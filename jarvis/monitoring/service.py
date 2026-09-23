"""Continuous observation (spec §10, §29, §78-81, §94, §97, §135-136).

Lightweight local monitors turn the environment into structured state and
events. None of them calls a language model: EVENT → deterministic rule →
action; the model is only consulted later if interpretation is needed.
"""

from __future__ import annotations

import asyncio
import socket
import time
from collections import deque
from typing import Any, Awaitable, Callable

import psutil

from jarvis.clock import Clock, SystemClock, format_duration
from jarvis.config import MonitoringConfig
from jarvis.core.types import (Confidence, HealthStatus, NetworkState, Provenance, ProvenanceKind, Severity)
from jarvis.database.db import Database
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.log import get_logger
from jarvis.models.router import ModelRouter
from jarvis.monitoring.analysis import ThresholdDetector, TrendTracker
from jarvis.monitoring.metrics import MetricsSource
from jarvis.state.engine import StateEngine
from jarvis.state.health import HealthRegistry
from jarvis.world.model import LOADED_INTO, RUNS_ON, WorldModel

log = get_logger("monitoring")

NetworkProbe = Callable[[], Awaitable[tuple[bool, float | None]]]
_PSUTIL = Provenance(ProvenanceKind.SYSTEM_STATE, "system monitor")


class SystemMonitor:
    def __init__(self, metrics: MetricsSource, state: StateEngine, config: MonitoringConfig, *,
                 bus: EventBus | None = None, clock: Clock | None = None, world: WorldModel | None = None,
                 health: HealthRegistry | None = None) -> None:
        self.metrics = metrics
        self.state = state
        self.config = config
        self.bus = bus
        self.clock = clock or SystemClock()
        self.world = world
        self.health = health
        self.thresholds = ThresholdDetector(config.thresholds)
        self.trends = TrendTracker(config.trend_window)
        self._last_prediction: dict[str, float] = {}
        self._last_trend: dict[str, float] = {}
        self.host = socket.gethostname()

    async def sample_once(self) -> dict[str, Any]:
        sample = await asyncio.to_thread(self.metrics.sample)
        now = self.clock.now()
        ttl = max(3 * self.config.system_interval_s, 30.0)
        for key, value in sample.items():
            if isinstance(value, (int, float, str, bool)):
                self.state.set(f"resources.{key}", value, provenance=_PSUTIL, ttl=ttl, persist=False)
        if self.world is not None:
            attrs = {k: sample.get(k) for k in ("cpu_percent", "memory_percent", "disk_percent", "gpu_percent")
                     if k in sample}
            self.world.upsert_entity("machine", self.host, attrs, source="system monitor")
        if self.bus:
            self.bus.emit(Event(EventType.SYSTEM_METRICS, "system_monitor", sample, severity=Severity.DEBUG))
        for ev in self.thresholds.evaluate(sample, now):
            self._emit(EventType.RESOURCE_THRESHOLD_EXCEEDED if ev.exceeded else EventType.RESOURCE_THRESHOLD_CLEARED,
                       ev.payload(), Severity.CRITICAL if ev.exceeded and ev.severity == "critical"
                       else Severity.WARNING if ev.exceeded else Severity.INFO)
        self.trends.add_sample(sample, now)
        self._predict(now)
        if self.health:
            self.health.report("system_monitor", HealthStatus.HEALTHY, "sampling")
        return sample

    def _predict(self, now: float) -> None:
        disk_th = self.config.thresholds.get("disk_percent")
        if disk_th:
            eta = self.trends.time_to_threshold("disk_percent", disk_th.value)
            if eta is not None and 0 < eta < 7 * 86400 and now - self._last_prediction.get("disk", 0) > 6 * 3600:
                self._last_prediction["disk"] = now
                self._emit(EventType.PREDICTIVE_WARNING, {
                    "metric": "disk_percent", "eta_s": eta, "threshold": disk_th.value,
                    "confidence": Confidence.ESTIMATED.value,
                    "message": f"At the current rate, disk usage will reach {disk_th.value:.0f}% in approximately "
                               f"{format_duration(eta)} (estimate)"}, Severity.WARNING)
        mem = self.trends.steady_trend("memory_percent", min_abs_per_hour=5.0)
        if mem and mem.slope_per_hour > 0 and mem.span_s >= 1800 and now - self._last_trend.get("memory", 0) > 3600:
            self._last_trend["memory"] = now
            self._emit(EventType.TREND_DETECTED, {
                "metric": "memory_percent", "slope_per_hour": round(mem.slope_per_hour, 2), "r2": round(mem.r2, 2),
                "message": f"Memory usage has increased steadily for the last {format_duration(mem.span_s)} "
                           f"(+{mem.slope_per_hour:.1f}%/hour)"}, Severity.INFO)

    def _emit(self, etype: EventType, payload: dict[str, Any], severity: Severity) -> None:
        if self.bus:
            self.bus.emit(Event(etype, "system_monitor", payload, severity=severity, entity_id=f"machine:{self.host}"))


async def tcp_probe(host: str, port: int, timeout: float = 3.0) -> tuple[bool, float | None]:
    started = time.monotonic()
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True, (time.monotonic() - started) * 1000
    except (OSError, asyncio.TimeoutError):
        return False, None


def interfaces_up() -> bool:
    try:
        stats = psutil.net_if_stats()
    except Exception:
        return True
    return any(s.isup for name, s in stats.items() if not name.startswith("lo"))


class NetworkMonitor:
    def __init__(self, state: StateEngine, config: MonitoringConfig, *, bus: EventBus | None = None,
                 clock: Clock | None = None, probe: NetworkProbe | None = None,
                 interfaces: Callable[[], bool] | None = None) -> None:
        self.state = state
        self.config = config
        self.bus = bus
        self.clock = clock or SystemClock()
        self.probe = probe or (lambda: tcp_probe(config.network_probe_host, config.network_probe_port))
        self.interfaces = interfaces or interfaces_up
        self.changes: deque[float] = deque(maxlen=20)

    async def check_once(self) -> NetworkState:
        reachable, latency = await self.probe()
        if reachable:
            status = NetworkState.HIGH_LATENCY if latency and latency > self.config.network_high_latency_ms \
                else NetworkState.ONLINE
        else:
            status = NetworkState.LIMITED if self.interfaces() else NetworkState.OFFLINE
        now = self.clock.now()
        previous = self.state.value("network.state")
        recent = [t for t in self.changes if now - t < 600]
        if previous is not None and previous != status.value:
            self.changes.append(now)
            recent.append(now)
        if len(recent) >= 4 and status != NetworkState.OFFLINE:
            status = NetworkState.UNSTABLE
        self.state.set("network.state", status.value, provenance=Provenance(ProvenanceKind.SYSTEM_STATE, "network probe"),
                       ttl=3 * self.config.network_interval_s, persist=False)
        if latency is not None:
            self.state.set("network.latency_ms", round(latency, 1), ttl=3 * self.config.network_interval_s,
                           persist=False)
        if previous is not None and previous != status.value and self.bus:
            self.bus.emit(Event(EventType.NETWORK_CHANGED, "network_monitor",
                                {"state": status.value, "previous": previous, "latency_ms": latency},
                                severity=Severity.WARNING if status == NetworkState.OFFLINE else Severity.INFO))
        return status


class ModelMonitor:
    def __init__(self, router: ModelRouter, state: StateEngine, *, world: WorldModel | None = None,
                 host: str | None = None) -> None:
        self.router = router
        self.state = state
        self.world = world
        self.host = host or socket.gethostname()

    async def check_once(self) -> None:
        inventory = await self.router.refresh()
        prov = Provenance(ProvenanceKind.SYSTEM_STATE, "model manager")
        self.state.set("models.available", [m.name for m in inventory], provenance=prov, persist=False)
        self.state.set("models.loaded", [m.name for m in inventory if m.loaded], provenance=prov, persist=False)
        self.state.set("models.providers", dict(self.router.provider_status), provenance=prov, persist=False)
        if self.world is not None:
            for m in inventory:
                entity = self.world.upsert_entity("model", m.name, m.to_dict(), id=f"model:{m.name}",
                                                  source="model manager")
                if m.local:
                    self.world.relate(entity.id, RUNS_ON, f"machine:{self.host}")
                if m.loaded:
                    self.world.relate(entity.id, LOADED_INTO, f"machine:{self.host}", {"vram_bytes": m.vram_bytes})
                else:
                    self.world.unrelate(entity.id, LOADED_INTO)


class SelfMonitor:
    def __init__(self, db: Database, bus: EventBus, health: HealthRegistry) -> None:
        self.db = db
        self.bus = bus
        self.health = health

    def check_once(self) -> None:
        self.health.report("database", HealthStatus.HEALTHY if self.db.healthy() else HealthStatus.CRITICAL,
                           critical=True)
        bus_status = HealthStatus.HEALTHY if self.bus.persist_failures == 0 else HealthStatus.DEGRADED
        self.health.report("event_bus", bus_status, f"{self.bus.published} published, "
                           f"{self.bus.handler_errors} handler errors", critical=True)
        self.health.check_heartbeats()


class MonitoringService:
    """Runs each monitor on its own interval; a failing monitor degrades, never crashes, the system."""

    def __init__(self, config: MonitoringConfig, *, system: SystemMonitor | None = None,
                 network: NetworkMonitor | None = None, models: ModelMonitor | None = None,
                 self_monitor: SelfMonitor | None = None, health: HealthRegistry | None = None,
                 extra: list[tuple[str, float, Callable[[], Awaitable[Any]]]] | None = None) -> None:
        self.config = config
        self.system = system
        self.network = network
        self.models = models
        self.self_monitor = self_monitor
        self.health = health
        self.extra = extra or []
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        loops: list[tuple[str, float, Callable[[], Awaitable[Any]]]] = []
        if self.system:
            loops.append(("system_monitor", self.config.system_interval_s, self.system.sample_once))
        if self.network:
            loops.append(("network_monitor", self.config.network_interval_s, self.network.check_once))
        if self.models:
            loops.append(("model_monitor", self.config.model_interval_s, self.models.check_once))
        if self.self_monitor:
            async def self_check() -> None:
                self.self_monitor.check_once()
            loops.append(("self_monitor", 15.0, self_check))
        loops += self.extra
        for name, interval, fn in loops:
            self._tasks.append(asyncio.create_task(self._loop(name, interval, fn), name=f"monitor-{name}"))

    async def _loop(self, name: str, interval: float, fn: Callable[[], Awaitable[Any]]) -> None:
        while True:
            try:
                await fn()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("monitor_failed", monitor=name, error=repr(exc))
                if self.health:
                    self.health.report(name, HealthStatus.DEGRADED, f"check failed: {exc}")
            await asyncio.sleep(interval)

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()
