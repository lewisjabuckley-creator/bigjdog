"""Health model and self-monitoring (spec §78, §93-95).

Each subsystem reports its own health. Overall health is derived from those
observable signals — never asserted. Outages are recorded so JARVIS can later
explain itself ("the local model was unavailable for 18 seconds").
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from jarvis.clock import Clock, SystemClock
from jarvis.core.types import HealthStatus, Severity
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    detail: str = ""
    since: float = 0.0
    critical: bool = False
    last_heartbeat: float | None = None
    heartbeat_timeout_s: float | None = None


@dataclass
class Outage:
    component: str
    status: HealthStatus
    detail: str
    started: float
    ended: float | None = None

    @property
    def duration(self) -> float | None:
        return None if self.ended is None else self.ended - self.started


@dataclass
class HealthRegistry:
    bus: EventBus | None = None
    clock: Clock = field(default_factory=SystemClock)
    components: dict[str, ComponentHealth] = field(default_factory=dict)
    outages: deque[Outage] = field(default_factory=lambda: deque(maxlen=200))
    last_overall: HealthStatus | None = None

    def register(self, name: str, *, critical: bool = False, heartbeat_timeout_s: float | None = None) -> None:
        if name not in self.components:
            self.components[name] = ComponentHealth(name, critical=critical, since=self.clock.now(),
                                                    heartbeat_timeout_s=heartbeat_timeout_s)

    def report(self, name: str, status: HealthStatus, detail: str = "", *, critical: bool | None = None) -> None:
        now = self.clock.now()
        comp = self.components.get(name)
        if comp is None:
            comp = self.components[name] = ComponentHealth(name, critical=bool(critical), since=now)
        if critical is not None:
            comp.critical = critical
        previous = comp.status
        comp.detail = detail
        comp.last_heartbeat = now
        if status == previous:
            return
        comp.status = status
        comp.since = now
        bad = status >= HealthStatus.DEGRADED
        was_bad = previous >= HealthStatus.DEGRADED
        if bad and not was_bad:
            self.outages.append(Outage(name, status, detail, now))
            self._emit(EventType.SUBSYSTEM_DEGRADED, name, status, detail,
                       Severity.ERROR if comp.critical else Severity.WARNING)
        elif was_bad and not bad:
            outage = self._open_outage(name)
            if outage:
                outage.ended = now
            self._emit(EventType.SUBSYSTEM_RECOVERED, name, status, detail, Severity.INFO,
                       duration=outage.duration if outage else None)
        self._check_overall()

    def _check_overall(self) -> None:
        """Publish HEALTH_CHANGED only when the overall state changes (never once per sample)."""
        overall = self.overall()
        previous, self.last_overall = self.last_overall, overall
        if previous is None or previous == overall or self.bus is None:
            return
        if previous == HealthStatus.UNKNOWN and overall == HealthStatus.HEALTHY:
            return      # components finishing their first check (e.g. at startup) is not news
        severity = Severity.INFO if overall <= HealthStatus.HEALTHY else \
            Severity.ERROR if overall >= HealthStatus.CRITICAL else Severity.WARNING
        self.bus.emit(Event(EventType.HEALTH_CHANGED, "health",
                            {"from": previous.label, "to": overall.label,
                             "unhealthy": [f"{c.name} {c.status.label}" for c in self.unhealthy()][:6]},
                            severity=severity))

    def heartbeat(self, name: str) -> None:
        comp = self.components.get(name)
        if comp:
            comp.last_heartbeat = self.clock.now()

    def check_heartbeats(self) -> list[str]:
        """Mark components whose heartbeat is overdue as offline. Returns their names."""
        now = self.clock.now()
        stale = []
        for comp in self.components.values():
            if comp.heartbeat_timeout_s and comp.last_heartbeat is not None \
                    and now - comp.last_heartbeat > comp.heartbeat_timeout_s \
                    and comp.status != HealthStatus.OFFLINE:
                stale.append(comp.name)
                self.report(comp.name, HealthStatus.OFFLINE, "heartbeat overdue")
        return stale

    def overall(self) -> HealthStatus:
        if not self.components:
            return HealthStatus.UNKNOWN
        worst = HealthStatus.HEALTHY
        for comp in self.components.values():
            status = comp.status
            if not comp.critical and status > HealthStatus.DEGRADED:
                status = HealthStatus.DEGRADED  # optional components can only degrade the whole
            worst = max(worst, status)
        return worst

    def unhealthy(self) -> list[ComponentHealth]:
        return [c for c in self.components.values() if c.status >= HealthStatus.DEGRADED]

    def recent_outages(self, since: float | None = None, component: str | None = None) -> list[Outage]:
        return [o for o in self.outages
                if (since is None or (o.ended or self.clock.now()) >= since)
                and (component is None or o.component == component)]

    def _open_outage(self, name: str) -> Outage | None:
        for outage in reversed(self.outages):
            if outage.component == name and outage.ended is None:
                return outage
        return None

    def _emit(self, etype: EventType, name: str, status: HealthStatus, detail: str, severity: Severity,
              **extra: object) -> None:
        if self.bus is not None:
            payload = {"component": name, "status": status.label, "detail": detail, **extra}
            self.bus.emit(Event(etype, "health", payload, severity=severity, entity_id=f"subsystem:{name}"))
