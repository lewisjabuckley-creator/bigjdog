"""One runtime health check and one live-state snapshot.

Health is derived from observable signals (a database query, loop heartbeats,
provider probes), never asserted. A signal that cannot be read is reported as
``unknown`` rather than assumed healthy.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from jarvis import __version__
from jarvis.core.types import HealthStatus
from jarvis.tasks.models import EXECUTING, OPEN, TaskKind, TaskStatus

if TYPE_CHECKING:
    from jarvis.runtime import Runtime

CRITICAL = ("runtime", "database", "task_engine", "workers", "scheduler", "event_bus")


def _c(status: HealthStatus | str, detail: str = "", **extra: Any) -> dict[str, Any]:
    label = status.label if isinstance(status, HealthStatus) else status
    return {"status": label, "detail": detail, **extra}


async def health_report(runtime: "Runtime", *, probe: bool = True) -> dict[str, Any]:
    """Runtime, database, task engine, workers, scheduler, event bus, monitoring, model layer and Ollama."""
    svc = runtime.svc
    if svc is None or not runtime.started:
        return {"overall": "offline", "components": {"runtime": _c(HealthStatus.OFFLINE, "not started")}}
    now = svc.clock.now()
    comps: dict[str, dict[str, Any]] = {}
    uptime = now - (runtime.started_at or now)
    comps["runtime"] = _c(HealthStatus.HEALTHY, f"{runtime.mode}, pid {os.getpid()}, up {int(uptime)}s",
                          pid=os.getpid(), mode=runtime.mode, run=runtime.run_id, version=__version__,
                          uptime_s=round(uptime, 1))
    db_ok = svc.db.healthy()
    size = None
    try:
        size = round(os.path.getsize(svc.db.path) / 2**20, 2) if svc.db.path != ":memory:" else None
    except OSError:
        pass
    comps["database"] = _c(HealthStatus.HEALTHY if db_ok else HealthStatus.CRITICAL,
                           f"schema v{svc.db.schema_version()}" if db_ok else "queries fail", size_mb=size)
    try:
        counts: dict[str, int] = {}
        for task in svc.tasks.list_tasks(OPEN, limit=1000):
            counts[task.status.value] = counts.get(task.status.value, 0) + 1
        comps["task_engine"] = _c(HealthStatus.HEALTHY, f"{sum(counts.values())} open task(s)", open=counts)
    except Exception as exc:
        comps["task_engine"] = _c(HealthStatus.CRITICAL, f"cannot read tasks: {exc}")
    pool = svc.pool
    loop_alive = pool._loop_task is not None and not pool._loop_task.done()
    worker_health = svc.health.components.get("workers")
    comps["workers"] = _c(HealthStatus.HEALTHY if loop_alive else HealthStatus.CRITICAL,
                          f"{len(pool.running)} running, max {pool.config.max_concurrent}"
                          if loop_alive else "the scheduler loop is not running",
                          running=list(pool.running), throttled=sorted(svc.resources.throttled),
                          registry=worker_health.status.label if worker_health else "unknown")
    ticks = runtime.loop_ticks()
    tick_s = svc.config.scheduler.tick_s
    last = ticks.get("scheduler")
    if last is None:
        sched = _c(HealthStatus.UNKNOWN, "no tick yet")
    elif now - last > max(3 * tick_s, 30.0):
        sched = _c(HealthStatus.DEGRADED, f"last tick {int(now - last)}s ago", last_tick=last)
    else:
        upcoming = svc.automations.upcoming(limit=1)
        sched = _c(HealthStatus.HEALTHY, f"next: {upcoming[0].name}" if upcoming else "nothing scheduled",
                   last_tick=last, schedules=len([a for a in svc.automations.list(enabled_only=True)
                                                   if a.kind == "schedule"]))
    comps["scheduler"] = sched
    bus = svc.bus
    comps["event_bus"] = _c(HealthStatus.HEALTHY if bus.persist_failures == 0 else HealthStatus.DEGRADED,
                            f"{bus.published} published, {bus.handler_errors} handler error(s), "
                            f"{bus.persist_failures} persist failure(s)")
    if svc.monitoring is None:
        comps["monitoring"] = _c("disabled", "continuous monitoring is off (heartbeat still checks the core)")
    else:
        alive = [t for t in svc.monitoring._tasks if not t.done()]
        stale = svc.state.is_stale("resources.cpu_percent")
        status = HealthStatus.HEALTHY if alive and not stale else HealthStatus.DEGRADED
        comps["monitoring"] = _c(status, f"{len(alive)} monitor loop(s)" + ("; system sample is stale" if stale else ""))
    router = svc.router
    rstatus = router.status()
    if router.available():
        comps["model_layer"] = _c(HealthStatus.HEALTHY, f"conversation model: {rstatus['conversation_model']}",
                                  active_requests=rstatus["active_requests"], last_failure=rstatus["last_failure"],
                                  last_fallback=rstatus["last_fallback"])
    else:
        comps["model_layer"] = _c(HealthStatus.DEGRADED, "no language model is available; deterministic "
                                  "capabilities still work", last_failure=rstatus["last_failure"])
    ollama_cfg = svc.config.models.ollama
    provider = router.providers.get("ollama")
    if not ollama_cfg.enabled or provider is None:
        comps["ollama"] = _c("disabled", "not configured")
    elif probe:
        try:
            h = await asyncio.wait_for(provider.health(), timeout=5)
            comps["ollama"] = _c(HealthStatus.HEALTHY if h.available else HealthStatus.OFFLINE,
                                 (f"version {h.version}" if h.available else h.detail) or "",
                                 url=ollama_cfg.base_url, latency_ms=h.latency_ms, version=h.version)
        except Exception as exc:
            comps["ollama"] = _c(HealthStatus.UNKNOWN, f"probe failed: {exc}", url=ollama_cfg.base_url)
    else:
        up = router.provider_status.get("ollama")
        comps["ollama"] = _c(HealthStatus.UNKNOWN if up is None else HealthStatus.HEALTHY if up else
                             HealthStatus.OFFLINE, "last known state", url=ollama_cfg.base_url)
    order = {s.label: s for s in HealthStatus}
    worst = HealthStatus.HEALTHY
    for name, comp in comps.items():
        status = order.get(comp["status"])
        if status is None:        # disabled
            continue
        if name not in CRITICAL and status > HealthStatus.DEGRADED:
            status = HealthStatus.DEGRADED     # optional parts only degrade the whole
        worst = max(worst, status)
    return {"overall": worst.label, "checked_at": now, "components": comps}


def _fresh(svc: Any, key: str) -> Any:
    """A live value, or None when the last observation is too old to trust."""
    return svc.state.value(key, allow_stale=False)


def state_snapshot(runtime: "Runtime") -> dict[str, Any]:
    """Live state: health, resources, network, GPU, battery, Ollama, models, tasks, workers, alerts."""
    svc = runtime.svc
    assert svc is not None
    now = svc.clock.now()
    res = {k.split(".", 1)[1]: v for k, v in svc.state.values("resources.").items()}
    stale = svc.state.is_stale("resources.cpu_percent")
    resources = {key: (None if stale else res.get(key)) for key in (
        "cpu_percent", "memory_percent", "memory_used_gb", "memory_total_gb", "disk_percent", "disk_free_gb",
        "swap_percent", "load_avg_1m", "cpu_temp_c")}
    battery = None if "battery_percent" not in res else {"percent": res.get("battery_percent"),
                                                         "plugged": res.get("battery_plugged"), "stale": stale}
    gpu = None if "gpu_percent" not in res else {"percent": res.get("gpu_percent"), "vram_used_gb": res.get("vram_used_gb"),
                                                 "vram_total_gb": res.get("vram_total_gb"),
                                                 "temp_c": res.get("gpu_temp_c"), "stale": stale}
    tasks = svc.tasks.list_tasks(OPEN, limit=200)
    by_status: dict[str, int] = {}
    for t in tasks:
        by_status[t.status.value] = by_status.get(t.status.value, 0) + 1
    running = [{"id": t.id, "title": t.title, "progress": t.compute_progress(),
                "step": t.current_step.description if t.current_step else None, "kind": t.kind.value}
               for t in tasks if t.status in EXECUTING]
    router_status = svc.router.status()
    alerts = [{"id": n.id, "text": n.text(), "priority": n.priority.name.lower()}
              for n in svc.notifications.pending()]
    alerts += [{"id": f"health:{c.name}", "text": f"{c.name} is {c.status.label}"
                + (f": {c.detail}" if c.detail else ""), "priority": "important"} for c in svc.health.unhealthy()]
    return {
        "ts": now,
        "runtime": {"pid": os.getpid(), "mode": runtime.mode, "run": runtime.run_id, "version": __version__,
                    "started_at": runtime.started_at, "uptime_s": round(now - (runtime.started_at or now), 1),
                    "simulated": runtime.simulated,
                    "self": {k.split(".")[-1]: v for k, v in svc.state.values("runtime.self.").items()}},
        "health": {"overall": svc.health.overall().label,
                   "unhealthy": [f"{c.name} {c.status.label}" for c in svc.health.unhealthy()]},
        "mode": svc.modes.current.value,
        "resources": resources, "resources_stale": stale, "battery": battery, "gpu": gpu,
        "network": {"state": _fresh(svc, "network.state"), "latency_ms": _fresh(svc, "network.latency_ms")},
        "ollama": {"url": svc.config.models.ollama.base_url if svc.config.models.ollama.enabled else None,
                   "reachable": svc.router.provider_status.get("ollama")},
        "models": {"conversation_model": router_status["conversation_model"], "loaded": router_status["loaded"],
                   "active_requests": router_status["active_requests"], "last_failure": router_status["last_failure"],
                   "providers": router_status["providers"]},
        "tasks": {"open": by_status, "running": running,
                  "monitors": len([t for t in tasks if t.kind == TaskKind.MONITOR and t.status == TaskStatus.RUNNING])},
        "workers": {"running": len(svc.pool.running), "max_concurrent": svc.pool.config.max_concurrent,
                    "throttled": sorted(svc.resources.throttled),
                    "low_priority_policy": svc.resources.low_priority_policy},
        "alerts": alerts,
        "presence": svc.presence.snapshot() if svc.presence else None,
        "schedule": [a.to_api() for a in svc.automations.upcoming(limit=5)],
    }
