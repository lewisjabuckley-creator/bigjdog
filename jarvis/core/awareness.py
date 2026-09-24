"""What happened while the user was away, and the morning briefing, built only from records.

Both answers are assembled from the task store, the event store, the
notification log, the runtime's own run records and live state. The language
model is not involved, so the answer is the same whether or not a model is
available, and it never contains a result that is not on record.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from jarvis.clock import format_datetime, format_duration, format_time
from jarvis.core.personality import join_clauses, sentence
from jarvis.core.services import Services
from jarvis.core.types import HealthStatus, NotificationPriority, Severity, new_id
from jarvis.database.db import dumps, loads
from jarvis.events.types import Event, EventType
from jarvis.tasks.models import OPEN, Task, TaskKind, TaskStatus

S = TaskStatus
RESULT_CHARS = 3000

_HEALTH_EVENTS = [EventType.SUBSYSTEM_DEGRADED, EventType.SUBSYSTEM_RECOVERED, EventType.MODEL_UNAVAILABLE,
                  EventType.MODEL_RECOVERED, EventType.NETWORK_CHANGED, EventType.RESOURCE_THRESHOLD_EXCEEDED,
                  EventType.RESOURCE_THRESHOLD_CLEARED, EventType.PREDICTIVE_WARNING, EventType.SECURITY_EVENT,
                  EventType.EMERGENCY_ENTERED, EventType.EMERGENCY_EXITED, EventType.RESOURCE_THROTTLED]


def _when(ts: float | None, now: float) -> str:
    if ts is None:
        return "now"
    same_day = time.localtime(ts)[:3] == time.localtime(now)[:3]
    return format_time(ts) if same_day else format_datetime(ts)


def _clip(text: str, limit: int = RESULT_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


# -- "what happened while I was away?" -------------------------------------------------------------

@dataclass
class AwayReport:
    since: float
    until: float | None                  # None: the user is still away (asked through the API)
    now: float
    kept_running: bool = True
    gaps: list[dict[str, Any]] = field(default_factory=list)          # periods JARVIS was not running
    unclean: list[dict[str, Any]] = field(default_factory=list)       # unexpected stops
    finished: list[dict[str, Any]] = field(default_factory=list)
    open: list[dict[str, Any]] = field(default_factory=list)
    schedules: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    notifications: list[dict[str, Any]] = field(default_factory=list)
    reported_task_ids: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.finished or self.open or self.schedules or self.events or self.notifications
                    or self.gaps or self.unclean)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def text(self) -> str:
        end = self.until or self.now
        head = f"While you were away ({_when(self.since, self.now)}–{_when(end, self.now)}, " \
               f"{format_duration(end - self.since)})"
        lines: list[str] = []
        if self.unclean:
            for u in self.unclean:
                back = f" and started again at {_when(u['restarted'], self.now)}" if u.get("restarted") else ""
                lines.append(f"JARVIS stopped unexpectedly around {_when(u['last_seen'], self.now)}{back}.")
        for gap in self.gaps:
            if not any(abs(gap["from"] - u["last_seen"]) < 120 for u in self.unclean):
                lines.append(f"JARVIS was not running from {_when(gap['from'], self.now)} to "
                             f"{_when(gap['to'], self.now)}.")
        if self.kept_running and not self.unclean:
            lines.append("JARVIS kept running the whole time.")
        if self.empty:
            return f"{head}: nothing happened. No tasks ran, nothing needs you and there were no alerts. " + \
                " ".join(lines)
        if self.finished:
            lines.append("Finished:")
            for i, t in enumerate(self.finished):
                status = t["status"] if t.get("outcome") in (None, "complete", "failed") else \
                    f"{t['status']} ({t['outcome']})"
                lines.append(f"• {t['title']} — {status} at {_when(t['finished_at'], self.now)}.")
                if t["status"] == "completed" and t.get("result") and i < 3:
                    lines += ["    " + ln for ln in _clip(t["result"]).splitlines()]
                elif t["status"] != "completed" and (t.get("error") or t.get("status_reason")):
                    lines.append(f"    {t.get('status_reason') or t.get('error')}")
        if self.open:
            lines.append("Still open:")
            for t in self.open:
                reason = f": {t['status_reason']}" if t.get("status_reason") else ""
                lines.append(f"• {t['title']} — {t['status']}{reason}")
        if self.schedules:
            lines.append("Scheduled: " + "; ".join(s["text"] for s in self.schedules) + ".")
        if self.events:
            lines.append("Also: " + "; ".join(f"{_when(e['ts'], self.now)} {e['text']}" for e in self.events) + ".")
        if self.notifications:
            lines.append("Notifications: " + "; ".join(n["text"] for n in self.notifications) + ".")
        return f"{head}:\n" + "\n".join(lines)


def away_window(svc: Services) -> tuple[float, float | None]:
    since, until = (svc.presence.away_window() if svc.presence is not None else (None, None))
    if since is None:
        last = svc.state.value("session.last_seen")
        since = float(last) if isinstance(last, (int, float)) else svc.clock.now() - 86400
    return since, until


def away_report(svc: Services, since: float | None = None, until: float | None = None) -> AwayReport:
    now = svc.clock.now()
    if since is None:
        since, until = away_window(svc)
    end = until or now
    report = AwayReport(since=since, until=until, now=now)
    _runtime_coverage(svc, report, since, end)

    for task in svc.tasks.list_tasks([S.COMPLETED, S.FAILED, S.CANCELLED], since=since, order="recent", limit=50):
        if task.kind == TaskKind.MONITOR or task.finished_at is None or not since <= task.finished_at <= end:
            continue
        report.finished.append(_task_brief(task))
    # plus anything you asked for whose outcome you haven't been shown yet, even if it finished before this
    # absence began (e.g. the runtime restarted while a window was open, which restarts the absence clock)
    seen_ids = {t["id"] for t in report.finished}
    for task in unreported_results(svc):
        if task.id not in seen_ids:
            report.finished.append(_task_brief(task))
    report.finished.sort(key=lambda t: t["finished_at"])
    for task in svc.tasks.list_tasks(OPEN, order="recent", limit=50):
        if task.kind == TaskKind.MONITOR:
            continue
        report.open.append(_task_brief(task))
    report.reported_task_ids = [t["id"] for t in report.finished] + \
        [t["id"] for t in report.open if t["status"] in ("waiting", "blocked")]

    for e in svc.events.query(since=since, until=end, types=[EventType.AUTOMATION_TRIGGERED,
                                                              EventType.SCHEDULE_MISSED], limit=50,
                              newest_first=False):
        p = e.payload
        if e.type == EventType.SCHEDULE_MISSED:
            report.schedules.append({"ts": e.ts, "name": p.get("name"), "missed": True,
                                     "text": f"{p.get('name')} was missed ({p.get('reason')})"})
        elif p.get("trigger") == "schedule":
            report.schedules.append({"ts": e.ts, "name": p.get("automation"), "missed": False,
                                     "text": f"{p.get('automation')} ran at {_when(e.ts, now)}"})

    for e in svc.events.query(since=since, until=end, types=_HEALTH_EVENTS, limit=40, newest_first=False):
        text = _health_text(e)
        if text:
            report.events.append({"ts": e.ts, "type": str(e.type), "text": text})
    report.events = report.events[-8:]

    task_ids = {t["id"] for t in report.finished + report.open}
    for n in reversed(svc.notifications.list(since=since, limit=50)):
        if n["ts"] > end or n.get("task_id") in task_ids:
            continue      # the task itself is reported above
        if NotificationPriority[n["priority"].upper()] < NotificationPriority.IMPORTANT:
            continue
        if str(n.get("source", "")).startswith("health") or n.get("dedupe_key", "") in ("network",):
            continue      # already covered by the health events
        report.notifications.append({"id": n["id"], "ts": n["ts"], "text": n["text"], "state": n["state"]})
    report.notifications = report.notifications[-6:]
    return report


REPORTED = "result_reported_at"


def unreported_results(svc: Services, days: float = 3.0, limit: int = 5) -> list[Task]:
    """Your most recent finished tasks (last ``days``) whose outcome has not been reported to you in a
    conversation."""
    since = svc.clock.now() - days * 86400
    return [t for t in svc.tasks.list_tasks([S.COMPLETED, S.FAILED], since=since, order="recent", limit=100)
            if t.kind != TaskKind.MONITOR and t.created_by.startswith("user") and not t.outputs.get(REPORTED)][:limit]


def mark_reported(svc: Services, task_ids: list[str]) -> None:
    """The user has now been told these outcomes (in a reply or a "while you were away" answer)."""
    now = svc.clock.now()
    for task_id in task_ids:
        task = svc.tasks.get_task(task_id)
        if task is not None and task.terminal and not task.outputs.get(REPORTED):
            task.outputs[REPORTED] = now
            svc.tasks.save(task)


def _task_brief(task: Task) -> dict[str, Any]:
    api = task.to_api()
    return {"id": task.id, "title": task.title, "status": task.status.value,
            "outcome": task.outcome.value if task.outcome else None, "status_reason": task.status_reason,
            "result": api["result"] if task.status == S.COMPLETED else "", "error": task.error,
            "finished_at": task.finished_at, "progress": api["progress"], "artifacts": task.artifacts[-5:]}


def _runtime_coverage(svc: Services, report: AwayReport, since: float, end: float) -> None:
    """Was JARVIS running the whole time? Derived from the run records (start, heartbeat, clean stop)."""
    rows = svc.db.query("SELECT * FROM runtime_runs WHERE started_at <= ? ORDER BY started_at", (end,))
    current = (svc.extra.get("run") or {}).get("id")
    intervals: list[tuple[float, float]] = []
    runs = [dict(r) for r in rows]
    for i, r in enumerate(runs):
        stop = report.now if r["id"] == current else (r["stopped_at"] or r["heartbeat_at"] or r["started_at"])
        if stop < since:
            continue
        intervals.append((r["started_at"], stop))
        if r["id"] != current and not r["clean"] and since <= stop <= end:
            nxt = runs[i + 1]["started_at"] if i + 1 < len(runs) else None
            report.unclean.append({"run": r["id"], "last_seen": stop, "restarted": nxt})
    if not intervals:          # no run record covers any of it (e.g. asked while no runtime is running)
        report.gaps.append({"from": since, "to": end})
        report.kept_running = False
        return
    cursor = since
    slack = max(60.0, 3 * svc.config.runtime.heartbeat_s)
    for start, stop in sorted(intervals):
        if start - cursor > slack:
            report.gaps.append({"from": cursor, "to": start})
        cursor = max(cursor, stop)
    if end - cursor > slack:
        report.gaps.append({"from": cursor, "to": end})
    report.kept_running = not report.gaps


def _health_text(e: Event) -> str | None:
    p = e.payload
    t = e.type
    if t == EventType.SUBSYSTEM_DEGRADED:
        if str(p.get("component", "")).startswith("model:"):
            return None
        return f"{p.get('component')} became {p.get('status')}" + (f" ({p['detail']})" if p.get("detail") else "")
    if t == EventType.SUBSYSTEM_RECOVERED:
        if str(p.get("component", "")).startswith("model:"):
            return None
        took = f" after {format_duration(p['duration'])}" if p.get("duration") else ""
        return f"{p.get('component')} recovered{took}"
    if t == EventType.MODEL_UNAVAILABLE:
        return f"model provider {p.get('provider')} became unavailable" + \
            (f" ({p.get('model')})" if p.get("model") else "")
    if t == EventType.MODEL_RECOVERED:
        return f"model provider {p.get('provider')} came back"
    if t == EventType.NETWORK_CHANGED:
        return f"network became {str(p.get('state', '')).replace('_', ' ')}"
    if t == EventType.RESOURCE_THROTTLED:
        return str(p.get("reason") or "work was throttled")
    for key in ("message", "reason", "summary"):
        if p.get(key):
            return str(p[key])
    return str(t).lower().replace("_", " ")


# -- morning briefing ----------------------------------------------------------------------------

def briefing_data(svc: Services, since: float | None = None) -> dict[str, Any]:
    """Everything a briefing says, as data (the HUD, voice or text renderers all read this)."""
    now = svc.clock.now()
    if since is None:
        last = svc.state.value("session.last_seen")
        since = float(last) if isinstance(last, (int, float)) else now - 86400
    local = time.localtime(now)
    finished = [t for t in svc.tasks.list_tasks([S.COMPLETED, S.FAILED], since=since, order="recent", limit=10)
                if t.kind != TaskKind.MONITOR]
    failed = [t for t in finished if t.status == S.FAILED]
    open_tasks = [t for t in svc.tasks.open_tasks() if t.kind != TaskKind.MONITOR]
    waiting = [t for t in open_tasks if t.status in (S.WAITING, S.BLOCKED)]
    overall = svc.health.overall()
    warnings = svc.events.query(since=since, min_severity=Severity.WARNING,
                                types=[EventType.RESOURCE_THRESHOLD_EXCEEDED, EventType.PREDICTIVE_WARNING,
                                       EventType.SECURITY_EVENT, EventType.TREND_DETECTED, EventType.SCHEDULE_MISSED,
                                       EventType.SYSTEM_RECOVERED], limit=5)
    resources = {k: svc.state.value(f"resources.{k}", allow_stale=False)
                 for k in ("cpu_percent", "memory_percent", "disk_percent", "disk_free_gb", "battery_percent",
                           "gpu_percent")}
    project = svc.projects.active()
    upcoming = svc.automations.upcoming(limit=3) if svc.automations else []
    queued = svc.notifications.pending()
    readiness = svc.extra.get("model_readiness")
    from jarvis.core.reports import activity
    headline_parts = []
    if finished:
        headline_parts.append(f"{len(finished)} task{'s' if len(finished) != 1 else ''} finished"
                              + (f", {len(failed)} failed" if failed else ""))
    if waiting:
        headline_parts.append(f"{len(waiting)} waiting on you")
    headline_parts.append("all systems healthy" if overall == HealthStatus.HEALTHY else f"health {overall.label}")
    return {
        "generated_at": now, "since": since,
        "time": time.strftime("%H:%M", local), "day": time.strftime("%A %d %B", local),
        "headline": sentence("; ".join(headline_parts)),
        "tasks": {"finished": [_task_brief(t) for t in finished], "failed": [t.id for t in failed],
                  "open": [_task_brief(t) for t in open_tasks[:10]], "waiting_on_you": [t.id for t in waiting]},
        "activity": activity(svc) if svc.tasks.open_tasks() else "",
        "health": {"overall": overall.label,
                   "unhealthy": [f"{c.name} {c.status.label}" for c in svc.health.unhealthy()[:5]]},
        "warnings": [_health_text(e) or str(e.type) for e in warnings],
        "resources": {k: v for k, v in resources.items() if v is not None},
        "model": readiness.summary() if readiness is not None and readiness.can_converse else None,
        "project": project.name if project else None,
        "schedule": [{"name": a.name, "next_run": a.next_run} for a in upcoming],
        "notifications": [n.text() for n in queued[:5]],
    }


def render_briefing(data: dict[str, Any]) -> str:
    parts = [f"It's {data['time']} on {data['day']}."]
    finished = data["tasks"]["finished"]
    failed = [t for t in finished if t["id"] in data["tasks"]["failed"]]
    if finished:
        text = f"{len(finished)} task{'s' if len(finished) != 1 else ''} finished since you were last here"
        if failed:
            text += f", {len(failed)} failed ({join_clauses([t['title'] for t in failed[:3]])})"
        parts.append(sentence(text))
    if data.get("activity"):
        parts.append(data["activity"])
    if data["health"]["overall"] == "healthy":
        parts.append("All systems are healthy.")
    else:
        parts.append(sentence(f"System health is {data['health']['overall']}: " +
                              join_clauses(data["health"]["unhealthy"][:3])))
    for warning in data["warnings"][:3]:
        parts.append(sentence(warning))
    if data.get("project"):
        parts.append(f"Active project: {data['project']}.")
    if data["schedule"]:
        parts.append(sentence("Scheduled: " + join_clauses([f"{s['name']} at {format_time(s['next_run'])}"
                                                            for s in data["schedule"] if s.get("next_run")])))
    return " ".join(parts)


def prepare_briefing(svc: Services, kind: str = "scheduled", briefing_id: str | None = None) -> dict[str, Any]:
    """Build, store and announce a briefing (the scheduled 'briefing' action). Idempotent per id."""
    previous = svc.db.query_one("SELECT ts FROM briefings ORDER BY ts DESC LIMIT 1")
    data = briefing_data(svc, since=previous["ts"] if previous else None)
    bid = briefing_id or new_id("brief")
    data["id"] = bid
    text = render_briefing(data)
    inserted = svc.db.execute("INSERT OR IGNORE INTO briefings(id, ts, kind, data, text) VALUES(?,?,?,?,?)",
                              (bid, data["generated_at"], kind, dumps(data), text))
    if inserted:
        svc.bus.emit(Event(EventType.BRIEFING_READY, "briefing", {"id": bid, "headline": data["headline"],
                                                                  "kind": kind}))
    return data


def latest_briefing(svc: Services) -> dict[str, Any] | None:
    row = svc.db.query_one("SELECT * FROM briefings ORDER BY ts DESC LIMIT 1")
    if row is None:
        return None
    return {"id": row["id"], "ts": row["ts"], "kind": row["kind"], "text": row["text"], "data": loads(row["data"], {})}
