"""Scheduled and conditional automation (spec §60-61, §136) — the durable scheduler.

WHEN <event> IF <condition> DO <action> → VERIFY → NOTIFY, plus schedules:
once, daily, weekly and interval. Schedules live in SQLite and are driven by
the runtime's own scheduler loop, so they run whether or not an interface is
open or monitoring is enabled.

Durability: each run belongs to a *slot* (its due time). The action for a slot
carries an idempotency key derived from it, so a crash between running an
action and recording it can never run the same slot twice. Runs missed while
JARVIS was not running are made up once if they are within the catch-up window
and otherwise recorded as missed (and reported), never silently dropped.

Authority: automations act with automation authority, not the user's. A
scheduled job gains nothing by being scheduled: anything above the automation
baseline needs an explicit, scoped grant, and goes through the same tool
registry pipeline (authorization, permission, execution, verification, audit)
as an interactive command. Every automation is inspectable, can be disabled,
and records when it last ran and how that went.

The schedule format is a plain dict, so a natural-language front end can
produce it later: ``{"type": "weekly", "days": ["mon", "thu"], "at": "08:00"}``.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from jarvis.clock import Clock, SystemClock
from jarvis.clock import format_datetime
from jarvis.config import WEEKDAYS
from jarvis.core.types import NotificationPriority, Priority, Severity, new_id
from jarvis.database.db import Database, dumps, loads
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.log import get_logger
from jarvis.tasks.manager import TaskManager
from jarvis.tasks.models import Step

log = get_logger("automation")

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    ">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq, "!=": operator.ne,
    "in": lambda a, b: a in b, "contains": lambda a, b: b in a if a is not None else False,
}

ActionHandler = Callable[[dict[str, Any], "Automation", Event | None], Awaitable[None]]


@dataclass
class Automation:
    id: str
    name: str
    kind: str                      # schedule | rule
    spec: dict[str, Any]
    enabled: bool = True
    owner: str = "owner"
    created_at: float = 0.0
    last_run: float | None = None
    next_run: float | None = None
    run_count: int = 0
    last_status: str | None = None     # ok | failed | missed
    last_error: str | None = None
    last_task_id: str | None = None
    missed: int = 0

    @property
    def schedule(self) -> dict[str, Any] | None:
        return schedule_of(self.spec) if self.kind == "schedule" else None

    def to_api(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "kind": self.kind, "schedule": self.schedule,
                "when": self.spec.get("when"), "action": self.spec.get("do"), "enabled": self.enabled,
                "owner": self.owner, "created_at": self.created_at, "last_run": self.last_run,
                "next_run": self.next_run, "run_count": self.run_count, "last_status": self.last_status,
                "last_error": self.last_error, "last_task_id": self.last_task_id, "missed": self.missed,
                "describe": describe_schedule(self.schedule) if self.schedule else None}


def _lookup(data: dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def matches(condition: dict[str, Any] | None, data: dict[str, Any]) -> bool:
    """``{"payload.outcome": "complete", "payload.value": {">": 90}}`` style conditions (all must hold)."""
    if not condition:
        return True
    for path, expected in condition.items():
        actual = _lookup(data, path)
        if isinstance(expected, dict):
            for op, value in expected.items():
                fn = _OPS.get(op)
                try:
                    if fn is None or not fn(actual, value):
                        return False
                except TypeError:
                    return False
        elif actual != expected:
            return False
    return True


def schedule_of(spec: dict[str, Any]) -> dict[str, Any]:
    """The normalised schedule of a spec (older specs used ``every_s`` / ``daily_at`` at the top level)."""
    if isinstance(spec.get("schedule"), dict):
        return spec["schedule"]
    if "every_s" in spec:
        return {"type": "interval", "every_s": float(spec["every_s"])}
    if "daily_at" in spec:
        return {"type": "daily", "at": str(spec["daily_at"])}
    if "at" in spec:
        return {"type": "once", "at": float(spec["at"])}
    raise ValueError("a schedule needs a 'schedule' ({type: once|daily|weekly|interval, ...})")


def validate_schedule(sched: dict[str, Any]) -> dict[str, Any]:
    kind = sched.get("type")
    if kind == "once":
        at = sched["at"]
        if isinstance(at, str) and not at.replace(".", "", 1).isdigit():
            from datetime import datetime
            try:
                at = datetime.fromisoformat(at).timestamp()     # "2026-10-01T09:00" (local time)
            except ValueError:
                raise ValueError(f"'at' must be a timestamp or an ISO date and time, not {at!r}") from None
        return {"type": "once", "at": float(at)}
    if kind == "interval":
        every = float(sched.get("every_s", 0))
        if every < 1:
            raise ValueError("an interval schedule needs every_s >= 1")
        out: dict[str, Any] = {"type": "interval", "every_s": every}
        if sched.get("start") is not None:
            out["start"] = float(sched["start"])
        return out
    if kind in ("daily", "weekly"):
        hh, mm = _hhmm(str(sched.get("at", "")))
        out = {"type": kind, "at": f"{hh:02d}:{mm:02d}"}
        if kind == "weekly":
            days = [str(d).lower()[:3] for d in sched.get("days") or []]
            if not days or any(d not in WEEKDAYS for d in days):
                raise ValueError(f"a weekly schedule needs days from {', '.join(WEEKDAYS)}")
            out["days"] = sorted(set(days), key=WEEKDAYS.index)
        return out
    raise ValueError("schedule type must be once, daily, weekly or interval")


def _hhmm(text: str) -> tuple[int, int]:
    try:
        hh, mm = (int(x) for x in text.split(":"))
    except ValueError:
        raise ValueError(f"time must be HH:MM, not {text!r}") from None
    if not (0 <= hh < 24 and 0 <= mm < 60):
        raise ValueError(f"time must be HH:MM, not {text!r}")
    return hh, mm


def _local_at(day_ts: float, hh: int, mm: int) -> float:
    local = time.localtime(day_ts)
    # isdst=-1: let the C library resolve daylight-saving time for that date
    return time.mktime((local.tm_year, local.tm_mon, local.tm_mday, hh, mm, 0, 0, 0, -1))


def next_run_for(spec: dict[str, Any], after: float) -> float | None:
    """The first slot strictly after ``after`` (local wall-clock time for daily and weekly schedules)."""
    sched = schedule_of(spec)
    kind = sched["type"]
    if kind == "once":
        at = float(sched["at"])
        return at if at > after else None
    if kind == "interval":
        every = float(sched["every_s"])
        start = sched.get("start")
        if start is None:
            return after + every
        start = float(start)
        if after < start:
            return start
        return start + (int((after - start) // every) + 1) * every
    hh, mm = _hhmm(str(sched["at"]))
    days = sched.get("days") if kind == "weekly" else list(WEEKDAYS)
    for offset in range(0, 9):
        candidate = _local_at(after + offset * 86400, hh, mm)
        if candidate <= after:
            continue
        if WEEKDAYS[time.localtime(candidate).tm_wday] in days:
            return candidate
    return None


def describe_schedule(sched: dict[str, Any]) -> str:
    kind = sched.get("type")
    if kind == "once":
        return f"once at {format_datetime(float(sched['at']))}"
    if kind == "interval":
        every = float(sched["every_s"])
        unit = (every / 86400, "day") if every % 86400 == 0 else (every / 3600, "hour") if every % 3600 == 0 else \
            (every / 60, "minute") if every % 60 == 0 else (every, "second")
        n = int(unit[0])
        return f"every {unit[1]}" if n == 1 else f"every {n} {unit[1]}s"
    if kind == "daily":
        return f"every day at {sched['at']}"
    if kind == "weekly":
        return f"every {', '.join(d.capitalize() for d in sched.get('days', []))} at {sched['at']}"
    return str(sched)


class AutomationEngine:
    def __init__(self, db: Database, tasks: TaskManager, *, bus: EventBus | None = None,
                 clock: Clock | None = None, catch_up_window_s: float = 21600.0) -> None:
        self.db = db
        self.tasks = tasks
        self.bus = bus
        self.clock = clock or SystemClock()
        self.catch_up_window_s = catch_up_window_s
        self.handlers: dict[str, ActionHandler] = {"task": self._task_action}
        self._cache: list[Automation] | None = None
        self.max_fires_per_minute = 10
        self._fires: dict[str, list[float]] = {}
        self._slot: dict[str, float] = {}       # automation id -> slot being run (for idempotency keys)

    # -- CRUD ------------------------------------------------------------------------------
    def create(self, name: str, kind: str, spec: dict[str, Any], *, owner: str = "owner") -> Automation:
        if kind not in ("schedule", "rule"):
            raise ValueError("kind must be 'schedule' or 'rule'")
        if "do" not in spec:
            raise ValueError("an automation needs a 'do' action")
        if not isinstance(spec["do"], dict) or spec["do"].get("type") not in self.handlers:
            raise ValueError(f"unknown action; available: {', '.join(sorted(self.handlers))}")
        now = self.clock.now()
        if kind == "schedule":
            spec = {**{k: v for k, v in spec.items() if k not in ("every_s", "daily_at", "at")},
                    "schedule": validate_schedule(schedule_of(spec))}
        auto = Automation(new_id("auto"), name, kind, spec, True, owner, now)
        if kind == "schedule":
            auto.next_run = next_run_for(spec, now)
            if auto.next_run is None:
                raise ValueError("that time has already passed")
        self.db.execute("INSERT INTO automations(id, name, kind, spec, enabled, owner, created_at, next_run) "
                        "VALUES(?,?,?,?,1,?,?,?)", (auto.id, name, kind, dumps(spec), owner, now, auto.next_run))
        self._cache = None
        return auto

    def schedule(self, name: str, when: dict[str, Any], action: dict[str, Any], *,
                 owner: str = "owner", catch_up: str = "once") -> Automation:
        """Convenience: ``schedule("backup", {"type": "daily", "at": "02:00"}, {"type": "task", ...})``."""
        if catch_up not in ("once", "skip"):
            raise ValueError("catch_up must be 'once' or 'skip'")
        return self.create(name, "schedule", {"schedule": when, "do": action, "catch_up": catch_up}, owner=owner)

    def get(self, automation_id: str) -> Automation | None:
        return next((a for a in self.list() if a.id == automation_id), None)

    def find(self, name: str, owner: str | None = None) -> Automation | None:
        return next((a for a in self.list() if a.name == name and (owner is None or a.owner == owner)), None)

    def list(self, enabled_only: bool = False) -> list[Automation]:
        if self._cache is None:
            self._cache = [_row(r) for r in self.db.query("SELECT * FROM automations ORDER BY created_at")]
        return [a for a in self._cache if a.enabled or not enabled_only]

    def set_enabled(self, automation_id: str, enabled: bool) -> bool:
        auto = self.get(automation_id)
        next_run = auto.next_run if auto else None
        if enabled and auto is not None and auto.kind == "schedule" and (next_run is None or
                                                                          next_run <= self.clock.now()):
            next_run = next_run_for(auto.spec, self.clock.now())   # re-enabling never replays old slots
        changed = self.db.execute("UPDATE automations SET enabled=?, next_run=? WHERE id=?",
                                  (int(enabled), next_run, automation_id))
        self._cache = None
        return bool(changed)

    def delete(self, automation_id: str) -> bool:
        changed = self.db.execute("DELETE FROM automations WHERE id=?", (automation_id,))
        self._cache = None
        return bool(changed)

    def upcoming(self, limit: int = 5) -> list[Automation]:
        scheduled = [a for a in self.list(enabled_only=True) if a.kind == "schedule" and a.next_run]
        return sorted(scheduled, key=lambda a: a.next_run or 0)[:limit]

    # -- execution ----------------------------------------------------------------------------
    def attach(self) -> None:
        if self.bus is not None:
            self.bus.subscribe("*", self._on_event, name="automation")

    async def tick(self) -> int:
        """Run due schedules (each due slot at most once). Returns how many fired."""
        now = self.clock.now()
        fired = 0
        for auto in self.list(enabled_only=True):
            if auto.kind != "schedule" or auto.next_run is None or auto.next_run > now:
                continue
            slot = auto.next_run
            following = next_run_for(auto.spec, now)     # later missed slots collapse into this one run
            late = now - slot
            if late > self.catch_up_window_s or (late > 60 and auto.spec.get("catch_up") == "skip"):
                self._missed(auto, slot, following)
                continue
            auto.next_run = following
            self._slot[auto.id] = slot
            try:
                await self._fire(auto, None)
            finally:
                self._slot.pop(auto.id, None)
            fired += 1
        return fired

    def _missed(self, auto: Automation, slot: float, following: float | None) -> None:
        auto.missed += 1
        auto.next_run = following
        auto.last_status = "missed"
        reason = f"it was due at {format_datetime(slot)} while JARVIS was not running"
        auto.last_error = reason
        self.db.execute("UPDATE automations SET next_run=?, missed=?, last_status=?, last_error=? WHERE id=?",
                        (following, auto.missed, "missed", reason, auto.id))
        log.warning("schedule_missed", automation=auto.id, slot=slot)
        if self.bus:
            self.bus.emit(Event(EventType.SCHEDULE_MISSED, "scheduler",
                                {"id": auto.id, "name": auto.name, "slot": slot, "next_run": following,
                                 "reason": reason}, severity=Severity.WARNING))

    async def _on_event(self, event: Event) -> None:
        if event.source == "automation":
            return   # never trigger on our own events (no loops)
        for auto in self.list(enabled_only=True):
            if auto.kind != "rule" or auto.spec.get("when") != str(event.type):
                continue
            if event.payload.get("created_by") == f"automation:{auto.id}":
                continue   # an automation never reacts to the work it created itself
            if self._rate_limited(auto):
                continue
            if matches(auto.spec.get("if"), {"payload": event.payload, "task_id": event.task_id,
                                             "severity": event.severity.name.lower(), "source": event.source}):
                await self._fire(auto, event)

    def _rate_limited(self, auto: Automation) -> bool:
        """Defence in depth against cascades between automations."""
        now = self.clock.now()
        recent = [t for t in self._fires.get(auto.id, []) if now - t < 60]
        self._fires[auto.id] = recent
        if len(recent) >= self.max_fires_per_minute:
            log.warning("automation_rate_limited", automation=auto.id)
            return True
        return False

    async def _fire(self, auto: Automation, event: Event | None) -> None:
        now = self.clock.now()
        self._fires.setdefault(auto.id, []).append(now)
        action = auto.spec["do"]
        handler = self.handlers.get(action.get("type", ""))
        if self.bus:
            self.bus.emit(Event(EventType.AUTOMATION_TRIGGERED, "automation",
                                {"automation": auto.name, "id": auto.id, "action": action.get("type"),
                                 "trigger": str(event.type) if event else "schedule",
                                 "slot": self._slot.get(auto.id)}))
        status, error = "ok", None
        if handler is None:
            status, error = "failed", f"unknown action {action.get('type')!r}"
            log.error("unknown_automation_action", automation=auto.id, action=action)
        else:
            try:
                await handler(action, auto, event)
            except Exception as exc:
                status, error = "failed", str(exc)
                log.error("automation_failed", automation=auto.id, error=repr(exc))
                if self.bus:
                    self.bus.emit(Event(EventType.ERROR_DETECTED, "automation", {"automation": auto.name,
                                                                                 "error": str(exc)},
                                        severity=Severity.WARNING))
        # recorded after the action: a crash in between re-runs the slot, and the slot's idempotency key
        # makes that re-run return what the first attempt created
        auto.last_run = now
        auto.run_count += 1
        auto.last_status, auto.last_error = status, error
        self.db.execute("UPDATE automations SET last_run=?, next_run=?, run_count=?, last_status=?, last_error=?, "
                        "last_task_id=? WHERE id=?", (now, auto.next_run, auto.run_count, status, error,
                                                     auto.last_task_id, auto.id))

    def idempotency_key(self, auto: Automation, event: Event | None) -> str:
        slot = self._slot.get(auto.id)
        if event is not None:
            return f"rule:{auto.id}:{event.id}"
        return f"schedule:{auto.id}:{slot if slot is not None else self.clock.now()}"

    async def _task_action(self, action: dict[str, Any], auto: Automation, event: Event | None) -> None:
        steps = [Step(s.get("description") or s["tool"], s["tool"], s.get("args", {}), allow_failure=s.get("allow_failure", False))
                 for s in action.get("steps", [])]
        objective = action.get("objective") or auto.name
        if event is not None and event.payload.get("title"):
            objective = objective.replace("{title}", str(event.payload["title"]))
        # automation authority, never the user's: scheduling something does not grant it anything
        task = self.tasks.create_task(objective, title=action.get("title") or objective, steps=steps or None,
                                      priority=Priority[action.get("priority", "P3")],
                                      created_by=f"automation:{auto.id}", authority={"interactive": False},
                                      cwd=action.get("cwd"), success_condition=action.get("success_condition"),
                                      dependencies=[event.task_id] if event is not None and event.task_id and
                                      action.get("after_trigger_task") else None,
                                      request=objective, origin=f"{auto.kind}:{auto.id}",
                                      idempotency_key=self.idempotency_key(auto, event))
        auto.last_task_id = task.id


def notify_action(notifier: Any) -> ActionHandler:
    async def handler(action: dict[str, Any], auto: Automation, event: Event | None) -> None:
        priority = NotificationPriority[action.get("priority", "IMPORTANT").upper()]
        title = action.get("title", auto.name)
        if event is not None:
            title = title.replace("{title}", str(event.payload.get("title", "")))
        notifier.notify(priority, title, action.get("body", ""), source=f"automation:{auto.id}")
    return handler


def _row(r: Any) -> Automation:
    keys = r.keys()
    return Automation(r["id"], r["name"], r["kind"], loads(r["spec"], {}), bool(r["enabled"]), r["owner"],
                      r["created_at"], r["last_run"], r["next_run"], r["run_count"],
                      r["last_status"] if "last_status" in keys else None,
                      r["last_error"] if "last_error" in keys else None,
                      r["last_task_id"] if "last_task_id" in keys else None,
                      r["missed"] if "missed" in keys else 0)
