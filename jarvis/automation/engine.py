"""Scheduled and conditional automation (spec §60-61, §136).

WHEN <event> IF <condition> DO <action> → VERIFY → NOTIFY, plus recurring
schedules. Automations run with automation authority (no interactive approval),
so anything consequential needs an explicit, scoped grant. Every automation is
inspectable, can be disabled, and records when it last ran.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from jarvis.clock import Clock, SystemClock
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


def next_run_for(spec: dict[str, Any], after: float) -> float | None:
    if "every_s" in spec:
        return after + float(spec["every_s"])
    if "daily_at" in spec:
        hh, mm = (int(x) for x in str(spec["daily_at"]).split(":"))
        local = time.localtime(after)
        candidate = time.mktime((local.tm_year, local.tm_mon, local.tm_mday, hh, mm, 0, 0, 0, -1))
        if candidate <= after:
            candidate += 86400
        return candidate
    return None


class AutomationEngine:
    def __init__(self, db: Database, tasks: TaskManager, *, bus: EventBus | None = None,
                 clock: Clock | None = None) -> None:
        self.db = db
        self.tasks = tasks
        self.bus = bus
        self.clock = clock or SystemClock()
        self.handlers: dict[str, ActionHandler] = {"task": self._task_action}
        self._cache: list[Automation] | None = None

    # -- CRUD ------------------------------------------------------------------------------
    def create(self, name: str, kind: str, spec: dict[str, Any], *, owner: str = "owner") -> Automation:
        if kind not in ("schedule", "rule"):
            raise ValueError("kind must be 'schedule' or 'rule'")
        if "do" not in spec:
            raise ValueError("an automation needs a 'do' action")
        now = self.clock.now()
        auto = Automation(new_id("auto"), name, kind, spec, True, owner, now)
        if kind == "schedule":
            auto.next_run = next_run_for(spec, now)
        self.db.execute("INSERT INTO automations(id, name, kind, spec, enabled, owner, created_at, next_run) "
                        "VALUES(?,?,?,?,1,?,?,?)", (auto.id, name, kind, dumps(spec), owner, now, auto.next_run))
        self._cache = None
        return auto

    def list(self, enabled_only: bool = False) -> list[Automation]:
        if self._cache is None:
            self._cache = [_row(r) for r in self.db.query("SELECT * FROM automations ORDER BY created_at")]
        return [a for a in self._cache if a.enabled or not enabled_only]

    def set_enabled(self, automation_id: str, enabled: bool) -> bool:
        changed = self.db.execute("UPDATE automations SET enabled=? WHERE id=?", (int(enabled), automation_id))
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
        """Run due schedules. Returns how many fired."""
        now = self.clock.now()
        fired = 0
        for auto in self.list(enabled_only=True):
            if auto.kind == "schedule" and auto.next_run is not None and auto.next_run <= now:
                auto.next_run = next_run_for(auto.spec, now)
                await self._fire(auto, None)
                fired += 1
        return fired

    async def _on_event(self, event: Event) -> None:
        if event.source == "automation":
            return   # never trigger on our own events (no loops)
        for auto in self.list(enabled_only=True):
            if auto.kind != "rule" or auto.spec.get("when") != str(event.type):
                continue
            if matches(auto.spec.get("if"), {"payload": event.payload, "task_id": event.task_id,
                                             "severity": event.severity.name.lower(), "source": event.source}):
                await self._fire(auto, event)

    async def _fire(self, auto: Automation, event: Event | None) -> None:
        now = self.clock.now()
        auto.last_run = now
        auto.run_count += 1
        self.db.execute("UPDATE automations SET last_run=?, next_run=?, run_count=? WHERE id=?",
                        (now, auto.next_run, auto.run_count, auto.id))
        action = auto.spec["do"]
        handler = self.handlers.get(action.get("type", ""))
        if self.bus:
            self.bus.emit(Event(EventType.AUTOMATION_TRIGGERED, "automation",
                                {"automation": auto.name, "id": auto.id, "action": action.get("type"),
                                 "trigger": str(event.type) if event else "schedule"}))
        if handler is None:
            log.error("unknown_automation_action", automation=auto.id, action=action)
            return
        try:
            await handler(action, auto, event)
        except Exception as exc:
            log.error("automation_failed", automation=auto.id, error=repr(exc))
            if self.bus:
                self.bus.emit(Event(EventType.ERROR_DETECTED, "automation", {"automation": auto.name,
                                                                             "error": str(exc)},
                                    severity=Severity.WARNING))

    async def _task_action(self, action: dict[str, Any], auto: Automation, event: Event | None) -> None:
        steps = [Step(s.get("description") or s["tool"], s["tool"], s.get("args", {}), allow_failure=s.get("allow_failure", False))
                 for s in action.get("steps", [])]
        objective = action.get("objective") or auto.name
        if event is not None and event.payload.get("title"):
            objective = objective.replace("{title}", str(event.payload["title"]))
        self.tasks.create_task(objective, title=action.get("title") or objective, steps=steps or None,
                               priority=Priority[action.get("priority", "P3")], created_by=f"automation:{auto.id}",
                               authority={"interactive": False}, cwd=action.get("cwd"),
                               success_condition=action.get("success_condition"),
                               dependencies=[event.task_id] if event is not None and event.task_id and
                               action.get("after_trigger_task") else None)


def notify_action(notifier: Any) -> ActionHandler:
    async def handler(action: dict[str, Any], auto: Automation, event: Event | None) -> None:
        priority = NotificationPriority[action.get("priority", "IMPORTANT").upper()]
        title = action.get("title", auto.name)
        if event is not None:
            title = title.replace("{title}", str(event.payload.get("title", "")))
        notifier.notify(priority, title, action.get("body", ""), source=f"automation:{auto.id}")
    return handler


def _row(r: Any) -> Automation:
    return Automation(r["id"], r["name"], r["kind"], loads(r["spec"], {}), bool(r["enabled"]), r["owner"],
                      r["created_at"], r["last_run"], r["next_run"], r["run_count"])
