"""Notifications and the interrupt policy (spec §11-12, §76-77).

The user's attention is a resource. Every notification carries a priority,
expiry and acknowledgement state; the interrupt policy combines priority, mode,
quiet state and what the user is doing to decide: deliver now, queue for the
next suitable moment, or record silently. Repeats are deduplicated and counted
("it has failed 4 times in 20 minutes") instead of re-announced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from jarvis.clock import Clock, SystemClock, format_duration
from jarvis.config import NotificationsConfig
from jarvis.core.modes import ModeManager
from jarvis.core.types import NotificationPriority, Severity, new_id
from jarvis.database.db import Database
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType

NP = NotificationPriority
Sink = Callable[["Notification"], None]


@dataclass
class Notification:
    id: str
    priority: NotificationPriority
    title: str
    body: str = ""
    source: str = ""
    task_id: str | None = None
    dedupe_key: str | None = None
    count: int = 1
    state: str = "new"             # delivered | queued | logged | expired | acknowledged
    ts: float = 0.0
    first_ts: float = 0.0
    expires_at: float | None = None
    delivered_at: float | None = None
    acknowledged_at: float | None = None

    def text(self) -> str:
        if not self.body:
            return self.title
        sep = " " if self.title.endswith((".", "!", "?", ":")) else ". "
        return f"{self.title}{sep}{self.body}"


class NotificationManager:
    def __init__(self, db: Database, modes: ModeManager, *, config: NotificationsConfig | None = None,
                 bus: EventBus | None = None, clock: Clock | None = None) -> None:
        self.db = db
        self.modes = modes
        self.config = config or NotificationsConfig()
        self.bus = bus
        self.clock = clock or SystemClock()
        self.sinks: list[Sink] = []
        self.last_user_input: float | None = None
        self.user_activity = "idle"     # idle | typing | speaking | away
        self._recent_interrupts: list[float] = []
        self._by_key: dict[str, Notification] = {}
        self._queue: dict[str, Notification] = {}

    # -- attention ------------------------------------------------------------------
    def on_user_input(self) -> None:
        self.last_user_input = self.clock.now()

    def set_activity(self, activity: str) -> None:
        self.user_activity = activity

    def user_busy(self) -> bool:
        # "conversing": JARVIS is answering the user right now; the answer is the right place for news.
        return self.user_activity in ("typing", "speaking", "conversing", "presenting")

    def user_idle_for(self) -> float:
        return float("inf") if self.last_user_input is None else self.clock.now() - self.last_user_input

    # -- policy -----------------------------------------------------------------------
    def decide(self, n: Notification) -> str:
        if n.priority <= NP.DEBUG:
            return "logged"
        if n.priority == NP.CRITICAL:
            return "delivered"
        policy = self.modes.effective().policy
        threshold = policy.busy_interrupt_min if self.user_busy() else policy.interrupt_min
        if n.priority >= threshold:
            if self._rate_limited():
                return "queued"
            return "delivered"
        if n.priority == NP.INFORMATIONAL:
            return "logged"
        return "queued"

    def _rate_limited(self) -> bool:
        now = self.clock.now()
        window = self.config.interrupt_window_s
        self._recent_interrupts = [t for t in self._recent_interrupts if now - t < window]
        return len(self._recent_interrupts) >= self.config.max_interrupts_per_window

    # -- API --------------------------------------------------------------------------
    def notify(self, priority: NotificationPriority, title: str, body: str = "", *, source: str = "jarvis",
               task_id: str | None = None, dedupe_key: str | None = None,
               expires_in: float | None = None) -> Notification:
        now = self.clock.now()
        if dedupe_key and dedupe_key in self._by_key:
            prev = self._by_key[dedupe_key]
            if now - prev.ts < self.config.dedupe_window_s and prev.state != "acknowledged":
                prev.count += 1
                prev.ts = now
                prev.body = body
                prev.priority = max(prev.priority, priority)
                span = format_duration(now - prev.first_ts)
                prev.title = f"{title} (again — {prev.count} times in {span})"
                # repeated problems escalate one level, but never to CRITICAL by repetition alone
                if prev.count >= 3 and prev.priority < NP.URGENT:
                    prev.priority = NotificationPriority(prev.priority + 1)
                self._route(prev)
                return prev
        n = Notification(new_id("ntf"), NotificationPriority(priority), title, body, source, task_id, dedupe_key,
                         1, "new", now, now, now + expires_in if expires_in else None)
        if dedupe_key:
            self._by_key[dedupe_key] = n
        self._route(n)
        return n

    def _route(self, n: Notification) -> None:
        decision = self.decide(n)
        n.state = decision
        if decision == "delivered":
            n.delivered_at = self.clock.now()
            self._recent_interrupts.append(n.delivered_at)
            self._queue.pop(n.id, None)
            self._deliver(n)
        elif decision == "queued":
            self._queue[n.id] = n
        self._persist(n)

    def _deliver(self, n: Notification) -> None:
        for sink in list(self.sinks):
            try:
                sink(n)
            except Exception:
                pass

    def _persist(self, n: Notification) -> None:
        self.db.execute(
            "INSERT INTO notifications(id, ts, priority, title, body, source, task_id, dedupe_key, count, state, "
            "expires_at, delivered_at, acknowledged_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET ts=excluded.ts, priority=excluded.priority, title=excluded.title, "
            "body=excluded.body, count=excluded.count, state=excluded.state, delivered_at=excluded.delivered_at, "
            "acknowledged_at=excluded.acknowledged_at",
            (n.id, n.ts, int(n.priority), n.title, n.body, n.source, n.task_id, n.dedupe_key, n.count, n.state,
             n.expires_at, n.delivered_at, n.acknowledged_at))

    def pending(self) -> list[Notification]:
        now = self.clock.now()
        for n in list(self._queue.values()):
            if n.expires_at and n.expires_at < now:
                n.state = "expired"
                self._queue.pop(n.id)
                self._persist(n)
        return sorted(self._queue.values(), key=lambda n: (-int(n.priority), n.ts))

    def drain(self, limit: int = 5, *, min_priority: NotificationPriority = NP.IMPORTANT) -> list[Notification]:
        """Deliver queued notifications at a suitable moment (e.g. right after answering the user)."""
        out = []
        for n in self.pending():
            if n.priority < min_priority or len(out) >= limit:
                continue
            n.state = "delivered"
            n.delivered_at = self.clock.now()
            self._queue.pop(n.id, None)
            self._persist(n)
            out.append(n)
        return out

    def drain_if_idle(self, idle_after_s: float = 30.0) -> list[Notification]:
        policy = self.modes.effective().policy
        if not policy.deliver_queued_when_idle or self.user_busy() or self.user_idle_for() < idle_after_s:
            return []
        items = self.drain()
        for n in items:
            self._deliver(n)
        return items

    def acknowledge(self, notification_id: str | None = None) -> int:
        now = self.clock.now()
        targets = [n for n in self._by_key.values() if notification_id in (None, n.id)]
        targets += [n for n in self._queue.values() if notification_id in (None, n.id) and n not in targets]
        for n in targets:
            n.state = "acknowledged"
            n.acknowledged_at = now
            self._queue.pop(n.id, None)
            self._persist(n)
        return len(targets)

    def acknowledge_task(self, task_id: str) -> int:
        """Mark queued notifications about a task as seen (e.g. its result was just reported inline)."""
        count = 0
        for n in list(self._queue.values()):
            if n.task_id == task_id:
                n.state = "acknowledged"
                n.acknowledged_at = self.clock.now()
                self._queue.pop(n.id, None)
                self._persist(n)
                count += 1
        return count

    def history(self, limit: int = 20, min_priority: NotificationPriority = NP.INFORMATIONAL) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM notifications WHERE priority >= ? ORDER BY ts DESC, rowid DESC LIMIT ?",
                             (int(min_priority), limit))
        return [dict(r) for r in rows]

    # -- event rules --------------------------------------------------------------------
    def attach(self, bus: EventBus) -> None:
        bus.subscribe("*", self._on_event, name="notifications")

    def _on_event(self, e: Event) -> None:
        rule = self._rule(e)
        if rule is None:
            return
        priority, title, body, key = rule
        self.notify(priority, title, body, source=e.source, task_id=e.task_id, dedupe_key=key)

    def _rule(self, e: Event) -> tuple[NotificationPriority, str, str, str | None] | None:
        p = e.payload
        t = e.type
        user_task = str(p.get("created_by", "user")).startswith("user")
        title = p.get("title", "the task")
        if t == EventType.TASK_COMPLETED:
            if p.get("task_kind") == "monitor":
                return None     # monitors report through MONITOR_TRIGGERED
            outcome = p.get("outcome")
            prio = NP.IMPORTANT if user_task else NP.INFORMATIONAL
            label = {"partial": "partly finished", "unknown": "finished (unverified)"}.get(outcome, "finished")
            return prio, f"{_cap(title)} {label}", p.get("reason", ""), None
        if t == EventType.TASK_FAILED:
            return (NP.URGENT if user_task else NP.IMPORTANT), f"{_cap(title)} failed", p.get("reason", ""), \
                f"task-failed:{e.task_id}"
        if t == EventType.TASK_WAITING and "approval" in p.get("reason", ""):
            return NP.URGENT, f"{_cap(title)} needs your approval", p.get("reason", ""), f"approval:{e.task_id}"
        if t == EventType.TASK_BLOCKED:
            return NP.IMPORTANT, f"{_cap(title)} is blocked", p.get("reason", ""), f"blocked:{e.task_id}"
        if t == EventType.TASK_INTERRUPTED:
            return NP.IMPORTANT, f"{_cap(title)} was interrupted", p.get("summary", ""), f"interrupted:{e.task_id}"
        if t == EventType.MONITOR_TRIGGERED:
            prio = _priority(p.get("notify", "important"))
            if e.severity >= Severity.ERROR:
                prio = max(prio, NP.URGENT)
            return prio, _cap(p.get("detail", "monitor triggered")), "", f"monitor:{e.task_id}:{p.get('detail')}"
        if t == EventType.RESOURCE_THRESHOLD_EXCEEDED:
            prio = NP.CRITICAL if p.get("severity") == "critical" else NP.IMPORTANT
            return prio, p.get("message", "resource threshold exceeded"), "", f"threshold:{p.get('name')}"
        if t == EventType.RESOURCE_THRESHOLD_CLEARED:
            return NP.INFORMATIONAL, p.get("message", "resource back to normal"), "", None
        if t == EventType.PREDICTIVE_WARNING:
            return NP.IMPORTANT, p.get("message", "predicted problem"), "", f"predict:{p.get('metric')}"
        if t == EventType.TREND_DETECTED:
            return NP.INFORMATIONAL, p.get("message", "trend detected"), "", f"trend:{p.get('metric')}"
        if t == EventType.SUBSYSTEM_DEGRADED:
            prio = NP.URGENT if e.severity >= Severity.ERROR else NP.IMPORTANT
            return prio, f"{p.get('component')} is {p.get('status')}", p.get("detail", ""), f"subsystem:{p.get('component')}"
        if t == EventType.SUBSYSTEM_RECOVERED:
            return NP.INFORMATIONAL, f"{p.get('component')} recovered", "", None
        if t == EventType.MODEL_UNAVAILABLE:
            return NP.IMPORTANT, f"Model provider {p.get('provider')} is unavailable", p.get("error", ""), \
                f"model:{p.get('provider')}"
        if t == EventType.MODEL_FALLBACK:
            return NP.INFORMATIONAL, f"Used {p.get('used')} because {p.get('intended')} failed", "", None
        if t == EventType.SECURITY_EVENT:
            prio = NP.CRITICAL if e.severity >= Severity.CRITICAL else NP.URGENT
            return prio, "Security: " + str(p.get("reason", "security event")), "", f"security:{p.get('reason')}"
        if t == EventType.DEADLINE_AT_RISK:
            return NP.IMPORTANT, f"{_cap(title)} is unlikely to meet its deadline", "", f"deadline:{e.task_id}"
        if t == EventType.RESOURCE_THROTTLED:
            return NP.INFORMATIONAL, p.get("reason", "work throttled"), "", None
        if t == EventType.NETWORK_CHANGED:
            new = p.get("state")
            prio = NP.IMPORTANT if new in ("offline", "unstable") else NP.INFORMATIONAL
            return prio, f"Network is now {str(new).replace('_', ' ')}", "", "network"
        if t in (EventType.DEVICE_DISCONNECTED, EventType.DEVICE_FAILURE):
            return NP.IMPORTANT, f"{p.get('name', 'A device')} {'failed' if t == EventType.DEVICE_FAILURE else 'disconnected'}", \
                "", f"device:{p.get('name')}"
        if t == EventType.EMERGENCY_ENTERED:
            return NP.CRITICAL, "Emergency mode", p.get("reason", ""), "emergency"
        if t == EventType.EMERGENCY_EXITED:
            return NP.URGENT, "Emergency resolved", p.get("reason", ""), None
        if t in (EventType.CALL_RECEIVED, EventType.MESSAGE_RECEIVED):
            prio = _priority(p.get("priority", "important"))
            who = p.get("from", "unknown")
            kind = "call" if t == EventType.CALL_RECEIVED else "message"
            return prio, f"Incoming {kind} from {who}", p.get("summary", ""), f"{kind}:{who}"
        return None


def _priority(name: str) -> NotificationPriority:
    try:
        return NotificationPriority[str(name).upper()]
    except KeyError:
        return NP.IMPORTANT


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text
