"""Emergency mode as architecture, not personality (spec §28, §184).

On a critical event: identify critical state, deprioritise non-essential work
(the resource manager pauses P3+ while emergency mode is active), preserve
evidence, notify, take only authorised protective actions, keep monitoring, and
exit only when the triggering condition has cleared.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jarvis.audit.log import AuditLog
from jarvis.clock import Clock, SystemClock
from jarvis.core.modes import Mode, ModeManager
from jarvis.core.types import OperationalReason, Severity
from jarvis.events.bus import EventBus
from jarvis.events.store import EventStore
from jarvis.events.types import Event, EventType
from jarvis.state.engine import StateEngine


class EmergencyController:
    def __init__(self, modes: ModeManager, state: StateEngine, bus: EventBus, audit: AuditLog, *,
                 events: EventStore | None = None, clock: Clock | None = None, evidence_dir: str | None = None) -> None:
        self.modes = modes
        self.state = state
        self.bus = bus
        self.audit = audit
        self.events = events
        self.clock = clock or SystemClock()
        self.evidence_dir = Path(evidence_dir).expanduser() if evidence_dir else None
        self.trigger: dict[str, Any] | None = None

    def attach(self) -> None:
        self.bus.subscribe("*", self._on_event, name="emergency")

    @property
    def active(self) -> bool:
        return self.modes.current == Mode.EMERGENCY

    async def _on_event(self, e: Event) -> None:
        if e.type == EventType.RESOURCE_THRESHOLD_CLEARED and self.active and self.trigger \
                and self.trigger.get("metric") == e.payload.get("metric"):
            await self.exit(f"{e.payload.get('message', 'condition cleared')}")
            return
        if e.severity >= Severity.CRITICAL and not self.active and e.type in (
                EventType.RESOURCE_THRESHOLD_EXCEEDED, EventType.SECURITY_EVENT, EventType.DEVICE_FAILURE,
                EventType.SUBSYSTEM_DEGRADED):
            await self.enter(e)

    async def enter(self, trigger: Event | None = None, *, reason: str = "") -> dict[str, Any]:
        if self.active:
            return self.trigger or {}
        why = reason or (trigger.payload.get("message") or trigger.payload.get("reason") or str(trigger.type)
                         if trigger else "manual activation")
        self.trigger = {"type": str(trigger.type) if trigger else "manual", "reason": why,
                        "metric": trigger.payload.get("metric") if trigger else None, "at": self.clock.now()}
        critical_state = {k: f.value for k, f in self.state.snapshot("resources.").items()}
        critical_state.update({k: f.value for k, f in self.state.snapshot("network.").items()})
        evidence = self._preserve_evidence(critical_state)
        self.modes.set_mode(Mode.EMERGENCY, by="emergency", reason=why)
        self.audit.record(actor="system:emergency", action="enter_emergency", summary=why,
                          reason=OperationalReason(why, "critical events activate emergency mode",
                                                   "entered emergency mode and deprioritised non-essential work",
                                                   "Only authorised protective actions will run."),
                          rollback={"evidence": evidence} if evidence else None)
        self.bus.emit(Event(EventType.EMERGENCY_ENTERED, "emergency",
                            {"reason": why, "evidence": evidence}, severity=Severity.CRITICAL))
        return self.trigger

    async def exit(self, reason: str = "conditions cleared", *, by: str = "emergency") -> bool:
        if not self.active:
            return False
        previous = self.state.value("mode.before_emergency", "normal")
        try:
            target = Mode(previous)
        except ValueError:
            target = Mode.NORMAL
        self.modes.set_mode(target, by=by, reason=reason)
        self.audit.record(actor="system:emergency" if by == "emergency" else f"user:{by}", action="exit_emergency",
                          summary=reason, reason=OperationalReason(reason, "exit only when conditions permit",
                                                                   f"returned to {target.value} mode"))
        self.bus.emit(Event(EventType.EMERGENCY_EXITED, "emergency", {"reason": reason, "mode": target.value}))
        self.trigger = None
        return True

    def _preserve_evidence(self, critical_state: dict[str, Any]) -> str | None:
        if self.evidence_dir is None:
            return None
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        recent = [e.to_dict() for e in (self.events.query(limit=500) if self.events else self.bus.recent_events(500))]
        path = self.evidence_dir / f"emergency-{int(self.clock.now())}.json"
        path.write_text(json.dumps({"at": self.clock.now(), "trigger": self.trigger, "state": critical_state,
                                    "events": recent}, default=str, indent=1))
        return str(path)
