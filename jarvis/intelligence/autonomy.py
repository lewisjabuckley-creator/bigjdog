"""Autonomy levels (Phase 3 §20, §38-39).

Autonomy decides how much JARVIS does *on its own initiative* and how much it checks with you first. It never
decides what is permitted: the permission system is consulted for every action at every level, and nothing in
this module is read by it. Higher autonomy can only mean fewer questions about things that are already
allowed, never access to things that aren't.

* LOW — every plan is shown before it starts; nothing happens on JARVIS's own initiative.
* NORMAL — plans with consequential steps are previewed; events produce suggestions, not actions.
* HIGH — plans start without a preview; notable events are investigated (observe-only) and the findings offered.
* EMERGENCY — follows emergency mode: urgent work first, investigations start by themselves; approvals are still
  required for anything consequential.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from jarvis.core.types import OperationalReason
from jarvis.events.types import Event, EventType


class AutonomyLevel(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EMERGENCY = "emergency"


_REACTIONS = {AutonomyLevel.LOW: "off", AutonomyLevel.NORMAL: "suggest", AutonomyLevel.HIGH: "investigate",
              AutonomyLevel.EMERGENCY: "investigate"}


class Autonomy:
    def __init__(self, state: Any, config: Any, *, modes: Any = None, bus: Any = None, audit: Any = None) -> None:
        self.state = state
        self.config = config
        self.modes = modes
        self.bus = bus
        self.audit = audit

    @property
    def level(self) -> AutonomyLevel:
        if self.modes is not None and getattr(self.modes.current, "value", "") == "emergency":
            return AutonomyLevel.EMERGENCY
        stored = self.state.value("intelligence.autonomy") if self.state is not None else None
        try:
            return AutonomyLevel(stored or self.config.autonomy)
        except ValueError:
            return AutonomyLevel.NORMAL

    def set(self, level: str, *, by: str = "user") -> AutonomyLevel:
        new = AutonomyLevel(level)
        if new == AutonomyLevel.EMERGENCY:
            raise ValueError("emergency autonomy follows emergency mode; it can't be set directly")
        old = self.level
        self.state.set("intelligence.autonomy", new.value)
        if self.bus is not None:
            self.bus.emit(Event(EventType.AUTONOMY_CHANGED, "planner", {"from": old.value, "to": new.value, "by": by}))
        if self.audit is not None:
            self.audit.record(actor=by if ":" in by else f"user:{by}", action="autonomy_changed",
                              summary=f"autonomy {old.value} → {new.value}",
                              reason=OperationalReason(f"{by} set autonomy to {new.value}",
                                                       "autonomy is the user's choice", f"set autonomy to {new.value}",
                                                       "Permissions are unchanged."))
        return new

    def reactions(self) -> str:
        if not getattr(self.config, "reactions", True):
            return "off"
        return _REACTIONS[self.level]

    def describe(self) -> str:
        level = self.level
        what = {
            AutonomyLevel.LOW: "I show you every plan before starting it and do nothing on my own initiative",
            AutonomyLevel.NORMAL: "I show you plans with consequential steps first, and turn events into "
                                  "suggestions",
            AutonomyLevel.HIGH: "I start plans without a preview and look into notable events by myself "
                                "(observing only)",
            AutonomyLevel.EMERGENCY: "emergency mode: urgent work first, and I investigate problems as they appear",
        }[level]
        return (f"Autonomy is {level.value}: {what}. Anything consequential still needs your approval — autonomy "
                "never changes what's permitted.")
