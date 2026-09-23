"""Instruction hierarchy for conflicting directives (spec §26).

When sources disagree (user says stop, an automation says continue, an agent
says resume), precedence is decided here — deterministically:

  1. safety/security constraints
  2. explicit current user instruction
  3. explicit delegated authority
  4. active task policy
  5. automation
  6. default behaviour

Within the same rank the newer directive wins. A lower-ranked source can never
override a higher-ranked one, however recent: stale autonomous instructions do
not beat an explicit user command.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable


class InstructionSource(IntEnum):
    SAFETY = 1
    USER = 2
    DELEGATED = 3
    TASK_POLICY = 4
    AUTOMATION = 5
    DEFAULT = 6


@dataclass(frozen=True)
class Directive:
    source: InstructionSource
    action: str                 # e.g. pause | resume | cancel | continue | halt
    issued_at: float
    issued_by: str = ""
    target: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"source": int(self.source), "action": self.action, "issued_at": self.issued_at,
                "issued_by": self.issued_by, "target": self.target}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Directive":
        return cls(InstructionSource(data["source"]), data["action"], float(data["issued_at"]),
                   data.get("issued_by", ""), data.get("target", ""))


def may_override(new: Directive, existing: Directive | None) -> bool:
    if existing is None:
        return True
    if new.source != existing.source:
        return new.source < existing.source
    return new.issued_at >= existing.issued_at


def resolve(directives: Iterable[Directive]) -> Directive | None:
    """The directive that currently governs a target."""
    best: Directive | None = None
    for d in directives:
        if best is None or d.source < best.source or (d.source == best.source and d.issued_at >= best.issued_at):
            best = d
    return best


def source_for_actor_kind(kind: str) -> InstructionSource:
    return {
        "safety": InstructionSource.SAFETY,
        "user": InstructionSource.USER,
        "delegated": InstructionSource.DELEGATED,
        "task": InstructionSource.TASK_POLICY,
        "agent": InstructionSource.TASK_POLICY,
        "automation": InstructionSource.AUTOMATION,
        "system": InstructionSource.DEFAULT,
    }.get(kind, InstructionSource.DEFAULT)
