"""Permission vocabulary (spec §22-23, §146)."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class PermissionLevel(IntEnum):
    OBSERVE = 0                 # inspect
    RECOMMEND = 1               # propose actions
    PREPARE = 2                 # draft commands, files, messages, plans
    EXECUTE_REVERSIBLE = 3      # low-risk, reversible operations
    EXECUTE_CONSEQUENTIAL = 4   # needs explicit authorization unless delegated
    AUTONOMOUS = 5              # bounded independent operation inside explicit limits

    @property
    def label(self) -> str:
        return self.name.lower().replace("_", " ")


@dataclass(frozen=True)
class Actor:
    """Who is asking for an action.

    ``interactive`` marks actions that directly serve an explicit, current user
    request: the request itself confers scoped authority (spec §145) and the user
    is available to approve anything beyond it.
    """

    kind: str                   # user | task | agent | automation | system
    id: str
    on_behalf_of: str = "owner"
    interactive: bool = False

    @property
    def subject(self) -> str:
        return f"{self.kind}:{self.id}"

    @classmethod
    def user(cls, user_id: str = "owner") -> "Actor":
        return cls("user", user_id, user_id, interactive=True)

    @classmethod
    def system(cls) -> "Actor":
        return cls("system", "jarvis", "owner", interactive=False)


@dataclass
class Grant:
    """Scoped, inspectable, revocable, time-limited delegation of authority."""

    id: str
    subject: str                # "user:owner", "task:<id>", "agent:<id>", "automation:<id>", "*"
    level: PermissionLevel
    tools: list[str] = field(default_factory=lambda: ["*"])   # glob patterns
    paths: list[str] = field(default_factory=list)            # path prefixes; empty = any allowed path
    project_id: str | None = None
    task_id: str | None = None
    expires_at: float | None = None
    max_uses: int | None = None
    uses: int = 0
    reason: str = ""
    created_by: str = "owner"
    created_at: float = 0.0
    revoked_at: float | None = None

    def active(self, now: float) -> bool:
        if self.revoked_at is not None:
            return False
        if self.expires_at is not None and now >= self.expires_at:
            return False
        if self.max_uses is not None and self.uses >= self.max_uses:
            return False
        return True

    def covers(self, *, subjects: set[str], tool: str, paths: list[str], project_id: str | None,
               task_id: str | None) -> bool:
        if self.subject != "*" and self.subject not in subjects:
            return False
        if not any(fnmatch.fnmatchcase(tool, pattern) for pattern in self.tools):
            return False
        if self.project_id and self.project_id != project_id:
            return False
        if self.task_id and self.task_id != task_id:
            return False
        if self.paths:
            prefixes = [os.path.realpath(os.path.expanduser(p)) for p in self.paths]
            for path in paths:
                if not any(_within(path, prefix) for prefix in prefixes):
                    return False
        return True

    def describe(self) -> str:
        scope = ", ".join(self.tools)
        if self.paths:
            scope += f" in {', '.join(self.paths)}"
        text = f"{self.level.label} for {scope}"
        if self.expires_at:
            text += " (time-limited)"
        if self.max_uses:
            text += f" ({self.max_uses - self.uses} use(s) left)"
        return text


@dataclass
class AccessRequest:
    actor: Actor
    tool: str
    level: PermissionLevel
    paths: list[str] = field(default_factory=list)
    project_id: str | None = None
    task_id: str | None = None
    summary: str = ""


@dataclass
class Decision:
    allowed: bool
    needs_approval: bool = False
    reason: str = ""
    basis: str = ""             # baseline | grant | safety | scope | mode | permission
    grant_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "needs_approval": self.needs_approval, "reason": self.reason,
                "basis": self.basis, "grant_id": self.grant_id}


def _within(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip(os.sep) + os.sep)


def within(path: str, prefix: str) -> bool:
    return _within(os.path.realpath(os.path.expanduser(path)), os.path.realpath(os.path.expanduser(prefix)))
