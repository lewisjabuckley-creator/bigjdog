"""Permission manager: capability is not authority (spec §23).

Every action is checked against (in order):
  1. safety constraints — absolute, never overridden by grants;
  2. scope — filesystem roots, denied paths, project isolation;
  3. authority — baseline level for the actor, then explicit grants;
  4. otherwise: ask the user (interactive) or deny (autonomous).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from jarvis.clock import Clock, SystemClock
from jarvis.config import PermissionsConfig
from jarvis.core.types import RiskLevel, Severity, new_id
from jarvis.database.db import Database, dumps, loads
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.permissions.model import AccessRequest, Actor, Decision, Grant, PermissionLevel, _within


class PathPolicy:
    """Filesystem scope. Paths are resolved (symlinks included) before checking."""

    def __init__(self, allowed_roots: list[str], denied_paths: list[str]) -> None:
        self.allowed_roots = [self._norm(p) for p in allowed_roots]
        self.denied_paths = [self._norm(p) for p in denied_paths]

    @staticmethod
    def _norm(path: str) -> str:
        return os.path.realpath(os.path.expanduser(path))

    def resolve(self, path: str, cwd: str | None = None) -> str:
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            path = os.path.join(cwd or os.getcwd(), path)
        return os.path.realpath(path)

    def check(self, resolved: str, *, allowed_dirs: list[str] | None = None,
              granted_paths: list[str] | None = None) -> tuple[bool, str]:
        granted = [self._norm(p) for p in (granted_paths or [])]
        for denied in self.denied_paths:
            if _within(resolved, denied) and not any(_within(resolved, g) and _within(g, denied) for g in granted):
                return False, f"{resolved} is inside protected path {denied}"
        if not any(_within(resolved, root) for root in self.allowed_roots) \
                and not any(_within(resolved, g) for g in granted):
            return False, f"{resolved} is outside the allowed roots"
        if allowed_dirs is not None:
            dirs = [self._norm(d) for d in allowed_dirs]
            if not any(_within(resolved, d) for d in dirs) and not any(_within(resolved, g) for g in granted):
                return False, f"{resolved} is outside the active project's allowed directories"
        return True, ""


class PermissionManager:
    def __init__(self, db: Database, config: PermissionsConfig | None = None, *,
                 bus: EventBus | None = None, clock: Clock | None = None) -> None:
        self.db = db
        self.config = config or PermissionsConfig()
        self.bus = bus
        self.clock = clock or SystemClock()
        self.paths = PathPolicy(self.config.allowed_roots, self.config.denied_paths)

    # -- grants -----------------------------------------------------------------
    def grant(self, subject: str, level: PermissionLevel, *, tools: list[str] | None = None,
              paths: list[str] | None = None, project_id: str | None = None, task_id: str | None = None,
              ttl_s: float | None = None, max_uses: int | None = None, reason: str = "",
              created_by: str = "owner") -> Grant:
        now = self.clock.now()
        g = Grant(new_id("grant"), subject, PermissionLevel(level), tools or ["*"], paths or [], project_id,
                  task_id, now + ttl_s if ttl_s else None, max_uses, 0, reason, created_by, now)
        self.db.execute(
            "INSERT INTO grants(id, subject, level, tools, paths, project_id, task_id, expires_at, max_uses, uses, "
            "reason, created_by, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (g.id, g.subject, int(g.level), dumps(g.tools), dumps(g.paths), g.project_id, g.task_id,
             g.expires_at, g.max_uses, 0, g.reason, g.created_by, g.created_at),
        )
        self._emit(EventType.PERMISSION_GRANTED, {"grant_id": g.id, "subject": subject, "scope": g.describe(),
                                                  "reason": reason})
        return g

    def revoke(self, grant_id: str) -> bool:
        changed = self.db.execute("UPDATE grants SET revoked_at=? WHERE id=? AND revoked_at IS NULL",
                                  (self.clock.now(), grant_id))
        if changed:
            self._emit(EventType.PERMISSION_REVOKED, {"grant_id": grant_id})
        return bool(changed)

    def revoke_for(self, *, task_id: str | None = None, subject: str | None = None) -> int:
        grants = [g for g in self.list_grants() if (task_id and g.task_id == task_id) or (subject and g.subject == subject)]
        return sum(self.revoke(g.id) for g in grants)

    def list_grants(self, active_only: bool = True) -> list[Grant]:
        now = self.clock.now()
        grants = [_row_to_grant(r) for r in self.db.query("SELECT * FROM grants ORDER BY created_at")]
        return [g for g in grants if g.active(now)] if active_only else grants

    def granted_paths(self, actor: Actor, tool: str, task_id: str | None) -> list[str]:
        """Paths explicitly delegated to this actor for this tool (can reach into protected locations)."""
        subjects = self._subjects(actor, task_id)
        out: list[str] = []
        for g in self.list_grants():
            if g.paths and g.covers(subjects=subjects, tool=tool, paths=[], project_id=g.project_id,
                                    task_id=task_id):
                out.extend(g.paths)
        return out

    # -- decisions ----------------------------------------------------------------
    def baseline(self, actor: Actor) -> PermissionLevel:
        if actor.interactive:
            return PermissionLevel(self.config.interactive_level)
        return PermissionLevel(self.config.automation_level)

    def check(self, request: AccessRequest, *, consume: bool = True) -> Decision:
        baseline = self.baseline(request.actor)
        if request.level <= baseline:
            return Decision(True, reason=f"within {baseline.label} baseline for {request.actor.kind}",
                            basis="baseline")
        subjects = self._subjects(request.actor, request.task_id)
        now = self.clock.now()
        for g in self.list_grants():
            if g.level >= request.level and g.active(now) and g.covers(
                    subjects=subjects, tool=request.tool, paths=request.paths,
                    project_id=request.project_id, task_id=request.task_id):
                if consume:
                    self.db.execute("UPDATE grants SET uses = uses + 1 WHERE id=?", (g.id,))
                return Decision(True, reason=f"delegated: {g.describe()}", basis="grant", grant_id=g.id)
        if request.actor.interactive:
            return Decision(False, needs_approval=True, basis="permission",
                            reason=f"{request.tool} requires {request.level.label} authorization")
        self._emit(EventType.PERMISSION_DENIED, {"tool": request.tool, "actor": request.actor.subject,
                                                 "level": request.level.label}, Severity.WARNING)
        return Decision(False, basis="permission",
                        reason=f"{request.actor.kind} lacks {request.level.label} authority for {request.tool}")

    @staticmethod
    def _subjects(actor: Actor, task_id: str | None) -> set[str]:
        subjects = {actor.subject, f"user:{actor.on_behalf_of}"} if actor.kind == "user" else {actor.subject}
        if task_id:
            subjects.add(f"task:{task_id}")
        return subjects

    def _emit(self, etype: EventType, payload: dict, severity: Severity = Severity.INFO) -> None:
        if self.bus is not None:
            self.bus.emit(Event(etype, "permissions", payload, severity=severity))


# -- approvals -----------------------------------------------------------------------

@dataclass
class ApprovalRequest:
    id: str
    tool: str
    args: dict
    summary: str
    risk: RiskLevel
    level: PermissionLevel
    requested_by: str
    status: str = "pending"          # pending | approved | denied | expired
    task_id: str | None = None
    step_id: str | None = None
    reason: str = ""
    created_at: float = 0.0
    decided_at: float | None = None
    decided_by: str | None = None
    expires_at: float | None = None
    meta: dict = field(default_factory=dict)


class ApprovalManager:
    """Pending authorization requests. Approval creates a single-use, task-scoped grant."""

    def __init__(self, db: Database, permissions: PermissionManager, *, bus: EventBus | None = None,
                 clock: Clock | None = None) -> None:
        self.db = db
        self.permissions = permissions
        self.bus = bus
        self.clock = clock or SystemClock()

    def request(self, *, tool: str, args: dict, summary: str, risk: RiskLevel, level: PermissionLevel,
                requested_by: str, task_id: str | None = None, step_id: str | None = None,
                reason: str = "") -> ApprovalRequest:
        # One pending request per task step: re-requesting returns the existing one.
        if task_id and step_id:
            existing = self.db.query_one("SELECT * FROM approvals WHERE task_id=? AND step_id=? AND status='pending'",
                                         (task_id, step_id))
            if existing:
                return _row_to_approval(existing)
        now = self.clock.now()
        timeout = self.permissions.config.approval_timeout_s
        req = ApprovalRequest(new_id("appr"), tool, args, summary, RiskLevel(risk), PermissionLevel(level),
                              requested_by, "pending", task_id, step_id, reason, now,
                              expires_at=now + timeout if timeout else None)
        self.db.execute(
            "INSERT INTO approvals(id, task_id, step_id, tool, args, summary, risk, level, reason, requested_by, "
            "status, created_at, expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (req.id, task_id, step_id, tool, dumps(args), summary, int(req.risk), int(req.level), reason,
             requested_by, "pending", now, req.expires_at),
        )
        if self.bus:
            self.bus.emit(Event(EventType.APPROVAL_REQUESTED, "permissions",
                                {"approval_id": req.id, "tool": tool, "summary": summary, "risk": req.risk.name.lower()},
                                severity=Severity.WARNING, task_id=task_id))
        return req

    def get(self, approval_id: str) -> ApprovalRequest | None:
        row = self.db.query_one("SELECT * FROM approvals WHERE id=?", (approval_id,))
        return _row_to_approval(row) if row else None

    def pending(self) -> list[ApprovalRequest]:
        self.expire()
        return [_row_to_approval(r) for r in
                self.db.query("SELECT * FROM approvals WHERE status='pending' ORDER BY created_at")]

    def approve(self, approval_id: str, by: str = "owner") -> Grant | None:
        req = self.get(approval_id)
        if req is None or req.status != "pending":
            return None
        self._decide(req, "approved", by)
        subject = f"task:{req.task_id}" if req.task_id else f"user:{by}"
        return self.permissions.grant(subject, req.level, tools=[req.tool], task_id=req.task_id, max_uses=1,
                                      ttl_s=self.permissions.config.approval_timeout_s,
                                      reason=f"approved: {req.summary}", created_by=by)

    def deny(self, approval_id: str, by: str = "owner") -> bool:
        req = self.get(approval_id)
        if req is None or req.status != "pending":
            return False
        self._decide(req, "denied", by)
        return True

    def cancel_for_task(self, task_id: str) -> int:
        return self.db.execute("UPDATE approvals SET status='expired', decided_at=? WHERE task_id=? AND status='pending'",
                               (self.clock.now(), task_id))

    def expire(self) -> int:
        return self.db.execute("UPDATE approvals SET status='expired' WHERE status='pending' AND expires_at IS NOT NULL "
                               "AND expires_at < ?", (self.clock.now(),))

    def _decide(self, req: ApprovalRequest, status: str, by: str) -> None:
        self.db.execute("UPDATE approvals SET status=?, decided_at=?, decided_by=? WHERE id=?",
                        (status, self.clock.now(), by, req.id))
        if self.bus:
            self.bus.emit(Event(EventType.APPROVAL_DECIDED, "permissions",
                                {"approval_id": req.id, "decision": status, "tool": req.tool, "by": by},
                                task_id=req.task_id))


def _row_to_grant(r) -> Grant:
    return Grant(r["id"], r["subject"], PermissionLevel(r["level"]), loads(r["tools"], ["*"]), loads(r["paths"], []),
                 r["project_id"], r["task_id"], r["expires_at"], r["max_uses"], r["uses"], r["reason"] or "",
                 r["created_by"], r["created_at"], r["revoked_at"])


def _row_to_approval(r) -> ApprovalRequest:
    return ApprovalRequest(r["id"], r["tool"], loads(r["args"], {}), r["summary"], RiskLevel(r["risk"]),
                           PermissionLevel(r["level"]), r["requested_by"], r["status"], r["task_id"], r["step_id"],
                           r["reason"] or "", r["created_at"], r["decided_at"], r["decided_by"], r["expires_at"])
