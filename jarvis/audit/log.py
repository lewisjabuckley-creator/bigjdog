"""Action log / audit trail (spec §46).

Every tool execution and every autonomous control decision is recorded with its
authorization, verification and operational reason, so JARVIS can answer
"what did you do?" and "why did you do that?" from facts rather than recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jarvis.clock import Clock, SystemClock
from jarvis.core.types import OperationalReason, new_id
from jarvis.database.db import Database, dumps, loads
from jarvis.security.redaction import redact


@dataclass
class AuditEntry:
    id: str
    ts: float
    actor: str
    action: str
    tool: str | None = None
    task_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    ok: bool | None = None
    outcome: str | None = None
    summary: str = ""
    verification: dict[str, Any] | None = None
    authorization: dict[str, Any] | None = None
    reason: dict[str, Any] | None = None
    model: str | None = None
    agent: str | None = None
    rollback: dict[str, Any] | None = None
    duration_s: float | None = None
    dry_run: bool = False

    def reason_sentence(self) -> str | None:
        if not self.reason:
            return None
        return OperationalReason(self.reason.get("condition", ""), self.reason.get("rule", ""),
                                 self.reason.get("action", self.action), self.reason.get("expected", "")).sentence()


class AuditLog:
    def __init__(self, db: Database, clock: Clock | None = None) -> None:
        self.db = db
        self.clock = clock or SystemClock()

    def record(self, *, actor: str, action: str, tool: str | None = None, task_id: str | None = None,
               params: dict[str, Any] | None = None, ok: bool | None = None, outcome: str | None = None,
               summary: str = "", verification: dict[str, Any] | None = None,
               authorization: dict[str, Any] | None = None, reason: OperationalReason | dict | None = None,
               model: str | None = None, agent: str | None = None, rollback: dict[str, Any] | None = None,
               duration_s: float | None = None, dry_run: bool = False) -> AuditEntry:
        reason_dict = reason.to_dict() if isinstance(reason, OperationalReason) else reason
        entry = AuditEntry(new_id("act"), self.clock.now(), actor, action, tool, task_id,
                           redact(params or {}), ok, outcome, redact(summary), verification, authorization,
                           reason_dict, model, agent, rollback, duration_s, dry_run)
        self.db.execute(
            "INSERT INTO audit(id, ts, actor, task_id, action, tool, params, ok, outcome, summary, verification, "
            "authorization, reason, model, agent, rollback, duration_s, dry_run) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (entry.id, entry.ts, actor, task_id, action, tool, dumps(entry.params),
             None if ok is None else int(ok), outcome, entry.summary, dumps(verification) if verification else None,
             dumps(authorization) if authorization else None, dumps(reason_dict) if reason_dict else None,
             model, agent, dumps(rollback) if rollback else None, duration_s, int(dry_run)),
        )
        return entry

    def query(self, *, since: float | None = None, task_id: str | None = None, actor: str | None = None,
              action: str | None = None, include_dry_runs: bool = False, limit: int = 50) -> list[AuditEntry]:
        clauses, params = [], []
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        if actor:
            clauses.append("actor = ?")
            params.append(actor)
        if action:
            clauses.append("action = ?")
            params.append(action)
        if not include_dry_runs:
            clauses.append("dry_run = 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.query(f"SELECT * FROM audit {where} ORDER BY ts DESC, rowid DESC LIMIT ?", (*params, limit))
        return [_row(r) for r in rows]

    def get(self, entry_id: str) -> AuditEntry | None:
        row = self.db.query_one("SELECT * FROM audit WHERE id=?", (entry_id,))
        return _row(row) if row else None

    def last_with_reason(self, *, autonomous_only: bool = False) -> AuditEntry | None:
        sql = "SELECT * FROM audit WHERE reason IS NOT NULL AND dry_run = 0"
        if autonomous_only:
            sql += " AND actor NOT LIKE 'user:%'"
        row = self.db.query_one(sql + " ORDER BY ts DESC, rowid DESC LIMIT 1")
        return _row(row) if row else None


    def last_decision(self, since: float | None = None) -> AuditEntry | None:
        """Most recent autonomous control decision (pause, resume, retry, recovery...) made by JARVIS itself."""
        sql = "SELECT * FROM audit WHERE reason IS NOT NULL AND actor LIKE 'system:%'"
        params: list[Any] = []
        if since is not None:
            sql += " AND ts >= ?"
            params.append(since)
        row = self.db.query_one(sql + " ORDER BY ts DESC, rowid DESC LIMIT 1", params)
        return _row(row) if row else None


def _row(r: Any) -> AuditEntry:
    return AuditEntry(r["id"], r["ts"], r["actor"], r["action"], r["tool"], r["task_id"], loads(r["params"], {}),
                      None if r["ok"] is None else bool(r["ok"]), r["outcome"], r["summary"] or "",
                      loads(r["verification"]), loads(r["authorization"]), loads(r["reason"]), r["model"],
                      r["agent"], loads(r["rollback"]), r["duration_s"], bool(r["dry_run"]))
