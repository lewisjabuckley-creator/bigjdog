"""Decision history (spec §45): what was decided, why, and what happened."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from jarvis.clock import Clock, SystemClock
from jarvis.core.types import new_id
from jarvis.database.db import Database, dumps, loads


@dataclass
class DecisionRecord:
    id: str
    ts: float
    title: str
    decision: str
    context: str = ""
    alternatives: list[str] = field(default_factory=list)
    reason: str = ""
    user_involvement: str = ""
    outcome: str = ""
    project_id: str | None = None
    tags: list[str] = field(default_factory=list)

    def explain(self) -> str:
        text = f"We chose {self.decision}"
        if self.reason:
            text += f" because {self.reason.rstrip('.')}"
        text += "."
        if self.alternatives:
            text += f" Alternatives considered: {', '.join(self.alternatives)}."
        if self.user_involvement:
            text += f" ({self.user_involvement})"
        if self.outcome:
            text += f" Outcome so far: {self.outcome}."
        return text


class DecisionLog:
    def __init__(self, db: Database, clock: Clock | None = None) -> None:
        self.db = db
        self.clock = clock or SystemClock()

    def record(self, title: str, decision: str, *, context: str = "", alternatives: list[str] | None = None,
               reason: str = "", user_involvement: str = "", project_id: str | None = None,
               tags: list[str] | None = None) -> DecisionRecord:
        rec = DecisionRecord(new_id("dec"), self.clock.now(), title, decision, context, list(alternatives or []),
                             reason, user_involvement, "", project_id, list(tags or []))
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO decisions(id, ts, project_id, title, decision, context, alternatives, reason, "
                "user_involvement, outcome, tags) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (rec.id, rec.ts, project_id, title, decision, context, dumps(rec.alternatives), reason,
                 user_involvement, "", dumps(rec.tags)))
            self.db.execute("INSERT INTO decisions_fts(decision_id, title, decision, context, reason) VALUES(?,?,?,?,?)",
                            (rec.id, title, decision, context, reason))
        return rec

    def set_outcome(self, decision_id: str, outcome: str) -> None:
        self.db.execute("UPDATE decisions SET outcome=? WHERE id=?", (outcome, decision_id))

    def list(self, project_id: str | None = None, limit: int = 50) -> list[DecisionRecord]:
        if project_id:
            rows = self.db.query("SELECT * FROM decisions WHERE project_id=? ORDER BY ts DESC LIMIT ?", (project_id, limit))
        else:
            rows = self.db.query("SELECT * FROM decisions ORDER BY ts DESC LIMIT ?", (limit,))
        return [_row(r) for r in rows]

    def search(self, query: str, project_id: str | None = None, limit: int = 5) -> list[DecisionRecord]:
        terms = [w for w in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(w) > 2 and w not in
                 {"why", "did", "we", "the", "choose", "chose", "pick", "use", "decide", "decided", "was", "what"}]
        if not terms:
            return self.list(project_id, limit)
        sql = ("SELECT d.*, bm25(decisions_fts) rank FROM decisions_fts JOIN decisions d ON d.id = decisions_fts.decision_id "
               "WHERE decisions_fts MATCH ?")
        params: list[Any] = [" OR ".join(f'"{t}"*' for t in terms)]
        if project_id:
            sql += " AND (d.project_id = ? OR d.project_id IS NULL)"
            params.append(project_id)
        return [_row(r) for r in self.db.query(sql + " ORDER BY rank LIMIT ?", (*params, limit))]


def _row(r: Any) -> DecisionRecord:
    return DecisionRecord(r["id"], r["ts"], r["title"], r["decision"], r["context"] or "", loads(r["alternatives"], []),
                          r["reason"] or "", r["user_involvement"] or "", r["outcome"] or "", r["project_id"],
                          loads(r["tags"], []))
