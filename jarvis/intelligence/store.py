"""Durable goals and plans (Phase 3 §6, §31-32).

Every plan change is written to SQLite before anything acts on it: the goal, the graph, which task carries
which node, completed and pending nodes, facts, assumptions, decisions, failures, approvals and the resources
it holds. A replan first stores the previous graph as a revision, so plan history can say what changed and why.
"""

from __future__ import annotations

from typing import Iterable

from jarvis.clock import Clock, SystemClock
from jarvis.database.db import Database, dumps, loads
from jarvis.intelligence.goals import Goal
from jarvis.intelligence.plans import PLAN_OPEN, Plan, PlanStatus


class PlanStore:
    def __init__(self, db: Database, clock: Clock | None = None) -> None:
        self.db = db
        self.clock = clock or SystemClock()

    # -- goals -----------------------------------------------------------------------------------------
    def save_goal(self, goal: Goal, *, session_id: str | None = None, project_id: str | None = None) -> Goal:
        if not goal.created_at:
            goal.created_at = self.clock.now()
        self.db.execute(
            "INSERT INTO goals(id, text, kind, mode, priority, complexity, session_id, project_id, data, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, mode=excluded.mode, "
            "priority=excluded.priority, complexity=excluded.complexity, data=excluded.data",
            (goal.id, goal.text, goal.kind, goal.mode.value, goal.priority.value, int(goal.complexity), session_id,
             project_id, dumps(goal.to_dict()), goal.created_at))
        return goal

    def get_goal(self, goal_id: str) -> Goal | None:
        row = self.db.query_one("SELECT data FROM goals WHERE id=?", (goal_id,))
        return Goal.from_dict(loads(row["data"], {})) if row else None

    def recent_goals(self, limit: int = 20) -> list[Goal]:
        rows = self.db.query("SELECT data FROM goals ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,))
        return [Goal.from_dict(loads(r["data"], {})) for r in rows]

    # -- plans -----------------------------------------------------------------------------------------
    def save(self, plan: Plan) -> Plan:
        now = self.clock.now()
        if not plan.created_at:
            plan.created_at = now
        plan.updated_at = now
        self.db.execute(
            "INSERT INTO plans(id, goal_id, title, status, status_reason, priority, mode, source, created_by, origin, "
            "session_id, project_id, quality, data, version, revision, created_at, updated_at, started_at, "
            "finished_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "title=excluded.title, status=excluded.status, status_reason=excluded.status_reason, "
            "priority=excluded.priority, mode=excluded.mode, source=excluded.source, quality=excluded.quality, "
            "data=excluded.data, version=plans.version + 1, revision=excluded.revision, "
            "updated_at=excluded.updated_at, started_at=excluded.started_at, finished_at=excluded.finished_at",
            (plan.id, plan.goal.id, plan.title, plan.status.value, plan.status_reason, plan.priority.value,
             plan.mode.value, plan.source, plan.created_by, plan.origin, plan.session_id, plan.project_id,
             plan.quality.value if plan.quality else None, dumps(plan.to_dict()), plan.version, plan.created_at,
             plan.updated_at, plan.started_at, plan.finished_at))
        return plan

    def get(self, plan_id: str) -> Plan | None:
        row = self.db.query_one("SELECT data FROM plans WHERE id=?", (plan_id,))
        return Plan.from_dict(loads(row["data"], {})) if row else None

    def list(self, statuses: Iterable[PlanStatus] | None = None, *, limit: int = 50,
             session_id: str | None = None, since: float | None = None) -> list[Plan]:
        clauses, params = [], []
        if statuses is not None:
            statuses = list(statuses)
            clauses.append(f"status IN ({','.join('?' * len(statuses))})")
            params += [s.value for s in statuses]
        if session_id:
            clauses.append("session_id=?")
            params.append(session_id)
        if since is not None:
            clauses.append("updated_at >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.query(f"SELECT data FROM plans {where} ORDER BY updated_at DESC, rowid DESC LIMIT ?",
                             (*params, limit))
        return [Plan.from_dict(loads(r["data"], {})) for r in rows]

    def open_plans(self) -> list[Plan]:
        return self.list(PLAN_OPEN, limit=200)

    def for_goal(self, goal_id: str) -> list[Plan]:
        rows = self.db.query("SELECT data FROM plans WHERE goal_id=? ORDER BY created_at", (goal_id,))
        return [Plan.from_dict(loads(r["data"], {})) for r in rows]

    def find(self, text: str, statuses: Iterable[PlanStatus] | None = None) -> list[Plan]:
        needle = text.lower().strip()
        if not needle:
            return []
        words = [w for w in needle.split() if w not in ("the", "my", "a", "an", "plan")]
        out = []
        for plan in self.list(statuses, limit=200):
            hay = f"{plan.title} {plan.goal.text} {plan.goal.objective} {plan.goal.target or ''}".lower()
            if plan.id == text or (words and all(w in hay for w in words)):
                out.append(plan)
        return out

    # -- revisions ---------------------------------------------------------------------------------------
    def record_revision(self, plan: Plan, trigger: str, summary: str) -> None:
        """Keep the graph as it was before a replan (plan history: what changed and why)."""
        self.db.execute(
            "INSERT OR REPLACE INTO plan_revisions(plan_id, version, ts, trigger, summary, nodes) VALUES(?,?,?,?,?,?)",
            (plan.id, plan.version, self.clock.now(), trigger, summary, dumps([n.to_dict() for n in plan.nodes])))

    def revisions(self, plan_id: str) -> list[dict]:
        rows = self.db.query("SELECT version, ts, trigger, summary, nodes FROM plan_revisions WHERE plan_id=? "
                             "ORDER BY version", (plan_id,))
        return [{"version": r["version"], "ts": r["ts"], "trigger": r["trigger"], "summary": r["summary"],
                 "nodes": loads(r["nodes"], [])} for r in rows]
