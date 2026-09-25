"""Memory-aware planning and decision memory (Phase 3 §14-15).

Before planning, JARVIS looks at what happened the last times it pursued the same kind of goal: which actions
were taken and whether independent verification said they helped. An action that already failed to help is
demoted, and the plan says so ("we already tried that on Tuesday"). Every decision a plan takes is recorded
in the decision log with the evidence it was based on, and its outcome is filled in once verification is done.
"""

from __future__ import annotations

from typing import Any

from jarvis.clock import Clock, SystemClock, format_datetime
from jarvis.core.types import Provenance, ProvenanceKind
from jarvis.intelligence.goals import Goal
from jarvis.intelligence.plans import PLAN_TERMINAL, Plan, PlanStatus, Quality
from jarvis.intelligence.store import PlanStore
from jarvis.memory.decisions import DecisionLog
from jarvis.memory.store import MemoryKind, MemoryStore


class PlanMemory:
    def __init__(self, store: PlanStore, decisions: DecisionLog | None = None, memory: MemoryStore | None = None,
                 clock: Clock | None = None) -> None:
        self.store = store
        self.decisions = decisions
        self.memory = memory
        self.clock = clock or SystemClock()

    # -- reading ------------------------------------------------------------------------------------------
    def _similar(self, goal: Goal, days: float = 30) -> list[Plan]:
        since = self.clock.now() - days * 86400
        return [p for p in self.store.list(PLAN_TERMINAL, limit=100, since=since)
                if p.goal.kind == goal.kind and p.goal.kind != "generic"]

    def history(self, goal: Goal) -> list[dict[str, Any]]:
        """Earlier outcomes of concrete actions for this kind of goal, newest first."""
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for plan in self._similar(goal):
            for decision in plan.decisions:
                for action in decision.get("actions", []):
                    key = action.get("key")
                    outcome = action.get("outcome")
                    if key and outcome and key not in seen:
                        seen.add(key)
                        out.append({"key": key, "outcome": outcome, "when": _when(self.clock, plan.finished_at
                                                                                   or plan.updated_at),
                                    "plan": plan.id, "title": action.get("label") or action.get("title", key)})
        return out

    def lessons(self, goal: Goal) -> list[str]:
        """Short notes the plan preview and report can show ("we already tried that")."""
        notes = []
        for entry in self.history(goal)[:3]:
            if entry["outcome"] in ("no_effect", "failed"):
                notes.append(f"Last time ({entry['when']}) I tried {entry['title']}; it didn't help, so I won't "
                             "lead with that again.")
            elif entry["outcome"] in ("helped", "verified"):
                notes.append(f"Last time ({entry['when']}), {entry['title']} helped.")
        recent_failures = [p for p in self._similar(goal, days=2) if p.status == PlanStatus.FAILED]
        if recent_failures and not notes:
            notes.append(f"A similar plan failed {_when(self.clock, recent_failures[0].updated_at)}: "
                         f"{recent_failures[0].status_reason or recent_failures[0].result[:120]}.")
        return notes

    # -- writing ---------------------------------------------------------------------------------------------
    def record_decision(self, plan: Plan, title: str, decision: str, *, evidence: list[str],
                        alternatives: list[str], reason: str) -> str | None:
        if self.decisions is None:
            return None
        rec = self.decisions.record(title, decision, context="Evidence: " + "; ".join(evidence)[:1500],
                                    alternatives=alternatives[:6], reason=reason,
                                    user_involvement="pending your approval" if plan.interactive else "",
                                    project_id=plan.project_id, tags=["plan", plan.goal.kind, plan.id])
        return rec.id

    async def record_outcome(self, plan: Plan) -> None:
        """Close the loop: decision outcomes, an episodic note, and what worked (procedural memory)."""
        for decision in plan.decisions:
            outcome = decision.get("outcome")
            if self.decisions is not None and decision.get("decision_log_id") and outcome:
                self.decisions.set_outcome(decision["decision_log_id"], outcome)
        if self.memory is None or plan.goal.kind == "generic":
            return
        first = (plan.result or plan.status_reason or "").strip().splitlines()
        summary = f"Plan '{plan.title}' {plan.status.value}" + (f" ({plan.quality.label})" if plan.quality else "")
        if first:
            summary += f": {first[0][:200]}"
        await self.memory.remember(summary, kind=MemoryKind.EPISODIC, subject=plan.goal.kind, project_id=plan.project_id,
                                   tags=["plan", plan.goal.kind], importance=0.4, force=True,
                                   provenance=Provenance(ProvenanceKind.DATABASE, "plan history", plan.id))
        for decision in plan.decisions:
            for action in decision.get("actions", []):
                if action.get("outcome") == "helped" and plan.quality == Quality.VERIFIED:
                    await self.memory.remember(
                        f"For '{plan.goal.kind}' problems, {action.get('title')} helped (verified).",
                        kind=MemoryKind.PROCEDURAL, subject=plan.goal.kind, project_id=plan.project_id,
                        tags=["plan", "lesson"], importance=0.5, force=True,
                        provenance=Provenance(ProvenanceKind.TOOL_OUTPUT, "plan verification", plan.id))


def _when(clock: Clock, ts: float | None) -> str:
    if not ts:
        return "earlier"
    age = clock.now() - ts
    if age < 3600:
        return "less than an hour ago"
    if age < 86400:
        return f"{int(age // 3600)} hour{'s' if age >= 7200 else ''} ago"
    return f"on {format_datetime(ts)[:10]}"
