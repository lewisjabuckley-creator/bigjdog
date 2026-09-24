"""The conversation side of planning (Phase 3 §2, §21-25, §30-31, §41, §52-54).

Kept out of the orchestrator so it doesn't grow into a monolith: the orchestrator routes, this module handles
goals that need a plan, advice, simulations and predictions, plan previews and confirmation, corrections, "what
are you doing / why did you do that" for plans, stopping and resuming plans, and autonomy.

Replies are short and factual, built from the plan record. When a plan reaches an approval point within a few
seconds, the reply is the approval question itself, with the evidence behind it.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from jarvis.core.intent import Intent, IntentKind, is_pronoun
from jarvis.core.types import Provenance, ProvenanceKind
from jarvis.intelligence import explain
from jarvis.intelligence.goals import ExecutionMode, Goal
from jarvis.intelligence.plans import PLAN_OPEN, NodeKind, NodeStatus, Plan, PlanStatus
from jarvis.intelligence.replanning import parse_correction

if TYPE_CHECKING:
    from jarvis.core.orchestrator import Orchestrator, Response

P = PlanStatus
_FIX_IT = re.compile(r"^(ok(ay)?,?\s+|yes,?\s+|then\s+|right,?\s+)?(please\s+)?(go ahead and\s+)?(fix|sort|solve|deal with)"
                     r"\s+(it|that|this|them)(\s+then)?(\s+please)?[.!]?$", re.I)
_INLINE_WAIT_S = 10.0


class PlanDialogue:
    def __init__(self, orch: "Orchestrator") -> None:
        self.o = orch
        self.svc = orch.svc
        self.focus_plan: str | None = None
        self.focus_is_plan = False              # the conversation is currently about a plan
        self.reported_inline: list[str] = []    # plans whose result this reply shows (acknowledged afterwards)

    @property
    def intel(self) -> Any:
        return self.svc.intelligence

    def _reply(self, text: str, intent: Intent, **kw: Any) -> "Response":
        return self.o._reply(text, intent, **kw)

    # -- which plan the user means ----------------------------------------------------------------------------
    def plan_in_focus(self, statuses: Any = None, *, max_age_s: float = 6 * 3600) -> Plan | None:
        if self.intel is None:
            return None
        if self.focus_plan:
            plan = self.intel.get(self.focus_plan)
            if plan is not None and (statuses is None or plan.status in statuses):
                return plan
        plan = self.intel.latest(statuses=statuses, session_id=self.o.session_id)
        if plan is not None and self.svc.clock.now() - plan.updated_at <= max_age_s:
            return plan
        return None

    def find_plan(self, target: str | None, statuses: Any = None) -> Plan | None:
        if self.intel is None:
            return None
        if target is None or is_pronoun(target):
            return self.plan_in_focus(statuses)
        matches = self.intel.store.find(target, statuses)
        return matches[0] if matches else None

    # -- goals ---------------------------------------------------------------------------------------------------
    async def maybe_goal(self, intent: Intent) -> "Response | None":
        """A request the grammar didn't claim: does it need a plan, advice, a simulation or a prediction?"""
        if self.intel is None or not self.intel.config.enabled:
            return None
        if _FIX_IT.match(intent.text.strip()):
            fixed = await self.fix_it(intent)
            if fixed is not None:
                return fixed
        goal = self.intel.understand(intent.text, dry_run=intent.dry_run,
                                     context={"session": self.o.session_id, "cwd": self.o._work_root()})
        if goal.mode == ExecutionMode.SIMULATE:
            return await self.simulate(intent)
        if goal.mode == ExecutionMode.PREDICT:
            return await self.predict(intent)
        if goal.mode == ExecutionMode.ADVISE:
            return await self.advise(intent, goal)
        if self.intel.should_plan(goal):
            return await self.goal(intent, goal)
        return None

    async def goal(self, intent: Intent, goal: Goal) -> "Response":
        amb = goal.ambiguity
        if amb is not None and amb.must_ask:
            self.o.pending_question = {"entity": "goal_clarify", "text": goal.text, "missing": amb.missing,
                                       "kind": goal.kind, "ids": [], "labels": []}
            self.svc.bus.emit(_event("GOAL_CLARIFICATION_NEEDED", {"goal": goal.text, "missing": amb.missing,
                                                                  "class": amb.klass.value}))
            return self._reply(amb.question, intent, kind="question")
        started = await self.intel.run(goal, cwd=self.o._work_root(), session_id=self.o.session_id,
                                       origin=f"conversation:{self.o.session_id}")
        if started.plan is None or started.problems:
            problems = started.problems or ["I couldn't build a plan for that"]
            return self._reply(_sentence(f"I can't plan that yet: {'; '.join(problems[:2])}"), intent)
        plan = started.plan
        self.focus_plan = plan.id
        prefix = _sentence(f"Assuming {amb.assumption}") + " " if amb is not None and amb.assumption else ""
        if started.preview:
            return self._reply(prefix + explain.preview(plan) + "\nShall I start?", intent, kind="question",
                               data={"plan_id": plan.id, "format": "block"})
        plan = await self.wait(plan.id, _INLINE_WAIT_S)
        return self.plan_reply(plan, intent, prefix=prefix, opening=self._opening(plan))

    def _opening(self, plan: Plan) -> str:
        first = next((n.title for n in plan.nodes if n.kind != NodeKind.REPORT), plan.title)
        text = f"On it: {plan.title[:1].lower() + plan.title[1:]}. First: {first[:1].lower() + first[1:]}."
        if plan.goal.permissions == "changes" or plan.goal.wants_fix:
            text += " I'll ask before changing anything."
        return text

    async def wait(self, plan_id: str, timeout: float) -> Plan:
        """Wait briefly for a plan to finish or reach a point where it needs the user."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        plan = self.intel.get(plan_id)
        while loop.time() < deadline:
            plan = self.intel.get(plan_id)
            if plan is None or plan.terminal or plan.status in (P.BLOCKED, P.PAUSED) or \
                    (plan.status == P.WAITING and self._pending_gate(plan) is not None):
                break
            await asyncio.sleep(0.05)
        return plan

    def _pending_gate(self, plan: Plan) -> Any:
        for node in plan.nodes:
            if node.status == NodeStatus.WAITING and node.task_id:
                pending = [a for a in self.svc.approvals.pending() if a.task_id == node.task_id]
                if pending:
                    return node, pending[0]
        return None

    def plan_reply(self, plan: Plan, intent: Intent, *, prefix: str = "", opening: str = "") -> "Response":
        provs = [Provenance(ProvenanceKind.TOOL_OUTPUT, "plan steps"), Provenance(ProvenanceKind.DATABASE, "plan record")]
        gate = self._pending_gate(plan) if plan.status == P.WAITING else None
        if gate is not None:
            node, approval = gate
            self.o._inline_tasks.append(node.task_id)
            findings = explain.compose_report(plan).splitlines()
            evidence = " ".join(line for line in findings[:2] if not line.startswith("Result:"))
            text = prefix + (evidence + " " if evidence else "") + f"I'd like to {approval.summary}. Proceed?"
            return self._reply(text.strip(), intent, kind="question", task_id=node.task_id, approval_id=approval.id,
                               provenance=provs, data={"plan_id": plan.id})
        if plan.terminal:
            self._reported(plan)
            text = prefix + (plan.result or explain.status_line(plan))
            if plan.status == P.FAILED and plan.status_reason and plan.status_reason not in text:
                text += f"\n({plan.status_reason})"
            return self._reply(text.strip(), intent, kind="answer", provenance=provs,
                               data={"plan_id": plan.id, "format": "block"})
        if plan.status in (P.BLOCKED, P.PAUSED, P.WAITING):
            return self._reply(prefix + _sentence(explain.status_line(plan)), intent, data={"plan_id": plan.id})
        tail = " It keeps going if you close this window; I'll tell you when it's done."
        return self._reply((prefix + opening + tail).strip(), intent, kind="action", data={"plan_id": plan.id})

    def _reported(self, plan: Plan) -> None:
        from jarvis.core.awareness import mark_reported
        mark_reported(self.svc, [plan.id])
        self.reported_inline.append(plan.id)
        self.svc.notifications.acknowledge_task(f"plan:{plan.id}")
        for node in plan.nodes:
            for task_id in node.task_ids:
                self.svc.notifications.acknowledge_task(task_id)

    async def answer_clarification(self, text: str, pq: dict[str, Any]) -> "Response":
        original = pq.get("text", "")
        answer = text.strip().rstrip(".")
        if pq.get("kind") == "backup" and pq.get("missing") == "the destination":
            combined = f"{original.rstrip('.?! ')} to {answer}"
        elif pq.get("kind") == "backup":
            combined = f"back up {answer}"
        else:
            combined = f"{original.rstrip('.?! ')} ({answer})"
        return await self.o._dispatch(combined)

    async def fix_it(self, intent: Intent) -> "Response | None":
        """"Fix it" after an investigation: the same goal, now allowed to change things (asking first)."""
        plan = self.plan_in_focus([P.COMPLETED], max_age_s=3600)
        if plan is None or plan.goal.wants_fix or plan.goal.kind not in ("performance", "disk_cleanup"):
            return None
        goal = self.intel.understand({"performance": "my computer is slow, fix it",
                                      "disk_cleanup": "free up disk space"}[plan.goal.kind])
        goal.constraints = plan.goal.constraints
        goal.target = plan.goal.target
        self.intel.attention.acted(plan.goal.kind)
        return await self.goal(intent, goal)

    # -- advice, simulation, prediction ------------------------------------------------------------------------------
    async def advise(self, intent: Intent, goal: Goal | None = None) -> "Response":
        goal = goal or self.intel.understand(intent.text)
        goal.mode = ExecutionMode.ADVISE
        goal.wants_fix = False
        if goal.kind in ("performance", "disk_cleanup", "research"):
            started = await self.intel.run(goal, cwd=self.o._work_root(), session_id=self.o.session_id,
                                           origin=f"conversation:{self.o.session_id}")
            if started.plan is not None and not started.problems:
                self.focus_plan = started.plan.id
                plan = await self.wait(started.plan.id, 15.0)
                reply = self.plan_reply(plan, intent, opening="Looking into it first (observing only).")
                if plan.terminal and plan.goal.kind in ("performance", "disk_cleanup"):
                    reply.text += "\nI haven't changed anything. Say 'fix it' and I'll do the first recommendation, " \
                                  "asking you first."
                return reply
        return await self.o._chat(Intent(IntentKind.CHAT, intent.text, dry_run=intent.dry_run, source="advice"),
                                  advisory=True)

    async def simulate(self, intent: Intent) -> "Response":
        estimate = await self.intel.simulate(intent.text, cwd=self.o._work_root())
        return self._reply(estimate.render(), intent, data={"estimate": estimate.data, "basis": estimate.basis},
                           provenance=[Provenance(ProvenanceKind.INFERENCE, "simulation", estimate.basis)])

    async def predict(self, intent: Intent) -> "Response":
        estimate = self.intel.predict(intent.text)
        return self._reply(estimate.render(), intent, data={"estimate": estimate.data, "basis": estimate.basis},
                           provenance=[Provenance(ProvenanceKind.DATABASE, "task history", estimate.basis)])

    # -- plans in conversation --------------------------------------------------------------------------------------
    async def show(self, intent: Intent) -> "Response":
        plan = self.find_plan(intent.target, PLAN_OPEN) or self.plan_in_focus()
        if plan is None:
            return self._reply("There's no plan in progress.", intent)
        self.focus_plan = plan.id
        text = explain.preview(plan) + "\n" + _sentence(explain.status_line(plan))
        if plan.replans:
            text += "\n" + explain.changes(plan)
        return self._reply(text, intent, data={"plan_id": plan.id, "format": "block"})

    async def history(self, intent: Intent) -> "Response":
        if re.search(r"\b(change|changed|revis)", intent.text, re.I):
            plan = self.plan_in_focus()
            if plan is not None:
                return self._reply(explain.changes(plan), intent, data={"plan_id": plan.id})
        return self._reply(explain.history(self.intel.recent(8)), intent, data={"format": "block"})

    async def autonomy(self, intent: Intent) -> "Response":
        level = (intent.params.get("level") or "").lower()
        auto = self.intel.autonomy
        if level in ("low", "normal", "high"):
            auto.set(level, by=self.svc.user)
        return self._reply(auto.describe(), intent)

    async def correct(self, intent: Intent) -> "Response | None":
        plan = self.plan_in_focus(PLAN_OPEN)
        if plan is None or parse_correction(intent.text) is None:
            return None
        applied, detail = await self.intel.correct(intent.text, plan)
        if not applied:
            return self._reply(_sentence(detail or "That doesn't change the current plan"), intent,
                               data={"plan_id": plan.id})
        self.focus_plan = plan.id
        return self._reply(_sentence(f"Updated {plan.title[:1].lower() + plan.title[1:]}: {detail}") +
                           " Completed work is kept.", intent, kind="action", data={"plan_id": plan.id})

    # -- hooks into the existing control handlers ---------------------------------------------------------------------
    async def stop(self, intent: Intent, hard: bool) -> "Response | None":
        target = (intent.target or "").lower()
        if self.intel is None:
            return None
        if target in ("everything", "all", "all tasks"):
            for plan in self.intel.open_plans():
                if hard:
                    await self.intel.engine.cancel(plan.id, by=self.svc.user, reason="cancelled by you")
                else:
                    await self.intel.engine.pause(plan.id, by=self.svc.user, reason="stopped by you")
            return None            # the task-level handler reports the totals
        if (intent.target is None or is_pronoun(intent.target)) and not self.focus_is_plan and any(
                t.status.value in ("running", "planning", "verifying", "queued") and not t.outputs.get("plan_id")
                for t in self.svc.tasks.open_tasks()):
            return None            # "stop" while talking about a task: the task handler owns it
        plan = self.find_plan(intent.target, [P.RUNNING, P.WAITING, P.BLOCKED, P.READY, P.REPLANNING] +
                              ([P.PAUSED] if hard else []))
        if plan is None:
            return None
        if hard:
            ok, message = await self.intel.engine.cancel(plan.id, by=self.svc.user, reason="cancelled by you")
            text = f"Cancelled {plan.title[:1].lower() + plan.title[1:]}. Completed steps are kept in the history."
        else:
            ok, message = await self.intel.engine.pause(plan.id, by=self.svc.user, reason="stopped by you")
            text = f"Stopped {plan.title[:1].lower() + plan.title[1:]}. It's checkpointed; say 'continue' to resume."
        return self._reply(text if ok else _sentence(message), intent, kind="action", data={"plan_id": plan.id})

    async def resume(self, intent: Intent) -> "Response | None":
        if self.intel is None:
            return None
        statuses = [P.PAUSED, P.BLOCKED, P.WAITING, P.FAILED, P.READY]
        target = intent.target
        plan = self.find_plan(target, statuses)
        if plan is None:
            return None
        if target is None or is_pronoun(target):
            # "continue" alone: a paused task mentioned more recently than the plan wins
            task = self.o.resolver.task(None, statuses=None).item
            if task is not None and not getattr(task, "outputs", {}).get("plan_id") and \
                    task.updated_at > plan.updated_at and task.status.value in ("paused", "blocked", "waiting"):
                return None
        if plan.status == P.WAITING and self._pending_gate(plan) is not None:
            node, approval = self._pending_gate(plan)
            return self._reply(f"{plan.title} is waiting for your approval to {approval.summary}. Proceed?", intent,
                               kind="question", task_id=node.task_id, approval_id=approval.id)
        ok, message = await self.intel.engine.resume(plan.id, by=self.svc.user)
        self.focus_plan = plan.id
        if not ok:
            return self._reply(_sentence(message), intent)
        fresh = self.intel.get(plan.id) or plan
        done = [n.title for n in fresh.nodes if n.status == NodeStatus.DONE and n.kind != NodeKind.REPORT]
        nxt = next((n.title for n in fresh.nodes if not n.finished), None)
        text = f"Resuming {fresh.title[:1].lower() + fresh.title[1:]}"
        if done:
            text += f". Already done: {', '.join(done[-3:])}"
        if nxt:
            text += f". Next: {nxt[:1].lower() + nxt[1:]}"
        return self._reply(_sentence(text), intent, kind="action", data={"plan_id": plan.id})

    async def approve(self, intent: Intent) -> "Response | None":
        """"Yes" to a plan preview, or "do it" after advice. Approvals of steps go through the normal flow."""
        if self.intel is None:
            return None
        plan = self.plan_in_focus([P.READY])
        if plan is not None and plan.awaiting_confirmation:
            started = await self.intel.confirm(plan.id)
            plan = await self.wait(plan.id, _INLINE_WAIT_S) if started else plan
            return self.plan_reply(plan, intent, opening=self._opening(plan))
        if not self.svc.approvals.pending():
            return await self.fix_it(intent)
        return None

    async def after_approval(self, approval: Any, intent: Intent) -> "Response | None":
        """After approving a plan's gate, wait briefly and report what happened (verified or not)."""
        if self.intel is None or not approval.task_id:
            return None
        plan = self.intel.plan_for_task(approval.task_id)
        if plan is None:
            return None
        self.focus_plan = plan.id
        plan = await self.wait(plan.id, 15.0)
        if plan.terminal or (plan.status == P.WAITING and self._pending_gate(plan) is not None):
            return self.plan_reply(plan, intent)
        return self._reply(f"Proceeding: {approval.summary}. I'll check the result independently and tell you.",
                           intent, kind="action", data={"plan_id": plan.id})

    async def deny(self, intent: Intent) -> "Response | None":
        if self.intel is None:
            return None
        plan = self.plan_in_focus([P.READY])
        if plan is not None and plan.awaiting_confirmation:
            await self.intel.engine.cancel(plan.id, by=self.svc.user, reason="you decided not to start it")
            return self._reply("Understood — I won't start it.", intent, kind="action")
        return None

    async def why(self, intent: Intent) -> "Response | None":
        if self.intel is None:
            return None
        plan = self.find_plan(intent.target) if intent.target and not is_pronoun(intent.target) else \
            self.plan_in_focus(max_age_s=2 * 3600)
        if plan is None:
            return None
        return self._reply(explain.why(plan), intent, data={"plan_id": plan.id},
                           provenance=[Provenance(ProvenanceKind.DATABASE, "plan record and audit log")])

    def status_prefix(self) -> str:
        return explain.activity(self.intel.open_plans()) if self.intel is not None else ""


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def _event(etype: str, payload: dict[str, Any]) -> Any:
    from jarvis.events.types import Event
    return Event(etype, "conversation", payload)
