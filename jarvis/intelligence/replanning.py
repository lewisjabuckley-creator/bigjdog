"""Dynamic replanning (Phase 3 §5, §16, §24).

When something changes — a fix that verification says didn't work, an assumption that no longer holds, a
failed step with no alternative, a user correction — the plan is not restarted. The replanner pauses the plan
(REPLANNING), keeps everything already done and learned, works out which part of the remaining plan is affected,
rebuilds only that part, and resumes from the right point. The previous graph is stored as a revision first,
and every revision is bounded by the loop-protection limits and re-checked for cycles.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from jarvis.core.types import OperationalReason
from jarvis.events.types import EventType
from jarvis.intelligence.goals import Constraint, GoalParser
from jarvis.intelligence.plans import NODE_ACTIVE, NodeKind, NodeStatus, Plan, PlanNode, PlanStatus
from jarvis.intelligence.playbooks import PLAYBOOKS
from jarvis.log import get_logger

log = get_logger("replanning")
N = NodeStatus


class Replanner:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    async def replan(self, plan: Plan, trigger: str, reason: str, *, node: PlanNode | None = None,
                     assumption: Any = None, correction: dict[str, Any] | None = None) -> bool:
        eng = self.engine
        limit = eng.guard.replan(plan)
        if limit:
            eng._limit_event(plan, limit)
            return False
        snapshot = [n.to_dict() for n in plan.nodes]
        goal_snapshot = plan.goal.to_dict()
        previous = plan.status
        eng.store.record_revision(plan, trigger, reason)
        eng._set(plan, PlanStatus.REPLANNING, f"re-planning: {reason[:160]}", force=True)
        try:
            if trigger == "verification":
                changed = self._next_candidate(plan, node)
            elif trigger == "assumption":
                changed = self._assumption(plan, node, assumption)
            elif trigger == "correction":
                changed = await self._correction(plan, correction or {})
            elif trigger == "resources":
                changed = self._resources(plan, reason)
            else:
                changed = await self._repair(plan, node, reason)
        except Exception as exc:          # a replanning bug must leave the plan as it was
            log.error("replan_failed", plan_id=plan.id, error=repr(exc))
            changed = []
        problems = plan.problems()
        if not changed or problems:
            from jarvis.intelligence.goals import Goal
            # restore in place: callers may hold references to these nodes
            live = {n.id: n for n in plan.nodes}
            restored = []
            for d in snapshot:
                fresh = PlanNode.from_dict(d)
                if d["id"] in live:
                    live[d["id"]].__dict__.update(fresh.__dict__)
                    restored.append(live[d["id"]])
                else:
                    restored.append(fresh)
            plan.nodes = restored
            plan.goal = Goal.from_dict(goal_snapshot)
            eng._set(plan, previous, plan.status_reason, force=True)
            if problems:
                log.warning("replan_rejected", plan_id=plan.id, problems=problems)
            return False
        plan.version += 1
        plan.replans.append({"version": plan.version, "trigger": trigger, "reason": reason[:300],
                             "changed": changed, "ts": eng.clock.now()})
        # a plan that hadn't started (a preview) or was paused stays that way; otherwise it carries on
        after = previous if previous in (PlanStatus.READY, PlanStatus.PAUSED) else PlanStatus.RUNNING
        eng._set(plan, after, f"revised the plan: {changed[0]}", force=True)
        eng._emit(EventType.PLAN_REPLANNED, plan, {"trigger": trigger, "reason": reason[:200], "changed": changed,
                                                   "version": plan.version})
        eng._audit(plan, "plan_replanned", "; ".join(changed)[:300],
                   OperationalReason(reason[:200], "adapt the remaining plan instead of restarting it; keep what is done",
                                     f"revised the plan ({trigger})", "; ".join(changed)[:200]))
        return True

    # -- a fix that didn't work: try the next most likely one ----------------------------------------------------
    def _next_candidate(self, plan: Plan, verify: PlanNode | None) -> list[str]:
        if verify is None:
            return []
        decides = [n for n in plan.ancestors(verify.id) if n.kind == NodeKind.DECIDE]
        if not decides:
            return []
        decide = max(decides, key=lambda n: plan.nodes.index(n))
        for decision in plan.decisions:
            if decision.get("node") == decide.id:
                for action in decision.get("actions", []):
                    action["outcome"] = action.get("outcome") or "no_effect"
        analysis = plan.fact(f"{decide.meta.get('from')}.analysis") or {}
        from jarvis.intelligence.engine import action_identity
        tried = {a.get("identity") for d in plan.decisions for a in d.get("actions", [])}
        min_conf = float(decide.meta.get("min_confidence", 0.55))
        remaining = [c for c in analysis.get("candidates", [])
                     if (not c.get("blocked") or c.get("blocked") == "already tried in this plan")
                     and action_identity(c) not in tried and c.get("confidence", 0) >= min_conf]
        if not remaining:
            return []
        round_no = int(decide.meta.get("round", 1)) + 1
        prefix = decide.id[: -len("decide")] if decide.id.endswith("decide") else re.sub(r"decide\d*$", "", decide.id)
        new_decide_id = f"{prefix}decide{round_no}"
        new_verify_id = f"{prefix}verify{round_no}"
        new_decide = PlanNode(new_decide_id, "Try the next most likely fix", NodeKind.DECIDE, depends_on=[verify.id],
                              meta={**decide.meta, "round": round_no, "verify": new_verify_id})
        new_verify = PlanNode.from_dict(verify.to_dict())
        new_verify.id = new_verify_id
        new_verify.reset()
        new_verify.quality = None
        new_verify.task_ids = []
        new_verify.attempts = 0
        new_verify.meta = {}
        new_verify.depends_on = [new_decide_id]
        new_verify.condition = {"fact": f"{new_decide_id}.actions", "op": "nonempty"}
        for s in new_verify.steps:
            args = s.get("args") or {}
            if "before" in args:
                args["before"] = {"$fact": f"{verify.id}.after"}      # compare with the state after the first fix
            if "targets" in args:
                args["targets"] = {"$fact": f"{new_decide_id}.targets"}
            if "paths" in args:
                args["paths"] = {"$fact": f"{new_decide_id}.paths"}
        verify.meta["superseded_by"] = new_verify_id
        for n in plan.nodes:
            if n.kind == NodeKind.REPORT and verify.id in n.depends_on and not n.finished:
                n.depends_on = [new_verify_id if d == verify.id else d for d in n.depends_on]
        position = plan.nodes.index(verify) + 1
        plan.nodes[position:position] = [new_decide, new_verify]
        detail = (plan.facts.get(verify.id) or {}).get("verdict", {}).get("detail", "it didn't help")
        return [f"the first fix didn't help ({detail}); next: {remaining[0]['title']}"]

    # -- an assumption no longer holds -------------------------------------------------------------------------
    def _assumption(self, plan: Plan, node: PlanNode | None, assumption: Any) -> list[str]:
        if node is None or assumption is None:
            return []
        kind = (assumption.check or {}).get("type")
        if kind == "process_running" and node.kind == NodeKind.ACTION:
            node.status = N.SKIPPED
            node.note = f"no longer needed: {assumption.evidence}"
            return [f"skipped '{node.title}': {assumption.evidence}"]
        return []                 # a missing folder or drive needs the user: the engine blocks and asks

    # -- the user corrected something ---------------------------------------------------------------------------
    async def _correction(self, plan: Plan, correction: dict[str, Any]) -> list[str]:
        eng = self.engine
        kind = correction.get("kind")
        changed: list[str] = []
        if kind in ("protect", "constraint"):
            constraint = Constraint.from_dict(correction["constraint"]) if kind == "constraint" else \
                Constraint("protect_process", correction["value"], correction.get("text", ""))
            if not any(c.kind == constraint.kind and c.value.lower() == constraint.value.lower()
                       for c in plan.goal.constraints):
                plan.goal.constraints.append(constraint)
            changed.append(f"added: {constraint.describe()}")
            for n in plan.nodes:
                if n.finished or n.kind != NodeKind.ACTION:
                    continue
                if any(plan.goal.forbids(s["tool"], s.get("args") or {}, n.meta, level=4) for s in n.steps):
                    if n.task_id and n.status in NODE_ACTIVE:
                        eng.tasks.cancel_task(n.task_id, by="user", reason=f"you said: {constraint.describe()}")
                    n.status = N.SKIPPED
                    n.note = f"you said: {constraint.describe()}"
                    changed.append(f"dropped '{n.title}'")
            for gate in [g for g in plan.nodes if g.kind == NodeKind.GATE and not g.finished]:
                live = [nid for nid in gate.gate_for if (t := plan.node(nid)) and not t.finished]
                if not live:
                    if gate.task_id and gate.status in NODE_ACTIVE:
                        eng.tasks.cancel_task(gate.task_id, by="user", reason="nothing left to approve")
                    gate.status = N.SKIPPED
                    gate.note = "nothing left to approve"
            return changed
        if kind == "skip":
            words = [w for w in re.findall(r"[a-z0-9]+", str(correction.get("value", "")).lower())
                     if w not in ("the", "a", "step", "part")]
            for n in plan.nodes:
                if n.finished or not words or n.kind == NodeKind.REPORT:
                    continue
                title = n.title.lower()
                if all(w[:5] in title for w in words):
                    if n.task_id and n.status in NODE_ACTIVE:
                        eng.tasks.cancel_task(n.task_id, by="user", reason="you said to skip it")
                    n.status = N.SKIPPED
                    n.note = "you said to skip it"
                    changed.append(f"skipped '{n.title}'")
            return changed
        if kind == "retarget":
            new_target = str(correction.get("value", "")).strip()
            if not new_target or plan.goal.kind not in ("backup", "disk_cleanup", "research"):
                return []            # (a performance investigation has no target to change)
            old_target = plan.goal.target
            if plan.goal.kind == "backup" and old_target and "->" in old_target:
                src, dst = [x.strip() for x in old_target.split("->", 1)]
                which = correction.get("which") or "destination"
                plan.goal.target = f"{new_target} -> {dst}" if which == "source" else f"{src} -> {new_target}"
            else:
                plan.goal.target = new_target
            for n in plan.nodes:
                if n.finished:
                    continue
                if n.task_id and n.status in NODE_ACTIVE:
                    eng.tasks.cancel_task(n.task_id, by="user", reason=f"you changed the target to {new_target}")
                n.status = N.CANCELLED
                n.note = f"superseded: you changed the target to {new_target}"
            ctx = eng.builder.context(plan.goal, plan.cwd or ".", interactive=plan.interactive)
            blueprint = PLAYBOOKS[plan.goal.kind](plan.goal, ctx, prefix=f"r{plan.version + 1}_")
            if blueprint.problems:
                return []
            plan.nodes.extend(blueprint.nodes)
            plan.assumptions.extend(blueprint.assumptions)
            eng.builder.insert_gates(plan, [n for n in blueprint.nodes if n.kind in (NodeKind.ACTION, NodeKind.TASK)
                                            and n.steps])
            plan.title = blueprint.title
            return [f"target changed from {old_target or 'the default'} to {plan.goal.target}; completed work is kept"]
        return []

    # -- resources are short: drop optional heavy work ------------------------------------------------------------
    def _resources(self, plan: Plan, reason: str) -> list[str]:
        changed = []
        for n in plan.nodes:
            if not n.finished and n.optional and (n.estimate.get("model_calls") or n.kind == NodeKind.AGENT):
                if n.task_id and n.status in NODE_ACTIVE:
                    self.engine.tasks.cancel_task(n.task_id, by="system:planner", reason=f"resources: {reason}")
                n.status = N.SKIPPED
                n.note = f"skipped to save resources ({reason})"
                changed.append(f"skipped the optional '{n.title}'")
        return changed

    # -- a step failed and has no alternative: ask the model for a replacement ---------------------------------------
    async def _repair(self, plan: Plan, node: PlanNode | None, reason: str) -> list[str]:
        eng = self.engine
        if node is None or not (eng.router and eng.router.available()):
            return []
        if eng.guard.model_call(plan):
            return []
        ctx = eng.builder.context(plan.goal, plan.cwd or ".", interactive=plan.interactive)
        situation = (f"The step '{node.title}' ({', '.join(s['tool'] for s in node.steps) or 'planned'}) failed: "
                     f"{reason[:300]}. Propose steps that achieve what it was for another way. Completed so far: "
                     + "; ".join(n.title for n in plan.nodes if n.status == N.DONE)[:600])
        nodes, _notes = await eng.builder.propose(plan.goal, ctx, prompt_extra=situation, max_nodes=4,
                                                  prefix=f"{node.id}_r{plan.version + 1}_")
        if not nodes:
            return []
        entry_ids = {n.id for n in nodes}
        for n in nodes:
            if not any(d in entry_ids for d in n.depends_on):
                n.depends_on = list(dict.fromkeys(node.depends_on + n.depends_on))
        exits = [n.id for n in nodes if not any(n.id in m.depends_on for m in nodes)]
        for child in plan.children(node.id):
            child.depends_on = [d for d in child.depends_on if d != node.id] + exits
        node.status = N.SKIPPED
        node.note = f"replaced after it failed: {reason[:120]}"
        position = plan.nodes.index(node) + 1
        plan.nodes[position:position] = nodes
        eng.builder.insert_gates(plan, [n for n in nodes if n.kind == NodeKind.ACTION])
        return [f"replaced '{node.title}' with: " + "; ".join(n.title for n in nodes)]


def parse_correction(text: str) -> dict[str, Any] | None:
    """What a correction asks to change: "leave Chrome alone", "no, back it up to E:\\ instead",
    "skip the cleanup", "don't delete anything"."""
    t = text.strip()
    parser = GoalParser()
    constraints = parser.constraints(t)
    protect = next((c for c in constraints if c.kind == "protect_process"), None)
    if protect:
        return {"kind": "protect", "value": protect.value, "text": t}
    other = next((c for c in constraints if c.kind in ("no_delete", "no_restart", "protect_path", "read_only",
                                                        "no_network", "local_models")), None)
    if other:
        return {"kind": "constraint", "constraint": other.to_dict(), "text": t}
    m = re.search(r"\b(?:skip|drop|leave out|don'?t bother with)\s+(?:the\s+)?(?P<v>.+?)(?:\s+step)?[.!]?$", t, re.I)
    if m:
        return {"kind": "skip", "value": m.group("v"), "text": t}
    m = re.search(r"^(?:no,?\s+|actually,?\s+|not\s+that,?\s+)?(?:i\s+meant|i\s+mean|use|make\s+it|it'?s|put\s+it\s+(?:on|in)|"
                  r"(?:back\s*(?:it\s+)?up\s+)?to)\s+(?P<v>[~/.\\]?[\w:./\\ -]+?)(?:\s+instead)?[.!]?$", t, re.I)
    if m and re.match(r"^(no|actually|not that|i meant|i mean|use|to|back|put)", t, re.I):
        return {"kind": "retarget", "value": m.group("v").strip(), "text": t}
    return None


def copy_node(node: PlanNode) -> PlanNode:
    return copy.deepcopy(node)
