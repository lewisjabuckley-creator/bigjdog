"""Result quality (Phase 3 §17-19).

The executor never certifies its own work. A node that changed something is VERIFIED only when an independent
check confirmed it: either the tool's own post-condition check (read back from the system, not the tool's
claim) or a separate VERIFY node run under the verifier's identity, which re-measures and compares. Observations
and analysis have no quality label: they are evidence, reported as observed or inferred.
"""

from __future__ import annotations

from typing import Any

from jarvis.intelligence.plans import NodeKind, NodeStatus, Plan, PlanNode, Quality
from jarvis.tasks.models import StepStatus, Task

_RESULT_KINDS = (NodeKind.ACTION, NodeKind.TASK, NodeKind.AGENT, NodeKind.VERIFY)


def node_quality(node: PlanNode, task: Task | None, facts: dict[str, Any]) -> Quality | None:
    if node.kind == NodeKind.VERIFY:
        result = facts.get("verdict") if isinstance(facts, dict) else None
        if isinstance(result, dict) and result.get("quality"):
            try:
                return Quality(result["quality"])
            except ValueError:
                return Quality.UNVERIFIED
        return Quality.UNVERIFIED
    if node.kind not in _RESULT_KINDS or task is None:
        return None
    if node.kind == NodeKind.AGENT:
        # an agent's findings are evidence; artifacts it claims are checked on disk by the delegation tool
        step = task.plan[0] if task.plan else None
        v = (step.verification or {}) if step else {}
        if v.get("performed"):
            return Quality.VERIFIED if v.get("passed") else Quality.FAILED
        return Quality.UNVERIFIED
    done = [s for s in task.plan if s.status == StepStatus.DONE]
    if not done:
        return Quality.UNVERIFIED
    performed = [s for s in done if (s.verification or {}).get("performed")]
    if any((s.verification or {}).get("passed") is False for s in performed):
        return Quality.FAILED
    report = task.outputs.get("verification") or {}
    if report.get("outcome") == "failed":
        return Quality.FAILED
    if performed and len(performed) == len(done):
        return Quality.VERIFIED
    if performed:
        return Quality.PARTIALLY_VERIFIED
    return Quality.UNVERIFIED


def covered_by_verification(plan: Plan, node: PlanNode) -> bool:
    for v in plan.nodes:
        if v.kind == NodeKind.VERIFY and v.status == NodeStatus.DONE and node in plan.ancestors(v.id):
            return True
    return False


def plan_quality(plan: Plan) -> Quality | None:
    """The plan's overall quality, from its verification nodes first, then from its result nodes."""
    verifications = [n.quality for n in plan.nodes if n.kind == NodeKind.VERIFY and n.status == NodeStatus.DONE
                     and n.quality and not n.meta.get("superseded_by")]
    results = [n for n in plan.nodes if n.kind in (NodeKind.ACTION, NodeKind.TASK, NodeKind.AGENT)
               and n.status == NodeStatus.DONE]
    uncovered = [n for n in results if not covered_by_verification(plan, n)]
    qualities = list(verifications) + [n.quality or Quality.UNVERIFIED for n in uncovered]
    if not qualities:
        return None                       # nothing was changed or produced that needs verifying
    if Quality.CONFLICTING in qualities:
        return Quality.CONFLICTING
    if Quality.FAILED in qualities:
        return Quality.FAILED if all(q == Quality.FAILED for q in qualities) or verifications else \
            Quality.PARTIALLY_VERIFIED
    if all(q == Quality.VERIFIED for q in qualities):
        return Quality.VERIFIED
    if any(q in (Quality.VERIFIED, Quality.PARTIALLY_VERIFIED) for q in qualities):
        return Quality.PARTIALLY_VERIFIED
    return Quality.UNVERIFIED


def describe(quality: Quality | None) -> str:
    return {
        Quality.VERIFIED: "verified independently",
        Quality.PARTIALLY_VERIFIED: "partly verified",
        Quality.UNVERIFIED: "not independently verified",
        Quality.FAILED: "verification failed",
        Quality.CONFLICTING: "the evidence conflicts",
        None: "observed directly; nothing needed verifying",
    }[quality]
