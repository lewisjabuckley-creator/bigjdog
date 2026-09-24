"""Loop protection (Phase 3 §34).

Autonomy needs hard limits. Every retry, alternative, replan, model call and added node is counted against
configurable limits; a plan that reaches one stops and says so, instead of looping. Identical failures are
recognised by signature, so "the same thing failed the same way again" ends retrying early. Circular
dependencies are rejected before a plan runs and after every replan.
"""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.config import IntelligenceConfig
from jarvis.intelligence.failures import Failure
from jarvis.intelligence.plans import Plan, PlanNode


@dataclass
class Limit:
    name: str
    reason: str

    def sentence(self) -> str:
        return f"I stopped because {self.reason}, to avoid going round in circles"


class LoopGuard:
    def __init__(self, config: IntelligenceConfig | None = None) -> None:
        self.config = config or IntelligenceConfig()

    def node_attempt(self, plan: Plan, node: PlanNode) -> Limit | None:
        if node.attempts >= self.config.max_node_attempts:
            return Limit("node_attempts", f"'{node.title}' has already been tried {node.attempts} times")
        return None

    def replan(self, plan: Plan) -> Limit | None:
        if len(plan.replans) >= self.config.max_replans:
            return Limit("replans", f"the plan has already been revised {len(plan.replans)} times")
        return None

    def failure(self, plan: Plan, failure: Failure) -> Limit | None:
        """Count a failure; the same failure repeating is a loop even under the attempt limit."""
        key = f"failure:{failure.signature or failure.category.value}"
        plan.counters[key] = plan.counters.get(key, 0) + 1
        if plan.counters[key] > self.config.max_identical_failures:
            return Limit("identical_failures", f"the same failure happened {plan.counters[key]} times "
                                               f"({failure.detail[:120]})")
        return None

    def nodes(self, plan: Plan, extra: int = 0) -> Limit | None:
        if len(plan.nodes) + extra > self.config.max_nodes:
            return Limit("nodes", f"the plan would grow past {self.config.max_nodes} steps")
        return None

    def model_call(self, plan: Plan) -> Limit | None:
        if plan.counters.get("model_calls", 0) >= self.config.max_model_calls:
            return Limit("model_calls", f"the plan has used its {self.config.max_model_calls} model calls")
        plan.counters["model_calls"] = plan.counters.get("model_calls", 0) + 1
        return None

    def cycle(self, plan: Plan) -> Limit | None:
        cycle = plan.find_cycle()
        if cycle:
            return Limit("circular_dependency", "its steps depend on each other in a circle (" + " → ".join(cycle) + ")")
        return None

    def age(self, plan: Plan, now: float) -> Limit | None:
        started = plan.started_at or plan.created_at
        if started and now - started > self.config.max_plan_hours * 3600:
            return Limit("duration", f"the plan has been open for more than {self.config.max_plan_hours:g} hours")
        return None

    def resource_wait(self, node: PlanNode, now: float) -> Limit | None:
        since = node.meta.get("waiting_since")
        if since and now - since > self.config.resource_wait_max_s:
            return Limit("resource_wait", f"'{node.title}' has waited {int((now - since) // 60)} minutes for "
                                          "resources")
        return None

    def loop_iteration(self, node: PlanNode) -> Limit | None:
        maximum = int((node.loop or {}).get("max", 3))
        if node.iterations >= maximum:
            return Limit("loop_iterations", f"'{node.title}' has repeated {node.iterations} times (the limit is "
                                            f"{maximum})")
        return None
