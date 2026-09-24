"""Agent coordination (Phase 3 §8-10, §37, §45-46).

JARVIS stays the orchestrator. Work is handed to specialist agents only when that helps: there is enough
independent work to split, a model is available, and resources allow it. Otherwise the plan does the work
deterministically. Each agent node:

* runs as an ordinary task through ``delegate_to_agent``, so it has its own identity (``agent:<name>``), its
  tool allowlist and permission ceiling, its budgets, and an audit trail — and it cannot delegate further or
  grant itself anything;
* receives packed context (the goal, what is known so far), not the whole conversation;
* is isolated: if it fails, times out or exhausts its budget, only its node is affected, and the node falls
  back to a deterministic recipe for the same work;
* returns evidence, not truth: its findings count only once they carry a source, and every quoted finding is
  checked against that source by the verification step. Conflicting findings are reported as conflicting.
"""

from __future__ import annotations

import json
import os
from typing import Any

from jarvis.agents.base import AgentRegistry
from jarvis.events.types import Event, EventType
from jarvis.intelligence.context import ContextPacker
from jarvis.intelligence.plans import NodeKind, Plan, PlanNode
from jarvis.intelligence.playbooks import step


class AgentCoordinator:
    def __init__(self, agents: AgentRegistry, *, router: Any = None, resources: Any = None, runner: Any = None,
                 bus: Any = None, packer: ContextPacker | None = None, max_concurrent: int = 2,
                 files_per_agent: int = 4) -> None:
        self.agents = agents
        self.router = router
        self.resources = resources
        self.runner = runner
        self.bus = bus
        self.packer = packer or ContextPacker(max_tokens=1500)
        self.max_concurrent = max_concurrent
        self.files_per_agent = files_per_agent

    # -- when to use agents ------------------------------------------------------------------------------------
    def should_decompose(self, work_items: int, threshold: int = 8) -> tuple[bool, str]:
        if not (self.router and self.router.available()):
            return False, "no language model is available"
        if self.resources is not None:
            constrained, why = self.resources.pressure()
            if constrained:
                return False, f"resources are short ({why})"
        if work_items <= threshold:
            return False, f"{work_items} item(s) are quick to handle directly"
        return True, f"{work_items} items split across up to {self.max_concurrent} agents"

    # -- assigning work ---------------------------------------------------------------------------------------------
    def research_nodes(self, plan: Plan, parent: PlanNode, files: list[str], *, topic: str) -> list[PlanNode]:
        spec = self.agents.get("research")
        if spec is None:
            return []
        groups = [files[i:i + self.files_per_agent] for i in range(0, len(files), self.files_per_agent)]
        groups = groups[: max(1, self.max_concurrent * 2)]
        nodes = []
        for index, group in enumerate(groups, 1):
            context = self.packer.pack([("Goal", plan.goal.summary()),
                                        ("Files to read", "\n".join(group)),
                                        ("Report", "every finding with the exact quote, the file path and the line")])
            objective = f"Read these files and report what they say about {topic}: " + ", ".join(
                os.path.basename(f) for f in group)
            fallback = [step("file_read", {"path": f, "max_bytes": 200_000}, f"read {f}", f"doc{i}")
                        for i, f in enumerate(group)]
            nodes.append(PlanNode(
                f"{parent.id}_agent{index}", f"Research agent {index}: {len(group)} file(s)", NodeKind.AGENT,
                agent="research", depends_on=[parent.id], optional=True,
                steps=[step("delegate_to_agent", {"agent": "research", "objective": objective, "context": context},
                            f"research agent reads {len(group)} file(s)", "agent")],
                alternatives=[fallback],
                meta={"collect": "agent_claims", "fallback_collect": "documents", "files": group},
                estimate={"seconds": spec.max_seconds / 2, "model_calls": spec.max_model_calls}))
        return nodes

    def assign(self, plan: Plan, agent: str, objective: str, *, depends_on: list[str], node_id: str,
               evidence: Any = None) -> PlanNode | None:
        """A single specialist node with packed context (e.g. the analyst interpreting gathered evidence)."""
        spec = self.agents.get(agent)
        if spec is None:
            return None
        context = self.packer.plan_context(plan, extra=json.dumps(evidence, default=str)[:3000] if evidence else "")
        return PlanNode(node_id, f"{agent.capitalize()} agent: {objective[:60]}", NodeKind.AGENT, agent=agent,
                        depends_on=depends_on, optional=spec.on_failure != "fail",
                        steps=[step("delegate_to_agent", {"agent": agent, "objective": objective, "context": context},
                                    f"{agent} agent", "agent")],
                        estimate={"seconds": spec.max_seconds / 2, "model_calls": spec.max_model_calls})

    # -- collecting and merging -------------------------------------------------------------------------------------
    def collect(self, plan: Plan, node: PlanNode, facts: dict[str, Any]) -> None:
        """Turn an agent's result into evidence: sourced findings become claims to verify; the rest is kept as
        unverified notes; refused tool calls and warnings are recorded."""
        result = facts.get("agent") if isinstance(facts.get("agent"), dict) else \
            next((d for d in facts.get("steps") or [] if isinstance(d, dict)), {})
        findings = result.get("findings") or []
        claims, notes = [], []
        for f in findings:
            if isinstance(f, dict) and f.get("quote") and f.get("source"):
                claims.append({"text": str(f.get("text") or f["quote"])[:300], "quote": str(f["quote"])[:300],
                               "source": str(f["source"]), "line": f.get("line"), "by": f"agent:{node.agent}"})
            else:
                notes.append(str(f.get("text") if isinstance(f, dict) else f)[:200])
        facts["claims"] = claims
        facts["notes"] = notes
        facts["agent_status"] = {"status": result.get("status"), "confidence": result.get("confidence"),
                                 "warnings": result.get("warnings", []), "refused": result.get("refused_calls", []),
                                 "model_calls": result.get("model_calls"), "tool_calls": result.get("tool_calls")}
        if self.bus is not None:
            self.bus.emit(Event(EventType.AGENT_FINISHED, "planner",
                                {"plan_id": plan.id, "node": node.id, "agent": node.agent,
                                 "status": result.get("status"), "claims": len(claims), "unsourced": len(notes),
                                 "refused": result.get("refused_calls", [])}, entity_id=f"plan:{plan.id}"))

    def status(self) -> list[dict[str, Any]]:
        if self.runner is None:
            return []
        return list(self.runner.active.values())

    def contracts(self) -> list[dict[str, Any]]:
        return [spec.contract() for spec in self.agents.list()]
