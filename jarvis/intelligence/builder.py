"""Plan building and validation (Phase 3 §3, §9, §13-14, §21, §29, §44).

Sources, in order of preference:

1. a deterministic playbook for the goal's kind (no model needed);
2. a compound request mapped clause by clause onto the intent grammar and templates;
3. a graph proposed by a language model — treated as an untrusted proposal: unknown tools and invalid arguments
   are rejected, node kinds are derived from what each tool can do (never from the model's label), and
   consequential steps always get an approval gate;
4. a single objective handed to the existing task planner.

Validation checks structure (no cycles, known dependencies, size limits), tools and arguments, the user's
constraints, and that verification nodes only observe. Approval gates are inserted in front of every action
above the plan's permission baseline; the gate is an ordinary consequential tool call, so approving it goes
through the permission system like anything else.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from jarvis.config import IntelligenceConfig
from jarvis.core.types import new_id
from jarvis.intelligence.analysis import jarvis_pids
from jarvis.intelligence.context import ContextPacker
from jarvis.intelligence.goals import Complexity, ExecutionMode, Goal, GoalParser, fingerprint
from jarvis.intelligence.guard import LoopGuard
from jarvis.intelligence.memory import PlanMemory
from jarvis.intelligence.plans import NodeKind, Plan, PlanNode
from jarvis.intelligence.playbooks import PLAYBOOKS, Blueprint, PlanningContext, compound, step
from jarvis.log import get_logger
from jarvis.models.base import ChatMessage, ModelError, Purpose
from jarvis.models.router import TaskProfile
from jarvis.permissions.model import PermissionLevel
from jarvis.planner.planner import _extract_json
from jarvis.tools.schema import SchemaError, validate_args

log = get_logger("planning")

_INTERNAL_TOOLS = {"plan_gate", "plan_analyze", "plan_verify", "delegate_to_agent", "start_background_task",
                   "remember"}

_DAG_INSTRUCTIONS = """You are the planning component of JARVIS, a local AI operating environment.
Break the objective into a small graph of tool calls. Rules:
- Use only the tools listed; arguments must match each tool's parameters.
- Observe before acting. Steps that don't depend on each other must not list each other in depends_on
  (they run in parallel).
- No destructive actions unless the objective requires them. Do not add approval steps; JARVIS adds them.
- At most {max_nodes} nodes. Short lowercase ids.
Respond with JSON only:
{{"nodes": [{{"id": "...", "title": "...", "tool": "<tool>", "args": {{...}}, "depends_on": ["id"],
 "optional": false}}], "notes": "..."}}"""


@dataclass
class BuildResult:
    plan: Plan | None
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.problems


class PlanBuilder:
    def __init__(self, registry: Any, router: Any, permissions: Any, memory: PlanMemory, *,
                 config: IntelligenceConfig | None = None, projects: Any = None, resources: Any = None,
                 parser: GoalParser | None = None, packer: ContextPacker | None = None) -> None:
        self.registry = registry
        self.router = router
        self.permissions = permissions
        self.memory = memory
        self.config = config or IntelligenceConfig()
        self.projects = projects
        self.resources = resources
        self.parser = parser or GoalParser()
        self.guard = LoopGuard(self.config)
        self.packer = packer or ContextPacker()

    # -- context ----------------------------------------------------------------------------------------------
    def context(self, goal: Goal, cwd: str, *, interactive: bool = True, project: Any = None) -> PlanningContext:
        pressure = ""
        if self.resources is not None:
            constrained, why = self.resources.pressure()
            pressure = why if constrained else ""
        tools = {t.spec.name for t in self.registry.list()}
        return PlanningContext(
            cwd=cwd, project_name=getattr(project, "name", None), project_root=getattr(project, "root", None),
            model_available=bool(self.router and self.router.available()), tools=tools,
            history=self.memory.history(goal), lessons=self.memory.lessons(goal), pressure=pressure,
            interactive=interactive, jarvis_pids=sorted(jarvis_pids()), sample_s=self.config.process_sample_s)

    def baseline(self, interactive: bool) -> int:
        cfg = self.permissions.config
        return int(cfg.interactive_level if interactive else cfg.automation_level)

    # -- building --------------------------------------------------------------------------------------------
    async def build(self, goal: Goal, *, cwd: str, interactive: bool = True, created_by: str = "user:owner",
                    owner: str = "owner", origin: str = "", session_id: str | None = None, project: Any = None,
                    autonomy: str = "normal") -> BuildResult:
        ctx = self.context(goal, cwd, interactive=interactive, project=project)
        if goal.kind in PLAYBOOKS:
            blueprint = PLAYBOOKS[goal.kind](goal, ctx)
            source = f"playbook:{goal.kind}"
        elif goal.kind == "compound":
            blueprint = compound(goal, ctx, self.parser)
            source = "compound"
        else:
            blueprint, source = await self._model_blueprint(goal, ctx)
        if blueprint.problems:
            return BuildResult(None, blueprint.problems, blueprint.notes)
        title = blueprint.title or goal.objective
        if goal.mode == ExecutionMode.DRY_RUN and not title.lower().startswith("dry run"):
            title = "Dry run: " + title[:1].lower() + title[1:]      # a preview must never read like the real thing
        plan = Plan(goal, title, blueprint.nodes, source=source, priority=goal.priority,
                    mode=goal.mode, created_by=created_by, owner=owner, origin=origin, session_id=session_id,
                    project_id=getattr(project, "id", None), cwd=ctx.project_root or cwd, interactive=interactive,
                    autonomy=autonomy, assumptions=blueprint.assumptions, milestones=blueprint.milestones)
        plan.facts["_notes"] = blueprint.notes + ([f"Resources are constrained ({ctx.pressure}), so heavy work may "
                                                   "wait."] if ctx.pressure else [])
        self._drop_unavailable(plan, ctx)
        problems = self.validate(plan)
        if not problems:
            self.insert_gates(plan, [n for n in plan.nodes if n.kind in (NodeKind.ACTION, NodeKind.TASK)
                                     and n.steps])
            problems = self.validate(plan)
        return BuildResult(plan if not problems else None, problems, blueprint.notes)

    def _drop_unavailable(self, plan: Plan, ctx: PlanningContext) -> None:
        """Optional steps whose tool isn't available here are dropped instead of failing at run time."""
        for node in plan.nodes:
            kept = []
            for s in node.steps:
                if s["tool"] in ctx.tools or not s.get("allow_failure"):
                    kept.append(s)
            node.steps = kept

    # -- model-proposed graphs ----------------------------------------------------------------------------------
    async def _model_blueprint(self, goal: Goal, ctx: PlanningContext) -> tuple[Blueprint, str]:
        report = PlanNode("report", "Report", NodeKind.REPORT, run_on_failure=True)
        if not ctx.model_available:
            return Blueprint([], goal.objective, problems=[
                "planning this needs a language model, and none is available"]), "none"
        nodes, notes = await self.propose(goal, ctx)
        if nodes:
            report.depends_on = [n.id for n in nodes if not any(n.id in m.depends_on for m in nodes)]
            return Blueprint(nodes + [report], goal.objective, notes=notes), "model"
        # the model couldn't produce a usable graph: one task, planned step by step by the existing planner
        work = PlanNode("work", goal.objective[:80], NodeKind.TASK, objective=goal.objective,
                        estimate={"model_calls": 3})
        report.depends_on = [work.id]
        return Blueprint([work, report], goal.objective, notes=notes), "objective"

    async def propose(self, goal: Goal, ctx: PlanningContext, *, prompt_extra: str = "",
                      max_nodes: int = 8, prefix: str = "") -> tuple[list[PlanNode], list[str]]:
        """Ask the model for a graph of tool calls and keep only what validates."""
        catalog = self._catalog(goal)
        packed = self.packer.pack([("Objective", goal.summary()), ("Working folder", ctx.cwd),
                                   ("Project", ctx.project_name or ""), ("Earlier attempts", "\n".join(ctx.lessons)),
                                   ("Situation", prompt_extra), ("Available tools", catalog)])
        messages = [ChatMessage("system", _DAG_INSTRUCTIONS.format(max_nodes=max_nodes)),
                    ChatMessage("user", packed)]
        profile = TaskProfile(purpose=Purpose.PLANNING, complexity="medium", interactive=False)
        try:
            routed = await self.router.chat(profile, messages, format="json")
        except ModelError as exc:
            return [], [f"the planning model was unavailable ({exc})"]
        try:
            data = json.loads(_extract_json(routed.response.content))
        except ValueError:
            return [], ["the planning model returned an unreadable plan"]
        raw = data.get("nodes") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return [], ["the planning model returned no steps"]
        nodes, rejected = self.validate_proposal(raw, max_nodes=max_nodes, prefix=prefix)
        notes = [f"ignored {len(rejected)} proposed step(s): {'; '.join(rejected[:3])}"] if rejected else []
        return nodes, notes

    def validate_proposal(self, raw: list[Any], *, max_nodes: int = 8, prefix: str = "") -> tuple[list[PlanNode], list[str]]:
        nodes: list[PlanNode] = []
        rejected: list[str] = []
        ids: dict[str, str] = {}
        for index, item in enumerate(raw[:max_nodes]):
            if not isinstance(item, dict):
                rejected.append(f"not an object: {str(item)[:40]}")
                continue
            tool_name = str(item.get("tool") or "")
            tool = self.registry.get(tool_name)
            if tool is None or tool_name in _INTERNAL_TOOLS:
                rejected.append(f"unknown or internal tool {tool_name!r}")
                continue
            try:
                args = validate_args(tool.spec.parameters, item.get("args") or {})
            except SchemaError as exc:
                rejected.append(f"{tool_name}: {exc}")
                continue
            if tool.assess(args).blocked:
                rejected.append(f"{tool_name}: refused by the safety policy")
                continue
            raw_id = re.sub(r"[^a-z0-9_]", "", str(item.get("id") or f"n{index}").lower())[:20] or f"n{index}"
            node_id = f"{prefix}{raw_id}"
            while node_id in ids.values():
                node_id += "_"
            ids[str(item.get("id") or f"n{index}")] = node_id
            level = tool.assess(args).level
            # the kind follows from what the tool does, never from the model's own label
            kind = NodeKind.GATHER if level <= PermissionLevel.OBSERVE else NodeKind.ACTION
            nodes.append(PlanNode(node_id, str(item.get("title") or tool.preview(args))[:100], kind,
                                  steps=[step(tool_name, args, str(item.get("title") or tool.preview(args))[:200],
                                              "out")], optional=bool(item.get("optional", False)),
                                  meta={"proposed_by": "model"}))
            nodes[-1].meta["raw_depends_on"] = [str(d) for d in item.get("depends_on") or []]
        for node in nodes:
            deps = [ids.get(d) for d in node.meta.pop("raw_depends_on", [])]
            node.depends_on = [d for d in deps if d and d != node.id]
        probe = Plan(Goal("probe", "probe"), "probe", nodes)
        if probe.find_cycle():
            rejected.append("the proposed steps depend on each other in a circle")
            return [], rejected
        return nodes, rejected

    def _catalog(self, goal: Goal) -> str:
        lines = []
        for tool in self.registry.list():
            spec = tool.spec
            if spec.name in _INTERNAL_TOOLS or spec.category == "planning":
                continue
            if goal.mode != ExecutionMode.EXECUTE and spec.level > PermissionLevel.OBSERVE:
                continue
            params = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in spec.parameters.get("properties", {}).items())
            lines.append(f"- {spec.name}({params}) [level {int(spec.level)}]: {spec.description}")
        return "\n".join(lines)

    # -- validation ------------------------------------------------------------------------------------------------
    def node_level(self, node: PlanNode) -> int:
        level = 0
        for s in node.steps:
            tool = self.registry.get(s["tool"])
            if tool is None:
                continue
            try:
                assessed = tool.assess(s["args"]) if not _has_placeholders(s["args"]) else None
            except Exception:
                assessed = None
            level = max(level, int(assessed.level if assessed else tool.spec.level))
        return level

    def validate(self, plan: Plan) -> list[str]:
        problems = list(plan.problems())
        limit = self.guard.nodes(plan)
        if limit:
            problems.append(limit.reason)
        goal = plan.goal
        for node in plan.nodes:
            for s in node.steps:
                tool = self.registry.get(s["tool"])
                if tool is None:
                    problems.append(f"'{node.title}' needs the tool {s['tool']}, which isn't available")
                    continue
                if not _has_placeholders(s["args"]):
                    try:
                        s["args"] = validate_args(tool.spec.parameters, s["args"])
                    except SchemaError as exc:
                        problems.append(f"'{node.title}': {exc}")
                        continue
                    assessed = tool.assess(s["args"])
                    if assessed.blocked:
                        problems.append(f"'{node.title}' is refused by the safety policy ({assessed.reason})")
                    why = goal.forbids(s["tool"], s["args"], node.meta, level=int(assessed.level),
                                       requires_network=assessed.requires_network or tool.spec.requires_network)
                    if why and node.kind != NodeKind.GATE:
                        problems.append(f"'{node.title}' conflicts with your instructions: {why}")
            if node.actor == "verifier" and self.node_level(node) > self.baseline(False):
                problems.append(f"'{node.title}' is a verification step but would change something")
            if goal.mode.changes_nothing and node.kind in (NodeKind.ACTION, NodeKind.TASK) \
                    and self.node_level(node) > PermissionLevel.OBSERVE and node.steps:
                # dry runs, simulations and advice never execute changes: the step is shown, not run
                node.meta["dry_run_only"] = True
        return problems

    def insert_gates(self, plan: Plan, actions: list[PlanNode]) -> PlanNode | None:
        """Put one approval gate in front of the actions that need more authority than the plan's baseline."""
        baseline = self.baseline(plan.interactive)
        gated = [n for n in actions if self.node_level(n) > baseline and not n.meta.get("dry_run_only")
                 and not self._preauthorized(plan, n)
                 and not any(g.kind == NodeKind.GATE and n.id in g.gate_for for g in plan.nodes)]
        if not gated:
            return None
        gate_id = "gate" if plan.node("gate") is None else new_id("gate").replace("-", "_")[:12]
        deps: list[str] = []
        for n in gated:
            deps += [d for d in n.depends_on if d not in deps]
        entries = []
        for n in gated:
            for s in n.steps:
                tool = self.registry.get(s["tool"])
                assessed = tool.assess(s["args"]) if tool and not _has_placeholders(s["args"]) else None
                title = n.title[:1].lower() + n.title[1:]
                entries.append({"node": n.id, "tool": s["tool"], "preview": title if len(n.steps) == 1 else
                                (tool.preview(s["args"]) if tool else s["tool"]),
                                "level": int(assessed.level) if assessed else int(tool.spec.level if tool else 5),
                                "reversible": bool(assessed.reversible if assessed else (tool and tool.spec.reversible)),
                                "rollback": n.rollback, "fingerprint": fingerprint(s["tool"], s["args"])})
        summary = gate_summary(entries)
        gate = PlanNode(gate_id, "Ask before making changes", NodeKind.GATE, depends_on=deps,
                        steps=[step("plan_gate", {"plan_id": plan.id, "summary": summary, "actions": entries},
                                    f"approve: {summary}", "gate")], gate_for=[n.id for n in gated],
                        meta={"expected": entries})
        index = min(plan.nodes.index(n) for n in gated)
        plan.nodes.insert(index, gate)
        for n in gated:
            n.depends_on = [gate_id]
        return gate

    def _preauthorized(self, plan: Plan, node: PlanNode) -> bool:
        """Authority the user already delegated (a standing grant) needs no gate. This only decides whether to
        *ask*; the permission check still runs for every step when it executes."""
        from jarvis.permissions.model import AccessRequest, Actor
        actor = Actor("task", "planned", plan.owner, interactive=plan.interactive,
                      delegated_by=plan.created_by if plan.created_by.startswith(("automation:", "agent:")) else None)
        for s in node.steps:
            tool = self.registry.get(s["tool"])
            if tool is None or _has_placeholders(s["args"]):
                return False
            level = tool.assess(s["args"]).level
            decision = self.permissions.check(AccessRequest(actor, s["tool"], level, [], plan.project_id, None,
                                                            tool.preview(s["args"])), consume=False)
            if not (decision.allowed and decision.basis == "grant"):
                return False
        return bool(node.steps)

    def needs_preview(self, plan: Plan, autonomy: str) -> bool:
        policy = self.config.preview
        if autonomy == "low" or policy == "always":
            return True
        if re.search(r"\b(show me the plan|plan (it|this) (out|first)|before you (start|do anything)|"
                     r"what would you do first)\b", plan.goal.text, re.I):
            return True
        if policy == "consequential":
            return any(n.kind == NodeKind.GATE for n in plan.nodes) and plan.goal.complexity >= Complexity.MODERATE
        return False


def gate_summary(entries: list[dict[str, Any]]) -> str:
    parts = []
    for e in entries:
        text = e["preview"]
        if not e.get("reversible", True):
            text += " (can't be undone)"
        parts.append(text)
    if len(parts) == 1:
        return parts[0]
    return "; ".join(parts[:-1]) + "; and " + parts[-1]


def _has_placeholders(value: Any) -> bool:
    if isinstance(value, dict):
        if set(value) & {"$fact", "$from_step", "$collect"} and len(value) == 1:
            return True
        return any(_has_placeholders(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_placeholders(v) for v in value)
    return False


def has_placeholders(value: Any) -> bool:
    return _has_placeholders(value)
