"""Plans as structured data (Phase 3 §3-6, §16, §19, §30-32, §43).

A :class:`Plan` is a directed acyclic graph of :class:`PlanNode` s above the task system. Nodes carry their
dependencies (independent nodes run in parallel), optional conditions ("only if the tests pass"), loops with a
bound ("re-run until they pass, at most 3 times"), approval gates, verification, rollback notes, resource
estimates and priorities. Nodes that do work become ordinary durable tasks; the plan records which task
carries which node, what each produced (``facts``), the assumptions it relies on, the decisions taken and why,
failures, approvals and every revision. All of it is persisted after every change, so a restart resumes the
plan from where it was.

Lifecycle::

    CREATED → VALIDATING → READY → RUNNING ⇄ WAITING / BLOCKED / PAUSED
                                      │  └→ REPLANNING → RUNNING
                                      └→ VERIFYING → COMPLETED | FAILED        (CANCELLED from any open state)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Iterable

from jarvis.core.types import new_id
from jarvis.intelligence.goals import ExecutionMode, Goal, PlanPriority


class NodeKind(StrEnum):
    GATHER = "gather"      # observe: read-only tool steps
    ANALYZE = "analyze"    # deterministic analysis of gathered evidence
    DECIDE = "decide"      # choose actions from the analysis; recorded as a decision with its evidence
    GATE = "gate"          # an approval point (the existing "proceed?" flow)
    ACTION = "action"      # changes something, through the tool registry
    AGENT = "agent"        # a bounded specialist agent
    TASK = "task"          # an objective the existing planner turns into steps
    VERIFY = "verify"      # independent re-measurement and comparison, never by the executor itself
    REPORT = "report"      # the result, composed from what the plan observed


class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"        # its task exists and is queued or executing
    WAITING = "waiting"        # for approval, a model, or resources
    BLOCKED = "blocked"        # needs the user (not authorized, outcome unknown...)
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


NODE_FINISHED = frozenset({NodeStatus.DONE, NodeStatus.FAILED, NodeStatus.SKIPPED, NodeStatus.CANCELLED})
NODE_ACTIVE = frozenset({NodeStatus.RUNNING, NodeStatus.WAITING, NodeStatus.BLOCKED})


class PlanStatus(StrEnum):
    CREATED = "created"
    VALIDATING = "validating"
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    BLOCKED = "blocked"
    PAUSED = "paused"
    REPLANNING = "replanning"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


P = PlanStatus
PLAN_TERMINAL = frozenset({P.COMPLETED, P.FAILED, P.CANCELLED})
PLAN_OPEN = frozenset(set(P) - PLAN_TERMINAL)
PLAN_TRANSITIONS: dict[PlanStatus, frozenset[PlanStatus]] = {
    P.CREATED: frozenset({P.VALIDATING, P.FAILED, P.CANCELLED}),
    P.VALIDATING: frozenset({P.READY, P.BLOCKED, P.FAILED, P.CANCELLED}),
    P.READY: frozenset({P.RUNNING, P.PAUSED, P.VALIDATING, P.REPLANNING, P.CANCELLED, P.COMPLETED, P.FAILED}),
    P.RUNNING: frozenset({P.WAITING, P.BLOCKED, P.PAUSED, P.REPLANNING, P.VERIFYING, P.COMPLETED, P.FAILED,
                          P.CANCELLED}),
    P.WAITING: frozenset({P.RUNNING, P.BLOCKED, P.PAUSED, P.REPLANNING, P.VERIFYING, P.FAILED, P.CANCELLED}),
    P.BLOCKED: frozenset({P.RUNNING, P.WAITING, P.PAUSED, P.REPLANNING, P.VERIFYING, P.FAILED, P.CANCELLED}),
    P.PAUSED: frozenset({P.RUNNING, P.READY, P.WAITING, P.BLOCKED, P.REPLANNING, P.CANCELLED}),
    P.REPLANNING: frozenset({P.RUNNING, P.READY, P.WAITING, P.BLOCKED, P.PAUSED, P.FAILED, P.CANCELLED}),
    P.VERIFYING: frozenset({P.COMPLETED, P.FAILED, P.RUNNING, P.REPLANNING, P.CANCELLED}),
    P.COMPLETED: frozenset(),
    P.FAILED: frozenset({P.REPLANNING}),       # an explicit "try again"
    P.CANCELLED: frozenset(),
}


class InvalidPlanTransition(ValueError):
    pass


class Quality(StrEnum):
    """How well a result is established (Phase 3 §19). Never flatten unverified into verified."""

    VERIFIED = "verified"                      # independently checked and it holds
    PARTIALLY_VERIFIED = "partially_verified"  # some of it was checked
    UNVERIFIED = "unverified"                  # done, but nothing independent confirmed it
    FAILED = "failed"                          # checked and it does not hold
    CONFLICTING = "conflicting"                # the evidence disagrees with itself

    @property
    def label(self) -> str:
        return self.value.replace("_", " ")


@dataclass
class Assumption:
    """Something the plan relies on. Checked before the nodes it affects run and when relevant events arrive;
    if it no longer holds, the plan is replanned instead of carrying on blindly (Phase 3 §16)."""

    statement: str
    check: dict[str, Any] = field(default_factory=dict)   # {"type": "path_exists", "path": ...} etc.
    affects: list[str] = field(default_factory=list)       # node ids
    id: str = field(default_factory=lambda: new_id("asm"))
    status: str = "unchecked"                              # unchecked | holding | invalid
    evidence: str = ""
    checked_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Assumption":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class PlanNode:
    id: str
    title: str
    kind: NodeKind
    steps: list[dict[str, Any]] = field(default_factory=list)   # [{"tool", "args", "description", ...}]
    objective: str = ""                   # TASK nodes: the existing planner turns this into steps
    agent: str | None = None              # AGENT nodes: which specialist
    depends_on: list[str] = field(default_factory=list)
    condition: dict[str, Any] | None = None
    loop: dict[str, Any] | None = None    # {"until": condition, "max": n, "restart_from": node id}
    optional: bool = False                # its failure does not fail the plan
    run_on_failure: bool = False          # runs once its dependencies settle, even if they failed (reports)
    actor: str = "plan"                   # plan | verifier: who runs it (the verifier never executes changes)
    verify: dict[str, Any] | None = None  # the node task's success condition
    rollback: str = ""                    # how the change can be undone, in words
    gate_for: list[str] = field(default_factory=list)          # GATE: the action nodes it approves
    approved: dict[str, list[str]] = field(default_factory=dict)   # GATE: node id -> fingerprints the user approved
    meta: dict[str, Any] = field(default_factory=dict)
    resources: list[str] = field(default_factory=list)
    estimate: dict[str, Any] = field(default_factory=dict)     # {"seconds", "model_calls", "memory_mb", "cpu"}
    alternatives: list[list[dict[str, Any]]] = field(default_factory=list)   # other recipes if a tool fails
    priority: PlanPriority | None = None  # overrides the plan's priority for this node
    # -- runtime state ---------------------------------------------------------------------------
    status: NodeStatus = NodeStatus.PENDING
    task_id: str | None = None
    task_ids: list[str] = field(default_factory=list)          # every task that carried this node
    attempts: int = 0
    iterations: int = 0
    quality: Quality | None = None
    summary: str = ""
    error: str = ""
    failure: dict[str, Any] | None = None                      # {"category", "detail", "strategy"}
    started_at: float | None = None
    finished_at: float | None = None
    note: str = ""

    @property
    def finished(self) -> bool:
        return self.status in NODE_FINISHED

    @property
    def succeeded(self) -> bool:
        return self.status == NodeStatus.DONE and self.quality != Quality.FAILED

    @property
    def runs_task(self) -> bool:
        return self.kind not in (NodeKind.DECIDE, NodeKind.REPORT)

    def reset(self, note: str = "") -> None:
        self.status = NodeStatus.PENDING
        self.task_id = None
        self.quality = None
        self.summary = ""
        self.error = ""
        self.failure = None
        self.started_at = None
        self.finished_at = None
        self.note = note

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["status"] = self.status.value
        d["quality"] = self.quality.value if self.quality else None
        d["priority"] = self.priority.value if self.priority else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlanNode":
        known = set(cls.__dataclass_fields__)
        data = {k: v for k, v in d.items() if k in known}
        data["kind"] = NodeKind(data["kind"])
        data["status"] = NodeStatus(data.get("status", "pending"))
        data["quality"] = Quality(data["quality"]) if data.get("quality") else None
        data["priority"] = PlanPriority(data["priority"]) if data.get("priority") else None
        return cls(**data)

    def brief(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "kind": self.kind.value, "status": self.status.value,
                "depends_on": self.depends_on, "task_id": self.task_id, "quality": self.quality.value
                if self.quality else None, "summary": self.summary, "error": self.error, "attempts": self.attempts,
                "optional": self.optional, "note": self.note}


@dataclass
class Plan:
    goal: Goal
    title: str
    nodes: list[PlanNode] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("plan"))
    status: PlanStatus = PlanStatus.CREATED
    status_reason: str = ""
    source: str = ""                         # playbook:<name> | compound | model | objective
    priority: PlanPriority = PlanPriority.NORMAL
    mode: ExecutionMode = ExecutionMode.EXECUTE
    created_by: str = "user:owner"           # user:<id> | automation:<id> | system:reactions
    owner: str = "owner"
    origin: str = ""                         # conversation:<session> | api | event:<type>
    session_id: str | None = None
    project_id: str | None = None
    cwd: str | None = None
    interactive: bool = True                 # the user asked for it now and can be asked to approve
    autonomy: str = "normal"
    facts: dict[str, Any] = field(default_factory=dict)
    assumptions: list[Assumption] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    replans: list[dict[str, Any]] = field(default_factory=list)
    corrections: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    milestones: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    quality: Quality | None = None
    result: str = ""
    awaiting_confirmation: bool = False      # a preview was shown and the user hasn't said go yet
    paused_by: str | None = None
    parent_id: str | None = None
    version: int = 0                         # revision number (bumped by every replan)
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None

    # -- graph ---------------------------------------------------------------------------------------------
    @property
    def terminal(self) -> bool:
        return self.status in PLAN_TERMINAL

    def node(self, node_id: str) -> PlanNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def node_for_task(self, task_id: str) -> PlanNode | None:
        return next((n for n in self.nodes if n.task_id == task_id or task_id in n.task_ids), None)

    def children(self, node_id: str) -> list[PlanNode]:
        return [n for n in self.nodes if node_id in n.depends_on]

    def descendants(self, node_id: str) -> list[PlanNode]:
        out: list[PlanNode] = []
        frontier = [node_id]
        seen: set[str] = set()
        while frontier:
            current = frontier.pop()
            for child in self.children(current):
                if child.id not in seen:
                    seen.add(child.id)
                    out.append(child)
                    frontier.append(child.id)
        return out

    def ancestors(self, node_id: str) -> list[PlanNode]:
        node = self.node(node_id)
        out: list[PlanNode] = []
        seen: set[str] = set()
        frontier = list(node.depends_on) if node else []
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            parent = self.node(current)
            if parent is not None:
                out.append(parent)
                frontier.extend(parent.depends_on)
        return out

    def problems(self) -> list[str]:
        """Structural problems: duplicate ids, unknown dependencies, cycles."""
        out: list[str] = []
        ids = [n.id for n in self.nodes]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            out.append(f"duplicate node ids: {', '.join(dupes)}")
        known = set(ids)
        for n in self.nodes:
            for dep in n.depends_on:
                if dep not in known:
                    out.append(f"'{n.id}' depends on unknown node '{dep}'")
                if dep == n.id:
                    out.append(f"'{n.id}' depends on itself")
            if n.loop and n.loop.get("restart_from") and n.loop["restart_from"] not in known:
                out.append(f"'{n.id}' loops back to unknown node '{n.loop['restart_from']}'")
        cycle = self.find_cycle()
        if cycle:
            out.append("circular dependency: " + " → ".join(cycle))
        return out

    def find_cycle(self) -> list[str] | None:
        graph = {n.id: [d for d in n.depends_on if self.node(d) is not None] for n in self.nodes}
        state: dict[str, int] = {}
        stack: list[str] = []

        def visit(nid: str) -> list[str] | None:
            state[nid] = 1
            stack.append(nid)
            for dep in graph.get(nid, []):
                if state.get(dep) == 1:
                    return stack[stack.index(dep):] + [dep]
                if state.get(dep) is None:
                    found = visit(dep)
                    if found:
                        return found
            stack.pop()
            state[nid] = 2
            return None

        for nid in graph:
            if state.get(nid) is None:
                found = visit(nid)
                if found:
                    return found
        return None

    def topological(self) -> list[PlanNode]:
        order: list[PlanNode] = []
        placed: set[str] = set()
        remaining = list(self.nodes)
        while remaining:
            progress = False
            for n in list(remaining):
                if all(d in placed or self.node(d) is None for d in n.depends_on):
                    order.append(n)
                    placed.add(n.id)
                    remaining.remove(n)
                    progress = True
            if not progress:          # a cycle: append the rest in declaration order
                order.extend(remaining)
                break
        return order

    def waves(self) -> list[list[PlanNode]]:
        """Nodes grouped by dependency depth: every node in a wave can run in parallel."""
        depth: dict[str, int] = {}
        for n in self.topological():
            depth[n.id] = 1 + max((depth.get(d, 0) for d in n.depends_on), default=-1)
        waves: dict[int, list[PlanNode]] = {}
        for n in self.nodes:
            waves.setdefault(depth.get(n.id, 0), []).append(n)
        return [waves[k] for k in sorted(waves)]

    def deps_settled(self, node: PlanNode) -> bool:
        return all((self.node(d) is None) or self.node(d).finished for d in node.depends_on)  # type: ignore[union-attr]

    def blocking_dependency(self, node: PlanNode) -> PlanNode | None:
        """A dependency whose failure means this node must not run (it is not optional, and this node neither
        reports on failures nor has a condition about that dependency's failure)."""
        referenced = set(_condition_nodes(node.condition))
        for dep_id in node.depends_on:
            dep = self.node(dep_id)
            if dep is None:
                continue
            if dep.status in (NodeStatus.FAILED, NodeStatus.CANCELLED) and not dep.optional \
                    and not node.run_on_failure and dep_id not in referenced:
                return dep
        return None

    def ready_nodes(self) -> list[PlanNode]:
        return [n for n in self.nodes if n.status == NodeStatus.PENDING and self.deps_settled(n)]

    def active_nodes(self) -> list[PlanNode]:
        return [n for n in self.nodes if n.status in NODE_ACTIVE]

    def progress(self) -> float:
        if not self.nodes:
            return 0.0
        return round(sum(1 for n in self.nodes if n.finished) / len(self.nodes), 3)

    # -- facts and conditions ----------------------------------------------------------------------------
    def fact(self, path: str) -> Any:
        value: Any = self.facts
        for part in path.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list):
                try:
                    value = value[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
            if value is None:
                return None
        return value

    def evaluate(self, condition: dict[str, Any] | None) -> bool | None:
        """True / False, or None while it can't be decided yet (a node it depends on hasn't finished)."""
        if not condition:
            return True
        if "all" in condition:
            results = [self.evaluate(c) for c in condition["all"]]
            if any(r is False for r in results):
                return False
            return None if any(r is None for r in results) else True
        if "any" in condition:
            results = [self.evaluate(c) for c in condition["any"]]
            if any(r is True for r in results):
                return True
            return None if any(r is None for r in results) else False
        if "not" in condition:
            inner = self.evaluate(condition["not"])
            return None if inner is None else not inner
        if "node" in condition:
            node = self.node(condition["node"])
            if node is None:
                return False
            if not node.finished:
                return None
            ok = node.succeeded and (self.facts.get(node.id) or {}).get("ok", True) is not False
            return ok if condition.get("outcome", "success") == "success" else not ok
        if "fact" in condition:
            return _compare(self.fact(condition["fact"]), condition.get("op", "true"), condition.get("value"))
        return None

    # -- serialisation ------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "goal": self.goal.to_dict(), "title": self.title,
            "nodes": [n.to_dict() for n in self.nodes], "status": self.status.value,
            "status_reason": self.status_reason, "source": self.source, "priority": self.priority.value,
            "mode": self.mode.value, "created_by": self.created_by, "owner": self.owner, "origin": self.origin,
            "session_id": self.session_id, "project_id": self.project_id, "cwd": self.cwd,
            "interactive": self.interactive, "autonomy": self.autonomy, "facts": self.facts,
            "assumptions": [a.to_dict() for a in self.assumptions], "decisions": self.decisions,
            "failures": self.failures[-50:], "approvals": self.approvals, "replans": self.replans,
            "corrections": self.corrections, "history": self.history[-200:], "milestones": self.milestones,
            "counters": self.counters, "quality": self.quality.value if self.quality else None,
            "result": self.result, "awaiting_confirmation": self.awaiting_confirmation,
            "paused_by": self.paused_by, "parent_id": self.parent_id, "version": self.version,
            "created_at": self.created_at, "updated_at": self.updated_at, "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Plan":
        return cls(
            goal=Goal.from_dict(d["goal"]), title=d["title"], nodes=[PlanNode.from_dict(n) for n in d.get("nodes", [])],
            id=d["id"], status=PlanStatus(d.get("status", "created")), status_reason=d.get("status_reason", ""),
            source=d.get("source", ""), priority=PlanPriority(d.get("priority", "normal")),
            mode=ExecutionMode(d.get("mode", "execute")), created_by=d.get("created_by", "user:owner"),
            owner=d.get("owner", "owner"), origin=d.get("origin", ""), session_id=d.get("session_id"),
            project_id=d.get("project_id"), cwd=d.get("cwd"), interactive=bool(d.get("interactive", True)),
            autonomy=d.get("autonomy", "normal"), facts=dict(d.get("facts", {})),
            assumptions=[Assumption.from_dict(a) for a in d.get("assumptions", [])],
            decisions=list(d.get("decisions", [])), failures=list(d.get("failures", [])),
            approvals=list(d.get("approvals", [])), replans=list(d.get("replans", [])),
            corrections=list(d.get("corrections", [])), history=list(d.get("history", [])),
            milestones=list(d.get("milestones", [])), counters=dict(d.get("counters", {})),
            quality=Quality(d["quality"]) if d.get("quality") else None, result=d.get("result", ""),
            awaiting_confirmation=bool(d.get("awaiting_confirmation", False)), paused_by=d.get("paused_by"),
            parent_id=d.get("parent_id"), version=int(d.get("version", 0)), created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0), started_at=d.get("started_at"), finished_at=d.get("finished_at"))

    def to_api(self) -> dict[str, Any]:
        """The plan as the local API and interfaces see it (no raw facts: they can be large)."""
        return {
            "id": self.id, "title": self.title, "status": self.status.value, "status_reason": self.status_reason,
            "goal": {"id": self.goal.id, "text": self.goal.text, "objective": self.goal.objective,
                     "kind": self.goal.kind, "mode": self.goal.mode.value,
                     "constraints": [c.describe() for c in self.goal.constraints], "deadline": self.goal.deadline,
                     "complexity": self.goal.complexity.label},
            "source": self.source, "priority": self.priority.value, "mode": self.mode.value,
            "progress": self.progress(), "nodes": [n.brief() for n in self.nodes],
            "quality": self.quality.value if self.quality else None, "result": self.result,
            "assumptions": [{"statement": a.statement, "status": a.status, "evidence": a.evidence}
                            for a in self.assumptions],
            "decisions": [{k: v for k, v in d.items() if k != "evidence_data"} for d in self.decisions],
            "replans": self.replans, "approvals": self.approvals,
            "milestones": [{**m, "done": bool(m.get("nodes")) and all(
                (n := self.node(i)) is None or n.finished for i in m.get("nodes", []))} for m in self.milestones],
            "awaiting_confirmation": self.awaiting_confirmation, "created_by": self.created_by,
            "origin": self.origin, "version": self.version, "created_at": self.created_at,
            "updated_at": self.updated_at, "started_at": self.started_at, "finished_at": self.finished_at,
        }


def check_plan_transition(old: PlanStatus, new: PlanStatus) -> None:
    if new not in PLAN_TRANSITIONS[old]:
        raise InvalidPlanTransition(f"cannot move plan from {old.value} to {new.value}")


def _condition_nodes(condition: dict[str, Any] | None) -> Iterable[str]:
    if not condition:
        return []
    out: list[str] = []
    if "node" in condition:
        out.append(condition["node"])
    for key in ("all", "any"):
        for c in condition.get(key, []):
            out.extend(_condition_nodes(c))
    if "not" in condition:
        out.extend(_condition_nodes(condition["not"]))
    return out


def _compare(value: Any, op: str, expected: Any) -> bool:
    try:
        if op == "true":
            return bool(value)
        if op == "false":
            return not value
        if op == "nonempty":
            return bool(value)
        if op == "empty":
            return not value
        if op == "eq":
            return value == expected
        if op == "ne":
            return value != expected
        if op == "contains":
            return expected in (value or [])
        if value is None:
            return False
        if op == "gt":
            return float(value) > float(expected)
        if op == "ge":
            return float(value) >= float(expected)
        if op == "lt":
            return float(value) < float(expected)
        if op == "le":
            return float(value) <= float(expected)
    except (TypeError, ValueError):
        return False
    return False
