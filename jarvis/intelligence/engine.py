"""The plan engine: INTENT → PLAN → EXECUTE → OBSERVE → VERIFY → ADAPT → COMPLETE (Phase 3 §4-7, §18, §30-31).

Plans run on the existing task system. Every node that does work becomes an ordinary durable task, so it gets
the same security pipeline as anything else (request → authorization → permission → execution → verification
→ audit), the same checkpoints, recovery after a restart, resource admission, and approvals.

The engine follows the plan's tasks through their events (and a periodic reconciliation, in case an event was
missed), and after each change:

1. brings node states up to date from their tasks and collects what each produced (facts);
2. classifies failures and applies a recovery strategy (retry, alternative tool, wait, replan, ask), within
   the loop-protection limits;
3. checks the assumptions of the nodes about to run;
4. runs decisions and reports itself (they change nothing) and starts every node whose dependencies are done,
   up to the parallelism limit and subject to resources — independent nodes run in parallel;
5. works out the plan's status, and when every node is finished assesses quality and completes the plan;
6. writes the plan to the database before anything acts on it.

Approval gates are ordinary consequential tool calls. When the user approves one, the exact actions it listed
(matched by fingerprint) receive single-use, task-scoped grants; nothing else does.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable

from jarvis.audit.log import AuditLog
from jarvis.clock import Clock, SystemClock
from jarvis.config import IntelligenceConfig
from jarvis.core.types import OperationalReason, Outcome, Priority, Severity, new_id
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.intelligence import explain
from jarvis.intelligence.builder import PlanBuilder, has_placeholders
from jarvis.intelligence.failures import STRATEGIES, FailureCategory, Strategy, classify
from jarvis.intelligence.goals import ExecutionMode, PlanPriority, fingerprint
from jarvis.intelligence.guard import Limit, LoopGuard
from jarvis.intelligence.memory import PlanMemory
from jarvis.intelligence.plans import (NODE_ACTIVE, Assumption, NodeKind, NodeStatus, Plan, PlanNode, PlanStatus,
                                       Quality, check_plan_transition)
from jarvis.intelligence.playbooks import step as make_step
from jarvis.intelligence.quality import node_quality, plan_quality
from jarvis.intelligence.store import PlanStore
from jarvis.log import get_logger
from jarvis.permissions.hierarchy import InstructionSource
from jarvis.permissions.model import Actor
from jarvis.tasks.models import Step, StepStatus, Task, TaskPolicy, TaskStatus
from jarvis.tools.base import ToolContext
from jarvis.tools.registry import ExecStatus

log = get_logger("plans")
P = PlanStatus
N = NodeStatus
_RESERVED_FACTS = {"summary", "outcome", "steps", "ok", "tests", "text"}
_LIGHT_TOOLS = {"system_info", "process_list", "process_inspect", "model_status", "time_now", "plan_analyze",
                "plan_verify", "plan_gate"}

_STATUS_EVENTS = {P.RUNNING: EventType.PLAN_STARTED, P.WAITING: EventType.PLAN_WAITING,
                  P.BLOCKED: EventType.PLAN_BLOCKED, P.PAUSED: EventType.PLAN_PAUSED,
                  P.REPLANNING: EventType.PLAN_REPLANNED, P.COMPLETED: EventType.PLAN_COMPLETED,
                  P.FAILED: EventType.PLAN_FAILED, P.CANCELLED: EventType.PLAN_CANCELLED}


class PlanEngine:
    def __init__(self, *, store: PlanStore, tasks: Any, registry: Any, permissions: Any, approvals: Any,
                 audit: AuditLog, bus: EventBus | None, builder: PlanBuilder, memory: PlanMemory,
                 clock: Clock | None = None, config: IntelligenceConfig | None = None, resources: Any = None,
                 router: Any = None, replanner: Any = None, coordinator: Any = None, data_dir: str | None = None,
                 autonomy: Callable[[], str] | None = None) -> None:
        self.store = store
        self.tasks = tasks
        self.registry = registry
        self.permissions = permissions
        self.approvals = approvals
        self.audit = audit
        self.bus = bus
        self.builder = builder
        self.memory = memory
        self.clock = clock or SystemClock()
        self.config = config or IntelligenceConfig()
        self.resources = resources
        self.router = router
        self.replanner = replanner
        self.coordinator = coordinator
        self.data_dir = data_dir
        self.autonomy = autonomy or (lambda: "normal")
        self.guard = LoopGuard(self.config)
        self._locks: dict[str, asyncio.Lock] = {}
        self._scheduled: set[str] = set()
        self._background: set[asyncio.Task[Any]] = set()
        self.enabled = True

    # -- wiring ---------------------------------------------------------------------------------------------
    def attach(self) -> None:
        if self.bus is not None:
            self.bus.subscribe("TASK_*", self._on_task_event, name="plan-engine")

    def _on_task_event(self, event: Event) -> None:
        plan_id = event.payload.get("plan_id")
        if plan_id and event.type != str(EventType.TASK_PROGRESS):
            self.schedule(plan_id)

    def schedule(self, plan_id: str) -> None:
        """Advance a plan soon (coalesces bursts of events into one pass)."""
        if plan_id in self._scheduled:
            return
        self._scheduled.add(plan_id)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._scheduled.discard(plan_id)
            return

        async def run() -> None:
            await asyncio.sleep(0)
            self._scheduled.discard(plan_id)
            try:
                await self.advance(plan_id)
            except Exception as exc:     # a bug in one plan must never take the engine down
                log.error("plan_advance_failed", plan_id=plan_id, error=repr(exc))

        task = loop.create_task(run(), name=f"plan-{plan_id}")
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def settle(self, timeout: float = 5.0) -> None:
        """Wait for scheduled plan passes (tests and shutdown)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while self._background and loop.time() < deadline:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    def _lock(self, plan_id: str) -> asyncio.Lock:
        lock = self._locks.get(plan_id)
        if lock is None:
            lock = self._locks[plan_id] = asyncio.Lock()
        return lock

    # -- lifecycle of a plan -----------------------------------------------------------------------------------
    def submit(self, plan: Plan, *, preview: bool = False) -> Plan:
        """Store a freshly built plan and validate it. With ``preview`` it waits for the user's go-ahead."""
        self.store.save_goal(plan.goal, session_id=plan.session_id, project_id=plan.project_id)
        plan.history.append({"ts": self.clock.now(), "from": None, "to": P.CREATED.value, "reason": "created",
                             "by": plan.created_by})
        self.store.save(plan)
        self._emit(EventType.GOAL_CREATED, plan, {"goal": plan.goal.objective, "kind": plan.goal.kind,
                                                  "complexity": plan.goal.complexity.label})
        self._emit(EventType.PLAN_CREATED, plan, {"nodes": len(plan.nodes), "source": plan.source})
        self._set(plan, P.VALIDATING, "validating")
        problems = self.builder.validate(plan)
        if problems:
            self._set(plan, P.FAILED, "the plan is not valid: " + "; ".join(problems[:3]))
            self.store.save(plan)
            return plan
        self._set(plan, P.READY, "ready")
        self._emit(EventType.PLAN_VALIDATED, plan, {"nodes": len(plan.nodes)})
        plan.awaiting_confirmation = preview
        self.store.save(plan)
        self._audit(plan, "plan_created", f"planned '{plan.title}' ({len(plan.nodes)} steps, {plan.source})",
                    OperationalReason(f"you asked: {plan.goal.text[:120]}", "plan complex requests before acting",
                                      f"built a {len(plan.nodes)}-step plan from {plan.source}"))
        return plan

    async def start(self, plan_id: str, *, by: str = "user") -> Plan | None:
        async with self._lock(plan_id):
            plan = self.store.get(plan_id)
            if plan is None or plan.status != P.READY:
                return plan
            if plan.awaiting_confirmation:
                plan.awaiting_confirmation = False
                gates = [n for n in plan.nodes if n.kind == NodeKind.GATE]
                plan.approvals.append({"kind": "preview", "by": by, "ts": self.clock.now(),
                                       "fingerprints": [e["fingerprint"] for g in gates
                                                        for e in g.steps[0]["args"].get("actions", [])]})
            plan.started_at = self.clock.now()
            self._set(plan, P.RUNNING, "started", by=by)
            self.store.save(plan)
        if plan.priority in (PlanPriority.EMERGENCY, PlanPriority.CRITICAL):
            await self._preempt(plan)
        await self.advance(plan_id)
        return self.store.get(plan_id)

    async def _preempt(self, urgent: Plan) -> None:
        """Emergency and critical plans come first: background and low-priority plans pause at their next step
        boundary and resume when the urgent plan is finished."""
        for other in self.store.open_plans():
            if other.id == urgent.id or other.priority not in (PlanPriority.LOW, PlanPriority.BACKGROUND):
                continue
            if other.status not in (P.RUNNING, P.WAITING):
                continue
            async with self._lock(other.id):
                fresh = self.store.get(other.id)
                if fresh is None or fresh.terminal:
                    continue
                for node in fresh.active_nodes():
                    if node.task_id:
                        self.tasks.pause_task(node.task_id, by="system:planner", source=InstructionSource.DEFAULT,
                                              reason=f"making way for '{urgent.title}'", at_step_boundary=True)
                fresh.paused_by = f"system:planner:{urgent.id}"
                self._set(fresh, P.PAUSED, f"paused to make way for '{urgent.title}' ({urgent.priority.value})",
                          by="system:planner")
                self.store.save(fresh)

    def _release_preempted(self, urgent: Plan) -> None:
        for other in self.store.list([P.PAUSED], limit=50):
            if other.paused_by == f"system:planner:{urgent.id}":
                self.schedule_resume(other.id)

    def schedule_resume(self, plan_id: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.resume(plan_id, by="system:planner", reason="the urgent work is finished"))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def pause(self, plan_id: str, *, by: str = "user", reason: str = "") -> tuple[bool, str]:
        async with self._lock(plan_id):
            plan = self.store.get(plan_id)
            if plan is None or plan.terminal:
                return False, "that plan isn't running"
            if plan.status == P.PAUSED:
                return True, f"{plan.title} is already paused"
            for node in plan.active_nodes():
                if node.task_id:
                    self.tasks.pause_task(node.task_id, by=by, reason=reason or f"plan paused by {by}")
                    # a paused plan asks nothing: its approval question is withdrawn and asked again on resume
                    self.approvals.cancel_for_task(node.task_id)
            plan.paused_by = by
            self._set(plan, P.PAUSED, reason or f"paused by {by}", by=by)
            self.store.save(plan)
            return True, f"paused {plan.title}"

    async def resume(self, plan_id: str, *, by: str = "user", reason: str = "") -> tuple[bool, str]:
        async with self._lock(plan_id):
            plan = self.store.get(plan_id)
            if plan is None:
                return False, "no such plan"
            if plan.status == P.FAILED:
                if not self.replanner:
                    return False, f"{plan.title} failed"
                self._set(plan, P.REPLANNING, f"retry requested by {by}", by=by)
                for node in plan.nodes:
                    if node.status in (N.FAILED, N.SKIPPED, N.CANCELLED) and node.kind != NodeKind.GATE \
                            and "declin" not in node.note:
                        node.reset(f"retried at the request of {by}")
                        node.attempts = 0
                plan.finished_at = None
                self._set(plan, P.RUNNING, "retrying", by=by)
            elif plan.status in (P.PAUSED, P.BLOCKED, P.WAITING):
                # the world may have changed while the plan was stopped: re-check what it relies on
                for a in plan.assumptions:
                    a.status = "unchecked"
                for node in plan.nodes:
                    if node.status in NODE_ACTIVE and node.task_id:
                        task = self.tasks.get_task(node.task_id)
                        if task and task.status in (TaskStatus.PAUSED, TaskStatus.BLOCKED):
                            self.tasks.resume_task(node.task_id, by=by, reason=reason or f"plan resumed by {by}")
                        elif task and task.status == TaskStatus.FAILED:
                            self.tasks.retry_task(node.task_id, by=by)
                    if node.status == N.BLOCKED and not node.task_id:
                        node.reset(f"resumed by {by}")
                    node.meta.pop("waiting_since", None)
                plan.paused_by = None
                self._set(plan, P.RUNNING, reason or f"resumed by {by}", by=by)
            elif plan.status == P.READY:
                pass
            else:
                return False, f"{plan.title} is {plan.status.value}"
            self.store.save(plan)
        if plan.status == P.READY:
            await self.start(plan_id, by=by)
        else:
            await self.advance(plan_id)
        plan = self.store.get(plan_id)
        return True, f"resumed {plan.title}" if plan else "resumed"

    async def cancel(self, plan_id: str, *, by: str = "user", reason: str = "") -> tuple[bool, str]:
        async with self._lock(plan_id):
            plan = self.store.get(plan_id)
            if plan is None or plan.terminal:
                return False, "that plan isn't open"
            reason = reason or f"cancelled by {by}"
            for node in plan.nodes:
                if node.task_id and node.status in NODE_ACTIVE:
                    self.tasks.cancel_task(node.task_id, by=by, reason=reason)
                if not node.finished:
                    node.status = N.CANCELLED
                    node.note = reason
            self._finish(plan, P.CANCELLED, reason, by=by)
            self.store.save(plan)
            return True, f"cancelled {plan.title}"

    # -- restart recovery ---------------------------------------------------------------------------------------
    async def recover(self) -> list[dict[str, Any]]:
        """After a restart (task recovery has already decided what to do with each task): bring every open plan
        up to date with its tasks. A node whose step had an unknown outcome keeps the plan waiting for the user;
        nothing is repeated blindly."""
        reports = []
        for plan in self.store.open_plans():
            async with self._lock(plan.id):
                plan = self.store.get(plan.id)
                if plan is None:
                    continue
                self._sync(plan)
                unknown = [n for n in plan.nodes if n.status == N.BLOCKED and (n.failure or {}).get("category")
                           == FailureCategory.UNKNOWN.value]
                if unknown and plan.status not in (P.PAUSED,):
                    reason = (f"interrupted by a restart: I can't tell whether '{unknown[0].title}' finished, so "
                              f"I haven't repeated it. Say 'continue {plan.title.lower()}' to run it again")
                    self._set(plan, P.PAUSED, reason, by="system:recovery", force=True)
                self.store.save(plan)
                summary = f"{plan.title}: {plan.status.value}" + (f" ({plan.status_reason})" if plan.status_reason
                                                                  else "")
                reports.append({"plan_id": plan.id, "title": plan.title, "status": plan.status.value,
                                "summary": summary})
                self._emit(EventType.PLAN_RECOVERED, plan, {"status": plan.status.value,
                                                            "reason": plan.status_reason},
                           Severity.WARNING if plan.status in (P.PAUSED, P.BLOCKED) else Severity.INFO)
        for r in reports:
            if r["status"] not in (P.PAUSED.value, P.READY.value):
                self.schedule(r["plan_id"])
        return reports

    async def tick(self) -> None:
        """Periodic reconciliation: missed events, due retries, resource waits, assumptions, age limits."""
        if not self.enabled:
            return
        for plan in self.store.open_plans():
            if plan.status in (P.PAUSED, P.CREATED, P.VALIDATING) or (plan.status == P.READY):
                continue
            await self.advance(plan.id)

    # -- the core loop --------------------------------------------------------------------------------------
    async def advance(self, plan_id: str) -> Plan | None:
        async with self._lock(plan_id):
            plan = self.store.get(plan_id)
            if plan is None or plan.terminal or plan.status in (P.CREATED, P.VALIDATING, P.READY):
                return plan
            self._sync(plan)
            if plan.status == P.PAUSED:
                self.store.save(plan)
                return plan
            limit = self.guard.age(plan, self.clock.now())
            if limit and plan.status != P.BLOCKED:
                self._limit(plan, limit)
                self.store.save(plan)
                return plan
            await self._recover_failures(plan)
            await self._progress(plan)
            await self._update_status(plan)
            self._check_deadline(plan)
            self.store.save(plan)
            return plan

    def _check_deadline(self, plan: Plan) -> None:
        """Long-horizon goals: warn once when the deadline is missed or the current pace won't make it."""
        deadline = plan.goal.deadline
        if not deadline or plan.terminal or plan.counters.get("deadline_warned"):
            return
        now = self.clock.now()
        progress = plan.progress()
        eta = None
        if plan.started_at and progress > 0:
            eta = plan.started_at + (now - plan.started_at) / progress
        if now > deadline or (eta is not None and eta > deadline and progress < 1):
            plan.counters["deadline_warned"] = 1
            self._emit(EventType.DEADLINE_AT_RISK, plan, {"deadline": deadline, "eta": eta, "progress": progress,
                                                         "milestones": _milestone_status(plan)}, Severity.WARNING)

    def _sync(self, plan: Plan) -> None:
        """Node states from their tasks."""
        for node in plan.nodes:
            if not node.task_id or node.status not in (N.RUNNING, N.WAITING, N.BLOCKED):
                continue
            task = self.tasks.get_task(node.task_id)
            if task is None:
                node.status = N.FAILED
                node.failure = {"category": FailureCategory.UNKNOWN.value, "detail": "its task record is missing"}
                continue
            s = task.status
            deferred = self.resources.explain(task.id) if (s == TaskStatus.QUEUED and self.resources) else None
            if deferred is not None and task.id not in getattr(self.resources, "locks", {}).values():
                # admission control is holding it back: the plan is waiting for resources, not running
                node.status = N.WAITING
                node.note = f"waiting for resources: {deferred.condition}"
                node.meta.setdefault("waiting_since", self.clock.now())
            elif s in (TaskStatus.QUEUED, TaskStatus.PLANNING, TaskStatus.RUNNING, TaskStatus.VERIFYING):
                node.status = N.RUNNING
                node.note = task.status_reason if s == TaskStatus.QUEUED else ""
                node.meta.pop("waiting_since", None)
            elif s == TaskStatus.WAITING:
                node.status = N.WAITING
                node.note = task.status_reason
            elif s == TaskStatus.PAUSED:
                unknown = any(st.outcome_unknown and not st.finished for st in task.plan)
                if unknown:
                    node.status = N.BLOCKED
                    node.failure = {"category": FailureCategory.UNKNOWN.value,
                                    "detail": task.recovery or "its outcome is unknown"}
                else:
                    node.status = N.WAITING
                node.note = task.status_reason
                node.meta.setdefault("waiting_since", self.clock.now())
            elif s == TaskStatus.BLOCKED:
                if node.status != N.BLOCKED:
                    node.status = N.BLOCKED
                    failure = classify(task)
                    node.failure = {**failure.to_dict(), "strategy": Strategy.ASK.value}
                node.note = task.status_reason
            elif s == TaskStatus.COMPLETED:
                self._completed(plan, node, task)
            elif s == TaskStatus.FAILED:
                if node.status != N.FAILED:
                    node.status = N.FAILED
                    node.finished_at = self.clock.now()
                    failure = classify(task)
                    node.failure = failure.to_dict()
                    node.error = failure.detail
                    node.meta["needs_recovery"] = True
            elif s in (TaskStatus.CANCELLED, TaskStatus.ABANDONED):
                node.status = N.CANCELLED
                node.note = task.status_reason or "cancelled"
                node.finished_at = self.clock.now()

    def _completed(self, plan: Plan, node: PlanNode, task: Task) -> None:
        facts: dict[str, Any] = {"summary": _first_line(task.result or task.outputs.get("summary", "")),
                                 "outcome": task.outcome.value if task.outcome else None, "steps": []}
        for index, st in enumerate(task.plan):
            data = (st.result or {}).get("data") if st.result else None
            facts["steps"].append(data)
            if index < len(node.steps) and node.steps[index].get("name"):
                name = node.steps[index]["name"]
                facts[name if name not in _RESERVED_FACTS else f"{name}_"] = data
        tests = task.outputs.get("tests")
        if tests is not None:
            facts["tests"] = tests
        facts["ok"] = task.outcome in (Outcome.COMPLETE, Outcome.UNKNOWN) and (tests or {}).get("ok", True) is not False
        if task.result:
            facts["text"] = task.result
        declined = [st for st in task.plan if st.status == StepStatus.SKIPPED]
        plan.facts[node.id] = facts
        node.finished_at = self.clock.now()
        node.summary = facts["summary"]
        if node.kind in (NodeKind.GATHER, NodeKind.ANALYZE) and len(task.plan) > 1:
            # an observation's summary is everything it observed, not only its last step
            node.summary = "; ".join(str((st.result or {}).get("summary") or "") for st in task.plan
                                     if st.result and st.result.get("summary"))[:300]
        if node.kind == NodeKind.GATE:
            if declined or not task.plan or task.plan[0].status != StepStatus.DONE:
                node.status = N.SKIPPED
                node.note = "declined by you"
                for nid in node.gate_for:
                    target = plan.node(nid)
                    if target and not target.finished:
                        target.status = N.SKIPPED
                        target.note = "you declined it"
                self._record_approval(plan, node, task, approved=False)
            else:
                node.status = N.DONE
                approved: dict[str, list[str]] = {}
                for e in node.steps[0]["args"].get("actions", []):
                    approved.setdefault(e["node"], []).append(e["fingerprint"])
                node.approved = approved
                self._record_approval(plan, node, task, approved=True)
            self._node_event(plan, node)
            return
        node.status = N.DONE
        node.quality = node_quality(node, task, facts)
        if node.kind == NodeKind.ACTION:
            plan.facts.setdefault("actions", {})[node.id] = {"ok": facts["ok"] and node.quality != Quality.FAILED,
                                                             "summary": node.summary}
        if node.kind == NodeKind.AGENT and self.coordinator is not None:
            self.coordinator.collect(plan, node, facts)
        if node.kind == NodeKind.VERIFY:
            result = facts.get("verdict") or {}
            self._emit(EventType.VERIFICATION_PERFORMED, plan, {"node": node.id, "quality": node.quality.value
                                                                if node.quality else None,
                                                                "detail": result.get("detail", "")})
            self._audit(plan, "plan_verification", f"{node.title}: {result.get('detail', '')}",
                        OperationalReason(f"the verifier re-measured after '{plan.title}'",
                                          "the executor never certifies its own work",
                                          f"judged the result {node.quality.label if node.quality else 'unverified'}",
                                          result.get("detail", "")))
            if node.quality in (Quality.FAILED, Quality.CONFLICTING) or (
                    node.quality == Quality.PARTIALLY_VERIFIED and result.get("resolved") is False
                    and plan.goal.kind == "performance"):
                node.meta["needs_recovery"] = True
        self._close_loop(plan, node)
        self._node_event(plan, node)

    def _record_approval(self, plan: Plan, gate: PlanNode, task: Task, *, approved: bool) -> None:
        approval = next((a for a in self._approvals_for(task.id)), None)
        preview = next((a["by"] for a in plan.approvals if a.get("kind") == "preview"), None)
        by = (approval.decided_by if approval and approval.decided_by else preview) or plan.owner
        plan.approvals.append({"kind": "gate", "node": gate.id, "approved": approved, "by": by,
                               "approval_id": approval.id if approval else None, "ts": self.clock.now(),
                               "actions": [e["preview"] for e in gate.steps[0]["args"].get("actions", [])]})
        gate.meta["approved_by"] = by if approved else None

    def _approvals_for(self, task_id: str) -> list[Any]:
        rows = self.approvals.db.query("SELECT id FROM approvals WHERE task_id=? ORDER BY created_at DESC", (task_id,))
        return [a for a in (self.approvals.get(r["id"]) for r in rows) if a is not None]

    def _close_loop(self, plan: Plan, node: PlanNode) -> None:
        """Bounded loops: re-run a stretch of the plan until a condition holds (at most ``max`` times)."""
        if not node.loop or node.status != N.DONE:
            return
        until = plan.evaluate(node.loop.get("until"))
        if until is not False:
            return
        limit = self.guard.loop_iteration(node)
        if limit:
            node.note = limit.reason
            self._limit_event(plan, limit)
            return
        start = plan.node(node.loop.get("restart_from") or node.id) or node
        # every node on a path from the loop's start to this node runs again
        between = [n for n in [start] + plan.descendants(start.id)
                   if n.id == node.id or node.id in {d.id for d in plan.descendants(n.id)}]
        node.iterations += 1
        iterations = node.iterations
        for n in {n.id: n for n in between + [node]}.values():
            n.reset(f"repeating (round {iterations + 1})")
            n.attempts = 0                # a new round, not a retry: failure retries are counted per round
        node.iterations = iterations

    # -- failures ------------------------------------------------------------------------------------------------
    async def _recover_failures(self, plan: Plan) -> None:
        for node in list(plan.nodes):
            if not node.meta.pop("needs_recovery", False):
                continue
            task = self.tasks.get_task(node.task_id) if node.task_id else None
            verification = node.kind == NodeKind.VERIFY
            failure = classify(task, verification_failed=verification)
            if node.failure and not verification:
                failure.category = FailureCategory(node.failure.get("category", failure.category.value))
            plan.failures.append({"ts": self.clock.now(), "node": node.id, **failure.to_dict()})
            limit = self.guard.failure(plan, failure)
            if node.kind == NodeKind.AGENT and node.alternatives and not limit:
                # agent failures are isolated: the same work is done deterministically instead
                if await self._apply(plan, node, Strategy.ALTERNATIVE, failure):
                    node.failure = {**failure.to_dict(), "strategy": Strategy.ALTERNATIVE.value}
                    continue
            if node.optional and failure.category not in (FailureCategory.VERIFICATION,):
                node.note = f"optional; continuing without it ({failure.detail[:120]})"
                continue
            handled = False
            for strategy in STRATEGIES[failure.category]:
                if limit:
                    break
                if failure.category == FailureCategory.PERMISSION and "safety" in failure.detail:
                    break                     # never allowed: asking again would be pointless
                handled = await self._apply(plan, node, strategy, failure)
                if handled:
                    node.failure = {**failure.to_dict(), "strategy": strategy.value}
                    break
            if limit:
                self._limit_event(plan, limit)
                node.note = limit.reason
            if not handled and node.kind == NodeKind.VERIFY:
                # the verification ran and says the goal isn't met, and there's no other approach to try:
                # the result is reported honestly rather than retried
                node.note = node.note or "no other approach left to try"

    async def _apply(self, plan: Plan, node: PlanNode, strategy: Strategy, failure: Any) -> bool:
        now = self.clock.now()
        if strategy == Strategy.RETRY:
            if self.guard.node_attempt(plan, node):
                return False
            delay = self.config.retry_backoff_s * (2 ** max(0, node.attempts - 1))
            node.reset(f"retrying in {delay:g}s after: {failure.detail[:100]}")
            node.meta["retry_at"] = now + delay
            self._audit(plan, "plan_retry", f"retrying '{node.title}'",
                        OperationalReason(f"'{node.title}' failed ({failure.detail[:120]})",
                                          "transient failures are retried a limited number of times",
                                          "scheduled a retry", f"next attempt in {delay:g}s"))
            return True
        if strategy == Strategy.ALTERNATIVE:
            if not node.alternatives or self.guard.node_attempt(plan, node):
                return False
            previous = [s["tool"] for s in node.steps]
            node.steps = node.alternatives.pop(0)
            if node.kind == NodeKind.AGENT and node.meta.get("fallback_collect"):
                node.meta["was_agent"] = node.agent
                node.kind = NodeKind.GATHER
                node.meta["collect"] = node.meta.pop("fallback_collect")
            node.reset(f"using another way after {', '.join(previous)} failed")
            plan.replans.append({"version": plan.version, "trigger": "tool_failure", "ts": now,
                                 "reason": f"'{node.title}' failed ({failure.detail[:120]})",
                                 "changed": [f"{node.id}: {', '.join(previous)} → "
                                             f"{', '.join(s['tool'] for s in node.steps)}"]})
            self._audit(plan, "plan_alternative", f"switched '{node.title}' to another tool",
                        OperationalReason(f"{', '.join(previous)} failed ({failure.detail[:120]})",
                                          "when a tool fails, use another way to get the same information",
                                          f"switched to {', '.join(s['tool'] for s in node.steps)}",
                                          "Completed steps are kept."))
            self._emit(EventType.PLAN_REPLANNED, plan, {"trigger": "tool_failure", "node": node.id})
            return True
        if strategy == Strategy.WAIT:
            node.status = N.WAITING
            node.note = f"waiting for resources ({failure.detail[:100]})"
            node.meta.setdefault("waiting_since", now)
            node.meta["retry_when"] = "resources"
            node.task_id = None
            return True
        if strategy == Strategy.REPLAN:
            if self.replanner is None:
                return False
            trigger = "verification" if failure.category == FailureCategory.VERIFICATION else "failure"
            return await self.replanner.replan(plan, trigger, failure.detail, node=node)
        if strategy == Strategy.ASK:
            node.status = N.BLOCKED
            node.note = failure.detail[:200]
            return True
        if strategy == Strategy.SKIP:
            node.status = N.SKIPPED
            node.note = failure.detail[:200]
            return True
        return False

    # -- progress -----------------------------------------------------------------------------------------------
    async def _progress(self, plan: Plan) -> None:
        for _ in range(len(plan.nodes) * 3 + 5):          # bounded: each pass finishes or starts something
            progressed = False
            running = [n for n in plan.nodes if n.status == N.RUNNING]
            for node in plan.ready_nodes():
                blocking = plan.blocking_dependency(node)
                if blocking is not None:
                    node.status = N.SKIPPED
                    node.note = f"not run because '{blocking.title}' didn't succeed"
                    progressed = True
                    continue
                condition = plan.evaluate(node.condition)
                if condition is None:
                    continue
                if condition is False:
                    node.status = N.SKIPPED
                    node.note = node.note or _condition_note(plan, node.condition)
                    progressed = True
                    continue
                invalid = self._check_assumptions(plan, node)
                if invalid is not None:
                    await self._assumption_failed(plan, node, invalid)
                    progressed = True
                    break
                if node.kind == NodeKind.DECIDE:
                    await self._decide(plan, node)
                    progressed = True
                    continue
                if node.kind == NodeKind.REPORT:
                    self._report(plan, node)
                    progressed = True
                    continue
                if node.meta.get("retry_at") and node.meta["retry_at"] > self.clock.now():
                    continue
                if len(running) >= self.config.max_parallel_nodes:
                    break
                if self._defer_for_resources(plan, node):
                    continue
                await self._materialize(plan, node)
                if node.status == N.RUNNING:
                    running.append(node)
                progressed = True
            # nodes waiting for resources get another chance when pressure clears
            for node in plan.nodes:
                if node.status == N.WAITING and node.meta.get("retry_when") == "resources" and not node.task_id:
                    if not self._pressure()[0]:
                        node.reset("resources are available again")
                        node.meta.pop("retry_when", None)
                        progressed = True
                    else:
                        limit = self.guard.resource_wait(node, self.clock.now())
                        if limit:
                            node.status = N.BLOCKED
                            node.note = limit.reason
                            self._limit_event(plan, limit)
            if not progressed:
                return

    def _pressure(self) -> tuple[bool, str]:
        if self.resources is None:
            return False, ""
        return self.resources.pressure()

    def _defer_for_resources(self, plan: Plan, node: PlanNode) -> bool:
        """Resource-aware scheduling: heavy, low-priority work waits while the machine is under pressure."""
        constrained, why = self._pressure()
        priority = (node.priority or plan.priority).task_priority(plan.interactive)
        heavy = node.estimate.get("cpu") == "high" or node.estimate.get("model_calls", 0) > 0 or \
            node.kind in (NodeKind.AGENT,)
        if constrained and heavy and priority >= Priority.P2:
            if node.status != N.WAITING:
                node.status = N.WAITING
                node.note = f"waiting for resources: {why}"
                node.meta["retry_when"] = "resources"
                node.meta.setdefault("waiting_since", self.clock.now())
                self._audit(plan, "plan_deferred", f"deferred '{node.title}'",
                            OperationalReason(why, "heavy background work waits while resources are short",
                                              f"deferred '{node.title}'", "It starts automatically when possible."))
            return True
        return False

    # -- assumptions ----------------------------------------------------------------------------------------------
    def _check_assumptions(self, plan: Plan, node: PlanNode) -> Assumption | None:
        for a in plan.assumptions:
            if node.id not in a.affects:
                continue                  # (an invalid one is checked again: the world may have changed back)
            holds, evidence = self._evaluate_assumption(plan, a)
            a.checked_at = self.clock.now()
            a.evidence = evidence
            if holds is None:
                continue
            a.status = "holding" if holds else "invalid"
            if not holds:
                return a
        return None

    def _evaluate_assumption(self, plan: Plan, a: Assumption) -> tuple[bool | None, str]:
        check = a.check
        kind = check.get("type")
        if kind == "path_exists":
            path = os.path.expanduser(str(check.get("path", "")))
            exists = os.path.exists(path)
            return exists, f"{path} {'exists' if exists else 'is not there'}"
        if kind == "process_running":
            try:
                import psutil
                pid = int(check["pid"])
                alive = psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
                if alive and check.get("name"):
                    alive = psutil.Process(pid).name().lower() == str(check["name"]).lower()
            except Exception:
                alive = False
            return alive, f"PID {check.get('pid')} is {'running' if alive else 'no longer running'}"
        if kind == "model_available":
            ok = bool(self.router and self.router.available())
            return ok, "a language model is available" if ok else "no language model is available"
        if kind == "tool_available":
            ok = self.registry.get(check.get("tool", "")) is not None
            return ok, f"{check.get('tool')} is {'available' if ok else 'not available'}"
        if kind == "fact":
            result = plan.evaluate(check.get("condition"))
            return result, f"condition {'holds' if result else 'does not hold'}"
        return None, "unchecked"

    async def _assumption_failed(self, plan: Plan, node: PlanNode, assumption: Assumption) -> None:
        self._emit(EventType.ASSUMPTION_INVALIDATED, plan, {"assumption": assumption.statement,
                                                            "evidence": assumption.evidence, "node": node.id},
                   Severity.WARNING)
        self._audit(plan, "assumption_invalidated", assumption.statement,
                    OperationalReason(assumption.evidence, "a plan must not carry on when what it relies on changed",
                                      "stopped to re-plan", ""))
        applied = False
        if self.replanner is not None:
            applied = await self.replanner.replan(plan, "assumption", assumption.evidence, node=node,
                                                  assumption=assumption)
        if not applied:
            node.status = N.BLOCKED
            node.note = f"{assumption.statement} no longer holds ({assumption.evidence})"

    # -- decisions --------------------------------------------------------------------------------------------------
    async def _decide(self, plan: Plan, node: PlanNode) -> None:
        if node.meta.get("select") == "sources":
            self._select_sources(plan, node)
            return
        analysis = plan.fact(f"{node.meta.get('from')}.analysis") or {}
        candidates = [c for c in analysis.get("candidates", []) if isinstance(c, dict)]
        tried = {a.get("identity") for d in plan.decisions for a in d.get("actions", [])}
        fix = bool(node.meta.get("fix")) and plan.mode == ExecutionMode.EXECUTE
        min_conf = float(node.meta.get("min_confidence", 0.55))
        viable = []
        for c in candidates:
            why_not = c.get("blocked")
            tool = self.registry.get(c.get("tool", ""))
            if tool is None:
                why_not = why_not or f"{c.get('tool')} isn't available"
            else:
                assessed = tool.assess(c.get("args") or {})
                why_not = why_not or plan.goal.forbids(c["tool"], c.get("args") or {}, c.get("meta"),
                                                       level=int(assessed.level)) or \
                    ("refused by the safety policy" if assessed.blocked else None)
            if why_not:
                c["blocked"] = why_not
            elif action_identity(c) in tried:
                c["blocked"] = "already tried in this plan"
            elif c.get("confidence", 0) >= min_conf:
                viable.append(c)
        if not plan.interactive:
            # nobody asked for this right now (an event or a schedule started it), so nobody can approve a change
            # in the moment: changes above the plan's baseline become recommendations for the user
            baseline = self.builder.baseline(False)
            viable = [c for c in viable if int(c.get("level", 5)) <= baseline]
        chosen = viable[: int(node.meta.get("max_actions", 1))] if fix else []
        findings = [f.get("text", "") for f in analysis.get("findings", [])]
        decision_text = ("; ".join(c["title"] for c in chosen) if chosen else
                         ("recommend: " + "; ".join(analysis.get("recommendations", [])[:3])
                          if analysis.get("recommendations") else "change nothing"))
        if chosen:
            reason = chosen[0]["reason"]
        elif not fix:
            reason = "you asked me to find out, not to change anything" if plan.mode == ExecutionMode.EXECUTE else \
                f"this is {'a ' + plan.mode.value.replace('_', ' ') if plan.mode != ExecutionMode.ADVISE else 'advice'}; " \
                "nothing is changed"
        else:
            reason = "no safe change is supported strongly enough by the evidence"
        decision = {"id": new_id("pdec"), "node": node.id, "title": f"What to do about: {plan.goal.objective[:80]}",
                    "decision": decision_text, "reason": reason, "evidence": findings[:8],
                    "cause": analysis.get("cause", ""), "confidence": analysis.get("confidence"),
                    "alternatives": [c["title"] + (f" (ruled out: {c['blocked']})" if c.get("blocked") else "")
                                     for c in candidates if c not in chosen][:6],
                    "recommendations": analysis.get("recommendations", []),
                    "actions": [], "mode": plan.mode.value, "ts": self.clock.now()}
        decision["decision_log_id"] = self.memory.record_decision(
            plan, decision["title"], decision_text, evidence=findings[:8], alternatives=decision["alternatives"],
            reason=reason)
        facts: dict[str, Any] = {"actions": [], "targets": [], "paths": [], "cause": analysis.get("cause", ""),
                                 "recommendations": analysis.get("recommendations", [])}
        new_nodes: list[PlanNode] = []
        limit = self.guard.nodes(plan, extra=len(chosen) + 1)
        if limit and chosen:
            self._limit_event(plan, limit)
            chosen = []
        round_no = node.meta.get("round", 1)
        prefix = node.id.replace("decide", "") if node.id.endswith("decide") else ""
        for index, c in enumerate(chosen, 1):
            act_id = f"{prefix}act{round_no}_{index}"
            act = PlanNode(act_id, c["title"][:1].upper() + c["title"][1:], NodeKind.ACTION, depends_on=[node.id],
                           steps=[make_step(c["tool"], c.get("args") or {}, c["title"], "act")],
                           rollback=c.get("rollback", ""),
                           meta={**(c.get("meta") or {}), "key": c["key"], "expected": c.get("expected", ""),
                                 "confidence": c.get("confidence"), "reason": c.get("reason", "")})
            new_nodes.append(act)
            facts["actions"].append(act_id)
            facts["targets"].append({"node": act_id, "tool": c["tool"], "pid": (c.get("args") or {}).get("pid"),
                                     "target": (c.get("meta") or {}).get("target")})
            if c["tool"] == "file_delete":
                facts["paths"].append((c.get("args") or {}).get("path"))
            target = (c.get("meta") or {}).get("target")
            label = f"{c['tool'].split('_')[-1]}ping {target}" if c["tool"] == "process_stop" and target else c["title"]
            decision["actions"].append({"key": c["key"], "identity": action_identity(c), "title": c["title"],
                                        "label": label, "node": act_id, "outcome": None})
            if c["tool"] == "file_delete" and (c.get("args") or {}).get("path"):
                plan.assumptions.append(Assumption(
                    f"{os.path.basename(str(c['args']['path']))} is still there",
                    {"type": "path_exists", "path": c["args"]["path"]}, [act_id]))
            if c["tool"] == "process_stop" and (c.get("args") or {}).get("pid"):
                plan.assumptions.append(Assumption(
                    f"{(c.get('meta') or {}).get('target')} (PID {c['args']['pid']}) is still running",
                    {"type": "process_running", "pid": c["args"]["pid"], "name": (c.get("meta") or {}).get("target")},
                    [act_id]))
        if plan.mode == ExecutionMode.DRY_RUN and viable:
            facts["expected_changes"] = await self._dry_run(plan, viable[: int(node.meta.get("max_actions", 1))])
        plan.decisions.append(decision)
        plan.facts[node.id] = facts
        position = plan.nodes.index(node) + 1
        plan.nodes[position:position] = new_nodes
        if new_nodes:
            self.builder.insert_gates(plan, new_nodes)
        verify = plan.node(node.meta.get("verify", "")) if node.meta.get("verify") else None
        if verify is not None:
            verify.depends_on = list(dict.fromkeys(verify.depends_on + [n.id for n in new_nodes]))
            if new_nodes:
                verify.condition = {"any": [{"node": n.id} for n in new_nodes]}
        node.status = N.DONE
        node.finished_at = self.clock.now()
        node.summary = decision_text
        self._emit(EventType.DECISION_RECORDED, plan, {"node": node.id, "decision": decision_text,
                                                       "reason": reason, "actions": len(new_nodes)})
        self._audit(plan, "plan_decision", decision_text,
                    OperationalReason(analysis.get("cause") or "the analysis finished",
                                      "act on the strongest evidence; never touch protected processes or anything "
                                      "you ruled out", f"decided: {decision_text}",
                                      chosen[0].get("expected", "") if chosen else ""))
        self._node_event(plan, node)

    async def _dry_run(self, plan: Plan, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """PLAN → DRY RUN → EXPECTED CHANGES: evaluate each action through the registry without running it."""
        out = []
        ctx = ToolContext(actor=Actor("user" if plan.interactive else "system", plan.owner, plan.owner,
                                      interactive=plan.interactive), cwd=plan.cwd, dry_run=True, clock=self.clock,
                          data_dir=self.data_dir)
        for c in candidates:
            ex = await self.registry.execute(c["tool"], c.get("args") or {}, ctx, reason=OperationalReason(
                f"dry run of '{plan.title}'", "show expected changes before making them", f"evaluated {c['tool']}"))
            if ex.status == ExecStatus.DRY_RUN:
                would = "would need your approval" if ex.decision and ex.decision.needs_approval else "would run"
            else:
                would = f"would be refused: {ex.message}"
            out.append({"title": c["title"], "would": would, "expected": c.get("expected", ""),
                        "reversible": c.get("reversible", True)})
        return out

    def _select_sources(self, plan: Plan, node: PlanNode) -> None:
        """Research: choose the files to read from the search results, then read them in parallel (or hand
        groups of them to research agents when there are many and a model is available)."""
        search = plan.facts.get(node.meta.get("from", ""), {}) or {}
        root = node.meta.get("root") or plan.cwd or "."
        counts: dict[str, set[str]] = {}
        for key, value in search.items():
            if not key.startswith("kw") or not isinstance(value, dict):
                continue
            for m in value.get("matches") or []:
                counts.setdefault(m.get("path"), set()).add(key)
        ranked = sorted(counts, key=lambda p: (-len(counts[p]), p))
        files = [os.path.join(root, p) for p in ranked if p][: int(node.meta.get("max_sources", 8))]
        new_nodes: list[PlanNode] = []
        use_agents = bool(node.meta.get("agents")) and len(ranked) > int(node.meta.get("many", 8)) and \
            self.coordinator is not None and not self._pressure()[0]
        if use_agents:
            new_nodes = self.coordinator.research_nodes(plan, node, [os.path.join(root, p) for p in ranked[:16]],
                                                        topic=node.meta.get("topic", ""))
        else:
            for index, path in enumerate(files, 1):
                new_nodes.append(PlanNode(f"{node.id}_read{index}", f"Read {os.path.basename(path)}", NodeKind.GATHER,
                                          depends_on=[node.id], optional=True, meta={"collect": "documents"},
                                          steps=[make_step("file_read", {"path": path, "max_bytes": 200_000},
                                                           f"read {path}", "document")]))
        position = plan.nodes.index(node) + 1
        plan.nodes[position:position] = new_nodes
        extract = plan.node(node.meta.get("extract", ""))
        if extract is not None:
            extract.depends_on = list(dict.fromkeys(extract.depends_on + [n.id for n in new_nodes]))
        plan.facts[node.id] = {"files": files, "agents": use_agents, "matched_files": len(ranked)}
        node.status = N.DONE
        node.finished_at = self.clock.now()
        node.summary = (f"{len(ranked)} file(s) matched; " + (f"{len(new_nodes)} research agent(s) will read them"
                                                               if use_agents else f"reading {len(files)}"))
        self._node_event(plan, node)

    # -- reports -------------------------------------------------------------------------------------------------
    def _report(self, plan: Plan, node: PlanNode) -> None:
        plan.quality = plan_quality(plan)
        text = explain.compose_report(plan)
        plan.facts[node.id] = {"report": text}
        plan.result = text
        node.status = N.DONE
        node.finished_at = self.clock.now()
        node.summary = _first_line(text)
        self._node_event(plan, node)

    # -- materialising nodes as tasks ------------------------------------------------------------------------------
    async def _materialize(self, plan: Plan, node: PlanNode) -> None:
        limit = self.guard.node_attempt(plan, node)
        if limit:
            node.status = N.FAILED
            node.note = limit.reason
            self._limit_event(plan, limit)
            return
        if node.kind == NodeKind.GATE and not await self._prepare_gate(plan, node):
            return
        node.attempts += 1
        steps = []
        for s in node.steps:
            steps.append(Step(s.get("description") or s["tool"], s["tool"], self._resolve(plan, s.get("args") or {}),
                              allow_failure=bool(s.get("allow_failure"))))
        verifier = node.actor == "verifier"
        created_by = "agent:verifier" if verifier else plan.created_by
        dry_run = bool(node.meta.get("dry_run_only")) and plan.mode.changes_nothing
        on_failure = "replan" if (node.kind == NodeKind.TASK and not node.steps) else "fail"
        policy = TaskPolicy(max_retries=1, on_step_failure=on_failure, max_replans=1,
                            notify_on=["waiting"], resume_after_restart="safe")
        priority = (node.priority or plan.priority).task_priority(plan.interactive)
        if node.steps and all(s["tool"] in _LIGHT_TOOLS for s in node.steps) and priority > Priority.P1:
            # light measurements are never deferred by resource pressure: JARVIS must be able to measure the very
            # problem it is under (they take a second and a few megabytes)
            priority = Priority.P1
        task = self.tasks.create_task(
            node.objective or f"{node.title} — part of '{plan.title}'",
            title=plan.title if node.kind == NodeKind.GATE else node.title, steps=steps or None,
            priority=priority, owner=plan.owner,
            created_by=created_by, project_id=plan.project_id, cwd=plan.cwd, dry_run=dry_run,
            authority={"interactive": plan.interactive and not verifier}, success_condition=node.verify,
            outputs={"plan_id": plan.id, "plan_node": node.id}, policy=policy, resources=list(node.resources),
            budget={"max_steps": 20, "max_model_calls": 6}, request=plan.goal.text, origin=f"plan:{plan.id}",
            # one key per materialisation: a repeated pass (or a restart between creating the task and saving the
            # plan) finds the same task instead of starting the work twice
            idempotency_key=f"plan:{plan.id}:{node.id}:{len(node.task_ids)}")
        node.task_id = task.id
        if task.id not in node.task_ids:
            node.task_ids.append(task.id)
        node.status = N.RUNNING
        node.started_at = self.clock.now()
        node.meta.pop("retry_at", None)
        self._grant_approved(plan, node, task)
        self._emit(EventType.PLAN_NODE_STARTED, plan, {"node": node.id, "title": node.title, "kind": node.kind.value,
                                                       "task_id": task.id})
        if node.kind == NodeKind.AGENT:
            self._emit(EventType.AGENT_ASSIGNED, plan, {"node": node.id, "agent": node.agent, "task_id": task.id})

    async def _prepare_gate(self, plan: Plan, gate: PlanNode) -> bool:
        """Before asking: dry-run every gated action through the registry. Actions the registry would refuse
        (safety, scope) are dropped with the reason; the rest become the expected changes the user approves."""
        args = gate.steps[0]["args"]
        ctx = ToolContext(actor=Actor("task", "gate-preview", plan.owner, interactive=plan.interactive),
                          cwd=plan.cwd, dry_run=True, clock=self.clock, data_dir=self.data_dir)
        kept = []
        for entry in args.get("actions", []):
            target = plan.node(entry["node"])
            if target is None or target.finished:
                continue
            gone = self._target_gone(plan, target)
            if gone:
                # the file was removed or the process exited since the analysis: nothing to ask about
                target.status = N.SKIPPED
                target.note = f"no longer needed: {gone}"
                continue
            refused = None
            for s in target.steps:
                ex = await self.registry.execute(s["tool"], self._resolve(plan, s.get("args") or {}), ctx,
                                                 reason=OperationalReason(f"checking '{target.title}' before asking",
                                                                          "dry run before approval",
                                                                          f"evaluated {s['tool']}"))
                # refusals no approval can lift (safety, scope, mode) drop the action; a plain "needs more
                # authority" is exactly what the gate is for
                if ex.status == ExecStatus.DENIED and ex.decision is not None and ex.decision.basis == "permission":
                    continue
                if ex.status not in (ExecStatus.DRY_RUN,):
                    refused = ex.message
                    break
            if refused:
                target.status = N.SKIPPED
                target.note = f"refused before asking: {refused}"
            else:
                kept.append(entry)
        if not kept:
            gate.status = N.SKIPPED
            gate.note = "nothing left to approve"
            return False
        from jarvis.intelligence.builder import gate_summary
        args["actions"] = kept
        args["summary"] = gate_summary(kept)
        gate.gate_for = [e["node"] for e in kept]
        gate.meta["expected"] = kept
        return True

    def _target_gone(self, plan: Plan, node: PlanNode) -> str | None:
        """For a deletion or a process stop: the evidence that its target no longer exists, else None."""
        if not any(s["tool"] in ("file_delete", "process_stop") for s in node.steps):
            return None
        for a in plan.assumptions:
            if node.id in a.affects and (a.check or {}).get("type") in ("path_exists", "process_running"):
                holds, evidence = self._evaluate_assumption(plan, a)
                if holds is False:
                    a.status, a.evidence, a.checked_at = "invalid", evidence, self.clock.now()
                    return evidence
        return None

    def _grant_approved(self, plan: Plan, node: PlanNode, task: Task) -> None:
        """Approved actions get exactly the authority the user approved: single-use, scoped to this task and tool,
        and only if the action is byte-for-byte the one that was shown."""
        if node.kind == NodeKind.GATE:
            prints = {e["fingerprint"] for e in node.steps[0]["args"].get("actions", [])}
            previewed = {f for a in plan.approvals if a.get("kind") == "preview" for f in a.get("fingerprints", [])}
            if prints and prints <= previewed:
                who = next(a["by"] for a in plan.approvals if a.get("kind") == "preview")
                self.permissions.grant(f"task:{task.id}", 4, tools=["plan_gate"], task_id=task.id, max_uses=1,
                                       ttl_s=3600, reason=f"approved in the plan preview: {plan.title}",
                                       created_by=_approver(who))
            return
        gate = next((g for g in plan.nodes if g.kind == NodeKind.GATE and node.id in g.gate_for
                     and g.status == N.DONE), None)
        if gate is None:
            return
        expected = list(gate.approved.get(node.id) or [])
        actual = [fingerprint(s["tool"], s.get("args") or {}) for s in node.steps]
        if sorted(actual) != sorted(expected):
            log.warning("approval_mismatch", plan_id=plan.id, node=node.id)
            return                         # changed since it was approved: it must be approved again
        tools = sorted({s["tool"] for s in node.steps})
        level = max((int(self.registry.get(s["tool"]).assess(s.get("args") or {}).level) for s in node.steps
                     if self.registry.get(s["tool"])), default=4)
        self.permissions.grant(f"task:{task.id}", min(level, 4), tools=tools, task_id=task.id,
                               max_uses=len(node.steps), ttl_s=3600,
                               reason=f"approved as part of '{plan.title}': {node.title}",
                               created_by=_approver(gate.meta.get("approved_by") or plan.owner))

    def _resolve(self, plan: Plan, value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"$fact"}:
                return plan.fact(str(value["$fact"]))
            if set(value) == {"$collect"}:
                return self._collect(plan, str(value["$collect"]))
            if set(value) == {"$from_step"}:
                return value                 # resolved by the executor inside the task
            return {k: self._resolve(plan, v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve(plan, v) for v in value]
        return value

    def _collect(self, plan: Plan, what: str) -> list[Any]:
        out: list[Any] = []
        for n in plan.nodes:
            if n.meta.get("collect") != what or n.status != N.DONE:
                continue
            facts = plan.facts.get(n.id) or {}
            if what == "agent_claims":
                out.extend(facts.get("claims") or [])
            else:
                out.extend(d for d in facts.get("steps") or [] if d is not None)
        return out

    # -- status --------------------------------------------------------------------------------------------------
    async def _update_status(self, plan: Plan) -> None:
        if plan.status in (P.PAUSED, P.REPLANNING) or plan.terminal:
            return
        nodes = plan.nodes
        if all(n.finished for n in nodes):
            await self._complete(plan)
            return
        running = [n for n in nodes if n.status == N.RUNNING]
        waiting = [n for n in nodes if n.status == N.WAITING]
        blocked = [n for n in nodes if n.status == N.BLOCKED]
        if running or plan.ready_nodes():
            self._set(plan, P.RUNNING, _running_reason(running))
        elif blocked:
            self._set(plan, P.BLOCKED, f"'{blocked[0].title}': {blocked[0].note or 'needs you'}")
        elif waiting:
            self._set(plan, P.WAITING, f"'{waiting[0].title}': {waiting[0].note or 'waiting'}")
        else:
            pending = [n for n in nodes if not n.finished]
            self._set(plan, P.BLOCKED, f"'{pending[0].title}' can't start" if pending else "stuck")

    async def _complete(self, plan: Plan) -> None:
        self._set(plan, P.VERIFYING, "checking the result")
        plan.quality = plan_quality(plan)
        failed = [n for n in plan.nodes if n.status == N.FAILED and not n.optional]
        # the goal is met only if the latest independent check says so: a change that was made but didn't
        # resolve the problem is not "done"
        verify_failed = [n for n in plan.nodes if n.kind == NodeKind.VERIFY and n.status == N.DONE
                         and not n.meta.get("superseded_by")
                         and (n.quality == Quality.FAILED
                              or (plan.facts.get(n.id) or {}).get("verdict", {}).get("resolved") is False)]
        if not plan.result:
            plan.result = explain.compose_report(plan)
        self._record_action_outcomes(plan)
        if failed:
            reason = f"'{failed[0].title}' failed: {failed[0].error or failed[0].note or 'see the report'}"
            self._finish(plan, P.FAILED, reason)
        elif verify_failed:
            detail = (plan.facts.get(verify_failed[-1].id) or {}).get("verdict", {}).get("detail", "")
            self._finish(plan, P.FAILED, f"verification: {detail}" if detail else "verification failed")
        else:
            self._finish(plan, P.COMPLETED, _first_line(plan.result) or "done")
        await self.memory.record_outcome(plan)

    def _record_action_outcomes(self, plan: Plan) -> None:
        verify_quality = {n.id: n for n in plan.nodes if n.kind == NodeKind.VERIFY}
        for decision in plan.decisions:
            for action in decision.get("actions", []):
                node = plan.node(action.get("node", ""))
                if node is None:
                    continue
                if node.status == N.SKIPPED:
                    action["outcome"] = "declined" if "declin" in node.note else "skipped"
                elif node.status != N.DONE:
                    action["outcome"] = "failed"
                else:
                    covering = [v for v in verify_quality.values() if node in plan.ancestors(v.id) and v.quality]
                    q = covering[0].quality if covering else node.quality
                    action["outcome"] = {Quality.VERIFIED: "helped", Quality.PARTIALLY_VERIFIED: "no_effect",
                                         Quality.FAILED: "failed", Quality.CONFLICTING: "failed"}.get(q, "unverified")
            outcomes = [a["outcome"] for a in decision.get("actions", []) if a.get("outcome")]
            if outcomes:
                decision["outcome"] = ", ".join(sorted(set(outcomes)))
            elif not decision.get("outcome"):
                decision["outcome"] = "no change made"

    def _finish(self, plan: Plan, status: PlanStatus, reason: str, *, by: str = "system") -> None:
        plan.finished_at = self.clock.now()
        self._set(plan, status, reason, by=by)
        if plan.priority in (PlanPriority.EMERGENCY, PlanPriority.CRITICAL):
            self._release_preempted(plan)
        for node in plan.nodes:
            for task_id in node.task_ids:
                self.permissions.revoke_for(task_id=task_id)

    # -- helpers ------------------------------------------------------------------------------------------------
    def _set(self, plan: Plan, status: PlanStatus, reason: str = "", *, by: str = "system",
             force: bool = False) -> None:
        if plan.status == status:
            if reason and reason != plan.status_reason:
                plan.status_reason = reason
            return
        if not force:
            check_plan_transition(plan.status, status)
        old = plan.status
        plan.status = status
        plan.status_reason = reason
        plan.history.append({"ts": self.clock.now(), "from": old.value, "to": status.value, "reason": reason, "by": by})
        etype = _STATUS_EVENTS.get(status)
        if etype == EventType.PLAN_STARTED and old != P.READY:
            etype = EventType.PLAN_RESUMED if old in (P.PAUSED, P.BLOCKED, P.WAITING) else None
        if status == P.REPLANNING:
            etype = None                  # the replanner reports what changed
        if etype is not None:
            payload = {"from": old.value, "to": status.value, "reason": reason, "by": by,
                       "quality": plan.quality.value if plan.quality else None}
            if status in (P.COMPLETED, P.FAILED):
                payload["result"] = _first_line(plan.result, 200)
                payload["more"] = len((plan.result or "").strip().splitlines()) > 1   # details on request
            severity = Severity.WARNING if status in (P.FAILED, P.BLOCKED) else Severity.INFO
            self._emit(etype, plan, payload, severity)

    def _node_event(self, plan: Plan, node: PlanNode) -> None:
        self._emit(EventType.PLAN_NODE_FINISHED, plan, {"node": node.id, "title": node.title, "status": node.status.value,
                                                        "quality": node.quality.value if node.quality else None,
                                                        "summary": node.summary[:200]})

    def _limit(self, plan: Plan, limit: Limit) -> None:
        self._limit_event(plan, limit)
        for node in plan.active_nodes():
            if node.task_id:
                self.tasks.pause_task(node.task_id, by="system:planner", source=InstructionSource.DEFAULT,
                                      reason=limit.reason)
        self._set(plan, P.BLOCKED, limit.sentence())

    def _limit_event(self, plan: Plan, limit: Limit) -> None:
        self._emit(EventType.LOOP_LIMIT_REACHED, plan, {"limit": limit.name, "reason": limit.reason}, Severity.WARNING)
        self._audit(plan, "loop_limit", limit.reason,
                    OperationalReason(limit.reason, "autonomy has hard limits", "stopped repeating", ""))

    def _emit(self, etype: EventType, plan: Plan, payload: dict[str, Any], severity: Severity = Severity.INFO) -> None:
        if self.bus is None:
            return
        body = {"plan_id": plan.id, "title": plan.title, "goal": plan.goal.objective[:200],
                "created_by": plan.created_by, "session_id": plan.session_id, **payload}
        self.bus.emit(Event(etype, "planner", body, severity=severity, entity_id=f"plan:{plan.id}"))

    def _audit(self, plan: Plan, action: str, summary: str, reason: OperationalReason) -> None:
        self.audit.record(actor="system:planner", action=action, summary=f"[{plan.title}] {summary}"[:500],
                          params={"plan_id": plan.id}, reason=reason)


def _condition_note(plan: Plan, condition: dict[str, Any] | None) -> str:
    """Why a conditional step didn't run, in words ("the tests didn't pass")."""
    refs = []

    def walk(c: dict[str, Any] | None) -> None:
        if not c:
            return
        if "node" in c:
            refs.append(c)
        for key in ("all", "any"):
            for sub in c.get(key, []):
                walk(sub)
    walk(condition)
    if refs:
        ref = refs[0]
        node = plan.node(ref["node"])
        title = (node.title[:1].lower() + node.title[1:]) if node else ref["node"]
        if ref.get("outcome", "success") == "success":
            return f"only if '{title}' succeeded, and it didn't"
        return f"only if '{title}' failed, and it didn't"
    return "not needed"


def action_identity(candidate: dict[str, Any]) -> str:
    """One concrete action (tool and exact arguments): two processes with the same name are two actions."""
    import json
    return f"{candidate.get('tool')}:{json.dumps(candidate.get('args') or {}, sort_keys=True)}"


def _first_line(text: str, limit: int = 160) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= limit else line[:limit - 1] + "…"
    return ""


def _milestone_status(plan: Plan) -> list[dict[str, Any]]:
    out = []
    for m in plan.milestones:
        nodes = [plan.node(n) for n in m.get("nodes", [])]
        nodes = [n for n in nodes if n is not None]
        done = all(n.finished for n in nodes) if nodes else False
        out.append({"title": m.get("title"), "done": done,
                    "failed": any(n.status == N.FAILED for n in nodes)})
    return out


def _running_reason(running: list[PlanNode]) -> str:
    if not running:
        return "running"
    names = [n.title for n in running[:3]]
    return "working on: " + "; ".join(names)


def _approver(who: str) -> str:
    who = who or "owner"
    if who.startswith(("user:", "system:")):
        return who
    return f"user:{who}" if who not in ("you",) else "user:owner"


__all__ = ["PlanEngine", "has_placeholders"]
