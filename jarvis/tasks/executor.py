"""Task executor: OBSERVE → PLAN → ACT → VERIFY → REPLAN (spec §17, §134, §183).

Runs one task inside a worker. Each step goes through the tool registry (so
authority, scope and audit always apply), is verified, and is checkpointed
before the next begins — a crash never loses completed work. Consequential
steps pause the task for approval instead of blocking a worker. Pause, cancel
and shutdown requests take effect promptly, cancelling in-flight tools.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from jarvis.audit.log import AuditLog
from jarvis.clock import Clock, SystemClock
from jarvis.core.types import OperationalReason, Outcome, Severity
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.log import get_logger
from jarvis.monitoring.watchers import MonitorChecker
from jarvis.permissions.manager import ApprovalManager, PermissionManager
from jarvis.permissions.model import Actor
from jarvis.planner.planner import Planner
from jarvis.tasks.manager import TaskController, TaskManager
from jarvis.tasks.models import Step, StepStatus, Task, TaskKind, TaskStatus
from jarvis.tasks.recovery import RecoveryAction, RecoveryPolicy
from jarvis.tools.base import ToolContext
from jarvis.tools.registry import Execution, ExecStatus, ToolRegistry
from jarvis.verification.engine import Verifier

log = get_logger("executor")


class _Interrupted(Exception):
    pass


class TaskExecutor:
    def __init__(self, manager: TaskManager, registry: ToolRegistry, approvals: ApprovalManager,
                 permissions: PermissionManager, audit: AuditLog, *, planner: Planner | None = None,
                 verifier: Verifier | None = None, monitor_checker: MonitorChecker | None = None,
                 bus: EventBus | None = None, clock: Clock | None = None, data_dir: str | None = None,
                 context_provider: Callable[[Task], str] | None = None) -> None:
        self.manager = manager
        self.registry = registry
        self.approvals = approvals
        self.permissions = permissions
        self.audit = audit
        self.planner = planner
        self.verifier = verifier or Verifier()
        self.monitor_checker = monitor_checker
        self.bus = bus
        self.clock = clock or SystemClock()
        self.data_dir = data_dir
        self.context_provider = context_provider or (lambda task: "")
        self.recovery = RecoveryPolicy()

    # -- entry point ---------------------------------------------------------------
    async def run(self, task: Task, controller: TaskController) -> Task:
        try:
            if task.kind == TaskKind.MONITOR:
                return await self._run_monitor(task, controller)
            return await self._run_plan(task, controller)
        except (_Interrupted, asyncio.CancelledError):
            return self._handle_interrupt(task, controller)

    def _actor(self, task: Task) -> Actor:
        delegated = task.created_by if task.created_by.startswith(("automation:", "agent:")) else None
        return Actor("task", task.id, task.owner, interactive=bool(task.authority.get("interactive", False)),
                     delegated_by=delegated, external=bool(task.authority.get("external", False)))

    def _ctx(self, task: Task, controller: TaskController, step: Step | None = None) -> ToolContext:
        return ToolContext(actor=self._actor(task), task_id=task.id, cwd=task.cwd, dry_run=task.dry_run,
                           cancel=controller.cancel_event, clock=self.clock, data_dir=self.data_dir,
                           step_id=step.id if step else None)

    def _check_control(self, controller: TaskController) -> None:
        if controller.cancel_event.is_set():
            raise _Interrupted()

    # -- plans ----------------------------------------------------------------------
    async def _run_plan(self, task: Task, controller: TaskController) -> Task:
        if not task.plan:
            self.manager.transition(task, TaskStatus.PLANNING, "planning")
            result = await self.planner.plan(task, self.context_provider(task)) if self.planner else None
            self._check_control(controller)
            if result is None or not result.steps:
                note = result.notes if result else "no planner is configured"
                if result and result.rejected:
                    note += f" (rejected: {'; '.join(result.rejected[:3])})"
                return self._finish(task, TaskStatus.FAILED, Outcome.FAILED, f"could not plan: {note}")
            task.plan = result.steps
            task.outputs["plan_source"] = result.source
            self.manager.checkpoint_task(task, f"planned {len(task.plan)} step(s) via {result.source}")
        self.manager.transition(task, TaskStatus.RUNNING, f"working on '{task.current_step.description}'"
                                if task.current_step else "running")
        while True:
            self._check_control(controller)
            if controller.apply_edits(task):
                self.manager.checkpoint_task(task, "plan modified")
            step = task.current_step
            if step is None:
                break
            if controller.pause_after_step:
                self.manager.checkpoint_task(task, "paused between steps")
                self.manager.transition(task, TaskStatus.PAUSED, controller.pause_after_step)
                return task
            if task.usage.get("steps", 0) >= int(task.budget.get("max_steps", 40)):
                return self._finish(task, TaskStatus.FAILED, Outcome.PARTIAL,
                                    f"step budget of {task.budget.get('max_steps')} exhausted")
            outcome = await self._run_step(task, step, controller)
            if outcome == "wait":
                return task
            if outcome == "stop":
                return task
        return await self._verify_and_finish(task, controller)

    async def _run_step(self, task: Task, step: Step, controller: TaskController) -> str:
        if step.tool is None:
            step.status = StepStatus.DONE
            self.manager.checkpoint_task(task)
            return "next"
        step.status = StepStatus.RUNNING
        step.attempts += 1
        step.started_at = self.clock.now()
        task.usage["steps"] = task.usage.get("steps", 0) + 1
        if task.status_reason != f"working on '{step.description}'":
            task.status_reason = f"working on '{step.description}'"
        self.manager.checkpoint_task(task, f"started {step.description}")
        self._progress(task, step)

        reason = OperationalReason(f"it is step {task.plan.index(step) + 1} of '{task.title}'",
                                   f"objective: {task.objective}", f"ran {step.tool}")
        execution = await self.registry.execute(step.tool, self._resolve_refs(task, step.args),
                                                self._ctx(task, controller, step), reason=reason)
        if execution.status == ExecStatus.CANCELLED or controller.cancel_event.is_set():
            step.status = StepStatus.PENDING
            step.interrupted = True
            raise _Interrupted()
        step.audit_id = execution.audit_id or step.audit_id
        step.finished_at = self.clock.now()
        step.result = execution.for_model() if execution.result or execution.message else None
        step.verification = execution.verification.to_dict() if execution.verification else None

        if execution.status == ExecStatus.DRY_RUN:
            step.status = StepStatus.DONE
            step.note = execution.message
            task.outputs.setdefault("dry_run", []).append(execution.preview)
            self.manager.checkpoint_task(task)
            return "next"
        if execution.status == ExecStatus.NEEDS_APPROVAL:
            assessment = execution.assessment
            approval = self.approvals.request(
                tool=step.tool, args=execution.args, summary=execution.preview,
                risk=assessment.risk if assessment else 0, level=assessment.level if assessment else 4,
                requested_by=f"task:{task.id}", task_id=task.id, step_id=step.id,
                reason=execution.decision.reason if execution.decision else "")
            step.status = StepStatus.WAITING_APPROVAL
            step.approval_id = approval.id
            step.attempts -= 1
            task.usage["steps"] -= 1
            self.manager.checkpoint_task(task, "awaiting approval")
            self.manager.transition(task, TaskStatus.WAITING, f"awaiting your approval to {execution.preview}")
            return "wait"
        if not execution.ok and execution.result is not None and execution.result.error == "model_unavailable":
            step.status = StepStatus.PENDING
            step.attempts -= 1
            task.usage["steps"] -= 1
            self._wait_for_model(task, f"waiting for a language model for '{step.description}'")
            return "wait"
        if execution.ok or (execution.status == ExecStatus.EXECUTED and step.allow_failure
                            and execution.verification is not None and execution.verification.passed is not False):
            step.status = StepStatus.DONE
            step.error = None if execution.ok else _error_text(execution)
            self._collect(task, step, execution)
            self.manager.checkpoint_task(task, f"completed {step.description}")
            self._progress(task, step)
            return "next"
        return await self._recover(task, step, execution, controller)

    async def _recover(self, task: Task, step: Step, execution: Execution, controller: TaskController) -> str:
        step.error = _error_text(execution)
        tool = self.registry.get(step.tool) if step.tool else None
        can_replan = self.planner is not None and self.planner.available and task.policy.on_step_failure == "replan"
        decision = self.recovery.decide(task, step, execution, idempotent=bool(tool and tool.spec.idempotent),
                                        can_replan=can_replan)
        task.errors.append({"ts": self.clock.now(), "step": step.description, "error": step.error,
                            "decision": decision.action.value})
        self.audit.record(actor="system:recovery", action=f"recovery_{decision.action.value}", task_id=task.id,
                          summary=decision.reason.sentence(), reason=decision.reason)
        if decision.action == RecoveryAction.RETRY:
            step.status = StepStatus.PENDING
            task.retry_count += 1
            self.manager.checkpoint_task(task, f"retrying {step.description}")
            try:
                await asyncio.wait_for(controller.cancel_event.wait(), timeout=decision.delay_s)
                raise _Interrupted()
            except asyncio.TimeoutError:
                return "next"
        if decision.action == RecoveryAction.CONTINUE:
            step.status = StepStatus.FAILED
            step.allow_failure = True
            self.manager.checkpoint_task(task)
            return "next"
        if decision.action == RecoveryAction.BLOCK:
            step.status = StepStatus.PENDING
            self.manager.checkpoint_task(task, "blocked")
            self.manager.transition(task, TaskStatus.BLOCKED, decision.reason.condition)
            return "stop"
        if decision.action == RecoveryAction.REPLAN and self.planner is not None:
            task.usage["replans"] = task.usage.get("replans", 0) + 1
            self.manager.transition(task, TaskStatus.PLANNING, f"replanning after '{step.description}' failed")
            result = await self.planner.replan(task, step, execution, self.context_provider(task))
            self._check_control(controller)
            if result.steps:
                step.status = StepStatus.FAILED
                step.allow_failure = True
                step.note = "superseded by a revised plan"
                keep = [s for s in task.plan if s.finished]
                task.plan = keep + result.steps
                self.manager.checkpoint_task(task, f"replanned: {len(result.steps)} new step(s)")
                self.manager.transition(task, TaskStatus.RUNNING, f"working on '{result.steps[0].description}'")
                return "next"
            step.status = StepStatus.FAILED
            self.manager.checkpoint_task(task)
            return self._fail(task, f"'{step.description}' failed ({step.error}) and no alternative plan was found: "
                                    f"{result.notes}")
        step.status = StepStatus.FAILED
        self.manager.checkpoint_task(task)
        return self._fail(task, f"'{step.description}' failed: {step.error}")

    def _fail(self, task: Task, reason: str) -> str:
        done = task.completed_steps()
        outcome = Outcome.PARTIAL if done else Outcome.FAILED
        task.outputs["summary"] = reason
        self._finish(task, TaskStatus.FAILED, outcome, reason)
        return "stop"

    async def _verify_and_finish(self, task: Task, controller: TaskController) -> Task:
        self.manager.transition(task, TaskStatus.VERIFYING, "verifying the result")
        report = await self.verifier.verify_task(task)
        self._check_control(controller)
        task.outputs["verification"] = report.to_dict()
        task.outputs["summary"] = report.summary
        if task.dry_run:
            planned = task.outputs.get("dry_run", [])
            task.outputs["summary"] = f"dry run: {len(planned)} action(s) planned, none executed"
            return self._finish(task, TaskStatus.COMPLETED, Outcome.UNKNOWN, task.outputs["summary"])
        if report.outcome in (Outcome.COMPLETE, Outcome.PARTIAL):
            tests = task.outputs.get("tests")
            if tests and not tests.get("ok") and self.bus:
                self.bus.emit(Event(EventType.TEST_FAILED, "executor", {"title": task.title,
                                                                        "summary": tests.get("summary"),
                                                                        "failures": tests.get("failures", [])[:10]},
                                    severity=Severity.WARNING, task_id=task.id))
            return self._finish(task, TaskStatus.COMPLETED, report.outcome, report.summary)
        if report.outcome == Outcome.UNKNOWN:
            return self._finish(task, TaskStatus.COMPLETED, Outcome.UNKNOWN, f"finished, but {report.summary}")
        return self._finish(task, TaskStatus.FAILED, report.outcome, report.summary)

    def _wait_for_model(self, task: Task, reason: str) -> Task:
        """No model is reachable: wait (resumed automatically when one comes back) instead of failing."""
        self.manager.checkpoint_task(task, "waiting for a language model")
        task.checkpoint["waiting_for"] = "model"        # (checkpoint_task rebuilds the checkpoint)
        self.manager.transition(task, TaskStatus.WAITING, reason)
        return task

    @staticmethod
    def _resolve_refs(task: Task, args: dict[str, Any]) -> dict[str, Any]:
        """Steps may take an earlier step's output as an argument: ``{"$from_step": 0}`` (the step's index)."""
        def resolve(value: Any) -> Any:
            if isinstance(value, dict) and set(value) == {"$from_step"}:
                index = int(value["$from_step"])
                if 0 <= index < len(task.plan) and task.plan[index].result:
                    result = task.plan[index].result or {}
                    return result.get("data") if result.get("data") is not None else result.get("summary")
                return None
            return value
        return {k: resolve(v) for k, v in args.items()}

    def _collect(self, task: Task, step: Step, execution: Execution) -> None:
        """Record what a completed step produced: files it wrote and reports it generated."""
        data = execution.result.data if execution.result else None
        if step.tool == "file_write" and execution.args.get("path"):
            task.artifacts.append({"type": "file", "path": execution.args["path"], "step": step.description})
        if isinstance(data, dict):
            for artifact in data.get("artifacts") or []:
                if isinstance(artifact, dict):
                    task.artifacts.append({**artifact, "step": step.description})
            if isinstance(data.get("report"), str) and data["report"].strip():
                task.result = data["report"].strip()
                task.outputs["result_preview"] = _preview(task.result)

    @staticmethod
    def _derive_result(task: Task) -> str:
        """The final result in words when no step produced an explicit report: the last step's output."""
        for step in reversed(task.plan):
            if step.status != StepStatus.DONE or not step.result:
                continue
            data = step.result.get("data")
            if isinstance(data, dict) and str(data.get("stdout") or "").strip():
                lines = [ln for ln in str(data["stdout"]).strip().splitlines()]
                tail = "\n".join(lines[-20:])
                # a command's summary is usually its last line (ping statistics, "5 passed", "Done")
                last = next((ln.strip() for ln in reversed(lines) if ln.strip()), "")
                task.outputs["result_preview"] = _preview(last)
                return tail if len(tail) <= 2000 else "…" + tail[-2000:]
            if step.result.get("summary"):
                return str(step.result["summary"])
        return ""

    def _finish(self, task: Task, status: TaskStatus, outcome: Outcome, reason: str) -> Task:
        task.outputs.setdefault("summary", reason)
        if status == TaskStatus.COMPLETED and not task.result and task.kind != TaskKind.MONITOR:
            task.result = self._derive_result(task) or task.outputs.get("summary", "")
        self.manager.checkpoint_task(task, reason)
        self.manager.transition(task, status, reason, outcome=outcome)
        self.permissions.revoke_for(task_id=task.id)
        return task

    def _progress(self, task: Task, step: Step) -> None:
        if self.bus:
            self.bus.emit(Event(EventType.TASK_PROGRESS, "executor",
                                {"title": task.title, "progress": task.compute_progress(), "step": step.description,
                                 "step_status": step.status.value}, task_id=task.id))

    # -- monitors ---------------------------------------------------------------------
    async def _run_monitor(self, task: Task, controller: TaskController) -> Task:
        spec = task.monitor
        if spec is None or self.monitor_checker is None:
            return self._finish(task, TaskStatus.FAILED, Outcome.FAILED, "monitor has no target or checker")
        self.manager.transition(task, TaskStatus.RUNNING, f"monitoring: {task.title}")
        while True:
            self._check_control(controller)
            now = self.clock.now()
            if spec.expires_at is not None and now >= spec.expires_at:
                return self._finish_monitor(task, "expired", "monitoring period ended")
            obs = await self.monitor_checker.check(spec, self._ctx(task, controller))
            spec.last_check = self.clock.now()
            spec.last_observation = {"state": obs.state, "detail": obs.detail, "ts": spec.last_check}
            task.monitor = spec
            if obs.triggered:
                spec.triggers += 1
                if self.bus:
                    self.bus.emit(Event(EventType.MONITOR_TRIGGERED, "monitor",
                                        {"title": task.title, "detail": obs.detail, "notify": spec.notify,
                                         "resolved": obs.resolved, "triggers": spec.triggers, **obs.data},
                                        severity=obs.severity, task_id=task.id))
            task.status_reason = f"monitoring: {obs.detail}"
            self.manager.save(task)
            if obs.resolved and ("target_resolved" in spec.stop_when or "condition_resolved" in spec.stop_when):
                return self._finish_monitor(task, "resolved", obs.detail)
            if obs.triggered and "triggered_once" in spec.stop_when:
                return self._finish_monitor(task, "triggered", obs.detail)
            try:
                await asyncio.wait_for(controller.cancel_event.wait(), timeout=spec.interval_s)
                raise _Interrupted()
            except asyncio.TimeoutError:
                continue

    def _finish_monitor(self, task: Task, why: str, detail: str) -> Task:
        self.manager.transition(task, TaskStatus.VERIFYING, f"monitor stop condition met ({why})")
        task.outputs["summary"] = detail
        task.outputs["stop_reason"] = why
        return self._finish(task, TaskStatus.COMPLETED, Outcome.COMPLETE, f"stopped monitoring ({why}): {detail}")

    # -- interruption -------------------------------------------------------------------
    def _handle_interrupt(self, task: Task, controller: TaskController) -> Task:
        fresh = self.manager.get_task(task.id) or task
        # keep the in-memory plan state (it is newer than what is persisted for the running step)
        fresh.plan = task.plan
        fresh.usage = task.usage
        fresh.outputs = task.outputs
        if task.monitor:
            fresh.monitor = task.monitor
        controller.apply_edits(fresh)
        for step in fresh.plan:
            if step.status == StepStatus.RUNNING:
                step.status = StepStatus.PENDING
                step.interrupted = True
        intent = controller.intent or "shutdown"
        if intent == "cancel":
            for step in fresh.plan:
                if not step.finished:
                    step.status = StepStatus.SKIPPED
                    step.note = "cancelled"
            fresh.outputs["summary"] = controller.reason or "cancelled"
            self.manager.checkpoint_task(fresh, "cancelled")
            self.manager.transition(fresh, TaskStatus.CANCELLED, controller.reason or "cancelled",
                                    by="user", outcome=Outcome.PARTIAL if fresh.completed_steps() else None)
            self.permissions.revoke_for(task_id=fresh.id)
            self.approvals.cancel_for_task(fresh.id)
            for callback in self.manager.on_cancel:
                callback(fresh)
        elif intent == "pause":
            self.manager.checkpoint_task(fresh, "paused")
            self.manager.transition(fresh, TaskStatus.PAUSED, controller.reason or "paused")
        else:
            self.manager.checkpoint_task(fresh, "interrupted by shutdown")
            self.manager.mark_interrupted(fresh)
        return fresh


def _preview(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit - 1].rstrip() + "…"


def _error_text(execution: Execution) -> str:
    """A human-readable error: machine codes like 'not_found' are replaced by the tool's own summary."""
    result = execution.result
    if result is None:
        return execution.message or "unknown error"
    error = result.error or ""
    if not error or (" " not in error and len(error) < 30):
        return result.summary or error or execution.message
    return error


def summarize_step_result(step: Step) -> dict[str, Any]:
    return {"description": step.description, "status": step.status.value, "result": step.result}
