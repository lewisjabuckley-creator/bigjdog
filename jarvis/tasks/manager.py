"""Durable task manager (spec §18-20, §57-59, §102, §155).

Tasks live in SQLite, not in process memory: they survive restarts, and every
transition is validated against the state machine, recorded in the task's
history, and published as an event. Control commands (pause, resume, cancel)
respect the instruction hierarchy, so a stale automation cannot undo an
explicit user command.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from jarvis.clock import Clock, SystemClock
from jarvis.core.types import Outcome, Priority, Severity
from jarvis.database.db import Database, dumps, loads
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.permissions.hierarchy import Directive, InstructionSource, may_override
from jarvis.tasks.models import (EXECUTING, OPEN, TERMINAL, InvalidTransition, MonitorSpec, Step, StepStatus, Task,
                                 TaskKind, TaskPolicy, TaskStatus, check_transition)
from jarvis.world.model import WorldModel

_STATUS_EVENTS = {
    TaskStatus.RUNNING: EventType.TASK_STARTED,
    TaskStatus.WAITING: EventType.TASK_WAITING,
    TaskStatus.BLOCKED: EventType.TASK_BLOCKED,
    TaskStatus.PAUSED: EventType.TASK_PAUSED,
    TaskStatus.COMPLETED: EventType.TASK_COMPLETED,
    TaskStatus.FAILED: EventType.TASK_FAILED,
    TaskStatus.CANCELLED: EventType.TASK_CANCELLED,
}
_SEVERITY = {TaskStatus.FAILED: Severity.ERROR, TaskStatus.BLOCKED: Severity.WARNING,
             TaskStatus.WAITING: Severity.INFO}


@dataclass
class TaskController:
    """Live handle a worker holds for a running task (not persisted).

    Plan edits for a running task are queued here and applied by the worker at the next step
    boundary, so the worker's in-memory copy of the plan never silently discards them.
    """

    task_id: str
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    intent: str | None = None           # pause | cancel | shutdown — why the cancel event was set
    reason: str = ""
    runner: asyncio.Task[Any] | None = None
    edits: list[Callable[[Task], None]] = field(default_factory=list)
    pause_after_step: str | None = None  # pause at the next step boundary (the running step finishes first)

    def request(self, intent: str, reason: str = "") -> None:
        self.intent = intent
        self.reason = reason
        self.cancel_event.set()

    def apply_edits(self, task: Task) -> bool:
        applied = bool(self.edits)
        while self.edits:
            self.edits.pop(0)(task)
        return applied


@dataclass
class ControlResult:
    ok: bool
    message: str
    task: Task | None = None


@dataclass
class RecoveryReport:
    task_id: str
    title: str
    summary: str
    resumed: bool
    decision: str = ""              # resumed | paused | blocked
    outcome_unknown: bool = False   # the step in flight may or may not have taken effect


class TaskManager:
    def __init__(self, db: Database, *, bus: EventBus | None = None, clock: Clock | None = None,
                 world: WorldModel | None = None) -> None:
        self.db = db
        self.bus = bus
        self.clock = clock or SystemClock()
        self.world = world
        self.controllers: dict[str, TaskController] = {}
        self._wakeup: Callable[[], None] | None = None
        self.on_cancel: list[Callable[[Task], None]] = []

    def set_wakeup(self, callback: Callable[[], None]) -> None:
        self._wakeup = callback

    def _wake(self) -> None:
        if self._wakeup:
            self._wakeup()

    # -- creation -----------------------------------------------------------------
    def create_task(self, objective: str, *, title: str = "", kind: TaskKind = TaskKind.ONESHOT,
                    steps: Iterable[Step] | None = None, priority: Priority = Priority.P1, owner: str = "owner",
                    created_by: str = "user", project_id: str | None = None, parent_id: str | None = None,
                    dependencies: list[str] | None = None, resources: list[str] | None = None,
                    success_condition: dict[str, Any] | None = None, policy: TaskPolicy | None = None,
                    monitor: MonitorSpec | None = None, deadline: float | None = None, cwd: str | None = None,
                    dry_run: bool = False, authority: dict[str, Any] | None = None,
                    outputs: dict[str, Any] | None = None, budget: dict[str, Any] | None = None,
                    request: str = "", origin: str = "", idempotency_key: str | None = None) -> Task:
        """Create a durable task. With an ``idempotency_key`` a repeated request (a client retrying after a
        dropped connection, a scheduler slot fired twice) returns the task that already exists instead of
        creating a second one."""
        if idempotency_key:
            existing = self.find_by_key(idempotency_key)
            if existing is not None:
                return existing
        now = self.clock.now()
        interactive = created_by == "user" or created_by.startswith("user:")
        task = Task(objective=objective, title=title, kind=kind, owner=owner, created_by=created_by,
                    priority=Priority(priority), project_id=project_id, parent_id=parent_id,
                    plan=list(steps or []), dependencies=list(dependencies or []), resources=list(resources or []),
                    success_condition=success_condition, policy=policy or TaskPolicy(), monitor=monitor,
                    deadline=deadline, cwd=cwd, dry_run=dry_run,
                    authority=authority or {"interactive": interactive}, outputs=dict(outputs or {}),
                    request=request, origin=origin or ("conversation" if interactive else created_by),
                    idempotency_key=idempotency_key, created_at=now, updated_at=now)
        if budget:
            task.budget.update(budget)
        task.history.append({"ts": now, "from": None, "to": task.status.value, "reason": "created", "by": created_by})
        try:
            self.db.execute(
                "INSERT INTO tasks(id, kind, title, objective, owner, created_by, priority, status, status_reason, "
                "outcome, progress, project_id, parent_id, data, deadline, created_at, updated_at, started_at, "
                "finished_at, version, idempotency_key) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)",
                (task.id, task.kind.value, task.title, task.objective, task.owner, task.created_by, int(task.priority),
                 task.status.value, task.status_reason, None, 0.0, task.project_id, task.parent_id,
                 dumps(task.data_blob()), task.deadline, now, now, None, None, idempotency_key),
            )
        except sqlite3.IntegrityError:
            existing = self.find_by_key(idempotency_key) if idempotency_key else None
            if existing is None:
                raise
            return existing
        if self.world is not None:
            self.world.upsert_entity("task", task.title, {"status": task.status.value, "kind": task.kind.value},
                                     id=f"task:{task.id}", source="tasks")
            if project_id:
                self.world.relate(f"project:{project_id}", "contains", f"task:{task.id}")
        self._emit(EventType.TASK_CREATED, task, {"title": task.title, "kind": task.kind.value,
                                                  "priority": task.priority.name})
        self._wake()
        return task

    # -- reads --------------------------------------------------------------------
    def get_task(self, task_id: str) -> Task | None:
        row = self.db.query_one("SELECT * FROM tasks WHERE id=?", (task_id,))
        return Task.from_row(row, loads(row["data"], {})) if row else None

    def list_tasks(self, statuses: Iterable[TaskStatus] | None = None, *, kinds: Iterable[TaskKind] | None = None,
                   project_id: str | None = None, since: float | None = None, limit: int = 100,
                   order: str = "priority") -> list[Task]:
        clauses, params = [], []
        if statuses is not None:
            statuses = list(statuses)
            clauses.append(f"status IN ({','.join('?' * len(statuses))})")
            params += [s.value for s in statuses]
        if kinds is not None:
            kinds = list(kinds)
            clauses.append(f"kind IN ({','.join('?' * len(kinds))})")
            params += [k.value for k in kinds]
        if project_id:
            clauses.append("project_id=?")
            params.append(project_id)
        if since is not None:
            clauses.append("updated_at >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_by = {"priority": "priority ASC, created_at ASC", "recent": "updated_at DESC",
                    "created": "created_at ASC"}[order]
        rows = self.db.query(f"SELECT * FROM tasks {where} ORDER BY {order_by} LIMIT ?", (*params, limit))
        return [Task.from_row(r, loads(r["data"], {})) for r in rows]

    def find_by_key(self, idempotency_key: str) -> Task | None:
        row = self.db.query_one("SELECT * FROM tasks WHERE idempotency_key=?", (idempotency_key,))
        return Task.from_row(row, loads(row["data"], {})) if row else None

    def open_tasks(self) -> list[Task]:
        return self.list_tasks(OPEN)

    def most_recent(self, *, statuses: Iterable[TaskStatus] | None = None,
                    kinds: Iterable[TaskKind] | None = None) -> Task | None:
        tasks = self.list_tasks(statuses, kinds=kinds, order="recent", limit=1)
        return tasks[0] if tasks else None

    def find(self, text: str, statuses: Iterable[TaskStatus] | None = None) -> list[Task]:
        needle = text.lower().strip()
        return [t for t in self.list_tasks(statuses, order="recent", limit=200)
                if needle and (needle in t.title.lower() or needle in t.objective.lower() or t.id == text)]

    # -- persistence ----------------------------------------------------------------
    def save(self, task: Task) -> Task:
        if not task.control_dirty:
            row = self.db.query_one("SELECT priority, deadline, data FROM tasks WHERE id=?", (task.id,))
            if row is not None:
                task.priority = Priority(row["priority"])
                task.deadline = row["deadline"]
                task.control = loads(row["data"], {}).get("control")
        task.control_dirty = False
        task.updated_at = self.clock.now()
        task.version += 1
        self.db.execute(
            "UPDATE tasks SET title=?, priority=?, status=?, status_reason=?, outcome=?, progress=?, data=?, deadline=?, "
            "updated_at=?, started_at=?, finished_at=?, version=? WHERE id=?",
            (task.title, int(task.priority), task.status.value, task.status_reason,
             task.outcome.value if task.outcome else None, task.progress, dumps(task.data_blob()), task.deadline,
             task.updated_at, task.started_at, task.finished_at, task.version, task.id),
        )
        return task

    def checkpoint_task(self, task: Task, note: str = "") -> Task:
        current = task.current_step
        task.checkpoint = {
            "at": self.clock.now(),
            "completed_steps": [s.id for s in task.completed_steps()],
            "current_step": current.id if current else None,
            "current_description": current.description if current else None,
            "outputs_keys": sorted(task.outputs),
            "note": note,
        }
        task.progress = task.compute_progress()
        return self.save(task)

    def transition(self, task: Task, status: TaskStatus, reason: str = "", *, by: str = "system",
                   outcome: Outcome | None = None, emit: bool = True) -> Task:
        if task.status == status:
            if reason and reason != task.status_reason:
                task.status_reason = reason
                self.save(task)
            return task
        check_transition(task.status, status)
        old = task.status
        now = self.clock.now()
        task.status = status
        task.status_reason = reason
        if outcome is not None:
            task.outcome = outcome
        if status == TaskStatus.RUNNING and task.started_at is None:
            task.started_at = now
        if status in TERMINAL:
            task.finished_at = now
            task.progress = 1.0 if status == TaskStatus.COMPLETED else task.compute_progress()
        if old in TERMINAL and status not in TERMINAL:
            task.finished_at = None
        task.history.append({"ts": now, "from": old.value, "to": status.value, "reason": reason, "by": by})
        self.save(task)
        if self.world is not None:
            self.world.upsert_entity("task", task.title, {"status": status.value}, id=f"task:{task.id}")
        if emit:
            etype = _STATUS_EVENTS.get(status, EventType.TASK_STATUS_CHANGED)
            if old == TaskStatus.PAUSED and status == TaskStatus.QUEUED:
                etype = EventType.TASK_RESUMED
            payload = {"title": task.title, "from": old.value, "to": status.value, "reason": reason,
                       "outcome": task.outcome.value if task.outcome else None, "by": by}
            if status == TaskStatus.COMPLETED and task.result:
                preview = task.outputs.get("result_preview") or " ".join(task.result.split())
                payload["result"] = preview if len(preview) <= 200 else preview[:199].rstrip() + "…"
            self._emit(etype, task, payload, _SEVERITY.get(status, Severity.INFO))
        if status in (TaskStatus.QUEUED,):
            self._wake()
        return task

    # -- control ----------------------------------------------------------------------
    def _directive(self, action: str, by: str, source: InstructionSource) -> Directive:
        return Directive(source, action, self.clock.now(), by)

    def _governing(self, task: Task) -> Directive | None:
        return Directive.from_dict(task.control) if task.control else None

    def pause_task(self, task_id: str, *, by: str = "user", source: InstructionSource = InstructionSource.USER,
                   reason: str = "", at_step_boundary: bool = False) -> ControlResult:
        task = self.get_task(task_id)
        if task is None:
            return ControlResult(False, "no such task")
        if task.terminal:
            return ControlResult(False, f"{task.title} already {task.status.value}", task)
        directive = self._directive("pause", by, source)
        if not may_override(directive, self._governing(task)):
            return ControlResult(False, f"{task.title} is governed by a higher-priority instruction", task)
        task.control = directive.to_dict()
        task.control_dirty = True
        reason = reason or f"paused by {by}"
        controller = self.controllers.get(task_id)
        if controller and task.status in EXECUTING:
            self.save(task)
            if at_step_boundary:
                controller.pause_after_step = reason   # the step in flight is not interrupted
                return ControlResult(True, f"pausing {task.title} after its current step", task)
            controller.request("pause", reason)   # the worker checkpoints and transitions
            return ControlResult(True, f"pausing {task.title}", task)
        self.transition(task, TaskStatus.PAUSED, reason, by=by)
        return ControlResult(True, f"paused {task.title}", task)

    def resume_task(self, task_id: str, *, by: str = "user", source: InstructionSource = InstructionSource.USER,
                    reason: str = "") -> ControlResult:
        task = self.get_task(task_id)
        if task is None:
            return ControlResult(False, "no such task")
        if task.status == TaskStatus.FAILED and source == InstructionSource.USER:
            return self.retry_task(task_id, by=by)
        if task.status not in (TaskStatus.PAUSED, TaskStatus.BLOCKED, TaskStatus.WAITING):
            return ControlResult(False, f"{task.title} is {task.status.value}, not paused", task)
        directive = self._directive("resume", by, source)
        if not may_override(directive, self._governing(task)):
            governing = self._governing(task)
            return ControlResult(False, f"{task.title} was {governing.action}d by {governing.issued_by}; "
                                        f"only an equal or higher authority can resume it", task)
        task.control = directive.to_dict()
        task.control_dirty = True
        for step in task.plan:
            if step.status == StepStatus.RUNNING:
                step.status = StepStatus.PENDING
            step.interrupted = False
            if step.outcome_unknown and not step.finished:
                # an explicit decision to run a step whose earlier attempt has an unknown outcome
                step.outcome_unknown = False
                step.note = f"re-run at the request of {by} after an unknown outcome"
        self.transition(task, TaskStatus.QUEUED, reason or f"resumed by {by}", by=by)
        return ControlResult(True, f"resumed {task.title}", task)

    def cancel_task(self, task_id: str, *, by: str = "user", source: InstructionSource = InstructionSource.USER,
                    reason: str = "") -> ControlResult:
        task = self.get_task(task_id)
        if task is None:
            return ControlResult(False, "no such task")
        if task.terminal:
            return ControlResult(False, f"{task.title} already {task.status.value}", task)
        directive = self._directive("cancel", by, source)
        if not may_override(directive, self._governing(task)):
            return ControlResult(False, f"{task.title} is governed by a higher-priority instruction", task)
        task.control = directive.to_dict()
        task.control_dirty = True
        reason = reason or f"cancelled by {by}"
        controller = self.controllers.get(task_id)
        if controller and task.status in EXECUTING:
            self.save(task)
            controller.request("cancel", reason)
            return ControlResult(True, f"cancelling {task.title}", task)
        for step in task.plan:
            if not step.finished:
                step.status = StepStatus.SKIPPED
                step.note = "cancelled"
        self.transition(task, TaskStatus.CANCELLED, reason, by=by)
        for callback in self.on_cancel:
            callback(task)
        return ControlResult(True, f"cancelled {task.title}", task)

    def retry_task(self, task_id: str, *, by: str = "user") -> ControlResult:
        task = self.get_task(task_id)
        if task is None or task.status != TaskStatus.FAILED:
            return ControlResult(False, "only failed tasks can be retried", task)
        for step in task.plan:
            if step.status in (StepStatus.FAILED, StepStatus.RUNNING, StepStatus.WAITING_APPROVAL):
                step.status = StepStatus.PENDING
                step.attempts = 0
                step.error = None
        task.outcome = None
        task.retry_count += 1
        task.control = self._directive("resume", by, InstructionSource.USER).to_dict()
        task.control_dirty = True
        self.transition(task, TaskStatus.QUEUED, f"retry requested by {by}", by=by)
        return ControlResult(True, f"retrying {task.title} from the last checkpoint", task)

    def reprioritize(self, task_id: str, priority: Priority, *, by: str = "user") -> ControlResult:
        task = self.get_task(task_id)
        if task is None:
            return ControlResult(False, "no such task")
        task.priority = Priority(priority)
        task.control_dirty = True
        task.history.append({"ts": self.clock.now(), "from": task.status.value, "to": task.status.value,
                             "reason": f"priority set to {task.priority.name}", "by": by})
        self.save(task)
        self._wake()
        return ControlResult(True, f"{task.title} is now {task.priority.name}", task)

    def skip_steps(self, task_id: str, predicate: Callable[[Step], bool], *, by: str = "user",
                   reason: str = "") -> ControlResult:
        """Modify a plan mid-flight without discarding completed work (spec §58)."""
        task = self.get_task(task_id)
        if task is None:
            return ControlResult(False, "no such task")
        note = reason or f"removed by {by}"

        def edit(t: Task) -> list[str]:
            names = []
            for step in t.plan:
                if not step.finished and step.status != StepStatus.RUNNING and predicate(step):
                    step.status = StepStatus.SKIPPED
                    step.note = note
                    names.append(step.description)
            if names:
                t.history.append({"ts": self.clock.now(), "from": t.status.value, "to": t.status.value,
                                  "reason": f"plan modified: skipped {', '.join(names)}", "by": by})
            return names

        preview = [s.description for s in task.plan
                   if not s.finished and s.status != StepStatus.RUNNING and predicate(s)]
        if not preview:
            return ControlResult(False, "no pending steps matched", task)
        controller = self.controllers.get(task_id)
        if controller is not None and task.status in EXECUTING:
            controller.edits.append(edit)
        else:
            edit(task)
            self.save(task)
        return ControlResult(True, f"removed from the plan: {', '.join(preview)}", task)

    def add_steps(self, task_id: str, steps: list[Step], *, by: str = "user") -> ControlResult:
        task = self.get_task(task_id)
        if task is None or task.terminal:
            return ControlResult(False, "task is not open", task)
        controller = self.controllers.get(task_id)
        if controller is not None and task.status in EXECUTING:
            controller.edits.append(lambda t: t.plan.extend(steps))
        else:
            task.plan.extend(steps)
            self.save(task)
        return ControlResult(True, f"added {len(steps)} step(s) to {task.title}", task)

    # -- recovery -----------------------------------------------------------------------
    def recover_interrupted(self, idempotent: Callable[[str | None], bool], *,
                            safe_to_repeat: Callable[[Step], bool] | None = None, max_age_s: float | None = None,
                            automation_state: Callable[[str], bool | None] | None = None,
                            audit: Any = None) -> list[RecoveryReport]:
        """After a restart, reconcile tasks that were executing when the process stopped (spec §155).

        For each interrupted task: restore its checkpoint, validate it (age, working folder, dependencies, the
        automation that created it), decide whether the step that was in flight is safe to repeat, then resume
        it, pause it for the user, or block it, and record the decision (task history, audit, events).

        A step that was running when the process stopped has an unknown outcome: it may or may not have taken
        effect. It is repeated automatically only if doing so is safe (idempotent or read-only); otherwise the
        task waits for the user. Completed steps are never re-run. Monitors resume (they only observe).
        """
        repeatable = safe_to_repeat or (lambda step: idempotent(step.tool))
        now = self.clock.now()
        reports = []
        candidates = self.list_tasks(list(EXECUTING) + [TaskStatus.PAUSED])
        for task in candidates:
            if task.status == TaskStatus.PAUSED and not task.checkpoint.get("interrupted"):
                continue
            crashed = task.status in EXECUTING          # no shutdown checkpoint: the process died
            in_flight = next((s for s in task.plan if s.status == StepStatus.RUNNING), None) or \
                next((s for s in task.plan if s.interrupted and not s.finished), None)
            where_step = in_flight or task.current_step
            done = [s.description for s in task.completed_steps()]
            pending = [s.description for s in task.pending_steps()]
            for step in task.plan:
                if step.status == StepStatus.RUNNING:
                    step.status = StepStatus.PENDING
                    step.interrupted = True
            where = f" during '{where_step.description}'" if where_step else ""
            summary = f"{task.title} was interrupted{where}."
            if done:
                summary += f" Completed: {', '.join(done)}."
            if pending:
                summary += f" Pending: {', '.join(pending)}."

            # -- validate ------------------------------------------------------------------
            problem: str | None = None
            hold: str | None = None
            if task.cwd and not os.path.isdir(task.cwd):
                problem = f"its working folder {task.cwd} no longer exists"
            for dep_id in task.dependencies:
                dep = self.get_task(dep_id)
                if dep is None or dep.status in (TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.ABANDONED):
                    problem = problem or f"its dependency '{dep.title if dep else dep_id}' " \
                                         f"{'is missing' if dep is None else dep.status.value}"
            if task.created_by.startswith("automation:") and automation_state is not None:
                enabled = automation_state(task.created_by.split(":", 1)[1])
                if not enabled:
                    hold = "the automation that created it is " + ("disabled" if enabled is False else "gone")
            age = now - (task.updated_at or now)
            if max_age_s is not None and age > max_age_s and task.kind != TaskKind.MONITOR:
                hold = hold or f"it was interrupted {int(age // 3600)} hours ago"

            # -- decide ----------------------------------------------------------------------
            unknown = in_flight is not None and in_flight.tool is not None and not repeatable(in_flight)
            if unknown:
                in_flight.outcome_unknown = True
                in_flight.note = "outcome unknown: it was running when JARVIS stopped"
            policy = task.policy.resume_after_restart
            if task.kind == TaskKind.MONITOR:
                decision, why = ("blocked", problem) if problem else ("resumed", "monitors only observe")
            elif problem:
                decision, why = "blocked", problem
            elif unknown:
                decision = "paused"
                why = (f"'{in_flight.description}' was running when JARVIS stopped, so I can't tell whether it "
                       "finished, and repeating it could apply it twice")
            elif hold:
                decision, why = "paused", hold
            elif policy == "ask":
                decision, why = "paused", "its policy is to ask before resuming after a restart"
            else:
                decision = "resumed"
                why = f"'{in_flight.description}' is safe to repeat" if in_flight else \
                    "no step was in flight when JARVIS stopped"
            if decision == "resumed" and task.kind != TaskKind.MONITOR:
                summary += f" I resumed it because {why}."
            elif decision == "paused":
                summary += f" I haven't resumed it: {why}. Say 'continue' to resume."
            elif decision == "blocked":
                summary += f" It is blocked: {why}."
            task.recovery = summary
            task.checkpoint["interrupted"] = False
            task.checkpoint["recovered_at"] = now
            task.checkpoint["recovery_decision"] = decision
            if task.status in EXECUTING:
                # force through the state machine: EXECUTING -> PAUSED
                self.transition(task, TaskStatus.PAUSED, "interrupted by restart" if crashed
                                else "interrupted by shutdown", by="system")
            else:
                self.save(task)
            self._emit(EventType.TASK_INTERRUPTED, task, {"title": task.title, "summary": summary}, Severity.WARNING)
            if decision == "resumed":
                task.control = None
                task.control_dirty = True
                self.transition(task, TaskStatus.QUEUED, "resumed after restart", by="system")
            elif decision == "blocked":
                self.transition(task, TaskStatus.BLOCKED, why or "blocked after restart", by="system")
            else:
                task.status_reason = f"interrupted by restart: {why}"
                self.save(task)
            self._emit(EventType.TASK_RECOVERED, task,
                       {"title": task.title, "decision": decision, "reason": why, "crashed": crashed,
                        "step": in_flight.description if in_flight else None, "outcome_unknown": unknown},
                       Severity.WARNING if decision != "resumed" else Severity.INFO)
            if audit is not None:
                audit.record(actor="system:recovery", action="recovery_decision", task_id=task.id,
                             summary=summary, outcome="unknown" if unknown else None,
                             reason={"condition": f"'{task.title}' was interrupted"
                                                  f"{' by a crash' if crashed else ' by a shutdown'}{where}",
                                     "rule": "never repeat a step with an unknown outcome unless it is safe to "
                                             "repeat; validate before resuming",
                                     "action": {"resumed": "resumed the task", "paused": "paused it for you",
                                                "blocked": "blocked it"}[decision],
                                     "expected": why or ""})
            reports.append(RecoveryReport(task.id, task.title, summary, decision == "resumed", decision, unknown))
        return reports

    def mark_interrupted(self, task: Task, reason: str = "interrupted: shutdown") -> None:
        for step in task.plan:
            if step.status == StepStatus.RUNNING:
                step.status = StepStatus.PENDING
                step.interrupted = True
        task.checkpoint["interrupted"] = True
        try:
            self.transition(task, TaskStatus.PAUSED, reason, by="system")
        except InvalidTransition:
            self.save(task)

    # -- events -------------------------------------------------------------------------
    def _emit(self, etype: EventType, task: Task, payload: dict[str, Any], severity: Severity = Severity.INFO) -> None:
        if self.bus is not None:
            payload = {**payload, "task_kind": task.kind.value, "priority": task.priority.name,
                       "created_by": task.created_by}
            if task.outputs.get("plan_id"):
                payload["plan_id"] = task.outputs["plan_id"]
                payload["plan_node"] = task.outputs.get("plan_node")
            self.bus.emit(Event(etype, "tasks", payload, severity=severity, task_id=task.id,
                                entity_id=f"task:{task.id}"))

