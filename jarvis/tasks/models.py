"""Task model and its explicit state machine (spec §18, §20, §140)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from jarvis.core.types import Outcome, Priority, new_id


class TaskStatus(StrEnum):
    QUEUED = "queued"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING = "waiting"
    BLOCKED = "blocked"
    PAUSED = "paused"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"


S = TaskStatus
TERMINAL = frozenset({S.COMPLETED, S.FAILED, S.CANCELLED, S.ABANDONED})
EXECUTING = frozenset({S.PLANNING, S.RUNNING, S.VERIFYING})   # occupying a worker
OPEN = frozenset(set(S) - TERMINAL)

TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    S.QUEUED: frozenset({S.PLANNING, S.RUNNING, S.BLOCKED, S.PAUSED, S.CANCELLED, S.ABANDONED, S.WAITING}),
    S.PLANNING: frozenset({S.RUNNING, S.WAITING, S.BLOCKED, S.PAUSED, S.FAILED, S.CANCELLED, S.QUEUED}),
    S.RUNNING: frozenset({S.VERIFYING, S.WAITING, S.BLOCKED, S.PAUSED, S.FAILED, S.CANCELLED, S.PLANNING,
                          S.QUEUED}),
    S.WAITING: frozenset({S.QUEUED, S.RUNNING, S.BLOCKED, S.PAUSED, S.FAILED, S.CANCELLED}),
    S.BLOCKED: frozenset({S.QUEUED, S.RUNNING, S.WAITING, S.PAUSED, S.FAILED, S.CANCELLED, S.ABANDONED}),
    S.PAUSED: frozenset({S.QUEUED, S.BLOCKED, S.CANCELLED, S.ABANDONED}),
    S.VERIFYING: frozenset({S.COMPLETED, S.FAILED, S.RUNNING, S.BLOCKED, S.PAUSED, S.CANCELLED, S.QUEUED}),
    S.FAILED: frozenset({S.QUEUED}),           # explicit user retry only
    S.COMPLETED: frozenset(),
    S.CANCELLED: frozenset(),
    S.ABANDONED: frozenset(),
}


class InvalidTransition(ValueError):
    pass


def check_transition(old: TaskStatus, new: TaskStatus) -> None:
    if new not in TRANSITIONS[old]:
        raise InvalidTransition(f"cannot move task from {old.value} to {new.value}")


class TaskKind(StrEnum):
    ONESHOT = "oneshot"    # plan of steps toward an objective
    MONITOR = "monitor"    # delegated observation responsibility with a stop condition


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING_APPROVAL = "waiting_approval"


@dataclass
class Step:
    description: str
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("step"))
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    allow_failure: bool = False     # a failed result is a valid observation (e.g. failing tests)
    result: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    error: str | None = None
    approval_id: str | None = None
    audit_id: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    interrupted: bool = False
    outcome_unknown: bool = False   # it was running when the process died: its effect may or may not have happened
    note: str = ""

    @property
    def finished(self) -> bool:
        return self.status in (StepStatus.DONE, StepStatus.SKIPPED, StepStatus.FAILED)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Step":
        d = dict(d)
        d["status"] = StepStatus(d.get("status", "pending"))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class MonitorSpec:
    target: dict[str, Any]                     # {"type": "task"|"process"|"path"|"command"|"metric"|"url", ...}
    interval_s: float = 5.0
    notify: str = "important"                  # notification priority name
    stop_when: list[str] = field(default_factory=lambda: ["target_resolved"])
    expires_at: float | None = None
    last_check: float | None = None
    last_observation: dict[str, Any] | None = None
    triggers: int = 0
    baseline: dict[str, Any] | None = None     # e.g. file snapshot, previous exit code

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MonitorSpec":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TaskPolicy:
    max_retries: int = 2
    retry_backoff_s: float = 2.0
    on_step_failure: str = "replan"            # replan | fail | continue
    max_replans: int = 2
    # after a restart: "safe" resumes unless the interrupted step's outcome is unknown and repeating it could
    # apply an effect twice; "ask" always waits for the user; "auto" is kept as an alias of "safe"
    resume_after_restart: str = "safe"
    notify_on: list[str] = field(default_factory=lambda: ["completed", "failed", "blocked", "waiting"])
    notify_priority: str = "important"
    cancellable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskPolicy":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Task:
    objective: str
    title: str = ""
    kind: TaskKind = TaskKind.ONESHOT
    id: str = field(default_factory=lambda: new_id("task"))
    owner: str = "owner"
    created_by: str = "user"                   # user | automation:<id> | agent:<id> | system
    priority: Priority = Priority.P1
    status: TaskStatus = TaskStatus.QUEUED
    status_reason: str = ""
    outcome: Outcome | None = None
    progress: float = 0.0
    project_id: str | None = None
    parent_id: str | None = None
    plan: list[Step] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    authority: dict[str, Any] = field(default_factory=lambda: {"interactive": True})
    success_condition: dict[str, Any] | None = None
    policy: TaskPolicy = field(default_factory=TaskPolicy)
    monitor: MonitorSpec | None = None
    checkpoint: dict[str, Any] = field(default_factory=dict)
    recovery: str = ""
    control: dict[str, Any] | None = None      # governing directive (instruction hierarchy)
    budget: dict[str, Any] = field(default_factory=lambda: {"max_steps": 40, "max_model_calls": 20})
    usage: dict[str, Any] = field(default_factory=lambda: {"steps": 0, "model_calls": 0, "replans": 0})
    deadline: float | None = None
    cwd: str | None = None
    dry_run: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    request: str = ""                          # the user's own words, when a request started the task
    origin: str = ""                           # conversation:<session> | api | schedule:<id> | rule:<id> | system
    artifacts: list[dict[str, Any]] = field(default_factory=list)   # files and reports the task produced
    result: str = ""                           # the task's final result, in words
    retry_count: int = 0
    idempotency_key: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    version: int = 0
    # Not persisted: set when control-plane fields (priority, control, deadline) were changed
    # deliberately; otherwise saves preserve the stored values so a worker's stale copy can't undo them.
    control_dirty: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.title:
            self.title = self.objective[:80]

    # -- derived ------------------------------------------------------------------
    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def current_step(self) -> Step | None:
        for step in self.plan:
            if not step.finished:
                return step
        return None

    def completed_steps(self) -> list[Step]:
        return [s for s in self.plan if s.status in (StepStatus.DONE, StepStatus.SKIPPED)]

    def pending_steps(self) -> list[Step]:
        return [s for s in self.plan if not s.finished]

    def compute_progress(self) -> float:
        if not self.plan:
            return 0.0
        return round(len([s for s in self.plan if s.finished]) / len(self.plan), 3)

    def step(self, step_id: str) -> Step | None:
        return next((s for s in self.plan if s.id == step_id), None)

    def failed_steps(self) -> list[Step]:
        return [s for s in self.plan if s.status == StepStatus.FAILED]

    @property
    def error(self) -> str | None:
        """The most recent error, if the task has one."""
        if self.errors:
            return str(self.errors[-1].get("error") or "") or None
        return None

    def to_api(self) -> dict[str, Any]:
        """The task as the local API and interfaces see it."""
        current = self.current_step

        def brief(step: Step) -> dict[str, Any]:
            return {"id": step.id, "description": step.description, "tool": step.tool, "status": step.status.value,
                    "attempts": step.attempts, "error": step.error, "outcome_unknown": step.outcome_unknown}

        return {
            "id": self.id, "title": self.title, "request": self.request, "goal": self.objective,
            "kind": self.kind.value, "status": self.status.value, "status_reason": self.status_reason,
            "outcome": self.outcome.value if self.outcome else None, "priority": self.priority.name,
            "progress": self.compute_progress() if self.plan else self.progress,
            "origin": self.origin, "created_by": self.created_by, "owner": self.owner,
            "project_id": self.project_id, "cwd": self.cwd,
            "current_step": brief(current) if current else None,
            "completed_steps": [brief(s) for s in self.completed_steps()],
            "failed_steps": [brief(s) for s in self.failed_steps()],
            "pending_steps": [brief(s) for s in self.pending_steps()],
            "checkpoint": self.checkpoint, "retry_count": self.retry_count, "dependencies": self.dependencies,
            "permissions": {"interactive": bool(self.authority.get("interactive")), "authority": self.authority},
            "artifacts": self.artifacts, "result": self.result or self.outputs.get("summary", ""),
            "error": self.error, "recovery": self.recovery, "deadline": self.deadline,
            "created_at": self.created_at, "updated_at": self.updated_at, "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    # -- persistence ----------------------------------------------------------------
    def data_blob(self) -> dict[str, Any]:
        return {
            "plan": [s.to_dict() for s in self.plan],
            "outputs": self.outputs,
            "errors": self.errors,
            "dependencies": self.dependencies,
            "resources": self.resources,
            "authority": self.authority,
            "success_condition": self.success_condition,
            "policy": self.policy.to_dict(),
            "monitor": self.monitor.to_dict() if self.monitor else None,
            "checkpoint": self.checkpoint,
            "recovery": self.recovery,
            "control": self.control,
            "budget": self.budget,
            "usage": self.usage,
            "cwd": self.cwd,
            "dry_run": self.dry_run,
            "history": self.history[-100:],
            "request": self.request,
            "origin": self.origin,
            "artifacts": self.artifacts[-50:],
            "result": self.result,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_row(cls, row: Any, data: dict[str, Any]) -> "Task":
        return cls(
            objective=row["objective"], title=row["title"], kind=TaskKind(row["kind"]), id=row["id"],
            owner=row["owner"], created_by=row["created_by"], priority=Priority(row["priority"]),
            status=TaskStatus(row["status"]), status_reason=row["status_reason"] or "",
            outcome=Outcome(row["outcome"]) if row["outcome"] else None, progress=row["progress"],
            project_id=row["project_id"], parent_id=row["parent_id"],
            plan=[Step.from_dict(s) for s in data.get("plan", [])], outputs=data.get("outputs", {}),
            errors=data.get("errors", []), dependencies=data.get("dependencies", []),
            resources=data.get("resources", []), authority=data.get("authority", {}),
            success_condition=data.get("success_condition"),
            policy=TaskPolicy.from_dict(data.get("policy", {})),
            monitor=MonitorSpec.from_dict(data["monitor"]) if data.get("monitor") else None,
            checkpoint=data.get("checkpoint", {}), recovery=data.get("recovery", ""), control=data.get("control"),
            budget=data.get("budget", {}), usage=data.get("usage", {}), deadline=row["deadline"],
            cwd=data.get("cwd"), dry_run=data.get("dry_run", False), history=data.get("history", []),
            request=data.get("request", ""), origin=data.get("origin", ""), artifacts=data.get("artifacts", []),
            result=data.get("result", ""), retry_count=data.get("retry_count", 0),
            idempotency_key=row["idempotency_key"] if "idempotency_key" in row.keys() else None,
            created_at=row["created_at"], updated_at=row["updated_at"], started_at=row["started_at"],
            finished_at=row["finished_at"], version=row["version"],
        )
