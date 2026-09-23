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
    S.PAUSED: frozenset({S.QUEUED, S.CANCELLED, S.ABANDONED}),
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
    resume_after_restart: str = "ask"          # ask | auto (auto only honoured for idempotent steps)
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
            created_at=row["created_at"], updated_at=row["updated_at"], started_at=row["started_at"],
            finished_at=row["finished_at"], version=row["version"],
        )
