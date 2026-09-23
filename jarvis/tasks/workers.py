"""Worker pool and scheduler (spec §19, §102-105, §142).

Runs queued tasks as independent asyncio workers so long-running work never
blocks conversation. Admission honours priority, dependencies, concurrency
limits and resource constraints; under pressure, low-priority work is throttled
and later resumed automatically. Deadlines are watched and at-risk work is
reported early.
"""

from __future__ import annotations

import asyncio
from typing import Iterable

from jarvis.audit.log import AuditLog
from jarvis.clock import Clock, SystemClock
from jarvis.config import TasksConfig
from jarvis.core.types import HealthStatus, Priority, Severity
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.log import get_logger
from jarvis.permissions.hierarchy import InstructionSource
from jarvis.state.health import HealthRegistry
from jarvis.tasks.executor import TaskExecutor
from jarvis.tasks.manager import TaskController, TaskManager
from jarvis.tasks.models import TERMINAL, Task, TaskKind, TaskStatus
from jarvis.tasks.resources import RESOURCE_MANAGER, ResourceManager

log = get_logger("workers")


class WorkerPool:
    def __init__(self, manager: TaskManager, executor: TaskExecutor, resources: ResourceManager, *,
                 config: TasksConfig | None = None, bus: EventBus | None = None, clock: Clock | None = None,
                 health: HealthRegistry | None = None, audit: AuditLog | None = None) -> None:
        self.manager = manager
        self.executor = executor
        self.resources = resources
        self.config = config or TasksConfig()
        self.bus = bus
        self.clock = clock or SystemClock()
        self.health = health
        self.audit = audit
        self.running: dict[str, asyncio.Task[Task]] = {}
        self._kick: asyncio.Event | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._stopping = False
        self._deadline_warned: set[str] = set()

    # -- lifecycle -------------------------------------------------------------------
    async def start(self) -> None:
        self._kick = asyncio.Event()
        self._stopping = False
        self.manager.set_wakeup(self.kick)
        self._loop_task = asyncio.create_task(self._loop(), name="jarvis-scheduler")
        if self.health:
            self.health.register("workers", critical=True, heartbeat_timeout_s=max(30.0, self.config.scheduler_interval_s * 20))
            self.health.report("workers", HealthStatus.HEALTHY, "scheduler running")

    def kick(self) -> None:
        if self._kick is not None:
            self._kick.set()

    async def stop(self, timeout: float = 10.0) -> None:
        self._stopping = True
        self.kick()
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):
                pass
        for task_id in list(self.running):
            controller = self.manager.controllers.get(task_id)
            if controller:
                controller.request("shutdown", "JARVIS is shutting down")
        if self.running:
            await asyncio.wait(list(self.running.values()), timeout=timeout)
        for runner in list(self.running.values()):
            runner.cancel()
        if self.health:
            self.health.report("workers", HealthStatus.OFFLINE, "stopped")

    async def _loop(self) -> None:
        assert self._kick is not None
        while not self._stopping:
            try:
                await self.tick()
                if self.health:
                    self.health.report("workers", HealthStatus.HEALTHY, f"{len(self.running)} running")
            except Exception as exc:  # the scheduler must survive its own bugs
                log.error("scheduler_tick_failed", error=repr(exc))
                if self.health:
                    self.health.report("workers", HealthStatus.DEGRADED, f"tick failed: {exc}")
            try:
                await asyncio.wait_for(self._kick.wait(), timeout=self.config.scheduler_interval_s)
            except asyncio.TimeoutError:
                pass
            self._kick.clear()

    # -- scheduling ---------------------------------------------------------------------
    def _running_tasks(self) -> list[Task]:
        return [t for t in (self.manager.get_task(tid) for tid in self.running) if t is not None]

    async def tick(self) -> None:
        running = self._running_tasks()
        self._throttle(running)
        self._check_deadlines(running)
        busy = sum(1 for t in running if t.kind != TaskKind.MONITOR)
        for task in self.manager.list_tasks([TaskStatus.QUEUED]):
            if task.id in self.running:
                continue
            if not self._dependencies_ready(task):
                continue
            if task.kind != TaskKind.MONITOR and task.priority != Priority.P0 and busy >= self.config.max_concurrent:
                self._note(task, f"queued behind {busy} running task(s)")
                continue
            ok, why = self.resources.admit(task, running)
            if not ok:
                self._note(task, why)
                continue
            self.launch(task)
            running.append(task)
            if task.kind != TaskKind.MONITOR:
                busy += 1

    def _dependencies_ready(self, task: Task) -> bool:
        for dep_id in task.dependencies:
            dep = self.manager.get_task(dep_id)
            if dep is None or dep.status in (TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.ABANDONED):
                what = "is missing" if dep is None else dep.status.value
                self.manager.transition(task, TaskStatus.BLOCKED,
                                        f"dependency '{dep.title if dep else dep_id}' {what}; not running this step "
                                        "unless you override")
                return False
            if dep.status != TaskStatus.COMPLETED:
                self._note(task, f"waiting for '{dep.title}' to finish")
                return False
        return True

    def _note(self, task: Task, reason: str) -> None:
        if task.status_reason != reason:
            task.status_reason = reason
            self.manager.save(task)

    def launch(self, task: Task) -> None:
        controller = TaskController(task.id)
        self.manager.controllers[task.id] = controller
        self.resources.acquire(task)
        runner = asyncio.create_task(self._run(task, controller), name=f"task-{task.id}")
        controller.runner = runner
        self.running[task.id] = runner

    async def _run(self, task: Task, controller: TaskController) -> Task:
        try:
            return await self.executor.run(task, controller)
        except Exception as exc:
            log.error("task_crashed", task_id=task.id, error=repr(exc))
            current = self.manager.get_task(task.id) or task
            if not current.terminal:
                current.errors.append({"ts": self.clock.now(), "error": repr(exc), "step": None})
                current.outputs["summary"] = f"internal error: {exc}"
                try:
                    self.manager.transition(current, TaskStatus.FAILED, f"internal error: {exc}")
                except Exception:
                    self.manager.save(current)
            return current
        finally:
            self.running.pop(task.id, None)
            self.manager.controllers.pop(task.id, None)
            self.resources.release(task)
            self.kick()

    def _throttle(self, running: list[Task]) -> None:
        for task, reason in self.resources.throttle(running):
            result = self.manager.pause_task(task.id, by=RESOURCE_MANAGER, source=InstructionSource.DEFAULT,
                                             reason=reason.sentence())
            if result.ok:
                if self.audit:
                    self.audit.record(actor=f"system:{RESOURCE_MANAGER}", action="pause_task", task_id=task.id,
                                      summary=reason.sentence(), reason=reason)
                if self.bus:
                    self.bus.emit(Event(EventType.RESOURCE_THROTTLED, RESOURCE_MANAGER,
                                        {"title": task.title, "reason": reason.sentence()},
                                        severity=Severity.WARNING, task_id=task.id))
        for task_id in self.resources.resumable():
            task = self.manager.get_task(task_id)
            if task and task.status == TaskStatus.PAUSED and task.control \
                    and task.control.get("issued_by") == RESOURCE_MANAGER:
                self.manager.resume_task(task_id, by=RESOURCE_MANAGER, source=InstructionSource.DEFAULT,
                                         reason="resource pressure cleared")
                if self.audit:
                    self.audit.record(actor=f"system:{RESOURCE_MANAGER}", action="resume_task", task_id=task_id,
                                      summary="resumed after resource pressure cleared",
                                      reason={"condition": "resource pressure cleared",
                                              "rule": "throttled work resumes automatically",
                                              "action": f"resumed '{task.title}'"})

    def _check_deadlines(self, running: Iterable[Task]) -> None:
        now = self.clock.now()
        for task in running:
            if not task.deadline or task.id in self._deadline_warned or task.started_at is None:
                continue
            progress = task.compute_progress()
            elapsed = now - task.started_at
            if progress <= 0 or elapsed <= 0:
                eta = None
            else:
                eta = task.started_at + elapsed / progress
            if now > task.deadline or (eta is not None and eta > task.deadline):
                self._deadline_warned.add(task.id)
                if self.bus:
                    self.bus.emit(Event(EventType.DEADLINE_AT_RISK, "scheduler",
                                        {"title": task.title, "deadline": task.deadline, "eta": eta,
                                         "progress": progress}, severity=Severity.WARNING, task_id=task.id))

    # -- test / CLI helpers ------------------------------------------------------------------
    async def wait_for(self, task_id: str, statuses: Iterable[TaskStatus] | None = None,
                       timeout: float = 10.0) -> Task:
        wanted = set(statuses) if statuses is not None else set(TERMINAL)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            task = self.manager.get_task(task_id)
            # for terminal states also wait for the worker to release its resources
            if task is not None and task.status in wanted and (task.status not in TERMINAL
                                                               or task_id not in self.running):
                return task
            if loop.time() > deadline:
                raise TimeoutError(f"task {task_id} did not reach {sorted(s.value for s in wanted)} "
                                   f"(currently {task.status.value if task else 'missing'})")
            await asyncio.sleep(0.01)

    async def wait_idle(self, timeout: float = 10.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            non_monitor = [tid for tid in self.running
                           if (t := self.manager.get_task(tid)) and t.kind != TaskKind.MONITOR]
            queued = [t for t in self.manager.list_tasks([TaskStatus.QUEUED]) if t.kind != TaskKind.MONITOR]
            if not non_monitor and not queued:
                return
            if loop.time() > deadline:
                raise TimeoutError("workers did not become idle")
            await asyncio.sleep(0.01)
