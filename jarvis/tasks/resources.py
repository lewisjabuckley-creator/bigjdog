"""Resource manager: admission, locks and throttling (spec §29-30, §104-106, §132).

Resources influence planning. Work declares what it needs; exclusive resources
(files, GPU, devices) are locked; under pressure, low-priority work is deferred
or paused. Every such decision is recorded with an operational reason so the
user can ask "why is that task slow?".
"""

from __future__ import annotations

from typing import Callable

from jarvis.clock import Clock, SystemClock
from jarvis.core.types import OperationalReason, Priority
from jarvis.state.engine import StateEngine
from jarvis.tasks.models import Task, TaskKind

EXCLUSIVE_PREFIXES = ("file:", "gpu", "device:", "lock:", "port:")
RESOURCE_MANAGER = "resource-manager"


class ResourceManager:
    def __init__(self, state: StateEngine, *, clock: Clock | None = None, memory_critical: float = 92.0,
                 cpu_critical: float = 97.0, vram_critical: float = 90.0,
                 mode_provider: Callable[[], str] | None = None) -> None:
        self.state = state
        self.clock = clock or SystemClock()
        self.memory_critical = memory_critical
        self.cpu_critical = cpu_critical
        self.vram_critical = vram_critical
        self.mode_provider = mode_provider or (lambda: "normal")
        self.locks: dict[str, str] = {}
        self.reasons: dict[str, OperationalReason] = {}
        self.throttled: set[str] = set()

    @staticmethod
    def exclusive(resource: str) -> bool:
        return resource.startswith(EXCLUSIVE_PREFIXES)

    # -- pressure -------------------------------------------------------------------
    def pressure(self) -> tuple[bool, str]:
        mode = self.mode_provider()
        if mode in ("low_resource", "emergency"):
            return True, f"{mode.replace('_', '-')} mode is active"
        mem = self.state.value("resources.memory_percent", allow_stale=False)
        if isinstance(mem, (int, float)) and mem >= self.memory_critical:
            return True, f"memory usage is {mem:.0f}% (limit {self.memory_critical:.0f}%)"
        cpu = self.state.value("resources.cpu_percent", allow_stale=False)
        if isinstance(cpu, (int, float)) and cpu >= self.cpu_critical:
            return True, f"CPU usage is {cpu:.0f}%"
        return False, ""

    def gpu_pressure(self) -> tuple[bool, str]:
        used = self.state.value("resources.vram_used_gb", allow_stale=False)
        total = self.state.value("resources.vram_total_gb", allow_stale=False)
        if isinstance(used, (int, float)) and isinstance(total, (int, float)) and total > 0:
            pct = used / total * 100
            if pct >= self.vram_critical:
                holder = self.state.value("models.loaded")
                who = f" by {', '.join(holder)}" if holder else ""
                return True, f"GPU memory is {pct:.0f}% allocated{who}"
        return False, ""

    # -- admission ------------------------------------------------------------------
    def admit(self, task: Task, running: list[Task]) -> tuple[bool, str]:
        titles = {t.id: t.title for t in running}
        for resource in task.resources:
            holder = self.locks.get(resource)
            if self.exclusive(resource) and holder and holder != task.id:
                return self._defer(task, f"{resource} is in use by '{titles.get(holder, holder)}'",
                                   "tasks that modify the same resource run one at a time", "queued the task")
        if task.priority <= Priority.P1 or task.kind == TaskKind.MONITOR:
            return True, ""
        constrained, why = self.pressure()
        if constrained:
            return self._defer(task, why, f"under resource pressure only P0-P2 work starts; this is {task.priority.name}",
                               "deferred the task")
        if "gpu" in task.resources:
            gpu, why = self.gpu_pressure()
            if gpu:
                return self._defer(task, why, "GPU work below P1 waits for GPU memory", "deferred the task")
        self.reasons.pop(task.id, None)
        return True, ""

    def _defer(self, task: Task, condition: str, rule: str, action: str) -> tuple[bool, str]:
        reason = OperationalReason(condition, rule, action, "It will start automatically when possible.")
        self.reasons[task.id] = reason
        return False, reason.sentence()

    def acquire(self, task: Task) -> None:
        for resource in task.resources:
            if self.exclusive(resource):
                self.locks[resource] = task.id

    def release(self, task: Task) -> None:
        for resource, holder in list(self.locks.items()):
            if holder == task.id:
                del self.locks[resource]

    # -- throttling -------------------------------------------------------------------
    def throttle(self, running: list[Task]) -> list[tuple[Task, OperationalReason]]:
        """Running work to pause because resources are constrained (lowest priority first)."""
        constrained, why = self.pressure()
        if not constrained:
            return []
        out = []
        for task in sorted(running, key=lambda t: -int(t.priority)):
            if task.priority >= Priority.P3 and task.kind != TaskKind.MONITOR and task.id not in self.throttled:
                reason = OperationalReason(why, f"{task.priority.name} work yields under resource pressure",
                                           f"paused '{task.title}'", "It resumes automatically when pressure clears.")
                self.reasons[task.id] = reason
                self.throttled.add(task.id)
                out.append((task, reason))
        return out

    def resumable(self) -> list[str]:
        """Tasks this manager paused that may now resume."""
        if self.pressure()[0] or not self.throttled:
            return []
        ids = list(self.throttled)
        self.throttled.clear()
        return ids

    def explain(self, task_id: str) -> OperationalReason | None:
        return self.reasons.get(task_id)
