"""Observation checks used by monitor tasks (spec §21, §130-131, §180).

A monitor task is a delegated responsibility: "keep an eye on the build",
"watch this folder", "tell me when it's done". Each check is deterministic and
cheap — no language model is involved in watching.
"""

from __future__ import annotations

import operator
import os
from dataclasses import dataclass, field
from typing import Any

import psutil

from jarvis.core.types import Severity
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.state.engine import StateEngine
from jarvis.tasks.manager import TaskManager
from jarvis.tasks.models import TERMINAL, MonitorSpec, TaskStatus
from jarvis.tools.base import ToolContext
from jarvis.tools.registry import ExecStatus, ToolRegistry

_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq,
        "!=": operator.ne}
MAX_WATCHED_FILES = 5000


@dataclass
class Observation:
    state: str                       # ok | triggered | resolved | error
    detail: str
    triggered: bool = False          # something the user should hear about happened
    resolved: bool = False           # the watched thing reached its end state
    severity: Severity = Severity.INFO
    data: dict[str, Any] = field(default_factory=dict)


class MonitorChecker:
    def __init__(self, tasks: TaskManager, registry: ToolRegistry, state: StateEngine, bus: EventBus | None = None):
        self.tasks = tasks
        self.registry = registry
        self.state = state
        self.bus = bus

    async def check(self, spec: MonitorSpec, ctx: ToolContext) -> Observation:
        target = spec.target
        kind = target.get("type")
        spec.baseline = spec.baseline or {}
        try:
            if kind == "task":
                return self._task(spec)
            if kind == "process":
                return self._process(spec)
            if kind == "path":
                return self._path(spec)
            if kind == "command":
                return await self._command(spec, ctx)
            if kind == "metric":
                return self._metric(spec)
        except Exception as exc:  # a broken check must be visible, not silent
            return Observation("error", f"check failed: {exc}", severity=Severity.WARNING)
        return Observation("error", f"unknown monitor target {kind!r}", severity=Severity.WARNING)

    # -- targets -----------------------------------------------------------------------
    def _task(self, spec: MonitorSpec) -> Observation:
        watched = self.tasks.get_task(spec.target["task_id"])
        if watched is None:
            return Observation("resolved", "the watched task no longer exists", triggered=True, resolved=True)
        last = spec.baseline.get("last_status")
        spec.baseline["last_status"] = watched.status.value
        if watched.status in TERMINAL:
            summary = watched.outputs.get("summary") or watched.status_reason or watched.status.value
            sev = Severity.INFO if watched.status == TaskStatus.COMPLETED else Severity.ERROR
            return Observation("resolved", f"{watched.title} {watched.status.value}: {summary}", triggered=True,
                               resolved=True, severity=sev,
                               data={"status": watched.status.value, "outcome": watched.outcome.value
                                     if watched.outcome else None})
        if watched.status in (TaskStatus.BLOCKED, TaskStatus.WAITING) and last != watched.status.value:
            return Observation("triggered", f"{watched.title} is {watched.status.value}: {watched.status_reason}",
                               triggered=True, severity=Severity.WARNING)
        return Observation("ok", f"{watched.title} is {watched.status.value} ({int(watched.progress * 100)}%)")

    def _process(self, spec: MonitorSpec) -> Observation:
        pid = int(spec.target["pid"])
        name = spec.target.get("name") or f"PID {pid}"
        try:
            proc = psutil.Process(pid)
            alive = proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            return Observation("ok", f"{name} is running")
        self._emit(EventType.PROCESS_FINISHED, {"pid": pid, "name": name})
        return Observation("resolved", f"{name} has exited", triggered=True, resolved=True)

    def _snapshot(self, root: str, recursive: bool) -> dict[str, list[float]]:
        snap: dict[str, list[float]] = {}
        if os.path.isfile(root):
            st = os.stat(root)
            return {root: [st.st_mtime, st.st_size]}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")] if recursive else []
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                snap[path] = [st.st_mtime, st.st_size]
                if len(snap) >= MAX_WATCHED_FILES:
                    return snap
        return snap

    def _path(self, spec: MonitorSpec) -> Observation:
        root = os.path.expanduser(spec.target["path"])
        if not os.path.exists(root):
            if spec.baseline.get("missing"):
                return Observation("ok", f"{root} is still missing")
            spec.baseline["missing"] = True
            self._emit(EventType.FILE_DELETED, {"path": root})
            return Observation("triggered", f"{root} no longer exists", triggered=True, severity=Severity.WARNING)
        spec.baseline.pop("missing", None)
        current = self._snapshot(root, bool(spec.target.get("recursive", True)))
        previous = spec.baseline.get("files")
        spec.baseline["files"] = current
        if previous is None:
            return Observation("ok", f"watching {len(current)} file(s) in {root}")
        created = sorted(set(current) - set(previous))
        deleted = sorted(set(previous) - set(current))
        changed = sorted(p for p in set(current) & set(previous) if current[p] != previous[p])
        for p in created:
            self._emit(EventType.FILE_CREATED, {"path": p})
        for p in changed:
            self._emit(EventType.FILE_CHANGED, {"path": p})
        for p in deleted:
            self._emit(EventType.FILE_DELETED, {"path": p})
        if not (created or deleted or changed):
            return Observation("ok", f"no changes in {root}")
        parts = []
        if created:
            parts.append(f"{len(created)} created")
        if changed:
            parts.append(f"{len(changed)} changed")
        if deleted:
            parts.append(f"{len(deleted)} deleted")
        sample = (created + changed + deleted)[:3]
        detail = f"{', '.join(parts)} in {root}: " + ", ".join(os.path.relpath(p, root) if p != root else p
                                                               for p in sample)
        return Observation("triggered", detail, triggered=True,
                           data={"created": created[:50], "changed": changed[:50], "deleted": deleted[:50]})

    async def _command(self, spec: MonitorSpec, ctx: ToolContext) -> Observation:
        command = spec.target["command"]
        expected = int(spec.target.get("expect_exit", 0))
        execution = await self.registry.execute("shell_execute", {"command": command,
                                                                  "cwd": spec.target.get("cwd") or ctx.cwd or None,
                                                                  "timeout_s": spec.target.get("timeout_s", 30)}, ctx)
        if execution.status != ExecStatus.EXECUTED or execution.result is None:
            return Observation("error", f"could not run check: {execution.message}", severity=Severity.WARNING)
        code = execution.result.exit_code
        healthy = code == expected
        was_healthy = spec.baseline.get("healthy")
        spec.baseline["healthy"] = healthy
        label = spec.target.get("label") or command
        if healthy and was_healthy is False:
            return Observation("triggered", f"{label} has recovered", triggered=True,
                               resolved=bool(spec.target.get("resolve_on_recovery")))
        if not healthy and was_healthy is not False:
            return Observation("triggered", f"{label} is failing (exit {code})", triggered=True,
                               severity=Severity.ERROR, data={"exit_code": code})
        return Observation("ok", f"{label} {'healthy' if healthy else 'still failing'}")

    def _metric(self, spec: MonitorSpec) -> Observation:
        key, op, threshold = spec.target["key"], spec.target.get("op", ">"), spec.target["value"]
        value = self.state.value(key, allow_stale=False)
        if value is None:
            return Observation("ok", f"{key} is not currently observed")
        breached = _OPS[op](value, threshold)
        was = spec.baseline.get("breached", False)
        spec.baseline["breached"] = breached
        if breached and not was:
            return Observation("triggered", f"{key} is {value} ({op} {threshold})", triggered=True,
                               severity=Severity.WARNING, data={"value": value})
        if was and not breached:
            return Observation("triggered", f"{key} is back to {value}", triggered=True,
                               resolved=bool(spec.target.get("resolve_on_recovery")))
        return Observation("ok", f"{key} = {value}")

    def _emit(self, etype: EventType, payload: dict[str, Any]) -> None:
        if self.bus is not None:
            self.bus.emit(Event(etype, "monitor", payload, severity=Severity.INFO))
