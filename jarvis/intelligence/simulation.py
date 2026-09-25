"""Simulation and prediction (Phase 3 §26-27).

Four different things, never blurred together:

* **actual execution** — something was changed (only through the tool registry, with approval when needed);
* **dry run** — the plan runs its observation steps for real and evaluates each change through the registry
  without making it: "these are the changes I would make, and which would need your approval";
* **simulation** — "what would happen if I stopped Chrome?": an estimate of the effect of a hypothetical
  action, computed from what is measured right now (the process's CPU and memory share, a folder's size);
* **prediction** — "how long will the tests take?": an estimate from history (earlier runs of the same work).

Simulations and predictions run only read-only tools, are labelled as estimates with the basis they came from,
and say plainly when there is nothing to base an estimate on.
"""

from __future__ import annotations

import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from jarvis.clock import format_duration
from jarvis.core.types import OperationalReason
from jarvis.intelligence.analysis import processes, size_text
from jarvis.permissions.model import Actor
from jarvis.tasks.models import TaskStatus
from jarvis.tools.base import ToolContext
from jarvis.tools.registry import ExecStatus


@dataclass
class Estimate:
    text: str
    basis: str
    kind: str                                   # simulation | prediction
    confident: bool = True
    data: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        label = "Estimate" if self.kind == "simulation" else "Prediction"
        return f"{label}: {self.text} (based on {self.basis}). Nothing was changed."


_STOP = re.compile(r"\b(?:stopped|stop|closed|close|killed|kill|quit|ended|end)\s+(?:the\s+|my\s+)?(?P<name>[\w.+ -]+?)"
                   r"(?:\s+process)?\s*\??$", re.I)
_DELETE = re.compile(r"\b(?:deleted|delete|removed|remove|cleared|clear|emptied|empty)\s+(?:the\s+|my\s+)?"
                     r"(?P<path>[~/.\\]?[\w:./\\ -]+?)\s*\??$", re.I)
_UNLOAD = re.compile(r"\bunload(?:ed)?\s+(?:the\s+)?(?:model\s+)?(?P<name>[\w.:-]+)", re.I)


class Simulator:
    def __init__(self, registry: Any, tasks: Any, *, router: Any = None, clock: Any = None, owner: str = "owner",
                 data_dir: str | None = None) -> None:
        self.registry = registry
        self.tasks = tasks
        self.router = router
        self.clock = clock
        self.owner = owner
        self.data_dir = data_dir

    async def _observe(self, tool: str, args: dict[str, Any], cwd: str | None, why: str) -> Any:
        ctx = ToolContext(actor=Actor.user(self.owner), cwd=cwd, data_dir=self.data_dir,
                          **({"clock": self.clock} if self.clock else {}))
        ex = await self.registry.execute(tool, args, ctx, reason=OperationalReason(
            why, "simulations only observe", f"read {tool}"))
        if ex.status != ExecStatus.EXECUTED or ex.result is None or not ex.result.ok:
            return None
        return ex.result.data

    # -- simulation ------------------------------------------------------------------------------------------
    async def simulate(self, text: str, *, cwd: str | None = None) -> Estimate:
        body = re.sub(r"^(simulate:?|what\s+(would|will)\s+happen\s+if|what\s+if)\s+(i|you|we)?\s*", "",
                      text.strip(), flags=re.I)
        m = _UNLOAD.search(body)
        if m:
            return self._unload(m.group("name"))
        m = _STOP.search(body)
        if m:
            return await self._stop(m.group("name").strip(), cwd)
        m = _DELETE.search(body)
        if m:
            return await self._delete(m.group("path").strip(), cwd)
        return Estimate("I can't simulate that reliably. I can estimate the effect of stopping a program, deleting "
                        "a file or folder, or unloading a model; for anything else, try a dry run "
                        "('dry run: ...') to see exactly what I would do", "no model of that action", "simulation",
                        confident=False)

    async def _stop(self, name: str, cwd: str | None) -> Estimate:
        system = await self._observe("system_info", {}, cwd, f"simulating stopping {name}") or {}
        listing = await self._observe("process_list", {"name": name.split()[0], "limit": 50, "sample_s": 1.0,
                                                       "sort": "cpu"}, cwd, f"simulating stopping {name}")
        procs = processes(listing or {})
        if not procs:
            return Estimate(f"nothing called '{name}' is running, so stopping it would change nothing",
                            "the current process list", "simulation")
        ncpu = int(system.get("cpu_count") or (listing or {}).get("cpu_count") or os.cpu_count() or 1)
        cpu_share = sum(float(p.get("cpu_percent") or 0) for p in procs) / ncpu
        mem_share = sum(float(p.get("memory_percent") or 0) for p in procs)
        cpu, mem = system.get("cpu_percent"), system.get("memory_percent")
        parts = [f"{len(procs)} process(es) named like '{name}' would close"]
        if isinstance(cpu, (int, float)):
            parts.append(f"CPU would go from about {cpu:.0f}% to about {max(0.0, cpu - cpu_share):.0f}%")
        if isinstance(mem, (int, float)):
            if len(procs) > 1:
                # each process's figure counts memory they share, so the sum overstates what closing them frees
                parts.append(f"memory could drop from about {mem:.0f}% to as low as about "
                             f"{max(0.0, mem - mem_share):.0f}% (the processes share some memory, so the real drop "
                             f"is usually smaller)")
            else:
                parts.append(f"memory from about {mem:.0f}% to about {max(0.0, mem - mem_share):.0f}%")
        parts.append("anything unsaved in it would be lost")
        return Estimate("; ".join(parts), "its CPU and memory use measured just now", "simulation",
                        data={"processes": len(procs), "cpu_share": round(cpu_share, 1), "memory_share": round(mem_share, 1)})

    async def _delete(self, path_text: str, cwd: str | None) -> Estimate:
        path = os.path.expanduser(path_text)
        if not os.path.isabs(path):
            path = os.path.join(cwd or os.getcwd(), path)
        if not os.path.exists(path):
            return Estimate(f"{path} doesn't exist, so deleting it would change nothing", "the file system",
                            "simulation")
        usage = await self._observe("disk_usage", {"path": path if os.path.isdir(path) else os.path.dirname(path),
                                                   "top": 50}, cwd, f"simulating deleting {path}") or {}
        if os.path.isdir(path):
            size = sum(c.get("size", 0) for c in usage.get("children", []))
        else:
            size = os.path.getsize(path)
        free = usage.get("free")
        text = f"it would free about {size_text(size)}"
        if isinstance(free, (int, float)):
            text += f" (free space from {size_text(free)} to about {size_text(free + size)})"
        text += "; if I did it, the files would go to JARVIS's trash first, so it could be undone"
        return Estimate(text, "the sizes measured just now", "simulation", data={"bytes": size})

    def _unload(self, name: str) -> Estimate:
        info = self.router.find(name) if self.router else None
        if info is None:
            return Estimate(f"no model called '{name}' is installed", "the model list", "simulation")
        if not info.loaded:
            return Estimate(f"{info.name} isn't loaded, so unloading it would change nothing", "the model list",
                            "simulation")
        held = info.vram_bytes or info.size_bytes
        text = f"{info.name} would release its memory" + (f" (about {size_text(held)})" if held else "")
        text += "; the next request that needs it would wait while it loads again"
        return Estimate(text, "the model runtime's report", "simulation")

    # -- prediction ----------------------------------------------------------------------------------------------
    def predict(self, text: str, *, project_id: str | None = None) -> Estimate:
        lowered = text.lower()
        template = "run_tests" if "test" in lowered else "build" if "build" in lowered else None
        running = self._running_eta(lowered)
        if running is not None:
            return running
        if template is None:
            return Estimate("I don't have a way to predict that", "no comparable history", "prediction",
                            confident=False)
        durations = []
        for task in self.tasks.list_tasks([TaskStatus.COMPLETED, TaskStatus.FAILED], order="recent", limit=200):
            if task.outputs.get("template") != template or not task.started_at or not task.finished_at:
                continue
            if project_id and task.project_id != project_id:
                continue
            durations.append(task.finished_at - task.started_at)
            if len(durations) >= 10:
                break
        what = "the tests" if template == "run_tests" else "the build"
        if not durations:
            return Estimate(f"I haven't run {what} here before, so I have nothing to estimate from",
                            "no earlier runs", "prediction", confident=False)
        median = statistics.median(durations)
        spread = f", between {format_duration(min(durations))} and {format_duration(max(durations))}" \
            if len(durations) > 1 else ""
        return Estimate(f"{what} should take about {format_duration(median)}{spread}",
                        f"the last {len(durations)} run(s)", "prediction", confident=len(durations) >= 3,
                        data={"median_s": median, "runs": len(durations)})

    def _running_eta(self, lowered: str) -> Estimate | None:
        m = re.search(r"when will (?:the |my )?(?P<what>.+?) (?:finish|be done)", lowered)
        if not m:
            return None
        matches = self.tasks.find(m.group("what"), [TaskStatus.RUNNING, TaskStatus.QUEUED])
        if not matches:
            return Estimate(f"nothing called '{m.group('what')}' is running", "the task list", "prediction",
                            confident=False)
        task = matches[0]
        progress = task.compute_progress()
        if not task.started_at or progress <= 0 or self.clock is None:
            return Estimate(f"{task.title} has only just started, so there's no rate to extrapolate from yet",
                            "its progress so far", "prediction", confident=False)
        elapsed = self.clock.now() - task.started_at
        remaining = elapsed / progress - elapsed
        return Estimate(f"{task.title} is {progress:.0%} done; at this rate about {format_duration(remaining)} more",
                        "its progress so far (steps vary in length, so this is rough)", "prediction", confident=False)
