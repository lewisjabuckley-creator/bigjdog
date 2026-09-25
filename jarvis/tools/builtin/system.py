"""System, process and time tools backed by psutil (spec §29, §31)."""

from __future__ import annotations

import os
import platform
import socket
import time
from typing import Any

import psutil

from jarvis.core.types import Provenance, ProvenanceKind, RiskLevel
from jarvis.permissions.model import PermissionLevel
from jarvis.tools.base import Tool, ToolContext, ToolResult, ToolSpec, Verification

_PROV = Provenance(ProvenanceKind.SYSTEM_STATE, "psutil")


class SystemInfoTool(Tool):
    spec = ToolSpec(
        name="system_info",
        description="Report live CPU, memory, disk, battery, temperature, OS and uptime information.",
        parameters={"type": "object", "properties": {}},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="read-only", category="system",
    )

    def __init__(self, metrics_source: Any | None = None) -> None:
        self.metrics_source = metrics_source

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if self.metrics_source is not None:
            data = dict(self.metrics_source.sample())
        else:
            from jarvis.monitoring.metrics import PsutilMetrics
            data = dict(PsutilMetrics().sample())
        data.update({"os": f"{platform.system()} {platform.release()}", "hostname": socket.gethostname(),
                     "python": platform.python_version()})
        summary = f"CPU {data.get('cpu_percent', '?')}%, memory {data.get('memory_percent', '?')}%, " \
                  f"disk {data.get('disk_percent', '?')}%"
        return ToolResult(True, summary, data, provenance=_PROV)


class ProcessListTool(Tool):
    spec = ToolSpec(
        name="process_list",
        description="List running processes, optionally filtered by name, sorted by CPU or memory.",
        parameters={"type": "object", "properties": {
            "name": {"type": "string", "default": ""},
            "sort": {"type": "string", "enum": ["cpu", "memory"], "default": "cpu"},
            "limit": {"type": "integer", "default": 15, "minimum": 1},
            # psutil's first CPU reading for a process is always 0: measure over a window for real figures
            "sample_s": {"type": "number", "default": 0, "minimum": 0, "maximum": 10},
        }},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="read-only", category="system",
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        import asyncio
        procs = []
        needle = args["name"].lower()
        sample = float(args.get("sample_s") or 0)
        if sample > 0:
            for p in psutil.process_iter():
                try:
                    p.cpu_percent(None)
                except psutil.Error:
                    pass
            await asyncio.sleep(sample)
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status", "username"]):
            info = p.info
            if needle and needle not in (info.get("name") or "").lower():
                continue
            procs.append({"pid": info["pid"], "name": info.get("name"), "cpu_percent": info.get("cpu_percent") or 0.0,
                          "memory_percent": round(info.get("memory_percent") or 0.0, 2), "status": info.get("status"),
                          "user": info.get("username")})
        key = "cpu_percent" if args["sort"] == "cpu" else "memory_percent"
        procs.sort(key=lambda p: p[key], reverse=True)
        procs = procs[: args["limit"]]
        return ToolResult(True, f"{len(procs)} process(es)", {"processes": procs, "cpu_count": psutil.cpu_count(),
                                                             "sampled_s": sample}, provenance=_PROV)


class ProcessInspectTool(Tool):
    spec = ToolSpec(
        name="process_inspect",
        description="Inspect one process by PID: command line, status, resource usage, children.",
        parameters={"type": "object", "properties": {"pid": {"type": "integer"}}, "required": ["pid"]},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="read-only", category="system",
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            p = psutil.Process(args["pid"])
            with p.oneshot():
                data = {"pid": p.pid, "name": p.name(), "status": p.status(), "cmdline": p.cmdline()[:20],
                        "cpu_percent": p.cpu_percent(interval=0.1), "memory_rss": p.memory_info().rss,
                        "started": p.create_time(), "children": [c.pid for c in p.children()][:50],
                        "cwd": _safe(p.cwd)}
        except psutil.NoSuchProcess:
            return ToolResult(False, f"no process with PID {args['pid']}", error="not_found", provenance=_PROV)
        except psutil.AccessDenied:
            return ToolResult(False, f"access denied for PID {args['pid']}", error="access_denied", provenance=_PROV)
        return ToolResult(True, f"PID {p.pid} ({data['name']}) is {data['status']}", data, provenance=_PROV)


class ProcessStopTool(Tool):
    spec = ToolSpec(
        name="process_stop",
        description="Stop a process by PID (SIGTERM, or SIGKILL when force=true). Requires authorization.",
        parameters={"type": "object", "properties": {
            "pid": {"type": "integer"},
            "force": {"type": "boolean", "default": False},
            "grace_s": {"type": "number", "default": 5, "minimum": 0},
        }, "required": ["pid"]},
        level=PermissionLevel.EXECUTE_CONSEQUENTIAL, risk=RiskLevel.HIGH, side_effects=("process.signal",),
        reversible=False, verification="process no longer running", category="system",
    )

    def preview(self, args: dict[str, Any]) -> str:
        pid = args.get("pid")
        try:
            name = psutil.Process(int(pid)).name()
        except (psutil.Error, TypeError, ValueError):
            name = None
        what = f"{name} (PID {pid})" if name else f"process {pid}"
        return f"{'force-stop' if args.get('force') else 'stop'} {what}"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if args["pid"] in (0, 1, os.getpid()):
            return ToolResult(False, "refusing to stop a protected process", error="protected")
        try:
            p = psutil.Process(args["pid"])
            name = p.name()
            p.kill() if args["force"] else p.terminate()
            try:
                p.wait(timeout=args["grace_s"])
            except psutil.TimeoutExpired:
                return ToolResult(False, f"{name} ({p.pid}) did not exit within {args['grace_s']}s",
                                  error="timeout", data={"pid": p.pid})
        except psutil.NoSuchProcess:
            return ToolResult(False, f"no process with PID {args['pid']}", error="not_found")
        except psutil.AccessDenied:
            return ToolResult(False, f"access denied for PID {args['pid']}", error="access_denied")
        return ToolResult(True, f"stopped {name} ({args['pid']})", {"pid": args["pid"], "name": name},
                          provenance=_PROV)

    async def verify(self, args: dict[str, Any], result: ToolResult, ctx: ToolContext) -> Verification:
        if not result.ok:
            return Verification.not_performed("stop failed")
        try:
            running = psutil.Process(args["pid"]).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            running = False
        return Verification(True, not running, "process_table",
                            "process is gone" if not running else "process is still running")


class TimeTool(Tool):
    spec = ToolSpec(
        name="time_now",
        description="Current local date, time and timezone.",
        parameters={"type": "object", "properties": {}},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="read-only", category="system",
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        now = ctx.clock.now()
        local = time.localtime(now)
        data = {"epoch": now, "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", local),
                "time": time.strftime("%H:%M", local), "date": time.strftime("%A %d %B %Y", local),
                "timezone": time.strftime("%Z", local)}
        return ToolResult(True, f"{data['time']} {data['timezone']}, {data['date']}", data,
                          provenance=Provenance(ProvenanceKind.SYSTEM_STATE, "clock"))


def _safe(fn: Any) -> Any:
    try:
        return fn()
    except (psutil.Error, OSError):
        return None
