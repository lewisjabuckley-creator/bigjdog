"""Live end-to-end checks against the configured model runtime (``jarvis doctor --live``).

These verify the real chain on the user's own machine: Ollama reachable → a
chat model answers → answers stream → natural-language requests reach the tool
registry → results are verified → memory flows into the model's context →
consequential actions stop for approval → background work becomes a durable
task. They run in a throwaway data directory, so nothing is written to the
user's real JARVIS memory, tasks or audit log.

Result levels: PASS; WARN — the infrastructure works but the model did not do
what was asked (common with very small models); FAIL — something is broken;
SKIP — the check could not run because an earlier one failed.
"""

from __future__ import annotations

import copy
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from jarvis.config import JarvisConfig
from jarvis.models.base import ChatMessage, ModelError, NoModelAvailable
from jarvis.models.router import TaskProfile
from jarvis.runtime import Runtime
from jarvis.tasks.models import TaskStatus


@dataclass
class CheckResult:
    name: str
    status: str            # PASS | WARN | FAIL | SKIP
    detail: str
    seconds: float = 0.0

    def line(self) -> str:
        return f"[{self.status}] {self.name} — {self.detail} ({self.seconds:.1f}s)"


Reporter = Callable[[CheckResult], None]


def live_config(config: JarvisConfig, workdir: Path) -> JarvisConfig:
    cfg = copy.deepcopy(config)
    cfg.general.data_dir = str(workdir / "data")
    cfg.permissions.allowed_roots = list(cfg.permissions.allowed_roots) + [str(workdir)]
    cfg.monitoring.enabled = False
    return cfg


async def live_checks(config: JarvisConfig, *, model: str | None = None, level: str = "full",
                      workdir: Path | None = None, report: Reporter | None = None,
                      runtime_kwargs: dict | None = None) -> list[CheckResult]:
    workdir = workdir or Path(tempfile.mkdtemp(prefix="jarvis-live-"))
    results: list[CheckResult] = []

    def record(name: str, status: str, detail: str, started: float) -> CheckResult:
        result = CheckResult(name, status, detail, time.monotonic() - started)
        results.append(result)
        if report:
            report(result)
        return result

    runtime = Runtime(live_config(config, workdir), **(runtime_kwargs or {}))
    t = time.monotonic()
    startup = await runtime.start(monitoring=False)
    svc = runtime.svc
    assert svc is not None
    try:
        health = await svc.router.provider_health()
        up = {name: h for name, h in health.items() if h["available"]}
        if not up:
            detail = "; ".join(f"{n}: {h['detail']}" for n, h in health.items()) or "no model provider configured"
            record("model runtime reachable", "FAIL", detail, t)
            return results
        record("model runtime reachable", "PASS",
               ", ".join(f"{n} {h.get('version') or ''}".strip() for n, h in up.items()), t)

        t = time.monotonic()
        if model:
            try:
                svc.router.pin(model)
            except NoModelAvailable as exc:
                record("chat model installed", "FAIL", str(exc), t)
                return results
        readiness = startup.readiness
        if readiness is None or not readiness.can_converse:
            record("chat model installed", "FAIL", " ".join(readiness.issues if readiness else []) or "none", t)
            return results
        chat_model = model or readiness.chat_model
        record("chat model installed", "PASS" if readiness.tools or model else "WARN",
               f"{chat_model}" + ("" if readiness.tools else " (cannot call tools)"), t)

        t = time.monotonic()
        try:
            routed = await svc.router.chat(TaskProfile(), [
                ChatMessage("system", "You are a test harness. Follow instructions exactly."),
                ChatMessage("user", "Reply with the single word: ready")])
            text = routed.response.content.strip()
            record("model answers", "PASS" if text else "FAIL",
                   f"{routed.response.model} replied {text[:60]!r}", t)
        except ModelError as exc:
            record("model answers", "FAIL", str(exc), t)
            return results

        t = time.monotonic()
        try:
            chunks = [c async for c in svc.router.stream(TaskProfile(), [ChatMessage("user", "Count from 1 to 5.")])]
            deltas = [c.delta for c in chunks if c.delta]
            done = bool(chunks and chunks[-1].done)
            record("streaming", "PASS" if deltas and done else "FAIL",
                   f"{len(deltas)} chunk(s), final={'yes' if done else 'no'}", t)
        except ModelError as exc:
            record("streaming", "FAIL", str(exc), t)

        if level == "contract":
            return results

        orch = runtime.orchestrator()

        # natural language → tool registry → verified result
        t = time.monotonic()
        before = len(svc.audit.query(action="tool_execute", limit=500))
        reply = await orch.handle("Use your system_info tool to check this computer's current memory usage, "
                                  "then tell me the percentage.")
        runs = svc.audit.query(action="tool_execute", limit=500)
        new_runs = runs[: len(runs) - before]
        used = [e for e in new_runs if e.actor == f"user:{svc.user}"]
        if used:
            record("request reaches tools", "PASS", f"ran {', '.join(sorted({e.tool for e in used}))}; "
                                                    f"replied {reply.text[:80]!r}", t)
        else:
            record("request reaches tools", "WARN", f"the model answered without calling a tool: {reply.text[:100]!r}", t)

        # file creation through a tool, verified on disk
        t = time.monotonic()
        target = workdir / "hello.txt"
        reply = await orch.handle(f"Create a file at {target} containing exactly this text: hi from jarvis")
        await _settle(svc)
        if target.exists() and "hi from jarvis" in target.read_text(errors="replace"):
            writes = [e for e in svc.audit.query(action="tool_execute", limit=50) if e.tool == "file_write"]
            verified = bool(writes and writes[0].verification and writes[0].verification.get("passed"))
            record("file write verified", "PASS", "file created" + (" and read back" if verified else ""), t)
        else:
            record("file write verified", "WARN", f"no file was written: {reply.text[:100]!r}", t)

        # memory → model context
        t = time.monotonic()
        await orch.handle("Remember that the diagnostic code word is aubergine")
        reply = await orch.handle("What is the diagnostic code word? Answer with just the word.")
        record("memory reaches the model", "PASS" if "aubergine" in reply.text.lower() else "WARN",
               f"replied {reply.text[:80]!r}", t)

        # consequential action stops for approval; declining leaves the file alone
        t = time.monotonic()
        probe = workdir / "keep-me.txt"
        probe.write_text("do not delete")
        reply = await orch.handle(f"Delete the file {probe}")
        if reply.kind == "question" and "approval" in reply.text.lower():
            declined = await orch.handle("no")
            await _settle(svc)
            status = "PASS" if probe.exists() else "FAIL"
            record("approval gate", status, "deletion paused for approval; declined; file kept"
                   if probe.exists() else f"file deleted despite declining ({declined.text[:60]!r})", t)
        elif not probe.exists():
            record("approval gate", "FAIL", "the file was deleted without approval", t)
        else:
            record("approval gate", "WARN", f"the model didn't attempt the deletion: {reply.text[:100]!r}", t)

        # background work becomes a durable task
        t = time.monotonic()
        before_tasks = {task.id for task in svc.tasks.list_tasks(limit=500)}
        reply = await orch.handle("Run the shell command `echo jarvis-live-check` as a background task.")
        await _settle(svc)
        created = [task for task in svc.tasks.list_tasks(limit=500) if task.id not in before_tasks
                   and any(s.tool == "shell_execute" for s in task.plan)]
        if created:
            task = await svc.pool.wait_for(created[0].id, timeout=60)
            ok = task.status == TaskStatus.COMPLETED
            record("background task", "PASS" if ok else "WARN",
                   f"task {task.id} {task.status.value}: {task.outputs.get('summary', '')}"[:120], t)
        else:
            record("background task", "WARN", f"no task was created: {reply.text[:100]!r}", t)
        return results
    finally:
        await runtime.stop()


async def _settle(svc: object, timeout: float = 30.0) -> None:
    """Wait for any work the last request started (tasks run in the worker pool)."""
    try:
        await svc.pool.wait_idle(timeout=timeout)  # type: ignore[attr-defined]
    except TimeoutError:
        pass
    await svc.bus.drain()  # type: ignore[attr-defined]
