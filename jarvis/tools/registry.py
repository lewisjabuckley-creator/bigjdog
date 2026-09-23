"""Unified tool registry: the single gate between intent and side effects.

``execute`` is the only path by which any model, agent, task or automation
causes an effect. It validates arguments, evaluates risk for the specific
arguments, applies safety and scope constraints, checks authority (capability is
not permission), honours dry runs, runs with timeout and cancellation, verifies
the result independently where possible, and writes the audit trail.
"""

from __future__ import annotations

import asyncio
import fnmatch
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

from jarvis.audit.log import AuditLog
from jarvis.core.types import OperationalReason, Outcome, Severity
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.log import get_logger
from jarvis.permissions.manager import PermissionManager
from jarvis.permissions.model import AccessRequest, Decision
from jarvis.tools.base import Assessment, ExecutionPolicy, Tool, ToolContext, ToolResult, Verification
from jarvis.tools.schema import SchemaError, validate_args

log = get_logger("tools")


class ExecStatus(StrEnum):
    EXECUTED = "executed"
    DRY_RUN = "dry_run"
    NEEDS_APPROVAL = "needs_approval"
    DENIED = "denied"
    INVALID = "invalid"
    UNKNOWN_TOOL = "unknown_tool"
    FAILED = "failed"          # raised, timed out, or unsupported platform
    CANCELLED = "cancelled"


@dataclass
class Execution:
    status: ExecStatus
    tool: str
    args: dict[str, Any]
    message: str = ""
    assessment: Assessment | None = None
    decision: Decision | None = None
    result: ToolResult | None = None
    verification: Verification | None = None
    audit_id: str | None = None
    preview: str = ""
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return (self.status == ExecStatus.EXECUTED and self.result is not None and self.result.ok
                and (self.verification is None or self.verification.passed is not False))

    @property
    def verified(self) -> bool:
        return bool(self.verification and self.verification.performed and self.verification.passed)

    def describe(self) -> str:
        if self.status == ExecStatus.EXECUTED and self.result is not None:
            text = self.result.summary
            if self.verification and self.verification.performed:
                text += " (verified)" if self.verification.passed else f" (verification failed: {self.verification.detail})"
            return text
        return self.message

    def for_model(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status.value, "tool": self.tool}
        if self.result is not None:
            out.update(self.result.to_dict())
        if self.verification is not None:
            out["verification"] = self.verification.to_dict()
        if self.message:
            out["message"] = self.message
        return out


PolicyProvider = Callable[[], ExecutionPolicy]


class ToolRegistry:
    def __init__(self, permissions: PermissionManager, audit: AuditLog, *, bus: EventBus | None = None,
                 policy_provider: PolicyProvider | None = None) -> None:
        self.permissions = permissions
        self.audit = audit
        self.bus = bus
        self.policy_provider = policy_provider or ExecutionPolicy
        self._tools: dict[str, Tool] = {}
        self.stats: dict[str, dict[str, int]] = {}

    # -- registry ---------------------------------------------------------------
    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"tool {tool.spec.name!r} already registered")
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self, *, category: str | None = None) -> list[Tool]:
        return [t for t in self._tools.values()
                if t.spec.supported() and (category is None or t.spec.category == category)]

    def model_schemas(self, names: list[str] | None = None, *, max_level: int | None = None) -> list[dict[str, Any]]:
        policy = self.policy_provider()
        tools = []
        for tool in self.list():
            if names is not None and tool.spec.name not in names:
                continue
            if max_level is not None and tool.spec.level > max_level:
                continue
            if tool.spec.requires_network and not policy.network_allowed:
                continue
            if policy.allowed_tools is not None and not _allowed(tool.spec.name, policy.allowed_tools):
                continue
            tools.append(tool.spec.model_schema())
        return tools

    # -- authorization ----------------------------------------------------------
    def authorize(self, tool: Tool, args: dict[str, Any], ctx: ToolContext,
                  policy: ExecutionPolicy) -> tuple[Assessment, Decision, list[str]]:
        assessment = tool.assess(args)
        paths = [self.permissions.paths.resolve(str(args[p]), ctx.cwd) for p in tool.spec.path_params
                 if args.get(p) not in (None, "")]
        if assessment.blocked:
            return assessment, Decision(False, reason=f"blocked by safety policy: {assessment.reason}",
                                        basis="safety"), paths
        if (assessment.requires_network or tool.spec.requires_network) and not policy.network_allowed:
            return assessment, Decision(False, reason=policy.network_block_reason or "network access is disabled",
                                        basis="mode"), paths
        if policy.allowed_tools is not None and not _allowed(tool.spec.name, policy.allowed_tools):
            return assessment, Decision(False, reason=f"{tool.spec.name} is not allowed in the active project",
                                        basis="scope"), paths
        granted = self.permissions.granted_paths(ctx.actor, tool.spec.name, ctx.task_id)
        for path in paths:
            ok, why = self.permissions.paths.check(path, allowed_dirs=policy.allowed_dirs, granted_paths=granted)
            if not ok:
                return assessment, Decision(False, reason=why, basis="scope"), paths
        request = AccessRequest(ctx.actor, tool.spec.name, assessment.level, paths, policy.project_id, ctx.task_id,
                                tool.preview(args))
        return assessment, self.permissions.check(request, consume=not ctx.dry_run), paths

    # -- execution --------------------------------------------------------------
    async def execute(self, name: str, args: dict[str, Any] | None, ctx: ToolContext, *,
                      reason: OperationalReason | None = None, policy: ExecutionPolicy | None = None,
                      timeout_s: float | None = None) -> Execution:
        tool = self._tools.get(name)
        raw_args = dict(args or {})
        if tool is None:
            return Execution(ExecStatus.UNKNOWN_TOOL, name, raw_args, f"no tool named {name!r}")
        if not tool.spec.supported():
            return Execution(ExecStatus.FAILED, name, raw_args, f"{name} is not supported on this platform")
        try:
            valid = validate_args(tool.spec.parameters, raw_args)
        except SchemaError as exc:
            return Execution(ExecStatus.INVALID, name, raw_args, f"invalid arguments: {exc}")

        policy = policy or self.policy_provider()
        dry_run = ctx.dry_run or policy.dry_run_only
        assessment, decision, _paths = self.authorize(tool, valid, ctx, policy)
        preview = tool.preview(valid)

        if not decision.allowed and not decision.needs_approval:
            entry = self.audit.record(actor=ctx.actor.subject, action="tool_denied", tool=name, task_id=ctx.task_id,
                                      params=valid, ok=False, summary=decision.reason,
                                      authorization=decision.to_dict(), reason=reason)
            if decision.basis == "safety" and self.bus:
                self.bus.emit(Event(EventType.SECURITY_EVENT, "tools",
                                    {"tool": name, "reason": decision.reason, "actor": ctx.actor.subject},
                                    severity=Severity.WARNING, task_id=ctx.task_id))
            return Execution(ExecStatus.DENIED, name, valid, decision.reason, assessment, decision,
                             audit_id=entry.id, preview=preview)
        if dry_run:
            entry = self.audit.record(actor=ctx.actor.subject, action="tool_dry_run", tool=name, task_id=ctx.task_id,
                                      params=valid, summary=f"would run: {preview}",
                                      authorization=decision.to_dict(), reason=reason, dry_run=True)
            note = " (would require approval)" if decision.needs_approval else ""
            return Execution(ExecStatus.DRY_RUN, name, valid, f"dry run: {preview}{note}", assessment, decision,
                             audit_id=entry.id, preview=preview)
        if decision.needs_approval:
            return Execution(ExecStatus.NEEDS_APPROVAL, name, valid, decision.reason, assessment, decision,
                             preview=preview)

        started = time.monotonic()
        status = ExecStatus.EXECUTED
        result: ToolResult | None
        try:
            result = await self._run_with_limits(tool, valid, ctx, timeout_s or float(valid.get("timeout_s") or 0)
                                                 or tool.spec.timeout_s)
        except asyncio.CancelledError:
            if ctx.cancel.is_set():
                status, result = ExecStatus.CANCELLED, ToolResult(False, f"{name} cancelled", error="cancelled",
                                                                  outcome=Outcome.UNKNOWN)
            else:
                raise
        except asyncio.TimeoutError:
            status, result = ExecStatus.FAILED, ToolResult(False, f"{name} timed out", error="timeout")
        except Exception as exc:  # tool bug or environment failure — report, never hide
            log.error("tool_crashed", tool=name, error=repr(exc))
            status, result = ExecStatus.FAILED, ToolResult(False, f"{name} failed: {exc}", error=repr(exc))
        duration = time.monotonic() - started

        verification: Verification | None = None
        if status == ExecStatus.EXECUTED and result is not None:
            try:
                verification = await tool.verify(valid, result, ctx)
            except Exception as exc:
                verification = Verification(True, False, "verify", f"verification raised: {exc}")
            if verification.performed and verification.passed is False:
                result.outcome = Outcome.FAILED
                if self.bus:
                    self.bus.emit(Event(EventType.VERIFICATION_FAILED, "tools",
                                        {"tool": name, "detail": verification.detail, "claimed": result.summary},
                                        severity=Severity.WARNING, task_id=ctx.task_id))

        entry = self.audit.record(
            actor=ctx.actor.subject, action="tool_execute", tool=name, task_id=ctx.task_id, params=valid,
            ok=bool(result and result.ok and (verification is None or verification.passed is not False)),
            outcome=result.outcome.value if result and result.outcome else None,
            summary=result.summary if result else "", verification=verification.to_dict() if verification else None,
            authorization=decision.to_dict(), reason=reason, rollback=result.rollback if result else None,
            duration_s=round(duration, 4))
        execution = Execution(status, name, valid, result.summary if result else "", assessment, decision, result,
                              verification, entry.id, preview, duration)
        self._count(name, execution)
        if self.bus:
            etype = EventType.TOOL_EXECUTED if execution.ok else EventType.TOOL_FAILED
            self.bus.emit(Event(etype, "tools", {"tool": name, "summary": execution.describe(), "audit_id": entry.id,
                                                 "status": status.value},
                                severity=Severity.INFO if execution.ok else Severity.WARNING, task_id=ctx.task_id))
        return execution

    async def _run_with_limits(self, tool: Tool, args: dict[str, Any], ctx: ToolContext, timeout: float) -> ToolResult:
        run_task = asyncio.ensure_future(tool.run(args, ctx))
        cancel_task = asyncio.ensure_future(ctx.cancel.wait())
        try:
            done, _ = await asyncio.wait({run_task, cancel_task}, timeout=timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
            if run_task in done:
                return run_task.result()
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, Exception):
                pass
            if cancel_task in done:
                raise asyncio.CancelledError()
            raise asyncio.TimeoutError()
        finally:
            cancel_task.cancel()
            if not run_task.done():  # outer cancellation: never leave the tool running unobserved
                run_task.cancel()

    async def rollback(self, audit_id: str, ctx: ToolContext) -> ToolResult:
        entry = self.audit.get(audit_id)
        if entry is None or not entry.rollback or not entry.tool:
            return ToolResult(False, "no rollback information recorded for that action", error="no_rollback")
        tool = self._tools.get(entry.tool)
        if tool is None:
            return ToolResult(False, f"tool {entry.tool} is no longer available", error="unknown_tool")
        result = await tool.rollback(entry.rollback, ctx)
        self.audit.record(actor=ctx.actor.subject, action="rollback", tool=entry.tool, task_id=entry.task_id,
                          params={"audit_id": audit_id}, ok=result.ok, summary=result.summary)
        return result

    def _count(self, name: str, execution: Execution) -> None:
        stats = self.stats.setdefault(name, {"ok": 0, "failed": 0})
        stats["ok" if execution.ok else "failed"] += 1


def _allowed(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)
