"""Specialised agents under JARVIS's supervision (spec §50-52, §130-133, §170).

An agent is a bounded model-driven loop with a role, a tool allowlist, a
permission ceiling, a model purpose, budgets and a structured output contract.
JARVIS remains the orchestrator:

* agents cannot spawn agents (the delegation tool is never in their allowlist);
* concurrent agent runs are capped;
* every tool call goes through the registry as ``agent:<name>`` and is refused
  above the agent's permission ceiling, whatever the tool could do;
* budgets (steps, model calls, wall time) end runaway loops;
* agent output is evidence, not truth: a claim of verification by the agent is
  recorded as such, and JARVIS independently checks what it can (artifacts).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from jarvis.core.types import OperationalReason, Provenance, ProvenanceKind
from jarvis.models.base import ChatMessage, ModelError, Purpose
from jarvis.models.router import ModelRouter, TaskProfile
from jarvis.permissions.model import Actor, PermissionLevel
from jarvis.planner.planner import _extract_json
from jarvis.tools.base import Tool, ToolContext, ToolResult, ToolSpec, Verification
from jarvis.tools.registry import ExecStatus, ToolRegistry

DELEGATION_TOOL = "delegate_to_agent"


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    tools: tuple[str, ...]
    max_level: PermissionLevel = PermissionLevel.OBSERVE
    purpose: Purpose = Purpose.REASONING
    max_steps: int = 8
    max_model_calls: int = 10
    max_seconds: float = 300.0


@dataclass
class AgentResult:
    status: str                      # completed | failed | budget_exhausted | model_unavailable
    summary: str
    findings: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0
    verification: dict[str, Any] = field(default_factory=lambda: {"performed": False, "result": "none", "by": "agent"})
    tool_calls: int = 0
    model_calls: int = 0
    refused_calls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CONTRACT = """When you are finished, reply with JSON only:
{"status": "completed" | "failed", "summary": "...", "findings": ["..."], "artifacts": ["paths you created"],
 "warnings": ["..."], "confidence": 0.0-1.0, "verification": {"performed": true|false, "result": "..."}}
Only report what your tool results show. Do not invent results."""


BUILTIN_AGENTS = [
    AgentSpec("research", "You are JARVIS's research agent. Investigate the question using local files and "
              "system information; cross-check important claims.",
              ("file_list", "file_read", "file_search", "system_info", "search_memory"), PermissionLevel.OBSERVE,
              Purpose.REASONING),
    AgentSpec("testing", "You are JARVIS's testing agent. Run and analyse the project's tests; classify failures.",
              ("file_list", "file_read", "file_search", "shell_execute"), PermissionLevel.EXECUTE_REVERSIBLE,
              Purpose.CODING),
    AgentSpec("documentation", "You are JARVIS's documentation agent. Read the code and draft or update "
              "documentation files.", ("file_list", "file_read", "file_search", "file_write"),
              PermissionLevel.EXECUTE_REVERSIBLE, Purpose.SUMMARIZATION),
    AgentSpec("system", "You are JARVIS's system agent. Inspect processes and resources to explain system "
              "behaviour. You only observe.", ("system_info", "process_list", "process_inspect", "file_read"),
              PermissionLevel.OBSERVE, Purpose.REASONING),
]


class AgentRegistry:
    def __init__(self, specs: list[AgentSpec] | None = None) -> None:
        self.specs: dict[str, AgentSpec] = {}
        for spec in specs if specs is not None else BUILTIN_AGENTS:
            self.register(spec)

    def register(self, spec: AgentSpec) -> None:
        if DELEGATION_TOOL in spec.tools:
            raise ValueError("agents may not delegate to other agents")
        self.specs[spec.name] = spec

    def get(self, name: str) -> AgentSpec | None:
        return self.specs.get(name)

    def list(self) -> list[AgentSpec]:
        return list(self.specs.values())


class AgentRunner:
    def __init__(self, registry: ToolRegistry, router: ModelRouter, *, max_concurrent: int = 2) -> None:
        self.registry = registry
        self.router = router
        self._slots = asyncio.Semaphore(max_concurrent)

    async def run(self, spec: AgentSpec, objective: str, ctx: ToolContext, context: str = "") -> AgentResult:
        async with self._slots:
            return await self._run(spec, objective, ctx, context)

    async def _run(self, spec: AgentSpec, objective: str, ctx: ToolContext, context: str) -> AgentResult:
        started = time.monotonic()
        allowed = [t for t in spec.tools if t != DELEGATION_TOOL]
        schemas = self.registry.model_schemas(allowed, max_level=spec.max_level)
        agent_ctx = ToolContext(actor=Actor("agent", spec.name, ctx.actor.on_behalf_of, interactive=False),
                                task_id=ctx.task_id, cwd=ctx.cwd, dry_run=ctx.dry_run, cancel=ctx.cancel,
                                clock=ctx.clock, data_dir=ctx.data_dir)
        messages = [ChatMessage("system", f"{spec.role}\nUse only the tools provided.\n{_CONTRACT}"),
                    ChatMessage("user", f"Objective: {objective}\n{context}".strip())]
        result = AgentResult("failed", "")
        profile = TaskProfile(purpose=spec.purpose, complexity="medium", needs_tools=bool(schemas), interactive=False)
        while True:
            if ctx.cancel.is_set():
                result.status, result.summary = "failed", "cancelled"
                return result
            if result.model_calls >= spec.max_model_calls or result.tool_calls >= spec.max_steps or \
                    time.monotonic() - started > spec.max_seconds:
                result.status = "budget_exhausted"
                result.summary = result.summary or "stopped: budget exhausted before a conclusion"
                return result
            try:
                routed = await self.router.chat(profile, messages, tools=schemas or None)
            except ModelError as exc:
                result.status, result.summary = "model_unavailable", f"no model available: {exc}"
                return result
            result.model_calls += 1
            calls = routed.response.tool_calls
            if not calls:
                return self._final(routed.response.content, result)
            messages.append(ChatMessage("assistant", routed.response.content, tool_calls=calls))
            for call in calls:
                result.tool_calls += 1
                if call.name not in allowed:
                    result.refused_calls.append(call.name)
                    payload: dict[str, Any] = {"status": "refused",
                                               "message": f"{call.name} is not available to the {spec.name} agent"}
                else:
                    tool = self.registry.get(call.name)
                    level = tool.assess(call.arguments).level if tool else PermissionLevel.AUTONOMOUS
                    if level > spec.max_level:
                        result.refused_calls.append(call.name)
                        payload = {"status": "refused", "message": f"exceeds the {spec.name} agent's permission "
                                                                   f"ceiling ({spec.max_level.label})"}
                    else:
                        execution = await self.registry.execute(
                            call.name, call.arguments, agent_ctx,
                            reason=OperationalReason(f"the {spec.name} agent is working on: {objective[:80]}",
                                                     "delegated work stays within the agent's scope",
                                                     f"ran {call.name}"))
                        payload = execution.for_model()
                        if execution.status in (ExecStatus.DENIED, ExecStatus.NEEDS_APPROVAL):
                            result.refused_calls.append(call.name)
                messages.append(ChatMessage("tool", json.dumps(payload, default=str)[:6000], name=call.name,
                                            tool_call_id=call.id))

    @staticmethod
    def _final(text: str, result: AgentResult) -> AgentResult:
        try:
            data = json.loads(_extract_json(text))
        except ValueError:
            data = None
        if not isinstance(data, dict):
            result.status = "completed"
            result.summary = text.strip()[:2000]
            result.confidence = 0.3
            result.warnings.append("the agent did not follow the output contract")
            return result
        result.status = "completed" if data.get("status", "completed") == "completed" else "failed"
        result.summary = str(data.get("summary", ""))[:2000]
        result.findings = [str(x) for x in data.get("findings", [])][:50]
        result.artifacts = [str(x) for x in data.get("artifacts", [])][:50]
        result.warnings = [str(x) for x in data.get("warnings", [])][:20]
        try:
            result.confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        except (TypeError, ValueError):
            result.confidence = 0.5
        claimed = data.get("verification") or {}
        # the agent's own claim is kept but attributed: it is evidence, not verification by JARVIS
        result.verification = {"performed": bool(claimed.get("performed")), "result": str(claimed.get("result", "")),
                               "by": "agent"}
        return result


class DelegateToAgentTool(Tool):
    spec = ToolSpec(
        name=DELEGATION_TOOL,
        description=("Delegate a bounded sub-task to a specialised agent (research, testing, documentation, system). "
                     "Returns the agent's structured findings, which are evidence to check, not verified facts."),
        parameters={"type": "object", "properties": {
            "agent": {"type": "string"}, "objective": {"type": "string"},
            "context": {"type": "string", "default": ""}}, "required": ["agent", "objective"]},
        level=PermissionLevel.PREPARE, verification="artifacts checked on disk; claims treated as evidence",
        long_running=True, category="agents",
    )

    def __init__(self, agents: AgentRegistry, runner: AgentRunner) -> None:
        self.agents = agents
        self.runner = runner

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        spec = self.agents.get(args["agent"])
        if spec is None:
            return ToolResult(False, f"no agent named {args['agent']!r}; available: "
                                     f"{', '.join(a.name for a in self.agents.list())}", error="unknown_agent")
        if ctx.actor.kind == "agent":
            return ToolResult(False, "agents cannot delegate to other agents", error="recursion_refused")
        result = await self.runner.run(spec, args["objective"], ctx, args.get("context", ""))
        ok = result.status == "completed"
        return ToolResult(ok, f"{spec.name} agent {result.status}: {result.summary[:200]}", result.to_dict(),
                          error=None if ok else result.status,
                          provenance=Provenance(ProvenanceKind.INFERENCE, f"{spec.name} agent", "unverified evidence"))

    async def verify(self, args: dict[str, Any], result: ToolResult, ctx: ToolContext) -> Verification:
        data = result.data or {}
        artifacts = data.get("artifacts") or []
        if not artifacts:
            return Verification(False, None, "evidence",
                                "agent findings are evidence, not independently verified")
        base = ctx.cwd or os.getcwd()
        missing = [a for a in artifacts if not os.path.exists(a if os.path.isabs(a) else os.path.join(base, a))]
        if missing:
            return Verification(True, False, "artifact_check", f"claimed artifacts missing: {', '.join(missing[:5])}")
        return Verification(True, True, "artifact_check", f"{len(artifacts)} claimed artifact(s) exist")
