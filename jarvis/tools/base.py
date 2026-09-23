"""Tool contract (spec §53).

Every tool declares what it does, what it needs, what it can break and how its
result is verified. The registry — not the tool — enforces authority.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from jarvis.clock import Clock, SystemClock
from jarvis.core.types import Outcome, Provenance, ProvenanceKind, RiskLevel
from jarvis.permissions.model import Actor, PermissionLevel
from jarvis.security.secrets import SecretStore


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    level: PermissionLevel
    risk: RiskLevel = RiskLevel.NONE
    side_effects: tuple[str, ...] = ()
    reversible: bool = True
    idempotent: bool = False          # safe to retry automatically
    requires_network: bool = False
    resources: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ("linux", "darwin", "win32")
    timeout_s: float = 60.0
    verification: str = ""            # how success is confirmed
    path_params: tuple[str, ...] = () # arguments that are filesystem paths (scope-checked)
    long_running: bool = False
    category: str = "general"

    def supported(self) -> bool:
        return any(sys.platform.startswith(p) for p in self.platforms)

    def model_schema(self) -> dict[str, Any]:
        """Function-calling schema in the common (Ollama/OpenAI) format."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass
class Assessment:
    """Risk evaluated for *these* arguments (e.g. ``rm -rf`` vs ``ls``)."""

    level: PermissionLevel
    risk: RiskLevel
    reason: str = ""
    requires_network: bool = False
    blocked: bool = False             # absolute safety constraint — never executable
    reversible: bool = True


@dataclass
class ExecutionPolicy:
    """Environmental constraints from mission mode and the active project."""

    network_allowed: bool = True
    network_block_reason: str = ""
    allowed_tools: list[str] | None = None     # None = no project restriction
    allowed_dirs: list[str] | None = None
    project_id: str | None = None
    dry_run_only: bool = False


@dataclass
class ToolContext:
    actor: Actor
    task_id: str | None = None
    cwd: str | None = None
    dry_run: bool = False
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    clock: Clock = field(default_factory=SystemClock)
    secrets: SecretStore = field(default_factory=SecretStore)
    progress: Callable[[float, str], None] | None = None
    data_dir: str | None = None
    step_id: str | None = None


@dataclass
class ToolResult:
    ok: bool
    summary: str
    data: Any = None
    error: str | None = None
    exit_code: int | None = None
    rollback: dict[str, Any] | None = None
    provenance: Provenance | None = None
    outcome: Outcome | None = None

    def __post_init__(self) -> None:
        if self.outcome is None:
            self.outcome = Outcome.COMPLETE if self.ok else Outcome.FAILED

    def to_dict(self, max_data_chars: int = 4000) -> dict[str, Any]:
        data = self.data
        text = repr(data) if not isinstance(data, (dict, list, str, int, float, bool, type(None))) else data
        if isinstance(text, str) and len(text) > max_data_chars:
            text = text[:max_data_chars] + "…[truncated]"
        return {"ok": self.ok, "summary": self.summary, "data": text, "error": self.error,
                "exit_code": self.exit_code, "outcome": self.outcome.value if self.outcome else None}


@dataclass
class Verification:
    performed: bool
    passed: bool | None = None
    method: str = ""
    detail: str = ""

    @classmethod
    def not_performed(cls, why: str = "no verification strategy") -> "Verification":
        return cls(False, None, "none", why)

    def to_dict(self) -> dict[str, Any]:
        return {"performed": self.performed, "passed": self.passed, "method": self.method, "detail": self.detail}


class Tool:
    spec: ToolSpec

    def assess(self, args: dict[str, Any]) -> Assessment:
        return Assessment(self.spec.level, self.spec.risk, requires_network=self.spec.requires_network,
                          reversible=self.spec.reversible)

    def preview(self, args: dict[str, Any]) -> str:
        shown = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
        return f"{self.spec.name}({shown})"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise NotImplementedError

    async def verify(self, args: dict[str, Any], result: ToolResult, ctx: ToolContext) -> Verification:
        return Verification.not_performed()

    async def rollback(self, info: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(False, "rollback not supported", error="unsupported")


def tool_output(data: Any, summary: str, source: str, *, ok: bool = True, **kw: Any) -> ToolResult:
    return ToolResult(ok, summary, data, provenance=Provenance(ProvenanceKind.TOOL_OUTPUT, source), **kw)


def _short(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"
