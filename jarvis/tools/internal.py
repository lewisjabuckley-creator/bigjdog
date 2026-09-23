"""Tools that expose JARVIS's own subsystems to the language model.

They go through the same registry as every other tool, so they are audited and
permission-checked like anything else.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from jarvis.core.types import Priority, Provenance, ProvenanceKind
from jarvis.memory.store import MemoryKind, MemoryStore
from jarvis.permissions.model import PermissionLevel
from jarvis.tools.base import Tool, ToolContext, ToolResult, ToolSpec

TaskStarter = Callable[[str, list[dict[str, Any]] | None, Priority, bool], Awaitable[dict[str, Any]]]
StateReader = Callable[[str], dict[str, Any]]


class StartTaskTool(Tool):
    spec = ToolSpec(
        name="start_background_task",
        description=("Start a durable background task for long-running or multi-step work. Optionally give explicit "
                     "steps (each a tool call); otherwise JARVIS plans them. The user is notified when it finishes."),
        parameters={"type": "object", "properties": {
            "objective": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "object"}, "default": []},
            "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"], "default": "P2"},
            "notify_when_done": {"type": "boolean", "default": True},
        }, "required": ["objective"]},
        level=PermissionLevel.PREPARE, verification="task record created", category="jarvis",
    )

    def __init__(self, starter: TaskStarter) -> None:
        self.starter = starter

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        info = await self.starter(args["objective"], args.get("steps") or None, Priority[args["priority"]],
                                  args["notify_when_done"])
        if info.get("error"):
            return ToolResult(False, info["error"], info, error="rejected")
        return ToolResult(True, f"started task {info['id']}: {info['title']}", info,
                          provenance=Provenance(ProvenanceKind.DATABASE, "tasks"))


class RememberTool(Tool):
    spec = ToolSpec(
        name="remember",
        description="Store a durable fact, preference or procedure the user wants remembered.",
        parameters={"type": "object", "properties": {
            "content": {"type": "string"},
            "kind": {"type": "string", "enum": [k.value for k in MemoryKind], "default": "semantic"},
        }, "required": ["content"]},
        level=PermissionLevel.PREPARE, reversible=True, verification="memory stored", category="jarvis",
    )

    def __init__(self, memory: MemoryStore, project_id: Callable[[], str | None]) -> None:
        self.memory = memory
        self.project_id = project_id

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        kind = MemoryKind(args["kind"])
        item = await self.memory.remember(args["content"], kind=kind,
                                          project_id=self.project_id() if kind == MemoryKind.PROJECT else None)
        if item is None:
            return ToolResult(False, "memory is suppressed for this conversation ('don't remember this')",
                              error="suppressed")
        return ToolResult(True, "remembered", {"memory_id": item.id})


class SearchMemoryTool(Tool):
    spec = ToolSpec(
        name="search_memory",
        description="Search JARVIS's memory (facts, preferences, project knowledge, past events).",
        parameters={"type": "object", "properties": {"query": {"type": "string"},
                                                     "limit": {"type": "integer", "default": 5}},
                    "required": ["query"]},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="read-only", category="jarvis",
    )

    def __init__(self, memory: MemoryStore, project_id: Callable[[], str | None]) -> None:
        self.memory = memory
        self.project_id = project_id

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        hits = await self.memory.retrieve(args["query"], project_id=self.project_id(), limit=args["limit"])
        data = [{"id": h.item.id, "kind": h.item.kind.value, "content": h.item.content, "score": round(h.score, 3)}
                for h in hits]
        return ToolResult(True, f"{len(data)} memory item(s)", data,
                          provenance=Provenance(ProvenanceKind.MEMORY, "memory search"))


class LiveStateTool(Tool):
    spec = ToolSpec(
        name="get_live_state",
        description="Read JARVIS's live state: resources, tasks, health, network, models or the active project.",
        parameters={"type": "object", "properties": {
            "section": {"type": "string", "enum": ["resources", "tasks", "health", "network", "models", "project",
                                                   "summary", "all"], "default": "summary"}}},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="read-only", category="jarvis",
    )

    def __init__(self, reader: StateReader) -> None:
        self.reader = reader

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(True, f"live state: {args['section']}", self.reader(args["section"]),
                          provenance=Provenance(ProvenanceKind.SYSTEM_STATE, "live state"))
