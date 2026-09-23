import json

import pytest

from jarvis.agents.base import AgentRegistry, AgentRunner, AgentSpec, DelegateToAgentTool, DELEGATION_TOOL
from jarvis.audit.log import AuditLog
from jarvis.config import PermissionsConfig
from jarvis.models.base import ChatResponse, ToolCall
from jarvis.models.fake import ScriptedProvider
from jarvis.models.router import ModelRouter
from jarvis.permissions.manager import PermissionManager
from jarvis.permissions.model import Actor, PermissionLevel
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin import register_builtin_tools
from jarvis.tools.registry import ExecStatus, ToolRegistry


def setup(db, tmp_path, script):
    perms = PermissionManager(db, PermissionsConfig(allowed_roots=[str(tmp_path)], denied_paths=[]))
    registry = ToolRegistry(perms, AuditLog(db))
    register_builtin_tools(registry)
    provider = ScriptedProvider()
    calls = iter(script)
    provider.responder = lambda model, messages, tools: next(calls)
    router = ModelRouter([provider])
    agents = AgentRegistry()
    runner = AgentRunner(registry, router)
    registry.register(DelegateToAgentTool(agents, runner))
    return registry, agents, runner, provider


def ctx(tmp_path, actor=None):
    return ToolContext(actor=actor or Actor.user(), cwd=str(tmp_path), data_dir=str(tmp_path / ".j"))


async def test_agent_uses_allowed_tools_and_returns_structured_evidence(db, tmp_path):
    (tmp_path / "notes.md").write_text("The API rate limit is 100 requests per minute.")
    final = {"status": "completed", "summary": "rate limit is 100 rpm", "findings": ["limit=100/min in notes.md"],
             "confidence": 0.9, "verification": {"performed": True, "result": "cross-checked"}}
    registry, agents, runner, provider = setup(db, tmp_path, [
        ChatResponse("", "m", "p", [ToolCall("file_search", {"path": str(tmp_path), "query": "rate limit"})]),
        ChatResponse(json.dumps(final), "m", "p")])
    await registry.execute("time_now", {}, ctx(tmp_path))
    ex = await registry.execute(DELEGATION_TOOL, {"agent": "research", "objective": "what is the rate limit?"},
                                ctx(tmp_path))
    assert ex.status == ExecStatus.EXECUTED and ex.result.ok
    data = ex.result.data
    assert data["findings"] == ["limit=100/min in notes.md"] and data["tool_calls"] == 1
    assert data["verification"]["by"] == "agent"               # the claim is attributed, not trusted
    assert ex.verification.performed is False                   # JARVIS did not verify it
    actors = {e.actor for e in AuditLog(db).query(action="tool_execute")}
    assert "agent:research" in actors


async def test_agent_cannot_exceed_its_permission_ceiling_or_allowlist(db, tmp_path):
    registry, agents, runner, provider = setup(db, tmp_path, [
        ChatResponse("", "m", "p", [ToolCall("file_write", {"path": str(tmp_path / "x"), "content": "x"}),
                                    ToolCall("shell_execute", {"command": "ls"})]),
        ChatResponse(json.dumps({"status": "completed", "summary": "done"}), "m", "p")])
    ex = await registry.execute(DELEGATION_TOOL, {"agent": "research", "objective": "tidy up"}, ctx(tmp_path))
    assert set(ex.result.data["refused_calls"]) == {"file_write", "shell_execute"}
    assert not (tmp_path / "x").exists()


async def test_budget_stops_runaway_agents(db, tmp_path):
    loop_forever = [ChatResponse("", "m", "p", [ToolCall("time_now", {})])] * 50
    registry, agents, runner, provider = setup(db, tmp_path, loop_forever)
    agents.register(AgentSpec("looper", "loops", ("time_now",), max_steps=3))
    ex = await registry.execute(DELEGATION_TOOL, {"agent": "looper", "objective": "spin"}, ctx(tmp_path))
    assert ex.result.data["status"] == "budget_exhausted" and ex.result.data["tool_calls"] == 3


async def test_no_recursive_delegation(db, tmp_path):
    registry, agents, runner, provider = setup(db, tmp_path, [])
    with pytest.raises(ValueError):
        agents.register(AgentSpec("spawner", "spawns", (DELEGATION_TOOL,)))
    agent_actor = Actor("agent", "research")
    ex = await registry.execute(DELEGATION_TOOL, {"agent": "research", "objective": "x"}, ctx(tmp_path, agent_actor))
    assert ex.status in (ExecStatus.EXECUTED, ExecStatus.DENIED)
    assert not ex.ok


async def test_claimed_artifacts_are_checked(db, tmp_path):
    final = {"status": "completed", "summary": "wrote docs", "artifacts": ["docs/API.md"]}
    registry, agents, runner, provider = setup(db, tmp_path, [ChatResponse(json.dumps(final), "m", "p")])
    ex = await registry.execute(DELEGATION_TOOL, {"agent": "documentation", "objective": "document the API"},
                                ctx(tmp_path))
    assert ex.verification.performed and ex.verification.passed is False   # the file was never written
    assert not ex.ok
