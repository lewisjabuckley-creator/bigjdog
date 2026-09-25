"""End-to-end conversations: natural language → real Ollama → JARVIS tools, tasks, permissions, memory,
verification → real Ollama → answer. Uses deterministic puppet models, so every assertion is exact."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from jarvis.config import config_from_dict
from jarvis.core.intent import IntentKind
from jarvis.diagnostics import live_checks
from jarvis.runtime import Runtime
from jarvis.tasks.models import TaskStatus


def live_config(tmp_path: Path, ollama_url: str, extra_roots: list[str] | None = None):
    return config_from_dict({
        "general": {"data_dir": str(tmp_path / "data")},
        "models": {"ollama": {"base_url": ollama_url, "num_ctx": 4096}},
        "permissions": {"allowed_roots": [str(tmp_path)] + (extra_roots or [])},
        "monitoring": {"enabled": False},
        "tasks": {"scheduler_interval_s": 0.05},
    })


@asynccontextmanager
async def live_jarvis(tmp_path: Path, ollama_url: str, model: str | None,
                      extra_roots: list[str] | None = None) -> AsyncIterator[tuple[Runtime, list[dict[str, Any]]]]:
    runtime = Runtime(live_config(tmp_path, ollama_url, extra_roots))       # real providers from config
    await runtime.start(monitoring=False)
    sent: list[dict[str, Any]] = []

    async def capture(request):                     # record exactly what JARVIS sends to Ollama
        if request.url.path == "/api/chat":
            sent.append(json.loads(request.content))

    provider = runtime.svc.router.providers["ollama"]
    provider._client.event_hooks["request"].append(capture)
    if model:
        runtime.svc.router.pin(model)
    try:
        yield runtime, sent
    finally:
        await runtime.stop()


async def test_request_reaches_a_tool_and_the_answer_comes_back(tmp_path, ollama_url, puppets):
    async with live_jarvis(tmp_path, ollama_url, puppets["sysinfo"]) as (rt, sent):
        reply = await rt.orchestrator().handle("How is this computer doing right now?")
        assert reply.intent == IntentKind.CHAT
        assert reply.text == "SYSINFO-ANSWER: the machine looks fine." and reply.model == puppets["sysinfo"]
        # the tool ran through the registry, as the user, and was audited
        runs = rt.svc.audit.query(action="tool_execute")
        assert [(e.tool, e.actor, e.ok) for e in runs] == [("system_info", "user:owner", True)]
        assert any(p.kind.value == "system_state" and p.source == "psutil" for p in reply.provenance)
        # what JARVIS actually sent: tool definitions, live state, then the tool result
        first, second = sent
        assert "system_info" in {t["function"]["name"] for t in first["tools"]}
        assert "device_command" not in {t["function"]["name"] for t in first["tools"]}   # irrelevant tools hidden
        assert first["options"]["num_ctx"] == 4096
        assert "LIVE STATE (observed just now)" in first["messages"][0]["content"]
        tool_msg = second["messages"][-1]
        assert tool_msg["role"] == "tool" and tool_msg["tool_name"] == "system_info"
        assert "memory_percent" in tool_msg["content"]


async def test_answers_stream_to_the_interface(tmp_path, ollama_url, puppets):
    async with live_jarvis(tmp_path, ollama_url, puppets["sysinfo"]) as (rt, sent):
        tokens: list[str] = []
        reply = await rt.orchestrator().handle("How is this computer doing?", on_token=tokens.append)
        assert reply.streamed and "".join(tokens) == reply.text
        assert all(body["stream"] for body in sent)


async def test_file_write_request_is_executed_and_verified(tmp_path, ollama_url, puppets):
    async with live_jarvis(tmp_path, ollama_url, puppets["write"], [str(puppets.dir)]) as (rt, _):
        reply = await rt.orchestrator().handle("Please write that note to a file for me.")
        assert reply.text == "WRITE-ANSWER: the file is written."
        assert puppets.write_target.read_text() == "written through ollama"
        write = rt.svc.audit.query(action="tool_execute")[0]
        assert write.tool == "file_write" and write.verification["passed"] is True   # read back from disk


async def test_consequential_request_waits_for_approval(tmp_path, ollama_url, puppets):
    target = puppets.delete_target
    target.parent.mkdir(parents=True, exist_ok=True)
    async with live_jarvis(tmp_path, ollama_url, puppets["delete"], [str(puppets.dir)]) as (rt, _):
        orch = rt.orchestrator()
        target.write_text("important")
        ask = await orch.handle("Get rid of that old file.")
        assert ask.kind == "question" and "approval" in ask.text and ask.task_id
        assert target.exists()                                   # nothing happened without authority
        assert len(rt.svc.approvals.pending()) == 1
        declined = await orch.handle("no")
        assert "won't" in declined.text
        await rt.svc.pool.wait_for(ask.task_id)
        assert target.exists()

        again = await orch.handle("Get rid of that old file.")
        assert again.kind == "question"
        await orch.handle("proceed")
        done = await rt.svc.pool.wait_for(again.task_id)
        assert done.status == TaskStatus.COMPLETED and not target.exists()
        executed = [e for e in rt.svc.audit.query(task_id=again.task_id) if e.action == "tool_execute"]
        assert executed[0].authorization["basis"] == "grant"     # authority came from the approval
        assert executed[0].verification["passed"] is True        # gone from disk, recoverable from trash


async def test_background_request_becomes_a_durable_task(tmp_path, ollama_url, puppets):
    async with live_jarvis(tmp_path, ollama_url, puppets["background"]) as (rt, _):
        reply = await rt.orchestrator().handle("Run the marker job in the background.")
        assert reply.text == "BACKGROUND-ANSWER: started."
        tasks = [t for t in rt.svc.tasks.list_tasks() if t.plan and t.plan[0].tool == "shell_execute"]
        assert len(tasks) == 1 and tasks[0].created_by == "user:owner"
        done = await rt.svc.pool.wait_for(tasks[0].id, timeout=60)
        assert done.status == TaskStatus.COMPLETED
        assert "puppet-background-marker" in done.plan[0].result["data"]["stdout"]


async def test_relative_working_directory_from_the_model_is_safe(tmp_path, ollama_url, puppets):
    # JARVIS itself runs from a folder outside the allowed roots (like D:\\JARVIS on Windows); a model's "."
    # must mean the user's work area, not JARVIS's launch folder, and must not block the task.
    async with live_jarvis(tmp_path, ollama_url, puppets["background_cwd"]) as (rt, _):
        reply = await rt.orchestrator().handle("Run the marker job in the background.")
        assert reply.text == "BACKGROUND-CWD-ANSWER: started."
        task = next(t for t in rt.svc.tasks.list_tasks() if t.plan and t.plan[0].tool == "shell_execute")
        done = await rt.svc.pool.wait_for(task.id, [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED],
                                          timeout=60)
        assert done.status == TaskStatus.COMPLETED, done.status_reason
        assert "puppet-cwd-marker" in done.plan[0].result["data"]["stdout"]


async def test_long_running_tool_runs_as_a_task(tmp_path, ollama_url, puppets):
    async with live_jarvis(tmp_path, ollama_url, puppets["shell"]) as (rt, sent):
        reply = await rt.orchestrator().handle("Echo the shell marker.")
        assert reply.text == "SHELL-ANSWER: the command ran."
        task = next(t for t in rt.svc.tasks.list_tasks() if t.plan and t.plan[0].tool == "shell_execute")
        assert task.status == TaskStatus.COMPLETED
        assert "puppet-shell-marker" in sent[-1]["messages"][-1]["content"]   # the task's result went back


async def test_sloppy_model_arguments_are_sanitised(tmp_path, ollama_url, puppets):
    async with live_jarvis(tmp_path, ollama_url, puppets["hallucinate"], [str(puppets.dir)]) as (rt, sent):
        reply = await rt.orchestrator().handle("Read that file.")
        assert reply.text == "HALLUCINATE-ANSWER: handled."
        read = rt.svc.audit.query(action="tool_execute")[0]
        assert read.tool == "file_read" and "bogus" not in read.params           # stray argument dropped
        assert read.ok is False and "does not exist" in read.summary             # honest failure fed back
        assert "does not exist" in sent[-1]["messages"][-1]["content"]


async def test_memory_and_live_state_reach_the_model(tmp_path, ollama_url, puppets):
    async with live_jarvis(tmp_path, ollama_url, puppets["text"]) as (rt, sent):
        orch = rt.orchestrator()
        noted = await orch.handle("Remember that the staging server listens on port 8443")
        assert noted.intent == IntentKind.REMEMBER and not sent          # deterministic: no model call
        reply = await orch.handle("Which port does the staging server use?")
        assert reply.text == "Plain answer from the model."
        system = sent[-1]["messages"][0]["content"]
        assert "staging server listens on port 8443" in system and "RELEVANT MEMORY" in system
        assert "LIVE STATE" in system and "Mode: normal" in system


async def test_unrecognised_grammar_targets_fall_through_to_the_model(tmp_path, ollama_url, puppets):
    async with live_jarvis(tmp_path, ollama_url, puppets["text"]) as (rt, sent):
        reply = await rt.orchestrator().handle("How's the weather looking?")
        assert reply.text == "Plain answer from the model." and sent        # not "I'm not tracking 'weather'"


async def test_unreachable_ollama_is_reported_plainly(tmp_path, puppets):
    async with live_jarvis(tmp_path, "http://127.0.0.1:9", None) as (rt, _):
        readiness = rt.svc.extra["model_readiness"]
        assert not readiness.can_converse and "Ollama isn't running" in readiness.issues[0]
        reply = await rt.orchestrator().handle("Tell me a joke.")
        assert reply.kind == "error" and "unavailable" in reply.text
        assert (await rt.orchestrator().handle("What are you doing?")).text     # the rest still works


async def test_live_doctor_contract_checks(tmp_path, ollama_url, puppets):
    results = await live_checks(live_config(tmp_path, ollama_url), model=puppets["text"], level="contract",
                                workdir=tmp_path / "doctor")
    assert [(r.name, r.status) for r in results] == [
        ("model runtime reachable", "PASS"), ("chat model installed", "PASS"), ("model answers", "PASS"),
        ("streaming", "PASS")]
