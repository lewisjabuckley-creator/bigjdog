"""Offline unit tests for the real-model conversation path (streaming, readiness, argument hygiene,
model fallback, Ollama request options). Live counterparts are in tests/integration."""

from __future__ import annotations

import json

import httpx

from jarvis.config import ModelsConfig, config_from_dict
from jarvis.core.intent import IntentKind, parse
from jarvis.core.orchestrator import _StreamCleaner
from jarvis.models.base import Capability, ChatMessage, ModelInfo, ToolCall
from jarvis.models.fake import ScriptedProvider
from jarvis.models.ollama import OllamaProvider, strip_thinking
from jarvis.models.readiness import assess
from jarvis.models.router import ModelRouter
from jarvis.runtime import Runtime
from jarvis.simulation.environment import SimulatedEnvironment


# -- streaming filter -------------------------------------------------------------------------------

def test_stream_cleaner_drops_thinking_and_filler():
    out: list[str] = []
    cleaner = _StreamCleaner(out.append)
    for piece in ["<think>let me reason", " about this</think>", "Certainly! ", "The disk is ",
                  "62% full, which is fine for now."]:
        cleaner.feed(piece)
    cleaner.close()
    assert "".join(out) == "The disk is 62% full, which is fine for now."
    assert cleaner.emitted


def test_stream_cleaner_short_answer_flushes_on_close():
    out: list[str] = []
    cleaner = _StreamCleaner(out.append)
    cleaner.feed("Done.")
    assert out == []
    cleaner.close()
    assert out == ["Done."]


def test_strip_thinking():
    assert strip_thinking("<think>hmm</think>\nAnswer") == "Answer"
    assert strip_thinking("<think>unfinished") == ""
    assert strip_thinking("no tags") == "no tags"


# -- Ollama request options ------------------------------------------------------------------------

async def test_ollama_sends_context_window_and_options():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "m", "message": {"role": "assistant",
                                                                   "content": "<think>x</think>ok"}})

    provider = OllamaProvider(transport=httpx.MockTransport(handler), default_options={"num_ctx": 8192,
                                                                                      "temperature": 0.2})
    reply = await provider.chat("m", [ChatMessage("user", "hi")], options={"temperature": 0.0})
    assert bodies[0]["options"] == {"num_ctx": 8192, "temperature": 0.0}     # per-call overrides defaults
    assert reply.content == "ok"
    await provider.chat("m", [ChatMessage("assistant", "", tool_calls=[ToolCall("t", {"a": 1}, "call_1")]),
                              ChatMessage("tool", "{}", name="t", tool_call_id="call_1")])
    sent = bodies[1]["messages"]
    assert sent[0]["tool_calls"][0] == {"id": "call_1", "function": {"index": 0, "name": "t", "arguments": {"a": 1}}}
    assert sent[1]["tool_call_id"] == "call_1" and sent[1]["tool_name"] == "t"
    await provider.close()


def test_local_ollama_bypasses_system_proxies():
    assert OllamaProvider("http://127.0.0.1:11434")._client._trust_env is False
    assert OllamaProvider("http://gpu-box.lan:11434")._client._trust_env is True


def test_config_exposes_ollama_context_and_options():
    cfg = config_from_dict({"models": {"ollama": {"num_ctx": 16384, "options": {"temperature": 0.3}}}})
    assert cfg.models.ollama.num_ctx == 16384 and cfg.models.ollama.options == {"temperature": 0.3}


# -- readiness guidance ------------------------------------------------------------------------------

async def test_readiness_guidance():
    down = ScriptedProvider("ollama")
    down.online = False
    router = ModelRouter([down], ModelsConfig(profiles={}))
    await router.refresh()
    r = assess(router, "http://127.0.0.1:11434")
    assert not r.can_converse and "Ollama isn't running at http://127.0.0.1:11434" in r.issues[0]

    empty = ModelRouter([ScriptedProvider("ollama", models=[])], ModelsConfig(profiles={}))
    await empty.refresh()
    r = assess(empty)
    assert "no chat model is installed" in r.issues[0] and "ollama pull" in r.issues[0]

    no_tools = ScriptedProvider("ollama", models=[ModelInfo("gemma2:2b", "ollama", True, frozenset({Capability.CHAT}))])
    router = ModelRouter([no_tools], ModelsConfig(profiles={}))
    await router.refresh()
    r = assess(router)
    assert r.can_converse and not r.tools and "can't call tools" in r.issues[0]
    assert r.summary() == "gemma2:2b (local, no tool use)"

    good = ModelRouter([ScriptedProvider("ollama")], ModelsConfig(profiles={}))
    await good.refresh()
    r = assess(good)
    assert r.can_converse and r.tools and not r.issues and r.embedding_model == "sim-embed"


# -- orchestrator behaviour with a model ------------------------------------------------------------------

async def _runtime(tmp_path):
    sim = SimulatedEnvironment()
    cfg = config_from_dict({"general": {"data_dir": str(tmp_path / "d")}, "monitoring": {"enabled": False},
                            "permissions": {"allowed_roots": [str(tmp_path)]}, "tasks": {"scheduler_interval_s": 0.02}})
    rt = Runtime(cfg, providers=[sim.provider], metrics=sim.metrics, network_probe=sim.probe, simulated=True)
    await rt.start(monitoring=False)
    return rt, sim


async def test_streaming_reaches_the_interface(tmp_path):
    rt, sim = await _runtime(tmp_path)
    try:
        sim.provider.when("tell me something", "Absolutely! Here is a fact about disks.")
        tokens: list[str] = []
        reply = await rt.orchestrator().handle("tell me something", on_token=tokens.append)
        assert reply.streamed and "".join(tokens) == "Here is a fact about disks." == reply.text
    finally:
        await rt.stop()


async def test_grammar_misfires_fall_through_to_the_model(tmp_path):
    rt, sim = await _runtime(tmp_path)
    try:
        sim.provider.when("weather", "I can't check the weather from here.")
        sim.provider.when("calculator", "I can't open desktop applications yet.")
        orch = rt.orchestrator()
        assert (await orch.handle("How's the weather?")).text == "I can't check the weather from here."
        assert (await orch.handle("Open the calculator")).text == "I can't open desktop applications yet."
        sim.provider.online = False            # without a model the deterministic answer remains
        assert "I'm not tracking anything called" in (await orch.handle("How's the weather?")).text
    finally:
        await rt.stop()


async def test_model_tool_arguments_are_cleaned_and_irrelevant_tools_hidden(tmp_path):
    rt, sim = await _runtime(tmp_path)
    try:
        (tmp_path / "a.txt").write_text("alpha")
        sim.provider.when("read the note", tool_calls=[ToolCall("file_read", {"path": str(tmp_path / "a.txt"),
                                                                              "encoding": "utf8", "why": None})],
                          once=True)
        sim.provider.when("alpha", "It says alpha.")
        reply = await rt.orchestrator().handle("read the note")
        assert reply.text == "It says alpha."
        read = rt.svc.audit.query(action="tool_execute")[0]
        assert read.tool == "file_read" and read.ok and set(read.params) == {"path", "max_bytes"}
        offered = {t["function"]["name"] for t in sim.provider.calls[0]["tools"]}
        assert "file_read" in offered and "device_command" not in offered and "delegate_to_agent" not in offered
    finally:
        await rt.stop()


async def test_greeting_names_the_conversation_model(tmp_path):
    rt, sim = await _runtime(tmp_path)
    try:
        readiness = rt.svc.extra["model_readiness"]
        assert readiness.can_converse and readiness.tools
    finally:
        await rt.stop()


def test_for_me_is_not_a_target():
    intent = parse("Could you run the test suite for me?")
    assert intent.kind == IntentKind.RUN_TESTS and intent.params.get("target") is None
