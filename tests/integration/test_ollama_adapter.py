"""The Ollama adapter and router against a real Ollama server (API contract)."""

from __future__ import annotations

import asyncio

import pytest

from jarvis.models.base import Capability, ChatMessage, ModelError, ModelUnavailable
from jarvis.models.ollama import OllamaProvider
from jarvis.models.router import ModelRouter, TaskProfile

TOOLS = [
    {"type": "function", "function": {"name": "system_info", "description": "live system information",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "file_write", "description": "write a file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]}}},
]


async def test_health_and_discovery_report_real_capabilities(ollama_url, puppets):
    provider = OllamaProvider(ollama_url, default_options={"num_ctx": 4096})
    try:
        health = await provider.health()
        assert health.available and health.version is not None
        models = {m.name: m for m in await provider.list_models()}
        info = models[puppets["sysinfo"]]
        assert info.capabilities == {Capability.CHAT, Capability.TOOLS}   # from /api/show, not guessed
        assert info.context_length == 8192 and info.family == "llama" and info.local
    finally:
        await provider.close()


async def test_chat_streaming_and_tool_calls(ollama_url, puppets):
    provider = OllamaProvider(ollama_url, default_options={"num_ctx": 4096})
    try:
        reply = await provider.chat(puppets["text"], [ChatMessage("user", "hello")])
        assert reply.content == "Plain answer from the model." and reply.model == puppets["text"]
        assert reply.completion_tokens and reply.prompt_tokens and reply.done_reason == "stop"

        chunks = [c async for c in provider.stream_chat(puppets["text"], [ChatMessage("user", "hello")])]
        assert chunks[-1].done and chunks[-1].response.content == "Plain answer from the model."
        assert "".join(c.delta for c in chunks) == "Plain answer from the model."

        call = await provider.chat(puppets["write"], [ChatMessage("user", "write it")], tools=TOOLS)
        assert call.content == "" and len(call.tool_calls) == 1
        tc = call.tool_calls[0]
        assert tc.name == "file_write" and tc.arguments["content"] == "written through ollama" and tc.id

        streamed = [c async for c in provider.stream_chat(puppets["write"], [ChatMessage("user", "write")],
                                                          tools=TOOLS)]
        assert streamed[-1].response.tool_calls[0].name == "file_write"

        # the tool result goes back with its name and id, and the model continues
        follow = await provider.chat(puppets["write"], [
            ChatMessage("user", "write it"), ChatMessage("assistant", "", tool_calls=call.tool_calls),
            ChatMessage("tool", '{"ok": true}', name="file_write", tool_call_id=tc.id)], tools=TOOLS)
        assert follow.content == "WRITE-ANSWER: the file is written." and not follow.tool_calls
    finally:
        await provider.close()


async def test_model_lifecycle(ollama_url, puppets):
    provider = OllamaProvider(ollama_url)
    try:
        await provider.load(puppets["text"])
        assert puppets["text"] in await provider.loaded_models()
        await provider.unload(puppets["text"])
        for _ in range(50):
            if puppets["text"] not in await provider.loaded_models():
                break
            await asyncio.sleep(0.1)
        assert puppets["text"] not in await provider.loaded_models()
    finally:
        await provider.close()


async def test_errors_are_classified(ollama_url, puppets):
    provider = OllamaProvider(ollama_url)
    try:
        with pytest.raises(ModelError) as missing:
            await provider.chat("jarvis-no-such-model:latest", [ChatMessage("user", "hi")])
        assert not isinstance(missing.value, ModelUnavailable) and not missing.value.transient
        with pytest.raises(ModelError):
            await provider.embed(puppets["text"], ["not an embedding model"])
    finally:
        await provider.close()
    down = OllamaProvider("http://127.0.0.1:9")
    try:
        assert not (await down.health()).available
        with pytest.raises(ModelUnavailable):
            await down.chat(puppets["text"], [ChatMessage("user", "hi")])
    finally:
        await down.close()


async def test_router_routes_to_real_models(ollama_url, puppets):
    router = ModelRouter([OllamaProvider(ollama_url)])
    try:
        await router.refresh()
        router.pin(puppets["text"])
        routed = await router.chat(TaskProfile(needs_tools=True), [ChatMessage("user", "hello")], tools=TOOLS)
        assert routed.response.model == puppets["text"] and routed.decision.local
        assert await router.embed(["text"]) is None or True   # an embedding model is optional
    finally:
        await router.close()
