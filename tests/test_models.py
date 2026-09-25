import json

import httpx
import pytest

from jarvis.config import ModelsConfig
from jarvis.core.types import HealthStatus
from jarvis.events.bus import EventBus
from jarvis.models.base import Capability, ChatMessage, ModelInfo, ModelUnavailable, NoModelAvailable, Purpose, ToolCall
from jarvis.models.fake import ScriptedProvider
from jarvis.models.ollama import OllamaProvider
from jarvis.models.openai_compat import OpenAICompatibleProvider
from jarvis.models.router import ModelRouter, TaskProfile
from jarvis.state.health import HealthRegistry


def fake_ollama(requests: list):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        requests.append((request.method, request.url.path, body))
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.9.0"})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "llama3.1:8b", "size": 4_900_000_000,
                 "details": {"family": "llama", "parameter_size": "8.0B", "quantization_level": "Q4_K_M"}},
                {"name": "llava:13b", "size": 8_000_000_000,
                 "details": {"family": "llama", "families": ["llama", "clip"], "parameter_size": "13B"}},
                {"name": "nomic-embed-text:latest", "size": 270_000_000,
                 "details": {"family": "nomic-bert", "parameter_size": "137M"}},
            ]})
        if path == "/api/show":
            if body["model"] == "llama3.1:8b":
                return httpx.Response(200, json={"capabilities": ["completion", "tools"],
                                                 "model_info": {"llama.context_length": 131072}})
            return httpx.Response(200, json={})  # older server without capabilities
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "llama3.1:8b", "size_vram": 5_000_000_000}]})
        if path == "/api/chat":
            if body.get("stream"):
                lines = [{"message": {"content": "Hel"}, "done": False},
                         {"message": {"content": "lo"}, "done": False},
                         {"message": {"content": ""}, "done": True, "eval_count": 2, "done_reason": "stop"}]
                return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())
            if body.get("tools"):
                return httpx.Response(200, json={"model": body["model"], "message": {
                    "role": "assistant", "content": "",
                    "tool_calls": [{"function": {"name": "system_info", "arguments": {}}}]},
                    "done_reason": "stop", "prompt_eval_count": 50, "eval_count": 5})
            return httpx.Response(200, json={"model": body["model"], "message": {"role": "assistant", "content": "Hi."},
                                             "prompt_eval_count": 10, "eval_count": 2})
        if path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2] for _ in body["input"]]})
        if path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        return httpx.Response(404, json={"error": "not found"})
    return httpx.MockTransport(handler)


async def test_ollama_discovery_and_capabilities():
    requests = []
    p = OllamaProvider(transport=fake_ollama(requests))
    assert (await p.health()).version == "0.9.0"
    models = {m.name: m for m in await p.list_models()}
    assert models["llama3.1:8b"].capabilities == {Capability.CHAT, Capability.TOOLS}
    assert models["llama3.1:8b"].context_length == 131072
    assert models["llama3.1:8b"].loaded and models["llama3.1:8b"].vram_bytes == 5_000_000_000
    assert Capability.VISION in models["llava:13b"].capabilities       # inferred from clip family
    assert models["nomic-embed-text:latest"].capabilities == {Capability.EMBEDDING}
    assert models["llama3.1:8b"].params_b == 8.0
    await p.close()


async def test_ollama_chat_tools_stream_embed_lifecycle():
    requests = []
    p = OllamaProvider(transport=fake_ollama(requests))
    msgs = [ChatMessage("user", "status?", images=["aGk="])]
    r = await p.chat("llama3.1:8b", msgs, tools=[{"type": "function", "function": {"name": "system_info"}}])
    assert r.tool_calls[0].name == "system_info" and r.prompt_tokens == 50
    sent = requests[-1][2]
    assert sent["messages"][0]["images"] == ["aGk="] and sent["stream"] is False
    chunks = [c async for c in p.stream_chat("llama3.1:8b", [ChatMessage("user", "hi")])]
    assert "".join(c.delta for c in chunks) == "Hello" and chunks[-1].done
    assert chunks[-1].response.content == "Hello"
    assert await p.embed("nomic-embed-text", ["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    await p.unload("llama3.1:8b")
    assert requests[-1][2]["keep_alive"] == 0
    # tool results are sent back with the tool name
    await p.chat("llama3.1:8b", [ChatMessage("assistant", "", tool_calls=[ToolCall("x", {"a": 1})]),
                                 ChatMessage("tool", "{}", name="x")])
    assert requests[-1][2]["messages"][1]["tool_name"] == "x"
    await p.close()


async def test_ollama_unreachable_raises_unavailable():
    def refuse(request):
        raise httpx.ConnectError("connection refused")
    p = OllamaProvider(transport=httpx.MockTransport(refuse))
    assert not (await p.health()).available
    with pytest.raises(ModelUnavailable):
        await p.chat("x", [ChatMessage("user", "hi")])
    await p.close()


async def test_openai_compatible_adapter():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer k"
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json={"model": "m", "choices": [{"message": {
                "content": "", "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{\"a\": 1}"}}]},
                "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}})
        return httpx.Response(200, json={"data": [{"id": "m"}]})
    p = OpenAICompatibleProvider("http://x/v1", api_key="k", transport=httpx.MockTransport(handler))
    assert [m.name for m in await p.list_models()] == ["m"]
    r = await p.chat("m", [ChatMessage("user", "hi")])
    assert r.tool_calls[0].arguments == {"a": 1}
    await p.close()


# -- router -------------------------------------------------------------------------------

def _router(providers, **kw):
    cfg = ModelsConfig(profiles={"conversation": ["sim-small:3b"], "coding": ["sim-coder"],
                                 "embedding": ["sim-embed"]})
    return ModelRouter(providers, cfg, **kw)


async def test_router_prefers_profile_and_respects_capabilities():
    sim = ScriptedProvider()
    router = _router([sim])
    await router.refresh()
    assert router.select(TaskProfile()).model == "sim-small:3b"
    # tools required: the small model can't call tools
    assert router.select(TaskProfile(needs_tools=True)).model == "sim-chat:8b"
    # no vision model installed: explain rather than guess
    with pytest.raises(NoModelAvailable, match="vision"):
        router.select(TaskProfile(needs_vision=True))
    assert router.select(TaskProfile(purpose=Purpose.EMBEDDING)).model == "sim-embed"


async def test_router_complexity_prefers_bigger_model_without_preferences():
    sim = ScriptedProvider()
    router = ModelRouter([sim], ModelsConfig(profiles={}))
    await router.refresh()
    assert router.select(TaskProfile(complexity="low")).model == "sim-small:3b"
    assert router.select(TaskProfile(complexity="high")).model == "sim-chat:8b"


async def test_router_pin_overrides_preference():
    router = _router([ScriptedProvider()])
    await router.refresh()
    router.pin("sim-chat")
    d = router.select(TaskProfile())
    assert d.model == "sim-chat:8b" and d.reason == "selected by you"


async def test_router_falls_back_and_reports_it(clock):
    sim = ScriptedProvider()
    bus = EventBus(clock=clock)
    events = []
    bus.subscribe("MODEL_*", lambda e: events.append(e.type))
    router = _router([sim], bus=bus, clock=clock)
    await router.refresh()
    sim.failing_models.add("sim-small:3b")
    routed = await router.chat(TaskProfile(), [ChatMessage("user", "hi")])
    assert routed.used_fallback and routed.response.model == "sim-chat:8b"
    assert "unavailable" in routed.fallback_note
    await bus.drain()
    assert "MODEL_FALLBACK" in events


async def test_router_private_mode_excludes_cloud():
    local = ScriptedProvider("local")
    cloud = ScriptedProvider("cloud", local=False, models=[
        ModelInfo("cloud-big", "cloud", False, frozenset({Capability.CHAT, Capability.TOOLS, Capability.VISION}))])
    private = {"on": False}
    router = ModelRouter([local, cloud], ModelsConfig(profiles={}), local_only=lambda: private["on"],
                         allow_cloud=lambda: True)
    await router.refresh()
    assert router.select(TaskProfile(needs_vision=True)).model == "cloud-big"
    private["on"] = True
    with pytest.raises(NoModelAvailable, match="local"):
        router.select(TaskProfile(needs_vision=True))
    # cloud is opt-in: without allow_cloud it is never chosen
    router2 = ModelRouter([cloud], ModelsConfig(profiles={}))
    await router2.refresh()
    assert not router2.available()


async def test_provider_outage_and_recovery(clock):
    sim = ScriptedProvider()
    health = HealthRegistry(None, clock)
    router = _router([sim], clock=clock, health=health)
    await router.refresh()
    sim.online = False
    with pytest.raises(NoModelAvailable):
        await router.chat(TaskProfile(), [ChatMessage("user", "hi")])
    assert health.components["model:scripted"].status == HealthStatus.OFFLINE
    clock.advance(18)
    sim.online = True
    routed = await router.chat(TaskProfile(), [ChatMessage("user", "hi")])  # re-probes and recovers
    assert routed.response.content == "Understood."
    assert health.recent_outages(component="model:scripted")[0].duration == 18


async def test_router_embed_returns_none_when_unavailable():
    router = ModelRouter([ScriptedProvider(models=[])], ModelsConfig(profiles={}))
    await router.refresh()
    assert await router.embed(["x"]) is None
