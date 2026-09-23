"""Ollama adapter (spec §99).

Implements discovery (/api/tags + /api/show capabilities), lifecycle (/api/ps,
load/unload via keep_alive), chat with tool calling and images, streaming
(NDJSON), embeddings (/api/embed) and health (/api/version). Nothing outside
this module knows it is talking to Ollama.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx

from jarvis.models.base import (Capability, ChatChunk, ChatMessage, ChatResponse, ModelError, ModelInfo,
                                ModelProvider, ModelUnavailable, ProviderHealth, ToolCall)


class OllamaProvider(ModelProvider):
    local = True

    def __init__(self, base_url: str = "http://127.0.0.1:11434", *, keep_alive: str = "5m",
                 timeout: float = 300.0, transport: httpx.AsyncBaseTransport | None = None,
                 name: str = "ollama") -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self.timeout = timeout
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(timeout, connect=3.0),
                                         transport=transport)
        self._show_cache: dict[str, dict[str, Any]] = {}

    # -- helpers ------------------------------------------------------------------
    async def _request(self, method: str, path: str, *, json_body: Any = None, timeout: float | None = None) -> Any:
        try:
            resp = await self._client.request(method, path, json=json_body,
                                              timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ModelUnavailable(f"Ollama is not reachable at {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ModelError(f"Ollama request timed out: {path}", transient=True) from exc
        except httpx.HTTPError as exc:
            raise ModelUnavailable(f"Ollama transport error: {exc}") from exc
        if resp.status_code >= 400:
            raise _http_error(resp)
        return resp.json() if resp.content else {}

    # -- discovery ------------------------------------------------------------------
    async def health(self) -> ProviderHealth:
        started = time.monotonic()
        try:
            data = await self._request("GET", "/api/version", timeout=3.0)
        except ModelError as exc:
            return ProviderHealth(False, str(exc))
        return ProviderHealth(True, "ok", round((time.monotonic() - started) * 1000, 1), data.get("version"))

    async def list_models(self) -> list[ModelInfo]:
        tags = await self._request("GET", "/api/tags")
        loaded = await self._loaded()
        models = []
        for entry in tags.get("models", []):
            name = entry.get("name") or entry.get("model")
            details = entry.get("details") or {}
            show = await self._show(name)
            caps = _capabilities(name, details, show)
            models.append(ModelInfo(
                name=name, provider=self.name, local=True, capabilities=caps,
                context_length=_context_length(show), parameter_size=details.get("parameter_size"),
                size_bytes=entry.get("size"), quantization=details.get("quantization_level"),
                family=details.get("family"), loaded=name in loaded,
                vram_bytes=loaded.get(name)))
        return models

    async def _show(self, name: str) -> dict[str, Any]:
        if name not in self._show_cache:
            try:
                self._show_cache[name] = await self._request("POST", "/api/show", json_body={"model": name})
            except ModelError:
                return {}
        return self._show_cache[name]

    async def _loaded(self) -> dict[str, int | None]:
        try:
            data = await self._request("GET", "/api/ps", timeout=5.0)
        except ModelError:
            return {}
        return {m.get("name") or m.get("model"): m.get("size_vram") for m in data.get("models", [])}

    async def loaded_models(self) -> list[str]:
        return list((await self._loaded()).keys())

    # -- lifecycle --------------------------------------------------------------------
    async def load(self, model: str) -> None:
        await self._request("POST", "/api/generate", json_body={"model": model, "prompt": "",
                                                                "keep_alive": self.keep_alive})

    async def unload(self, model: str) -> None:
        await self._request("POST", "/api/generate", json_body={"model": model, "prompt": "", "keep_alive": 0})

    # -- inference ---------------------------------------------------------------------
    def _chat_body(self, model: str, messages: list[ChatMessage], tools: list[dict[str, Any]] | None,
                   options: dict[str, Any] | None, format: Any, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": [_to_ollama(m) for m in messages], "stream": stream,
                                "keep_alive": self.keep_alive}
        if tools:
            body["tools"] = tools
        if options:
            body["options"] = options
        if format is not None:
            body["format"] = format
        return body

    async def chat(self, model: str, messages: list[ChatMessage], *, tools: list[dict[str, Any]] | None = None,
                   options: dict[str, Any] | None = None, format: Any = None,
                   timeout: float | None = None) -> ChatResponse:
        started = time.monotonic()
        data = await self._request("POST", "/api/chat",
                                   json_body=self._chat_body(model, messages, tools, options, format, False),
                                   timeout=timeout)
        return _from_ollama(data, model, self.name, time.monotonic() - started)

    async def stream_chat(self, model: str, messages: list[ChatMessage], *,
                          tools: list[dict[str, Any]] | None = None, options: dict[str, Any] | None = None,
                          timeout: float | None = None) -> AsyncIterator[ChatChunk]:
        started = time.monotonic()
        body = self._chat_body(model, messages, tools, options, None, True)
        content: list[str] = []
        tool_calls: list[ToolCall] = []
        try:
            async with self._client.stream("POST", "/api/chat", json=body,
                                           timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise _http_error(resp)
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise ModelError(str(data["error"]))
                    msg = data.get("message") or {}
                    delta = msg.get("content") or ""
                    tool_calls += _tool_calls(msg)
                    if delta:
                        content.append(delta)
                        yield ChatChunk(delta)
                    if data.get("done"):
                        final = _from_ollama({**data, "message": {"content": "".join(content)}}, model, self.name,
                                             time.monotonic() - started)
                        final.tool_calls = tool_calls
                        yield ChatChunk("", done=True, response=final)
                        return
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ModelUnavailable(f"Ollama is not reachable at {self.base_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ModelError("Ollama stream timed out", transient=True) from exc

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        data = await self._request("POST", "/api/embed", json_body={"model": model, "input": texts})
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ModelError("unexpected embedding response shape")
        return vectors

    async def close(self) -> None:
        await self._client.aclose()


def _http_error(resp: httpx.Response) -> ModelError:
    try:
        detail = resp.json().get("error", resp.text)
    except (ValueError, AttributeError):
        detail = resp.text
    text = str(detail)
    lowered = text.lower()
    if resp.status_code == 404:
        return ModelError(f"model not found: {text}")
    if "memory" in lowered or "cuda" in lowered or "vram" in lowered:
        return ModelError(f"insufficient resources: {text}", resource=True, transient=True)
    if "does not support tools" in lowered:
        return ModelError(text)
    return ModelError(f"Ollama error {resp.status_code}: {text}", transient=resp.status_code >= 500)


def _capabilities(name: str, details: dict[str, Any], show: dict[str, Any]) -> frozenset[Capability]:
    reported = show.get("capabilities")
    caps: set[Capability] = set()
    if isinstance(reported, list):
        mapping = {"completion": Capability.CHAT, "tools": Capability.TOOLS, "vision": Capability.VISION,
                   "embedding": Capability.EMBEDDING, "thinking": Capability.REASONING}
        caps = {mapping[c] for c in reported if c in mapping}
        if caps:
            return frozenset(caps)
    # Older Ollama versions: infer conservatively from families and name.
    families = set(details.get("families") or []) | {details.get("family") or ""}
    lowered = name.lower()
    if "embed" in lowered or families & {"bert", "nomic-bert"}:
        return frozenset({Capability.EMBEDDING})
    caps.add(Capability.CHAT)
    if "clip" in families or any(k in lowered for k in ("llava", "vision", "moondream", "bakllava")):
        caps.add(Capability.VISION)
    return frozenset(caps)


def _context_length(show: dict[str, Any]) -> int | None:
    info = show.get("model_info") or {}
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


def _to_ollama(m: ChatMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.images:
        out["images"] = m.images
    if m.tool_calls:
        out["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls]
    if m.role == "tool" and m.name:
        out["tool_name"] = m.name
    return out


def _tool_calls(msg: dict[str, Any]) -> list[ToolCall]:
    calls = []
    for raw in msg.get("tool_calls") or []:
        fn = raw.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {"_raw": args}
        calls.append(ToolCall(fn.get("name", ""), args, raw.get("id")))
    return calls


def _from_ollama(data: dict[str, Any], model: str, provider: str, latency: float) -> ChatResponse:
    msg = data.get("message") or {}
    return ChatResponse(content=msg.get("content") or "", model=data.get("model") or model, provider=provider,
                        tool_calls=_tool_calls(msg), prompt_tokens=data.get("prompt_eval_count"),
                        completion_tokens=data.get("eval_count"), latency_s=round(latency, 3),
                        done_reason=data.get("done_reason"))
