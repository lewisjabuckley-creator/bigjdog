"""Optional adapter for OpenAI-compatible endpoints.

Covers local servers that speak this protocol (llama.cpp server, vLLM, LM Studio)
as well as cloud APIs. Disabled by default; when it points at a cloud endpoint it
is ``local=False`` and is therefore excluded in private mode.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from jarvis.models.base import (Capability, ChatMessage, ChatResponse, ModelError, ModelInfo, ModelProvider,
                                ModelUnavailable, ProviderHealth, ToolCall)


class OpenAICompatibleProvider(ModelProvider):
    def __init__(self, base_url: str, *, api_key: str | None = None, local: bool = False,
                 models: list[str] | None = None, name: str = "openai_compatible", timeout: float = 120.0,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.name = name
        self.local = local
        self._models = models or []
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers,
                                         timeout=httpx.Timeout(timeout, connect=5.0), transport=transport)

    async def _request(self, method: str, path: str, json_body: Any = None, timeout: float | None = None) -> Any:
        try:
            resp = await self._client.request(method, path, json=json_body,
                                              timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ModelUnavailable(f"{self.name} unreachable: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ModelError(f"{self.name} timed out", transient=True) from exc
        except httpx.HTTPError as exc:
            raise ModelUnavailable(f"{self.name} transport error: {exc}") from exc
        if resp.status_code in (401, 403):
            raise ModelError(f"{self.name} rejected credentials ({resp.status_code})")
        if resp.status_code == 429:
            raise ModelError(f"{self.name} rate limited", transient=True)
        if resp.status_code >= 400:
            raise ModelError(f"{self.name} error {resp.status_code}: {resp.text[:300]}",
                             transient=resp.status_code >= 500)
        return resp.json()

    async def health(self) -> ProviderHealth:
        started = time.monotonic()
        try:
            await self._request("GET", "/models", timeout=5.0)
        except ModelError as exc:
            return ProviderHealth(False, str(exc))
        return ProviderHealth(True, "ok", round((time.monotonic() - started) * 1000, 1))

    async def list_models(self) -> list[ModelInfo]:
        names = list(self._models)
        if not names:
            data = await self._request("GET", "/models")
            names = [m["id"] for m in data.get("data", [])]
        infos = []
        for name in names:
            caps = {Capability.EMBEDDING} if "embed" in name.lower() else {Capability.CHAT, Capability.TOOLS}
            infos.append(ModelInfo(name, self.name, self.local, frozenset(caps)))
        return infos

    async def chat(self, model: str, messages: list[ChatMessage], *, tools: list[dict[str, Any]] | None = None,
                   options: dict[str, Any] | None = None, format: Any = None,
                   timeout: float | None = None) -> ChatResponse:
        body: dict[str, Any] = {"model": model, "messages": [_to_openai(m) for m in messages]}
        if tools:
            body["tools"] = tools
        if options:
            if "temperature" in options:
                body["temperature"] = options["temperature"]
            if "num_predict" in options:
                body["max_tokens"] = options["num_predict"]
        if format == "json" or isinstance(format, dict):
            body["response_format"] = {"type": "json_object"}
        started = time.monotonic()
        data = await self._request("POST", "/chat/completions", body, timeout)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls = []
        for raw in msg.get("tool_calls") or []:
            fn = raw.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {"_raw": fn.get("arguments")}
            calls.append(ToolCall(fn.get("name", ""), args, raw.get("id")))
        usage = data.get("usage") or {}
        return ChatResponse(msg.get("content") or "", data.get("model", model), self.name, calls,
                            usage.get("prompt_tokens"), usage.get("completion_tokens"),
                            round(time.monotonic() - started, 3), choice.get("finish_reason"))

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        data = await self._request("POST", "/embeddings", {"model": model, "input": texts})
        return [d["embedding"] for d in sorted(data.get("data", []), key=lambda d: d.get("index", 0))]

    async def close(self) -> None:
        await self._client.aclose()


def _to_openai(m: ChatMessage) -> dict[str, Any]:
    if m.images:
        content: Any = [{"type": "text", "text": m.content}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in m.images]
    else:
        content = m.content
    out: dict[str, Any] = {"role": m.role, "content": content}
    if m.tool_calls:
        out["tool_calls"] = [{"id": c.id or f"call_{i}", "type": "function",
                              "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                             for i, c in enumerate(m.tool_calls)]
    if m.role == "tool":
        out["tool_call_id"] = m.tool_call_id or "call_0"
    return out
