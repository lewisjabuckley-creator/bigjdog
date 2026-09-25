"""Scripted model provider for tests and simulation mode (spec §162).

It is honest about being fake: responses come from explicit rules supplied by
the test or simulation, and it can be switched offline to exercise fallbacks.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from jarvis.models.base import (Capability, ChatMessage, ChatResponse, ModelError, ModelInfo, ModelProvider,
                                ModelUnavailable, ProviderHealth, ToolCall)

Responder = Callable[[str, list[ChatMessage], list[dict[str, Any]] | None], ChatResponse | str | None]


@dataclass
class Rule:
    pattern: re.Pattern[str]
    reply: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    once: bool = False
    used: bool = False


class ScriptedProvider(ModelProvider):
    def __init__(self, name: str = "scripted", *, local: bool = True, models: list[ModelInfo] | None = None,
                 default_reply: str = "Understood.") -> None:
        self.name = name
        self.local = local
        self.models = models if models is not None else [
            ModelInfo("sim-chat:8b", name, local, frozenset({Capability.CHAT, Capability.TOOLS}),
                      context_length=8192, parameter_size="8B"),
            ModelInfo("sim-small:3b", name, local, frozenset({Capability.CHAT}), context_length=4096,
                      parameter_size="3B"),
            ModelInfo("sim-embed", name, local, frozenset({Capability.EMBEDDING})),
        ]
        self.default_reply = default_reply
        self.rules: list[Rule] = []
        self.responder: Responder | None = None
        self.online = True
        self.failing_models: set[str] = set()
        self.calls: list[dict[str, Any]] = []

    # -- scripting ---------------------------------------------------------------
    def when(self, pattern: str, reply: str = "", *, tool_calls: list[ToolCall] | None = None,
             once: bool = False) -> "ScriptedProvider":
        self.rules.append(Rule(re.compile(pattern, re.IGNORECASE), reply, tool_calls or [], once))
        return self

    # -- provider API ------------------------------------------------------------
    async def health(self) -> ProviderHealth:
        return ProviderHealth(self.online, "ok" if self.online else "offline (simulated)", 1.0, "sim")

    async def list_models(self) -> list[ModelInfo]:
        if not self.online:
            raise ModelUnavailable(f"{self.name} is offline (simulated)")
        return list(self.models)

    async def chat(self, model: str, messages: list[ChatMessage], *, tools: list[dict[str, Any]] | None = None,
                   options: dict[str, Any] | None = None, format: Any = None,
                   timeout: float | None = None) -> ChatResponse:
        self.calls.append({"model": model, "messages": messages, "tools": tools, "format": format})
        if not self.online:
            raise ModelUnavailable(f"{self.name} is offline (simulated)")
        if model in self.failing_models:
            raise ModelError(f"{model} failed (simulated)", transient=True)
        if self.responder is not None:
            out = self.responder(model, messages, tools)
            if isinstance(out, ChatResponse):
                return out
            if isinstance(out, str):
                return ChatResponse(out, model, self.name)
        last_user = next((m.content for m in reversed(messages) if m.role in ("user", "tool")), "")
        for rule in self.rules:
            if rule.once and rule.used:
                continue
            if rule.pattern.search(last_user):
                rule.used = True
                return ChatResponse(rule.reply, model, self.name, list(rule.tool_calls))
        if format is not None:
            return ChatResponse(json.dumps({"intent": "chat"}), model, self.name)
        return ChatResponse(self.default_reply, model, self.name)

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if not self.online:
            raise ModelUnavailable(f"{self.name} is offline (simulated)")
        return [_bag_of_words_vector(t) for t in texts]


def _bag_of_words_vector(text: str, dim: int = 64) -> list[float]:
    """Deterministic pseudo-embedding: hashed bag of words. Similar texts share dimensions."""
    vec = [0.0] * dim
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
        vec[idx] += 1.0
    return vec
