"""Model provider abstraction (spec §4, §98).

JARVIS is never hard-coded to one model or runtime. Every inference backend
implements :class:`ModelProvider`; everything above this layer talks to the
router, never to a specific provider.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, AsyncIterator


class Capability(StrEnum):
    CHAT = "chat"
    TOOLS = "tools"
    VISION = "vision"
    EMBEDDING = "embedding"
    REASONING = "reasoning"


class Purpose(StrEnum):
    CONVERSATION = "conversation"
    REASONING = "reasoning"
    PLANNING = "planning"
    CODING = "coding"
    VISION = "vision"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    BACKGROUND = "background"
    EMBEDDING = "embedding"


@dataclass
class ModelInfo:
    name: str
    provider: str
    local: bool
    capabilities: frozenset[Capability] = frozenset({Capability.CHAT})
    context_length: int | None = None
    parameter_size: str | None = None      # e.g. "8.0B"
    size_bytes: int | None = None
    quantization: str | None = None
    family: str | None = None
    loaded: bool = False
    vram_bytes: int | None = None

    @property
    def params_b(self) -> float | None:
        if self.parameter_size:
            m = re.match(r"([\d.]+)\s*([BMK])", self.parameter_size.upper())
            if m:
                scale = {"B": 1.0, "M": 1e-3, "K": 1e-6}[m.group(2)]
                return float(m.group(1)) * scale
        m = re.search(r"(\d+(?:\.\d+)?)b\b", self.name.lower())
        return float(m.group(1)) if m else None

    def has(self, cap: Capability) -> bool:
        return cap in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "provider": self.provider, "local": self.local,
                "capabilities": sorted(c.value for c in self.capabilities), "context_length": self.context_length,
                "parameter_size": self.parameter_size, "size_bytes": self.size_bytes, "loaded": self.loaded,
                "vram_bytes": self.vram_bytes}


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass
class ChatMessage:
    role: str                      # system | user | assistant | tool
    content: str = ""
    images: list[str] = field(default_factory=list)       # base64-encoded
    tool_calls: list[ToolCall] = field(default_factory=list)
    name: str | None = None        # tool name for role=tool
    tool_call_id: str | None = None


@dataclass
class ChatResponse:
    content: str
    model: str
    provider: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None
    done_reason: str | None = None


@dataclass
class ChatChunk:
    delta: str = ""
    done: bool = False
    response: ChatResponse | None = None


@dataclass
class ProviderHealth:
    available: bool
    detail: str = ""
    latency_ms: float | None = None
    version: str | None = None


class ModelError(Exception):
    """A model call failed. ``transient`` failures may succeed on another attempt or model."""

    def __init__(self, message: str, *, transient: bool = False, resource: bool = False) -> None:
        super().__init__(message)
        self.transient = transient
        self.resource = resource   # e.g. out of memory — a smaller model may work


class ModelUnavailable(ModelError):
    def __init__(self, message: str) -> None:
        super().__init__(message, transient=True)


class NoModelAvailable(ModelError):
    def __init__(self, message: str, attempts: list[str] | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []


class ModelProvider(ABC):
    name: str
    local: bool

    @abstractmethod
    async def health(self) -> ProviderHealth: ...

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]: ...

    @abstractmethod
    async def chat(self, model: str, messages: list[ChatMessage], *, tools: list[dict[str, Any]] | None = None,
                   options: dict[str, Any] | None = None, format: Any = None,
                   timeout: float | None = None) -> ChatResponse: ...

    async def stream_chat(self, model: str, messages: list[ChatMessage], *,
                          tools: list[dict[str, Any]] | None = None, options: dict[str, Any] | None = None,
                          timeout: float | None = None) -> AsyncIterator[ChatChunk]:
        response = await self.chat(model, messages, tools=tools, options=options, timeout=timeout)
        yield ChatChunk(response.content, done=True, response=response)

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        raise ModelError(f"{self.name} does not support embeddings")

    async def loaded_models(self) -> list[str]:
        return []

    async def load(self, model: str) -> None:
        return None

    async def unload(self, model: str) -> None:
        return None

    async def close(self) -> None:
        return None
