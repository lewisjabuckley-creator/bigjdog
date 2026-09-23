"""Model router (spec §4-5, §39, §98).

Chooses a model per request from task requirements (purpose, complexity,
tools, vision, context, latency, privacy) and live availability; tracks model
health; falls back across models and providers; and reports — never hides —
when a fallback changed behaviour.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from jarvis.clock import Clock, SystemClock
from jarvis.config import ModelsConfig
from jarvis.core.types import HealthStatus, Severity
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.log import get_logger
from jarvis.models.base import (Capability, ChatChunk, ChatMessage, ChatResponse, ModelError, ModelInfo,
                                ModelProvider, ModelUnavailable, NoModelAvailable, Purpose)
from jarvis.state.health import HealthRegistry

log = get_logger("models")


@dataclass
class TaskProfile:
    purpose: Purpose = Purpose.CONVERSATION
    complexity: str = "low"          # low | medium | high
    needs_tools: bool = False
    needs_vision: bool = False
    min_context: int = 0
    interactive: bool = True
    local_only: bool = False


@dataclass
class RouteDecision:
    provider: str
    model: str
    reason: str
    fallbacks: list[tuple[str, str]] = field(default_factory=list)
    local: bool = True


@dataclass
class RoutedResponse:
    response: ChatResponse
    decision: RouteDecision
    used_fallback: bool = False
    attempts: list[str] = field(default_factory=list)

    @property
    def fallback_note(self) -> str | None:
        if not self.used_fallback:
            return None
        return (f"{self.decision.model} was unavailable, so I used {self.response.model} instead")


@dataclass
class _ModelHealth:
    failures: int = 0
    unhealthy_until: float = 0.0
    last_error: str = ""


class ModelRouter:
    def __init__(self, providers: list[ModelProvider], config: ModelsConfig | None = None, *,
                 bus: EventBus | None = None, clock: Clock | None = None, health: HealthRegistry | None = None,
                 local_only: Callable[[], bool] | None = None, allow_cloud: Callable[[], bool] | None = None) -> None:
        self.providers = {p.name: p for p in providers}
        self.config = config or ModelsConfig()
        self.bus = bus
        self.clock = clock or SystemClock()
        self.health = health
        self._local_only = local_only or (lambda: False)
        self._allow_cloud = allow_cloud or (lambda: False)
        self.inventory: list[ModelInfo] = []
        self.provider_status: dict[str, bool] = {}
        self._model_health: dict[tuple[str, str], _ModelHealth] = {}
        self.pins: dict[str, str] = {}         # purpose (or "*") -> model name chosen by the user
        self.last_refresh: float | None = None
        self.calls = 0
        self.tokens = {"prompt": 0, "completion": 0}

    # -- inventory ----------------------------------------------------------------
    async def refresh(self) -> list[ModelInfo]:
        inventory: list[ModelInfo] = []
        for provider in self.providers.values():
            was = self.provider_status.get(provider.name)
            try:
                models = await provider.list_models()
                inventory.extend(models)
                self.provider_status[provider.name] = True
                self._report(provider.name, HealthStatus.HEALTHY, f"{len(models)} model(s)")
                if was is False:
                    self._emit(EventType.MODEL_RECOVERED, {"provider": provider.name}, Severity.INFO)
            except ModelError as exc:
                self.provider_status[provider.name] = False
                self._report(provider.name, HealthStatus.OFFLINE, str(exc))
                if was is not False:
                    self._emit(EventType.MODEL_UNAVAILABLE, {"provider": provider.name, "error": str(exc)},
                               Severity.WARNING)
        previous_loaded = {m.name for m in self.inventory if m.loaded}
        for m in inventory:
            if m.loaded and m.name not in previous_loaded and self.last_refresh is not None:
                self._emit(EventType.MODEL_LOADED, {"model": m.name, "provider": m.provider}, Severity.INFO)
        self.inventory = inventory
        self.last_refresh = self.clock.now()
        return inventory

    def available(self, capability: Capability = Capability.CHAT) -> bool:
        return any(m.has(capability) and self._usable_provider(m) for m in self.inventory)

    def _usable_provider(self, m: ModelInfo) -> bool:
        if not self.provider_status.get(m.provider, False):
            return False
        if not m.local and (self._local_only() or not self._allow_cloud()):
            return False
        return True

    def find(self, name: str) -> ModelInfo | None:
        for m in self.inventory:
            if _name_matches(m.name, name):
                return m
        return None

    # -- selection --------------------------------------------------------------
    def pin(self, model: str, purpose: str = "*") -> ModelInfo:
        info = self.find(model)
        if info is None:
            raise NoModelAvailable(f"no installed model matches {model!r}")
        self.pins[purpose] = info.name
        return info

    def unpin(self, purpose: str = "*") -> None:
        self.pins.pop(purpose, None)

    def candidates(self, profile: TaskProfile) -> list[ModelInfo]:
        need = Capability.EMBEDDING if profile.purpose == Purpose.EMBEDDING else Capability.CHAT
        out = []
        for m in self.inventory:
            if not self._usable_provider(m):
                continue
            if profile.local_only and not m.local:
                continue
            if not m.has(need):
                continue
            if need == Capability.CHAT and m.has(Capability.EMBEDDING) and not m.has(Capability.CHAT):
                continue
            if profile.needs_tools and not m.has(Capability.TOOLS):
                continue
            if profile.needs_vision and not m.has(Capability.VISION):
                continue
            if profile.min_context and m.context_length and m.context_length < profile.min_context:
                continue
            out.append(m)
        return out

    def select(self, profile: TaskProfile) -> RouteDecision:
        candidates = self.candidates(profile)
        if not candidates:
            raise NoModelAvailable(self._explain_none(profile))
        now = self.clock.now()
        healthy = [m for m in candidates if self._health(m).unhealthy_until <= now] or candidates
        ranked = self._rank(healthy, profile)
        purpose = profile.purpose.value
        pinned = self.pins.get(purpose) or self.pins.get("*")
        primary, reason = ranked[0], ""
        if pinned:
            match = next((m for m in ranked if m.name == pinned), None)
            if match:
                primary, reason = match, "selected by you"
        if not reason:
            prefs = self.config.profiles.get(purpose, [])
            preferred = [m for p in prefs for m in ranked if _name_matches(m.name, p)]
            if preferred:
                primary, reason = preferred[0], f"preferred {purpose} model"
            else:
                reason = f"best available for {profile.complexity}-complexity {purpose}"
        rest = [m for m in ranked if m is not primary]
        return RouteDecision(primary.provider, primary.name, reason, [(m.provider, m.name) for m in rest],
                             primary.local)

    def _rank(self, models: list[ModelInfo], profile: TaskProfile) -> list[ModelInfo]:
        prefs = self.config.profiles.get(profile.purpose.value, [])

        def pref_index(m: ModelInfo) -> int:
            for i, p in enumerate(prefs):
                if _name_matches(m.name, p):
                    return i
            return len(prefs)

        def size(m: ModelInfo) -> float:
            return m.params_b if m.params_b is not None else 7.0

        if profile.complexity == "high":
            key = lambda m: (pref_index(m), -size(m))
        else:
            # simple / interactive work: prefer already-loaded, then smaller (faster) models
            key = lambda m: (pref_index(m), not m.loaded if profile.interactive else False, size(m))
        return sorted(models, key=key)

    def _explain_none(self, profile: TaskProfile) -> str:
        if not self.inventory:
            down = [p for p, ok in self.provider_status.items() if not ok]
            if down:
                return f"no model provider is reachable ({', '.join(down)})"
            return "no models are installed"
        needs = []
        if profile.needs_vision:
            needs.append("vision")
        if profile.needs_tools:
            needs.append("tool calling")
        if profile.local_only or self._local_only():
            needs.append("local execution")
        suffix = f" with {', '.join(needs)}" if needs else ""
        return f"no available model supports {profile.purpose.value}{suffix}"

    # -- invocation ---------------------------------------------------------------
    async def chat(self, profile: TaskProfile, messages: list[ChatMessage], *,
                   tools: list[dict[str, Any]] | None = None, format: Any = None,
                   options: dict[str, Any] | None = None, timeout: float | None = None,
                   max_attempts: int = 3) -> RoutedResponse:
        decision = await self._decide(profile)
        queue = [(decision.provider, decision.model)] + decision.fallbacks
        attempts: list[str] = []
        tried = 0
        while queue and tried < max_attempts:
            provider_name, model = queue.pop(0)
            tried += 1
            provider = self.providers[provider_name]
            try:
                response = await provider.chat(model, messages, tools=tools, format=format, options=options,
                                               timeout=timeout)
            except ModelError as exc:
                attempts.append(f"{model}: {exc}")
                self._record_failure(provider_name, model, exc)
                if isinstance(exc, ModelUnavailable):
                    # the whole provider is down; skip its other models
                    queue = [o for o in queue if o[0] != provider_name]
                continue
            self._record_success(provider_name, model)
            self.calls += 1
            self.tokens["prompt"] += response.prompt_tokens or 0
            self.tokens["completion"] += response.completion_tokens or 0
            used_fallback = (provider_name, model) != (decision.provider, decision.model)
            if used_fallback:
                self._emit(EventType.MODEL_FALLBACK, {"intended": decision.model, "used": model,
                                                      "attempts": attempts}, Severity.WARNING)
            return RoutedResponse(response, decision, used_fallback, attempts)
        raise NoModelAvailable("all candidate models failed: " + "; ".join(attempts), attempts)

    async def _decide(self, profile: TaskProfile) -> RouteDecision:
        """Select a model, re-probing providers once if none is currently usable (they may have recovered)."""
        if not self.inventory:
            await self.refresh()
        try:
            return self.select(profile)
        except NoModelAvailable:
            if all(self.provider_status.values()) and self.inventory:
                raise
            await self.refresh()
            return self.select(profile)

    async def stream(self, profile: TaskProfile, messages: list[ChatMessage], *,
                     tools: list[dict[str, Any]] | None = None) -> AsyncIterator[ChatChunk]:
        decision = await self._decide(profile)
        provider = self.providers[decision.provider]
        try:
            async for chunk in provider.stream_chat(decision.model, messages, tools=tools):
                yield chunk
            self._record_success(decision.provider, decision.model)
        except ModelError as exc:
            self._record_failure(decision.provider, decision.model, exc)
            raise

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], str] | None:
        try:
            decision = self.select(TaskProfile(purpose=Purpose.EMBEDDING))
        except NoModelAvailable:
            return None
        try:
            vectors = await self.providers[decision.provider].embed(decision.model, texts)
        except ModelError as exc:
            self._record_failure(decision.provider, decision.model, exc)
            return None
        return vectors, decision.model

    async def load(self, model: str) -> ModelInfo:
        info = self.find(model)
        if info is None:
            raise NoModelAvailable(f"no installed model matches {model!r}")
        await self.providers[info.provider].load(info.name)
        info.loaded = True
        self._emit(EventType.MODEL_LOADED, {"model": info.name}, Severity.INFO)
        return info

    async def unload(self, model: str) -> ModelInfo:
        info = self.find(model)
        if info is None:
            raise NoModelAvailable(f"no installed model matches {model!r}")
        await self.providers[info.provider].unload(info.name)
        info.loaded = False
        self._emit(EventType.MODEL_UNLOADED, {"model": info.name}, Severity.INFO)
        return info

    async def provider_health(self) -> dict[str, dict[str, Any]]:
        results = await asyncio.gather(*(p.health() for p in self.providers.values()), return_exceptions=True)
        out = {}
        for provider, res in zip(self.providers.values(), results):
            if isinstance(res, BaseException):
                out[provider.name] = {"available": False, "detail": str(res)}
            else:
                out[provider.name] = {"available": res.available, "detail": res.detail, "latency_ms": res.latency_ms,
                                      "version": res.version}
        return out

    async def close(self) -> None:
        for provider in self.providers.values():
            try:
                await provider.close()
            except Exception:
                pass

    # -- health tracking ------------------------------------------------------------
    def _health(self, m: ModelInfo) -> _ModelHealth:
        return self._model_health.setdefault((m.provider, m.name), _ModelHealth())

    def _record_failure(self, provider: str, model: str, exc: ModelError) -> None:
        h = self._model_health.setdefault((provider, model), _ModelHealth())
        h.failures += 1
        h.last_error = str(exc)
        log.warning("model_call_failed", provider=provider, model=model, error=str(exc), failures=h.failures)
        if isinstance(exc, ModelUnavailable):
            if self.provider_status.get(provider, True):
                self.provider_status[provider] = False
                self._report(provider, HealthStatus.OFFLINE, str(exc))
                self._emit(EventType.MODEL_UNAVAILABLE, {"provider": provider, "model": model, "error": str(exc)},
                           Severity.WARNING)
        elif h.failures >= self.config.failure_threshold:
            h.unhealthy_until = self.clock.now() + self.config.unhealthy_cooldown_s
            self._emit(EventType.MODEL_UNAVAILABLE, {"provider": provider, "model": model, "error": str(exc)},
                       Severity.WARNING)

    def _record_success(self, provider: str, model: str) -> None:
        h = self._model_health.setdefault((provider, model), _ModelHealth())
        h.failures = 0
        h.unhealthy_until = 0.0

    def _report(self, provider: str, status: HealthStatus, detail: str) -> None:
        if self.health is not None:
            self.health.report(f"model:{provider}", status, detail)

    def _emit(self, etype: EventType, payload: dict[str, Any], severity: Severity) -> None:
        if self.bus is not None:
            self.bus.emit(Event(etype, "models", payload, severity=severity))


def _name_matches(name: str, wanted: str) -> bool:
    name, wanted = name.lower(), wanted.lower()
    if name == wanted:
        return True
    if name.removesuffix(":latest") == wanted.removesuffix(":latest"):
        return True
    return ":" not in wanted and name.split(":")[0] == wanted
