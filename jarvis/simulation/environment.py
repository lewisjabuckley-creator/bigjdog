"""Simulation environment (spec §162-163).

Lets the whole system run against fake hardware, metrics, network and models so
autonomous behaviour can be exercised without touching the real machine or
needing Ollama. Everything simulated is labelled as such.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from jarvis.models.base import ChatMessage, ChatResponse
from jarvis.models.fake import ScriptedProvider
from jarvis.monitoring.metrics import StaticMetrics


@dataclass
class SimulatedEnvironment:
    provider: ScriptedProvider = field(default_factory=lambda: ScriptedProvider("simulated"))
    metrics: StaticMetrics = field(default_factory=StaticMetrics)
    network_up: bool = True
    network_latency_ms: float = 25.0

    def __post_init__(self) -> None:
        self.provider.responder = self._respond

    async def probe(self) -> tuple[bool, float | None]:
        return (True, self.network_latency_ms) if self.network_up else (False, None)

    def _respond(self, model: str, messages: list[ChatMessage], tools: list[dict[str, Any]] | None) -> ChatResponse | None:
        """A deliberately simple, honest stand-in model: it can call a few read-only tools, otherwise it says
        plainly that it is simulated."""
        if self.provider.rules:
            return None   # explicit test rules take precedence
        last = messages[-1]
        if last.role == "tool":
            try:
                data = json.loads(last.content)
            except ValueError:
                data = {}
            summary = data.get("summary") or data.get("message") or "done"
            return ChatResponse(f"(simulated model) The {last.name} tool reports: {summary}.", model,
                                self.provider.name)
        text = last.content.lower()
        tool_names = {t["function"]["name"] for t in tools or []}
        from jarvis.models.base import ToolCall
        if "system_info" in tool_names and re.search(r"\b(cpu|memory|ram|disk|resources?)\b", text):
            return ChatResponse("", model, self.provider.name, [ToolCall("system_info", {})])
        if "file_list" in tool_names and re.search(r"\b(list|show) (the )?files\b", text):
            return ChatResponse("", model, self.provider.name, [ToolCall("file_list", {"path": "."})])
        if messages and messages[0].role == "system" and "planning component" in messages[0].content:
            return ChatResponse(json.dumps({"steps": [], "notes": "the simulated model cannot plan open-ended work"}),
                                model, self.provider.name)
        return ChatResponse("(simulated model) I'm running in simulation mode without a real language model, so I "
                            "can't reason about that. Deterministic commands work normally — try 'help'.",
                            model, self.provider.name)

    def control(self, command: str) -> str:
        """Handle `/sim ...` commands from the CLI: cpu 95 | memory 97 | disk 98 | model off | network off."""
        parts = command.split()
        if len(parts) >= 2 and parts[0] in ("cpu", "memory", "disk", "gpu", "battery"):
            key = {"cpu": "cpu_percent", "memory": "memory_percent", "disk": "disk_percent", "gpu": "gpu_percent",
                   "battery": "battery_percent"}[parts[0]]
            self.metrics.set(**{key: float(parts[1])})
            return f"simulated {key} = {parts[1]}"
        if len(parts) >= 2 and parts[0] == "model":
            self.provider.online = parts[1] in ("on", "up", "online")
            return f"simulated model provider {'online' if self.provider.online else 'offline'}"
        if len(parts) >= 2 and parts[0] == "network":
            self.network_up = parts[1] in ("on", "up", "online")
            return f"simulated network {'up' if self.network_up else 'down'}"
        return "usage: /sim cpu|memory|disk|gpu|battery <value> | /sim model on|off | /sim network on|off"
