"""Is a usable language model available, and what should the user do if not?

Turns router inventory and provider status into plain guidance at startup and
in ``jarvis doctor`` — e.g. "Ollama is running but no chat model is installed:
run `ollama pull llama3.1:8b`".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jarvis.models.base import NoModelAvailable, Purpose
from jarvis.models.router import ModelRouter, TaskProfile

RECOMMENDED_CHAT = "llama3.1:8b"
RECOMMENDED_SMALL = "llama3.2:3b"
RECOMMENDED_TOOLS = "qwen2.5:7b"
RECOMMENDED_EMBEDDING = "nomic-embed-text"


@dataclass
class ModelReadiness:
    chat_model: str | None = None
    chat_local: bool = True
    tools: bool = False
    embedding_model: str | None = None
    issues: list[str] = field(default_factory=list)    # things that block or limit conversation
    tips: list[str] = field(default_factory=list)      # optional improvements

    @property
    def can_converse(self) -> bool:
        return self.chat_model is not None

    def summary(self) -> str:
        if not self.chat_model:
            return "no language model available"
        where = "local" if self.chat_local else "cloud"
        return f"{self.chat_model} ({where}, {'tools enabled' if self.tools else 'no tool use'})"


def assess(router: ModelRouter, ollama_url: str | None = None) -> ModelReadiness:
    r = ModelReadiness()
    down = [name for name, ok in router.provider_status.items() if not ok]
    for name in down:
        if name == "ollama":
            where = f" at {ollama_url}" if ollama_url else ""
            r.issues.append(f"Ollama isn't running{where}. Start the Ollama app (or run `ollama serve`); I'll connect "
                            "as soon as it's up.")
        else:
            r.issues.append(f"The {name} model provider isn't reachable.")
    try:
        decision = router.select(TaskProfile())
        r.chat_model, r.chat_local = decision.model, decision.local
    except NoModelAvailable:
        if not down:
            r.issues.append(f"Ollama is running but no chat model is installed. In a terminal run "
                            f"`ollama pull {RECOMMENDED_CHAT}` (or `ollama pull {RECOMMENDED_SMALL}` on a computer "
                            "without a graphics card).")
        return r
    try:
        tool_decision = router.select(TaskProfile(needs_tools=True))
        r.tools = True
        if tool_decision.model != r.chat_model:
            r.chat_model = tool_decision.model
    except NoModelAvailable:
        r.issues.append(f"{r.chat_model} can't call tools, so I can talk but can't act on requests (files, commands, "
                        f"tasks). Install a tool-capable model: `ollama pull {RECOMMENDED_TOOLS}`.")
    try:
        r.embedding_model = router.select(TaskProfile(purpose=Purpose.EMBEDDING)).model
    except NoModelAvailable:
        r.tips.append(f"Memory recall uses keyword search. For recall by meaning: `ollama pull {RECOMMENDED_EMBEDDING}`.")
    return r
