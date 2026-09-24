"""Dynamic context assembly for model calls (spec §87-89, §139).

Layers: system rules and style, current mode, live state, active tasks,
relevant memory, project context, recent conversation, then the request. Each
layer is small and labelled with where it came from; the database is never
dumped wholesale into the prompt.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from jarvis.core import personality, reports
from jarvis.core.services import Services
from jarvis.core.types import Provenance, ProvenanceKind
from jarvis.memory.store import MemoryKind
from jarvis.models.base import ChatMessage
from jarvis.tasks.models import EXECUTING, TaskKind, TaskStatus

CAPABILITIES = ["filesystem read/write/search", "shell commands with risk checks", "process and system inspection",
                "background tasks with checkpoints", "monitoring", "memory", "decision history", "projects",
                "notifications", "model routing"]


@dataclass
class AssembledContext:
    messages: list[ChatMessage]
    provenance: list[Provenance] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)


class ContextAssembler:
    def __init__(self, svc: Services, *, max_chars: int = 12_000, history_turns: int = 12) -> None:
        self.svc = svc
        self.max_chars = max_chars
        self.history_turns = history_turns

    def live_state(self) -> str:
        svc = self.svc
        now = svc.clock.now()
        lines = [f"Time: {time.strftime('%A %d %B %Y %H:%M %Z', time.localtime(now))}",
                 f"Mode: {svc.modes.current.value}" + (" (quiet)" if svc.modes.quiet else "") +
                 (" (private: local only)" if svc.modes.private else "")]
        project = svc.projects.active()
        if project:
            ctx = svc.projects.context(project)
            detail = ", ".join(f"{k}={v}" for k, v in ctx.items() if v and k in ("root", "branch", "languages",
                                                                                 "test_command", "sensitive"))
            lines.append(f"Active project: {project.name} ({detail})")
        r = svc.state.values("resources.")
        if r and not svc.state.is_stale("resources.cpu_percent"):
            lines.append("Resources: " + ", ".join(
                f"{k.split('.', 1)[1]}={v}" for k, v in r.items()
                if k.split(".", 1)[1] in ("cpu_percent", "memory_percent", "disk_percent", "gpu_percent",
                                          "battery_percent", "memory_used_gb", "memory_total_gb")))
        lines.append(f"Network: {svc.state.value('network.state', 'unknown')}; health: {svc.health.overall().label}")
        tasks = svc.tasks.open_tasks()
        if tasks:
            lines.append("Open tasks:")
            for t in tasks[:8]:
                kind = "monitor" if t.kind == TaskKind.MONITOR else t.status.value
                lines.append(f"- [{t.id}] {t.title}: {kind}"
                             + (f", {int(t.compute_progress() * 100)}%" if t.status in EXECUTING and t.plan else "")
                             + (f" ({t.status_reason})" if t.status_reason else ""))
        recent = svc.tasks.list_tasks([TaskStatus.COMPLETED, TaskStatus.FAILED], order="recent", limit=3,
                                      since=now - 6 * 3600)
        if recent:
            lines.append("Recently finished: " + "; ".join(
                f"{t.title} ({t.status.value}: {str(t.outputs.get('summary', ''))[:120]})" for t in recent))
        approvals = svc.approvals.pending()
        if approvals:
            lines.append("Awaiting user approval: " + "; ".join(a.summary for a in approvals[:3]))
        return "\n".join(lines)

    async def build(self, user_text: str, history: list[ChatMessage]) -> AssembledContext:
        svc = self.svc
        eff = svc.modes.effective()
        system = personality.system_prompt(verbosity=svc.modes.verbosity(), humor=eff.policy.humor and svc.config.ui.humor,
                                           use_sir=svc.config.ui.use_sir, mode=eff.mode.value,
                                           capabilities=CAPABILITIES)
        provs = [Provenance(ProvenanceKind.SYSTEM_STATE, "live state")]
        blocks = [system, "COMPUTER: " + environment_note(), "LIVE STATE (observed just now):\n" + self.live_state()]
        project = svc.projects.active()
        memory_ids: list[str] = []
        if svc.config.memory.enabled:
            hits = await svc.memory.retrieve(user_text, project_id=project.id if project else None,
                                             limit=svc.config.memory.max_context_items)
            if hits:
                lines = []
                for h in hits:
                    when = time.strftime("%Y-%m-%d", time.localtime(h.item.updated_at))
                    lines.append(f"- ({h.item.kind.value}, {when}) {h.item.content}")
                    memory_ids.append(h.item.id)
                    provs.append(Provenance(ProvenanceKind.MEMORY, h.item.id, h.item.content[:60]))
                blocks.append("RELEVANT MEMORY (retrieved; may be out of date):\n" + "\n".join(lines))
        lowered = user_text.lower()
        if any(w in lowered for w in ("why", "decid", "chose", "choose", "architecture")):
            decisions = svc.decisions.search(user_text, project_id=project.id if project else None, limit=3)
            if decisions:
                blocks.append("DECISION HISTORY:\n" + "\n".join(f"- {d.title}: {d.explain()}" for d in decisions))
                provs += [Provenance(ProvenanceKind.DATABASE, f"decision {d.id}", d.title) for d in decisions]
        prefs = svc.memory.list(kind=MemoryKind.PREFERENCE, limit=5)
        if prefs:
            blocks.append("USER PREFERENCES:\n" + "\n".join(f"- {p.content}" for p in prefs))
        text = "\n\n".join(blocks)
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n…[context truncated]"
        messages = [ChatMessage("system", text)]
        for past in history[-self.history_turns:]:
            content = past.content if len(past.content) <= 2000 else past.content[:2000] + "…"
            messages.append(ChatMessage(past.role, content))
        messages.append(ChatMessage("user", user_text))
        return AssembledContext(messages, provs, memory_ids)


def environment_note() -> str:
    """Which operating system and shell the tools act on, so the model doesn't reach for `ps aux` on Windows."""
    import platform
    system = platform.system() or sys.platform
    release = platform.release()
    if sys.platform == "win32":
        return (f"{system} {release}. shell_execute runs commands in cmd.exe: use Windows commands (dir, type, "
                "tasklist, where, findstr, ipconfig), not Unix ones (ls, cat, ps, grep). For memory, CPU, disk and "
                "processes prefer the system_info and process_list tools over shell commands.")
    shell = "zsh or sh" if sys.platform == "darwin" else "sh"
    return (f"{system} {release}. shell_execute runs commands in {shell}. For memory, CPU, disk and processes prefer "
            "the system_info and process_list tools over shell commands.")


def summarize_state_for_tool(svc: Services, section: str) -> dict:
    data: dict = {}
    if section in ("resources", "all"):
        data["resources"] = svc.state.values("resources.")
    if section in ("tasks", "all"):
        data["tasks"] = [{"id": t.id, "title": t.title, "status": t.status.value, "progress": t.compute_progress(),
                          "reason": t.status_reason} for t in svc.tasks.open_tasks()[:20]]
    if section in ("health", "all"):
        data["health"] = {c.name: {"status": c.status.label, "detail": c.detail} for c in svc.health.components.values()}
    if section in ("network", "all"):
        data["network"] = svc.state.values("network.")
    if section in ("models", "all"):
        data["models"] = [m.to_dict() for m in svc.router.inventory]
    if section in ("project", "all"):
        project = svc.projects.active()
        data["project"] = svc.projects.context(project) if project else None
    if section == "summary":
        data["summary"] = reports.activity(svc)
    return data
