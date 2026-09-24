"""Context packing (Phase 3 §13).

A model gets the context it needs and no more: the goal and its constraints first, then what the plan has
established, then memory, trimmed to fit the chosen model's context window. Sections are packed in priority
order; a section that doesn't fit is shortened rather than silently dropped, and anything cut is marked so the
model knows the material is partial.
"""

from __future__ import annotations

import json
from typing import Any

CHARS_PER_TOKEN = 4


class ContextPacker:
    def __init__(self, max_tokens: int = 3000) -> None:
        self.max_tokens = max_tokens

    def budget_for(self, context_length: int | None, reserve: int = 1500) -> int:
        """Characters available for context given the model's window, keeping room for the answer."""
        tokens = min(self.max_tokens, max(500, (context_length or 8192) // 2 - reserve))
        return tokens * CHARS_PER_TOKEN

    def pack(self, sections: list[tuple[str, Any]], *, context_length: int | None = None) -> str:
        budget = self.budget_for(context_length)
        out: list[str] = []
        used = 0
        for title, content in sections:
            text = content if isinstance(content, str) else json.dumps(content, default=str, indent=None)
            text = text.strip()
            if not text:
                continue
            block = f"{title}:\n{text}"
            remaining = budget - used
            if remaining <= 80:
                out.append(f"({title} omitted: no room left)")
                break
            if len(block) > remaining:
                block = block[:remaining - 20].rstrip() + " …[truncated]"
            out.append(block)
            used += len(block) + 2
        return "\n\n".join(out)

    def plan_context(self, plan: Any, *, memories: list[str] | None = None, extra: str = "",
                     context_length: int | None = None) -> str:
        """What an agent or the replanner needs to know about a plan in progress."""
        done = [f"- {n.title}: {n.summary or n.status.value}" for n in plan.nodes if n.finished]
        pending = [f"- {n.title}" for n in plan.nodes if not n.finished]
        facts = {k: v for k, v in plan.facts.items() if not k.startswith("_")}
        brief_facts = {k: _brief(v) for k, v in facts.items()}
        return self.pack([
            ("Goal", plan.goal.summary()),
            ("Done so far", "\n".join(done)),
            ("Still to do", "\n".join(pending)),
            ("Assumptions", "\n".join(f"- {a.statement} ({a.status})" for a in plan.assumptions)),
            ("Situation", extra),
            ("What the plan observed", brief_facts),
            ("From memory", "\n".join(f"- {m}" for m in memories or [])),
        ], context_length=context_length)


def _brief(value: Any, limit: int = 400) -> Any:
    text = json.dumps(value, default=str)
    return value if len(text) <= limit else text[:limit] + "…"
