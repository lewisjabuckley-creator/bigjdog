"""Response layer: the voice is the final layer, not the architecture (spec §71-77, §195).

Calm, precise, understated. Concise by default, detailed when asked, terse in
emergencies. No filler openers, no theatrical grandeur, no movie quotations.
Internal state stays structured; this module turns it into language.
"""

from __future__ import annotations

import re

from jarvis.clock import format_duration

_FILLER_OPENERS = re.compile(
    r"^\s*(certainly|absolutely|of course|sure thing|sure|great question|excellent question|happy to help|"
    r"i'd be happy to help|no problem)[!,.]*\s*", re.IGNORECASE)
_FILLER_CLOSERS = re.compile(
    r"\s*(let me know if (there'?s|you need|you have) (anything|any)[^.]*\.|i hope (this|that) helps[^.]*\.|"
    r"feel free to ask[^.]*\.|is there anything else[^?]*\?)\s*$", re.IGNORECASE)


def system_prompt(*, verbosity: str = "normal", humor: bool = True, use_sir: bool = False, mode: str = "normal",
                  capabilities: list[str] | None = None) -> str:
    lines = [
        "You are JARVIS, the conversational layer of a local-first AI operating environment running on the "
        "user's own computer. You are not the whole system: live state, memory, tasks, tools, permissions and "
        "monitoring are separate components, and you see their current state in the context below.",
        "Behave like an intelligent operating system, not a chatbot with a persona.",
        "Style: calm, precise, understated, confident without pretending certainty. Plain sentences; avoid "
        "bullet lists in conversation unless listing genuinely separate items. Never open with filler such as "
        "'Certainly' or 'Absolutely'. No movie references, no theatrical grandeur, no fake British affectation.",
        "Honesty rules: never claim to have run a tool, read a file or checked a source unless a tool result in "
        "this conversation shows it. Distinguish observed facts, inferences and estimates. If something failed "
        "or was not verified, say so. If you cannot do something, say so and offer what you can do instead.",
        "Use tools to inspect reality instead of guessing about files, processes, services or system state.",
        "Consequential actions (deleting, sending, deploying, installing, stopping processes) require the "
        "user's authorization; the permission system will pause and ask. Do not try to work around it.",
    ]
    lines.append({
        "short": "Be brief: one to three sentences unless more is essential.",
        "detailed": "The user wants detail: explain thoroughly, including reasoning and trade-offs.",
    }.get(verbosity, "Be concise; expand only when the question needs it."))
    if mode == "emergency":
        lines.append("EMERGENCY MODE: be extremely concise. State only what matters and what to do. No humor.")
    elif humor:
        lines.append("Occasional dry, restrained humor is acceptable when context invites it; competence dominates.")
    else:
        lines.append("No humor right now.")
    if use_sir:
        lines.append("You may occasionally address the user as 'sir', sparingly and never mechanically.")
    if capabilities:
        lines.append("Implemented capabilities: " + ", ".join(capabilities) + ". Anything else is not implemented.")
    return "\n".join(lines)


def strip_opener(text: str) -> str:
    """Remove a leading filler opener ("Certainly!", "Absolutely,")."""
    stripped = _FILLER_OPENERS.sub("", text, count=1)
    if stripped != text and stripped and stripped[0].islower():
        stripped = stripped[0].upper() + stripped[1:]
    return stripped


def clean(text: str) -> str:
    """Strip filler that makes responses sound like a chatbot."""
    text = strip_opener(text.strip())
    text = _FILLER_CLOSERS.sub("", text)
    return text.strip()


def join_clauses(parts: list[str]) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?", ":")) else text + "."


def percent(progress: float) -> str:
    return f"{int(round(progress * 100))}%"


def failure_report(*, what: str, why: str = "", done: list[str] | None = None, not_done: list[str] | None = None,
                   actions: list[str] | None = None, next_steps: list[str] | None = None, short: bool = True) -> str:
    """Error communication format (spec §149): what failed, why, what completed, what didn't, what I did, next."""
    out = [sentence(f"{what} failed" + (f": {why}" if why else ""))]
    if done:
        out.append(sentence(f"Completed: {join_clauses(done)}"))
    if not_done:
        out.append(sentence(f"Not completed: {join_clauses(not_done)}"))
    if actions and not short:
        out.append(sentence(f"I {join_clauses(actions)}"))
    if next_steps:
        out.append(sentence(f"Next: {join_clauses(next_steps)}"))
    return " ".join(out)


def duration(seconds: float) -> str:
    return format_duration(seconds)
