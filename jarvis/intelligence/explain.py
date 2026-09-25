"""Explanations built from what actually happened (Phase 3 §21-23, §32, §53-55).

Reports, "what are you doing?", "why did you do that?", plan previews and plan history are composed from the
plan record: the evidence gathered, the decisions and their stated reasons, approvals, verification results
and revisions. Nothing here asks a model to explain itself, so there is no private reasoning to expose and
nothing invented: if a fact isn't in the record, it isn't in the answer.
"""

from __future__ import annotations

from typing import Any

from jarvis.clock import format_time
from jarvis.intelligence.analysis import size_text
from jarvis.intelligence.goals import ExecutionMode
from jarvis.intelligence.plans import NodeKind, NodeStatus, Plan, PlanStatus, Quality
from jarvis.intelligence.quality import describe

N = NodeStatus

_STATUS_WORDS = {
    PlanStatus.CREATED: "being planned", PlanStatus.VALIDATING: "being checked", PlanStatus.READY: "ready to start",
    PlanStatus.RUNNING: "in progress", PlanStatus.WAITING: "waiting", PlanStatus.BLOCKED: "blocked",
    PlanStatus.PAUSED: "paused", PlanStatus.REPLANNING: "being revised", PlanStatus.VERIFYING: "being verified",
    PlanStatus.COMPLETED: "done", PlanStatus.FAILED: "failed", PlanStatus.CANCELLED: "cancelled",
}


def _cap(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:] if text else text


def _end(text: str) -> str:
    text = text.strip()
    return text if not text or text.endswith((".", "!", "?", ":")) else text + "."


def _analysis(plan: Plan) -> dict[str, Any]:
    for node in reversed(plan.nodes):
        if node.kind == NodeKind.ANALYZE and node.status == N.DONE:
            data = (plan.facts.get(node.id) or {}).get("analysis")
            if isinstance(data, dict):
                return data
    return {}


def _latest_verify(plan: Plan):
    verifies = [n for n in plan.nodes if n.kind == NodeKind.VERIFY and n.status == N.DONE
                and not n.meta.get("superseded_by")]
    return verifies[-1] if verifies else None


# -- the report a plan ends with ------------------------------------------------------------------------------

def compose_report(plan: Plan) -> str:
    kind = plan.goal.kind
    if kind == "research":
        return _research_report(plan)
    lines: list[str] = []
    analysis = _analysis(plan)
    if analysis:
        if analysis.get("cause"):
            lines.append(_end(_cap(analysis["cause"])))
        cause = (analysis.get("cause") or "").lower()
        observed = [f.get("text", "") for f in analysis.get("findings", []) if f.get("confidence") == "observed"
                    and f.get("text", "").lower() not in cause]
        if observed and kind != "compound":
            lines.append("Observed: " + "; ".join(observed[:4]) + ".")
    actions = [n for n in plan.nodes if n.kind == NodeKind.ACTION]
    for node in actions:
        lines.append(_action_line(node, with_reason=not analysis))
    decided_nothing = analysis and not actions
    if decided_nothing or plan.mode != ExecutionMode.EXECUTE:
        recommendations = analysis.get("recommendations") or []
        if plan.mode == ExecutionMode.DRY_RUN:
            expected = next((plan.facts[d].get("expected_changes") for d in plan.facts
                             if isinstance(plan.facts[d], dict) and plan.facts[d].get("expected_changes")), None)
            if expected:
                lines.append("Dry run — expected changes: " + "; ".join(f"{e['title']} ({e['would']})"
                                                                    for e in expected) + ". Nothing was changed.")
            else:
                lines.append("Dry run: nothing would need changing. Nothing was changed.")
        elif recommendations:
            lines.append("Recommended: " + "; ".join(recommendations[:3]) + ".")
            if plan.mode == ExecutionMode.EXECUTE and not plan.goal.wants_fix and plan.goal.kind in (
                    "performance", "disk_cleanup"):
                lines.append("Say 'fix it' and I'll do that, asking you first.")
    if kind == "backup" or kind == "compound" or not analysis:
        lines += _step_lines(plan, skip_kinds=(NodeKind.ACTION,) if actions and analysis else ())
    verify = _latest_verify(plan)
    if verify is not None:
        detail = (plan.facts.get(verify.id) or {}).get("verdict", {}).get("detail", "")
        if detail:
            lines.append(f"Checked independently: {_end(detail)}")
    notes = list(analysis.get("notes", [])) if analysis else []
    for c in plan.goal.constraints:
        if c.kind == "protect_process":
            notes.append(f"I left {c.value} alone, as you asked")
    lessons = plan.facts.get("_notes") or []
    for text in notes[:3]:
        lines.append(_end(_cap(text)))
    for text in lessons[:2]:
        if "tried" in text or "Last time" in text:
            lines.append(_end(text))
    if any(n.kind == NodeKind.ACTION and n.status == N.DONE for n in plan.nodes) or plan.quality is not None:
        lines.append(f"Result: {describe(plan.quality)}.")
    return "\n".join(line for line in lines if line) or "Nothing to report."


def details(plan: Plan) -> str:
    """Everything the plan found, in full ("what are these files?"): the observations and each candidate with its
    evidence, not just the one-line summary a notification carries."""
    if plan.goal.kind == "research":
        return _research_report(plan)
    analysis = _analysis(plan)
    if not analysis:
        lines = _step_lines(plan)
        return "\n".join([f"{plan.title} ({_STATUS_WORDS[plan.status]}):"] + lines) if lines else \
            f"{plan.title} hasn't found anything yet ({_STATUS_WORDS[plan.status]})."
    lines = [f"{plan.title} ({_STATUS_WORDS[plan.status]}): {_end(analysis.get('cause') or 'here is what I found')}"]
    observed = [f.get("text", "") for f in analysis.get("findings", []) if f.get("confidence") == "observed"]
    if observed:
        lines.append("Observed: " + "; ".join(observed[:6]) + ".")
    candidates = [c for c in analysis.get("candidates", []) if isinstance(c, dict)]
    usable = [c for c in candidates if not c.get("blocked")]
    if usable:
        lines.append("Candidates:" if plan.goal.kind != "disk_cleanup" else "The files:")
        for i, c in enumerate(usable[:10], 1):
            lines.append(f"{i}. {_cap(c.get('reason') or c.get('title', ''))}")
    for c in [c for c in candidates if c.get("blocked")][:3]:
        lines.append(f"Not suggesting {c.get('title')}: {c['blocked']}.")
    done = [n for n in plan.nodes if n.kind == NodeKind.ACTION and n.status == N.DONE]
    if done:
        lines += [_action_line(n) for n in done]
    elif plan.terminal:
        lines.append("Nothing was changed.")
    return "\n".join(lines)


def candidates(plan: Plan) -> list[dict[str, Any]]:
    return [c for c in _analysis(plan).get("candidates", []) if isinstance(c, dict) and not c.get("blocked")]


def _action_line(node, *, with_reason: bool = True) -> str:
    title = node.title[:1].lower() + node.title[1:]
    if node.status == N.DONE:
        return f"I {_past(title)}" + (f" ({node.meta['reason']})" if with_reason and node.meta.get("reason")
                                      else "") + "."
    if node.status == N.SKIPPED and "declin" in node.note:
        return f"You declined to {title}, so I didn't."
    if node.status == N.SKIPPED:
        return f"I didn't {title}: {node.note}."
    if node.status == N.FAILED:
        return f"Trying to {title} failed: {node.error or node.note}."
    if node.status == N.CANCELLED:
        return f"I didn't {title} ({node.note or 'cancelled'})."
    return f"Still to do: {title}."


def _past(verb_phrase: str) -> str:
    irregular = {"stop": "stopped", "unload": "unloaded", "move": "moved", "copy": "copied", "run": "ran",
                 "build": "built", "delete": "deleted", "restart": "restarted", "kill": "killed", "free": "freed"}
    first, _, rest = verb_phrase.partition(" ")
    return f"{irregular.get(first, first + 'ed')} {rest}".strip()


def _step_lines(plan: Plan, skip_kinds: tuple = ()) -> list[str]:
    out = []
    for n in plan.nodes:
        if n.kind in (NodeKind.REPORT, NodeKind.DECIDE, NodeKind.GATE, NodeKind.VERIFY) or n.kind in skip_kinds:
            continue
        if n.status == N.DONE:
            mark = "✓"
            detail = n.summary
            tests = (plan.facts.get(n.id) or {}).get("tests")
            if tests and tests.get("summary"):
                detail = tests["summary"]
            out.append(f"{mark} {n.title}" + (f" — {detail}" if detail else ""))
        elif n.status == N.FAILED:
            out.append(f"✗ {n.title} — {n.error or n.note or 'failed'}")
        elif n.status in (N.SKIPPED, N.CANCELLED):
            out.append(f"– {n.title} (not run: {n.note or n.status.value})")
    return out


def _research_report(plan: Plan) -> str:
    verify = _latest_verify(plan)
    result = (plan.facts.get(verify.id) or {}).get("verdict", {}) if verify else {}
    analysis = _analysis(plan)
    claims = result.get("supported") if result else None
    if claims is None:
        claims = [{k: c.get(k) for k in ("text", "source", "line")} for c in analysis.get("claims", [])]
    lines = []
    topic = plan.goal.target or plan.goal.objective
    if not claims:
        lines.append(f"I found nothing about {topic} in the sources I searched.")
    else:
        sources = sorted({c["source"] for c in claims if c.get("source")})
        lines.append(f"What the sources say about {topic} ({len(claims)} statement(s) from {len(sources)} file(s)):")
        for i, c in enumerate(claims[:12], 1):
            where = f"{c.get('source')}:{c.get('line')}" if c.get("line") else str(c.get("source"))
            lines.append(f"[{i}] {c['text']} ({where})")
    for conflict in analysis.get("conflicts", [])[:3]:
        a, b = conflict["a"], conflict["b"]
        lines.append(f"Sources disagree: \"{a['text']}\" ({a['source']}) vs \"{b['text']}\" ({b['source']}).")
    unsupported = result.get("unsupported") or []
    if unsupported:
        lines.append(f"Left out {len(unsupported)} statement(s) I couldn't find in the cited source.")
    summary_node = plan.node("summary")
    if summary_node and summary_node.status == N.DONE:
        text = (plan.facts.get("summary") or {}).get("written") or {}
        report = text.get("report") if isinstance(text, dict) else None
        if report:
            lines.append(f"Summary (written by {text.get('model', 'the model')} from the statements above; check the "
                         f"cited sources): {report.strip()}")
    if verify is not None:
        lines.append(f"Result: {describe(verify.quality)} — {result.get('detail', '')}".rstrip(" —") + ".")
    return "\n".join(lines)


# -- "what are you doing?" ------------------------------------------------------------------------------------

def status_line(plan: Plan) -> str:
    done = sum(1 for n in plan.nodes if n.finished)
    total = len(plan.nodes)
    words = _STATUS_WORDS[plan.status]
    text = f"{plan.title}: {words} ({done}/{total} steps)"
    running = [n.title for n in plan.nodes if n.status == N.RUNNING]
    if plan.status == PlanStatus.RUNNING and running:
        text += f" — now: {', '.join(running[:2])}"
    elif plan.status in (PlanStatus.WAITING, PlanStatus.BLOCKED, PlanStatus.PAUSED) and plan.status_reason:
        text += f" — {plan.status_reason}"
    return text


def activity(plans: list[Plan]) -> str:
    open_plans = [p for p in plans if not p.terminal]
    if not open_plans:
        return ""
    return " ".join(_end(status_line(p)) for p in open_plans[:3])


# -- "why did you do that?" -------------------------------------------------------------------------------------

def why(plan: Plan) -> str:
    """The most recent decision in the plan, with its evidence, the approval and what verification found."""
    decision = next((d for d in reversed(plan.decisions)), None)
    if decision is None:
        if plan.replans:
            r = plan.replans[-1]
            return _end(f"I revised '{plan.title}' because {r['reason']}: {'; '.join(r['changed'])}")
        return _end(f"{plan.title} is {_STATUS_WORDS[plan.status]}" + (f" because {plan.status_reason}"
                                                                       if plan.status_reason else ""))
    parts = []
    acted = [a for a in decision.get("actions", [])]
    if acted:
        titles = "; ".join(a["title"] for a in acted)
        parts.append(f"I decided to {titles} because {decision['reason'].rstrip('.')}")
    else:
        parts.append(f"I decided to {decision['decision']} because {decision['reason'].rstrip('.')}")
    if decision.get("evidence"):
        parts.append("The evidence: " + "; ".join(decision["evidence"][:3]))
    ruled_out = [a for a in decision.get("alternatives", []) if "ruled out" in a]
    if ruled_out:
        parts.append("I didn't choose: " + "; ".join(ruled_out[:2]))
    gate = next((a for a in reversed(plan.approvals) if a.get("kind") == "gate"), None)
    if acted and gate:
        who = gate.get("by") or "you"
        who = "you" if str(who).startswith("user:") or who in ("owner", "you") else who
        parts.append(f"{_cap(who)} {'approved' if gate.get('approved') else 'declined'} it at {format_time(gate['ts'])}")
    verify = _latest_verify(plan)
    if acted and verify is not None:
        detail = (plan.facts.get(verify.id) or {}).get("verdict", {}).get("detail", "")
        parts.append(f"Afterwards, an independent check found: {detail} ({describe(verify.quality)})")
    elif acted and plan.status not in (PlanStatus.COMPLETED, PlanStatus.FAILED):
        parts.append("It hasn't been verified yet")
    if plan.replans:
        r = plan.replans[-1]
        parts.append(f"Then I revised the plan: {'; '.join(r['changed'])}")
    return " ".join(_end(p) for p in parts)


# -- previews and history -----------------------------------------------------------------------------------------

def preview(plan: Plan) -> str:
    lines = [f"Plan: {plan.title}."]
    step_no = 0
    for wave in plan.waves():
        titles = []
        for n in wave:
            label = n.title
            if n.kind == NodeKind.GATE:
                label = f"ask you before: {n.steps[0]['args'].get('summary', 'the changes')}"
            elif n.kind == NodeKind.DECIDE and plan.goal.wants_fix:
                label += " (anything I change, I'll ask you first)"
            elif n.kind == NodeKind.VERIFY:
                label += " (a separate check, not my own word)"
            titles.append(label)
        step_no += 1
        lines.append(f"{step_no}. " + ("; in parallel: " if len(titles) > 1 else "") + "; ".join(titles))
    if plan.goal.constraints:
        lines.append("Respecting: " + "; ".join(c.describe() for c in plan.goal.constraints) + ".")
    if plan.goal.deadline:
        lines.append(f"Deadline: {format_time(plan.goal.deadline)}.")
    for note in (plan.facts.get("_notes") or [])[:2]:
        lines.append(_end(note))
    estimate = sum(float(n.estimate.get("seconds", 0)) for n in plan.nodes)
    if estimate:
        lines.append(f"Rough time: {_duration(estimate)}.")
    return "\n".join(lines)


def history(plans: list[Plan]) -> str:
    if not plans:
        return "I haven't run any plans yet."
    lines = []
    for p in plans[:8]:
        when = format_time(p.created_at) if p.created_at else ""
        quality = f", {describe(p.quality)}" if p.quality else ""
        revised = f", revised {len(p.replans)}×" if p.replans else ""
        lines.append(f"{when} {p.title} — {_STATUS_WORDS[p.status]}{quality}{revised}".strip())
    return "\n".join(lines)


def changes(plan: Plan) -> str:
    if not plan.replans:
        return f"I haven't changed the plan for {plan.title}."
    return " ".join(_end(f"Revision {r['version']} ({r['trigger'].replace('_', ' ')}): {'; '.join(r['changed'])}")
                    for r in plan.replans[-3:])


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"under {max(1, int(round(seconds / 10.0)) * 10)} seconds" if seconds > 10 else "a few seconds"
    return f"about {int(round(seconds / 60))} minutes"


def disk_line(usage: dict[str, Any]) -> str:
    return f"{usage.get('percent', 0):.0f}% used, {size_text(usage.get('free'))} free"


def quality_word(quality: Quality | None) -> str:
    return describe(quality)
