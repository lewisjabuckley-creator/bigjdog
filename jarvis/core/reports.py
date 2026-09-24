"""Deterministic answers built from live state (spec §47-49, §75, §114-117, §151, §191).

"What are you doing?", "Where were we?", "What changed?", "Why did you do
that?", "What's wrong?" are answered from the task system, audit log, event
store and live state — never from the language model's recollection — so they
stay accurate even when no model is available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import psutil

from jarvis.clock import format_datetime, format_duration, format_time
from jarvis.core.personality import join_clauses, percent, sentence
from jarvis.core.services import Services
from jarvis.core.types import Confidence, NotificationPriority, Severity
from jarvis.events.types import EventType
from jarvis.tasks.models import EXECUTING, Task, TaskKind, TaskStatus

S = TaskStatus


def _label(task: Task) -> str:
    title = task.title.strip()
    if len(title) > 1 and title[:2].isupper():   # keep acronyms ("CI pipeline")
        return title
    return title[:1].lower() + title[1:]


def _elapsed(svc: Services, task: Task) -> str:
    if task.started_at is None:
        return ""
    return format_duration(svc.clock.now() - task.started_at)


# -- activity ------------------------------------------------------------------------------------

def activity(svc: Services) -> str:
    """Answer to "What are you doing?" — operational status, not chain-of-thought (spec §48)."""
    open_tasks = svc.tasks.open_tasks()
    running = [t for t in open_tasks if t.status in EXECUTING and t.kind != TaskKind.MONITOR]
    monitors = [t for t in open_tasks if t.kind == TaskKind.MONITOR and t.status in EXECUTING | {S.QUEUED}]
    waiting = [t for t in open_tasks if t.status == S.WAITING]
    blocked = [t for t in open_tasks if t.status == S.BLOCKED]
    queued = [t for t in open_tasks if t.status == S.QUEUED and t.kind != TaskKind.MONITOR]
    paused = [t for t in open_tasks if t.status == S.PAUSED]
    parts = []
    if running:
        items = []
        for t in running[:4]:
            detail = percent(t.compute_progress()) if t.plan and len(t.plan) > 1 else ""
            items.append(f"{_label(t)}" + (f" ({detail})" if detail else ""))
        parts.append(f"I'm working on {join_clauses(items)}")
    if monitors:
        parts.append(f"I'm monitoring {join_clauses([_label(t).removeprefix('watch ') for t in monitors[:4]])}")
    sentences = [sentence(p) for p in parts]
    if waiting:
        sentences.append(sentence(f"Waiting on {join_clauses([_label(t) + ' (' + t.status_reason + ')' for t in waiting[:3]])}"))
    if queued:
        sentences.append(sentence(f"{len(queued)} task{'s are' if len(queued) != 1 else ' is'} queued"))
    if paused:
        sentences.append(sentence(f"Paused: {join_clauses([_label(t) for t in paused[:3]])}"))
    if blocked:
        sentences.append(sentence(f"Blocked: {join_clauses([_label(t) + ' — ' + t.status_reason for t in blocked[:3]])}"))
    elif running or monitors:
        sentences.append("Nothing is blocked.")
    if not sentences:
        sentences.append("Nothing is running right now.")
    pending = svc.notifications.pending()
    if pending:
        sentences.append(f"{len(pending)} update{'s' if len(pending) != 1 else ''} waiting for you.")
    return " ".join(sentences)


def task_status(svc: Services, task: Task) -> str:
    title = task.title[:1].upper() + task.title[1:]
    if task.status in (S.COMPLETED, S.FAILED, S.CANCELLED):
        summary = task.outputs.get("summary") or task.status_reason
        when = f" {format_duration(svc.clock.now() - task.finished_at)} ago" if task.finished_at else ""
        outcome = f" ({task.outcome.value})" if task.outcome and task.status == S.COMPLETED and \
            task.outcome.value != "complete" else ""
        return sentence(f"{title} {task.status.value}{outcome}{when}: {summary}")
    if task.kind == TaskKind.MONITOR:
        obs = (task.monitor.last_observation or {}).get("detail") if task.monitor else None
        return sentence(f"I'm monitoring {_label(task)}" + (f"; last check: {obs}" if obs else ""))
    parts = [f"{title} is {task.status.value}"]
    if task.plan:
        done = len([s for s in task.plan if s.finished])
        parts.append(f"{percent(task.compute_progress())} through ({done} of {len(task.plan)} steps)")
    current = task.current_step
    if current and task.status in EXECUTING:
        parts.append(f"currently on '{current.description}'")
    elapsed = _elapsed(svc, task)
    if elapsed:
        parts.append(f"running for {elapsed}")
    text = ", ".join(parts)
    progress = task.compute_progress()
    if task.started_at and 0 < progress < 1 and task.status in EXECUTING:
        spent = svc.clock.now() - task.started_at
        remaining = spent / progress - spent
        text += f". At this rate roughly {format_duration(remaining)} remain (estimate)"
    if task.status in (S.WAITING, S.BLOCKED, S.PAUSED) and task.status_reason:
        text += f". {task.status_reason[:1].upper()}{task.status_reason[1:]}"
    explanation = svc.resources.explain(task.id)
    if explanation and task.status in (S.QUEUED, S.PAUSED):
        text += f". {explanation.sentence()}"
    return sentence(text)


# -- system status ------------------------------------------------------------------------------

def _fmt(value: Any, suffix: str = "%") -> str:
    return f"{value:.0f}{suffix}" if isinstance(value, (int, float)) else "n/a"


def model_line(svc: Services) -> str:
    if not svc.router.available():
        down = [p for p, ok in svc.router.provider_status.items() if not ok]
        return "unavailable" + (f" ({', '.join(down)} offline)" if down else "")
    try:
        from jarvis.models.router import TaskProfile
        decision = svc.router.select(TaskProfile())
        return f"{decision.model} ({'local' if decision.local else 'cloud'})"
    except Exception:
        return "unavailable"


def status_report(svc: Services) -> str:
    """Compact operational status block (spec §75)."""
    r = svc.state.values("resources.")
    lines = ["SYSTEM", f"Model: {model_line(svc)}"]
    cpu = r.get("resources.cpu_percent")
    mem_used, mem_total = r.get("resources.memory_used_gb"), r.get("resources.memory_total_gb")
    lines.append(f"CPU: {_fmt(cpu)}")
    if isinstance(mem_used, (int, float)) and isinstance(mem_total, (int, float)):
        lines.append(f"Memory: {mem_used:.1f} GB / {mem_total:.0f} GB")
    lines.append(f"Disk: {_fmt(r.get('resources.disk_percent'))}")
    if "resources.gpu_percent" in r:
        lines.append(f"GPU: {_fmt(r.get('resources.gpu_percent'))}")
    lines.append(f"Network: {str(svc.state.value('network.state', 'unknown')).replace('_', ' ')}")
    lines.append(f"Mode: {svc.modes.current.value}{' (quiet)' if svc.modes.quiet else ''}"
                 f"{' (private)' if svc.modes.private else ''}")
    if svc.state.is_stale("resources.cpu_percent") and cpu is not None:
        lines.append("(resource figures are stale — monitor not reporting)")
    lines.append("")
    lines.append("TASKS")
    tasks = svc.tasks.open_tasks()
    recent_done = svc.tasks.list_tasks([S.COMPLETED, S.FAILED], order="recent", limit=3,
                                       since=svc.clock.now() - 3600)
    if not tasks and not recent_done:
        lines.append("None")
    for t in tasks[:8]:
        extra = f" {percent(t.compute_progress())}" if t.status in EXECUTING and len(t.plan) > 1 else ""
        lines.append(f"{t.title[:40]}: {'monitoring' if t.kind == TaskKind.MONITOR else t.status.value}{extra}")
    for t in recent_done:
        lines.append(f"{t.title[:40]}: {t.status.value}")
    lines.append("")
    lines.append("ALERTS")
    alerts = [n for n in svc.notifications.pending() if n.priority >= NotificationPriority.IMPORTANT]
    unhealthy = svc.health.unhealthy()
    if not alerts and not unhealthy:
        lines.append("No critical alerts.")
    for c in unhealthy[:5]:
        lines.append(f"{c.name}: {c.status.label} {('— ' + c.detail) if c.detail else ''}".rstrip())
    for n in alerts[:5]:
        lines.append(n.text())
    return "\n".join(lines)


def are_we_good(svc: Services) -> str:
    unhealthy = svc.health.unhealthy()
    open_tasks = svc.tasks.open_tasks()
    blocked = [t for t in open_tasks if t.status in (S.BLOCKED, S.WAITING)]
    failed = svc.tasks.list_tasks([S.FAILED], since=svc.clock.now() - 6 * 3600, limit=5)
    alerts = [n for n in svc.notifications.pending() if n.priority >= NotificationPriority.IMPORTANT]
    issues = []
    for c in unhealthy:
        issues.append(f"{c.name} is {c.status.label}" + (f" ({c.detail})" if c.detail else ""))
    for t in blocked:
        issues.append(f"{_label(t)} is {t.status.value}: {t.status_reason}")
    for t in failed:
        issues.append(f"{_label(t)} failed {format_duration(svc.clock.now() - (t.finished_at or t.updated_at))} ago")
    if not issues and not alerts:
        running = [t for t in open_tasks if t.status in EXECUTING]
        tail = f" {len(running)} task{'s are' if len(running) != 1 else ' is'} running normally." if running else ""
        return "All good. Every monitored subsystem is healthy and nothing needs your attention." + tail
    if issues:
        lead = "Mostly." if len(issues) == 1 and not unhealthy else "Not entirely."
        text = f"{lead} {sentence(join_clauses(issues))}"
    else:
        text = "Everything is healthy."
    if alerts:
        text += f" {len(alerts)} update{'s' if len(alerts) != 1 else ''} waiting: " + "; ".join(n.text() for n in alerts[:3])
    return text


# -- re-entry, briefing, changes ------------------------------------------------------------------

def last_seen(svc: Services) -> float:
    value = svc.state.value("session.last_seen")
    return float(value) if isinstance(value, (int, float)) else svc.clock.now() - 86400


def reentry(svc: Services) -> str:
    """Answer to "Where were we?" (spec §151)."""
    parts = []
    project = svc.projects.active()
    if project:
        ctx = svc.projects.context(project)
        branch = f" on branch {ctx['branch']}" if ctx.get("branch") else ""
        parts.append(f"We're in the {project.name} project{branch}.")
    open_user = [t for t in svc.tasks.open_tasks() if t.created_by.startswith("user") and t.kind != TaskKind.MONITOR]
    recent = svc.tasks.list_tasks([S.COMPLETED, S.FAILED, S.CANCELLED], order="recent", limit=5,
                                  since=svc.clock.now() - 3 * 86400)
    if open_user:
        current = open_user[0]
        parts.append(f"The current objective is {_label(current)} ({current.status.value}).")
        done = [s.description for s in current.completed_steps()]
        pending = [s.description for s in current.pending_steps()]
        if current.recovery:
            parts.append(current.recovery)
        else:
            if done:
                parts.append(sentence(f"Completed so far: {join_clauses(done[-4:])}"))
            if pending:
                parts.append(sentence(f"Still pending: {join_clauses(pending[:4])}"))
    elif recent:
        last = recent[0]
        parts.append(sentence(f"The last thing we did was {_label(last)}, which {last.status.value}: "
                              f"{last.outputs.get('summary') or last.status_reason}"))
    blockers = [t for t in svc.tasks.open_tasks() if t.status in (S.BLOCKED, S.WAITING)]
    if blockers:
        parts.append(sentence(f"Waiting on you: {join_clauses([_label(t) + ' (' + t.status_reason + ')' for t in blockers[:3]])}"))
    since = last_seen(svc)
    events = svc.events.query(since=since, min_severity=Severity.WARNING, limit=5)
    if events:
        parts.append(sentence(f"Since you were last here: {join_clauses([_event_text(e) for e in events[:3]])}"))
    if not parts:
        return "There's no active project or open work. We're starting fresh."
    return " ".join(parts)


def _event_text(e: Any) -> str:
    p = e.payload
    for key in ("message", "summary", "reason", "detail"):
        if p.get(key):
            title = p.get("title")
            return f"{title}: {p[key]}" if title and title not in str(p[key]) else str(p[key])
    if p.get("title"):
        return f"{p['title']} ({str(e.type).lower().replace('_', ' ')})"
    return str(e.type).lower().replace("_", " ")


def what_changed(svc: Services, since: float | None = None) -> str:
    since = since if since is not None else last_seen(svc)
    actions = [a for a in svc.audit.query(since=since, limit=200) if a.action == "tool_execute" and a.ok]
    effects = [a for a in actions if a.tool not in ("file_read", "file_list", "file_search", "system_info",
                                                     "process_list", "process_inspect", "time_now")]
    files = sorted({a.params.get("path") for a in effects if a.tool in ("file_write", "file_delete")
                    and a.params.get("path")})
    commands = [a.params.get("command") for a in effects if a.tool == "shell_execute"]
    finished = svc.tasks.list_tasks([S.COMPLETED, S.FAILED, S.CANCELLED], since=since, order="recent", limit=10)
    events = svc.events.query(since=since, types=[EventType.MODE_CHANGED, EventType.PROJECT_CHANGED,
                                                  EventType.FILE_CREATED, EventType.FILE_CHANGED,
                                                  EventType.FILE_DELETED, EventType.PERMISSION_GRANTED,
                                                  EventType.MODEL_UNAVAILABLE, EventType.NETWORK_CHANGED],
                              limit=50)
    lines = []
    if files:
        lines.append(f"Files changed by me: {join_clauses(files[:6])}" + (f" and {len(files) - 6} more" if len(files) > 6 else ""))
    if commands:
        lines.append(f"Commands run: {join_clauses([f'`{c}`' for c in commands[:4]])}")
    if finished:
        lines.append("Tasks finished: " + join_clauses([f"{_label(t)} ({t.status.value})" for t in finished[:5]]))
    watched = [e for e in events if str(e.type).startswith("FILE_")]
    if watched:
        lines.append(f"{len(watched)} file change{'s' if len(watched) != 1 else ''} observed in watched locations")
    other = [e for e in events if not str(e.type).startswith("FILE_")]
    for e in other[:4]:
        p = e.payload
        if e.type == EventType.MODE_CHANGED and p.get("to"):
            lines.append(f"Mode changed to {p['to']}")
        elif e.type == EventType.PROJECT_CHANGED:
            lines.append(f"Switched to project {p.get('name')}")
        elif e.type == EventType.PERMISSION_GRANTED:
            lines.append(f"Authority granted: {p.get('scope')}")
        elif e.type == EventType.NETWORK_CHANGED:
            lines.append(f"Network became {p.get('state')}")
        elif e.type == EventType.MODEL_UNAVAILABLE:
            lines.append(f"Model provider {p.get('provider')} went unavailable")
    if not lines:
        return f"Nothing has changed since {format_time(since)}."
    return f"Since {format_time(since)}: " + "; ".join(lines) + "."


def briefing(svc: Services) -> str:
    """Concise morning briefing (spec §62), rendered from the same data the scheduled briefing stores."""
    from jarvis.core.awareness import briefing_data, render_briefing
    parts = [render_briefing(briefing_data(svc, since=last_seen(svc)))]
    pending = svc.notifications.drain(limit=5)
    if pending:
        parts.append("Updates: " + "; ".join(n.text() for n in pending) + ".")
    return " ".join(parts)


# -- explanations ------------------------------------------------------------------------------------

def why(svc: Services, task: Task | None = None) -> str:
    """Operational explanation of the most recent autonomous decision (spec §49)."""
    if task is not None:
        reason = svc.resources.explain(task.id)
        if reason:
            return reason.sentence()
        entries = [e for e in svc.audit.query(task_id=task.id, limit=20) if e.reason]
        decisions = [e for e in entries if e.actor.startswith("system:")] or entries
        if decisions:
            return decisions[0].reason_sentence() or decisions[0].summary
        if task.status_reason:
            return sentence(f"{task.title} is {task.status.value} because {task.status_reason}")
    entry = svc.audit.last_decision(since=svc.clock.now() - 6 * 3600)
    if entry is None:
        entry = svc.audit.last_with_reason()
    if entry is None:
        return "I haven't taken any autonomous action that needs explaining."
    return entry.reason_sentence() or entry.summary


def what_did_you_do(svc: Services, since: float | None = None) -> str:
    since = since if since is not None else svc.clock.now() - 24 * 3600
    entries = svc.audit.query(since=since, limit=100)
    if not entries:
        return "I haven't taken any actions recently."
    tool_runs = [e for e in entries if e.action == "tool_execute"]
    denied = [e for e in entries if e.action == "tool_denied"]
    decisions = [e for e in entries if e.actor.startswith("system:")]
    parts = []
    if tool_runs:
        ok = [e for e in tool_runs if e.ok]
        failed = [e for e in tool_runs if not e.ok]
        recent = [e.summary for e in tool_runs[:5]]
        parts.append(f"I ran {len(tool_runs)} action{'s' if len(tool_runs) != 1 else ''} ({len(ok)} succeeded"
                     f"{f', {len(failed)} failed' if failed else ''}). Most recent: " + "; ".join(recent))
    if decisions:
        parts.append("Decisions I made on my own: " + "; ".join(
            (e.reason_sentence() or e.summary).rstrip(".") for e in decisions[:3]))
    if denied:
        parts.append(f"{len(denied)} action{'s were' if len(denied) != 1 else ' was'} refused by policy: " +
                     "; ".join(e.summary for e in denied[:2]))
    return " ".join(sentence(p) for p in parts)


def provenance(svc: Services, provs: list[Any]) -> str:
    if not provs:
        return "That came from my own general knowledge (the language model), not from anything I checked."
    labels = []
    for p in provs:
        text = p.describe()
        if text not in labels:
            labels.append(text)
    return sentence("That came from " + join_clauses(labels[:6]))


# -- diagnostics ---------------------------------------------------------------------------------------

@dataclass
class Diagnosis:
    observed: list[str] = field(default_factory=list)
    causes: list[tuple[str, Confidence]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    subject: str = ""

    def format(self, short: bool = False) -> str:
        if not self.observed and not self.causes:
            return "I checked and found nothing wrong: subsystems are healthy, no recent failures and resources are normal."
        out = []
        if self.observed:
            out.append("Observed: " + "; ".join(self.observed[:6 if not short else 3]) + ".")
        if self.causes:
            out.append("Likely cause: " + "; ".join(f"{c} ({conf.value})" for c, conf in self.causes[:3]) + ".")
        if self.actions:
            out.append("Recommended: " + "; ".join(self.actions[:3]) + ".")
        return " ".join(out)


_ERROR_HINTS = [
    ("no such file", "a file or path the step needed does not exist", "check the path or create the missing file"),
    ("does not exist", "a file or path the step needed does not exist", "check the path or create the missing file"),
    ("not found", "a command, file or resource was not found", "check that the tool or file is installed/present"),
    ("permission denied", "the operation lacked filesystem or OS permissions", "check file permissions"),
    ("timeout", "the operation took longer than its time limit", "retry with a longer timeout or investigate the hang"),
    ("timed out", "the operation took longer than its time limit", "retry with a longer timeout or investigate the hang"),
    ("connection refused", "a service it depends on is not accepting connections",
     "check that the service is running"),
    ("out of memory", "the process ran out of memory", "free memory or use a smaller workload/model"),
    ("modulenotfounderror", "a Python dependency is missing", "install the missing package in the project environment"),
    ("syntaxerror", "there is a syntax error in the code", "fix the syntax error reported in the output"),
]


def diagnose(svc: Services, task: Task | None = None) -> Diagnosis:
    """Gather evidence before concluding (spec §33, §115)."""
    d = Diagnosis(subject=task.title if task else "system")
    if task is not None:
        d.observed.append(f"{task.title} is {task.status.value}" + (f" ({task.status_reason})" if task.status_reason else ""))
        tests = task.outputs.get("tests")
        if tests and not tests.get("ok"):
            failing = tests.get("failures", [])
            d.observed.append(f"tests: {tests.get('summary')}")
            d.causes.append((f"{len(failing) or tests.get('failed')} failing test(s)"
                             + (f": {', '.join(failing[:3])}" if failing else ""), Confidence.OBSERVED))
            d.actions.append("look at the failing tests (I can show the output)")
        for step in task.plan:
            if step.status.value == "failed" or (step.error and step.status.value != "done"):
                d.observed.append(f"step '{step.description}' failed: {(step.error or '')[:200]}")
                text = f"{step.error or ''} {(step.result or {}).get('data', '')}".lower()
                for needle, cause, action in _ERROR_HINTS:
                    if needle in text:
                        d.causes.append((cause, Confidence.INFERRED))
                        d.actions.append(action)
                        break
            if step.verification and step.verification.get("performed") and step.verification.get("passed") is False:
                d.observed.append(f"verification of '{step.description}' failed: {step.verification.get('detail')}")
        if task.status == S.WAITING:
            d.causes.append(("it is waiting for your approval", Confidence.OBSERVED))
            d.actions.append("say 'proceed' to approve or 'no' to skip that step")
        if task.status == S.BLOCKED:
            d.causes.append((task.status_reason, Confidence.OBSERVED))
        explanation = svc.resources.explain(task.id)
        if explanation:
            d.causes.append((explanation.condition, Confidence.OBSERVED))
    r = svc.state.values("resources.")
    mem = r.get("resources.memory_percent")
    cpu = r.get("resources.cpu_percent")
    disk = r.get("resources.disk_percent")
    stale = svc.state.is_stale("resources.cpu_percent")
    if not stale:
        if isinstance(mem, (int, float)) and mem >= 90:
            d.observed.append(f"memory usage is {mem:.0f}%")
            d.causes.append(("memory pressure", Confidence.INFERRED))
            d.actions.append("close memory-heavy applications or unload unused models")
        if isinstance(cpu, (int, float)) and cpu >= 90:
            top = _top_process()
            d.observed.append(f"CPU usage is {cpu:.0f}%" + (f"; top process {top}" if top else ""))
            if top:
                d.causes.append((f"{top} is consuming most of the CPU", Confidence.INFERRED))
        if isinstance(disk, (int, float)) and disk >= 90:
            d.observed.append(f"disk usage is {disk:.0f}%")
            d.causes.append(("the disk is nearly full", Confidence.OBSERVED))
            d.actions.append("free disk space")
    for comp in svc.health.unhealthy():
        d.observed.append(f"{comp.name} is {comp.status.label}" + (f": {comp.detail}" if comp.detail else ""))
        if comp.name.startswith("model:"):
            d.causes.append((f"the {comp.name[6:]} model runtime is not reachable", Confidence.OBSERVED))
            d.actions.append("start the model runtime (for Ollama: `ollama serve`)")
    if svc.state.value("network.state") == "offline":
        d.observed.append("the network is offline")
    if task is None:
        failures = svc.tasks.list_tasks([S.FAILED], since=svc.clock.now() - 6 * 3600, order="recent", limit=3)
        for t in failures:
            d.observed.append(f"{_label(t)} failed: {t.outputs.get('summary') or t.status_reason}")
        recent = svc.events.query(since=svc.clock.now() - 3600, min_severity=Severity.ERROR, limit=3)
        for e in recent:
            text = _event_text(e)
            if not any(text in o for o in d.observed):
                d.observed.append(text)
    return d


def _top_process(window_s: float = 0.2) -> str | None:
    """Name of the process using the most CPU, measured over a short window (psutil's first reading per
    process is always 0, so a single pass would name an arbitrary process)."""
    try:
        procs = list(psutil.process_iter(["name"]))
        for p in procs:
            try:
                p.cpu_percent(None)
            except psutil.Error:
                pass
        time.sleep(window_s)
        usage = []
        for p in procs:
            try:
                usage.append((p.cpu_percent(None), p.info.get("name")))
            except psutil.Error:
                continue
        usage.sort(reverse=True)
        return usage[0][1] if usage and usage[0][0] > 5.0 else None
    except Exception:
        return None


# -- misc ----------------------------------------------------------------------------------------------

def time_answer(svc: Services) -> str:
    now = svc.clock.now()
    local = time.localtime(now)
    return f"It's {time.strftime('%H:%M', local)} ({time.strftime('%Z', local)}), {time.strftime('%A %d %B %Y', local)}."


def self_description(svc: Services) -> str:
    tools = ", ".join(sorted(t.spec.name for t in svc.registry.list())[:12])
    model = model_line(svc)
    return ("I'm JARVIS: a local-first AI operating environment on this machine. I combine language models "
            f"(currently {model}), memory, live system state, tools, background tasks, monitoring, automation and "
            "permissions you control. I can run and monitor tasks, inspect files, processes and resources, remember "
            "things, explain what I did and why, and ask before anything consequential. "
            f"My tools right now: {tools}. Not implemented yet: voice, vision/screen awareness, communications "
            "(email, calls, messaging) and physical device control — the interfaces exist, the integrations don't.")


def help_text() -> str:
    return ("Things I understand without a language model: what are you doing / status / are we good / where were we / "
            "what changed / what's wrong / why did you do that / what did you do / where did you get that; "
            "stop, pause, continue, run that again, proceed, no; run the tests, build, $ <command>, "
            "open the <name> project; keep an eye on it, tell me when it's done, watch folder <path>; "
            "remember that ..., forget that, what do you remember about ..., why did we choose ...; "
            "focus mode, quiet, private mode, emergency mode, normal mode; use the local model, which models. "
            "Anything else goes to the language model with access to my tools.")


def models_report(svc: Services) -> str:
    inv = svc.router.inventory
    if not inv:
        down = [p for p, ok in svc.router.provider_status.items() if not ok]
        return "No models are available" + (f"; {', '.join(down)} is not reachable." if down else ".")
    parts = []
    for m in inv[:10]:
        caps = ",".join(sorted(c.value for c in m.capabilities))
        parts.append(f"{m.name} ({'loaded, ' if m.loaded else ''}{'local' if m.local else 'cloud'}; {caps})")
    pinned = svc.router.pins.get("*")
    text = f"Installed: {'; '.join(parts)}. For conversation I'd use {model_line(svc)}."
    if pinned:
        text += f" You pinned {pinned}."
    return text


def format_ts(ts: float) -> str:
    return format_datetime(ts)
