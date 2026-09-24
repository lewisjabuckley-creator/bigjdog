"""Playbooks: deterministic plan recipes for goals JARVIS understands well (Phase 3 §3, §29, §35, §41-44).

A playbook builds the skeleton of a plan without a language model: gather evidence, analyze it, decide, act
behind an approval gate, verify independently, report. What to change is decided at run time from the evidence
(the DECIDE node expands into concrete action nodes), so the plan adapts to what it finds instead of following
a fixed script.

Compound requests ("run the tests, and if they pass build it, then tell me") are mapped clause by clause onto
the existing intent grammar and templates, so the planner reuses what JARVIS already knows how to do instead of
duplicating it.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any

from jarvis.core.intent import IntentKind, parse
from jarvis.intelligence.analysis import keywords
from jarvis.intelligence.goals import Complexity, Goal, GoalParser
from jarvis.intelligence.plans import Assumption, NodeKind, PlanNode
from jarvis.planner.templates import build_plan, probe_project, run_tests_plan


@dataclass
class PlanningContext:
    cwd: str
    home: str = field(default_factory=lambda: os.path.expanduser("~"))
    platform: str = sys.platform
    project_name: str | None = None
    project_root: str | None = None
    model_available: bool = False
    tools: set[str] = field(default_factory=set)
    history: list[dict[str, Any]] = field(default_factory=list)     # earlier outcomes of the same actions
    lessons: list[str] = field(default_factory=list)                # "we already tried that" notes
    pressure: str = ""                                              # why resources are constrained, if they are
    interactive: bool = True
    jarvis_pids: list[int] = field(default_factory=list)
    many_sources: int = 8                                           # research: more files than this → agents
    sample_s: float = 1.5                                           # per-process CPU measurement window


@dataclass
class Blueprint:
    nodes: list[PlanNode]
    title: str
    assumptions: list[Assumption] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    milestones: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def step(tool: str, args: dict[str, Any] | None = None, description: str = "", name: str = "",
         allow_failure: bool = False) -> dict[str, Any]:
    return {"tool": tool, "args": dict(args or {}), "description": description or tool, "name": name or tool,
            "allow_failure": allow_failure}


def _constraints(goal: Goal) -> list[dict[str, Any]]:
    return [c.to_dict() for c in goal.constraints]


def _process_fallback(platform: str) -> list[dict[str, Any]]:
    if platform == "win32":
        return [step("shell_execute", {"command": "tasklist /fo csv /nh", "timeout_s": 30},
                     "list processes with tasklist", "by_cpu"),
                step("shell_execute", {"command": "tasklist /fo csv /nh", "timeout_s": 30},
                     "list processes with tasklist", "by_memory")]
    return [step("shell_execute", {"command": "ps -eo pid,pcpu,pmem,comm --sort=-pcpu | head -15", "timeout_s": 30},
                 "list processes with ps", "by_cpu"),
            step("shell_execute", {"command": "ps -eo pid,pcpu,pmem,comm --sort=-pmem | head -15", "timeout_s": 30},
                 "list processes with ps", "by_memory")]


# -- performance ---------------------------------------------------------------------------------------------

def performance(goal: Goal, ctx: PlanningContext, prefix: str = "") -> Blueprint:
    p = prefix
    measure = PlanNode(f"{p}measure", "Measure CPU, memory and the busiest processes", NodeKind.GATHER, steps=[
        step("system_info", {}, "read CPU, memory and disk use", "system"),
        step("process_list", {"sort": "cpu", "limit": 12, "sample_s": ctx.sample_s}, "sample the busiest processes",
             "by_cpu"),
        step("process_list", {"sort": "memory", "limit": 12}, "list the largest processes", "by_memory"),
    ], alternatives=[[step("system_info", {}, "read CPU, memory and disk use", "system"),
                      *_process_fallback(ctx.platform)]],
        estimate={"seconds": 4, "cpu": "low"})
    models = PlanNode(f"{p}models", "Check which language models are loaded", NodeKind.GATHER,
                      steps=[step("model_status", {}, "list loaded models", "models")], optional=True,
                      estimate={"seconds": 1})
    analyze = PlanNode(f"{p}analyze", "Work out the likely cause", NodeKind.ANALYZE, depends_on=[measure.id, models.id],
                       steps=[step("plan_analyze", {"analyzer": "performance", "evidence": {
                           "system": {"$fact": f"{measure.id}.system"}, "by_cpu": {"$fact": f"{measure.id}.by_cpu"},
                           "by_memory": {"$fact": f"{measure.id}.by_memory"}, "models": {"$fact": f"{models.id}.models"}},
                           "context": {"constraints": _constraints(goal), "history": ctx.history,
                                       "jarvis_pids": ctx.jarvis_pids}}, "analyze the measurements", "analysis")],
                       estimate={"seconds": 1})
    decide = PlanNode(f"{p}decide", "Decide what to change", NodeKind.DECIDE, depends_on=[analyze.id],
                      meta={"from": analyze.id, "max_actions": 1, "min_confidence": 0.55, "verify": f"{p}verify",
                            "fix": goal.wants_fix})
    verify = PlanNode(f"{p}verify", "Measure again, independently", NodeKind.VERIFY, actor="verifier",
                      depends_on=[decide.id], condition={"fact": f"{decide.id}.actions", "op": "nonempty"}, steps=[
                          step("system_info", {}, "read CPU and memory again", "after"),
                          step("process_list", {"sort": "cpu", "limit": 20, "sample_s": ctx.sample_s},
                               "sample the processes again", "after_processes"),
                          step("model_status", {}, "list loaded models again", "after_models"),
                          step("plan_verify", {"check": "performance", "before": {"$fact": f"{measure.id}.system"},
                                               "after": {"$from_step": 0}, "after_processes": {"$from_step": 1},
                                               "after_models": {"$from_step": 2},
                                               "targets": {"$fact": f"{decide.id}.targets"},
                                               "claimed": {"$fact": "actions"}},
                               "compare before and after", "verdict")],
                      estimate={"seconds": 4})
    report = PlanNode(f"{p}report", "Report what I found" + (" and did" if goal.wants_fix else ""), NodeKind.REPORT,
                      depends_on=[verify.id], run_on_failure=True)
    title = "Find out why the computer is slow" + (" and fix it" if goal.wants_fix else "")
    notes = []
    if ctx.lessons:
        notes += ctx.lessons
    return Blueprint([measure, models, analyze, decide, verify, report], title, notes=notes)


# -- disk cleanup ----------------------------------------------------------------------------------------------

def _downloads(home: str) -> str:
    return os.path.join(home, "Downloads")


def resolve_folder(name: str | None, ctx: PlanningContext) -> str:
    if not name:
        return ctx.home
    lowered = name.lower().strip()
    if lowered in ("downloads", "my downloads", "downloads folder"):
        return _downloads(ctx.home)
    if lowered in ("temp", "tmp", "temp files", "junk"):
        return tempfile.gettempdir()
    if lowered == "desktop":
        return os.path.join(ctx.home, "Desktop")
    path = os.path.expanduser(name)
    return path if os.path.isabs(path) else os.path.join(ctx.cwd, path)


def disk_cleanup(goal: Goal, ctx: PlanningContext, prefix: str = "") -> Blueprint:
    p = prefix
    target = resolve_folder(goal.target, ctx)
    steps = [step("disk_usage", {"path": target, "top": 10}, f"measure what uses space in {target}", "usage")]
    extra: list[dict[str, Any]] = []
    for folder, days, label in ((_downloads(ctx.home), 30, "downloads"), (tempfile.gettempdir(), 7, "temp")):
        if os.path.isdir(folder):
            steps.append(step("disk_usage", {"path": folder, "top": 5, "largest_files": 8, "older_than_days": days},
                              f"find large, old files in {folder}", label))
            extra.append({"$fact": f"{p}measure.{label}"})
    measure = PlanNode(f"{p}measure", "Measure disk use", NodeKind.GATHER, steps=steps, estimate={"seconds": 10})
    analyze = PlanNode(f"{p}analyze", "Work out what could go", NodeKind.ANALYZE, depends_on=[measure.id], steps=[
        step("plan_analyze", {"analyzer": "disk", "evidence": {"usage": {"$fact": f"{measure.id}.usage"}, "extra": extra},
                              "context": {"constraints": _constraints(goal), "history": ctx.history}},
             "analyze disk use", "analysis")])
    decide = PlanNode(f"{p}decide", "Decide what to remove", NodeKind.DECIDE, depends_on=[analyze.id],
                      meta={"from": analyze.id, "max_actions": 5, "min_confidence": 0.5, "verify": f"{p}verify",
                            "fix": goal.wants_fix})
    verify = PlanNode(f"{p}verify", "Check the files are gone and measure free space", NodeKind.VERIFY,
                      actor="verifier", depends_on=[decide.id],
                      condition={"fact": f"{decide.id}.actions", "op": "nonempty"}, steps=[
                          step("disk_usage", {"path": target, "top": 3}, "measure free space again", "after"),
                          step("plan_verify", {"check": "disk", "before": {"$fact": f"{measure.id}.usage"},
                                               "after": {"$from_step": 0}, "paths": {"$fact": f"{decide.id}.paths"}},
                               "compare before and after", "verdict")])
    report = PlanNode(f"{p}report", "Report", NodeKind.REPORT, depends_on=[verify.id], run_on_failure=True)
    title = "Free up disk space" if goal.wants_fix else "Find out what is using disk space"
    notes = [] if goal.target else ["I'm looking at your home folder; only Downloads and temp files are candidates "
                                    "for removal, and removing anything needs your approval."]
    return Blueprint([measure, analyze, decide, verify, report], title, notes=notes + ctx.lessons)


# -- backup ------------------------------------------------------------------------------------------------------

def backup(goal: Goal, ctx: PlanningContext, prefix: str = "") -> Blueprint:
    p = prefix
    if not goal.target or "->" not in goal.target:
        return Blueprint([], "Back up", problems=["I need to know what to back up and where to."])
    src_text, dst_text = [s.strip() for s in goal.target.split("->", 1)]
    src = resolve_folder(src_text, ctx)
    dst_root = resolve_folder(dst_text, ctx)
    dest = os.path.join(dst_root, os.path.basename(src.rstrip("/\\")) or "backup")
    check = PlanNode(f"{p}check", f"Look at {src}", NodeKind.GATHER, steps=[
        step("file_list", {"path": src, "recursive": True, "limit": 5000}, f"list {src}", "source")],
        estimate={"seconds": 5})
    copy = PlanNode(f"{p}copy", f"Copy {os.path.basename(src)} to {dst_root}", NodeKind.ACTION, depends_on=[check.id],
                    steps=[step("file_copy", {"source": src, "destination": dest}, f"copy {src} to {dest}", "copy")],
                    rollback="the copies can be deleted; the originals are never changed",
                    meta={"target": dest}, estimate={"seconds": 60})
    verify = PlanNode(f"{p}verify", "Check every file arrived", NodeKind.VERIFY, actor="verifier",
                      depends_on=[copy.id], steps=[
                          step("plan_verify", {"check": "backup", "source": src, "destination": dest},
                               "compare the copy with the original", "verdict")])
    report = PlanNode(f"{p}report", "Report", NodeKind.REPORT, depends_on=[verify.id], run_on_failure=True)
    assumptions = [
        Assumption(f"{src} exists", {"type": "path_exists", "path": src}, [copy.id]),
        Assumption(f"the destination {dst_root} is available",
                   {"type": "path_exists", "path": _existing_anchor(dst_root)}, [copy.id]),
    ]
    return Blueprint([check, copy, verify, report], f"Back up {os.path.basename(src) or src} to {dst_root}",
                     assumptions=assumptions)


def _existing_anchor(path: str) -> str:
    """The destination itself if it exists, else its drive or mount (a missing sub-folder is created)."""
    current = path
    while current and not os.path.exists(current):
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    drive = os.path.splitdrive(path)[0]
    return path if os.path.exists(path) else (drive + os.sep if drive else current or path)


# -- research ------------------------------------------------------------------------------------------------

def research(goal: Goal, ctx: PlanningContext, prefix: str = "") -> Blueprint:
    p = prefix
    m = re.search(r"\b(?:in|from|under|inside)\s+(?P<path>[~/.\\][^\s,;]+|[A-Za-z]:\\[^\s,;]+)", goal.text)
    root = os.path.expanduser(m.group("path")) if m else (ctx.project_root or ctx.cwd)
    topic = goal.target or goal.objective
    keys = keywords(topic) or keywords(goal.objective)
    if not keys:
        return Blueprint([], "Research", problems=["I couldn't tell what to research. What topic?"])
    search = PlanNode(f"{p}search", f"Search {root} for {', '.join(keys[:3])}", NodeKind.GATHER, steps=[
        step("file_search", {"path": root, "query": k, "max_results": 60}, f"search for '{k}'", f"kw{i}")
        for i, k in enumerate(keys[:4])] + [step("search_memory", {"query": topic, "limit": 5},
                                                 "check what I remember", "memory", allow_failure=True)],
        estimate={"seconds": 10})
    select = PlanNode(f"{p}select", "Choose the sources to read", NodeKind.DECIDE, depends_on=[search.id],
                      meta={"select": "sources", "from": search.id, "root": root, "max_sources": 8,
                            "agents": ctx.model_available and "delegate_to_agent" in ctx.tools,
                            "many": ctx.many_sources, "extract": f"{p}extract", "topic": topic})
    extract = PlanNode(f"{p}extract", "Extract and cross-check what the sources say", NodeKind.ANALYZE,
                       depends_on=[select.id], steps=[
                           step("plan_analyze", {"analyzer": "research", "evidence": {
                               "documents": {"$collect": "documents"}, "agent_claims": {"$collect": "agent_claims"}},
                               "context": {"topic": topic, "keywords": keys}}, "extract claims", "analysis")])
    nodes = [search, select, extract]
    verify_deps = [extract.id]
    if ctx.model_available and "model_report" in ctx.tools:
        summary = PlanNode(f"{p}summary", "Write a summary from the verified statements", NodeKind.TASK,
                           depends_on=[extract.id], optional=True, steps=[
                               step("model_report", {"instruction": (
                                   f"Summarise what these statements say about {topic}. Use only the statements; "
                                   "refer to them by number like [1]. Say where sources disagree. If they don't "
                                   "answer something, say so."), "material": {"$fact": f"{extract.id}.analysis.claims"}},
                                   "write the summary", "written")], estimate={"seconds": 60, "model_calls": 1})
        nodes.append(summary)
        verify_deps.append(summary.id)
    verify = PlanNode(f"{p}verify", "Check every statement against its source", NodeKind.VERIFY, actor="verifier",
                      depends_on=verify_deps, steps=[step("plan_verify", {
                          "check": "citations", "claims": {"$fact": f"{extract.id}.analysis.claims"},
                          "conflicts": {"$fact": f"{extract.id}.analysis.conflicts"}, "root": root},
                          "re-read each cited source", "verdict")])
    report = PlanNode(f"{p}report", "Report", NodeKind.REPORT, depends_on=[verify.id], run_on_failure=True)
    nodes += [verify, report]
    assumptions = [Assumption(f"{root} exists", {"type": "path_exists", "path": root}, [search.id])]
    return Blueprint(nodes, f"Research {topic}", assumptions=assumptions, notes=ctx.lessons)


PLAYBOOKS = {"performance": performance, "disk_cleanup": disk_cleanup, "backup": backup, "research": research}


# -- compound requests --------------------------------------------------------------------------------------------

_TELL = re.compile(r"^(?:and\s+)?(?:then\s+)?(?:tell|let|notify|ping|message)\s+me(?:\s+know)?"
                   r"(?:\s+(?:when|how|what|if|whether)\b.*)?$|^report(\s+back)?$|"
                   r"^(?:and\s+)?(?:then\s+)?(?:give\s+me\s+a\s+)?summary$", re.I)
_SYSTEM_QUERY = re.compile(r"\b(check|show|what'?s|how\s+much)\b.*\b(disk\s+space|free\s+space|storage|memory|ram|"
                           r"cpu|temperature|resources)\b", re.I)


def actionable(text: str) -> bool:
    """Whether a part of a request is something JARVIS can do itself (as opposed to a question for the
    conversation): tests, a build, a command, a project scan, a measurement, or a goal with a playbook."""
    intent = parse(text)
    if intent.kind in (IntentKind.RUN_TESTS, IntentKind.BUILD, IntentKind.SHELL, IntentKind.ANALYZE_PROJECT,
                       IntentKind.SYSTEM_QUERY):
        return True
    return GoalParser.kind(text) in PLAYBOOKS or bool(_SYSTEM_QUERY.search(text))


def compound(goal: Goal, ctx: PlanningContext, parser: GoalParser) -> Blueprint:
    nodes: list[PlanNode] = []
    exits: list[str] = []            # the node that represents each part's completion
    problems: list[str] = []
    notes: list[str] = []
    milestones: list[dict[str, Any]] = []
    root = ctx.project_root or ctx.cwd
    wants_report = False
    for index, part in enumerate(goal.subgoals):
        text = part.text.strip()
        p = f"s{index + 1}_"
        deps = [exits[i] for i in part.after if i < len(exits) and exits[i]]
        condition = None
        if part.condition and deps:
            condition = {"all": [{"node": d, "outcome": part.condition} for d in deps]}
        part_nodes: list[PlanNode] = []
        exit_id = ""
        if _TELL.search(text) and not _SYSTEM_QUERY.search(text):
            wants_report = True
            exits.append(exits[-1] if exits else "")
            continue
        intent = parse(text)
        sub_kind = GoalParser.kind(text)
        try:
            if intent.kind == IntentKind.RUN_TESTS:
                steps, success = run_tests_plan(root)
                part_nodes = [PlanNode(f"{p}tests", "Run the tests", NodeKind.ACTION, verify=success,
                                       steps=[_from_step(s, "tests") for s in steps], estimate={"seconds": 120})]
            elif intent.kind == IntentKind.BUILD:
                steps, success = build_plan(root)
                part_nodes = [PlanNode(f"{p}build", "Build the project", NodeKind.ACTION, verify=success,
                                       steps=[_from_step(s, "build") for s in steps], estimate={"seconds": 120})]
            elif intent.kind == IntentKind.SHELL and intent.params.get("command"):
                command = intent.params["command"]
                part_nodes = [PlanNode(f"{p}run", f"Run `{command[:50]}`", NodeKind.ACTION, steps=[
                    step("shell_execute", {"command": command, "cwd": root, "timeout_s": 600}, f"run `{command}`",
                         "run")])]
            elif intent.kind == IntentKind.ANALYZE_PROJECT:
                name = ctx.project_name or os.path.basename(root.rstrip(os.sep)) or root
                part_nodes = [PlanNode(f"{p}scan", f"Scan {name}", NodeKind.GATHER,
                                       steps=[step("project_scan", {"path": root}, f"scan {name}", "scan")])]
                if ctx.model_available:
                    part_nodes.append(PlanNode(f"{p}analysis", f"Write the analysis of {name}", NodeKind.TASK,
                                               depends_on=[part_nodes[0].id], steps=[step("model_report", {
                                                   "instruction": f"Analyze the software project '{name}': what it "
                                                                  "is, how it is built and tested, notable risks.",
                                                   "material": {"$fact": f"{p}scan.scan"}}, "write the analysis",
                                                   "report")]))
            elif sub_kind in PLAYBOOKS:
                sub_goal = parser.parse(text)
                sub_goal.constraints = goal.constraints
                blueprint = PLAYBOOKS[sub_kind](sub_goal, ctx, prefix=p)
                problems += blueprint.problems
                # the part's own report is folded into the plan's final report
                part_nodes = [n for n in blueprint.nodes if n.kind != NodeKind.REPORT]
                if part_nodes:
                    exit_id = part_nodes[-1].id
            elif _SYSTEM_QUERY.search(text):
                if re.search(r"disk|storage|space", text, re.I):
                    part_nodes = [PlanNode(f"{p}disk", "Measure disk space", NodeKind.GATHER, steps=[
                        step("disk_usage", {"path": ctx.home, "top": 5}, "measure disk space", "usage")])]
                else:
                    part_nodes = [PlanNode(f"{p}system", "Read the system's resources", NodeKind.GATHER,
                                           steps=[step("system_info", {}, "read CPU, memory and disk use", "system")])]
            elif ctx.model_available:
                part_nodes = [PlanNode(f"{p}work", text[:1].upper() + text[1:80], NodeKind.TASK, objective=text,
                                       estimate={"seconds": 120, "model_calls": 3})]
            else:
                problems.append(f"I can't work out how to '{text}' without a language model, and none is available.")
                exits.append("")
                continue
        except LookupError as exc:
            problems.append(f"{exc}.")
            exits.append("")
            continue
        entry = [n for n in part_nodes if not any(d in {m.id for m in part_nodes} for d in n.depends_on)]
        for n in entry:
            n.depends_on = list(dict.fromkeys(deps + n.depends_on))
            if condition:
                n.condition = condition
        nodes += part_nodes
        exit_id = exit_id or part_nodes[-1].id
        exits.append(exit_id)
        milestones.append({"id": f"m{index + 1}", "title": text[:1].upper() + text[1:], "nodes": [n.id for n in part_nodes]})
    finals = [e for e in exits if e]
    report = PlanNode("report", "Report", NodeKind.REPORT, depends_on=list(dict.fromkeys(finals[-1:] or finals)),
                      run_on_failure=True, meta={"notify": wants_report})
    # the report waits for every part, not only the last one
    report.depends_on = list(dict.fromkeys(finals))
    nodes.append(report)
    title = "; then ".join(s.text for s in goal.subgoals if not _TELL.search(s.text))[:80]
    if goal.complexity >= Complexity.VERY_COMPLEX and goal.deadline:
        notes.append("I'll track each part as a milestone against the deadline.")
    return Blueprint(nodes, title[:1].upper() + title[1:], problems=problems, notes=notes, milestones=milestones)


def _from_step(s: Any, name: str) -> dict[str, Any]:
    return {"tool": s.tool, "args": dict(s.args), "description": s.description, "name": name,
            "allow_failure": s.allow_failure}


def probe(root: str) -> dict[str, Any]:
    return probe_project(root).to_dict()
