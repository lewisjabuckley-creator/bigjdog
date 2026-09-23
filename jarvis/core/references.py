"""Conversational reference resolution (spec §13-14, §118, §120).

"Stop that", "open the other one", "keep an eye on it": pronouns and partial
names are resolved against the conversation focus and the live task/project
state. When more than one candidate is plausible and the choice matters, the
resolver returns the candidates so the orchestrator can ask one precise
question instead of guessing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from jarvis.core.intent import is_pronoun
from jarvis.core.types import Provenance
from jarvis.projects.manager import Project, ProjectManager
from jarvis.tasks.manager import TaskManager
from jarvis.tasks.models import EXECUTING, OPEN, Task, TaskKind, TaskStatus

_STOP_WORDS = ("the ", "my ", "our ", "that ", "this ")


@dataclass
class ConversationFocus:
    tasks: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    projects: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    candidates: list[str] = field(default_factory=list)   # last disambiguation list (ids)
    candidate_kind: str = ""
    last_user_text: str = ""
    last_reply: str = ""
    last_provenance: list[Provenance] = field(default_factory=list)
    last_command: str | None = None

    def touch_task(self, task_id: str) -> None:
        if task_id in self.tasks:
            self.tasks.remove(task_id)
        self.tasks.appendleft(task_id)

    def touch_project(self, project_id: str) -> None:
        if project_id in self.projects:
            self.projects.remove(project_id)
        self.projects.appendleft(project_id)


@dataclass
class Resolution:
    item: Task | Project | None
    candidates: list[Task] | list[Project] = field(default_factory=list)
    reason: str = ""

    @property
    def ambiguous(self) -> bool:
        return self.item is None and len(self.candidates) > 1


def _clean(target: str) -> str:
    t = target.lower().strip()
    for w in _STOP_WORDS:
        if t.startswith(w):
            t = t[len(w):]
    return t.removesuffix(" task").removesuffix(" job").strip()


class ReferenceResolver:
    def __init__(self, tasks: TaskManager, projects: ProjectManager, focus: ConversationFocus) -> None:
        self.tasks = tasks
        self.projects = projects
        self.focus = focus

    # -- tasks ---------------------------------------------------------------------------
    def task(self, target: str | None, *, statuses: Iterable[TaskStatus] | None = None,
             prefer: Iterable[TaskStatus] | None = None, include_monitors: bool = True) -> Resolution:
        allowed = set(statuses) if statuses is not None else None
        preferred = set(prefer or [])

        def ok(t: Task | None) -> bool:
            return t is not None and (allowed is None or t.status in allowed) and \
                (include_monitors or t.kind != TaskKind.MONITOR)

        if is_pronoun(target):
            # focus order first (what we've been talking about), then live state
            focused = [t for t in (self.tasks.get_task(tid) for tid in self.focus.tasks) if ok(t)]
            if preferred:
                hits = [t for t in focused if t.status in preferred]
                if hits:
                    return Resolution(hits[0], reason="most recently discussed")
            if focused:
                return Resolution(focused[0], reason="most recently discussed")
            live = [t for t in self.tasks.list_tasks(allowed or OPEN, order="recent", limit=20) if ok(t)]
            if preferred:
                live.sort(key=lambda t: t.status not in preferred)
            executing = [t for t in live if t.status in EXECUTING and t.kind != TaskKind.MONITOR]
            pool = executing or live
            if len(pool) == 1:
                return Resolution(pool[0], reason="the only matching task")
            if pool:
                return Resolution(pool[0], pool[:4], reason="most recent")
            return Resolution(None, reason="no matching task")
        name = _clean(target)
        if name in ("other one", "other"):
            others = [t for t in (self.tasks.get_task(tid) for tid in self.focus.candidates) if ok(t)]
            others = [t for t in others if not self.focus.tasks or t.id != self.focus.tasks[0]]
            if others:
                return Resolution(others[0], reason="the other candidate")
            recent = [t for t in (self.tasks.get_task(tid) for tid in list(self.focus.tasks)[1:]) if ok(t)]
            return Resolution(recent[0] if recent else None, reason="the previously discussed task")
        if name in ("everything", "all", "all tasks"):
            return Resolution(None, [t for t in self.tasks.list_tasks(allowed or OPEN) if ok(t)], reason="all")
        matches = [t for t in self.tasks.find(name, allowed) if ok(t)]
        if not matches and name.endswith("s"):
            matches = [t for t in self.tasks.find(name[:-1], allowed) if ok(t)]
        if not matches:
            return Resolution(None, reason=f"no task matches '{target}'")
        open_matches = [t for t in matches if t.status in OPEN]
        pool = open_matches or matches
        if preferred:
            pool.sort(key=lambda t: t.status not in preferred)
        if len(pool) == 1 or len({t.title for t in pool}) == 1:
            return Resolution(pool[0], reason="name match")
        executing = [t for t in pool if t.status in EXECUTING]
        if len(executing) == 1:
            return Resolution(executing[0], reason="the running one")
        return Resolution(None, pool[:4], reason="several tasks match")

    # -- projects ------------------------------------------------------------------------------
    def project(self, target: str | None) -> Resolution:
        if target is None or is_pronoun(target) or _clean(target) in ("project", "current project"):
            active = self.projects.active()
            return Resolution(active, reason="the active project" if active else "no active project")
        name = _clean(target).removesuffix(" project").strip()
        if name in ("other one", "other"):
            active = self.projects.active()
            candidates = [p for p in (self.projects.get(pid) for pid in self.focus.candidates
                                      if self.focus.candidate_kind == "project") if p]
            candidates += [p for p in (self.projects.get(pid) for pid in self.focus.projects) if p]
            candidates += self.projects.list()
            others = []
            for p in candidates:
                if (not active or p.id != active.id) and p.id not in {o.id for o in others}:
                    others.append(p)
            if len(others) == 1 or (others and self.focus.candidate_kind == "project"):
                return Resolution(others[0], reason="the other project")
            return Resolution(None, others[:4], reason="several other projects")
        matches = self.projects.find(name)
        if len(matches) == 1:
            return Resolution(matches[0], reason="name match")
        if not matches:
            return Resolution(None, reason=f"no project matches '{target}'")
        return Resolution(None, matches[:4], reason="several projects match")
