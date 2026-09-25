"""The conversation loop (spec §182).

USER SPEAKS → INTERPRET INTENT → RETRIEVE CONTEXT → CHECK CURRENT STATE →
CHECK AUTHORITY → DECIDE (answer | ask | plan | execute | delegate | monitor)
→ VERIFY IF ACTION OCCURRED → RESPOND.

The orchestrator coordinates subsystems; it does not do their work. Control
and status intents are handled deterministically from live state, so they keep
working when no model is available. Work is turned into durable tasks, which
the worker pool executes and the notification policy reports on.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from jarvis.core import personality, reports
from jarvis.core.context import ContextAssembler, summarize_state_for_tool
from jarvis.core.intent import Intent, IntentKind, is_everything, is_pronoun, parse
from jarvis.core.modes import Mode
from jarvis.core.plan_dialogue import PlanDialogue
from jarvis.core.references import ConversationFocus, ReferenceResolver
from jarvis.core.services import Services
from jarvis.core.types import OperationalReason, Priority, Provenance, ProvenanceKind, new_id
from jarvis.events.types import Event, EventType
from jarvis.log import get_logger
from jarvis.memory.store import MemoryKind
from jarvis.models.base import Capability, ChatMessage, ModelError, NoModelAvailable, Purpose, ToolCall
from jarvis.models.ollama import strip_thinking
from jarvis.models.router import TaskProfile
from jarvis.notifications.manager import Notification
from jarvis.permissions.hierarchy import InstructionSource
from jarvis.permissions.model import Actor
from jarvis.planner.templates import build_plan, probe_project, run_tests_plan
from jarvis.projects.manager import Project
from jarvis.tasks.models import EXECUTING, OPEN, MonitorSpec, Step, StepStatus, Task, TaskKind, TaskPolicy, TaskStatus
from jarvis.tools.base import ToolContext
from jarvis.tools.internal import LiveStateTool, RememberTool, SearchMemoryTool, StartTaskTool
from jarvis.tools.registry import ExecStatus

log = get_logger("orchestrator")
S = TaskStatus

_HARD_STOP = re.compile(r"^\W*(cancel|abort|kill|forget it|never ?mind|nevermind)", re.IGNORECASE)
_MAX_TOOL_ROUNDS = 6


@dataclass
class Response:
    text: str
    intent: IntentKind
    kind: str = "answer"                   # answer | question | action | error
    task_id: str | None = None
    approval_id: str | None = None
    provenance: list[Provenance] = field(default_factory=list)
    notifications: list[Notification] = field(default_factory=list)
    model: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    streamed: bool = False                 # the answer text was already shown token by token
    footnote: str = ""                     # e.g. a model fallback note

    def to_dict(self) -> dict[str, Any]:
        """The response as interfaces receive it over the local API."""
        return {"text": self.text, "intent": self.intent.value, "kind": self.kind, "task_id": self.task_id,
                "approval_id": self.approval_id, "provenance": [p.to_dict() for p in self.provenance],
                "notifications": [{"id": n.id, "text": n.text(), "priority": n.priority.name.lower()}
                                  for n in self.notifications],
                "model": self.model, "data": self.data, "streamed": self.streamed, "footnote": self.footnote,
                "rendered": self.render(), "extra": self.render(include_text=False)}

    def render(self, *, include_text: bool = True) -> str:
        parts = []
        if include_text:
            parts.append(self.text + (f" ({self.footnote})" if self.footnote else ""))
        elif self.footnote:
            parts.append(f"({self.footnote})")
        parts += [f"• {n.text()}" for n in self.notifications]
        return "\n".join(p for p in parts if p)


Handler = Callable[[Intent], Awaitable[Response]]


class Orchestrator:
    def __init__(self, svc: Services, *, session_id: str = "default") -> None:
        self.svc = svc
        self.session_id = session_id
        self.focus = ConversationFocus()
        self.resolver = ReferenceResolver(svc.tasks, svc.projects, self.focus)
        self.context = ContextAssembler(svc)
        self.history: list[ChatMessage] = []
        self.pending_question: dict[str, Any] | None = None
        self.verbosity_override: str | None = None
        previous = svc.state.value("session.last_seen")
        self.session_since: float = float(previous) if isinstance(previous, (int, float)) else svc.clock.now() - 86400
        self.last_changes_check: float | None = None
        self._inline_tasks: list[str] = []      # tasks whose outcome is reported in the current reply
        self._token_sink: Callable[[str], None] | None = None
        self.client_cwd: str | None = None      # the interface's working folder (the runtime may run elsewhere)
        self._current_text = ""
        self._advisory = False                  # "what should I do?": recommend, never act
        self._model_asked = False               # the last reply was the model asking the user something
        self._register_internal_tools()
        self.plans = PlanDialogue(self)
        self.handlers: dict[IntentKind, Handler] = {
            IntentKind.STATUS: self._status, IntentKind.REENTRY: self._reentry, IntentKind.BRIEFING: self._briefing,
            IntentKind.WHAT_CHANGED: self._what_changed, IntentKind.DIAGNOSE: self._diagnose, IntentKind.WHY: self._why,
            IntentKind.WHAT_DID_YOU_DO: self._what_did_you_do, IntentKind.PROVENANCE: self._provenance,
            IntentKind.STOP: self._stop, IntentKind.PAUSE: self._stop, IntentKind.RESUME: self._resume,
            IntentKind.RETRY: self._retry, IntentKind.APPROVE: self._approve, IntentKind.DENY: self._deny,
            IntentKind.MODIFY_PLAN: self._modify_plan, IntentKind.PRIORITY: self._priority, IntentKind.MODE: self._mode,
            IntentKind.REMEMBER: self._remember, IntentKind.FORGET: self._forget,
            IntentKind.DONT_REMEMBER: self._dont_remember, IntentKind.RECALL: self._recall,
            IntentKind.DECISION_WHY: self._decision_why, IntentKind.RECORD_DECISION: self._record_decision,
            IntentKind.LOG_THAT: self._log_that, IntentKind.MODEL_USE: self._model_use,
            IntentKind.MODEL_UNLOAD: self._model_unload, IntentKind.MODEL_LIST: self._model_list,
            IntentKind.MONITOR: self._monitor, IntentKind.RUN_TESTS: self._run_tests, IntentKind.BUILD: self._build,
            IntentKind.SHELL: self._shell, IntentKind.OPEN_PROJECT: self._open_project,
            IntentKind.REPEAT_FOR: self._repeat_for, IntentKind.TIME: self._time, IntentKind.SELF: self._self,
            IntentKind.HELP: self._help, IntentKind.SHORTER: self._shorter, IntentKind.LONGER: self._longer,
            IntentKind.CHAT: self._chat, IntentKind.AWAY: self._away, IntentKind.ANALYZE_PROJECT: self._analyze_project,
            IntentKind.CLI_COMMAND: self._cli_command, IntentKind.SYSTEM_QUERY: self._system_query,
            IntentKind.ADVISE: self.plans.advise, IntentKind.SIMULATE: self.plans.simulate,
            IntentKind.PREDICT: self.plans.predict, IntentKind.PLAN_SHOW: self.plans.show,
            IntentKind.PLAN_HISTORY: self.plans.history, IntentKind.AUTONOMY: self.plans.autonomy,
            IntentKind.PLAN_DETAILS: self.plans.details,
        }

    # -- entry point ------------------------------------------------------------------------
    async def handle(self, text: str, *, on_token: Callable[[str], None] | None = None,
                     cwd: str | None = None) -> Response:
        """Handle one user turn. ``on_token`` receives model answer text as it streams (interfaces may show it
        live); deterministic answers are returned whole. ``cwd`` is the interface's working folder, which is
        what "this project" and relative paths mean to the user."""
        if cwd and os.path.isdir(cwd):
            self.client_cwd = cwd
        self._current_text = text
        previous_activity = self.svc.notifications.user_activity
        self.svc.notifications.set_activity("conversing")
        self._token_sink = on_token
        try:
            return await self._handle(text)
        finally:
            self._token_sink = None
            self.svc.notifications.set_activity(previous_activity if previous_activity != "conversing" else "idle")

    async def _handle(self, text: str) -> Response:
        svc = self.svc
        self.plans.begin_turn()
        svc.notifications.on_user_input()
        svc.bus.emit(Event(EventType.USER_MESSAGE, "conversation", {"chars": len(text)}))
        self._record("user", text)
        response: Response | None = None
        if self.pending_question:
            response = await self._answer_pending(text)
        if response is None:
            response = await self._dispatch(text)
        inline = list(self._inline_tasks)
        self._inline_tasks.clear()
        if response.task_id and response.intent in (IntentKind.SHELL, IntentKind.CHAT):
            inline.append(response.task_id)
        if inline:
            await svc.bus.drain()
            from jarvis.core.awareness import mark_reported
            for task_id in inline:
                task = svc.tasks.get_task(task_id)
                if task is not None and (task.terminal or task.status == S.WAITING):
                    svc.notifications.acknowledge_task(task.id)   # already reported inline
                    if task.terminal:
                        mark_reported(svc, [task.id])
        if self.plans.reported_inline:
            await svc.bus.drain()        # the plan's own completion notice is queued by now: it was just shown
            for plan_id in self.plans.reported_inline:
                svc.notifications.acknowledge_task(f"plan:{plan_id}")
            self.plans.reported_inline.clear()
        if response.kind != "question" and self._may_deliver_queued(response.intent):
            delivered = svc.notifications.drain(limit=3)
            response.notifications = [n for n in delivered if n.title not in response.text]
        if response.data.get("plan_id"):
            self.plans.focus_plan = response.data["plan_id"]
            self.plans.focus_is_plan = True
        elif response.task_id:
            self.plans.focus_is_plan = False
        self.focus.last_user_text = text
        self.focus.last_reply = response.text
        self._model_asked = bool(response.model) and response.text.rstrip().endswith("?")
        if response.intent not in (IntentKind.PROVENANCE,):
            self.focus.last_provenance = response.provenance
        svc.state.set("session.last_seen", svc.clock.now())
        self._record("assistant", response.text, {"intent": response.intent.value, "model": response.model})
        return response

    async def _dispatch(self, text: str) -> Response:
        """Understand one request and hand it to its handler."""
        intent = parse(text)
        try:
            if intent.kind in (IntentKind.CHAT, IntentKind.MODIFY_PLAN):
                # a correction to the plan in progress ("leave Chrome alone", "no, back it up to E:\\")
                corrected = await self.plans.correct(intent)
                if corrected is not None:
                    return corrected
            handler = self.handlers.get(intent.kind, self._chat)
            return await handler(intent)
        except Exception as exc:
            log.error("handler_failed", intent=intent.kind.value, error=repr(exc))
            return Response(f"Something went wrong while handling that ({exc}). It's logged; nothing else "
                            "was affected.", intent.kind, kind="error")

    def restore_history(self, limit: int = 40) -> int:
        """Continue this session's conversation after a restart or from another interface."""
        rows = self.svc.db.query("SELECT role, content FROM conversation WHERE session_id=? "
                                 "ORDER BY ts DESC, rowid DESC LIMIT ?", (self.session_id, limit))
        self.history = [ChatMessage("user" if r["role"] == "user" else "assistant", r["content"])
                        for r in reversed(rows)]
        return len(self.history)

    def _may_deliver_queued(self, kind: IntentKind) -> bool:
        if kind in (IntentKind.STATUS, IntentKind.BRIEFING, IntentKind.REENTRY, IntentKind.WHAT_CHANGED):
            return True
        return self.svc.modes.effective().policy.deliver_queued_when_idle

    def _record(self, role: str, content: str, meta: dict[str, Any] | None = None) -> None:
        if role == "user":
            self.history.append(ChatMessage("user", content))
        else:
            self.history.append(ChatMessage("assistant", content))
        self.history = self.history[-40:]
        if self.svc.memory.suppressed:
            return   # "don't remember this" covers the conversation log too
        self.svc.db.execute("INSERT INTO conversation(id, ts, session_id, user_id, role, content, meta) "
                            "VALUES(?,?,?,?,?,?,?)", (new_id("msg"), self.svc.clock.now(), self.session_id,
                                                      self.svc.user, role, content, json.dumps(meta or {})))

    def _reply(self, text: str, intent: Intent, **kw: Any) -> Response:
        return Response(text, intent.kind, **kw)

    def _ask(self, question: str, intent: Intent, candidates: list[Any], entity: str) -> Response:
        self.pending_question = {"intent": intent, "ids": [c.id for c in candidates], "entity": entity,
                                 "labels": [getattr(c, "title", None) or getattr(c, "name", "") for c in candidates]}
        self.focus.candidates = [c.id for c in candidates]
        self.focus.candidate_kind = entity
        return Response(question, intent.kind, kind="question")

    async def _answer_pending(self, text: str) -> Response | None:
        pq = self.pending_question
        self.pending_question = None
        if pq is None:
            return None
        if pq.get("entity") == "monitor_spec":
            return await self._monitor_from_answer(text, pq)
        if pq.get("entity") == "goal_clarify":
            if parse(text).kind not in (IntentKind.CHAT, IntentKind.OPEN_PROJECT):
                return None        # a new request (or "cancel"), not an answer
            return await self.plans.answer_clarification(text, pq)
        lowered = text.lower().strip().rstrip(".!?")
        ordinals = {"1": 0, "first": 0, "the first": 0, "first one": 0, "the first one": 0, "2": 1, "second": 1,
                    "the second": 1, "second one": 1, "the second one": 1, "3": 2, "third": 2, "the third": 2,
                    "4": 3, "fourth": 3, "last": len(pq["ids"]) - 1, "the last one": len(pq["ids"]) - 1}
        index = ordinals.get(lowered)
        if index is None:
            fillers = {"the", "one", "a", "an", "that", "this", "please", "i", "mean", "its", "it's", "option",
                       "number", "yes", "go", "with"}
            tokens = [w for w in re.findall(r"[a-z0-9']+", lowered) if w not in fillers]
            matches = [i for i, label in enumerate(pq["labels"])
                       if label and tokens and all(any(lw.startswith(t) for lw in re.findall(r"[a-z0-9']+",
                                                                                              label.lower()))
                                                   for t in tokens)]
            index = matches[0] if len(matches) == 1 else None
        if index is None or index >= len(pq["ids"]):
            return None   # not an answer; treat as a new request
        intent: Intent = pq["intent"]
        intent.params["resolved_id"] = pq["ids"][index]
        return await self.handlers[intent.kind](intent)

    # -- resolution helpers --------------------------------------------------------------------
    def _resolved_task(self, intent: Intent) -> Task | None:
        rid = intent.params.get("resolved_id")
        return self.svc.tasks.get_task(rid) if rid else None

    def _resolved_project(self, intent: Intent) -> Project | None:
        rid = intent.params.get("resolved_id")
        return self.svc.projects.get(rid) if rid else None

    def _task_choice_question(self, verb: str, tasks: list[Task]) -> str:
        options = "; ".join(f"{i + 1}) {t.title} ({t.status.value})" for i, t in enumerate(tasks))
        return f"Which one should I {verb}? {options}"

    def _work_root(self) -> str:
        """Where relative paths and commands point: the active project, else the folder JARVIS was started
        from if it is inside the allowed scope, else the user's home folder (allowed by default), else the
        first allowed folder."""
        project = self.svc.projects.active()
        if project and project.root:
            return project.root
        paths = self.svc.permissions.paths
        for candidate in (self._launch_dir(), os.path.expanduser("~")):
            if paths.check(os.path.realpath(candidate))[0]:
                return candidate
        for root in paths.roots():                     # otherwise the first allowed folder that exists
            if os.path.isdir(root) and paths.check(root)[0]:
                return root
        return self._launch_dir()

    def _launch_dir(self) -> str:
        """The folder the user is working in: the interface's, else this process's."""
        return self.client_cwd if self.client_cwd and os.path.isdir(self.client_cwd) else os.getcwd()

    def _project_for(self, target: str | None) -> tuple[Project | None, list[Project]]:
        if target:
            res = self.resolver.project(target)
            if res.item is not None or res.candidates:
                return res.item, list(res.candidates)  # type: ignore[arg-type]
            expanded = os.path.expanduser(target)
            if os.path.isdir(expanded):
                return self.svc.projects.discover(expanded), []
            return None, []
        active = self.svc.projects.active()
        if active:
            return active, []
        cwd = self._launch_dir()
        probe = probe_project(cwd)
        if probe.test_command or probe.markers:
            return self.svc.projects.discover(cwd), []
        return None, []

    def _user_task(self, objective: str, *, steps: list[Step] | None = None, title: str = "",
                   success: dict[str, Any] | None = None, cwd: str | None = None, dry_run: bool = False,
                   priority: Priority = Priority.P1, project: Project | None = None, template: str | None = None,
                   dependencies: list[str] | None = None, policy: TaskPolicy | None = None) -> Task:
        project = project or self.svc.projects.active()
        task = self.svc.tasks.create_task(objective, title=title, steps=steps, success_condition=success,
                                          cwd=cwd or (project.root if project and project.root else None),
                                          dry_run=dry_run, priority=priority, created_by=f"user:{self.svc.user}",
                                          owner=self.svc.user, project_id=project.id if project else None,
                                          outputs={"template": template} if template else None,
                                          dependencies=dependencies, policy=policy,
                                          authority={"interactive": True}, request=self._current_text,
                                          origin=f"conversation:{self.session_id}")
        self.focus.touch_task(task.id)
        return task

    # -- awareness -------------------------------------------------------------------------------
    async def _status(self, intent: Intent) -> Response:
        svc = self.svc
        text = intent.text.lower()
        task = self._resolved_task(intent)
        if task is None and intent.target and not is_pronoun(intent.target):
            res = self.resolver.task(intent.target)
            if res.ambiguous:
                return self._ask(self._task_choice_question("report on", res.candidates), intent, res.candidates, "task")
            task = res.item  # type: ignore[assignment]
            if task is None:
                return await self._or_chat(intent, self._reply(
                    f"I'm not tracking anything called '{intent.target}'. {reports.activity(svc)}",
                    intent, provenance=[Provenance(ProvenanceKind.DATABASE, "tasks")]))
        elif task is None and intent.target:
            res = self.resolver.task(None)
            task = res.item  # type: ignore[assignment]
            if task is None:
                return self._reply("Nothing is running right now.", intent)
        prov = [Provenance(ProvenanceKind.DATABASE, "task manager"), Provenance(ProvenanceKind.SYSTEM_STATE, "live state")]
        if task is not None:
            self.focus.touch_task(task.id)
            return self._reply(reports.task_status(svc, task), intent, task_id=task.id, provenance=prov)
        if re.search(r"\b(good|anything|need to know|should know)\b", text):
            return self._reply(reports.are_we_good(svc), intent, provenance=prov)
        if re.search(r"\b(status|sitrep|report)\b", text) and "doing" not in text:
            return self._reply(reports.status_report(svc), intent, provenance=prov, data={"format": "block"})
        return self._reply(reports.activity(svc), intent, provenance=prov)

    async def _reentry(self, intent: Intent) -> Response:
        text = reports.reentry(self.svc)
        recovered = self.svc.extra.get("recovery_reports") or []
        pending_recovery = [r for r in recovered if not r.resumed]
        if pending_recovery and all(r.summary not in text for r in pending_recovery):
            extra = " ".join(r.summary for r in pending_recovery[:2])
            text += " " + (extra if "'continue'" in extra else extra + " Say 'continue' to resume.")
        return self._reply(text, intent, provenance=[Provenance(ProvenanceKind.DATABASE, "tasks and events")])

    async def _briefing(self, intent: Intent) -> Response:
        return self._reply(reports.briefing(self.svc), intent,
                           provenance=[Provenance(ProvenanceKind.SYSTEM_STATE, "live state"),
                                       Provenance(ProvenanceKind.DATABASE, "tasks and events")])

    async def _away(self, intent: Intent) -> Response:
        """"What happened while I was away?" — from tasks, events, notifications and state only."""
        from jarvis.core.awareness import away_report, mark_reported
        report = away_report(self.svc)
        self.last_changes_check = self.svc.clock.now()
        for task_id in report.reported_task_ids:
            self.svc.notifications.acknowledge_task(task_id)     # their results are in this answer
        mark_reported(self.svc, [t["id"] for t in report.finished])
        return self._reply(report.text(), intent, data={"away": report.to_dict(), "format": "block"},
                           provenance=[Provenance(ProvenanceKind.DATABASE, "tasks, events and notifications"),
                                       Provenance(ProvenanceKind.SYSTEM_STATE, "runtime records")])

    async def _cli_command(self, intent: Intent) -> Response:
        """`py -m jarvis runtime stop` typed into the conversation: it belongs in a terminal. Running it from here
        would run it on myself (and a stop would stop me in the middle of running it)."""
        command = intent.params.get("command", intent.text)
        text = (f"`{command}` is a command for Command Prompt (or a terminal), not something to say to me. If I ran "
                "it, I'd be running it on myself. Type /quit, then run it there.")
        if re.search(r"\bruntime\s+stop\b", command, re.IGNORECASE):
            text += " (Closing this window doesn't stop me; that command does.)"
        elif re.search(r"\bruntime\s+(status|health)\b", command, re.IGNORECASE):
            text += " Or ask me here: \"status\" or \"are we good?\"."
        return self._reply(text, intent)

    async def _analyze_project(self, intent: Intent) -> Response:
        """A durable task: measure the project, then have the model write the analysis from those facts."""
        resolved = self._resolved_project(intent)
        project, candidates = (resolved, []) if resolved else self._project_for(intent.target)
        if candidates:
            return self._ask("Which project should I analyze? " + "; ".join(
                f"{i + 1}) {p.name}" for i, p in enumerate(candidates)), intent, candidates, "project")
        if project is None and intent.target:
            return self._reply(f"I can't find a project called '{intent.target}'. Open it first (\"open the project "
                               "<folder>\") or run me from its folder.", intent)
        root = project.root if project and project.root else self._launch_dir()
        name = project.name if project else os.path.basename(root.rstrip(os.sep)) or root
        steps = [Step(f"scan {name}", "project_scan", {"path": root}),
                 Step("write the analysis", "model_report",
                      {"instruction": f"Analyze the software project '{name}' for its owner: what it is, how it is "
                                      "built, its size and structure, how it is tested, and notable risks or gaps.",
                       "material": {"$from_step": 0}})]
        task = self._user_task(f"Analyze the {name} project", steps=steps, title=f"Analyze {name}", cwd=root,
                               project=project)
        return self._reply(f"Analyzing {name} in the background (task {task.id}). It keeps running if you close "
                           "this window; I'll tell you when the analysis is ready.", intent, task_id=task.id,
                           provenance=[Provenance(ProvenanceKind.DATABASE, "task manager")])

    async def _what_changed(self, intent: Intent) -> Response:
        since = self.last_changes_check or self.session_since
        self.last_changes_check = self.svc.clock.now()
        return self._reply(reports.what_changed(self.svc, since), intent,
                           provenance=[Provenance(ProvenanceKind.DATABASE, "audit log and event store")])

    async def _diagnose(self, intent: Intent) -> Response:
        svc = self.svc
        task = self._resolved_task(intent)
        chain_text = ""
        if task is None:
            target = intent.target
            if target and not is_pronoun(target):
                res = self.resolver.task(target)
                if res.ambiguous:
                    return self._ask(self._task_choice_question("diagnose", res.candidates), intent, res.candidates, "task")
                task = res.item  # type: ignore[assignment]
                if task is None:
                    chain_text = self._dependency_chain(target)
                    if not chain_text and intent.text.lower().lstrip().startswith("what happened"):
                        return await self._or_chat(intent, self._reply(
                            f"I have no record of anything called '{target}'.", intent))
            else:
                focused = self.resolver.task(None, prefer=[S.FAILED, S.BLOCKED, S.WAITING],
                                             statuses=None).item
                recent_failed = svc.tasks.list_tasks([S.FAILED, S.BLOCKED], order="recent", limit=1,
                                                     since=svc.clock.now() - 6 * 3600)
                if focused is not None and focused.status in (S.FAILED, S.BLOCKED, S.WAITING):
                    task = focused  # type: ignore[assignment]
                elif recent_failed:
                    task = recent_failed[0]
        diagnosis = reports.diagnose(svc, task)
        text = diagnosis.format(short=self._short())
        if chain_text:
            text = chain_text + " " + text
        provs = [Provenance(ProvenanceKind.SYSTEM_STATE, "live state"), Provenance(ProvenanceKind.DATABASE, "task history")]
        model = None
        if diagnosis.observed and not diagnosis.causes and svc.router.available():
            interpretation, model = await self._interpret(diagnosis)
            if interpretation:
                text += f" My reading (inferred, not verified): {interpretation}"
                provs.append(Provenance(ProvenanceKind.INFERENCE, "language model", "interpretation of evidence"))
        if task is not None:
            self.focus.touch_task(task.id)
        return self._reply(text, intent, task_id=task.id if task else None, provenance=provs, model=model,
                           data={"observed": diagnosis.observed, "causes": [(c, k.value) for c, k in diagnosis.causes]})

    def _dependency_chain(self, target: str) -> str:
        entities = self.svc.world.resolve_name(re.sub(r"^(the|my)\s+", "", target.lower()),
                                               ["service", "application", "process", "project"])
        if not entities:
            return ""
        start = entities[0]
        chain = self.svc.world.chain(start.id)
        if not chain:
            return ""
        path = " → ".join([start.name] + [e.name + (f" ({_attrs_brief(e.attrs)})" if e.attrs else "") for e in chain])
        return f"Dependency chain: {path}."

    async def _interpret(self, diagnosis: reports.Diagnosis) -> tuple[str | None, str | None]:
        prompt = ("Observed facts:\n- " + "\n- ".join(diagnosis.observed) +
                  "\nIn one or two sentences, what is the most likely cause? Say if the evidence is insufficient.")
        try:
            routed = await self.svc.router.chat(TaskProfile(purpose=Purpose.REASONING, complexity="medium"),
                                                [ChatMessage("system", "You diagnose computer problems from evidence. "
                                                             "Be brief and label uncertainty."),
                                                 ChatMessage("user", prompt)])
        except ModelError:
            return None, None
        return personality.clean(routed.response.content), routed.response.model

    async def _why(self, intent: Intent) -> Response:
        task = self._resolved_task(intent)
        if task is None and (self.plans.focus_is_plan or (intent.target and not is_pronoun(intent.target)
                                                           and not self.resolver.task(intent.target).item)):
            answer = await self.plans.why(intent)
            if answer is not None:
                return answer
        if task is None and intent.target and not is_pronoun(intent.target):
            task = self.resolver.task(intent.target).item  # type: ignore[assignment]
        return self._reply(reports.why(self.svc, task), intent,
                           provenance=[Provenance(ProvenanceKind.DATABASE, "audit log")])

    async def _what_did_you_do(self, intent: Intent) -> Response:
        return self._reply(reports.what_did_you_do(self.svc, self.session_since if "away" in intent.text else None),
                           intent, provenance=[Provenance(ProvenanceKind.DATABASE, "audit log")])

    async def _provenance(self, intent: Intent) -> Response:
        return self._reply(reports.provenance(self.svc, self.focus.last_provenance), intent)

    # -- control --------------------------------------------------------------------------------
    async def _stop(self, intent: Intent) -> Response:
        svc = self.svc
        hard = bool(_HARD_STOP.match(intent.text))
        stopped = await self.plans.stop(intent, hard)
        if stopped is not None:
            return stopped
        verb = "cancel" if hard else "stop"
        if is_everything(intent.target):
            plans, self.plans.stopped_plans = list(self.plans.stopped_plans), []
            tasks = [t for t in svc.tasks.open_tasks() if not t.outputs.get("plan_id")]
            if not tasks and not plans:
                return self._reply(f"Nothing is running, so there's nothing to {verb}.", intent)
            for t in tasks:
                (svc.tasks.cancel_task if hard else svc.tasks.pause_task)(
                    t.id, by=svc.user, reason="cancelled by you" if hard else "stopped by you")
            done = "Cancelled" if hard else "Stopped"
            parts = [_lower(p) for p in plans[:3]] + ([f"{len(plans) - 3} more plans"] if len(plans) > 3 else [])
            if tasks:
                parts.append(f"{len(tasks)} task{'s' if len(tasks) != 1 else ''}")
            return self._reply(f"{done} {personality.join_clauses(parts)}."
                               + ("" if hard else " Everything is checkpointed; say 'continue' to resume."), intent,
                               kind="action")
        task = self._resolved_task(intent)
        if task is None:
            res = self.resolver.task(None if is_pronoun(intent.target) else intent.target,
                                     statuses=[S.QUEUED, S.PLANNING, S.RUNNING, S.VERIFYING, S.WAITING, S.BLOCKED,
                                               S.PAUSED] if hard else [S.QUEUED, S.PLANNING, S.RUNNING, S.VERIFYING,
                                                                       S.WAITING, S.BLOCKED],
                                     prefer=list(EXECUTING))
            task = res.item  # type: ignore[assignment]
            if task is None and res.ambiguous and not is_pronoun(intent.target):
                return self._ask(self._task_choice_question(verb, res.candidates), intent, res.candidates, "task")
            if task is None:
                finished = self._finished_work(intent.target)
                if finished:
                    return self._reply(finished, intent)
                nothing = self._reply(self._nothing_to_stop(intent, verb), intent)
                words = set(re.findall(r"[a-z]+", (intent.target or "").lower()))
                if is_pronoun(intent.target) or words & _WORK_WORDS or \
                        (self._open_work() and not _names_a_program(intent.target or "")):
                    # about JARVIS's own work: answer from the record, never from the model (it can't cancel
                    # plans or tasks, must not claim it did, and must not mistake it for something to delete)
                    return nothing
                # a program ("stop firefox"): the model can look it up and stop it, asking first
                return await self._or_chat(intent, nothing)
        plan_id = task.outputs.get("plan_id")
        if plan_id and svc.intelligence is not None and (plan := svc.intelligence.get(plan_id)) is not None \
                and not plan.terminal:
            # a step of a plan: stopping it means stopping the plan (never one step behind the plan's back)
            engine = svc.intelligence.engine
            ok, message = await (engine.cancel(plan.id, by=svc.user, reason="cancelled by you") if hard else
                                 engine.pause(plan.id, by=svc.user, reason="stopped by you"))
            text = (f"Cancelled {_lower(plan.title)}. Completed steps are kept in the history." if hard else
                    f"Stopped {_lower(plan.title)}. It's checkpointed; say 'continue' to resume.")
            return self._reply(text if ok else personality.sentence(message), intent, kind="action",
                               data={"plan_id": plan.id})
        if hard:
            result = svc.tasks.cancel_task(task.id, by=svc.user, reason="cancelled by you")
        else:
            result = svc.tasks.pause_task(task.id, by=svc.user, reason="stopped by you")
        self.focus.touch_task(task.id)
        if not result.ok:
            return self._reply(personality.sentence(result.message), intent, task_id=task.id)
        others = [t for t in svc.tasks.open_tasks() if t.id != task.id and t.status in EXECUTING
                  and t.kind != TaskKind.MONITOR]
        text = f"Cancelled {_lower(task.title)}." if hard else \
            f"Stopped {_lower(task.title)}. It's checkpointed; say 'continue' to resume."
        if others:
            text += f" Still running: {personality.join_clauses([_lower(t.title) for t in others[:3]])}."
        return self._reply(text, intent, kind="action", task_id=task.id)

    def _finished_work(self, target: str | None) -> str:
        """'cancel the free up disk space process' when that plan already ended: say so instead of guessing."""
        intel = self.svc.intelligence
        if intel is None or not target or is_pronoun(target):
            return ""
        found = self.plans.find_plans(target)
        plan = found[0] if found else None
        if plan is None or not plan.terminal:
            return ""
        elapsed = self.svc.clock.now() - (plan.finished_at or plan.updated_at)
        ago = "just now" if elapsed < 10 else f"{personality.duration(elapsed)} ago"
        state = "finished" if plan.status.value == "completed" else plan.status.value
        return f"{plan.title} was already {state} {ago}, so there's nothing to stop."

    def _open_work(self) -> list[str]:
        """Titles of the open plans and the open tasks outside plans (what "stop"/"cancel" could mean)."""
        svc = self.svc
        names = [p.title for p in svc.intelligence.open_plans()] if svc.intelligence is not None else []
        return names + [t.title for t in svc.tasks.open_tasks()
                        if not t.outputs.get("plan_id") and t.kind != TaskKind.MONITOR]

    def _nothing_to_stop(self, intent: Intent, verb: str) -> str:
        """No match for a stop/cancel/pause: say so plainly (nothing was changed) and list what could be stopped."""
        verb = "pause" if intent.kind == IntentKind.PAUSE else verb
        done = {"cancel": "cancelled", "stop": "stopped", "pause": "paused"}[verb]
        names = self._open_work()
        if not names:
            return f"Nothing is running, so there's nothing to {verb}."
        if intent.target and not is_pronoun(intent.target):
            head = f"I couldn't find anything matching '{intent.target}', so I haven't {done} anything."
        else:
            head = f"I'm not sure which one you mean, so I haven't {done} anything."
        listed = "; ".join(f"'{n}'" for n in names[:6])
        return f"{head} Open right now: {listed}. Tell me which one, for example '{verb} {_lower(names[0])}'."

    async def _resume(self, intent: Intent) -> Response:
        svc = self.svc
        task = self._resolved_task(intent)
        if task is None:
            resumed = await self.plans.resume(intent)
            if resumed is not None:
                return resumed
        if task is None:
            target = intent.target
            if target and target.lower().strip() in ("project", "the project", "current project"):
                target = None
            res = self.resolver.task(None if is_pronoun(target) else target,
                                     statuses=[S.PAUSED, S.BLOCKED, S.WAITING, S.FAILED],
                                     prefer=[S.PAUSED, S.BLOCKED, S.WAITING])
            if res.item is None and res.ambiguous:
                return self._ask(self._task_choice_question("continue", res.candidates), intent, res.candidates, "task")
            task = res.item  # type: ignore[assignment]
            if task is None and target and not is_pronoun(target):
                project = self.resolver.project(target).item
                if isinstance(project, Project):
                    paused = [t for t in svc.tasks.list_tasks([S.PAUSED, S.BLOCKED, S.FAILED], project_id=project.id,
                                                              order="recent")]
                    task = paused[0] if paused else None
        if task is None:
            return self._reply("There's nothing paused or interrupted to continue.", intent)
        self.focus.touch_task(task.id)
        if task.status == S.WAITING:
            pending = [a for a in svc.approvals.pending() if a.task_id == task.id]
            if pending:
                return self._reply(f"{task.title} is waiting for your approval to {pending[0].summary}. Proceed?",
                                   intent, kind="question", task_id=task.id, approval_id=pending[0].id)
        result = svc.tasks.resume_task(task.id, by=svc.user)
        if not result.ok:
            return self._reply(personality.sentence(result.message), intent, task_id=task.id)
        done = [s.description for s in task.completed_steps()]
        pending_steps = [s.description for s in task.pending_steps()]
        text = f"Resuming {_lower(task.title)}"
        if done:
            text += f". Already done: {personality.join_clauses(done[-3:])}"
        if pending_steps:
            text += f". Next: {pending_steps[0]}"
        return self._reply(personality.sentence(text), intent, kind="action", task_id=task.id)

    async def _retry(self, intent: Intent) -> Response:
        svc = self.svc
        source = self.resolver.task(None, statuses=None).item
        if source is None:
            recent = svc.tasks.most_recent(kinds=[TaskKind.ONESHOT])
            source = recent
        if source is None:
            return self._reply("There's nothing recent to run again.", intent)
        if source.status in OPEN and source.status not in (S.PAUSED,):
            return self._reply(f"{source.title} is still {source.status.value}.", intent, task_id=source.id)
        clone = self._clone(source)
        return self._reply(f"Running {_lower(source.title)} again.", intent, kind="action", task_id=clone.id)

    def _clone(self, source: Task, *, root: str | None = None, project: Project | None = None,
               title: str | None = None) -> Task:
        template = source.outputs.get("template")
        root = root or source.cwd
        if template == "run_tests" and root:
            steps, success = run_tests_plan(root)
        elif template == "build" and root:
            steps, success = build_plan(root)
        else:
            steps = [Step(s.description, s.tool, _rebase(dict(s.args), source.cwd, root), allow_failure=s.allow_failure)
                     for s in source.plan if s.tool and s.note != "superseded by a revised plan"]
            success = source.success_condition
        return self._user_task(source.objective, title=title or source.title, steps=steps or None, success=success,
                               cwd=root, dry_run=source.dry_run, project=project, template=template,
                               policy=source.policy)

    async def _approve(self, intent: Intent) -> Response:
        svc = self.svc
        planned = await self.plans.approve(intent)
        if planned is not None:
            return planned
        pending = svc.approvals.pending()
        if not pending:
            nothing = self._reply("There's nothing waiting for approval.", intent)
            if self._model_asked:
                return await self._or_chat(intent, nothing)    # "yes" to something the model asked
            return nothing
        chosen_id = intent.params.get("resolved_id")
        if chosen_id is None and len(pending) > 1 and "all" not in intent.text.lower():
            focused = [a for a in pending if self.focus.tasks and a.task_id == self.focus.tasks[0]]
            if len(focused) == 1:
                chosen = [focused[0]]
            else:
                options = "; ".join(f"{i + 1}) {a.summary}" for i, a in enumerate(pending[:4]))
                self.pending_question = {"intent": intent, "ids": [a.id for a in pending[:4]], "entity": "approval",
                                         "labels": [a.summary for a in pending[:4]]}
                return self._reply(f"{len(pending)} actions are waiting. Which should I approve? {options}", intent,
                                   kind="question")
        else:
            chosen = [a for a in pending if a.id == chosen_id] if chosen_id else \
                (pending if "all" in intent.text.lower() else [pending[-1]])
        summaries = []
        for approval in chosen:
            svc.approvals.approve(approval.id, by=svc.user)
            if approval.task_id:
                svc.tasks.resume_task(approval.task_id, by=svc.user, reason="approved by you")
                self.focus.touch_task(approval.task_id)
            summaries.append(approval.summary)
        if len(chosen) == 1:
            followed = await self.plans.after_approval(chosen[0], intent)
            if followed is not None:
                return followed
        return self._reply(f"Proceeding: {personality.join_clauses(summaries)}.", intent, kind="action",
                           task_id=chosen[0].task_id)

    async def _deny(self, intent: Intent) -> Response:
        svc = self.svc
        declined = await self.plans.deny(intent)
        if declined is not None:
            return declined
        pending = svc.approvals.pending()
        if not pending:
            return self._reply("Understood.", intent)
        approval = pending[-1]
        if self.focus.tasks:
            focused = [a for a in pending if a.task_id == self.focus.tasks[0]]
            approval = focused[-1] if focused else approval
        svc.approvals.deny(approval.id, by=svc.user)
        text = f"Understood — I won't {approval.summary}."
        if approval.task_id:
            task = svc.tasks.get_task(approval.task_id)
            if task:
                for step in task.plan:
                    if step.id == approval.step_id:
                        step.status = StepStatus.SKIPPED
                        step.note = "declined by you"
                svc.tasks.save(task)
                remaining = [s for s in task.pending_steps()]
                if not remaining and not task.outputs.get("plan_id") and \
                        not any(s.status == StepStatus.DONE for s in task.plan):
                    # nothing was done and nothing is left: it ends as declined, never as "completed"
                    # (a plan's approval step is different: the plan records the decline itself)
                    svc.tasks.cancel_task(task.id, by=svc.user, reason="you declined it")
                else:
                    svc.tasks.resume_task(task.id, by=svc.user, reason="continuing without the declined step")
                if remaining:
                    text += " The rest of the task continues."
        return self._reply(text, intent, kind="action", task_id=approval.task_id)

    async def _modify_plan(self, intent: Intent) -> Response:
        svc = self.svc
        words = [w for w in re.findall(r"[a-z0-9]+", (intent.target or "").lower()) if w not in ("the", "a", "it", "yet")]
        if not words:
            return self._reply("Which step should I drop?", intent, kind="question")
        stems = [w[:5] for w in words]

        def matches(text: str) -> bool:
            lowered = text.lower()
            return all(stem in lowered for stem in stems)

        changed = []
        for approval in svc.approvals.pending():
            if matches(approval.summary):
                await self._deny(Intent(IntentKind.DENY, "no"))
                changed.append(approval.summary)
        for task in svc.tasks.open_tasks():
            result = svc.tasks.skip_steps(task.id, lambda s: matches(s.description) or matches(json.dumps(s.args)),
                                          by=svc.user, reason=f"you said: {intent.text.strip()}")
            if result.ok:
                changed.append(f"{result.message} (in {_lower(task.title)})")
                self.focus.touch_task(task.id)
        if not changed:
            return self._reply(f"Nothing in the current plans matches '{intent.target}'.", intent)
        return self._reply(personality.sentence(f"Done: {'; '.join(changed)}") +
                           " Completed work is kept.", intent, kind="action")

    async def _priority(self, intent: Intent) -> Response:
        res = self.resolver.task(intent.target)
        task = self._resolved_task(intent) or res.item
        if task is None:
            if res.ambiguous:
                return self._ask(self._task_choice_question("reprioritize", res.candidates), intent, res.candidates, "task")
            return self._reply(f"I couldn't find a task matching '{intent.target}'.", intent)
        level = (intent.params.get("level") or "top").lower()
        priority = Priority.P1 if level in ("top", "high", "urgent") else Priority.P3
        result = self.svc.tasks.reprioritize(task.id, priority, by=self.svc.user)
        return self._reply(personality.sentence(result.message), intent, kind="action", task_id=task.id)

    async def _mode(self, intent: Intent) -> Response:
        svc = self.svc
        name = intent.params.get("mode", "normal")
        on = intent.params.get("on", True)
        if name == "quiet":
            svc.modes.set_quiet(on)
            return self._reply("Going quiet. Background work continues; only critical alerts will interrupt."
                               if on else "Notifications are back to normal.", intent, kind="action")
        if name == "private":
            svc.modes.set_private(on)
            return self._reply("Private mode. Local models only, network tools disabled, nothing leaves this machine."
                               if on else "Private mode off. Optional network tools are available again.", intent,
                               kind="action")
        if name == "emergency":
            if on:
                await svc.emergency.enter(reason="activated by you")
                return self._reply("Emergency mode. Non-essential work is deprioritised, evidence is being preserved, "
                                   "and only urgent alerts will reach you.", intent, kind="action")
            exited = await svc.emergency.exit("ended by you", by=svc.user)
            return self._reply("Emergency mode ended." if exited else "Emergency mode isn't active.", intent)
        name = {"dev": "development"}.get(name, name)
        try:
            mode = Mode(name)
        except ValueError:
            return self._reply(f"I don't have a {name} mode.", intent)
        if on:
            svc.modes.set_mode(mode, by=svc.user)
            descriptions = {
                Mode.FOCUS: "Focus mode. I'll hold anything that isn't urgent; background work continues.",
                Mode.PRESENTATION: "Presentation mode. Only critical alerts will interrupt.",
                Mode.DEBUG: "Debug mode. I'll surface operational detail and more notifications.",
                Mode.OFFLINE: "Offline mode. Network tools are disabled; local models and tools only.",
                Mode.LOW_RESOURCE: "Low-resource mode. Only essential background work will run.",
                Mode.NORMAL: "Back to normal.",
            }
            return self._reply(descriptions.get(mode, f"{mode.value.replace('_', ' ').capitalize()} mode."), intent,
                               kind="action")
        if svc.modes.current != mode:
            return self._reply(f"{mode.value.replace('_', ' ').capitalize()} mode isn't active.", intent)
        svc.modes.set_mode(Mode.NORMAL, by=svc.user)
        queued = svc.notifications.drain(limit=5)
        text = "Back to normal."
        if queued:
            text += " While you were focused: " + "; ".join(n.text() for n in queued)
        return self._reply(text, intent, kind="action")

    # -- memory ---------------------------------------------------------------------------------
    async def _remember(self, intent: Intent) -> Response:
        svc = self.svc
        content = intent.params.get("content")
        if not content:
            content = self._previous_user_statement()
            if not content:
                return self._reply("Remember what, exactly?", intent, kind="question")
        kind = _memory_kind(content, svc.projects.active() is not None)
        project = svc.projects.active()
        item = await svc.memory.remember(content, kind=kind, project_id=project.id if project and
                                         kind == MemoryKind.PROJECT else None,
                                         provenance=Provenance(ProvenanceKind.USER_STATEMENT, "you said"), force=True)
        if item is None:
            return self._reply("I couldn't store that.", intent, kind="error")
        scope = f" for {project.name}" if item.project_id and project else ""
        return self._reply(f"Noted{scope}.", intent, kind="action",
                           provenance=[Provenance(ProvenanceKind.USER_STATEMENT, "you")])

    def _previous_user_statement(self) -> str | None:
        users = [m.content for m in self.history if m.role == "user"]
        for text in reversed(users[:-1]):
            if parse(text).kind in (IntentKind.CHAT,) or len(text.split()) > 3:
                return text
        return None

    async def _dont_remember(self, intent: Intent) -> Response:
        self.svc.memory.suppressed = True
        return self._reply("Understood. Nothing from this conversation will be stored until you tell me to "
                           "remember something.", intent, kind="action")

    async def _forget(self, intent: Intent) -> Response:
        svc = self.svc
        target = intent.target
        if target is None:
            last = svc.memory.last_stored_id
            if last is None:
                return self._reply("I haven't stored anything recently to forget.", intent)
            item = svc.memory.get(last)
            svc.memory.forget(last)
            return self._reply(f"Forgotten: \"{item.content if item else 'that'}\".", intent, kind="action")
        project = self.resolver.project(target).item if "project" in target.lower() or intent.params.get("all") else None
        if isinstance(project, Project):
            n = svc.memory.forget(project_id=project.id)
            return self._reply(f"Deleted {n} memor{'ies' if n != 1 else 'y'} about {project.name}.", intent,
                               kind="action")
        n = svc.memory.forget(query=target)
        if n == 0:
            return self._reply(f"I don't have anything stored about {target}.", intent)
        return self._reply(f"Deleted {n} memor{'ies' if n != 1 else 'y'} about {target}.", intent, kind="action")

    async def _recall(self, intent: Intent) -> Response:
        svc = self.svc
        target = intent.target or ""
        project = self.resolver.project(target).item if target else None
        if isinstance(project, Project):
            items = svc.memory.list(project_id=project.id, limit=8)
            decisions = svc.decisions.list(project_id=project.id, limit=3)
            if not items and not decisions:
                return self._reply(f"I don't have anything stored about {project.name} yet.", intent)
            parts = [f"About {project.name}:"] + [f"{i.content}." for i in items[:6]]
            if decisions:
                parts.append("Decisions: " + "; ".join(f"{d.title}: {d.decision}" for d in decisions) + ".")
            provs = [Provenance(ProvenanceKind.MEMORY, i.id, i.content[:40]) for i in items]
            return self._reply(" ".join(parts), intent, provenance=provs)
        hits = await svc.memory.retrieve(target, limit=5)
        if not hits:
            return self._reply(f"I don't remember anything about {target}.", intent)
        lines = [h.item.content.rstrip(".") for h in hits]
        return self._reply("I remember: " + "; ".join(lines) + ".", intent,
                           provenance=[Provenance(ProvenanceKind.MEMORY, h.item.id, h.item.content[:40]) for h in hits])

    async def _decision_why(self, intent: Intent) -> Response:
        project = self.svc.projects.active()
        found = self.svc.decisions.search(intent.target or intent.text, project_id=project.id if project else None,
                                          limit=1)
        if not found:
            return self._reply(f"I don't have a recorded decision about {intent.target}. If you tell me the reasoning, "
                               "I'll log it.", intent)
        d = found[0]
        when = reports.format_ts(d.ts)
        return self._reply(f"{d.explain()} (recorded {when})", intent,
                           provenance=[Provenance(ProvenanceKind.DATABASE, f"decision history {d.id}", d.title)])

    async def _record_decision(self, intent: Intent) -> Response:
        content = intent.params.get("content", "").strip()
        decision, _, reason = content.partition(" because ")
        project = self.svc.projects.active()
        rec = self.svc.decisions.record(decision[:80], decision, reason=reason, user_involvement="stated by you",
                                        project_id=project.id if project else None)
        return self._reply(f"Logged the decision: {rec.decision}" + (f", because {rec.reason}." if rec.reason else "."),
                           intent, kind="action")

    async def _log_that(self, intent: Intent) -> Response:
        users = [m.content for m in self.history if m.role == "user"][:-1]
        last_user = users[-1] if users else ""
        last_reply = self.focus.last_reply
        if not last_user and not last_reply:
            return self._reply("There's nothing to log yet.", intent)
        project = self.svc.projects.active()
        await self.svc.memory.remember(f"{last_user} → {last_reply}"[:1000], kind=MemoryKind.EPISODIC,
                                       project_id=project.id if project else None, force=True,
                                       provenance=Provenance(ProvenanceKind.USER_STATEMENT, "log that"))
        return self._reply("Logged.", intent, kind="action")

    # -- models -------------------------------------------------------------------------------------
    async def _model_use(self, intent: Intent) -> Response:
        router = self.svc.router
        if not router.inventory:
            await router.refresh()
        target = (intent.target or "").lower()
        purpose_for = {"coding": Purpose.CODING, "code": Purpose.CODING, "vision": Purpose.VISION,
                       "reasoning": Purpose.REASONING, "fast": Purpose.BACKGROUND, "small": Purpose.BACKGROUND,
                       "big": Purpose.REASONING}
        try:
            if target == "default":
                router.unpin()
                self.svc.state.set("models.prefer_local", False)
                return self._reply("Back to automatic model selection.", intent, kind="action")
            if target == "local":
                decision = router.select(TaskProfile(local_only=True))
                router.pin(decision.model)
                self.svc.state.set("models.prefer_local", True)
                return self._reply(f"Using the local model {decision.model}.", intent, kind="action")
            if target in purpose_for:
                profile = TaskProfile(purpose=purpose_for[target],
                                      complexity="high" if target in ("big", "reasoning") else "low",
                                      needs_vision=target == "vision")
                decision = router.select(profile)
                router.pin(decision.model)
                return self._reply(f"Using {decision.model} ({decision.reason}).", intent, kind="action")
            info = router.pin(target)
            return self._reply(f"Using {info.name}.", intent, kind="action")
        except NoModelAvailable as exc:
            return self._reply(f"I can't: {exc}.", intent, kind="error")

    async def _model_unload(self, intent: Intent) -> Response:
        router = self.svc.router
        target = (intent.target or "").lower()
        candidates = [m for m in router.inventory if m.loaded]
        if target == "vision":
            candidates = [m for m in candidates if m.has(Capability.VISION)]
        elif target:
            candidates = [m for m in candidates if target in m.name.lower()] or \
                [m for m in router.inventory if target in m.name.lower()]
        if not candidates:
            return self._reply(f"No loaded model matches '{target}'.", intent)
        try:
            info = await router.unload(candidates[0].name)
        except ModelError as exc:
            return self._reply(f"Unloading failed: {exc}", intent, kind="error")
        self.svc.audit.record(actor=f"user:{self.svc.user}", action="unload_model", summary=f"unloaded {info.name}")
        return self._reply(f"Unloaded {info.name}.", intent, kind="action")

    async def _model_list(self, intent: Intent) -> Response:
        if not self.svc.router.inventory:
            try:
                await self.svc.router.refresh()
            except Exception:
                pass
        return self._reply(reports.models_report(self.svc), intent,
                           provenance=[Provenance(ProvenanceKind.SYSTEM_STATE, "model manager")])

    # -- delegation / monitoring -----------------------------------------------------------------------
    async def _monitor(self, intent: Intent) -> Response:
        svc = self.svc
        target = intent.target
        notify = intent.params.get("notify", "important")
        until_done = intent.params.get("until_done", False)
        now = svc.clock.now()
        looks_like_path = intent.params.get("type") == "path" or (target and target.strip()[:1] in ("/", "~", "."))
        if looks_like_path and target:
            path = os.path.realpath(os.path.expanduser(target.strip()))
            if not os.path.exists(path):
                return self._reply(f"{path} doesn't exist, so there's nothing to watch.", intent)
            ok, why = svc.permissions.paths.check(path)
            if not ok:
                return self._reply(f"I can't watch that: {why}.", intent)
            task = svc.tasks.create_task(f"watch {path}", title=f"watch {os.path.basename(path) or path}",
                                         kind=TaskKind.MONITOR, created_by=f"user:{svc.user}",
                                         monitor=MonitorSpec({"type": "path", "path": path}, interval_s=5.0,
                                                             notify=notify, stop_when=[],
                                                             expires_at=now + 7 * 86400),
                                         priority=Priority.P2)
            self.focus.touch_task(task.id)
            return self._reply(f"Watching {path}. I'll tell you about changes; the watch expires in 7 days unless you "
                               "stop it sooner.", intent, kind="action", task_id=task.id)
        watched = self._resolved_task(intent)
        if watched is None:
            if target is None or is_pronoun(target):
                res = self.resolver.task(None, statuses=[S.QUEUED, S.PLANNING, S.RUNNING, S.VERIFYING, S.WAITING,
                                                         S.BLOCKED, S.COMPLETED, S.FAILED],
                                         prefer=list(EXECUTING), include_monitors=False)
            else:
                res = self.resolver.task(target, include_monitors=False)
                if res.ambiguous:
                    return self._ask(self._task_choice_question("watch", res.candidates), intent, res.candidates, "task")
            watched = res.item  # type: ignore[assignment]
        if watched is not None:
            self.focus.touch_task(watched.id)
            if watched.terminal:
                return self._reply(f"It already finished. {reports.task_status(svc, watched)}", intent,
                                   task_id=watched.id)
            monitor = svc.tasks.create_task(f"keep an eye on {watched.title}", title=f"watch {watched.title}",
                                            kind=TaskKind.MONITOR, created_by=f"user:{svc.user}",
                                            monitor=MonitorSpec({"type": "task", "task_id": watched.id},
                                                                interval_s=1.0, notify=notify,
                                                                stop_when=["target_resolved"],
                                                                expires_at=now + 7 * 86400),
                                            priority=Priority.P2)
            text = f"I'll keep an eye on {_lower(watched.title)}" + \
                (" and tell you when it's done." if until_done else " and tell you if anything needs attention.")
            return self._reply(text, intent, kind="action", task_id=monitor.id)
        entity_cmd = self._health_command_for(target)
        if entity_cmd:
            return await self._create_command_monitor(intent, entity_cmd, target or "it", notify)
        self.pending_question = {"entity": "monitor_spec", "intent": intent, "target": target, "notify": notify}
        return self._reply(f"What should I check to know {target or 'it'} is healthy — a process ID, a folder, "
                           "or a command such as `curl -fsS localhost:8080/health`?", intent, kind="question")

    def _health_command_for(self, target: str | None) -> str | None:
        if not target:
            return None
        for e in self.svc.world.resolve_name(re.sub(r"^(the|my)\s+", "", target.lower()),
                                             ["service", "application"]):
            if e.attrs.get("health_command"):
                return str(e.attrs["health_command"])
        return None

    async def _monitor_from_answer(self, text: str, pq: dict[str, Any]) -> Response:
        intent: Intent = pq["intent"]
        answer = text.strip().strip("`")
        pid = re.fullmatch(r"(?:pid|process)?\s*(\d+)", answer, re.IGNORECASE)
        if pid:
            task = self.svc.tasks.create_task(f"watch process {pid.group(1)}", kind=TaskKind.MONITOR,
                                              created_by=f"user:{self.svc.user}", priority=Priority.P2,
                                              monitor=MonitorSpec({"type": "process", "pid": int(pid.group(1)),
                                                                   "name": pq.get("target")}, interval_s=2.0,
                                                                  notify=pq.get("notify", "important")))
            return self._reply(f"Watching process {pid.group(1)}.", intent, kind="action", task_id=task.id)
        if answer[:1] in ("/", "~", "."):
            return await self._monitor(Intent(IntentKind.MONITOR, answer, answer, {"type": "path"}))
        return await self._create_command_monitor(intent, answer, pq.get("target") or "it", pq.get("notify", "important"))

    async def _create_command_monitor(self, intent: Intent, command: str, label: str, notify: str) -> Response:
        svc = self.svc
        from jarvis.tools.builtin.shell import classify_command
        assessment = classify_command(command)
        if assessment.blocked or assessment.level > svc.permissions.config.interactive_level:
            return self._reply(f"`{command}` isn't a safe read-only check ({assessment.reason}), so I won't run it "
                               "repeatedly without a scoped grant.", intent)
        task = svc.tasks.create_task(f"keep an eye on {label}", title=f"watch {label}", kind=TaskKind.MONITOR,
                                     created_by=f"user:{svc.user}", priority=Priority.P2,
                                     cwd=self._work_root(),
                                     monitor=MonitorSpec({"type": "command", "command": command, "label": label},
                                                         interval_s=30.0, notify="urgent" if notify == "important"
                                                         else notify, stop_when=[],
                                                         expires_at=svc.clock.now() + 7 * 86400))
        self.focus.touch_task(task.id)
        return self._reply(f"I'll check {label} every 30 seconds with `{command}` and tell you if it fails or recovers.",
                           intent, kind="action", task_id=task.id)

    # -- engineering --------------------------------------------------------------------------------------
    async def _run_tests(self, intent: Intent) -> Response:
        project = self._resolved_project(intent)
        candidates: list[Project] = []
        if project is None:
            project, candidates = self._project_for(intent.target)
        if project is None:
            if candidates:
                return self._ask("Which project? " + "; ".join(f"{i + 1}) {p.name}" for i, p in enumerate(candidates)),
                                 intent, candidates, "project")
            return self._reply("I couldn't find a project to test. Open one first (e.g. 'open the robotics project') "
                               "or give me a path.", intent, kind="question")
        try:
            steps, success = run_tests_plan(project.root or self._work_root())
        except LookupError as exc:
            return self._reply(f"{exc}.", intent)
        self.focus.touch_project(project.id)
        task = self._user_task(f"run the tests for {project.name}", title=f"{project.name} tests", steps=steps,
                               success=success, cwd=project.root, dry_run=intent.dry_run, project=project,
                               template="run_tests")
        command = steps[0].args["command"]
        text = f"Running the {project.name} test suite (`{command}`). I'll let you know how it goes."
        rest = (intent.params.get("rest") or "").lower()
        if "fix" in rest:
            if self.svc.router.available(Capability.CHAT):
                fixer = self._user_task(
                    f"Fix the obvious, isolated failures from the latest test run of {project.name} "
                    f"(test task {task.id}). Inspect failing tests and code first; make minimal changes with "
                    "file_write; re-run the tests; leave anything architectural or unclear untouched and report it.",
                    title=f"fix obvious {project.name} test failures", cwd=project.root, project=project,
                    dependencies=[task.id], dry_run=intent.dry_run)
                text += (" Then I'll fix whatever is clearly isolated and re-run them; anything that needs a design "
                         "decision I'll leave for you.")
                self.focus.touch_task(fixer.id)
            else:
                text += " Fixing failures needs a language model, and none is available, so I'll only report them."
        if intent.dry_run:
            text = f"Dry run: I would run `{command}` in {project.root}. Nothing will be executed."
        return self._reply(text, intent, kind="action", task_id=task.id)

    async def _build(self, intent: Intent) -> Response:
        project, candidates = self._project_for(intent.target)
        if project is None:
            return self._reply("Which project should I build?", intent, kind="question")
        try:
            steps, success = build_plan(project.root or self._work_root())
        except LookupError as exc:
            return self._reply(f"{exc}.", intent)
        task = self._user_task(f"build {project.name}", title=f"{project.name} build", steps=steps, success=success,
                               cwd=project.root, dry_run=intent.dry_run, project=project, template="build")
        return self._reply(f"Building {project.name} (`{steps[0].args['command']}`).", intent, kind="action",
                           task_id=task.id)

    async def _shell(self, intent: Intent) -> Response:
        svc = self.svc
        command = (intent.params.get("command") or "").strip()
        if not command:
            return self._reply("Which command?", intent, kind="question")
        self.focus.last_command = command
        cwd = self._work_root()
        task = self._user_task(f"run `{command}`", title=f"$ {command[:60]}",
                               steps=[Step(f"run `{command}`", "shell_execute",
                                           {"command": command, "cwd": cwd, "timeout_s": 600})],
                               cwd=cwd, dry_run=intent.dry_run, policy=TaskPolicy(on_step_failure="fail"))
        try:
            done = await svc.pool.wait_for(task.id, [S.COMPLETED, S.FAILED, S.WAITING, S.BLOCKED, S.CANCELLED],
                                           timeout=10.0)
        except TimeoutError:
            return self._reply(f"`{command}` is still running. I'll tell you when it finishes.", intent,
                               kind="action", task_id=task.id)
        return self._shell_result(intent, command, done)

    def _shell_result(self, intent: Intent, command: str, task: Task) -> Response:
        step = task.plan[0] if task.plan else None
        prov = [Provenance(ProvenanceKind.TOOL_OUTPUT, "shell_execute", command)]
        if task.status == S.WAITING:
            pending = [a for a in self.svc.approvals.pending() if a.task_id == task.id]
            return self._reply(f"`{command}` needs your approval ({pending[0].reason if pending else 'consequential'}"
                               "). Proceed?", intent, kind="question", task_id=task.id,
                               approval_id=pending[0].id if pending else None)
        if task.status == S.BLOCKED:
            return self._reply(f"I won't run `{command}`: {task.status_reason}.", intent, task_id=task.id)
        if task.dry_run:
            return self._reply(f"Dry run: would run `{command}` in {task.cwd}. Nothing was executed.", intent,
                               task_id=task.id)
        data = (step.result or {}).get("data") if step else None
        if not isinstance(data, dict):
            return self._reply(personality.sentence(task.outputs.get("summary") or task.status_reason), intent,
                               task_id=task.id, provenance=prov)
        output = (data.get("stdout") or "").strip()
        err = (data.get("stderr") or "").strip()
        code = data.get("exit_code")
        shown = output[-1500:] if output else ""
        if code == 0:
            text = f"Done (exit 0)." + (f"\n{shown}" if shown else " No output.")
        else:
            text = f"`{command}` exited with {code}." + (f"\n{(err or output)[-1500:]}" if (err or output) else "")
        return self._reply(text, intent, kind="action", task_id=task.id, provenance=prov)

    async def _open_project(self, intent: Intent) -> Response:
        svc = self.svc
        project = self._resolved_project(intent)
        if project is None:
            target = intent.target or ""
            res = self.resolver.project(target)
            if res.item is None and res.candidates:
                return self._ask("Which one? " + "; ".join(f"{i + 1}) {p.name}" for i, p in enumerate(res.candidates)),
                                 intent, list(res.candidates), "project")
            project = res.item  # type: ignore[assignment]
            if project is None:
                expanded = os.path.expanduser(target)
                if os.path.isdir(expanded):
                    project = svc.projects.discover(expanded)
                else:
                    known = ", ".join(p.name for p in svc.projects.list()[:6])
                    return await self._or_chat(intent, self._reply(
                        f"I don't know a project called '{target}'."
                        + (f" Known projects: {known}." if known else " Give me its folder and I'll register it."),
                        intent))
        ctx = svc.projects.open(project)
        self.focus.touch_project(project.id)
        facts = []
        if ctx.get("branch"):
            facts.append(f"branch {ctx['branch']}")
        if ctx.get("languages"):
            facts.append("/".join(ctx["languages"]))
        if ctx.get("test_command"):
            facts.append(f"tests: `{ctx['test_command']}`")
        open_tasks = [t for t in svc.tasks.open_tasks() if t.project_id == project.id]
        mems = svc.memory.count(project.id)
        text = f"Opened {project.name}" + (f" ({', '.join(facts)})" if facts else "") + "."
        if open_tasks:
            text += f" {len(open_tasks)} open task{'s' if len(open_tasks) != 1 else ''}."
        if mems:
            text += f" I have {mems} note{'s' if mems != 1 else ''} about it."
        if project.policy.sensitive:
            text += " It's marked sensitive: local models only" + ("" if project.policy.network else
                                                                 ", network tools disabled") + "."
        return self._reply(text, intent, kind="action",
                           provenance=[Provenance(ProvenanceKind.LOCAL_FILE, project.root or project.name)],
                           data={"project": ctx})

    async def _repeat_for(self, intent: Intent) -> Response:
        svc = self.svc
        source = self.resolver.task(None, statuses=None, include_monitors=False).item
        if source is None:
            return self._reply("There's no recent task to repeat.", intent)
        target = intent.target or "other"
        project = self._resolved_project(intent)
        if project is None:
            res = self.resolver.project(target)
            if res.item is None and res.candidates:
                return self._ask("For which project? " + "; ".join(f"{i + 1}) {p.name}"
                                                                  for i, p in enumerate(res.candidates)),
                                 intent, list(res.candidates), "project")
            project = res.item  # type: ignore[assignment]
        if project is None or not project.root:
            return self._reply(f"I couldn't find the project '{target}'.", intent)
        title = source.title
        old_project = svc.projects.get(source.project_id) if source.project_id else None
        if old_project and old_project.name in title:
            title = title.replace(old_project.name, project.name)
        elif project.name not in title:
            title = f"{title} ({project.name})"
        clone = self._clone(source, root=project.root, project=project, title=title)
        self.focus.touch_project(project.id)
        return self._reply(f"Doing the same for {project.name}: {_lower(source.title)}.", intent, kind="action",
                           task_id=clone.id)

    # -- misc ---------------------------------------------------------------------------------------------
    async def _system_query(self, intent: Intent) -> Response:
        """One live number ("check my CPU temperature"): a single read-only tool call, no planning."""
        svc = self.svc
        ctx = ToolContext(actor=Actor.user(svc.user), cwd=self._work_root(), clock=svc.clock,
                          data_dir=str(svc.config.data_path))
        execution = await svc.registry.execute("system_info", {}, ctx, reason=OperationalReason(
            "you asked: " + intent.text[:80], "answer from live measurements", "read system_info"))
        if not execution.ok or execution.result is None:
            return self._reply(f"I couldn't read the system's state ({execution.message}).", intent, kind="error")
        text = reports.system_fact(execution.result.data or {}, intent.text)
        return self._reply(text, intent, provenance=[Provenance(ProvenanceKind.SYSTEM_STATE, "system_info",
                                                                "measured just now")])

    async def _time(self, intent: Intent) -> Response:
        return self._reply(reports.time_answer(self.svc), intent,
                           provenance=[Provenance(ProvenanceKind.SYSTEM_STATE, "system clock")])

    async def _self(self, intent: Intent) -> Response:
        return self._reply(reports.self_description(self.svc), intent,
                           provenance=[Provenance(ProvenanceKind.CONFIGURATION, "registered subsystems")])

    async def _help(self, intent: Intent) -> Response:
        return self._reply(reports.help_text(), intent)

    def _short(self) -> bool:
        return (self.verbosity_override or self.svc.modes.verbosity()) == "short"

    async def _shorter(self, intent: Intent) -> Response:
        self.verbosity_override = "short"
        return await self._rewrite(intent, "Rewrite this in one or two sentences, keeping only what matters:",
                                   "I'll keep it short.")

    async def _longer(self, intent: Intent) -> Response:
        self.verbosity_override = "detailed"
        return await self._rewrite(intent, "Expand this with full detail and reasoning:", "I'll go into more detail.")

    async def _rewrite(self, intent: Intent, instruction: str, fallback: str) -> Response:
        last = self.focus.last_reply
        if not last or not self.svc.router.available():
            return self._reply(fallback, intent)
        try:
            routed = await self.svc.router.chat(TaskProfile(purpose=Purpose.SUMMARIZATION),
                                                [ChatMessage("user", f"{instruction}\n\n{last}")])
        except ModelError:
            return self._reply(fallback, intent)
        return self._reply(personality.clean(routed.response.content), intent, model=routed.response.model)

    # -- conversation with the model ------------------------------------------------------------------------
    def _register_internal_tools(self) -> None:
        reg = self.svc.registry
        project_id = lambda: (p.id if (p := self.svc.projects.active()) else None)  # noqa: E731
        for tool in (StartTaskTool(self._start_task_from_model), RememberTool(self.svc.memory, project_id),
                     SearchMemoryTool(self.svc.memory, project_id),
                     LiveStateTool(lambda section: summarize_state_for_tool(self.svc, section))):
            if reg.get(tool.spec.name) is None:
                reg.register(tool)

    async def _start_task_from_model(self, objective: str, steps: list[dict[str, Any]] | None, priority: Priority,
                                     notify: bool) -> dict[str, Any]:
        planner_steps: list[Step] = []
        if steps:
            from jarvis.planner.planner import Planner
            validated, rejected = Planner(self.svc.registry).validate(steps, 40)
            if rejected and not validated:
                return {"error": f"invalid steps: {'; '.join(rejected[:3])}"}
            planner_steps = validated
        task = self._user_task(objective, steps=planner_steps or None, priority=priority, cwd=self._work_root())
        if not notify:
            task.policy.notify_on = ["failed"]
            self.svc.tasks.save(task)
        return {"id": task.id, "title": task.title, "steps": len(planner_steps)}

    def _profile(self, text: str, *, tools: bool) -> TaskProfile:
        lowered = text.lower()
        complex_words = ("analy", "design", "architect", "refactor", "implement", "migrat", "plan", "compare",
                         "investigate", "explain why", "prepare")
        complexity = "high" if any(w in lowered for w in complex_words) or len(text) > 400 else \
            "medium" if len(text) > 120 else "low"
        purpose = Purpose.CODING if any(w in lowered for w in ("code", "function", "bug", "stack trace", "compile",
                                                               "refactor", "test")) else Purpose.CONVERSATION
        local_only = bool(self.svc.state.value("models.prefer_local")) or self.svc.modes.effective().local_only
        return TaskProfile(purpose=purpose, complexity=complexity, needs_tools=tools, local_only=local_only)

    def _chat_tools(self, profile: TaskProfile) -> list[dict[str, Any]]:
        """Tool definitions offered to the model. Fewer, relevant tools make small local models far more reliable."""
        svc = self.svc
        exclude = set()
        if not svc.devices.list():
            exclude.add("device_command")
        if profile.complexity != "high":
            exclude.add("delegate_to_agent")
        names = [t.spec.name for t in svc.registry.list() if t.spec.name not in exclude
                 and t.spec.category != "planning"]
        if self._advisory:
            return svc.registry.model_schemas(names, max_level=0)    # advice: observe only
        return svc.registry.model_schemas(names)

    async def _chat(self, intent: Intent, *, advisory: bool = False) -> Response:
        text = intent.text
        if not advisory and intent.source == "model":
            # complexity decides the path: requests that need a plan (or advice, a simulation, a prediction) get
            # one; everything else is a conversation, as before
            planned = await self.plans.maybe_goal(intent)
            if planned is not None:
                return planned
        self._advisory = advisory
        try:
            return await self._converse(intent, text)
        finally:
            self._advisory = False

    async def _converse(self, intent: Intent, text: str) -> Response:
        svc = self.svc
        if not svc.router.available():
            try:
                await svc.router.refresh()
            except Exception:
                pass
        if not svc.router.available():
            down = [p for p, ok in svc.router.provider_status.items() if not ok]
            why = f" ({', '.join(down)} is not reachable)" if down else " (no models are installed)"
            return self._reply(
                f"The language model is unavailable{why}, so I can't handle open-ended requests right now. Everything "
                "else still works: tasks, monitoring, status, memory and commands. Say 'help' for what I can do "
                "without a model.", intent, kind="error")
        assembled = await self.context.build(text, self.history[:-1])
        messages = assembled.messages
        if self._advisory:
            messages.insert(1, ChatMessage("system", "The user is asking for advice. Recommend what to do and why; "
                                                     "you may look things up, but do not change anything."))
        provs = list(assembled.provenance)
        profile = self._profile(text, tools=True)
        try:
            svc.router.select(profile)
            tool_schemas = self._chat_tools(profile)
        except NoModelAvailable:
            profile = self._profile(text, tools=False)   # no tool-capable model: talk, but can't act
            tool_schemas = []
        ctx = ToolContext(actor=Actor.user(svc.user), cwd=self._work_root(), dry_run=intent.dry_run, clock=svc.clock,
                          data_dir=str(svc.config.data_path))
        model_name = None
        notes: list[str] = []
        streamed_any = False
        acted = False             # did any tool that changes something actually succeed this turn?
        for _round in range(_MAX_TOOL_ROUNDS):
            try:
                response, note, streamed = await self._model_round(profile, messages, tool_schemas or None,
                                                                   separator=streamed_any)
            except ModelError as exc:
                return self._reply(f"The model call failed: {exc}. Nothing was changed.", intent, kind="error")
            streamed_any = streamed_any or streamed
            model_name = response.model
            if note and note not in notes:
                notes.append(note)
            calls = response.tool_calls
            if not calls:
                answer = personality.clean(strip_thinking(response.content)) or "Done."
                footnote = f"Note: {'; '.join(notes)}" if notes else ""
                if not acted and _CLAIM.search(answer):
                    # the model said it did something, but nothing that changes anything ran: say so
                    footnote = (footnote + "\n" if footnote else "") + \
                        "Nothing was actually changed: I didn't run any action for that."
                return Response(answer, intent.kind, provenance=provs, model=model_name, streamed=streamed_any,
                                footnote=footnote)
            messages.append(ChatMessage("assistant", response.content, tool_calls=calls))
            for call in calls:
                outcome = await self._execute_model_tool(call, ctx, text)
                if isinstance(outcome, Response):
                    return outcome
                result, prov = outcome
                tool = svc.registry.get(call.name)
                if tool is not None and tool.spec.level > 0 and result.get("ok") and \
                        result.get("status") != ExecStatus.DRY_RUN.value:
                    acted = True
                if prov is not None:
                    provs.append(prov)
                messages.append(ChatMessage("tool", json.dumps(result, default=str)[:6000], name=call.name,
                                            tool_call_id=call.id))
        return self._reply("I stopped after several tool steps without reaching an answer. Here's where things stand: "
                           + reports.activity(svc), intent, provenance=provs, model=model_name)

    async def _model_round(self, profile: TaskProfile, messages: list[ChatMessage],
                           tools: list[dict[str, Any]] | None, *, separator: bool = False) -> tuple[Any, str | None, bool]:
        """One model call. Streams answer text to the interface when it asked for tokens; otherwise (or if
        streaming fails before anything was shown) uses the router's fallback-capable call."""
        sink = self._token_sink
        if sink is None:
            routed = await self.svc.router.chat(profile, messages, tools=tools)
            return routed.response, routed.fallback_note, False
        cleaner = _StreamCleaner(sink, prefix=" " if separator else "")
        final = None
        try:
            async for chunk in self.svc.router.stream(profile, messages, tools=tools):
                if chunk.delta:
                    cleaner.feed(chunk.delta)
                if chunk.done:
                    final = chunk.response
        except ModelError:
            if cleaner.emitted:
                raise
            routed = await self.svc.router.chat(profile, messages, tools=tools)
            return routed.response, routed.fallback_note, False
        cleaner.close()
        if final is None:
            raise ModelError("the model stream ended without a final message")
        return final, None, cleaner.emitted

    def _clean_args(self, call: ToolCall) -> dict[str, Any]:
        """Small models often add stray arguments or send JSON as a string: keep only what the tool declares."""
        args: Any = call.arguments
        if isinstance(args, dict) and set(args) == {"_raw"} and isinstance(args["_raw"], str):
            try:
                args = json.loads(args["_raw"])
            except ValueError:
                args = {}
        if not isinstance(args, dict):
            return {}
        tool = self.svc.registry.get(call.name)
        if tool is None or tool.spec.parameters.get("additionalProperties"):
            return dict(args)
        known = tool.spec.parameters.get("properties", {})
        return {k: v for k, v in args.items() if k in known and v is not None}

    async def _or_chat(self, intent: Intent, deterministic: Response) -> Response:
        """The grammar matched but its target means nothing to the task system ("how's the weather?"):
        let the model handle it when one is available. A fallback conversation never starts a plan: the
        grammar already decided this wasn't a new goal."""
        if self.svc.router.available():
            reply = await self._chat(Intent(IntentKind.CHAT, intent.text, dry_run=intent.dry_run, source="fallback"))
            if reply.kind != "error":
                return reply
        return deterministic       # no model (or it just failed): the deterministic answer still stands

    async def _execute_model_tool(self, call: ToolCall, ctx: ToolContext,
                                  user_text: str) -> tuple[dict[str, Any], Provenance | None] | Response:
        svc = self.svc
        reason = OperationalReason("you asked: " + user_text[:120], "act on explicit user requests",
                                   f"ran {call.name}")
        call = ToolCall(call.name, self._clean_args(call), call.id)
        tool = svc.registry.get(call.name)
        if self._advisory and tool is not None and tool.spec.level > 0 and not ctx.dry_run:
            # advice never acts: anything beyond observing is only previewed
            from dataclasses import replace
            ctx = replace(ctx, dry_run=True)
        long_running = bool(tool and tool.spec.long_running)
        if long_running and not ctx.dry_run:
            # durable + interruptible: run it as a task and wait briefly
            what = (tool.preview(call.arguments) if tool else call.name).split(" (in ")[0]
            task = self._user_task(f"{call.name} for: {user_text[:80]}", title=_short_title(what),
                                   steps=[Step(tool.preview(call.arguments) if tool else call.name, call.name,
                                               call.arguments)], policy=TaskPolicy(on_step_failure="fail"),
                                   cwd=ctx.cwd)
            try:
                done = await svc.pool.wait_for(task.id, [S.COMPLETED, S.FAILED, S.WAITING, S.BLOCKED, S.CANCELLED],
                                               timeout=20.0)
            except TimeoutError:
                return {"status": "running_in_background", "task_id": task.id,
                        "message": "still running; the user will be notified"}, None
            self._inline_tasks.append(task.id)
            if done.status == S.WAITING:
                pending = [a for a in svc.approvals.pending() if a.task_id == task.id]
                summary = pending[0].summary if pending else call.name
                return Response(f"That needs your approval: {summary}. Proceed?", IntentKind.CHAT, kind="question",
                                task_id=task.id, approval_id=pending[0].id if pending else None)
            step = done.plan[0] if done.plan else None
            result = dict(step.result or {}) if step else {}
            result.setdefault("status", done.status.value)
            if done.status == S.BLOCKED:
                result["message"] = done.status_reason
            return result, Provenance(ProvenanceKind.TOOL_OUTPUT, call.name)
        execution = await svc.registry.execute(call.name, call.arguments, ctx, reason=reason)
        if execution.status == ExecStatus.NEEDS_APPROVAL:
            task = self._user_task(f"{call.name} for: {user_text[:80]}", title=_short_title(execution.preview),
                                   steps=[Step(execution.preview, call.name, execution.args)],
                                   policy=TaskPolicy(on_step_failure="fail"), cwd=ctx.cwd)
            try:
                await svc.pool.wait_for(task.id, [S.WAITING, S.COMPLETED, S.FAILED, S.BLOCKED], timeout=5.0)
            except TimeoutError:
                pass
            pending = [a for a in svc.approvals.pending() if a.task_id == task.id]
            return Response(f"That needs your approval: {execution.preview}. Proceed?", IntentKind.CHAT,
                            kind="question", task_id=task.id, approval_id=pending[0].id if pending else None)
        prov = execution.result.provenance if execution.result and execution.result.provenance else \
            Provenance(ProvenanceKind.TOOL_OUTPUT, call.name)
        return execution.for_model(), prov


# words that say a stop/cancel is about JARVIS's own work, not a program on the computer
_WORK_WORDS = {"task", "tasks", "plan", "plans", "job", "jobs", "delete", "deletes", "deletion", "deletions",
               "deleting", "cleanup", "clean", "backup", "backups", "research", "scan", "investigation", "work",
               "free", "disk", "space"}


_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/)(?:[^\s'\"]*[\\/])+([^\s'\"\\/]+)")


def _short_title(preview: str, limit: int = 60) -> str:
    """A task title from an action preview: file paths shortened to their names ("move vc_redist.x64.exe to
    JARVIS's trash"), so the title still says what it does when it's cut to length."""
    text = _PATH.sub(lambda m: m.group(1), preview)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _names_a_program(target: str) -> bool:
    """Whether a phrase names a program running now ("firefox", "the chrome process")."""
    words = [w for w in re.findall(r"[a-z0-9]+", target.lower()) if w not in ("the", "my", "process", "app",
                                                                            "program", "window")]
    if not words:
        return False
    try:
        import psutil
        names = {os.path.splitext((p.info.get("name") or "").lower())[0] for p in psutil.process_iter(["name"])}
    except Exception:
        return False
    return any(n and (n == words[0] or n.startswith(words[0])) for n in names)


# first-person claims of having changed something ("I've deleted…", "all delete processes have been cancelled")
_DONE_VERBS = (r"(?:cancel+ed|stopped|paused|deleted|removed|moved|killed|terminated|closed|installed|uninstalled|"
               r"freed|cleaned|cleared|restarted|disabled|enabled|renamed|trashed|emptied)")
_CLAIM = re.compile(rf"\b(?:i(?:'ve| have)?\s+(?:now\s+|just\s+|successfully\s+)?{_DONE_VERBS}|"
                    rf"(?:has|have)\s+(?:now\s+)?been\s+(?:successfully\s+)?{_DONE_VERBS}|"
                    rf"(?:is|are)\s+now\s+{_DONE_VERBS})\b", re.I)


def _memory_kind(content: str, has_project: bool) -> MemoryKind:
    lowered = content.lower()
    if re.search(r"\b(i prefer|i like|i don'?t like|i hate|i want you to|always|never|call me)\b", lowered):
        return MemoryKind.PREFERENCE
    if re.search(r"\b(when i (ask|say)|whenever|the way i|my workflow)\b", lowered):
        return MemoryKind.PROCEDURAL
    if has_project:
        return MemoryKind.PROJECT
    return MemoryKind.SEMANTIC


def _lower(title: str) -> str:
    if len(title) > 1 and title[:2].isupper():
        return title
    return title[:1].lower() + title[1:]


def _rebase(args: dict[str, Any], old_root: str | None, new_root: str | None) -> dict[str, Any]:
    if not old_root or not new_root or old_root == new_root:
        return args
    return {k: (v.replace(old_root, new_root) if isinstance(v, str) else v) for k, v in args.items()}


def _attrs_brief(attrs: dict[str, Any]) -> str:
    items = [f"{k} {v}" for k, v in attrs.items() if isinstance(v, (int, float, str, bool)) and k != "root"][:2]
    return ", ".join(items)


class _StreamCleaner:
    """Filters streamed model text before it reaches the interface: drops inline reasoning traces
    (<think>…</think>) and holds back the first few words so chatbot filler openers can be removed."""

    _HOLD = 40

    def __init__(self, sink: Callable[[str], None], prefix: str = "") -> None:
        self.sink = sink
        self.prefix = prefix
        self.buffer = ""
        self.started = False
        self.in_think = False
        self.emitted = False

    def feed(self, delta: str) -> None:
        text = delta
        out = ""
        while text:
            if self.in_think:
                end = text.lower().find("</think>")
                if end < 0:
                    return
                text = text[end + len("</think>"):]
                self.in_think = False
            else:
                start = text.lower().find("<think>")
                if start < 0:
                    out += text
                    break
                out += text[:start]
                text = text[start + len("<think>"):]
                self.in_think = True
        if not out:
            return
        if self.started:
            self._emit(out)
            return
        self.buffer += out
        if len(self.buffer) >= self._HOLD:
            self._flush_start()

    def _flush_start(self) -> None:
        self.started = True
        text = personality.strip_opener(self.buffer.lstrip())
        self.buffer = ""
        if text:
            self._emit(self.prefix + text)

    def _emit(self, text: str) -> None:
        self.emitted = True
        self.sink(text)

    def close(self) -> None:
        if not self.started and self.buffer.strip():
            self._flush_start()
