"""The conversation side of perception (Phase 4 §2-3, §20-22, §27, §37, §49).

Typing stays the way the user talks to JARVIS. This module notices when a turn is about something the user showed
JARVIS — an attached image or document, a file path dropped into the window, "this screenshot", "the previous one",
"compare these two" — and answers it through the perception layer, combining the user's words with what the image or
document contains, the active project and the current system state. It also handles screen awareness ("turn on
screen awareness", "look at my screen", "stop watching my screen"), capability questions ("what can you see?"),
forgetting ("forget that screenshot"), and follow-ups on what was found ("can you fix it?").

It only claims to have seen what it actually analysed: without a vision model it says so (and helps from the image's
text when an OCR engine could read it). Ordinary text requests never touch vision: if a turn has no attachments and
refers to nothing shown, this module steps aside and the orchestrator handles the turn exactly as before.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

from jarvis.core.intent import Intent, IntentKind
from jarvis.core.types import Provenance, ProvenanceKind
from jarvis.perception import references
from jarvis.perception.inputs import Attachment, InputKind, InputOrigin, Observation, find_paths, strip_paths
from jarvis.perception.screen import ScreenAccessDenied, ScreenMode

if TYPE_CHECKING:
    from jarvis.core.orchestrator import Orchestrator, Response

_I = re.I
_SCREEN_WATCH = re.compile(r"\b(watch|monitor|keep\s+an\s+eye\s+on)\s+(my|the)\s+screen\b|"
                           r"\bscreen\s+(awareness|watching)\s+(to\s+)?watch(ing)?\b", _I)
_SCREEN_ON = re.compile(r"\b(turn|switch)\s+on\s+(the\s+)?screen(\s+(awareness|access|capture))?\b|"
                        r"\benable\s+screen(\s+(awareness|access|capture))?\b|"
                        r"\byou\s+(can|may)\s+(look\s+at|see|use)\s+my\s+screen\b|\ballow\s+screen\s+(access|capture)\b",
                        _I)
_SCREEN_OFF = re.compile(r"\b(turn|switch)\s+off\s+(the\s+)?screen(\s+(awareness|access|capture|watching))?\b|"
                         r"\bdisable\s+screen(\s+(awareness|access|capture))?\b|"
                         r"\bstop\s+(watching|looking\s+at|monitoring|capturing)\s+(my|the)\s+screen\b|"
                         r"\bdon'?t\s+(look\s+at|watch)\s+my\s+screen\b", _I)
_SCREEN_STATUS = re.compile(r"\bscreen\s+(awareness|access|capture)\s*(status|on|off|enabled)?\s*\?|"
                            r"\bare\s+you\s+(watching|looking\s+at|seeing)\s+my\s+screen\b|"
                            r"\bcan\s+you\s+see\s+my\s+screen\b", _I)
_SCREEN_LOOK = re.compile(r"\b(look\s+at|check|read|scan)\s+(my|the)\s+screen\b|\bwhat(?:['’]?s|\s+is)\s+on\s+(my|the)\s+screen\b|"
                          r"\bwhat\s+am\s+i\s+looking\s+at\b|\bwhat\s+(app|application|window|program)\s+(is\s+open|"
                          r"am\s+i\s+(in|using))\b|\btake\s+a\s+(screenshot|screen\s+shot)\b|"
                          r"\bwhat(?:['’]?s|\s+is)\s+(wrong|happening)\s+on\s+(my|the)\s+screen\b", _I)
_CAPABILITIES = re.compile(r"\bwhat\s+can\s+you\s+see\b|\bwhat\s+inputs?\b|\bwhat\s+(input\s+)?devices\b|"
                           r"\bcan\s+you\s+see\s+(images|pictures|photos|screenshots)\b|"
                           r"\bdo\s+you\s+have\s+(a\s+)?(camera|webcam|microphone|mic|vision|eyes|ocr)\b|"
                           r"\bwhat\s+(can|do)\s+you\s+(perceive|sense)\b|\bwhat\s+are\s+your\s+(senses|inputs)\b|"
                           r"\b(can|do)\s+you\s+(hear|listen)\b|\bcan\s+you\s+read\s+(images|pdfs?|documents)\b", _I)
_FORGET = re.compile(r"\b(forget|delete|remove|discard|throw\s+away)\s+(that|this|the|those|these)\s+"
                     r"(last\s+)?(screen\s?shots?|images?|pictures?|photos?|documents?|files?|pdfs?)\b|"
                     r"\bforget\s+(everything|all)\s+(i'?ve|i\s+have)\s+(shown|sent|given)\s+you\b", _I)
_READ_TEXT = re.compile(r"\b(read|transcribe|extract|copy)\b.{0,30}\b(text|words|writing)\b|"
                        r"\bwhat\s+does\s+(it|this|that|the\s+[\w ]{1,30}?)\s+say\b|\bwhat\s+text\b|\bocr\b|"
                        r"\bread\s+(it|this|that)(\s+out)?\s*(to\s+me)?\s*[.!?]?$", _I)
_UI = re.compile(r"\b(what|which|list)\s+(the\s+)?(buttons|controls|elements|options|menus|tabs|fields|toggles)\b|"
                 r"\b(ui|interface|on-?screen)\s+elements\b", _I)
_DIAGRAM = re.compile(r"\b(diagram|schematic|circuit|flow\s?chart|architecture\s+(diagram|drawing)|block\s+diagram|"
                      r"wiring)\b|\bcomponents?\s+and\s+(their\s+)?connections\b|\bhow\s+(is|are)\s+.+\s+connected\b",
                      _I)
_REQUIREMENTS = re.compile(r"\brequirements?\b|\bmust[- ]haves?\b|\bwhat\s+(is|are)\s+required\b|\bspec(ification)?s\b"
                           r"\s+(say|list)", _I)
_REFERENCES = re.compile(r"\b(find|show|list|where\s+are)\b.{0,20}\b(references?|mentions?)\s+(to|of)\s+(?P<t1>.+)|"
                         r"\bwhere\s+does\s+(it|this|that|the\s+\w+)\s+(mention|talk\s+about|say\s+anything\s+about|"
                         r"discuss)\s+(?P<t2>.+)|\b(find|search\s+for)\s+(?P<t3>.+?)\s+in\s+(this|the|that|it)"
                         r"(\s+(document|file|pdf|report|log))?\b", _I)
_SUMMARY = re.compile(r"\bsummari[sz]e\b|\bsummary\b|\boverview\b|\btl;?dr\b|\bwhat\s+is\s+(this|it|that)\s+about\b|"
                      r"\bimportant\s+(sections?|parts?|points?)\b|\bkey\s+points\b|\bread\s+(this|it|that)\b", _I)
_OUTLINE = re.compile(r"\boutline\b|\bstructure\b|\btable\s+of\s+contents\b|\bwhat\s+sections\b", _I)
_FIX = re.compile(r"\b(fix|solve|sort\s+out|resolve|repair|deal\s+with|get\s+rid\s+of|clean\s+(it\s+)?up)\b", _I)
_HOW_FIX = re.compile(r"\bhow\s+(do|can|should|would)\s+i\s+(fix|solve|resolve|sort|get\s+past|deal\s+with)\b|"
                      r"\bwhat\s+should\s+i\s+do\b|\bwhere\s+do\s+i\s+click\b|\bwhat\s+do\s+i\s+(do|click|press)\b", _I)
_FOLLOW_FIX = re.compile(r"^(ok(ay)?,?\s+|yes,?\s+|so,?\s+|then,?\s+|right,?\s+)?(please\s+)?(can|could|would|will)?\s*"
                         r"(you\s+)?(please\s+)?(go\s+ahead\s+and\s+)?(fix|solve|sort|resolve|repair|deal\s+with)\s+"
                         r"(it|that|this|them|the\s+(problem|issue|error))(\s+for\s+me)?(\s+then)?(\s+please)?\s*[.!?]*$",
                         _I)
_FOLLOW_HOW = re.compile(r"^(and\s+|so\s+|ok(ay)?,?\s+)?(how\s+(do|can|should)\s+i\s+(fix|solve|resolve|sort)\s+"
                         r"(it|that|this|them)|what\s+should\s+i\s+do(\s+about\s+(it|that|this))?|"
                         r"what\s+(caused|causes|is\s+causing)\s+(it|that|this)|why(\s+is\s+(it|that|this)\s+happening)?)"
                         r"\s*[.!?]*$", _I)
_CONTROL = re.compile(r"^\s*(cancel|stop|pause|resume|continue|yes|no|ok|okay|sure|help|status)\b", _I)
_PATHLIKE = re.compile(r"(?:^|\s)(?:[A-Za-z]:[\\/]|~[\\/]|/[\w.\-]+/)")
_NOUNS = (r"(screen\s?shots?|screengrabs?|snips?|images?|pictures?|photos?|pics?|diagrams?|charts?|graphs?|"
          r"schematics?|flow\s?charts?|mock-?ups?|documents?|docs?|files?|pdfs?|reports?|specs?|logs?)")
_INPUT_DEIXIS = re.compile(r"\b(this|that|these|those|the|last|previous|earlier|first|second|third|other|attached|"
                           r"both)\s+(two\s+)?" + _NOUNS + r"\b", _I)
_VISUAL_DEIXIS = re.compile(r"\b(this|that|these|those|the|my|attached|last|previous)\s+(screen\s?shots?|images?|"
                            r"pictures?|photos?|pics?|snips?)\b", _I)
_ASKS = re.compile(r"\b(what|which|where|why|how|explain|describe|look|read|analy[sz]e|check|wrong|compare|fix|"
                   r"tell\s+me|show)\b", _I)
_POINTING = re.compile(r"\b(look\s+at\s+(this|that|these|it)|what(?:['’]?s|\s+is)\s+wrong\s+(here|with\s+(this|that|it))|"
                       r"what\s+does\s+(this|that|it)\s+(say|show|mean)|what\s+is\s+(this|that)\b|"
                       r"read\s+(this|that|it)\b|(this|that)\s+error\b|compare\s+(these|them|this|those)|"
                       r"the\s+(previous|other|earlier|first|second|last)\s+one\b|where\s+do\s+i\s+click|"
                       r"what\s+should\s+i\s+(click|press)|explain\s+(this|that)\b|summari[sz]e\s+(this|it|that)\b|"
                       r"what\s+changed\b|what(?:['’]?s|\s+is)\s+different\b|describe\s+(this|it|that)\b)", _I)
_TIME_REF = re.compile(r"\b(yesterday|last\s+week|the\s+other\s+day|a\s+few\s+days\s+ago|(last|previous)\s+"
                       r"(debugging\s+)?session)\b", _I)


class PerceptionDialogue:
    def __init__(self, orch: "Orchestrator") -> None:
        self.o = orch
        self.svc = orch.svc
        self.turn = 0
        self.last: dict[str, Any] | None = None     # the latest finding: observations, problem, question, turn

    @property
    def p(self) -> Any:
        return self.svc.perception

    def _reply(self, text: str, **kw: Any) -> "Response":
        from jarvis.core.orchestrator import Response
        kind = kw.pop("intent", IntentKind.PERCEIVE)
        return Response(text, kind, **kw)

    # -- entry --------------------------------------------------------------------------------------------------
    async def intake(self, text: str, attachments: list[Attachment] | None = None) -> "Response | None":
        """Handle this turn if it's about perceived input (or screen/capabilities); otherwise None."""
        self.turn += 1
        if self.p is None:
            if attachments:
                return self._reply("Perception is switched off in my configuration, so I can't take in attachments.",
                                   kind="error")
            return None
        attachments = list(attachments or [])
        dropped = find_paths(text)
        if dropped:
            attachments += [Attachment(path=path, origin=InputOrigin.USER_PATH) for path in dropped]
            text = strip_paths(text, dropped)
        taken, problems = [], []
        if attachments:
            project = self.svc.projects.active()
            taken, problems = self.p.ingest(attachments, session_id=self.o.session_id,
                                            project_id=project.id if project else None)
            if not taken:
                return self._reply(" ".join(problems), kind="error")
        stripped = text.strip()
        if not taken:
            screen = await self._screen(stripped)
            if screen is not None:
                return screen
            if _CAPABILITIES.search(stripped):
                return self._reply(await self.p.capabilities_text(), intent=IntentKind.INPUTS,
                                   data={"capabilities": [c.to_dict() for c in await self.p.capabilities()]})
            if _FORGET.search(stripped):
                return self._forget(stripped)
            follow = await self._follow_up(stripped)
            if follow is not None:
                return follow
            if _CONTROL.match(stripped) or not self._refers_to_input(stripped):
                return None
        resolution = self.p.resolve(stripped, self.o.session_id, taken)
        if resolution.question:
            self.o.pending_question = {"entity": "visual_clarify", "intent": Intent(IntentKind.PERCEIVE, stripped),
                                       "ids": [x.id for x in resolution.observations],
                                       "labels": [f"{x.handle} {x.name}" for x in resolution.observations]}
            return self._reply(resolution.question, kind="question")
        if resolution.missing:
            return self._reply(f"I don't have {resolution.missing}. " + self._how_to_share(), kind="answer")
        if not resolution.observations:
            if taken or _VISUAL_DEIXIS.search(stripped):
                return self._reply("I don't see an image from you. " + self._how_to_share())
            return None
        response = await self.answer(stripped, resolution.observations, compare=resolution.compare)
        if problems:
            response.footnote = "; ".join(filter(None, [response.footnote, *problems]))
        return response

    async def answer_clarification(self, text: str, pq: dict[str, Any]) -> "Response | None":
        """"The second one" / "screenshot 2" after "which one do you mean?"."""
        options = [self.p.store.get(i) for i in pq.get("ids", [])]
        options = [o for o in options if o is not None]
        lowered = text.lower()
        pick = None
        ordinals = ["first", "second", "third", "fourth"]
        for i, o in enumerate(options):
            words = [str(i + 1)] + ([ordinals[i]] if i < len(ordinals) else [])
            if o.name.lower() in lowered or o.handle in lowered or \
                    any(re.search(rf"\b{w}\b", lowered) for w in words):
                pick = o
                break
        if pick is None and re.search(r"\b(last|latest)\b", lowered) and options:
            pick = options[-1]
        if pick is None:
            return None
        original: Intent = pq["intent"]
        return await self.answer(original.text, [pick])

    # -- answering about observations -------------------------------------------------------------------------
    async def answer(self, text: str, observations: list[Observation], *, compare: bool = False) -> "Response":
        visual = [o for o in observations if o.kind.visual]
        documents = [o for o in observations if o.kind in (InputKind.DOCUMENT, InputKind.FILE)]
        question = text.strip() or ("Describe this image: what it shows and anything notable (errors, warnings, "
                                    "important text)." if visual else "")
        if compare and len(observations) >= 2:
            seen = await self.p.compare(observations[0], observations[1], question or "What changed?")
        elif documents and not visual:
            seen = await self._document(question, documents)
        elif _READ_TEXT.search(question) and not _HOW_FIX.search(question):
            seen = await self.p.read_text(visual[0])
        elif _UI.search(question):
            seen = await self.p.ui(visual[0], question)
        elif _DIAGRAM.search(question) and not _FIX.search(question):
            seen = await self.p.diagram(visual[0], question)
        else:
            seen = await self.p.understand(visual, question, context=self._context())
        response = self._render(seen)
        if seen.failed:
            return response
        self.last = {"observations": [o.id for o in seen.observations], "problem": seen.problem,
                     "question": question, "turn": self.turn, "finding": seen.finding, "text": seen.text}
        wants_action = bool(_FIX.search(question)) and not _HOW_FIX.search(question)
        if wants_action and seen.problem:
            acted = await self.act(seen.problem, seen.observations, question)
            if acted is not None:
                acted.text = f"{seen.text}\n\n{acted.text}"
                acted.provenance = seen.provenance + acted.provenance
                acted.footnote = "; ".join(filter(None, [response.footnote, acted.footnote]))
                return acted
        elif wants_action and not seen.problem:
            response.text += "\n\nI can't tell what would fix that from the image alone, so I haven't changed " \
                             "anything."
        elif seen.problem and self._actionable(seen.problem):
            response.text += "\n" + self._offer(seen.problem)
        return response

    async def _document(self, question: str, documents: list[Observation]) -> Any:
        doc = documents[0]
        if len(documents) >= 2 and re.search(r"\b(changed|difference|different|compare|diff)\b", question, _I):
            return await self.p.compare_documents(documents[0], documents[1])
        if _REQUIREMENTS.search(question):
            return await self.p.document(doc, question, mode="requirements")
        m = _REFERENCES.search(question)
        if m:
            topic = next((m.group(g) for g in ("t1", "t2", "t3") if m.group(g)), "").strip(" ?.!\"'")
            return await self.p.document(doc, question, mode="references", topic=topic)
        if _OUTLINE.search(question):
            return await self.p.document(doc, question, mode="outline")
        if not question or _SUMMARY.search(question):
            return await self.p.document(doc, question, mode="summary")
        return await self.p.document(doc, question, mode="answer")

    def _render(self, seen: Any) -> "Response":
        data = {"observations": [o.id for o in seen.observations], "basis": seen.basis,
                "handles": [o.handle for o in seen.observations], **{k: v for k, v in seen.data.items()
                                                                    if k not in ("pixels",)}}
        if seen.problem:
            data["problem"] = seen.problem
        return self._reply(seen.text, kind="error" if seen.failed else "answer", provenance=seen.provenance,
                           model=seen.model, footnote="; ".join(seen.notes), data=data)

    def _context(self) -> str:
        """Trusted context for combining with the image: project, active window, recent failures."""
        parts = []
        project = self.svc.projects.active()
        if project:
            parts.append(f"The user's active project is {project.name}" + (f" at {project.root}" if project.root else "")
                         + ".")
        state = self.svc.state
        app = state.value("screen.active_app")
        if app:
            parts.append(f"The last application seen on screen was {app}.")
        failed = [t for t in self.svc.tasks.list_tasks(limit=5, order="recent") if t.status.value == "failed"]
        if failed:
            parts.append("Recently failed in JARVIS: " + "; ".join(f"{t.title} ({t.status_reason[:80]})"
                                                                   for t in failed[:2]) + ".")
        cpu, mem = state.value("resources.cpu_percent"), state.value("resources.memory_percent")
        if cpu is not None and mem is not None:
            parts.append(f"Right now CPU is at {cpu}% and memory at {mem}%.")
        return " ".join(parts)

    def _refers_to_input(self, text: str) -> bool:
        """Whether a message without attachments is about something the user shared (conservative: ordinary
        requests like "back up my documents" or anything naming a path are never taken over)."""
        if _PATHLIKE.search(text):
            return False
        now = self.svc.clock.now()
        fresh = [o for o in self.p.recent(self.o.session_id, limit=10) if now - o.created_at < 2 * 3600]
        if _TIME_REF.search(text) and references.wanted_kinds(text):
            return True                                    # "the diagram we looked at yesterday": searched by time
        if not fresh:
            # nothing shared: only an explicit "this screenshot / the image" gets an honest "I don't see one"
            return bool(_VISUAL_DEIXIS.search(text) and _ASKS.search(text))
        kinds = {o.kind for o in fresh}
        m = _INPUT_DEIXIS.search(text)
        if m:
            wanted = references.wanted_kinds(m.group(0)) or ()
            if not wanted or kinds & set(wanted):
                return True
        pointing = _POINTING.search(text)
        if pointing and re.match(r"what('?s|\s+is)?\s*(changed|different)", pointing.group(0), _I) and \
                len([o for o in fresh if o.kind.visual or o.kind == InputKind.DOCUMENT]) < 2:
            return False          # "what changed?" about the system, unless two inputs were just shared
        return bool(pointing or references._NUMBERED.search(text) or
                    references._ORDINAL_RE.search(text) and re.search(r"\bone\b", text, _I))

    @staticmethod
    def _how_to_share() -> str:
        return ("To show me one, drag the file into this window (or type its path) and press Enter, use /attach "
                "<file>, or copy a screenshot and type /paste.")

    # -- follow-ups on what was found ------------------------------------------------------------------------
    async def _follow_up(self, text: str) -> "Response | None":
        last = self.last
        if last is None or self.turn - last["turn"] > 3:
            return None
        observations = [o for o in (self.p.store.get(i) for i in last["observations"]) if o is not None]
        if not observations:
            return None
        if _FOLLOW_FIX.match(text):
            problem = last.get("problem")
            if problem and self._actionable(problem):
                acted = await self.act(problem, observations, text)
                if acted is not None:
                    return acted
            guidance = await self.answer(f"How do I fix this? (Earlier finding: {last['finding']})", observations)
            guidance.text = ("I can't fix this kind of problem myself, but here's what to do:\n" + guidance.text
                             if not (problem and self._actionable(problem)) else guidance.text)
            return guidance
        if _FOLLOW_HOW.match(text):
            return await self.answer(f"{text} (Earlier finding: {last['finding']})", observations)
        return None

    @staticmethod
    def _actionable(problem: dict[str, Any]) -> bool:
        return problem.get("kind") in ("disk_full", "performance", "missing_module", "code_error")

    @staticmethod
    def _offer(problem: dict[str, Any]) -> str:
        kind = problem.get("kind")
        if kind == "disk_full":
            return "Say 'fix it' and I'll look for space to free up, asking before I remove anything."
        if kind == "performance":
            return "Say 'fix it' and I'll find what's using the computer's resources, asking before I stop anything."
        if kind == "missing_module":
            return f"Say 'fix it' and I'll install {problem.get('module') or 'the missing package'} (asking first) " \
                   "and check it imports."
        if kind == "code_error":
            return "Say 'fix it' and I'll investigate it in your project: find the code, run the tests and explain " \
                   "the cause."
        return ""

    async def act(self, problem: dict[str, Any], observations: list[Observation], text: str) -> "Response | None":
        """Turn a problem seen in an image into a Phase 3 plan: the observation is the plan's first step."""
        plans = self.o.plans
        intel = self.svc.intelligence
        if intel is None or plans is None:
            return None
        from jarvis.intelligence.visual import goal_for_problem
        goal = goal_for_problem(intel, problem, observations, text, screen_on=self.p.screen.mode != ScreenMode.OFF)
        if goal is None:
            return None
        return await plans.goal(Intent(IntentKind.CHAT, text), goal)

    # -- screen ----------------------------------------------------------------------------------------------------
    async def _screen(self, text: str) -> "Response | None":
        screen = self.p.screen
        if _SCREEN_OFF.search(text):
            ok, message = screen.set_mode(ScreenMode.OFF, actor_kind="user", actor_id=self.svc.user)
            return self._reply(message, intent=IntentKind.SCREEN, kind="action")
        if _SCREEN_WATCH.search(text) or _SCREEN_ON.search(text):
            mode = ScreenMode.WATCHING if _SCREEN_WATCH.search(text) else ScreenMode.ON_REQUEST
            ok, message = screen.set_mode(mode, actor_kind="user", actor_id=self.svc.user)
            available, detail = screen.backend.capture_available()
            if ok and not available:
                message += f" (I can't actually capture the screen here yet: {detail}.)"
            elif ok and mode == ScreenMode.ON_REQUEST:
                message += " Say 'watch my screen' if you want me to keep an eye on it."
            return self._reply(message, intent=IntentKind.SCREEN, kind="action")
        if _SCREEN_STATUS.search(text) and not _SCREEN_LOOK.search(text):
            return self._reply(screen.describe_mode(), intent=IntentKind.SCREEN)
        if _SCREEN_LOOK.search(text):
            return await self._look(text)
        return None

    async def _look(self, text: str) -> "Response":
        screen = self.p.screen
        wants_detail = bool(re.search(r"\b(what(?:['’]?s|\s+is)\s+wrong|why|how|explain|help|what\s+should|error|problem)\b",
                                      text, _I))
        try:
            state, obs = await screen.look(reason=text[:120], by=f"user:{self.svc.user}",
                                           session_id=self.o.session_id, ui=bool(_UI.search(text)))
        except ScreenAccessDenied as exc:
            return self._reply(str(exc), intent=IntentKind.SCREEN)
        when = time.strftime("%H:%M:%S", time.localtime(state.ts))
        text_out = state.describe(when)
        provenance = [Provenance(ProvenanceKind.SCREEN_CAPTURE, state.id, when)]
        footnote = "; ".join(state.notes)
        model = None
        if obs is not None and (wants_detail or not state.text_excerpt):
            seen = await self.p.understand([obs], text if wants_detail else "What is on this screen? Name the "
                                                                                  "application and anything notable.",
                                           context=self._context())
            if not seen.failed:
                text_out = f"{text_out}\n\n{seen.text}"
                provenance += seen.provenance[1:]
                model = seen.model
                self.last = {"observations": [obs.id], "problem": seen.problem, "question": text,
                             "turn": self.turn, "finding": seen.finding, "text": seen.text}
            else:
                ok, why = self.p.vision.status()
                limit = (f"I could see which window is active, but not what's in it: {why}. 'What can you see?' "
                         "shows what to install") if not ok and not state.text_excerpt else seen.text
                footnote = "; ".join(filter(None, [footnote, limit]))
        return self._reply(text_out, intent=IntentKind.SCREEN, provenance=provenance, model=model, footnote=footnote,
                           data={"screen": state.to_dict(), "observations": [obs.id] if obs else []})

    # -- forgetting ------------------------------------------------------------------------------------------------
    def _forget(self, text: str) -> "Response":
        recent = self.p.recent(self.o.session_id)
        if re.search(r"\b(everything|all)\b", text, _I):
            gone = [self.p.forget(o.id) for o in recent]
            n = len([g for g in gone if g])
            self.last = None
            return self._reply(f"Done: I've deleted the {n} input(s) you shared in this conversation and what I "
                               "learned from them." if n else "You haven't shared anything in this conversation.",
                               kind="action")
        resolution = references.resolve(text, recent, now=self.svc.clock.now())
        if not resolution.observations:
            return self._reply("There's nothing like that to forget.")
        obs = resolution.observations[0]
        self.p.forget(obs.id)
        if self.last and obs.id in self.last["observations"]:
            self.last = None
        return self._reply(f"Done: I've deleted the {obs.handle} ({obs.name}) and what I learned from it.",
                           kind="action")
