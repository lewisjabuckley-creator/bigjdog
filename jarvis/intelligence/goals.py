"""Goal understanding: what the user wants, as structured data (Phase 3 §1-2, §28, §41-43, §52).

A request becomes a :class:`Goal` before anything is planned: its objective, the outcome that means success,
constraints ("don't close Chrome"), deadline, priority, execution mode (do it, dry run, simulate, predict,
advise), complexity, the parts of a compound request and how they depend on each other, and whatever is
ambiguous about it.

Everything here is deterministic. A language model is never needed to understand structure, and never
trusted to decide what the user allowed. Ambiguity is classified by what a wrong guess would cost:

* harmless — any reasonable reading is fine; proceed.
* recoverable — proceed on a stated assumption; a wrong guess can be undone.
* consequential — a wrong guess changes or loses something; ask one short question.
* dangerous — a wrong guess could destroy data or stop the system; ask, and require an explicit target.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from enum import IntEnum, StrEnum
from typing import Any

from jarvis.clock import Clock, SystemClock, to_local
from jarvis.core.types import Priority, new_id


class ExecutionMode(StrEnum):
    EXECUTE = "execute"        # do it
    DRY_RUN = "dry_run"        # observe for real, change nothing, show the expected changes
    SIMULATE = "simulate"      # predict the effect of hypothetical actions from observed state; run nothing
    PREDICT = "predict"        # estimate (duration, outcome) from history; run nothing
    ADVISE = "advise"          # gather evidence and recommend; take no consequential action

    @property
    def changes_nothing(self) -> bool:
        return self != ExecutionMode.EXECUTE


class PlanPriority(StrEnum):
    EMERGENCY = "emergency"
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"

    @property
    def rank(self) -> int:
        return list(PlanPriority).index(self)

    def task_priority(self, interactive: bool = True) -> Priority:
        """The task-level priority used for scheduling and resource admission."""
        return {PlanPriority.EMERGENCY: Priority.P0, PlanPriority.CRITICAL: Priority.P0,
                PlanPriority.HIGH: Priority.P1, PlanPriority.NORMAL: Priority.P1 if interactive else Priority.P2,
                PlanPriority.LOW: Priority.P3, PlanPriority.BACKGROUND: Priority.P4}[self]


class Complexity(IntEnum):
    SIMPLE = 0          # one direct answer or action: existing paths, no planning
    MODERATE = 1        # a short linear plan
    COMPLEX = 2         # a structured plan: investigate, decide, act, verify
    VERY_COMPLEX = 3    # a structured plan that also benefits from agents or milestones

    @property
    def label(self) -> str:
        return self.name.lower().replace("_", " ")


class AmbiguityClass(StrEnum):
    HARMLESS = "harmless"
    RECOVERABLE = "recoverable"
    CONSEQUENTIAL = "consequential"
    DANGEROUS = "dangerous"


@dataclass
class Ambiguity:
    klass: AmbiguityClass
    missing: str                          # what is unclear, e.g. "which files"
    question: str = ""                    # one concise question
    assumption: str = ""                  # what JARVIS assumes when it proceeds anyway
    options: list[str] = field(default_factory=list)

    @property
    def must_ask(self) -> bool:
        return self.klass in (AmbiguityClass.CONSEQUENTIAL, AmbiguityClass.DANGEROUS) or not self.assumption

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "klass": self.klass.value}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Ambiguity | None":
        if not d:
            return None
        return cls(AmbiguityClass(d["klass"]), d.get("missing", ""), d.get("question", ""), d.get("assumption", ""),
                   list(d.get("options", [])))


# -- constraints ------------------------------------------------------------------------------------------------

_DELETE_COMMAND = re.compile(r"\b(rm|rmdir|del|erase|rd|remove-item|shred|unlink)\b|\bclear-recyclebin\b",
                             re.IGNORECASE)
_RESTART_COMMAND = re.compile(r"\b(shutdown|reboot|restart-computer|restart|systemctl\s+(restart|reboot))\b",
                              re.IGNORECASE)
_KILL_COMMAND = re.compile(r"\b(kill|pkill|killall|taskkill|stop-process)\b", re.IGNORECASE)


@dataclass
class Constraint:
    """A limit the user set. Plans are validated against every constraint before and during execution."""

    kind: str       # protect_process | protect_path | only_path | no_delete | read_only | no_network |
                    # local_models | no_restart
    value: str = ""
    text: str = ""  # the user's words

    def describe(self) -> str:
        return {
            "protect_process": f"leave {self.value} running",
            "protect_path": f"don't touch {self.value}",
            "only_path": f"only work in {self.value}",
            "no_delete": "don't delete anything",
            "read_only": "don't change anything",
            "no_network": "don't use the network",
            "local_models": "use local models only",
            "no_restart": "don't restart anything",
        }.get(self.kind, self.text or self.kind)

    def forbids(self, tool: str, args: dict[str, Any], meta: dict[str, Any] | None = None, *,
                level: int = 0, requires_network: bool = False) -> str | None:
        """Why this constraint rules out the action, or None. ``meta`` carries what the action targets
        (e.g. the process name behind a PID) and ``level`` its assessed permission level."""
        meta = meta or {}
        command = str(args.get("command") or "")
        target_names = " ".join(str(v) for v in (meta.get("target"), meta.get("process"), args.get("name")) if v)
        value = self.value.lower()
        if self.kind == "protect_process":
            if tool == "process_stop" and value and value in target_names.lower():
                return f"you asked me to {self.describe()}"
            if tool == "shell_execute" and _KILL_COMMAND.search(command) and value and value in command.lower():
                return f"you asked me to {self.describe()}"
            return None
        if self.kind == "no_delete":
            if tool == "file_delete" or (tool == "shell_execute" and _DELETE_COMMAND.search(command)):
                return "you asked me not to delete anything"
            return None
        if self.kind == "no_restart":
            if tool == "shell_execute" and _RESTART_COMMAND.search(command):
                return "you asked me not to restart anything"
            return None
        if self.kind == "read_only":
            return "you asked me not to change anything" if level >= 3 else None
        if self.kind == "no_network":
            return "you asked me not to use the network" if requires_network else None
        if self.kind in ("protect_path", "only_path"):
            paths = [str(args[k]) for k in ("path", "source", "destination", "cwd") if args.get(k)]
            root = os.path.realpath(os.path.expanduser(self.value))
            for p in paths:
                real = os.path.realpath(os.path.expanduser(p))
                inside = real == root or real.startswith(root.rstrip(os.sep) + os.sep)
                if self.kind == "protect_path" and inside and level >= 3:
                    return f"you asked me not to touch {self.value}"
                if self.kind == "only_path" and not inside and level >= 3:
                    return f"you asked me to only work in {self.value}"
            return None
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Constraint":
        return cls(d["kind"], d.get("value", ""), d.get("text", ""))


# -- goals ---------------------------------------------------------------------------------------------------------

@dataclass
class SubGoal:
    """One part of a compound request ("run the tests, then build it")."""

    text: str
    after: list[int] = field(default_factory=list)   # indexes of earlier parts this one waits for
    condition: str | None = None                     # "success" | "failure" of the part it waits for

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Goal:
    text: str                                   # the user's own words
    objective: str                              # normalised objective
    kind: str = "generic"                       # a playbook id, "compound" or "generic"
    outcome: str = ""                           # the end state that means success
    target: str | None = None                   # what it is about (a folder, an application...)
    constraints: list[Constraint] = field(default_factory=list)
    deadline: float | None = None
    priority: PlanPriority = PlanPriority.NORMAL
    mode: ExecutionMode = ExecutionMode.EXECUTE
    complexity: Complexity = Complexity.SIMPLE
    ambiguity: Ambiguity | None = None
    success_criteria: list[str] = field(default_factory=list)
    failure_conditions: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    permissions: str = "read-only"              # what the goal is expected to need: read-only | changes | unknown
    context: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    subgoals: list[SubGoal] = field(default_factory=list)
    wants_fix: bool = False                     # the user asked for the problem to be solved, not only explained
    id: str = field(default_factory=lambda: new_id("goal"))
    created_at: float = 0.0

    @property
    def needs_planning(self) -> bool:
        return self.complexity >= Complexity.MODERATE

    def constraint(self, kind: str) -> Constraint | None:
        return next((c for c in self.constraints if c.kind == kind), None)

    def forbids(self, tool: str, args: dict[str, Any], meta: dict[str, Any] | None = None, *, level: int = 0,
                requires_network: bool = False) -> str | None:
        for constraint in self.constraints:
            why = constraint.forbids(tool, args, meta, level=level, requires_network=requires_network)
            if why:
                return why
        return None

    def summary(self) -> str:
        parts = [self.objective]
        if self.constraints:
            parts.append("constraints: " + "; ".join(c.describe() for c in self.constraints))
        if self.deadline:
            parts.append(f"deadline {to_local(self.deadline).strftime('%a %H:%M')}")
        if self.priority != PlanPriority.NORMAL:
            parts.append(f"{self.priority.value} priority")
        if self.mode != ExecutionMode.EXECUTE:
            parts.append(self.mode.value.replace("_", " "))
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "text": self.text, "objective": self.objective, "kind": self.kind, "outcome": self.outcome,
            "target": self.target, "constraints": [c.to_dict() for c in self.constraints], "deadline": self.deadline,
            "priority": self.priority.value, "mode": self.mode.value, "complexity": int(self.complexity),
            "ambiguity": self.ambiguity.to_dict() if self.ambiguity else None,
            "success_criteria": self.success_criteria, "failure_conditions": self.failure_conditions,
            "resources": self.resources, "permissions": self.permissions, "context": self.context,
            "dependencies": self.dependencies, "subgoals": [s.to_dict() for s in self.subgoals],
            "wants_fix": self.wants_fix, "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Goal":
        return cls(
            text=d["text"], objective=d.get("objective", d["text"]), kind=d.get("kind", "generic"),
            outcome=d.get("outcome", ""), target=d.get("target"),
            constraints=[Constraint.from_dict(c) for c in d.get("constraints", [])], deadline=d.get("deadline"),
            priority=PlanPriority(d.get("priority", "normal")), mode=ExecutionMode(d.get("mode", "execute")),
            complexity=Complexity(int(d.get("complexity", 0))), ambiguity=Ambiguity.from_dict(d.get("ambiguity")),
            success_criteria=list(d.get("success_criteria", [])),
            failure_conditions=list(d.get("failure_conditions", [])), resources=list(d.get("resources", [])),
            permissions=d.get("permissions", "read-only"), context=dict(d.get("context", {})),
            dependencies=list(d.get("dependencies", [])),
            subgoals=[SubGoal(s["text"], list(s.get("after", [])), s.get("condition")) for s in d.get("subgoals", [])],
            wants_fix=bool(d.get("wants_fix", False)), id=d.get("id") or new_id("goal"),
            created_at=d.get("created_at", 0.0))


# -- parsing -------------------------------------------------------------------------------------------------------

_LEAD = re.compile(r"^(hey |ok |okay )?(jarvis[,:!]?\s+)?(please\s+|could you\s+|can you\s+|would you\s+)?",
                   re.IGNORECASE)

# goal kinds that have a deterministic playbook, recognised from the whole request
_KINDS: list[tuple[str, re.Pattern[str]]] = [
    ("performance", re.compile(
        r"\b((computer|pc|laptop|machine|system|windows|it|everything)\s+(is|feels|seems|got|getting|running|'s|"
        r"being|keeps)\s+(\w+\s+)?(slow|sluggish|laggy|lagging|unresponsive|freezing|hanging)|"
        r"(slow|sluggish|laggy)\s+(computer|pc|laptop|machine|system)|"
        r"speed\s+(up|it up)\s*(my|the|this)?\s*(computer|pc|laptop|machine|system)?|"
        r"(high|100%?|maxed)\s+(cpu|memory|ram)|(cpu|memory|ram)\s+(is\s+)?(maxed|pegged|at 100)|"
        r"why\s+is\s+(my|the|this)\s+(computer|pc|laptop|machine|system)\s+(so\s+)?slow|"
        r"what'?s\s+(slowing|using up)\s+(down\s+)?(my|the)\s+(computer|pc|cpu|memory|ram|machine))", re.I)),
    ("disk_cleanup", re.compile(
        r"\b(free\s+up\s+(some\s+)?(disk\s+)?space|disk\s+(is\s+)?(almost\s+|nearly\s+)?full|running\s+out\s+of\s+"
        r"(disk\s+)?space|clean\s*(up)?\s+(my\s+|the\s+)?(disk|drive|temp|downloads|junk)|low\s+on\s+(disk\s+)?"
        r"space|what'?s\s+(taking|using)\s+(up\s+)?(all\s+)?(the\s+|my\s+)?(disk\s+)?space)", re.I)),
    ("backup", re.compile(r"\b(back\s*up|backup|make\s+a\s+copy\s+of|archive)\b", re.I)),
    ("research", re.compile(
        r"\b(research|investigate\s+what|look\s+into|find\s+(out\s+)?(everything|all|what)\s+(i|we|my|our)\s+"
        r"(have|know|wrote|notes?)|summari[sz]e\s+(what|everything)\s+(my|our|the)\s+(notes|docs|documents|files)|"
        r"what\s+do\s+(my|our|the)\s+(notes|docs|documents|files)\s+say\s+about|compile\s+(a\s+)?(report|summary)\s+"
        r"(on|about))\b", re.I)),
]

_FIX = re.compile(r"\b(fix|solve|resolve|repair|sort\s+(it|this|that)\s+out|speed\s+(it\s+)?up|make\s+it\s+faster|"
                  r"clean\s*(it\s+)?up|free\s+up|optimi[sz]e|get\s+rid\s+of|deal\s+with)\b", re.I)
_ADVISE = re.compile(r"^(so,?\s+)?(what\s+should\s+i\s+do|what\s+do\s+you\s+(recommend|suggest|advise)|"
                     r"what\s+would\s+you\s+(do|recommend|suggest)|should\s+i\b|any\s+(advice|suggestions|"
                     r"recommendations)|advise\s+me|what'?s\s+your\s+(advice|recommendation)|how\s+should\s+i\b)", re.I)
_SIMULATE = re.compile(r"^(simulate\b|what\s+(would|will)\s+happen\s+if\b|what\s+if\s+(i|you|we)\b|"
                       r"what\s+would\s+(it|that)\s+(do|change)\s+if\b)", re.I)
_PREDICT = re.compile(r"^(how\s+long\s+(will|would|does|should)\b|predict\b|estimate\s+(how\s+long|when)\b|"
                      r"when\s+will\s+.+\s+(finish|be\s+done)\b)", re.I)
_DRY_RUN = re.compile(r"^dry[- ]run[:,]?\s*", re.I)

_CONSTRAINT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:don'?t|do\s+not|never|without)\s+(?:closing|close|killing|kill|stopping|stop|quitting|quit)\s+"
                r"(?:my\s+|the\s+)?(?P<v>[\w.+ -]+?)(?=$|[,.;!]|\s+(?:and|but|or|then)\b)", re.I), "protect_process"),
    (re.compile(r"\b(?:leave|keep)\s+(?:my\s+|the\s+)?(?P<v>[\w.+ -]+?)\s+(?:alone|running|open)\b", re.I),
     "protect_process"),
    (re.compile(r"\b(?:don'?t|do\s+not|never)\s+touch\s+(?P<v>[~/.\\]?[\w:./\\ -]+?)(?=$|[,;!]|\s+(?:and|but)\b)",
                re.I), "protect_path"),
    (re.compile(r"\b(?:don'?t|do\s+not|never|without)\s+delet(?:e|ing)\b|\bnothing\s+(?:gets\s+)?deleted\b|"
                r"\bno\s+deleting\b", re.I), "no_delete"),
    (re.compile(r"\b(?:don'?t|do\s+not)\s+change\s+anything\b|\bwithout\s+changing\s+anything\b|\bread[- ]only\b|"
                r"\b(?:just|only)\s+(?:look|check|tell\s+me|report|investigate|diagnose)\b", re.I), "read_only"),
    (re.compile(r"\bwithout\s+(?:the\s+)?(?:internet|network)\b|\bno\s+(?:internet|network)\b|\boffline\s+only\b",
                re.I), "no_network"),
    (re.compile(r"\b(?:only|just)\s+(?:use\s+)?local\s+models?\b|\bkeep\s+it\s+local\b", re.I), "local_models"),
    (re.compile(r"\b(?:don'?t|do\s+not|without)\s+(?:restart(?:ing)?|reboot(?:ing)?)\b|\bno\s+reboots?\b", re.I),
     "no_restart"),
    (re.compile(r"\b(?:only|just)\s+(?:in|inside|within)\s+(?:the\s+|my\s+)?(?P<v>[~/.\\]?[\w:./\\ -]+?)"
                r"(?:\s+folder)?(?=$|[,;!]|\s+(?:and|but)\b)", re.I), "only_path"),
]

_PRIORITY_RULES: list[tuple[re.Pattern[str], PlanPriority]] = [
    (re.compile(r"\bemergency\b", re.I), PlanPriority.EMERGENCY),
    (re.compile(r"\bcritical\b", re.I), PlanPriority.CRITICAL),
    (re.compile(r"\b(urgent(ly)?|asap|right\s+now|immediately|high\s+priority)\b", re.I), PlanPriority.HIGH),
    (re.compile(r"\b(in\s+the\s+background|overnight|background\s+task)\b", re.I), PlanPriority.BACKGROUND),
    (re.compile(r"\b(when\s+you\s+(get\s+a\s+chance|have\s+(some\s+)?time)|no\s+rush|whenever|low\s+priority)\b",
                re.I), PlanPriority.LOW),
]

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# clause separators: sequential ("then"), conditional ("if it passes"), parallel ("and also", ";")
_THEN = re.compile(r"\s*(?:,\s*)?\b(?:and\s+then|then|after\s+that|afterwards|once\s+(?:that'?s|it'?s|they'?re)\s+"
                   r"(?:done|finished)|when\s+(?:that'?s|it'?s|they'?re)\s+(?:done|finished)|followed\s+by)\b[,\s]*",
                   re.I)
_IF = re.compile(r"\s*(?:,\s*)?\b(?:and\s+)?if\s+(?:it|they|that|the\s+\w+|those)\s+(?P<cond>pass(?:es)?|succeeds?|"
                 r"works?|is\s+ok|are\s+ok|goes\s+well|fails?|doesn'?t\s+work|breaks?)\b[,\s]*(?:then\s+)?", re.I)
_OTHERWISE = re.compile(r"\s*(?:,\s*)?\b(?:otherwise|if\s+not|or\s+else)\b[,\s]*", re.I)
_AND = re.compile(r"\s*(?:;|,\s*and\s+also|\band\s+also\b|,\s+and\s+|\s+and\s+(?=(?:run|build|check|back|clean|"
                  r"free|analy[sz]e|review|update|write|create|make|test|research|summari[sz]e|tell|let|notify|"
                  r"send|open|stop|restart|install|delete|remove|find|look|scan|measure|show|list|compile|"
                  r"document|prepare|email|ping|monitor|watch)\b))\s*", re.I)
_BUT = re.compile(r"\s*,?\s*\bbut\b\s*", re.I)

# a clause that is only a follow-up of the playbook ("and fix it", "tell me what you find")
_COVERED = re.compile(r"^(?:and\s+)?(?:then\s+)?(?:please\s+)?(?:fix|solve|resolve|repair|sort\s+out|deal\s+with|"
                      r"speed\s+up|clean\s+up|free\s+up|get\s+rid\s+of)?\s*(?:it|that|this|them|the\s+(?:problem|"
                      r"issue|cause)|whatever\s+(?:it\s+is|you\s+find|is\s+causing\s+it))?\s*$|"
                      r"^(?:and\s+)?(?:then\s+)?(?:find|figure|work)\s+out\s+why.*$|"
                      r"^(?:and\s+)?(?:then\s+)?(?:tell|let)\s+me(?:\s+know)?(?:\s+what.*|\s+why.*|\s+how.*)?$|"
                      r"^(?:and\s+)?(?:then\s+)?(?:report|explain|summari[sz]e)(?:\s+.*)?$|"
                      r"^(?:and\s+)?(?:then\s+)?make\s+it\s+(?:faster|quicker|snappier)$", re.I)

_VAGUE_OBJECT = re.compile(r"\b(delete|remove|wipe|erase|uninstall|kill|stop|close|format|purge|empty)\s+"
                           r"(?P<obj>it|them|that|those|this|everything|all(\s+of\s+it)?|stuff|things|the\s+old\s+"
                           r"(files|stuff|ones)|old\s+(files|stuff)|the\s+files|files|junk|the\s+rest)\b", re.I)
_DANGEROUS_VERBS = re.compile(r"\b(format|wipe|erase\s+everything|delete\s+everything|remove\s+everything|"
                              r"purge|factory\s+reset)\b", re.I)
_ACTION_VERBS = re.compile(r"\b(fix|clean|free|speed|optimi[sz]e|repair|resolve|back\s*up|backup|prepare|set\s+up|"
                           r"organi[sz]e|update|install|delete|remove|stop|kill|close|restart|build|deploy|write|"
                           r"create|move|copy|rename|run)\b", re.I)


class GoalParser:
    """Turns a request into a :class:`Goal`. Deterministic; no model involved."""

    def __init__(self, clock: Clock | None = None) -> None:
        self.clock = clock or SystemClock()

    # -- entry point ------------------------------------------------------------------------------------------
    def parse(self, text: str, *, dry_run: bool = False, context: dict[str, Any] | None = None) -> Goal:
        raw = text.strip()
        body = _LEAD.sub("", raw, count=1).strip()
        mode = ExecutionMode.EXECUTE
        if dry_run or _DRY_RUN.match(body):
            mode = ExecutionMode.DRY_RUN
            body = _DRY_RUN.sub("", body, count=1).strip()
        elif _SIMULATE.match(body):
            mode = ExecutionMode.SIMULATE
        elif _PREDICT.match(body):
            mode = ExecutionMode.PREDICT
        elif _ADVISE.match(body):
            mode = ExecutionMode.ADVISE
        constraints = self.constraints(body)
        if any(c.kind == "read_only" for c in constraints) and mode == ExecutionMode.EXECUTE:
            mode = ExecutionMode.ADVISE
        body_wo_constraints = self._strip_constraint_clauses(body)
        kind = self.kind(body)
        goal = Goal(text=raw, objective=_sentence(body_wo_constraints or body), kind=kind, mode=mode,
                    constraints=constraints, priority=self.priority(body), deadline=self.deadline(body),
                    context=dict(context or {}), created_at=self.clock.now())
        goal.wants_fix = bool(_FIX.search(body)) and mode == ExecutionMode.EXECUTE
        goal.target = self.target(body, kind)
        clauses = self.clauses(body_wo_constraints or body)
        substantive = [c for c in clauses if not (kind != "generic" and _COVERED.match(c.text.strip(" .!?")))]
        if kind != "generic" and len(substantive) <= 1:
            goal.subgoals = []
        elif len(clauses) >= 2:
            goal.kind = "compound"
            goal.subgoals = clauses
        self._describe(goal)
        goal.ambiguity = self.ambiguity(body, goal)
        goal.complexity = self.complexity(goal, body)
        return goal

    # -- pieces -----------------------------------------------------------------------------------------------
    @staticmethod
    def kind(text: str) -> str:
        for name, pattern in _KINDS:
            if pattern.search(text):
                return name
        return "generic"

    @staticmethod
    def constraints(text: str) -> list[Constraint]:
        out: list[Constraint] = []
        for pattern, kind in _CONSTRAINT_RULES:
            for m in pattern.finditer(text):
                value = (m.groupdict().get("v") or "").strip(" .,'\"")
                value = re.sub(r"^(my|the)\s+", "", value, flags=re.I)
                if kind == "protect_process" and (not value or value.lower() in ("anything", "it", "that", "this",
                                                                               "everything", "something")):
                    continue
                if kind in ("protect_path", "only_path") and not value:
                    continue
                if not any(c.kind == kind and c.value.lower() == value.lower() for c in out):
                    out.append(Constraint(kind, value, m.group(0).strip()))
        return out

    @staticmethod
    def priority(text: str) -> PlanPriority:
        for pattern, priority in _PRIORITY_RULES:
            if pattern.search(text):
                return priority
        return PlanPriority.NORMAL

    def deadline(self, text: str) -> float | None:
        now = to_local(self.clock.now())
        lowered = text.lower()
        m = re.search(r"\b(?:within|in)\s+(?:the\s+next\s+)?(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?|h|m)\b",
                      lowered)
        if m and re.search(r"\b(within|in the next|done in|finished in|ready in)\b", lowered):
            amount = float(m.group(1))
            seconds = amount * (3600 if m.group(2).startswith("h") else 60)
            return self.clock.now() + seconds
        m = re.search(r"\b(?:by|before|until)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", lowered)
        if m:
            hour, minute = int(m.group(1)), int(m.group(2) or 0)
            if m.group(3) == "pm" and hour < 12:
                hour += 12
            elif m.group(3) is None and hour < 8:
                hour += 12        # "by 5" means 17:00
            if hour <= 23 and minute <= 59:
                when = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if when <= now:
                    when += timedelta(days=1)
                return when.timestamp()
        m = re.search(r"\b(?:by|before)\s+(tonight|tomorrow(?:\s+morning)?|end\s+of\s+(?:the\s+)?day|eod|" +
                      "|".join(_WEEKDAYS) + r")\b", lowered)
        if m:
            word = m.group(1)
            if word == "tonight":
                when = now.replace(hour=21, minute=0, second=0, microsecond=0)
            elif word.startswith("tomorrow"):
                when = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
            elif word in ("eod",) or word.startswith("end of"):
                when = now.replace(hour=17, minute=0, second=0, microsecond=0)
            else:
                days = (_WEEKDAYS.index(word) - now.weekday()) % 7 or 7
                when = (now + timedelta(days=days)).replace(hour=17, minute=0, second=0, microsecond=0)
            if when <= now:
                when += timedelta(days=1)
            return when.timestamp()
        return None

    @staticmethod
    def target(text: str, kind: str) -> str | None:
        m = re.search(r"\b(?:back\s*up|backup|copy|archive)\s+(?:my\s+|the\s+)?(?P<src>[~/.\\]?[\w:./\\ -]+?)"
                      r"\s+(?:to|into|onto)\s+(?P<dst>[~/.\\]?[\w:./\\ -]+?)"
                      r"(?=$|[,;!?]|\.(?:\s|$)|\s+(?:and|then|but|by|before|until|within|tonight|every)\b)",
                      text, re.I)
        if m:
            return f"{m.group('src').strip()} -> {m.group('dst').strip()}"
        if kind == "backup":
            m = re.search(r"\b(?:back\s*up|backup|archive)\s+(?:my\s+|the\s+)?(?P<src>[~/.\\]?[\w:./\\ -]+?)"
                          r"(?=$|[,.;!?]|\s+(?:and|then|but|please)\b)", text, re.I)
            return m.group("src").strip() if m else None
        if kind == "research":
            m = re.search(r"\b(?:about|on|into|regarding)\s+(?P<topic>.+?)(?=$|[.;!?]|\s+(?:in|from|under|inside)\s+"
                          r"(?:my|the|our)\b|\s+(?:in|from|under|inside)\s+(?:[~/.\\]|[A-Za-z]:\\))", text, re.I)
            return m.group("topic").strip() if m else None
        if kind == "disk_cleanup":
            m = re.search(r"\b(?:clean(?:\s*up)?|free\s+up\s+space\s+(?:in|on))\s+(?:my\s+|the\s+)?"
                          r"(?P<where>downloads|temp|desktop|[~/.\\][\w:./\\ -]+)", text, re.I)
            return m.group("where").strip() if m else None
        return None

    @staticmethod
    def clauses(text: str) -> list[SubGoal]:
        """Split a compound request into parts with their dependencies and conditions."""
        parts: list[tuple[str, str, str | None]] = []   # (text, relation to previous, condition)
        # split on "if it passes" / "otherwise" first: they carry conditions
        pieces = _IF.split(text)
        # _IF.split yields [before, cond, after, cond, after...]
        segments: list[tuple[str, str | None]] = [(pieces[0], None)]
        for i in range(1, len(pieces), 2):
            cond_word = pieces[i].lower()
            cond = "failure" if re.match(r"fail|doesn|break", cond_word) else "success"
            segments.append((pieces[i + 1], cond))
        for seg_index, (segment, cond) in enumerate(segments):
            otherwise = _OTHERWISE.split(segment)
            for oi, piece in enumerate(otherwise):
                seq = [s for s in _THEN.split(piece)]
                for si, s in enumerate(seq):
                    par = [p for p in _AND.split(s)]
                    for pi, p in enumerate(par):
                        p = p.strip(" ,.;")
                        if not p:
                            continue
                        if oi > 0 and si == 0 and pi == 0:
                            relation, c = "then", "failure" if cond != "failure" else "success"
                        elif seg_index > 0 and oi == 0 and si == 0 and pi == 0:
                            relation, c = "then", cond
                        elif si > 0 and pi == 0:
                            relation, c = "then", None
                        elif pi > 0:
                            relation, c = "and", None
                        else:
                            relation, c = "then" if parts else "start", None
                        parts.append((p, relation, c))
        subgoals: list[SubGoal] = []
        last_seq: list[int] = []            # the group the next "then" waits for
        group: list[int] = []
        for index, (p, relation, cond) in enumerate(parts):
            if relation == "and" and subgoals:
                after = list(subgoals[-1].after)
                group.append(index)
            elif relation == "start":
                after = []
                group = [index]
            else:
                after = list(group) if group else list(last_seq)
                last_seq = list(group)
                group = [index]
            subgoals.append(SubGoal(p, after, cond))
        return subgoals

    def _strip_constraint_clauses(self, text: str) -> str:
        """The request without clauses that only state constraints ("..., but don't close Chrome")."""
        pieces = _BUT.split(text)
        kept = [p for p in pieces if not (self.constraints(p) and not _ACTION_VERBS.search(
            re.sub(r"\b(?:don'?t|do\s+not|never|without)\s+\w+", "", p, flags=re.I)))]
        out = " but ".join(kept).strip(" ,.")
        return out

    def ambiguity(self, text: str, goal: Goal) -> Ambiguity | None:
        m = _VAGUE_OBJECT.search(text)
        if m and goal.mode == ExecutionMode.EXECUTE:
            verb = m.group(1).lower()
            dangerous = bool(_DANGEROUS_VERBS.search(text)) or m.group("obj").lower().startswith(("everything", "all"))
            klass = AmbiguityClass.DANGEROUS if dangerous else AmbiguityClass.CONSEQUENTIAL
            if verb in ("kill", "stop", "close"):
                return Ambiguity(klass, "which program", f"Which program should I {verb}?")
            return Ambiguity(klass, "which files", f"Which files should I {verb}? Give me a folder or a pattern.")
        if goal.kind == "backup" and goal.mode == ExecutionMode.EXECUTE:
            if not goal.target:
                return Ambiguity(AmbiguityClass.RECOVERABLE, "what to back up", "What should I back up, and where to?")
            if "->" not in goal.target:
                return Ambiguity(AmbiguityClass.RECOVERABLE, "the destination",
                                 f"Where should I back up {goal.target} to?")
        if goal.kind == "disk_cleanup" and not goal.target:
            return Ambiguity(AmbiguityClass.RECOVERABLE, "where to clean up",
                             assumption="I'll look at the whole home folder and only recommend what to remove; "
                                        "deleting anything needs your approval")
        if goal.kind == "performance":
            return None
        if re.search(r"\b(make\s+it\s+better|improve\s+it|fix\s+it|sort\s+it)\b", text, re.I) and goal.kind == "generic":
            return Ambiguity(AmbiguityClass.HARMLESS, "what 'it' refers to",
                             assumption="the thing we were just talking about")
        return None

    @staticmethod
    def complexity(goal: Goal, text: str) -> Complexity:
        if goal.kind == "compound":
            actions = sum(1 for s in goal.subgoals if _ACTION_VERBS.search(s.text))
            conditional = any(s.condition for s in goal.subgoals)
            if len(goal.subgoals) >= 4 or goal.deadline:
                return Complexity.VERY_COMPLEX
            if len(goal.subgoals) >= 3 or conditional or actions >= 2:
                return Complexity.COMPLEX
            return Complexity.MODERATE
        if goal.kind == "performance":
            return Complexity.COMPLEX if goal.wants_fix else Complexity.MODERATE
        if goal.kind == "disk_cleanup":
            return Complexity.COMPLEX if goal.wants_fix else Complexity.MODERATE
        if goal.kind == "backup":
            return Complexity.MODERATE
        if goal.kind == "research":
            return Complexity.VERY_COMPLEX if re.search(r"\b(everything|all|thorough|in depth|detailed)\b", text, re.I) \
                else Complexity.COMPLEX
        if re.search(r"\b(step by step|make a plan|plan (it|this) out|in stages)\b", text, re.I):
            return Complexity.COMPLEX
        return Complexity.SIMPLE

    @staticmethod
    def _describe(goal: Goal) -> None:
        """Outcome, success criteria, failure conditions and expected permissions per kind."""
        if goal.kind == "performance":
            goal.outcome = "the computer is responsive again, with the cause identified" if goal.wants_fix else \
                "the cause of the slowness is identified"
            goal.success_criteria = ["the main resource bottleneck is identified with evidence"]
            if goal.wants_fix:
                goal.success_criteria.append("after the fix, an independent measurement shows the bottleneck eased")
            goal.failure_conditions = ["no evidence could be gathered", "the fix made things worse"]
            goal.resources = ["cpu"]
            goal.permissions = "changes" if goal.wants_fix else "read-only"
        elif goal.kind == "disk_cleanup":
            goal.outcome = "disk space is freed" if goal.wants_fix else "what uses the disk space is known"
            goal.success_criteria = ["the largest space users are identified with sizes"]
            if goal.wants_fix:
                goal.success_criteria.append("free space increased, measured independently")
            goal.permissions = "changes" if goal.wants_fix else "read-only"
        elif goal.kind == "backup":
            goal.outcome = "an up-to-date copy exists at the destination"
            goal.success_criteria = ["every source file exists at the destination with the same size"]
            goal.failure_conditions = ["the destination is not writable", "a file could not be copied"]
            goal.permissions = "changes"
        elif goal.kind == "research":
            goal.outcome = "a summary of what the sources say, with every claim traced to a source"
            goal.success_criteria = ["each claim cites a source that contains it"]
            goal.permissions = "read-only"
        elif goal.kind == "compound":
            goal.outcome = "every part of the request is done, in order"
            goal.success_criteria = [f"'{s.text}' done" for s in goal.subgoals]
            goal.permissions = "changes" if any(_ACTION_VERBS.search(s.text) for s in goal.subgoals) else "read-only"
        else:
            goal.outcome = goal.objective
            goal.permissions = "unknown"
        if goal.mode != ExecutionMode.EXECUTE:
            goal.permissions = "read-only"


def _sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" ,.!?")
    return text[:1].upper() + text[1:] if text else text


def fingerprint(tool: str, args: dict[str, Any]) -> str:
    """A stable identity for one exact action, used to bind an approval to what was shown."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps([tool, args], sort_keys=True, default=str).encode()).hexdigest()[:16]
