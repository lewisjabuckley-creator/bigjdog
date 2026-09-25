"""Intent engine (spec §13-15, §48-49, §56-59, §114-117, §145-146).

First a deterministic grammar recognises the control and status language the
system must always understand — "stop", "continue", "what are you doing?",
"keep an eye on it" — so these work with no language model at all. Anything
else is conversational and goes to the model with tools. Every intent carries
its goal, target reference and parameters; the orchestrator adds scope,
authority and success conditions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable


class IntentKind(StrEnum):
    STATUS = "status"
    REENTRY = "reentry"
    BRIEFING = "briefing"
    WHAT_CHANGED = "what_changed"
    AWAY = "away"
    DIAGNOSE = "diagnose"
    WHY = "why"
    WHAT_DID_YOU_DO = "what_did_you_do"
    PROVENANCE = "provenance"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    RETRY = "retry"
    APPROVE = "approve"
    DENY = "deny"
    MODIFY_PLAN = "modify_plan"
    MODE = "mode"
    REMEMBER = "remember"
    FORGET = "forget"
    DONT_REMEMBER = "dont_remember"
    RECALL = "recall"
    DECISION_WHY = "decision_why"
    RECORD_DECISION = "record_decision"
    LOG_THAT = "log_that"
    MODEL_USE = "model_use"
    MODEL_UNLOAD = "model_unload"
    MODEL_LIST = "model_list"
    MONITOR = "monitor"
    RUN_TESTS = "run_tests"
    BUILD = "build"
    SHELL = "shell"
    OPEN_PROJECT = "open_project"
    ANALYZE_PROJECT = "analyze_project"
    CLI_COMMAND = "cli_command"          # a Command Prompt command for JARVIS typed into the conversation
    REPEAT_FOR = "repeat_for"
    PRIORITY = "priority"
    TIME = "time"
    SELF = "self"
    HELP = "help"
    SHORTER = "shorter"
    LONGER = "longer"
    SYSTEM_QUERY = "system_query"        # one live number: CPU temperature, memory use...
    ADVISE = "advise"                    # "what should I do?" — recommend, don't act
    SIMULATE = "simulate"                # "what would happen if...?"
    PREDICT = "predict"                  # "how long will the tests take?"
    PLAN_SHOW = "plan_show"
    PLAN_DETAILS = "plan_details"        # "what are these files?", "what did you find?"
    PLAN_HISTORY = "plan_history"
    PERCEIVE = "perceive"                # about an image, screenshot or document the user shared (Phase 4)
    INPUTS = "inputs"                    # "what can you see?", "what inputs do you have?"
    SCREEN = "screen"                    # screen awareness: on / off / look
    AUTONOMY = "autonomy"
    CHAT = "chat"


@dataclass
class Intent:
    kind: IntentKind
    text: str
    target: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    source: str = "rules"       # rules | model
    confidence: float = 1.0


_PREFIX = re.compile(r"^(hey |ok |okay )?(jarvis[,:!]?\s+)?(please\s+|could you\s+|can you\s+|would you\s+)?",
                     re.IGNORECASE)
_SUFFIX = re.compile(r"(?:[\s,]*(?:please|thanks|thank you|jarvis|for me))*[\s.!?]*$", re.IGNORECASE)
_REF = r"(?P<target>.+?)"

Extractor = Callable[[re.Match[str]], dict[str, Any]]


def _t(m: re.Match[str]) -> dict[str, Any]:
    target = m.groupdict().get("target")
    return {"target": target.strip() if target else None}


RULES: list[tuple[re.Pattern[str], IntentKind, Extractor | None]] = []


def rule(pattern: str, kind: IntentKind, extractor: Extractor | None = None) -> None:
    RULES.append((re.compile(rf"^(?:{pattern})$", re.IGNORECASE), kind, extractor))


# -- control -----------------------------------------------------------------------------
rule(r"(stop|halt|abort|cancel|kill|forget it|never ?mind|nevermind)", IntentKind.STOP)
rule(r"(stop|halt|abort|cancel|kill) (that|it|this|everything|all( tasks)?|the (?P<target>.+)|(?P<target2>.+))",
     IntentKind.STOP, lambda m: {"target": m.group("target") or m.group("target2") or m.group(2)})
rule(r"(pause|hold on|wait|hold)( (that|it|this|the (?P<target>.+)))?", IntentKind.PAUSE, _t)
rule(r"(pause|hold) (?P<target>.+)", IntentKind.PAUSE, _t)
rule(r"(continue|resume|carry on|keep going|go on|keep working( on (it|this|that))?|pick up where we left off)"
     r"( (with )?(that|it|this|the (?P<target>.+)))?", IntentKind.RESUME, _t)
rule(r"continue (the )?(?P<target>.+?)( project)?", IntentKind.RESUME, _t)
rule(r"(run|do|try) (that|it|this) again|again|retry( (that|it))?|try again", IntentKind.RETRY)
rule(r"(yes|yeah|yep|proceed|go ahead|do it|approved?|confirm(ed)?|ok(ay)?,? (go|do it)|sure,? go ahead|make it so)",
     IntentKind.APPROVE)
rule(r"(no|nope|don'?t|do not|deny|denied|reject|don'?t do (that|it)|cancel that request)", IntentKind.DENY)
rule(r"actually,? (don'?t|do not|skip|no) (?P<target>.+?)( yet)?", IntentKind.MODIFY_PLAN, _t)
rule(r"(skip|drop|remove) (the )?(?P<target>.+?) step", IntentKind.MODIFY_PLAN, _t)
rule(r"(make|set) (?P<target>.+?) (a |to )?(?P<level>top|high|low|urgent|background) priority", IntentKind.PRIORITY,
     lambda m: {"target": m.group("target"), "level": m.group("level")})
rule(r"(prioriti[sz]e|focus on) (the )?(?P<target>.+)", IntentKind.PRIORITY, lambda m: {"target": m.group("target"),
                                                                                         "level": "top"})

# -- status / awareness --------------------------------------------------------------------
rule(r"what are you (doing|working on|up to)|what'?s running|what is running|what'?s (going on|happening)|"
     r"status|what'?s the status|system status|are we good|how are we doing|how'?s (it|everything) going|"
     r"anything i (need|should) (to )?know( about)?|sitrep|report", IntentKind.STATUS)
rule(r"how far along (is|are) (?P<target>.+?)|how'?s the (?P<target2>.+?)( going| doing)?|"
     r"(what'?s|what is) the status of (?P<target3>.+)|is (?P<target4>.+?) (done|finished|complete)( yet)?",
     IntentKind.STATUS,
     lambda m: {"target": next((g for g in (m.group("target"), m.group("target2"), m.group("target3"),
                                            m.group("target4")) if g), None)})
_AWAY = r"while i was (away|gone|out|asleep|offline)|since i (left|closed you|was last here)"
rule(rf"(so,? )?(what|anything) (happened|happen|went on|changed)( ?(,|-)? ?({_AWAY}))\??|"
     rf"what did i miss( ({_AWAY}))?|what (did you do|have you (done|been doing|been up to))( ({_AWAY}))|"
     rf"({_AWAY}),? what (happened|did i miss|did you do)|(any|what) news( ({_AWAY}))?|"
     r"what have you been (doing|up to)|i'?m back|fill me in", IntentKind.AWAY)
rule(r"where (were|are) we|where did we leave off|what are we (working on|doing)|catch me up|what was i doing",
     IntentKind.REENTRY)
rule(r"(good )?morning|(daily |morning )?briefing|brief me|what'?s (on )?today|daily summary", IntentKind.BRIEFING)
rule(r"what changed|what'?s changed|what has changed|what'?s new|what did i miss|anything new", IntentKind.WHAT_CHANGED)
rule(r"(what'?s|what is|is (there )?something|is anything) wrong( with (?P<target>.+))?|"
     r"what happened( (to|with) (?P<target2>.+))?|what'?s the problem|"
     r"(check |find out |tell me )?why (did )?(?P<target3>.+?) (fail(ed)?|crash(ed)?|break|broke|stop(ped)?)|"
     r"why is (?P<target4>.+?) (slow|failing|down|broken|not working|stuck|waiting|blocked)|diagnose( (?P<target5>.+))?|"
     r"show me what'?s causing the problem|why is it slow",
     IntentKind.DIAGNOSE,
     lambda m: {"target": next((g for g in (m.group("target"), m.group("target2"), m.group("target3"),
                                            m.group("target4"), m.group("target5")) if g), None)})
rule(r"why did you (do that|do this|pause|stop|cancel|retry|resume|throttle|switch|change) ?(?P<target>.*)",
     IntentKind.WHY, _t)
rule(r"why\??", IntentKind.WHY)
rule(r"what did you (do|change)( (today|recently|while i was away))?|what have you done", IntentKind.WHAT_DID_YOU_DO)
rule(r"where did you get that( from)?|how do you know( that)?|what'?s your source|source\??", IntentKind.PROVENANCE)
rule(r"what time is it|(what'?s|what is) the (time|date)|what day is it|time\??", IntentKind.TIME)
rule(r"what are you|who are you|what can you do|what are your (capabilities|limits)", IntentKind.SELF)
rule(r"help|commands", IntentKind.HELP)
rule(r"short(er)? version|shorter|be brief|tl;?dr", IntentKind.SHORTER)
rule(r"explain (everything|in detail|more)|more detail|go deeper|long version", IntentKind.LONGER)

# -- modes ----------------------------------------------------------------------------------
rule(r"(enter |switch to |go to |activate |enable )?(?P<mode>focus|normal|development|dev|research|presentation|travel|"
     r"maintenance|emergency|low[- ]resource|offline|private|debug) mode( on)?", IntentKind.MODE,
     lambda m: {"mode": m.group("mode").lower().replace("-", "_").replace(" ", "_"), "on": True})
rule(r"(exit|leave|end|disable|turn off|stop) (?P<mode>focus|presentation|emergency|private|debug|offline|"
     r"low[- ]resource|maintenance|travel|research|development|dev)( mode)?", IntentKind.MODE,
     lambda m: {"mode": m.group("mode").lower().replace("-", "_").replace(" ", "_"), "on": False})
rule(r"(quiet|be quiet|silence|shh+|mute)", IntentKind.MODE, lambda m: {"mode": "quiet", "on": True})
rule(r"(you can talk again|unmute|speak( up)?|resume (talking|notifications))", IntentKind.MODE,
     lambda m: {"mode": "quiet", "on": False})

# -- memory --------------------------------------------------------------------------------
rule(r"don'?t remember (this|that)|off the record|don'?t save (this|that)", IntentKind.DONT_REMEMBER)
rule(r"remember (that|this)", IntentKind.REMEMBER, lambda m: {"content": None})
rule(r"remember (that |this: |:)?(?P<content>.+)", IntentKind.REMEMBER, lambda m: {"content": m.group("content")})
rule(r"note (that|this)?:? ?(?P<content>.+)", IntentKind.REMEMBER, lambda m: {"content": m.group("content")})
rule(r"(forget|delete) (everything|all) (you (know|remember) )?about (?P<target>.+)", IntentKind.FORGET,
     lambda m: {"target": m.group("target"), "all": True})
rule(r"forget (that|this|what i (just )?said)", IntentKind.FORGET, lambda m: {"target": None})
rule(r"forget (about )?(?P<target>.+)", IntentKind.FORGET, _t)
rule(r"what do you (remember|know) about (?P<target>.+)|do you remember (?P<target2>.+)|"
     r"remember the (?P<target3>.+?)\?", IntentKind.RECALL,
     lambda m: {"target": m.group("target") or m.group("target2") or m.group("target3")})
rule(r"why did we (choose|pick|go with|decide on|use|switch to) (?P<target>.+)", IntentKind.DECISION_WHY, _t)
rule(r"(decision|we decided|we'?ll go with|we'?re going with)[: ]+(?P<content>.+)", IntentKind.RECORD_DECISION,
     lambda m: {"content": m.group("content")})
rule(r"log (that|this)", IntentKind.LOG_THAT)

# -- models ----------------------------------------------------------------------------------
rule(r"(use|switch to) (the )?(?P<target>local|coding|code|vision|reasoning|fast|small|big|default|[\w.:-]+) model",
     IntentKind.MODEL_USE, _t)
rule(r"(use|switch to) (model )?(?P<target>[\w.-]+:[\w.-]+)", IntentKind.MODEL_USE, _t)
rule(r"unload (the )?(?P<target>[\w.:-]+)( model)?", IntentKind.MODEL_UNLOAD, _t)
rule(r"(what|which) models?( are| do you have)?( available| installed| are you using| is active)?|list models|"
     r"(what|which) model are you( using)?", IntentKind.MODEL_LIST)

# -- monitoring / delegation ------------------------------------------------------------------
rule(r"(keep an eye on|watch|monitor|keep watching) (this |the )?(folder|directory|file) (?P<target>.+)",
     IntentKind.MONITOR, lambda m: {"target": m.group("target"), "type": "path"})
rule(r"(keep an eye on|watch|monitor|keep watching|track) (?P<target>it|that|this|the .+|[~/.].+|.+)",
     IntentKind.MONITOR, _t)
rule(r"(tell|let|notify|ping) me (know )?when (?P<target>.+?)(?:'s|\u2019s|\s+is|\s+are|\s+has)?\s+"
     r"(done|finished|finishes|complete[sd]?|ready|over)", IntentKind.MONITOR,
     lambda m: {"target": m.group("target"), "notify": "urgent", "until_done": True})

# -- engineering ------------------------------------------------------------------------------
rule(r"(run|execute) (the |all )?(unit |integration )?tests?( suite)?( for (?P<target>.+?))?( again)?"
     r"(,? (and )?(?P<rest>.+))?", IntentKind.RUN_TESTS,
     lambda m: {"target": m.group("target"), "rest": m.group("rest")})
rule(r"(build|compile) (it|the project|this|the (?P<target>.+?)( project)?)", IntentKind.BUILD, _t)
rule(r"(run|execute)[: ]+`(?P<cmd>[^`]+)`|\$ ?(?P<cmd2>.+)|(run|execute) (the )?command:? (?P<cmd3>.+)",
     IntentKind.SHELL, lambda m: {"command": m.group("cmd") or m.group("cmd2") or m.group("cmd3")})
_CODEBASE = r"(project|repo|repository|codebase|code ?base|code)"
rule(rf"(analy[sz]e|review|audit|assess|summari[sz]e|give me an overview of|tell me about) (this|the|my|our) "
     rf"{_CODEBASE}( (at|in) (?P<target>.+))?", IntentKind.ANALYZE_PROJECT, _t)
rule(rf"(analy[sz]e|review|audit|assess) (the )?(?P<target>[\w.~/\\:-]+) {_CODEBASE}", IntentKind.ANALYZE_PROJECT, _t)
rule(r"(open|switch to|load|work on) (the )?project (at |in )?(?P<target>.+)", IntentKind.OPEN_PROJECT, _t)
rule(r"(open|switch to|load|work on) (the )?(?P<target>other one|.+?)( project)?", IntentKind.OPEN_PROJECT, _t)
rule(r"do the same (thing )?for (the )?(?P<target>.+?)( project)?", IntentKind.REPEAT_FOR, _t)

# -- planning and autonomy (Phase 3) ------------------------------------------------------------------------------
rule(r"(check |what'?s |what is |how'?s |how hot is |show( me)? |tell me )?(my |the )?(cpu|processor|gpu|graphics card)"
     r"('?s)? (temperature|temp|usage|load|utili[sz]ation)( right now| now)?|"
     r"(how much )?(ram|memory) (am i using|is (used|free|in use|left))( right now)?|"
     r"(check |what'?s |what is |how much )?(my |the )?(free )?(disk space|free space|storage)( is (left|free|used))?"
     r"( left)?|(how'?s |check )(my |the )?(battery)( level)?|battery( level)?", IntentKind.SYSTEM_QUERY)
rule(r"(so,? )?(what should i do|what do you (recommend|suggest|advise)|what would you (do|recommend|suggest)|"
     r"any (advice|suggestions|recommendations)|advise me|what'?s your (advice|recommendation))"
     r"( (about|with|for|regarding) (?P<target>.+?))?", IntentKind.ADVISE, _t)
rule(r"(simulate:? .+|what (would|will) happen if .+|what if (i|you|we) .+)", IntentKind.SIMULATE)
rule(r"(how long (will|would|does|should) .+|when will .+ (finish|be done)|predict .+|estimate how long .+)",
     IntentKind.PREDICT)
rule(r"(what|which) (are|were) (these|those|the|they)( files| ones| things| items)?|(what|which) files"
     r"( are (they|those|these)| were (they|those|these))?|show me( the)? (details|files|list|results?|them|those)|"
     r"(more )?details|tell me more|what did you find( out)?|list (them|the files|those)|what (are|were) they",
     IntentKind.PLAN_DETAILS)
rule(r"((list|show)( me)?( all)?( the| my)? (tasks|processes|jobs)|all( the)? (tasks|processes|jobs)|"
     r"(what|which) (tasks|processes|jobs) are (running|open|there))", IntentKind.STATUS)
rule(r"(show( me)?|what'?s|what is|tell me)( the| your)? plan( for (?P<target>.+?))?|what'?s the plan|"
     r"show( me)? the plan", IntentKind.PLAN_SHOW, _t)
rule(r"(plan history|(what|which) plans (have you|did you) (run|do)|show( me)? (my |the |recent )?plans|list plans|"
     r"what did you change (in|about) the plan|what changed in the plan)", IntentKind.PLAN_HISTORY)
rule(r"((set|change|switch) )?(your |the )?autonomy( level)?( to)? (?P<level>low|normal|high)|"
     r"(be|work) (more )?(autonomous(ly)?|independent(ly)?)|be more (careful|cautious)|"
     r"(what'?s|what is) (your|the) autonomy( level)?|autonomy( level)?",
     IntentKind.AUTONOMY, lambda m: {"level": m.group("level") or ("high" if re.search(
         r"autonomous|independent", m.group(0), re.I) else "low" if re.search(r"careful|cautious", m.group(0), re.I)
         else None)})


# `py -m jarvis ...`, `python3 -m jarvis ...`, `jarvis runtime stop`: commands that control JARVIS itself. Typed into
# the conversation they belong in a terminal; run by JARVIS they would act on JARVIS mid-task (e.g. stop it).
_SELF_COMMAND = re.compile(
    r"(^|[;&|]\s*)\$?\s*[\"']?(?:[\w:\\/.-]*[\\/])?(py|pythonw?(\d+(\.\d+)?)?)(\.exe)?[\"']?\s+(-\d(\.\d+)?\s+)?-m\s+jarvis\b"
    r"|(^|[;&|]\s*)\$?\s*jarvis(\.exe)?\s+(runtime|doctor|ask|schedule|--embedded|--simulate|--data-dir|--config)\b",
    re.IGNORECASE)


def is_self_command(text: str) -> bool:
    return bool(_SELF_COMMAND.search(text.strip().strip("`")))


def normalise(text: str) -> str:
    text = text.strip()
    text = _PREFIX.sub("", text, count=1)
    text = _SUFFIX.sub("", text)
    text = re.sub(r"\s+for (me|us)$", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def parse(text: str) -> Intent:
    raw = text.strip()
    if is_self_command(raw):
        return Intent(IntentKind.CLI_COMMAND, raw, params={"command": raw.lstrip("$ ").strip("`").strip()})
    dry_run = False
    body = normalise(raw)
    m = re.match(r"^dry[- ]run[:,]?\s*(?P<rest>.*)$", body, re.IGNORECASE)
    if m:
        dry_run = True
        body = normalise(m.group("rest")) or body
    if raw.startswith("$ ") or raw.startswith("`"):
        cmd = raw.lstrip("$ ").strip("`").strip()
        return Intent(IntentKind.SHELL, raw, params={"command": cmd}, dry_run=dry_run)
    if is_affirmative(body):
        return Intent(IntentKind.APPROVE, raw, dry_run=dry_run)
    for pattern, kind, extractor in RULES:
        match = pattern.match(body)
        if match:
            params = extractor(match) if extractor else {}
            target = params.pop("target", None)
            if isinstance(target, str):
                target = target.strip() or None
            return Intent(kind, raw, target, params, dry_run)
    return Intent(IntentKind.CHAT, raw, dry_run=dry_run, source="model")


PRONOUNS = {"it", "that", "this", "them", "those", "the task", "the job", "the thing", "that one", "this one",
            "that task", "this task", "that process", "this process", "the process", "that plan", "this plan",
            "the plan", "that job", "this job"}

# words that only say yes ("sure, proceed", "yes please go ahead", "ok do it"), with at least one of _YES
_YES = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "proceed", "approve", "approved", "confirm", "confirmed",
        "alright", "absolutely", "definitely", "affirmative"}
_YES_FILLER = _YES | {"please", "go", "ahead", "do", "it", "for", "that's", "thats", "sounds", "good", "fine", "then",
                      "carry", "on", "right", "thing", "course", "of", "and", "you", "can", "just"}


_EVERYTHING = re.compile(r"^(everything|all|all of (them|it)|(all|every)( of)?( the| my)? (tasks?|process(es)?|jobs?|"
                         r"plans?|work|things?)( (running|open|going))?|everything (running|open|going on))$")


def is_everything(target: str | None) -> bool:
    """"all", "everything", "all processes", "all of them", "every task": all of JARVIS's open work."""
    return bool(target) and bool(_EVERYTHING.match(" ".join(re.findall(r"[a-z]+", target.lower()))))


def is_affirmative(body: str) -> bool:
    words = re.findall(r"[a-z']+", body.lower())
    joined = " ".join(words)
    return bool(words) and len(words) <= 8 and all(w in _YES_FILLER for w in words) and (
        any(w in _YES for w in words) or "go ahead" in joined or re.search(r"\bdo it\b", joined) is not None)


def is_pronoun(target: str | None) -> bool:
    return target is None or target.lower().strip() in PRONOUNS
