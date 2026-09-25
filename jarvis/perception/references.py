"""Resolving what the user means by "this", "the previous one", "the second screenshot" (Phase 4 §3, §22, §33).

References are resolved against recorded observations — their kind, order in the session, time, and what was learned
from them — never against a model's recollection. Rules, in order:

1. Inputs attached to this very message are what "this"/"these" mean.
2. Explicit numbers and ordinals: "screenshot 3", "image #2", "the second screenshot", "the first one", "the last one".
3. Relative ones: "the previous one", "the one before", "the earlier one", "the other one".
4. Comparison: "compare these two", "what changed", "both", "with the previous one" → two observations, earlier first.
5. Time and topic: "the screenshot from earlier", "the diagram we looked at yesterday", "the error image" → searched
   by time window, kind and what was seen in it; nothing found is reported as nothing found.
6. Otherwise a pronoun ("this", "it", "that", "here") means the most recent input, if it's recent.

When two readings are equally good (two screenshots arrived together and the user says "the screenshot"), the answer
is a short clarifying question instead of a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from jarvis.perception.inputs import InputKind, Observation

VISUAL = (InputKind.IMAGE, InputKind.SCREENSHOT, InputKind.CAMERA_FRAME)
DOCS = (InputKind.DOCUMENT, InputKind.FILE)

_NOUNS = {
    # people call any picture of a screen a screenshot, so it matches images too (screenshots first)
    "screenshot": (InputKind.SCREENSHOT, InputKind.IMAGE), "screen shot": (InputKind.SCREENSHOT, InputKind.IMAGE),
    "screengrab": (InputKind.SCREENSHOT, InputKind.IMAGE), "snip": (InputKind.SCREENSHOT, InputKind.IMAGE),
    "image": VISUAL, "picture": VISUAL, "photo": VISUAL, "pic": VISUAL,
    "diagram": VISUAL, "chart": VISUAL, "graph": VISUAL, "schematic": VISUAL, "drawing": VISUAL, "mockup": VISUAL,
    "mock-up": VISUAL, "flowchart": VISUAL, "circuit": VISUAL, "document": DOCS, "doc": DOCS, "file": DOCS,
    "pdf": DOCS, "report": DOCS, "spec": DOCS, "log": DOCS, "readme": DOCS, "csv": DOCS, "spreadsheet": DOCS,
}
_TOPIC_NOUNS = {"diagram", "chart", "graph", "schematic", "drawing", "mockup", "mock-up", "flowchart", "circuit"}
_NOUN_RE = re.compile(r"\b(screen\s?shots?|screengrabs?|snips?|images?|pictures?|photos?|pics?|diagrams?|charts?|"
                      r"graphs?|schematics?|drawings?|mock-?ups?|flowcharts?|circuits?|documents?|docs?|files?|pdfs?|"
                      r"reports?|specs?|logs?|readmes?|csvs?|spreadsheets?)\b", re.I)
_ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5,
             "5th": 5}
_NUMBERED = re.compile(r"\b(screen\s?shot|image|picture|photo|document|file|diagram)\s*(?:#|no\.?|number)?\s*(\d{1,3})\b",
                       re.I)
_ORDINAL_RE = re.compile(r"\b(the\s+)?(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|last|latest|newest|"
                         r"most\s+recent)\s+(one|screen\s?shot|image|picture|photo|document|file|diagram|pdf)?\b", re.I)
_PREVIOUS = re.compile(r"\b(previous|prior|earlier|older|before|other|original|last\s+time)\b(\s+one)?|"
                       r"\bthe\s+one\s+(before|earlier)\b|\bfrom\s+before\b", re.I)
_COMPARE = re.compile(r"\b(compare|comparison|difference|differences|diff|what\s+changed|what(?:['’]?s|\s+is)\s+changed|"
                      r"what\s+is\s+different|what(?:['’]?s|\s+is)\s+different|changed\s+between|these\s+two|those\s+two|both|"
                      r"side\s+by\s+side|versus|vs\.?)\b", re.I)
_PRONOUN = re.compile(r"\b(this|that|it|these|those|them|here|there|attached|above)\b", re.I)
_YESTERDAY = re.compile(r"\byesterday\b", re.I)
_LAST_WEEK = re.compile(r"\b(last\s+week|the\s+other\s+day|a\s+few\s+days\s+ago)\b", re.I)
_LAST_SESSION = re.compile(r"\b(last|previous)\s+(debugging\s+)?session\b", re.I)
_TOPIC = re.compile(r"\bthe\s+([a-z][a-z\-]{2,20})\s+(screen\s?shot|image|picture|photo|diagram|document|file|pdf)\b",
                    re.I)
_TOPIC_SKIP = {"first", "second", "third", "last", "previous", "other", "earlier", "same", "new", "old", "original",
               "latest", "whole", "entire", "attached", "above", "next"}


@dataclass
class Resolution:
    observations: list[Observation] = field(default_factory=list)
    reason: str = ""
    question: str | None = None            # a clarifying question when the reference is ambiguous
    compare: bool = False
    explicit: bool = False                 # the user named which one (ordinal, number, time, topic)
    missing: str | None = None             # what they referred to that doesn't exist ("a diagram from yesterday")


def wanted_kinds(text: str) -> tuple[InputKind, ...] | None:
    kinds: list[InputKind] = []
    for m in _NOUN_RE.finditer(text):
        word = m.group(1).lower().replace(" ", " ")
        base = re.sub(r"s$", "", word.replace("screen shot", "screenshot").replace("mock-up", "mockup"))
        base = "screenshot" if base.startswith("screen") else base
        for k in _NOUNS.get(base, ()):
            if k not in kinds:
                kinds.append(k)
    return tuple(kinds) or None


def mentions_input(text: str) -> bool:
    """Whether the text refers to something shown/attached (not whether anything exists)."""
    return bool(_NOUN_RE.search(text) or _NUMBERED.search(text) or re.search(
        r"\b(look\s+at\s+(this|that|these|it)|what(?:['’]?s|\s+is)\s+wrong\s+(here|with\s+(this|that|it))|what\s+does\s+(this|that|"
        r"it)\s+(say|show|mean)|what\s+is\s+(this|that)|read\s+(this|that|it)|(this|that)\s+error|"
        r"the\s+(previous|other|earlier|first|second|last)\s+one|compare\s+(these|them|this|those)|"
        r"where\s+do\s+i\s+click|what\s+should\s+i\s+(click|press)|explain\s+(this|that)\s+(screen|window)?)", text, re.I))


def resolve(text: str, recent: list[Observation], *, attached: list[Observation] | None = None, now: float = 0.0,
            store: Any = None, recent_s: float = 3600.0) -> Resolution:
    """Which observations the text refers to. ``recent`` is this session's, newest first."""
    attached = attached or []
    kinds = wanted_kinds(text)
    compare = bool(_COMPARE.search(text))
    pool = [o for o in recent if kinds is None or o.kind in kinds]

    # 5a. time and topic outside this session ("the diagram we looked at yesterday")
    window = _time_window(text, now)
    topic = _topic(text)
    if window is not None and store is not None:
        since, until, label = window
        found = [o for o in store.recent(None, kinds=kinds, since=since, until=until, limit=50)
                 if not topic or _about(o, topic)]
        if not found:
            what = f"{topic} " if topic else ""
            noun = (kinds[0].label if kinds and len(kinds) == 1 else "image or document")
            return Resolution(reason="nothing matches", missing=f"a {what}{noun} from {label}".replace("  ", " "),
                              explicit=True)
        return Resolution(found[:2 if compare else 1][::-1], f"from {label}", compare=compare, explicit=True)

    # 1. attached right now
    if attached:
        chosen = [o for o in attached if kinds is None or o.kind in kinds] or attached
        if compare and len(chosen) == 1 and _PREVIOUS.search(text):
            before = next((o for o in pool if o.id not in {a.id for a in attached}), None)
            if before is not None:
                return Resolution([before, chosen[0]], "this one and the previous one", compare=True)
        if not compare and len(chosen) > 1 and _singular_definite(text):
            return Resolution(chosen, "ambiguous", question=_which(chosen))
        return Resolution(chosen, "attached with this message", compare=compare and len(chosen) > 1)

    # 2. explicit number / ordinal
    m = _NUMBERED.search(text)
    if m:
        kind_word = m.group(1).lower().replace(" ", "")
        number = int(m.group(2))
        cands = [o for o in recent if o.ordinal == number and
                 (o.kind.value == kind_word or (kind_word in ("image", "picture", "photo", "diagram") and
                                                o.kind in VISUAL) or (kind_word in ("document", "file") and
                                                                      o.kind in DOCS))]
        if cands:
            return Resolution(cands[:1], f"{kind_word} {number}", explicit=True)
        return Resolution(reason="no such number", missing=f"{m.group(1).lower()} {number}", explicit=True)
    om = _ORDINAL_RE.search(text)
    if om and pool:
        word = om.group(2).lower()
        ordered = sorted(pool, key=lambda o: o.created_at)
        if word in ("last", "latest", "newest") or word.startswith("most"):
            return Resolution([ordered[-1]], "the latest", explicit=True)
        n = _ORDINALS.get(word)
        if n and n <= len(ordered):
            same_turn = [o for o in ordered if abs(o.created_at - ordered[-1].created_at) < 2.0]
            base = same_turn if len(same_turn) >= n else ordered
            return Resolution([base[n - 1]], f"the {word}", explicit=True)
        if n:
            return Resolution(reason="not that many", missing=f"a {word} {kinds[0].label if kinds else 'input'}",
                              explicit=True)

    # 4. comparison
    if compare:
        if len(pool) >= 2:
            return Resolution([pool[1], pool[0]], "the last two", compare=True)
        return Resolution(pool[:1], "only one available", compare=True,
                          missing="a second image to compare with" if pool else "two images to compare")

    # 3. relative
    if _PREVIOUS.search(text) and len(pool) >= 2:
        return Resolution([pool[1]], "the previous one", explicit=True)

    # 5b. topic within the session ("the error screenshot")
    if topic and pool:
        about = [o for o in pool if _about(o, topic)]
        if about:
            return Resolution(about[:1], f"the {topic} one", explicit=True)

    # 6. pronoun / bare noun: the most recent, if recent
    if pool and (now == 0.0 or now - pool[0].created_at <= recent_s):
        return Resolution(pool[:1], "the most recent")
    if pool:
        return Resolution(pool[:1], "the most recent (from a while ago)")
    return Resolution(reason="nothing recent")


def _singular_definite(text: str) -> bool:
    return bool(re.search(r"\bthe\s+(screen\s?shot|image|picture|photo|document|file|diagram|one)\b(?!s)", text, re.I)) \
        and not re.search(r"\b(both|these|those|all|them|each)\b", text, re.I)


def _which(obs: list[Observation]) -> str:
    options = " or ".join(f"{o.handle} ({o.name})" for o in obs[:4])
    return f"Which one do you mean: {options}?"


def _topic(text: str) -> str | None:
    m = _TOPIC.search(text)
    if m and m.group(1).lower() not in _TOPIC_SKIP:
        return m.group(1).lower()
    for noun in _TOPIC_NOUNS:
        if re.search(rf"\b{noun}s?\b", text, re.I):
            return noun
    return None


def _about(o: Observation, topic: str) -> bool:
    hay = " ".join([o.name, " ".join(o.labels), o.summary()]).lower()
    return topic.lower().rstrip("s") in hay


def _time_window(text: str, now: float) -> tuple[float, float, str] | None:
    if not now:
        return None
    today = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0)
    if _YESTERDAY.search(text):
        return (today - timedelta(days=1)).timestamp(), today.timestamp(), "yesterday"
    if _LAST_WEEK.search(text):
        return (today - timedelta(days=8)).timestamp(), today.timestamp(), "the last week"
    if _LAST_SESSION.search(text):
        return now - 30 * 86400, now - 1, "an earlier session"
    return None
