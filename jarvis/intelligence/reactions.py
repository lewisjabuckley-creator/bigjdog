"""Event-driven autonomy, proactive suggestions and the user attention model (Phase 3 §38-40).

Some events deserve a response before anyone asks: CPU pinned for minutes, a disk about to fill. What JARVIS
does depends on the autonomy level:

* suggest (normal) — offer to look into it, with the reason ("CPU has been above 90% for 5 minutes");
* investigate (high, emergency) — start an observe-only investigation by itself and offer the findings; any
  change it recommends needs the user, because a plan nobody asked for can't be approved by anyone in the
  moment (and a scheduled or event-driven plan never gains authority for being automatic);
* off (low) — nothing.

Reactions are loop-protected: a cooldown per kind, an hourly cap, no reaction to events caused by JARVIS's own
plans, and none while a similar plan is already open. The attention model decides whether a suggestion is worth
interrupting for: it holds back in focus or quiet modes, and backs off from kinds of suggestion the user keeps
ignoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jarvis.core.types import NotificationPriority, OperationalReason
from jarvis.events.types import Event, EventType
from jarvis.intelligence.goals import ExecutionMode, PlanPriority
from jarvis.log import get_logger

log = get_logger("reactions")


@dataclass
class Reaction:
    kind: str                # a goal kind: performance | disk_cleanup
    goal: str                # the request an investigation would answer
    headline: str            # why JARVIS is bringing it up
    suggestion: str          # what the user can say


def _rule(event: Event) -> Reaction | None:
    p = event.payload
    if p.get("plan_id"):
        return None                                   # caused by one of JARVIS's own plans
    if event.type == str(EventType.RESOURCE_THRESHOLD_EXCEEDED):
        name = str(p.get("name", ""))
        message = p.get("message") or f"{name} is high"
        if name in ("cpu_percent", "memory_percent", "swap_percent"):
            return Reaction("performance", "why is my computer slow", message,
                            "Say 'find out why my computer is slow' and I'll look into it.")
        if name == "disk_percent":
            return Reaction("disk_cleanup", "what is using my disk space", message,
                            "Say 'free up disk space' and I'll find what can go (asking before removing anything).")
    if event.type == str(EventType.PREDICTIVE_WARNING) and "disk" in str(p.get("metric", "")):
        return Reaction("disk_cleanup", "what is using my disk space", p.get("message", "the disk is filling up"),
                        "Say 'free up disk space' and I'll find what can go.")
    return None


class AttentionModel:
    """Whether a suggestion is worth the user's attention now, and whether they tend to want this kind."""

    def __init__(self, state: Any, modes: Any = None, clock: Any = None) -> None:
        self.state = state
        self.modes = modes
        self.clock = clock

    def _key(self, kind: str) -> str:
        return f"intelligence.suggestions.{kind}"

    def stats(self, kind: str) -> dict[str, float]:
        value = self.state.value(self._key(kind)) if self.state is not None else None
        return dict(value) if isinstance(value, dict) else {"offered": 0, "acted": 0, "last": 0.0}

    def offered(self, kind: str) -> None:
        s = self.stats(kind)
        s["offered"] = s.get("offered", 0) + 1
        s["last"] = self.clock.now() if self.clock else 0.0
        self.state.set(self._key(kind), s)

    def acted(self, kind: str) -> None:
        s = self.stats(kind)
        s["acted"] = s.get("acted", 0) + 1
        self.state.set(self._key(kind), s)

    def relevance(self, kind: str) -> float:
        s = self.stats(kind)
        return (s.get("acted", 0) + 1) / (s.get("offered", 0) + 2)      # starts at 0.5, learns from use

    def should_offer(self, kind: str, *, urgent: bool = False) -> tuple[bool, str]:
        mode = getattr(getattr(self.modes, "current", None), "value", "normal") if self.modes else "normal"
        if mode in ("focus", "presentation") and not urgent:
            return False, f"{mode} mode is on"
        s = self.stats(kind)
        if s.get("offered", 0) >= 3 and self.relevance(kind) < 0.25 and not urgent:
            return False, "you haven't taken up the last few suggestions like this"
        return True, ""


class ReactionEngine:
    def __init__(self, service: Any, *, bus: Any, notifications: Any, autonomy: Any, attention: AttentionModel,
                 config: Any, clock: Any, audit: Any = None) -> None:
        self.service = service
        self.bus = bus
        self.notifications = notifications
        self.autonomy = autonomy
        self.attention = attention
        self.config = config
        self.clock = clock
        self.audit = audit
        self._last: dict[str, float] = {}
        self._recent: list[float] = []
        self.enabled = True

    def attach(self) -> None:
        for etype in (EventType.RESOURCE_THRESHOLD_EXCEEDED, EventType.PREDICTIVE_WARNING):
            self.bus.subscribe(str(etype), self._on_event, name="reactions")

    async def _on_event(self, event: Event) -> None:
        if not self.enabled:
            return
        reaction = _rule(event)
        if reaction is None:
            return
        mode = self.autonomy.reactions()
        if mode == "off":
            return
        now = self.clock.now()
        if now - self._last.get(reaction.kind, -1e18) < self.config.reaction_cooldown_s:
            return
        self._recent = [t for t in self._recent if now - t < 3600]
        if len(self._recent) >= self.config.max_reactions_per_hour:
            return
        if any(p.goal.kind == reaction.kind for p in self.service.store.open_plans()):
            return
        self._last[reaction.kind] = now
        self._recent.append(now)
        reason = OperationalReason(reaction.headline, f"autonomy is {self.autonomy.level.value}: "
                                   + ("investigate notable events" if mode == "investigate" else "offer to help"),
                                   "started an observe-only investigation" if mode == "investigate"
                                   else "offered to look into it")
        if self.audit is not None:
            self.audit.record(actor="system:reactions", action=f"reaction_{mode}", summary=reaction.headline,
                              reason=reason)
        if mode == "investigate":
            await self._investigate(reaction, event)
        elif self.config.proactive:
            self._suggest(reaction, event)

    def _suggest(self, reaction: Reaction, event: Event) -> None:
        ok, why = self.attention.should_offer(reaction.kind)
        if not ok:
            log.info("suggestion_held_back", kind=reaction.kind, why=why)
            return
        self.attention.offered(reaction.kind)
        self.notifications.notify(NotificationPriority.IMPORTANT, reaction.headline, reaction.suggestion,
                                  source="reactions", dedupe_key=f"suggest:{reaction.kind}")
        self.bus.emit(Event(EventType.PROACTIVE_SUGGESTION, "reactions",
                            {"kind": reaction.kind, "headline": reaction.headline, "suggestion": reaction.suggestion,
                             "trigger": str(event.type), "rule": "resource threshold exceeded"}))

    async def _investigate(self, reaction: Reaction, event: Event) -> None:
        goal = self.service.parser.parse(reaction.goal)
        goal.mode = ExecutionMode.ADVISE
        goal.priority = PlanPriority.BACKGROUND
        goal.wants_fix = False
        result = await self.service.run(goal, cwd=self.service.home(), interactive=False,
                                        created_by="system:reactions", origin=f"event:{event.type}")
        if result.plan is not None:
            self.attention.offered(reaction.kind)
            self.bus.emit(Event(EventType.PROACTIVE_SUGGESTION, "reactions",
                                {"kind": reaction.kind, "headline": reaction.headline, "plan_id": result.plan.id,
                                 "trigger": str(event.type), "rule": "investigate notable events"}))
