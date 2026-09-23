"""Mission modes and effective execution policy (spec §27-28, §42, §96).

A mode changes behaviour, not just tone: which notifications interrupt, whether
network and cloud are allowed, how verbose responses are, and how much
background work runs. Private mode, an active sensitive project and a lost
network connection are overlays combined into one effective policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Callable

from jarvis.clock import Clock, SystemClock
from jarvis.config import JarvisConfig
from jarvis.core.types import Confidence, NetworkState, NotificationPriority, Provenance, ProvenanceKind, Severity
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.state.engine import StateEngine
from jarvis.tools.base import ExecutionPolicy

NP = NotificationPriority


class Mode(StrEnum):
    NORMAL = "normal"
    FOCUS = "focus"
    DEVELOPMENT = "development"
    RESEARCH = "research"
    PRESENTATION = "presentation"
    TRAVEL = "travel"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    LOW_RESOURCE = "low_resource"
    OFFLINE = "offline"
    PRIVATE = "private"
    DEBUG = "debug"


@dataclass(frozen=True)
class ModePolicy:
    interrupt_min: NotificationPriority = NP.URGENT       # delivered immediately when the user is not busy
    busy_interrupt_min: NotificationPriority = NP.CRITICAL  # delivered immediately even while the user is busy
    deliver_queued_when_idle: bool = True
    humor: bool = True
    verbosity: str | None = None                          # None = user preference
    allow_network: bool = True
    local_only: bool = False
    background: str = "normal"                            # normal | reduced | essential
    show_operational_detail: bool = False


POLICIES: dict[Mode, ModePolicy] = {
    Mode.NORMAL: ModePolicy(),
    Mode.FOCUS: ModePolicy(deliver_queued_when_idle=False, humor=False, verbosity="short"),
    Mode.DEVELOPMENT: ModePolicy(show_operational_detail=True),
    Mode.RESEARCH: ModePolicy(verbosity="detailed"),
    Mode.PRESENTATION: ModePolicy(interrupt_min=NP.CRITICAL, deliver_queued_when_idle=False, humor=False,
                                  verbosity="short"),
    Mode.TRAVEL: ModePolicy(background="reduced", verbosity="short"),
    Mode.MAINTENANCE: ModePolicy(show_operational_detail=True),
    Mode.EMERGENCY: ModePolicy(interrupt_min=NP.URGENT, busy_interrupt_min=NP.URGENT, deliver_queued_when_idle=False,
                               humor=False, verbosity="short", background="essential"),
    Mode.LOW_RESOURCE: ModePolicy(background="essential", verbosity="short"),
    Mode.OFFLINE: ModePolicy(allow_network=False, local_only=True),
    Mode.PRIVATE: ModePolicy(allow_network=False, local_only=True),
    Mode.DEBUG: ModePolicy(interrupt_min=NP.IMPORTANT, busy_interrupt_min=NP.URGENT, show_operational_detail=True),
}

QUIET_OVERLAY = dict(interrupt_min=NP.CRITICAL, busy_interrupt_min=NP.CRITICAL, deliver_queued_when_idle=False)


@dataclass
class EffectivePolicy:
    mode: Mode
    policy: ModePolicy
    quiet: bool
    network_allowed: bool
    network_block_reason: str
    local_only: bool
    local_only_reason: str


class ModeManager:
    def __init__(self, state: StateEngine, config: JarvisConfig, *, bus: EventBus | None = None,
                 clock: Clock | None = None, project_sensitive: Callable[[], bool] | None = None) -> None:
        self.state = state
        self.config = config
        self.bus = bus
        self.clock = clock or SystemClock()
        self.project_sensitive = project_sensitive or (lambda: False)
        if self.state.value("mode.current") is None:
            self._write(Mode(config.general.default_mode), "configured default")
        if config.privacy.private_mode and self.state.value("privacy.private") is None:
            self.state.set("privacy.private", True, confidence=Confidence.KNOWN,
                           provenance=Provenance(ProvenanceKind.CONFIGURATION, "privacy.private_mode"))

    def _write(self, mode: Mode, reason: str) -> None:
        self.state.set("mode.current", mode.value, confidence=Confidence.KNOWN,
                       provenance=Provenance(ProvenanceKind.USER_STATEMENT, reason))

    @property
    def current(self) -> Mode:
        try:
            return Mode(self.state.value("mode.current", "normal"))
        except ValueError:
            return Mode.NORMAL

    @property
    def quiet(self) -> bool:
        return bool(self.state.value("attention.quiet", False))

    @property
    def private(self) -> bool:
        return bool(self.state.value("privacy.private", False)) or self.current == Mode.PRIVATE

    def set_mode(self, mode: Mode, *, by: str = "user", reason: str = "") -> Mode:
        previous = self.current
        if mode == previous:
            return previous
        if mode != Mode.EMERGENCY:
            self.state.set("mode.previous", previous.value, persist=True)
        elif previous != Mode.EMERGENCY:
            self.state.set("mode.before_emergency", previous.value)
        self._write(mode, reason or f"set by {by}")
        if self.bus:
            self.bus.emit(Event(EventType.MODE_CHANGED, "modes", {"from": previous.value, "to": mode.value, "by": by,
                                                                  "reason": reason},
                                severity=Severity.WARNING if mode == Mode.EMERGENCY else Severity.INFO))
        return previous

    def set_quiet(self, quiet: bool) -> None:
        self.state.set("attention.quiet", quiet, confidence=Confidence.KNOWN)

    def set_private(self, private: bool) -> None:
        self.state.set("privacy.private", private, confidence=Confidence.KNOWN,
                       provenance=Provenance(ProvenanceKind.USER_STATEMENT, "private mode"))
        if self.bus:
            self.bus.emit(Event(EventType.MODE_CHANGED, "modes", {"private": private}))

    def effective(self) -> EffectivePolicy:
        mode = self.current
        policy = POLICIES[mode]
        if self.quiet:
            policy = replace(policy, **QUIET_OVERLAY)
        network_ok, why_net = policy.allow_network, ""
        local_only, why_local = policy.local_only, ""
        if mode == Mode.OFFLINE:
            why_net = why_local = "offline mode is active"
        if self.private:
            network_ok, local_only = False, True
            why_net = "Private mode is active, so nothing leaves this machine"
            why_local = "private mode is active"
        if self.project_sensitive():
            local_only = True
            why_local = why_local or "the active project is marked sensitive"
        net = self.state.value("network.state")
        if net == NetworkState.OFFLINE.value and network_ok:
            network_ok, why_net = False, "the network is offline"
        return EffectivePolicy(mode, policy, self.quiet, network_ok, why_net, local_only, why_local)

    def execution_policy(self, project_policy: ExecutionPolicy | None = None) -> ExecutionPolicy:
        eff = self.effective()
        base = project_policy or ExecutionPolicy()
        network = base.network_allowed and eff.network_allowed
        reason = eff.network_block_reason if not eff.network_allowed else base.network_block_reason
        return replace(base, network_allowed=network, network_block_reason=reason)

    def verbosity(self) -> str:
        return self.effective().policy.verbosity or self.config.ui.verbosity
