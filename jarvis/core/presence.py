"""Which interfaces are attached to the runtime, and since when the user has been away.

The runtime keeps working when every interface is closed. Presence records when
the last one went away (persisted, so it survives a runtime restart), which is
the reference point for "what happened while I was away?", and tells the
notification policy that nothing can be shown right now, so news is queued for
the user's return instead of being "delivered" to nobody.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jarvis.clock import Clock, SystemClock
from jarvis.core.types import new_id
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.state.engine import StateEngine

AWAY_SINCE = "presence.away_since"          # set while no interface is attached
LAST_AWAY = "presence.last_away"             # {"from": ts, "until": ts}: the most recent completed absence
LAST_ATTACHED = "presence.last_attached_at"


@dataclass
class Client:
    id: str
    kind: str                  # cli | api | test ...
    session_id: str
    attached_at: float
    last_seen: float
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Return:
    """What an interface learns when it attaches."""
    client_id: str
    away_since: float | None   # None when another interface was already attached
    first: bool                # the first interface to attach since the user left


class Presence:
    def __init__(self, state: StateEngine, *, bus: EventBus | None = None, clock: Clock | None = None,
                 timeout_s: float = 30.0, default_since: float | None = None) -> None:
        self.state = state
        self.bus = bus
        self.clock = clock or SystemClock()
        self.timeout_s = timeout_s
        self.clients: dict[str, Client] = {}
        if self.state.value(AWAY_SINCE) is None:
            # first run, or the previous runtime stopped while an interface was attached: the absence starts
            # when that runtime was last seen alive (or now)
            self._set_away(default_since or self.clock.now())

    @property
    def away(self) -> bool:
        return not self.clients

    def away_since(self) -> float | None:
        value = self.state.value(AWAY_SINCE)
        return float(value) if isinstance(value, (int, float)) else None

    def attach(self, kind: str = "cli", session_id: str = "default", client_id: str | None = None,
               info: dict[str, Any] | None = None) -> Return:
        now = self.clock.now()
        if client_id and client_id in self.clients:
            self.clients[client_id].last_seen = now
            return Return(client_id, None, False)
        first = not self.clients
        since = self.away_since() if first else None
        cid = client_id or new_id("ui")
        self.clients[cid] = Client(cid, kind, session_id, now, now, dict(info or {}))
        self.state.set(LAST_ATTACHED, now)
        if first:
            if since is not None:
                self.state.set(LAST_AWAY, {"from": since, "until": now})
            self.state.delete(AWAY_SINCE)
        if self.bus:
            self.bus.emit(Event(EventType.UI_ATTACHED, "presence",
                                {"client": cid, "kind": kind, "session": session_id, "away_since": since,
                                 "attached": len(self.clients)}))
        return Return(cid, since, first)

    def touch(self, client_id: str) -> bool:
        client = self.clients.get(client_id)
        if client is None:
            return False
        client.last_seen = self.clock.now()
        return True

    def detach(self, client_id: str, reason: str = "closed") -> bool:
        client = self.clients.pop(client_id, None)
        if client is None:
            return False
        if not self.clients:
            self._set_away(self.clock.now())
            if self.bus:
                self.bus.emit(Event(EventType.UI_DETACHED, "presence",
                                    {"client": client_id, "kind": client.kind, "reason": reason}))
        return True

    def expire(self) -> list[str]:
        """Interfaces that stopped checking in (closed window, killed terminal) count as gone."""
        now = self.clock.now()
        stale = [c.id for c in self.clients.values() if now - c.last_seen > self.timeout_s]
        for cid in stale:
            self.detach(cid, reason="timed out")
        return stale

    def away_window(self) -> tuple[float | None, float | None]:
        """(from, until) of the current absence (until=None) or, while attached, of the last one."""
        if self.away:
            return self.away_since(), None
        last = self.state.value(LAST_AWAY)
        if isinstance(last, dict) and isinstance(last.get("from"), (int, float)):
            return float(last["from"]), float(last["until"]) if last.get("until") else None
        return None, None

    def _set_away(self, ts: float) -> None:
        self.state.set(AWAY_SINCE, ts)

    def snapshot(self) -> dict[str, Any]:
        return {"away": self.away, "away_since": self.away_since() if self.away else None,
                "clients": [{"id": c.id, "kind": c.kind, "session": c.session_id, "attached_at": c.attached_at,
                             "last_seen": c.last_seen} for c in self.clients.values()]}
