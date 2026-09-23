"""In-process asynchronous event bus.

Components publish structured events and subscribe to the ones they care about
instead of calling each other directly (spec §9). Handler failures are isolated:
one broken subscriber never prevents delivery to others or crashes the publisher.
"""

from __future__ import annotations

import asyncio
import inspect
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from jarvis.clock import Clock, SystemClock
from jarvis.core.types import Severity
from jarvis.events.store import EventStore
from jarvis.events.types import Event
from jarvis.log import get_logger

Handler = Callable[[Event], Awaitable[None] | None]

log = get_logger("events")


@dataclass
class Subscription:
    pattern: str
    handler: Handler
    name: str

    def matches(self, event_type: str) -> bool:
        if self.pattern == "*":
            return True
        if self.pattern.endswith("*"):
            return event_type.startswith(self.pattern[:-1])
        return event_type == self.pattern


class EventBus:
    def __init__(self, store: EventStore | None = None, clock: Clock | None = None,
                 recent_size: int = 500, handler_timeout_s: float = 10.0) -> None:
        self.store = store
        self.clock = clock or SystemClock()
        self.recent: deque[Event] = deque(maxlen=recent_size)
        self._subs: list[Subscription] = []
        self._pending: set[asyncio.Task[Any]] = set()
        self.handler_timeout_s = handler_timeout_s
        self.handler_errors = 0
        self.published = 0
        self.persist_failures = 0

    def subscribe(self, pattern: str, handler: Handler, *, name: str | None = None) -> Subscription:
        sub = Subscription(str(pattern), handler, name or getattr(handler, "__qualname__", "handler"))
        self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)

    async def publish(self, event: Event) -> Event:
        if not event.ts:
            event.ts = self.clock.now()
        self.published += 1
        self.recent.append(event)
        if self.store is not None and event.should_persist():
            try:
                self.store.append(event)
            except Exception as exc:  # persistence failure must not stop live delivery
                self.persist_failures += 1
                log.error("event_persist_failed", event_type=str(event.type), error=str(exc))
        if event.severity >= Severity.WARNING:
            log.warning("event", event_type=str(event.type), source=event.source, task_id=event.task_id,
                        payload=event.payload)
        targets = [s for s in list(self._subs) if s.matches(str(event.type))]
        if targets:
            await asyncio.gather(*(self._deliver(s, event) for s in targets))
        return event

    def emit(self, event: Event) -> None:
        """Fire-and-forget publish for synchronous call sites. Use :meth:`drain` to wait."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop (e.g. during synchronous shutdown): persist and buffer only.
            if not event.ts:
                event.ts = self.clock.now()
            self.recent.append(event)
            if self.store is not None and event.should_persist():
                self.store.append(event)
            return
        task = loop.create_task(self.publish(event))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def drain(self) -> None:
        while self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    async def _deliver(self, sub: Subscription, event: Event) -> None:
        try:
            result = sub.handler(event)
            if inspect.isawaitable(result):
                await asyncio.wait_for(result, timeout=self.handler_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.handler_errors += 1
            log.error("event_handler_failed", handler=sub.name, event_type=str(event.type), error=repr(exc))

    def recent_events(self, limit: int = 50, min_severity: Severity = Severity.DEBUG,
                      types: set[str] | None = None) -> list[Event]:
        items = [e for e in self.recent if e.severity >= min_severity and (types is None or e.type in types)]
        return items[-limit:]
