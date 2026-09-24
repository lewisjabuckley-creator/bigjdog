"""Inference scheduling (Phase 3 §12).

A local machine runs one or two model requests well and many badly. The scheduler caps concurrent inference and
serves waiting requests by priority, so the user's conversation is never stuck behind background planning or
agents. Under resource pressure the cap drops to one. It changes only *when* a request runs, never which model
answers or whether it is allowed.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable


class InferenceScheduler:
    def __init__(self, max_concurrent: int = 2, *, pressure: Callable[[], tuple[bool, str]] | None = None) -> None:
        self.max_concurrent = max(1, max_concurrent)
        self.pressure = pressure or (lambda: (False, ""))
        self.active = 0
        self._waiting: list[tuple[int, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()
        self.served = 0
        self.max_wait_s = 0.0

    @property
    def limit(self) -> int:
        constrained, _ = self.pressure()
        return 1 if constrained else self.max_concurrent

    @property
    def queued(self) -> int:
        return sum(1 for _, _, f in self._waiting if not f.done())

    async def acquire(self, priority: int = 1) -> None:
        """Wait for a slot. Lower numbers run first: 0 = the user is waiting, 2+ = background work."""
        if self.active < self.limit and not self.queued:
            self.active += 1
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiting, (priority, next(self._seq), future))
        started = time.monotonic()
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self.release()              # the slot was handed over just as the caller gave up
            raise
        self.max_wait_s = max(self.max_wait_s, time.monotonic() - started)

    def release(self) -> None:
        self.active = max(0, self.active - 1)
        self.served += 1
        while self._waiting and self.active < self.limit:
            _, _, future = heapq.heappop(self._waiting)
            if future.done():
                continue
            self.active += 1
            future.set_result(None)

    @asynccontextmanager
    async def slot(self, priority: int = 1) -> AsyncIterator[None]:
        await self.acquire(priority)
        try:
            yield
        finally:
            self.release()

    def status(self) -> dict[str, float | int]:
        return {"active": self.active, "queued": self.queued, "limit": self.limit, "served": self.served,
                "max_wait_s": round(self.max_wait_s, 2)}
