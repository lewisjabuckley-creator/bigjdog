"""Time source abstraction.

JARVIS reasons about time deterministically (spec §141): durations, deadlines and
schedules come from this clock, never from model guesswork. Tests and the
simulation environment use :class:`FakeClock` to control time explicitly.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Wall-clock time as UTC epoch seconds."""

    def monotonic(self) -> float:
        """Monotonic seconds for measuring durations."""

    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class FakeClock:
    """Manually advanced clock. ``sleep`` yields to the loop and advances time."""

    def __init__(self, start: float = 1_750_000_000.0) -> None:
        self._now = start
        self._mono = 0.0

    def now(self) -> float:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._now += seconds
        self._mono += seconds

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)
        await asyncio.sleep(0)


def to_local(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()


def format_time(ts: float) -> str:
    return to_local(ts).strftime("%H:%M")


def format_datetime(ts: float) -> str:
    return to_local(ts).strftime("%Y-%m-%d %H:%M")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        text = f"{hours} hour{'s' if hours != 1 else ''}"
        if minutes:
            text += f" {minutes} minute{'s' if minutes != 1 else ''}"
        return text
    days = hours // 24
    return f"{days} days"
