"""Conversation turns for interfaces attached to the runtime.

A turn runs inside the runtime, not the interface: if the window closes mid-
answer, the turn (and any task it starts) carries on, and its response is kept.
Each turn has a request id chosen by the interface; sending the same id again
(a retry after a dropped connection, a reconnect after a restart) returns the
original response instead of handling the message twice, so nothing a message
started is ever duplicated.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

from jarvis.core.types import new_id
from jarvis.database.db import dumps, loads
from jarvis.log import get_logger

if TYPE_CHECKING:
    from jarvis.runtime import Runtime

log = get_logger("conversations")


def _error(text: str) -> dict[str, Any]:
    return {"text": text, "intent": "chat", "kind": "error", "task_id": None, "approval_id": None, "provenance": [],
            "notifications": [], "model": None, "data": {}, "streamed": False, "footnote": "", "rendered": text,
            "extra": ""}


class Conversations:
    def __init__(self, runtime: "Runtime") -> None:
        self.runtime = runtime
        self._locks: dict[str, asyncio.Lock] = {}
        self._inflight: dict[str, asyncio.Task[dict[str, Any]]] = {}

    async def handle(self, text: str, *, session: str = "default", request_id: str | None = None,
                     cwd: str | None = None, on_token: Callable[[str], None] | None = None,
                     attachments: list[Any] | None = None) -> tuple[str, dict[str, Any], bool]:
        """Returns (request id, response, replayed)."""
        svc = self.runtime.svc
        assert svc is not None
        rid = request_id or new_id("req")
        row = svc.db.query_one("SELECT status, response FROM requests WHERE id=?", (rid,))
        if row is not None:
            if row["status"] == "done":
                return rid, loads(row["response"], {}) or _error("(no stored response)"), True
            running = self._inflight.get(rid)
            if running is not None:
                return rid, await asyncio.shield(running), True
            return rid, _error("That message was being handled when JARVIS stopped, so I can't tell how far it got. "
                               "Ask \"what are you doing?\" to see any task it started."), True
        svc.db.execute("INSERT INTO requests(id, session_id, text, status, created_at) VALUES(?,?,?,?,?)",
                       (rid, session, "" if svc.memory.suppressed else text[:2000], "running", svc.clock.now()))
        task = asyncio.create_task(self._run(rid, text, session, cwd, on_token, attachments), name=f"turn-{rid}")
        self._inflight[rid] = task
        # shielded: the turn finishes even if the interface disconnects while waiting
        return rid, await asyncio.shield(task), False

    async def _run(self, rid: str, text: str, session: str, cwd: str | None,
                   on_token: Callable[[str], None] | None, attachments: list[Any] | None = None) -> dict[str, Any]:
        svc = self.runtime.svc
        assert svc is not None
        lock = self._locks.setdefault(session, asyncio.Lock())
        try:
            async with lock:       # one turn at a time per conversation, whichever interface sent it
                orch = self.runtime.orchestrator(session)

                def sink(piece: str) -> None:
                    if on_token is None:
                        return
                    try:
                        on_token(piece)
                    except Exception:      # the interface went away; the turn continues
                        pass

                try:
                    response = (await orch.handle(text, on_token=sink if on_token else None, cwd=cwd,
                                                  attachments=attachments)).to_dict()
                except Exception as exc:
                    log.error("turn_failed", request=rid, error=repr(exc))
                    response = _error(f"Something went wrong while handling that ({exc}). It's logged; nothing else "
                                      "was affected.")
            stored = response if not svc.memory.suppressed else _error("(off the record: not stored)")
            svc.db.execute("UPDATE requests SET status='done', response=?, finished_at=? WHERE id=?",
                           (dumps(stored), svc.clock.now(), rid))
            return response
        finally:
            self._inflight.pop(rid, None)

    def prune(self, older_than_s: float = 7 * 86400) -> int:
        svc = self.runtime.svc
        assert svc is not None
        return svc.db.execute("DELETE FROM requests WHERE status='done' AND created_at < ?",
                              (svc.clock.now() - older_than_s,))
