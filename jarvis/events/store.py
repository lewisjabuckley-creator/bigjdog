"""Persistent event store with retention (spec §101)."""

from __future__ import annotations

from typing import Iterable

from jarvis.config import EventsConfig
from jarvis.core.types import Severity
from jarvis.database.db import Database, dumps, loads
from jarvis.events.types import Event

DAY = 86_400.0


class EventStore:
    def __init__(self, db: Database, config: EventsConfig | None = None) -> None:
        self.db = db
        self.config = config or EventsConfig()

    def append(self, event: Event) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO events(id, ts, type, source, severity, entity_id, task_id, payload) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (event.id, event.ts, str(event.type), event.source, int(event.severity), event.entity_id,
             event.task_id, dumps(event.payload)),
        )

    def query(
        self,
        *,
        since: float | None = None,
        until: float | None = None,
        types: Iterable[str] | None = None,
        task_id: str | None = None,
        entity_id: str | None = None,
        min_severity: Severity | None = None,
        limit: int = 200,
        newest_first: bool = True,
    ) -> list[Event]:
        clauses, params = [], []
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        if until is not None:
            clauses.append("ts <= ?")
            params.append(until)
        if types:
            types = list(types)
            clauses.append(f"type IN ({','.join('?' * len(types))})")
            params.extend(str(t) for t in types)
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        if entity_id:
            clauses.append("entity_id = ?")
            params.append(entity_id)
        if min_severity is not None:
            clauses.append("severity >= ?")
            params.append(int(min_severity))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order = "DESC" if newest_first else "ASC"
        rows = self.db.query(f"SELECT * FROM events {where} ORDER BY ts {order}, rowid {order} LIMIT ?",
                             (*params, limit))
        return [
            Event(type=r["type"], source=r["source"], payload=loads(r["payload"], {}),
                  severity=Severity(r["severity"]), entity_id=r["entity_id"], task_id=r["task_id"],
                  ts=r["ts"], id=r["id"], persist=True)
            for r in rows
        ]

    def count(self) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM events")
        return int(row["n"]) if row else 0

    def prune(self, now: float) -> int:
        """Apply retention: debug events are short-lived, warnings and above are kept longest."""
        c = self.config
        removed = 0
        removed += self.db.execute("DELETE FROM events WHERE severity < ? AND ts < ?",
                                   (int(Severity.INFO), now - c.retention_days_debug * DAY))
        removed += self.db.execute("DELETE FROM events WHERE severity >= ? AND severity < ? AND ts < ?",
                                   (int(Severity.INFO), int(Severity.WARNING), now - c.retention_days_info * DAY))
        removed += self.db.execute("DELETE FROM events WHERE severity >= ? AND ts < ?",
                                   (int(Severity.WARNING), now - c.retention_days_warning * DAY))
        return removed
