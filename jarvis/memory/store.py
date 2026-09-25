"""Memory (spec §43-45, §89).

Several kinds of memory with explicit structure — not a transcript dump.
Retrieval combines full-text search (FTS5/BM25) with optional embedding
similarity, filtered by kind and project, weighted by recency and importance.
Memory is inspectable and deletable, and "don't remember this" is honoured.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Callable, Iterable

from jarvis.clock import Clock, SystemClock
from jarvis.core.types import Confidence, Provenance, ProvenanceKind, new_id
from jarvis.database.db import Database, dumps, loads
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType

Embedder = Callable[[list[str]], Awaitable[tuple[list[list[float]], str] | None]]

_STOPWORDS = frozenset("""a an and are as at be but by did do does for from had has have how i in is it its me my
of on or our so that the their them then there these they this to was we were what when where which who why will
with you your about into can could should would just remember forget know tell us""".split())


class MemoryKind(StrEnum):
    EPISODIC = "episodic"        # important past events
    SEMANTIC = "semantic"        # stable facts
    PROCEDURAL = "procedural"    # how the user likes things done
    PROJECT = "project"          # project-specific knowledge
    TASK = "task"                # history of long-running work
    PREFERENCE = "preference"    # user preferences


@dataclass
class MemoryItem:
    id: str
    kind: MemoryKind
    content: str
    subject: str | None = None
    project_id: str | None = None
    user_id: str | None = None
    tags: list[str] = field(default_factory=list)
    provenance: Provenance | None = None
    confidence: Confidence = Confidence.RETRIEVED
    importance: float = 0.5
    created_at: float = 0.0
    updated_at: float = 0.0
    last_accessed: float | None = None
    expires_at: float | None = None


@dataclass
class Retrieved:
    item: MemoryItem
    score: float
    matched: str          # fts | vector | both | recent


class MemoryStore:
    def __init__(self, db: Database, *, clock: Clock | None = None, embedder: Embedder | None = None,
                 bus: EventBus | None = None) -> None:
        self.db = db
        self.clock = clock or SystemClock()
        self.embedder = embedder
        self.bus = bus
        self.suppressed = False       # "don't remember this" — session-scoped
        self.last_stored_id: str | None = None

    # -- writes -----------------------------------------------------------------------
    async def remember(self, content: str, *, kind: MemoryKind = MemoryKind.SEMANTIC, subject: str | None = None,
                       project_id: str | None = None, user_id: str | None = "owner", tags: Iterable[str] = (),
                       provenance: Provenance | None = None, importance: float = 0.5,
                       expires_at: float | None = None, force: bool = False) -> MemoryItem | None:
        content = content.strip()
        if not content or (self.suppressed and not force):
            return None
        now = self.clock.now()
        existing = self.db.query_one(
            "SELECT * FROM memories WHERE kind=? AND content=? AND IFNULL(project_id,'')=IFNULL(?, '')",
            (kind.value, content, project_id))
        if existing:
            self.db.execute("UPDATE memories SET updated_at=?, importance=MAX(importance, ?) WHERE id=?",
                            (now, importance, existing["id"]))
            self.last_stored_id = existing["id"]
            return _row(existing)
        prov = provenance or Provenance(ProvenanceKind.USER_STATEMENT, "conversation")
        item = MemoryItem(new_id("mem"), kind, content, subject, project_id, user_id, list(tags), prov,
                          Confidence.RETRIEVED, importance, now, now, None, expires_at)
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO memories(id, kind, content, subject, project_id, user_id, tags, provenance, confidence, "
                "importance, created_at, updated_at, expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (item.id, kind.value, content, subject, project_id, user_id, dumps(item.tags), dumps(prov.to_dict()),
                 item.confidence.value, importance, now, now, expires_at))
            self.db.execute("INSERT INTO memories_fts(memory_id, content, subject, tags) VALUES(?,?,?,?)",
                            (item.id, content, subject or "", " ".join(item.tags)))
        await self._embed(item)
        self.last_stored_id = item.id
        if self.bus:
            self.bus.emit(Event(EventType.MEMORY_STORED, "memory", {"memory_id": item.id, "kind": kind.value,
                                                                    "project_id": project_id}))
        return item

    async def _embed(self, item: MemoryItem) -> None:
        if self.embedder is None:
            return
        try:
            result = await self.embedder([item.content])
        except Exception:
            return
        if not result:
            return
        vectors, model = result
        vec = vectors[0]
        self.db.execute("INSERT OR REPLACE INTO embeddings(memory_id, model, dim, vector) VALUES(?,?,?,?)",
                        (item.id, model, len(vec), struct.pack(f"{len(vec)}f", *vec)))

    def forget(self, memory_id: str | None = None, *, project_id: str | None = None, kind: MemoryKind | None = None,
               query: str | None = None) -> int:
        """Delete memories. Returns how many were removed."""
        ids: list[str] = []
        if memory_id:
            ids = [memory_id]
        else:
            clauses, params = [], []
            if project_id:
                clauses.append("project_id=?")
                params.append(project_id)
            if kind:
                clauses.append("kind=?")
                params.append(kind.value)
            if query:
                matched = [r.item.id for r in self._fts(query, limit=50)]
                if not matched:
                    return 0
                clauses.append(f"id IN ({','.join('?' * len(matched))})")
                params += matched
            if not clauses:
                raise ValueError("refusing to forget everything without an explicit scope")
            ids = [r["id"] for r in self.db.query(f"SELECT id FROM memories WHERE {' AND '.join(clauses)}", params)]
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        with self.db.transaction():
            n = self.db.execute(f"DELETE FROM memories WHERE id IN ({marks})", ids)
            self.db.execute(f"DELETE FROM memories_fts WHERE memory_id IN ({marks})", ids)
            self.db.execute(f"DELETE FROM embeddings WHERE memory_id IN ({marks})", ids)
        if self.last_stored_id in ids:
            self.last_stored_id = None
        if self.bus:
            self.bus.emit(Event(EventType.MEMORY_FORGOTTEN, "memory", {"count": n, "project_id": project_id}))
        return n

    def purge_expired(self) -> int:
        rows = self.db.query("SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?",
                             (self.clock.now(),))
        return sum(self.forget(r["id"]) for r in rows)

    # -- reads ------------------------------------------------------------------------
    def get(self, memory_id: str) -> MemoryItem | None:
        row = self.db.query_one("SELECT * FROM memories WHERE id=?", (memory_id,))
        return _row(row) if row else None

    def list(self, *, kind: MemoryKind | None = None, project_id: str | None = None, limit: int = 50) -> list[MemoryItem]:
        clauses, params = [], []
        if kind:
            clauses.append("kind=?")
            params.append(kind.value)
        if project_id:
            clauses.append("project_id=?")
            params.append(project_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return [_row(r) for r in self.db.query(f"SELECT * FROM memories {where} ORDER BY updated_at DESC LIMIT ?",
                                                (*params, limit))]

    def count(self, project_id: str | None = None) -> int:
        if project_id:
            row = self.db.query_one("SELECT COUNT(*) n FROM memories WHERE project_id=?", (project_id,))
        else:
            row = self.db.query_one("SELECT COUNT(*) n FROM memories")
        return int(row["n"]) if row else 0

    def _fts(self, query: str, *, kinds: Iterable[MemoryKind] | None = None, project_id: str | None = None,
             include_global: bool = True, limit: int = 20) -> list[Retrieved]:
        terms = _terms(query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"*' for t in terms)
        sql = ("SELECT m.*, bm25(memories_fts) AS rank FROM memories_fts JOIN memories m ON m.id = memories_fts.memory_id "
               "WHERE memories_fts MATCH ?")
        params: list[Any] = [match]
        kinds = list(kinds or [])
        if kinds:
            sql += f" AND m.kind IN ({','.join('?' * len(kinds))})"
            params += [k.value for k in kinds]
        if project_id:
            sql += " AND (m.project_id = ?" + (" OR m.project_id IS NULL)" if include_global else ")")
            params.append(project_id)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        rows = self.db.query(sql, params)
        return [Retrieved(_row(r), -float(r["rank"]), "fts") for r in rows]

    async def retrieve(self, query: str, *, kinds: Iterable[MemoryKind] | None = None, project_id: str | None = None,
                       include_global: bool = True, limit: int = 5) -> list[Retrieved]:
        kinds = list(kinds or [])
        now = self.clock.now()
        candidates: dict[str, Retrieved] = {}
        fts = self._fts(query, kinds=kinds, project_id=project_id, include_global=include_global, limit=limit * 4)
        top = max((r.score for r in fts), default=1.0) or 1.0
        terms = _terms(query)
        for r in fts:
            # BM25 is unstable on small personal corpora, so blend it with plain query-term coverage.
            words = set(re.findall(r"[a-z0-9]+", f"{r.item.content} {r.item.subject or ''} {' '.join(r.item.tags)}".lower()))
            coverage = sum(1 for t in terms if any(w.startswith(t) for w in words)) / len(terms) if terms else 0.0
            r.score = 0.6 * (0.5 * coverage + 0.5 * (r.score / top))
            candidates[r.item.id] = r
        qvec = None
        if self.embedder is not None:
            try:
                result = await self.embedder([query])
                qvec = result[0][0] if result else None
            except Exception:
                qvec = None
        if qvec is not None:
            for mem_id, vec in self._vectors(kinds, project_id, include_global):
                sim = _cosine(qvec, vec)
                if sim < 0.3:
                    continue
                if mem_id in candidates:
                    candidates[mem_id].score += 0.4 * sim
                    candidates[mem_id].matched = "both"
                else:
                    item = self.get(mem_id)
                    if item:
                        candidates[mem_id] = Retrieved(item, 0.4 * sim, "vector")
        results = []
        for r in candidates.values():
            if r.item.expires_at and r.item.expires_at < now:
                continue
            age_days = max(0.0, (now - r.item.updated_at) / 86400)
            r.score += 0.1 * math.exp(-age_days / 30) + 0.1 * r.item.importance
            if project_id and r.item.project_id == project_id:
                r.score += 0.3   # the active project's own knowledge is the most relevant context
            results.append(r)
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:limit]
        if results:
            marks = ",".join("?" * len(results))
            self.db.execute(f"UPDATE memories SET last_accessed=? WHERE id IN ({marks})",
                            (now, *[r.item.id for r in results]))
        return results

    search = retrieve

    def _vectors(self, kinds: list[MemoryKind], project_id: str | None,
                 include_global: bool) -> list[tuple[str, list[float]]]:
        sql = "SELECT e.memory_id, e.dim, e.vector FROM embeddings e JOIN memories m ON m.id = e.memory_id WHERE 1=1"
        params: list[Any] = []
        if kinds:
            sql += f" AND m.kind IN ({','.join('?' * len(kinds))})"
            params += [k.value for k in kinds]
        if project_id:
            sql += " AND (m.project_id = ?" + (" OR m.project_id IS NULL)" if include_global else ")")
            params.append(project_id)
        rows = self.db.query(sql + " LIMIT 5000", params)
        return [(r["memory_id"], list(struct.unpack(f"{r['dim']}f", r["vector"]))) for r in rows]


def _terms(query: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOPWORDS and len(w) > 1]


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _row(r: Any) -> MemoryItem:
    prov = loads(r["provenance"])
    return MemoryItem(r["id"], MemoryKind(r["kind"]), r["content"], r["subject"], r["project_id"], r["user_id"],
                      loads(r["tags"], []), Provenance.from_dict(prov) if prov else None,
                      Confidence(r["confidence"]) if r["confidence"] else Confidence.RETRIEVED, r["importance"],
                      r["created_at"], r["updated_at"], r["last_accessed"], r["expires_at"])
