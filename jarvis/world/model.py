"""World model: how everything relates (spec §7, §90).

Memory is the past, state is the present; the world model is the structure.
Entities (projects, services, processes, models, devices, tasks, people...) are
linked by typed relations, which lets JARVIS reason about systems — e.g. walk a
``depends_on`` chain to explain why a service is slow.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Iterable

from jarvis.clock import Clock, SystemClock
from jarvis.database.db import Database, dumps, loads

# Relation vocabulary (extensible; these are the ones core subsystems use).
DEPENDS_ON = "depends_on"
CONTAINS = "contains"
MODIFIES = "modifies"
RUNS_ON = "runs_on"
OWNS = "owns"
EXECUTING = "executing"
WAITING_ON = "waiting_on"
CONSUMING = "consuming"
LOADED_INTO = "loaded_into"
BELONGS_TO = "belongs_to"
MONITORS = "monitors"


@dataclass
class Entity:
    id: str
    type: str
    name: str
    attrs: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class Relation:
    src: str
    rel: str
    dst: str
    attrs: dict[str, Any] = field(default_factory=dict)


def entity_id(type_: str, name: str) -> str:
    return f"{type_}:{name}"


class WorldModel:
    def __init__(self, db: Database, clock: Clock | None = None) -> None:
        self.db = db
        self.clock = clock or SystemClock()

    # -- entities ---------------------------------------------------------------
    def upsert_entity(self, type_: str, name: str, attrs: dict[str, Any] | None = None, *,
                      id: str | None = None, source: str | None = None, merge: bool = True) -> Entity:
        eid = id or entity_id(type_, name)
        now = self.clock.now()
        with self.db.transaction():
            existing = self.get_entity(eid)
            merged = dict(existing.attrs) if (existing and merge) else {}
            merged.update(attrs or {})
            if existing:
                self.db.execute("UPDATE entities SET type=?, name=?, attrs=?, source=?, updated_at=? WHERE id=?",
                                (type_, name, dumps(merged), source or existing.source, now, eid))
                created = existing.created_at
            else:
                self.db.execute("INSERT INTO entities(id, type, name, attrs, source, created_at, updated_at) "
                                "VALUES(?,?,?,?,?,?,?)", (eid, type_, name, dumps(merged), source, now, now))
                created = now
        return Entity(eid, type_, name, merged, source, created, now)

    # Spec §169 name.
    update_entity = upsert_entity

    def get_entity(self, eid: str) -> Entity | None:
        row = self.db.query_one("SELECT * FROM entities WHERE id=?", (eid,))
        return _row_to_entity(row) if row else None

    def find_entities(self, type_: str | None = None, name: str | None = None,
                      where: dict[str, Any] | None = None, limit: int = 200) -> list[Entity]:
        clauses, params = [], []
        if type_:
            clauses.append("type=?")
            params.append(type_)
        if name:
            clauses.append("name=? COLLATE NOCASE")
            params.append(name)
        sql = "SELECT * FROM entities" + (f" WHERE {' AND '.join(clauses)}" if clauses else "")
        sql += " ORDER BY updated_at DESC LIMIT ?"
        entities = [_row_to_entity(r) for r in self.db.query(sql, (*params, limit))]
        if where:
            entities = [e for e in entities if all(e.attrs.get(k) == v for k, v in where.items())]
        return entities

    def delete_entity(self, eid: str) -> None:
        with self.db.transaction():
            self.db.execute("DELETE FROM relations WHERE src=? OR dst=?", (eid, eid))
            self.db.execute("DELETE FROM entities WHERE id=?", (eid,))

    # -- relations --------------------------------------------------------------
    def relate(self, src: str, rel: str, dst: str, attrs: dict[str, Any] | None = None) -> Relation:
        self.db.execute(
            "INSERT INTO relations(src, rel, dst, attrs, updated_at) VALUES(?,?,?,?,?) "
            "ON CONFLICT(src, rel, dst) DO UPDATE SET attrs=excluded.attrs, updated_at=excluded.updated_at",
            (src, rel, dst, dumps(attrs or {}), self.clock.now()),
        )
        return Relation(src, rel, dst, attrs or {})

    def unrelate(self, src: str, rel: str | None = None, dst: str | None = None) -> int:
        clauses, params = ["src=?"], [src]
        if rel:
            clauses.append("rel=?")
            params.append(rel)
        if dst:
            clauses.append("dst=?")
            params.append(dst)
        return self.db.execute(f"DELETE FROM relations WHERE {' AND '.join(clauses)}", params)

    def relations(self, eid: str, rel: str | None = None, direction: str = "out") -> list[Relation]:
        out: list[Relation] = []
        if direction in ("out", "both"):
            sql, params = "SELECT * FROM relations WHERE src=?", [eid]
            if rel:
                sql += " AND rel=?"
                params.append(rel)
            out += [Relation(r["src"], r["rel"], r["dst"], loads(r["attrs"], {})) for r in self.db.query(sql, params)]
        if direction in ("in", "both"):
            sql, params = "SELECT * FROM relations WHERE dst=?", [eid]
            if rel:
                sql += " AND rel=?"
                params.append(rel)
            out += [Relation(r["src"], r["rel"], r["dst"], loads(r["attrs"], {})) for r in self.db.query(sql, params)]
        return out

    def find_related(self, eid: str, rel: str | None = None, direction: str = "out") -> list[tuple[str, Entity]]:
        result = []
        for r in self.relations(eid, rel, direction):
            other = r.dst if r.src == eid else r.src
            entity = self.get_entity(other)
            if entity:
                result.append((r.rel, entity))
        return result

    def chain(self, eid: str, rel: str = DEPENDS_ON, max_depth: int = 8) -> list[Entity]:
        """Follow a relation transitively (cycle-safe). Used for causal explanations (spec §66)."""
        seen: set[str] = {eid}
        path: list[Entity] = []
        frontier = [eid]
        for _ in range(max_depth):
            nxt = []
            for node in frontier:
                for r in self.relations(node, rel, "out"):
                    if r.dst not in seen:
                        seen.add(r.dst)
                        entity = self.get_entity(r.dst)
                        if entity:
                            path.append(entity)
                            nxt.append(r.dst)
            if not nxt:
                break
            frontier = nxt
        return path

    def query_state(self, type_: str) -> dict[str, dict[str, Any]]:
        return {e.name: e.attrs for e in self.find_entities(type_)}

    def resolve_name(self, text: str, types: Iterable[str] | None = None, cutoff: float = 0.6) -> list[Entity]:
        """Fuzzy match a spoken name (e.g. "the robotics project") against known entities."""
        candidates = []
        for t in (list(types) if types else [None]):
            candidates += self.find_entities(t, limit=1000)
        needle = text.lower().strip()
        exact = [e for e in candidates if e.name.lower() == needle]
        if exact:
            return exact
        contains = [e for e in candidates if needle and (needle in e.name.lower() or e.name.lower() in needle)]
        if contains:
            return contains
        names = [e.name.lower() for e in candidates]
        close = set(difflib.get_close_matches(needle, names, n=5, cutoff=cutoff))
        return [e for e in candidates if e.name.lower() in close]


def _row_to_entity(row: Any) -> Entity:
    return Entity(row["id"], row["type"], row["name"], loads(row["attrs"], {}), row["source"],
                  row["created_at"], row["updated_at"])
