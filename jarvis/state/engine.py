"""Live state: what is happening *now* (spec §8, §139).

State is distinct from memory. Every value is a :class:`Fact` carrying its
confidence, provenance and observation time, so JARVIS can answer "how do you
know?" and can tell when a value has gone stale. Critical state lives here — in
structured storage — never only inside a prompt.
"""

from __future__ import annotations

from typing import Any, Callable

from jarvis.clock import Clock, SystemClock
from jarvis.core.types import Confidence, Fact, Provenance, ProvenanceKind
from jarvis.database.db import Database, dumps, loads

ChangeCallback = Callable[[str, Fact | None, Fact | None], None]

_DEFAULT_PROVENANCE = Provenance(ProvenanceKind.SYSTEM_STATE, "jarvis")


class StateEngine:
    def __init__(self, db: Database | None = None, clock: Clock | None = None) -> None:
        self.db = db
        self.clock = clock or SystemClock()
        self._facts: dict[str, Fact] = {}
        self._persisted: set[str] = set()
        self._callbacks: list[tuple[str, ChangeCallback]] = []

    # -- writes -----------------------------------------------------------------
    def set(
        self,
        key: str,
        value: Any,
        *,
        confidence: Confidence = Confidence.OBSERVED,
        provenance: Provenance | None = None,
        ttl: float | None = None,
        persist: bool = True,
    ) -> Fact:
        old = self._facts.get(key)
        fact = Fact(value, confidence, provenance or _DEFAULT_PROVENANCE, self.clock.now(), ttl)
        self._facts[key] = fact
        if persist and self.db is not None:
            self._persisted.add(key)
            self.db.execute(
                "INSERT INTO state(key, fact, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET fact=excluded.fact, updated_at=excluded.updated_at",
                (key, dumps(fact.to_dict()), fact.observed_at),
            )
        if old is None or old.value != value:
            self._notify(key, old, fact)
        return fact

    def delete(self, key: str) -> None:
        old = self._facts.pop(key, None)
        if self.db is not None:
            self.db.execute("DELETE FROM state WHERE key=?", (key,))
        self._persisted.discard(key)
        if old is not None:
            self._notify(key, old, None)

    # -- reads ------------------------------------------------------------------
    def get(self, key: str) -> Fact | None:
        return self._facts.get(key)

    def value(self, key: str, default: Any = None, *, allow_stale: bool = True) -> Any:
        fact = self._facts.get(key)
        if fact is None:
            return default
        if not allow_stale and fact.is_stale(self.clock.now()):
            return default
        return fact.value

    def is_stale(self, key: str) -> bool:
        fact = self._facts.get(key)
        return fact is None or fact.is_stale(self.clock.now())

    def snapshot(self, prefix: str = "") -> dict[str, Fact]:
        return {k: v for k, v in sorted(self._facts.items()) if k.startswith(prefix)}

    def values(self, prefix: str = "") -> dict[str, Any]:
        return {k: f.value for k, f in self.snapshot(prefix).items()}

    # -- lifecycle --------------------------------------------------------------
    def restore(self) -> int:
        """Reload persisted state after a restart. Values keep their original observation time,
        so anything with a TTL is correctly reported as stale until re-observed."""
        if self.db is None:
            return 0
        count = 0
        for row in self.db.query("SELECT key, fact FROM state"):
            try:
                self._facts[row["key"]] = Fact.from_dict(loads(row["fact"]))
                self._persisted.add(row["key"])
                count += 1
            except (KeyError, ValueError, TypeError):
                continue
        return count

    def on_change(self, prefix: str, callback: ChangeCallback) -> None:
        self._callbacks.append((prefix, callback))

    def _notify(self, key: str, old: Fact | None, new: Fact | None) -> None:
        for prefix, callback in self._callbacks:
            if key.startswith(prefix):
                try:
                    callback(key, old, new)
                except Exception:
                    pass
