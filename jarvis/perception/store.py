"""Observations: what JARVIS was shown, stored once, analysed on demand, forgotten on request (Phase 4 §6, §41).

An input is validated before anything else happens (type, size, integrity, whether the path may be read), then its
bytes are stored content-addressed (the same screenshot sent twice is stored once and never analysed twice) and a
row records the observation: kind, origin, session, project, ordinal, sensitivity, labels and derived results.

Derived results (OCR text, a description, UI elements, a document's structure) live on the observation and in a
cache keyed by content hash and operation, so repeating a question costs nothing. Retention deletes stored copies
after a while (screen captures sooner) and keeps what was learned; "forget that screenshot" deletes both.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

from jarvis.clock import Clock, SystemClock
from jarvis.core.types import Severity, new_id
from jarvis.database.db import Database, dumps, loads
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.perception import images
from jarvis.perception.inputs import (Attachment, InputKind, InputOrigin, Observation, Sensitivity, Status,
                                      classify)


class InputRejected(ValueError):
    """An input JARVIS won't take in, with the reason in words."""


_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/bmp": ".bmp", "image/webp": ".webp",
        "image/tiff": ".tif", "application/pdf": ".pdf"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PerceptionStore:
    def __init__(self, db: Database, root: str | os.PathLike[str], *, clock: Clock | None = None,
                 bus: EventBus | None = None, max_bytes: int = 40_000_000, retention_s: float = 7 * 86400,
                 screen_retention_s: float = 86400,
                 path_check: Callable[[str], tuple[bool, str]] | None = None) -> None:
        self.db = db
        self.root = Path(root)
        self.clock = clock or SystemClock()
        self.bus = bus
        self.max_bytes = max_bytes
        self.retention_s = retention_s
        self.screen_retention_s = screen_retention_s
        # denied paths (keys, credentials) stay denied even when the user names a file inside them
        self.path_check = path_check or (lambda p: (True, ""))

    # -- taking inputs in ------------------------------------------------------------------------------------
    def ingest(self, attachment: Attachment, *, session_id: str | None = None, project_id: str | None = None,
               sensitivity: Sensitivity | None = None) -> Observation:
        """Validate and record one input. Raises InputRejected with a user-facing reason."""
        if attachment.observation_id:
            existing = self.get(attachment.observation_id)
            if existing is None:
                raise InputRejected(f"there's no input called {attachment.observation_id}")
            return existing
        data, name, source_path = self._read(attachment)
        if len(data) > self.max_bytes:
            self._reject(name, f"it's {len(data) / 1e6:.1f} MB, over the {self.max_bytes / 1e6:.0f} MB limit")
        if not data:
            self._reject(name, "it's empty")
        digest = sha256(data)
        fmt = images.sniff(data)
        width = height = None
        mime = ""
        if fmt is not None:
            try:
                info = images.image_info(data)
            except images.ImageError as exc:
                self._reject(name, str(exc))
            width, height, mime = info.width, info.height, info.mime
        elif data.startswith(b"%PDF"):
            mime = "application/pdf"
        else:
            mime = _guess_text_mime(name, data)
        if fmt and (attachment.kind == InputKind.SCREENSHOT or attachment.origin == InputOrigin.SCREEN_CAPTURE):
            kind = InputKind.SCREENSHOT
        else:
            kind = classify(name, mime, width, height)
        if kind == InputKind.FILE and attachment.kind in (InputKind.DOCUMENT,):
            kind = InputKind.DOCUMENT
        if kind == InputKind.FILE and fmt is None and not mime.startswith("text/"):
            self._reject(name, "I can read images, PDFs and text-based files (documents, code, logs, data), but not "
                               "this kind of file")
        origin = attachment.origin
        if sensitivity is None:
            sensitivity = Sensitivity.PRIVATE if origin == InputOrigin.SCREEN_CAPTURE else Sensitivity.NORMAL
        stored = self._store_bytes(digest, data, _EXT.get(mime) or os.path.splitext(name)[1].lower() or ".bin")
        now = self.clock.now()
        keep = self.screen_retention_s if origin == InputOrigin.SCREEN_CAPTURE else self.retention_s
        obs = Observation(new_id("obs"), kind, origin, name, now, session_id=session_id, project_id=project_id,
                          mime=mime, size_bytes=len(data), sha256=digest, path=str(stored), source_path=source_path,
                          width=width, height=height, ordinal=self._next_ordinal(session_id, kind),
                          sensitivity=sensitivity, updated_at=now, expires_at=now + keep if keep else None)
        self._insert(obs)
        self._emit(EventType.INPUT_RECEIVED, {"observation_id": obs.id, "kind": kind.value, "origin": origin.value,
                                              "name": name, "size_bytes": len(data), "width": width,
                                              "height": height, "session_id": session_id,
                                              "sensitivity": sensitivity.value})
        return obs

    def _read(self, a: Attachment) -> tuple[bytes, str, str | None]:
        if a.data is not None:
            return a.data, a.name or "upload", None
        if not a.path:
            raise InputRejected("nothing was attached")
        path = os.path.realpath(os.path.expanduser(a.path))
        name = os.path.basename(path)
        if not os.path.isfile(path):
            self._reject(name, f"{a.path} isn't a file I can find")
        ok, why = self.path_check(path)
        if not ok:
            self._reject(name, why)
        size = os.path.getsize(path)
        if size > self.max_bytes:
            self._reject(name, f"it's {size / 1e6:.1f} MB, over the {self.max_bytes / 1e6:.0f} MB limit")
        try:
            with open(path, "rb") as fh:
                return fh.read(), name, path
        except OSError as exc:
            self._reject(name, f"it couldn't be read ({exc.strerror or exc})")
        raise AssertionError("unreachable")

    def _reject(self, name: str, reason: str) -> None:
        self._emit(EventType.INPUT_REJECTED, {"name": name, "reason": reason}, Severity.WARNING)
        raise InputRejected(f"I couldn't take in {name}: {reason}")

    def _store_bytes(self, digest: str, data: bytes, ext: str) -> Path:
        folder = self.root / "objects" / digest[:2]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{digest}{ext if len(ext) <= 8 else ''}"
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        return path

    def _next_ordinal(self, session_id: str | None, kind: InputKind) -> int:
        row = self.db.query_one("SELECT MAX(ordinal) AS n FROM observations WHERE kind=? AND "
                                "(session_id=? OR (session_id IS NULL AND ? IS NULL))",
                                (kind.value, session_id, session_id))
        return int((row["n"] if row and row["n"] is not None else 0)) + 1

    # -- reading and updating --------------------------------------------------------------------------------
    def get(self, obs_id: str) -> Observation | None:
        row = self.db.query_one("SELECT * FROM observations WHERE id=?", (obs_id,))
        return _row(row) if row else None

    def data(self, obs: Observation) -> bytes:
        if not obs.available:
            raise InputRejected(f"I no longer have the {obs.handle} itself (stored copies are kept for a while and "
                                "then deleted); send it again if you'd like me to look at it")
        with open(obs.path, "rb") as fh:          # type: ignore[arg-type]
            return fh.read()

    def recent(self, session_id: str | None = None, *, kinds: Iterable[InputKind] | None = None, limit: int = 20,
               since: float | None = None, until: float | None = None) -> list[Observation]:
        """Newest first."""
        clauses, params = [], []
        if session_id is not None:
            clauses.append("session_id=?")
            params.append(session_id)
        if kinds:
            kinds = list(kinds)
            clauses.append(f"kind IN ({','.join('?' * len(kinds))})")
            params += [k.value for k in kinds]
        if since is not None:
            clauses.append("created_at>=?")
            params.append(since)
        if until is not None:
            clauses.append("created_at<?")
            params.append(until)
        sql = "SELECT * FROM observations" + (f" WHERE {' AND '.join(clauses)}" if clauses else "")
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        return [_row(r) for r in self.db.query(sql, (*params, limit))]

    def update(self, obs: Observation) -> Observation:
        obs.updated_at = self.clock.now()
        self.db.execute("UPDATE observations SET status=?, error=?, sensitivity=?, labels=?, derived=?, pages=?, "
                        "path=?, updated_at=?, expires_at=? WHERE id=?",
                        (obs.status.value, obs.error, obs.sensitivity.value, dumps(obs.labels), dumps(obs.derived),
                         obs.pages, obs.path, obs.updated_at, obs.expires_at, obs.id))
        return obs

    def _insert(self, o: Observation) -> None:
        self.db.execute(
            "INSERT INTO observations(id, kind, origin, session_id, project_id, name, mime, size_bytes, sha256, path, "
            "source_path, width, height, pages, ordinal, status, error, sensitivity, labels, derived, created_at, "
            "updated_at, expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (o.id, o.kind.value, o.origin.value, o.session_id, o.project_id, o.name, o.mime, o.size_bytes, o.sha256,
             o.path, o.source_path, o.width, o.height, o.pages, o.ordinal, o.status.value, o.error,
             o.sensitivity.value, dumps(o.labels), dumps(o.derived), o.created_at, o.updated_at, o.expires_at))

    # -- the cache of derived results -------------------------------------------------------------------------
    @staticmethod
    def _key(digest: str, op: str, params: dict[str, Any] | None) -> str:
        blob = json.dumps(params or {}, sort_keys=True, default=str)
        return hashlib.sha256(f"{digest}|{op}|{blob}".encode()).hexdigest()

    def cached(self, digest: str, op: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT result FROM perception_cache WHERE key=?", (self._key(digest, op, params),))
        return loads(row["result"], None) if row else None

    def cache(self, digest: str, op: str, result: dict[str, Any], params: dict[str, Any] | None = None, *,
              model: str | None = None) -> None:
        self.db.execute("INSERT OR REPLACE INTO perception_cache(key, sha256, op, result, model, created_at) "
                        "VALUES(?,?,?,?,?,?)", (self._key(digest, op, params), digest, op, dumps(result), model,
                                                self.clock.now()))

    # -- retention and forgetting ------------------------------------------------------------------------------
    def forget(self, obs_id: str, *, by: str = "user") -> Observation | None:
        """Delete an observation, its stored copy (unless another observation shares it) and what was derived."""
        obs = self.get(obs_id)
        if obs is None:
            return None
        self.db.execute("DELETE FROM observations WHERE id=?", (obs_id,))
        if obs.sha256 and not self.db.query_one("SELECT 1 FROM observations WHERE sha256=?", (obs.sha256,)):
            self.db.execute("DELETE FROM perception_cache WHERE sha256=?", (obs.sha256,))
            self._delete_file(obs.path)
        self._emit(EventType.INPUT_FORGOTTEN, {"observation_id": obs_id, "kind": obs.kind.value, "by": by})
        return obs

    def purge_expired(self) -> int:
        """Delete stored copies past their retention; keep the observation and what was learned from it."""
        now = self.clock.now()
        rows = self.db.query("SELECT * FROM observations WHERE expires_at IS NOT NULL AND expires_at < ? AND "
                             "status != ?", (now, Status.EXPIRED.value))
        count = 0
        for r in rows:
            obs = _row(r)
            obs.status, old = Status.EXPIRED, obs.path
            obs.path = None
            self.update(obs)
            if old and not self.db.query_one("SELECT 1 FROM observations WHERE path=? AND status != ?",
                                             (old, Status.EXPIRED.value)):
                self._delete_file(old)
            count += 1
        return count

    def _delete_file(self, path: str | None) -> None:
        if path:
            try:
                os.remove(path)
            except OSError:
                pass

    def wipe(self) -> None:
        """Everything perceived, gone (used by tests and "forget everything you've seen")."""
        self.db.execute("DELETE FROM observations")
        self.db.execute("DELETE FROM perception_cache")
        shutil.rmtree(self.root / "objects", ignore_errors=True)

    def _emit(self, etype: EventType, payload: dict[str, Any], severity: Severity = Severity.INFO) -> None:
        if self.bus is not None:
            self.bus.emit(Event(etype, "perception", payload, severity=severity,
                                entity_id=f"observation:{payload['observation_id']}" if payload.get("observation_id")
                                else None))


def _guess_text_mime(name: str, data: bytes) -> str:
    sample = data[:4096]
    if b"\x00" in sample:
        return "application/octet-stream"
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        try:
            sample.decode("cp1252")
        except UnicodeDecodeError:
            return "application/octet-stream"
    ext = os.path.splitext(name.lower())[1]
    return {".md": "text/markdown", ".json": "application/json", ".csv": "text/csv", ".html": "text/html",
            ".xml": "text/xml"}.get(ext, "text/plain")


def _row(r: Any) -> Observation:
    return Observation(
        id=r["id"], kind=InputKind(r["kind"]), origin=InputOrigin(r["origin"]), name=r["name"],
        created_at=r["created_at"], session_id=r["session_id"], project_id=r["project_id"], mime=r["mime"] or "",
        size_bytes=r["size_bytes"] or 0, sha256=r["sha256"] or "", path=r["path"], source_path=r["source_path"],
        width=r["width"], height=r["height"], pages=r["pages"], ordinal=r["ordinal"] or 0, status=Status(r["status"]),
        error=r["error"] or "", sensitivity=Sensitivity(r["sensitivity"]), labels=loads(r["labels"], []),
        derived=loads(r["derived"], {}), updated_at=r["updated_at"], expires_at=r["expires_at"])
