"""The input abstraction: every way something reaches JARVIS becomes the same shape.

    TEXT  IMAGE  SCREENSHOT  DOCUMENT  FILE  AUDIO (future)  CAMERA_FRAME (future)  SYSTEM_EVENT
                                        ↓
                             INPUT NORMALISATION (NormalizedInput)
                                        ↓
                        intent / context → planner / orchestrator

An interface (the CLI today; a HUD, a phone or a microphone later) hands JARVIS a :class:`NormalizedInput`: the
words (typed, or one day transcribed) plus any attachments, each an :class:`Observation`. Nothing downstream cares
which interface or input source produced it. Voice and camera exist here as interfaces only: they report that they
are not available, and when they are built they produce the same NormalizedInput, so the planner, memory, agents,
permissions and tasks need no rewrite.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from jarvis.core.types import Provenance, ProvenanceKind


class InputKind(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    SCREENSHOT = "screenshot"
    DOCUMENT = "document"
    FILE = "file"
    AUDIO = "audio"                  # future: microphone / voice notes
    CAMERA_FRAME = "camera_frame"    # future: a camera
    SYSTEM_EVENT = "system_event"

    @property
    def visual(self) -> bool:
        return self in (InputKind.IMAGE, InputKind.SCREENSHOT, InputKind.CAMERA_FRAME)

    @property
    def label(self) -> str:
        return {"image": "image", "screenshot": "screenshot", "document": "document", "file": "file",
                "audio": "recording", "camera_frame": "camera frame", "text": "text",
                "system_event": "event"}[self.value]


class InputOrigin(StrEnum):
    USER_UPLOAD = "user_upload"      # sent by an interface (dragged in, pasted, uploaded)
    USER_PATH = "user_path"          # the user named a file on this computer
    CLIPBOARD = "clipboard"
    SCREEN_CAPTURE = "screen_capture"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    TOOL = "tool"                    # produced by a tool JARVIS ran
    API = "api"
    SYSTEM = "system"


class Sensitivity(StrEnum):
    NORMAL = "normal"
    PRIVATE = "private"              # screen captures, personal documents: local processing only
    SECRET = "secret"                # credentials seen in it: local only, short retention, never remembered

    def at_least(self, other: "Sensitivity") -> bool:
        order = [Sensitivity.NORMAL, Sensitivity.PRIVATE, Sensitivity.SECRET]
        return order.index(self) >= order.index(other)


class Status(StrEnum):
    RECEIVED = "received"
    PROCESSED = "processed"
    FAILED = "failed"
    EXPIRED = "expired"              # the stored copy was deleted by retention; what was learned is kept


@dataclass
class Observation:
    """One thing JARVIS was shown (or captured), with where it came from and what has been learned from it."""
    id: str
    kind: InputKind
    origin: InputOrigin
    name: str
    created_at: float
    session_id: str | None = None
    project_id: str | None = None
    mime: str = ""
    size_bytes: int = 0
    sha256: str = ""
    path: str | None = None          # the stored copy (content-addressed), while retained
    source_path: str | None = None   # where the user's file was
    width: int | None = None
    height: int | None = None
    pages: int | None = None
    ordinal: int = 0                 # "screenshot 2": its number among inputs of its kind in the session
    status: Status = Status.RECEIVED
    error: str = ""
    sensitivity: Sensitivity = Sensitivity.NORMAL
    labels: list[str] = field(default_factory=list)       # what it shows, in words (for "the error screenshot")
    derived: dict[str, Any] = field(default_factory=dict)  # description, ocr, ui, document structure, findings
    updated_at: float = 0.0
    expires_at: float | None = None

    @property
    def handle(self) -> str:
        """How it is referred to in conversation: "screenshot 2" or "image 1" or "report.pdf"."""
        if self.kind in (InputKind.DOCUMENT, InputKind.FILE):
            return self.name
        return f"{self.kind.label} {self.ordinal}" if self.ordinal else self.kind.label

    @property
    def available(self) -> bool:
        return bool(self.path) and self.status != Status.EXPIRED and os.path.exists(self.path)

    def provenance(self, detail: str = "") -> Provenance:
        kind = {InputKind.SCREENSHOT: ProvenanceKind.SCREENSHOT, InputKind.IMAGE: ProvenanceKind.IMAGE,
                InputKind.CAMERA_FRAME: ProvenanceKind.IMAGE, InputKind.DOCUMENT: ProvenanceKind.DOCUMENT,
                InputKind.FILE: ProvenanceKind.DOCUMENT}.get(self.kind, ProvenanceKind.USER_STATEMENT)
        if self.origin == InputOrigin.SCREEN_CAPTURE:
            kind = ProvenanceKind.SCREEN_CAPTURE
        return Provenance(kind, f"{self.handle} ({self.id})", detail or self.name)

    def summary(self) -> str:
        """One line from what has been learned, for context and memory."""
        d = self.derived
        text = d.get("finding") or d.get("description") or d.get("summary") or ""
        if not text and d.get("ocr"):
            text = "text: " + (d["ocr"].get("text") or "")[:160]
        return " ".join(str(text).split())[:240]

    def to_api(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind.value, "origin": self.origin.value, "name": self.name,
                "handle": self.handle, "created_at": self.created_at, "session_id": self.session_id,
                "project_id": self.project_id, "mime": self.mime, "size_bytes": self.size_bytes,
                "width": self.width, "height": self.height, "pages": self.pages, "ordinal": self.ordinal,
                "status": self.status.value, "error": self.error, "sensitivity": self.sensitivity.value,
                "labels": list(self.labels), "summary": self.summary(), "available": self.available,
                "expires_at": self.expires_at,
                "derived": {k: v for k, v in self.derived.items() if k in ("description", "finding", "summary",
                                                                             "ocr", "ui", "document", "diagram",
                                                                             "injection_warning", "screen")}}


@dataclass
class Attachment:
    """What an interface sends: raw bytes with a name, a path on this computer, or an existing observation."""
    name: str = ""
    data: bytes | None = None
    path: str | None = None
    observation_id: str | None = None
    origin: InputOrigin = InputOrigin.USER_UPLOAD
    kind: InputKind | None = None    # a hint; the content decides

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> "Attachment":
        import base64
        if raw.get("id"):
            return cls(observation_id=str(raw["id"]))
        if raw.get("data"):
            data = base64.b64decode(str(raw["data"]), validate=False)
            return cls(name=str(raw.get("name") or "upload"), data=data,
                       origin=InputOrigin(raw.get("origin") or InputOrigin.USER_UPLOAD.value),
                       kind=InputKind(raw["kind"]) if raw.get("kind") else None)
        if raw.get("path"):
            return cls(name=os.path.basename(str(raw["path"])), path=str(raw["path"]),
                       origin=InputOrigin(raw.get("origin") or InputOrigin.USER_PATH.value),
                       kind=InputKind(raw["kind"]) if raw.get("kind") else None)
        raise ValueError("an attachment needs an id, data or a path")


@dataclass
class NormalizedInput:
    """One user turn, whatever produced it: the words plus what came with them."""
    text: str
    attachments: list[Observation] = field(default_factory=list)
    source: str = "text"             # text | voice | api | hud — who produced the words
    interface: str = "cli"
    session_id: str | None = None
    confidence: float = 1.0          # typed text is exact; a transcript would carry its recogniser's confidence
    received_at: float = 0.0
    problems: list[str] = field(default_factory=list)   # attachments that could not be taken in, and why


# -- input sources ------------------------------------------------------------------------------------------------

class InputSource(ABC):
    """A way input can arrive. Only text is required; the rest report honestly whether they exist."""
    kind: InputKind
    name: str

    @abstractmethod
    def status(self) -> tuple[str, str]:
        """(state, detail): state is available | disabled | not_available | not_implemented."""


class TextInput(InputSource):
    kind, name = InputKind.TEXT, "text"

    def status(self) -> tuple[str, str]:
        return "available", "typing in a JARVIS interface (the primary input)"

    @staticmethod
    def normalize(text: str, **kw: Any) -> NormalizedInput:
        return NormalizedInput(text=text.strip(), source="text", **kw)


class ImageInput(InputSource):
    kind, name = InputKind.IMAGE, "images"

    def status(self) -> tuple[str, str]:
        return "available", "attach a file, drop it into the window or paste it (/attach, /paste)"


class DocumentInput(InputSource):
    kind, name = InputKind.DOCUMENT, "documents"

    def status(self) -> tuple[str, str]:
        return "available", "text, Markdown, code, JSON, CSV, logs, configuration files and PDFs"


class VoiceInput(InputSource):
    """Future: microphone → speech recognition → text → the same NormalizedInput as typing.

    Deliberately not implemented in Phase 4 (no microphone is required or used). When it is built, ``transcribe``
    returns NormalizedInput(text=transcript, source="voice", confidence=<recogniser confidence>) and nothing else in
    JARVIS changes."""
    kind, name = InputKind.AUDIO, "voice"

    def __init__(self, microphone_detected: bool = False) -> None:
        self.microphone_detected = microphone_detected

    def status(self) -> tuple[str, str]:
        if self.microphone_detected:
            return "not_implemented", "a microphone is connected, but voice input isn't built yet; typing works"
        return "not_available", "no microphone is set up; typing works for everything"

    async def transcribe(self, audio: bytes) -> NormalizedInput:
        raise NotImplementedError("speech recognition is not part of JARVIS yet")


class CameraInput(InputSource):
    """Future: camera → capture → vision → observation → world state. Never switched on silently."""
    kind, name = InputKind.CAMERA_FRAME, "camera"

    def __init__(self, cameras_detected: int = 0) -> None:
        self.cameras_detected = cameras_detected

    def status(self) -> tuple[str, str]:
        if self.cameras_detected:
            return "not_implemented", (f"{self.cameras_detected} camera(s) detected; JARVIS doesn't use cameras yet "
                                       "(and never would without you switching it on)")
        return "not_available", "no camera detected"

    async def capture(self) -> Observation:
        raise NotImplementedError("camera input is not part of JARVIS yet")


# -- recognising what an input is ---------------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
DOCUMENT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".json", ".csv", ".tsv", ".log", ".pdf", ".ini", ".cfg",
                       ".conf", ".toml", ".yaml", ".yml", ".xml", ".html", ".htm", ".py", ".js", ".ts", ".tsx", ".jsx",
                       ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".ps1", ".bat",
                       ".sql", ".kt", ".swift", ".lua", ".env.example", ".properties", ".gradle", ".cmake", ".mk"}
_SCREENSHOT_NAME = re.compile(r"(screen\s*shot|screenshot|screen[_ -]?capture|snip|capture|printscreen|scr[_-]\d)",
                              re.I)
_COMMON_SCREENS = {(1280, 720), (1280, 800), (1366, 768), (1440, 900), (1536, 864), (1600, 900), (1680, 1050),
                   (1920, 1080), (1920, 1200), (2560, 1080), (2560, 1440), (2560, 1600), (2880, 1800), (3440, 1440),
                   (3840, 2160), (3024, 1964), (2736, 1824), (2256, 1504)}


def classify(name: str, mime: str, width: int | None = None, height: int | None = None) -> InputKind:
    ext = os.path.splitext(name.lower())[1]
    if mime.startswith("image/") or ext in IMAGE_EXTENSIONS:
        if _SCREENSHOT_NAME.search(name) or (width, height) in _COMMON_SCREENS:
            return InputKind.SCREENSHOT
        return InputKind.IMAGE
    if mime == "application/pdf" or ext in DOCUMENT_EXTENSIONS or mime.startswith("text/"):
        return InputKind.DOCUMENT
    return InputKind.FILE


# -- paths typed or dropped into the text ------------------------------------------------------------------------

# Dropping a file onto a terminal pastes its path: quoted on Windows when it has spaces ("C:\\Users\\Me\\a b.png"),
# sometimes with a leading & (PowerShell); escaped spaces on macOS/Linux (/Users/me/a\\ b.png).
_QUOTED = re.compile(r"""(?:^|[\s(&])["']((?:[A-Za-z]:[\\/]|\\\\|~[\\/]|/)[^"'\n]+?)["']""")
_BARE = re.compile(r"""(?:^|[\s(])((?:[A-Za-z]:[\\/]|\\\\[^\s\\]+\\|~[\\/]|/)(?:[^\s"'<>|]|\\ )+)""")


def find_paths(text: str, *, exists: Any = os.path.isfile) -> list[str]:
    """Existing files named in the text (as typed, dropped or pasted), in order, without duplicates."""
    found: list[str] = []
    for pattern in (_QUOTED, _BARE):
        for m in pattern.finditer(text):
            raw = m.group(1).strip().rstrip(".,;:!?)")
            candidate = os.path.expanduser(raw.replace("\\ ", " ") if not re.match(r"^[A-Za-z]:\\", raw) else raw)
            if candidate not in found and exists(candidate):
                found.append(candidate)
    return found


def strip_paths(text: str, paths: list[str]) -> str:
    """The text without the file paths that were taken in as attachments ("what's wrong here? C:\\x.png")."""
    out = text
    for p in paths:
        for variant in (f'"{p}"', f"'{p}'", f"& '{p}'", p, p.replace(" ", "\\ ")):
            out = out.replace(variant, " ")
    return " ".join(out.split())
