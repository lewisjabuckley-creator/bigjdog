"""Reading text in images (Phase 4 §7), with confidence and provenance, never presented as more exact than it is.

Engines, in order of preference (``perception.ocr = auto``):

* Tesseract, if installed: a real OCR engine. Its TSV output gives a confidence per word, so every line carries a
  measured confidence and low-confidence lines are marked [?].
* A vision model asked to transcribe: often good, but its confidence isn't measured, so the result says it was read
  by a model and may contain mistakes.

If neither is available JARVIS says so instead of guessing. Results are cached by image content.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from jarvis.platforms import hidden_window_kwargs


@dataclass
class OCRLine:
    text: str
    confidence: float | None = None          # 0..1, measured by the engine; None when not measured
    box: tuple[int, int, int, int] | None = None   # x, y, width, height in image pixels

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "confidence": self.confidence, "box": list(self.box) if self.box else None}


@dataclass
class OCRResult:
    engine: str                              # tesseract | vision:<model> | none
    text: str
    lines: list[OCRLine] = field(default_factory=list)
    mean_confidence: float | None = None
    measured: bool = False                   # confidence measured by an OCR engine (not a model's impression)
    note: str = ""
    observation_id: str = ""
    created_at: float = 0.0
    image_size: tuple[int, int] | None = None

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    def to_dict(self) -> dict[str, Any]:
        return {"engine": self.engine, "text": self.text, "lines": [l.to_dict() for l in self.lines],
                "mean_confidence": self.mean_confidence, "measured": self.measured, "note": self.note,
                "observation_id": self.observation_id, "created_at": self.created_at,
                "image_size": list(self.image_size) if self.image_size else None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OCRResult":
        return cls(d.get("engine", "none"), d.get("text", ""),
                   [OCRLine(l["text"], l.get("confidence"), tuple(l["box"]) if l.get("box") else None)
                    for l in d.get("lines", [])], d.get("mean_confidence"), bool(d.get("measured")),
                   d.get("note", ""), d.get("observation_id", ""), d.get("created_at", 0.0),
                   tuple(d["image_size"]) if d.get("image_size") else None)

    def marked_text(self, threshold: float = 0.6) -> str:
        """The text with lines the engine was unsure of marked [?] (so uncertain OCR never reads as exact)."""
        if not self.lines:
            return self.text
        return "\n".join(f"{l.text} [?]" if l.confidence is not None and l.confidence < threshold else l.text
                         for l in self.lines)

    def region(self, where: str) -> list[OCRLine]:
        """Lines in a part of the image ("bottom", "top", "left", "right"), when positions are known."""
        if not self.image_size or not any(l.box for l in self.lines):
            return []
        w, h = self.image_size
        out = []
        for l in self.lines:
            if not l.box:
                continue
            cx, cy = l.box[0] + l.box[2] / 2, l.box[1] + l.box[3] / 2
            if (where == "bottom" and cy > h * 2 / 3) or (where == "top" and cy < h / 3) or \
                    (where == "left" and cx < w / 3) or (where == "right" and cx > w * 2 / 3) or \
                    (where in ("middle", "centre", "center") and h / 3 <= cy <= h * 2 / 3):
                out.append(l)
        return out


def describe(result: OCRResult, handle: str, when: str) -> str:
    """The provenance block the user sees: source, text, confidence, time."""
    if result.measured and result.mean_confidence is not None:
        confidence = f"{result.mean_confidence:.0%} average ({result.engine}); lines marked [?] are uncertain"
    elif result.engine.startswith("vision:"):
        confidence = f"not measured: read by the vision model {result.engine.split(':', 1)[1]}, which can make mistakes"
    else:
        confidence = "not measured"
    text = result.marked_text().strip() or "(no text found)"
    return f"Source: {handle}\nExtracted text:\n{text}\nConfidence: {confidence}\nRead: {when}"


class OCREngine:
    name = "none"

    def available(self) -> tuple[bool, str]:
        return False, "no OCR engine"

    async def extract(self, data: bytes, path: str | None) -> OCRResult:
        raise NotImplementedError


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(self, binary: str = "", *, timeout_s: float = 60.0, languages: str = "eng") -> None:
        self.binary = binary or find_tesseract() or ""
        self.timeout_s = timeout_s
        self.languages = languages

    def available(self) -> tuple[bool, str]:
        if self.binary and os.path.exists(self.binary) or (self.binary and shutil.which(self.binary)):
            return True, f"Tesseract ({self.binary})"
        return False, "Tesseract isn't installed"

    async def extract(self, data: bytes, path: str | None) -> OCRResult:
        if not path:
            raise RuntimeError("tesseract needs the image as a file")
        proc = await asyncio.to_thread(
            subprocess.run, [self.binary, path, "stdout", "-l", self.languages, "tsv"], capture_output=True,
            timeout=self.timeout_s, **hidden_window_kwargs())
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or b"").decode(errors="replace").strip()[:300] or "tesseract failed")
        return parse_tsv(proc.stdout.decode("utf-8", errors="replace"))


def find_tesseract() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    if sys.platform == "win32":
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                     os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                     os.path.expandvars(r"%LOCALAPPDATA%\Programs")):
            candidate = os.path.join(base, "Tesseract-OCR", "tesseract.exe")
            if os.path.exists(candidate):
                return candidate
    return None


def parse_tsv(tsv: str) -> OCRResult:
    """Tesseract TSV → lines (words grouped by block/paragraph/line) with mean word confidence and boxes."""
    lines: dict[tuple[int, int, int], list[tuple[str, float, tuple[int, int, int, int]]]] = {}
    order: list[tuple[int, int, int]] = []
    for raw in tsv.splitlines()[1:]:
        cols = raw.split("\t")
        if len(cols) < 12 or cols[0] != "5":
            continue
        word = cols[11].strip()
        try:
            conf = float(cols[10])
        except ValueError:
            continue
        if not word or conf < 0:
            continue
        key = (int(cols[2]), int(cols[3]), int(cols[4]))
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append((word, conf / 100.0, (int(cols[6]), int(cols[7]), int(cols[8]), int(cols[9]))))
    out: list[OCRLine] = []
    for key in order:
        words = lines[key]
        x0 = min(b[0] for _, _, b in words)
        y0 = min(b[1] for _, _, b in words)
        x1 = max(b[0] + b[2] for _, _, b in words)
        y1 = max(b[1] + b[3] for _, _, b in words)
        out.append(OCRLine(" ".join(w for w, _, _ in words), round(sum(c for _, c, _ in words) / len(words), 3),
                           (x0, y0, x1 - x0, y1 - y0)))
    confs = [l.confidence for l in out if l.confidence is not None]
    return OCRResult("tesseract", "\n".join(l.text for l in out), out,
                     round(sum(confs) / len(confs), 3) if confs else None, measured=True)


Transcriber = Callable[[Any], Awaitable[tuple[str, str]]]     # observation -> (text, model)


class OCRService:
    """Picks an engine, caches by image content, records provenance."""

    def __init__(self, store: Any, *, mode: str = "auto", tesseract: TesseractEngine | None = None,
                 transcriber: Transcriber | None = None, vision_available: Callable[[], bool] | None = None,
                 clock: Any = None, engines: list[OCREngine] | None = None) -> None:
        self.store = store
        self.mode = mode
        self.tesseract = tesseract or TesseractEngine()
        self.transcriber = transcriber
        self.vision_available = vision_available or (lambda: False)
        self.clock = clock
        self.extra_engines = engines or []        # injected engines (tests, future local OCR packages)

    def status(self) -> tuple[bool, str]:
        if self.mode == "off":
            return False, "switched off in the configuration"
        for engine in self.extra_engines:
            ok, detail = engine.available()
            if ok:
                return True, detail
        if self.mode in ("auto", "tesseract"):
            ok, detail = self.tesseract.available()
            if ok:
                return True, detail
        if self.mode in ("auto", "vision") and self.transcriber is not None and self.vision_available():
            return True, "through the vision model (confidence not measured)"
        if self.mode == "tesseract":
            return False, "Tesseract isn't installed"
        return False, "no OCR engine: install Tesseract, or a vision model in Ollama (e.g. ollama pull llava)"

    async def read(self, obs: Any, *, force: bool = False) -> OCRResult:
        """OCR for one image observation (cached). Raises RuntimeError with a reason when no engine can read it."""
        if not force:
            hit = self.store.cached(obs.sha256, "ocr")
            if hit:
                result = OCRResult.from_dict(hit)
                result.observation_id = obs.id
                return result
        data = self.store.data(obs)
        errors: list[str] = []
        result: OCRResult | None = None
        engines: list[OCREngine] = [e for e in self.extra_engines if e.available()[0]]
        if self.mode in ("auto", "tesseract") and self.tesseract.available()[0]:
            engines.append(self.tesseract)
        for engine in engines:
            try:
                result = await engine.extract(data, obs.path)
                break
            except Exception as exc:
                errors.append(f"{engine.name}: {exc}")
        if result is None and self.mode in ("auto", "vision") and self.transcriber is not None and \
                self.vision_available():
            try:
                text, model = await self.transcriber(obs)
                lines = [OCRLine(t.strip()) for t in text.splitlines() if t.strip()]
                result = OCRResult(f"vision:{model}", "\n".join(l.text for l in lines), lines, None, measured=False,
                                   note="read by a vision model; confidence isn't measured and it can misread text")
            except Exception as exc:
                errors.append(f"vision model: {exc}")
        if result is None:
            why = "; ".join(errors) if errors else self.status()[1]
            raise RuntimeError(f"no OCR engine could read it ({why})")
        result.observation_id = obs.id
        result.created_at = self.clock.now() if self.clock else 0.0
        if obs.width and obs.height:
            result.image_size = (obs.width, obs.height)
        self.store.cache(obs.sha256, "ocr", result.to_dict(), model=result.engine)
        return result
