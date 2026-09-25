"""Screen awareness (Phase 4 §9-14, §35): off unless the user turns it on, visible while on, stoppable at any time.

Modes (``perception.screen_mode``, then whatever the user last chose):

* off         JARVIS cannot capture the screen at all (the default).
* on_request  JARVIS looks only when the user asks ("look at my screen", "what's on my screen?").
* watching    JARVIS also checks periodically and reports important changes (an error or a permission dialog
              appearing, a build failing) through the normal notification rules.

Only the user can switch it on (an automation, agent, plan or model tool cannot); anyone and anything can switch it
off. Every capture is audited; captures are stored as *private* observations (never sent to a cloud model), kept for
a day, and deleted at once if credentials are visible on them.

Pipeline:  current screen → capture → OCR (+ vision on request) → UI state → ScreenState → world state / events.

Watching is cheap-first: each check reads only the active window (no image); a full capture + OCR happens when the
window changes or every Nth check; vision is never used by the watcher (it's expensive) — only when the user asks.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from jarvis.core.types import Confidence, Provenance, ProvenanceKind, Severity, new_id
from jarvis.database.db import dumps, loads
from jarvis.perception import images, safety
from jarvis.perception.inputs import Attachment, InputKind, InputOrigin, Sensitivity, Status
from jarvis.platforms import hidden_window_kwargs


class ScreenMode(StrEnum):
    OFF = "off"
    ON_REQUEST = "on_request"
    WATCHING = "watching"


class ScreenAccessDenied(PermissionError):
    """The screen can't be captured now (off, or not permitted); the message says how the user can allow it."""


@dataclass
class WindowInfo:
    title: str
    app: str = ""
    pid: int | None = None
    active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Detection:
    kind: str            # error | build_failed | build_completed | permission_prompt | dialog | warning
    evidence: str
    confidence: float
    source: str          # window title | ocr | vision

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# without Pillow, screenshots larger than this aren't compared pixel by pixel while watching (too slow to repeat)
_PURE_PYTHON_DIFF_PIXELS = 2_400_000

_LABEL = {"error": "Error visible", "build_failed": "Build failed", "build_completed": "Build completed",
          "permission_prompt": "Permission dialog open", "dialog": "Dialog open", "warning": "Warning visible"}


@dataclass
class ScreenState:
    id: str
    ts: float
    active_app: str = ""
    window_title: str = ""
    windows: list[WindowInfo] = field(default_factory=list)
    observation_id: str | None = None
    text_excerpt: str = ""
    text_hash: str = ""
    detections: list[Detection] = field(default_factory=list)
    ui: dict[str, Any] | None = None
    source: str = "window list"          # window list | capture + OCR | capture + vision
    ocr_engine: str = ""
    pixel_change: float | None = None
    notes: list[str] = field(default_factory=list)

    def describe(self, when: str = "") -> str:
        lines = [f"Active application: {self.active_app or 'unknown'}",
                 f"Visible window: {self.window_title or 'unknown'}"]
        if self.detections:
            lines.append("Detected state: " + "; ".join(dict.fromkeys(_LABEL.get(d.kind, d.kind)
                                                                     for d in self.detections)))
            relevant = next((d for d in self.detections if d.kind in ("error", "build_failed", "permission_prompt")),
                            None)
            if relevant:
                lines.append(f"Relevant text: {relevant.evidence}")
        if self.ui and self.ui.get("elements"):
            lines.append("Elements: " + "; ".join(f"{e.get('type', '')} '{e.get('label', '')}'"
                                                  + (f" ({e['state']})" if e.get("state") else "")
                                                  for e in self.ui["elements"][:10]))
        others = [w.title for w in self.windows if not w.active and w.title][:5]
        if others:
            lines.append("Other windows: " + "; ".join(others))
        lines.append(f"Seen: {when or 'just now'} ({self.source})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScreenState":
        return cls(d["id"], d["ts"], d.get("active_app", ""), d.get("window_title", ""),
                   [WindowInfo(**w) for w in d.get("windows", [])], d.get("observation_id"), d.get("text_excerpt", ""),
                   d.get("text_hash", ""), [Detection(**x) for x in d.get("detections", [])], d.get("ui"),
                   d.get("source", "window list"), d.get("ocr_engine", ""), d.get("pixel_change"), d.get("notes", []))


# -- detection rules ----------------------------------------------------------------------------------------------

_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("build_failed", re.compile(r"\b(build failed|compilation failed|failed to compile|npm ERR!|could not compile|"
                                r"BUILD FAILURE|\d+ failed(,| in)|FAILED \(|error\[E\d+\]|MSB\d{4}: error)", re.I)),
    ("build_completed", re.compile(r"\b(build succeeded|build successful|compiled successfully|BUILD SUCCESS|"
                                   r"build completed|all tests passed|\b\d+ passed in\b)", re.I)),
    ("permission_prompt", re.compile(r"(do you want to allow|user account control|wants to make changes to your|"
                                     r"allow access|permission required|requires administrator|grant access|"
                                     r"would like to access|is requesting permission)", re.I)),
    ("error", re.compile(r"\b(error|exception|traceback|fatal|crash(ed)?|not responding|stopped working|"
                         r"failed to|access is denied|cannot find|no such file)\b", re.I)),
    ("dialog", re.compile(r"(are you sure|do you want to save|save changes|confirm|unsaved changes|"
                          r"\bOK\s+Cancel\b|\bYes\s+No\b)", re.I)),
    ("warning", re.compile(r"\bwarning\b", re.I)),
]
_NOT_ERROR = re.compile(r"\b(0|no) (errors?|failures?|failed)\b|\berrors?:\s*0\b", re.I)


def detect(text: str, title: str = "", *, confidences: dict[str, float] | None = None,
           source: str = "ocr") -> list[Detection]:
    """What the visible text says about the screen's state (deterministic; evidence kept)."""
    found: list[Detection] = []
    seen: set[str] = set()
    for origin, body, base in (("window title", title, 0.8), (source, text, 0.7)):
        for line in (body or "").splitlines():
            clean = line.strip()
            if not clean or _NOT_ERROR.search(clean):
                continue
            for kind, pattern in _RULES:
                if pattern.search(clean):
                    conf = (confidences or {}).get(clean, base)
                    key = f"{kind}:{clean[:80]}"
                    if key in seen:
                        continue
                    seen.add(key)
                    seen.add(kind)
                    found.append(Detection(kind, clean[:200], round(conf, 2), origin))
                    break
    order = {"build_failed": 0, "permission_prompt": 1, "error": 2, "dialog": 3, "build_completed": 4, "warning": 5}
    return sorted(found, key=lambda d: order.get(d.kind, 9))[:12]


# -- capture backends -----------------------------------------------------------------------------------------------

class ScreenBackend(ABC):
    name = "none"

    @abstractmethod
    def capture_available(self) -> tuple[bool, str]: ...

    @abstractmethod
    async def capture(self) -> bytes: ...

    async def active_window(self) -> WindowInfo | None:
        return None

    async def windows(self) -> list[WindowInfo]:
        active = await self.active_window()
        return [active] if active else []


async def _run(argv: list[str], timeout: float = 20.0) -> subprocess.CompletedProcess:
    return await asyncio.to_thread(subprocess.run, argv, capture_output=True, timeout=timeout, **hidden_window_kwargs())


_WIN_CAPTURE = r"""
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$b = [System.Windows.Forms.SystemInformation]::VirtualScreen
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Left, $b.Top, 0, 0, $bmp.Size)
$bmp.Save('{path}', [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
"""


class WindowsScreenBackend(ScreenBackend):
    name = "windows"

    def capture_available(self) -> tuple[bool, str]:
        return (True, "Windows screen capture (built in)") if shutil.which("powershell") else \
            (False, "PowerShell isn't available")

    async def capture(self) -> bytes:
        fd, path = tempfile.mkstemp(suffix=".png", prefix="jarvis-screen-")
        os.close(fd)
        try:
            proc = await _run(["powershell", "-NoProfile", "-NonInteractive", "-STA", "-Command",
                               _WIN_CAPTURE.replace("{path}", path.replace("'", "''"))], timeout=30)
            if proc.returncode != 0 or not os.path.getsize(path):
                raise RuntimeError((proc.stderr or b"").decode(errors="replace").strip()[:200] or "capture failed")
            with open(path, "rb") as fh:
                return fh.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    async def active_window(self) -> WindowInfo | None:
        return await asyncio.to_thread(_win_foreground)

    async def windows(self) -> list[WindowInfo]:
        return await asyncio.to_thread(_win_windows)


def _win_foreground() -> WindowInfo | None:
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        return _win_info(hwnd, active=True)
    except Exception:
        return None


def _win_info(hwnd: Any, active: bool = False) -> WindowInfo:
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    app = ""
    try:
        import psutil
        app = psutil.Process(pid.value).name()
    except Exception:
        pass
    return WindowInfo(buf.value, os.path.splitext(app)[0], pid.value or None, active)


def _win_windows() -> list[WindowInfo]:
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        foreground = user32.GetForegroundWindow()
        out: list[WindowInfo] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def callback(hwnd, _):
            if user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd) > 0 and len(out) < 40:
                out.append(_win_info(hwnd, active=hwnd == foreground))
            return True
        user32.EnumWindows(callback, 0)
        return out
    except Exception:
        return []


class MacScreenBackend(ScreenBackend):
    name = "macos"

    def capture_available(self) -> tuple[bool, str]:
        return (True, "macOS screencapture") if shutil.which("screencapture") else (False, "screencapture is missing")

    async def capture(self) -> bytes:
        fd, path = tempfile.mkstemp(suffix=".png", prefix="jarvis-screen-")
        os.close(fd)
        try:
            proc = await _run(["screencapture", "-x", "-t", "png", path])
            if proc.returncode != 0 or not os.path.getsize(path):
                raise RuntimeError("capture failed (macOS may need Screen Recording permission for the terminal)")
            with open(path, "rb") as fh:
                return fh.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    async def active_window(self) -> WindowInfo | None:
        script = ('tell application "System Events" to set p to first application process whose frontmost is true\n'
                  'set n to name of p\nset t to ""\ntry\nset t to name of front window of p\nend try\nreturn n & "|" & t')
        try:
            proc = await _run(["osascript", "-e", script], timeout=10)
            app, _, title = proc.stdout.decode(errors="replace").strip().partition("|")
            return WindowInfo(title, app, None, True) if app else None
        except Exception:
            return None


class LinuxScreenBackend(ScreenBackend):
    name = "linux"
    _TOOLS = [("grim", ["grim", "{path}"]), ("gnome-screenshot", ["gnome-screenshot", "-f", "{path}"]),
              ("spectacle", ["spectacle", "-b", "-n", "-o", "{path}"]), ("scrot", ["scrot", "-o", "{path}"]),
              ("import", ["import", "-window", "root", "{path}"])]

    def _tool(self) -> list[str] | None:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return None
        for name, argv in self._TOOLS:
            if shutil.which(name):
                return argv
        return None

    def capture_available(self) -> tuple[bool, str]:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return False, "there's no graphical display here"
        tool = self._tool()
        return (True, f"{tool[0]}") if tool else (False, "no screenshot tool is installed (grim, gnome-screenshot, "
                                                          "scrot or ImageMagick's import)")

    async def capture(self) -> bytes:
        argv = self._tool()
        if argv is None:
            raise RuntimeError(self.capture_available()[1])
        fd, path = tempfile.mkstemp(suffix=".png", prefix="jarvis-screen-")
        os.close(fd)
        try:
            proc = await _run([a.replace("{path}", path) for a in argv])
            if proc.returncode != 0 or not os.path.getsize(path):
                raise RuntimeError((proc.stderr or b"").decode(errors="replace").strip()[:200] or "capture failed")
            with open(path, "rb") as fh:
                return fh.read()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    async def active_window(self) -> WindowInfo | None:
        if not shutil.which("xdotool") or not os.environ.get("DISPLAY"):
            return None
        try:
            name = (await _run(["xdotool", "getactivewindow", "getwindowname"], 5)).stdout.decode(errors="replace")
            pid = (await _run(["xdotool", "getactivewindow", "getwindowpid"], 5)).stdout.decode().strip()
            app = ""
            if pid.isdigit():
                import psutil
                try:
                    app = psutil.Process(int(pid)).name()
                except Exception:
                    pass
            return WindowInfo(name.strip(), app, int(pid) if pid.isdigit() else None, True)
        except Exception:
            return None


class SimulatedScreenBackend(ScreenBackend):
    """A scripted screen for tests and simulation mode: set what is "on screen"."""
    name = "simulated"

    def __init__(self) -> None:
        self.image = images.solid_png(320, 200)
        self.window = WindowInfo("Desktop", "explorer", 1, True)
        self.others: list[WindowInfo] = []
        self.fail: str | None = None
        self.captures = 0

    def show(self, image: bytes | None = None, *, title: str | None = None, app: str | None = None,
             others: list[WindowInfo] | None = None) -> None:
        if image is not None:
            self.image = image
        if title is not None or app is not None:
            self.window = WindowInfo(title if title is not None else self.window.title,
                                     app if app is not None else self.window.app, self.window.pid, True)
        if others is not None:
            self.others = others

    def capture_available(self) -> tuple[bool, str]:
        return True, "simulated screen"

    async def capture(self) -> bytes:
        if self.fail:
            raise RuntimeError(self.fail)
        self.captures += 1
        return self.image

    async def active_window(self) -> WindowInfo | None:
        return self.window

    async def windows(self) -> list[WindowInfo]:
        return [self.window] + self.others


def default_backend(simulated: bool = False) -> ScreenBackend:
    if simulated:
        return SimulatedScreenBackend()
    if sys.platform == "win32":
        return WindowsScreenBackend()
    if sys.platform == "darwin":
        return MacScreenBackend()
    return LinuxScreenBackend()


# -- screen awareness ---------------------------------------------------------------------------------------------

class ScreenAwareness:
    STATE_KEY = "perception.screen.mode"

    def __init__(self, backend: ScreenBackend, store: Any, *, ocr: Any = None, vision: Any = None, state: Any = None,
                 bus: Any = None, audit: Any = None, world: Any = None, config: Any = None, clock: Any = None) -> None:
        self.backend = backend
        self.store = store
        self.ocr = ocr
        self.vision = vision
        self.state = state
        self.bus = bus
        self.audit = audit
        self.world = world
        self.config = config
        self.clock = clock
        self.last: ScreenState | None = None
        self.captures = 0
        self._ticks = 0
        self._last_window: WindowInfo | None = None
        self._last_windows: set[str] = set()
        self._lock = asyncio.Lock()
        self.last_error = ""

    # -- the switch ---------------------------------------------------------------------------------------------
    @property
    def mode(self) -> ScreenMode:
        value = self.state.value(self.STATE_KEY) if self.state is not None else None
        if value in (m.value for m in ScreenMode):
            return ScreenMode(value)
        return ScreenMode(getattr(self.config, "screen_mode", "off"))

    @property
    def since(self) -> float | None:
        if self.state is None:
            return None
        fact = self.state.get(self.STATE_KEY) if hasattr(self.state, "get") else None
        return getattr(fact, "observed_at", None)

    def set_mode(self, mode: ScreenMode | str, *, actor_kind: str, actor_id: str = "owner",
                 interactive: bool = True) -> tuple[bool, str]:
        """Switch screen awareness. Only the user can switch it on; anything can switch it off."""
        mode = ScreenMode(mode)
        previous = self.mode
        if mode != ScreenMode.OFF and not (actor_kind == "user" and interactive):
            if self.audit is not None:
                self.audit.record(actor=f"{actor_kind}:{actor_id}", action="screen_awareness_refused", ok=False,
                                  summary=f"refused to turn screen awareness {mode.value}: only the user can")
            return False, "Only you can turn screen awareness on."
        if self.state is not None:
            self.state.set(self.STATE_KEY, mode.value, confidence=Confidence.KNOWN,
                           provenance=Provenance(ProvenanceKind.USER_STATEMENT if actor_kind == "user" else
                                                 ProvenanceKind.CONFIGURATION, f"{actor_kind}:{actor_id}"))
        if mode == ScreenMode.OFF:
            self._last_window, self._last_windows = None, set()
        if self.audit is not None:
            self.audit.record(actor=f"{actor_kind}:{actor_id}", action="screen_awareness", ok=True,
                              summary=f"screen awareness {previous.value} → {mode.value}")
        self._emit("SCREEN_AWARENESS_CHANGED", {"from": previous.value, "to": mode.value,
                                                "by": f"{actor_kind}:{actor_id}"},
                   Severity.WARNING if mode == ScreenMode.WATCHING else Severity.INFO)
        return True, self.describe_mode(mode)

    def describe_mode(self, mode: ScreenMode | None = None) -> str:
        mode = mode or self.mode
        if mode == ScreenMode.OFF:
            return "Screen awareness is off: I can't see your screen."
        if mode == ScreenMode.ON_REQUEST:
            return "Screen awareness is on, on request: I'll look at your screen only when you ask."
        every = getattr(self.config, "screen_interval_s", 30.0)
        return (f"Screen awareness is on and watching: I check the active window every {every:g} seconds and read the "
                "screen when something changes, and I'll tell you about errors, permission dialogs or failed builds. "
                "Say 'stop watching my screen' to stop.")

    def status(self) -> dict[str, Any]:
        available, detail = self.backend.capture_available()
        return {"mode": self.mode.value, "since": self.since, "capture_available": available,
                "capture_method": detail, "interval_s": getattr(self.config, "screen_interval_s", 30.0),
                "captures": self.captures, "last_seen": self.last.ts if self.last else None,
                "last_state": self.last.to_dict() if self.last else None, "last_error": self.last_error,
                "description": self.describe_mode()}

    # -- looking -------------------------------------------------------------------------------------------------
    def _require_access(self) -> None:
        if self.mode == ScreenMode.OFF:
            raise ScreenAccessDenied("Screen awareness is off, so I can't see your screen. Say 'turn on screen "
                                     "awareness' if you'd like me to look when you ask.")
        ok, why = self.backend.capture_available()
        if not ok:
            raise ScreenAccessDenied(f"I can't capture the screen here: {why}.")

    async def look(self, *, reason: str, by: str = "user", session_id: str | None = None, read_text: bool = True,
                   ui: bool = False, question: str | None = None) -> tuple[ScreenState, Any]:
        """Capture the screen now (only when awareness is on). Returns (state, observation)."""
        self._require_access()
        async with self._lock:
            window = await self.backend.active_window()
            windows = await self.backend.windows()
            try:
                data = await self.backend.capture()
            except Exception as exc:
                self.last_error = str(exc)
                self._audit(by, reason, False, str(exc))
                raise ScreenAccessDenied(f"the screen capture failed ({exc})") from None
            self.captures += 1
            obs = self.store.ingest(Attachment(name=f"screen-{int(self._now())}.png", data=data,
                                               origin=InputOrigin.SCREEN_CAPTURE, kind=InputKind.SCREENSHOT),
                                    session_id=session_id, sensitivity=Sensitivity.PRIVATE)
            self._audit(by, reason, True, f"captured the screen ({obs.width}×{obs.height}) as {obs.id}")
            state = ScreenState(new_id("scr"), self._now(), (window.app if window else ""),
                                (window.title if window else ""), windows, obs.id, source="capture")
            text, confidences, engine = "", {}, ""
            if read_text and self.ocr is not None and self.ocr.status()[0]:
                try:
                    result = await self.ocr.read(obs)
                    text, engine = result.text, result.engine
                    confidences = {l.text: l.confidence for l in result.lines if l.confidence is not None}
                    obs.derived["ocr"] = result.to_dict()
                    state.source = "capture + OCR"
                except Exception as exc:
                    state.notes.append(f"couldn't read the text on screen ({exc})")
            if ui and self.vision is not None:
                try:
                    vr = await self.vision.ui_elements(obs, context=question or "")
                    if isinstance(vr.data, dict):
                        state.ui = vr.data
                        text = text or " ".join(str(e.get("label", "")) for e in vr.data.get("elements", []))
                    state.source = "capture + vision"
                except Exception as exc:
                    state.notes.append(f"couldn't identify the on-screen elements ({exc})")
            state.ocr_engine = engine
            if safety.contains_secret(text):
                # a password or key is visible: keep nothing of the image and only a scrubbed reading
                obs.sensitivity = Sensitivity.SECRET
                obs.derived["ocr"] = {"text": safety.scrub(text), "engine": engine, "scrubbed": True}
                self.store.update(obs)
                self.store.forget(obs.id, by="system:privacy")
                state.observation_id = None
                state.notes.append("credentials were visible, so I deleted the capture and kept only a scrubbed "
                                   "reading")
                text = safety.scrub(text)
            else:
                obs.status = Status.PROCESSED
                obs.labels = list(dict.fromkeys([window.app] if window and window.app else []))
                self.store.update(obs)
            state.text_excerpt = safety.scrub(text)[:2000]
            state.text_hash = hashlib.sha256(text.encode()).hexdigest()[:16] if text else ""
            state.detections = detect(text, state.window_title, confidences=confidences,
                                      source="ocr" if engine else "vision")
            if self.last is not None and self.last.observation_id and state.observation_id and \
                    (images.pillow_available() or (obs.width or 0) * (obs.height or 0) <= _PURE_PYTHON_DIFF_PIXELS):
                prev = self.store.get(self.last.observation_id)
                if prev is not None and prev.available:
                    try:
                        # off the event loop: decoding two screenshots without Pillow takes a moment
                        diff = await asyncio.to_thread(images.pixel_diff, self.store.data(prev), data)
                        state.pixel_change = diff.changed_fraction if diff.comparable else None
                    except Exception:
                        pass
            self._record(state)
            return state, (obs if state.observation_id else None)

    async def tick(self) -> list[tuple[str, dict[str, Any]]]:
        """One watching check (the runtime calls this periodically). Cheap unless something changed."""
        if self.mode != ScreenMode.WATCHING or not self.backend.capture_available()[0]:
            return []
        self._ticks += 1
        window = await self.backend.active_window()
        titles = {w.title for w in await self.backend.windows() if w.title}
        changed = window is not None and (self._last_window is None or window.title != self._last_window.title or
                                          window.app != self._last_window.app)
        events: list[tuple[str, dict[str, Any]]] = []
        if changed and self._last_window is not None and window.app != self._last_window.app:
            events.append(("APPLICATION_CHANGED", {"from": self._last_window.app, "to": window.app,
                                                   "window": window.title}))
        for gone in sorted(self._last_windows - titles)[:3]:
            events.append(("APPLICATION_CLOSED", {"window": gone}))
        self._last_window, self._last_windows = window, titles
        every = max(1, int(getattr(self.config, "screen_capture_every", 4)))
        if changed or self._ticks % every == 1 or every == 1:
            previous = self.last
            try:
                state, _obs = await self.look(reason="watching (periodic check)", by="system:screen-watch")
                events += changes(previous, state)
            except ScreenAccessDenied as exc:
                self.last_error = str(exc)
        for etype, payload in events:
            self._emit(etype, payload, Severity.WARNING if etype in ("SCREEN_ERROR_DETECTED", "BUILD_FAILED",
                                                                     "DIALOG_APPEARED") else Severity.INFO)
        return events

    async def check_absent(self, texts: list[str], *, by: str = "system:verifier") -> tuple[bool | None, str]:
        """Visual verification: is this text (an error message) gone from the screen? None when it can't be checked."""
        try:
            state, _obs = await self.look(reason="checking the result on screen", by=by)
        except ScreenAccessDenied as exc:
            return None, str(exc)
        if not state.ocr_engine:
            return None, "the screen was captured, but no OCR engine could read its text"
        visible = state.text_excerpt.lower()
        still = [t for t in texts if t and t.lower()[:60] in visible]
        if still:
            return False, f"still on screen: \"{still[0][:120]}\""
        return True, "the message is no longer on screen"

    # -- bookkeeping -------------------------------------------------------------------------------------------
    def _record(self, state: ScreenState) -> None:
        self.last = state
        db = getattr(self.store, "db", None)
        if db is not None:
            db.execute("INSERT INTO screen_states(id, ts, active_app, window_title, observation_id, data) "
                       "VALUES(?,?,?,?,?,?)", (state.id, state.ts, state.active_app, state.window_title,
                                               state.observation_id, dumps(state.to_dict())))
            db.execute("DELETE FROM screen_states WHERE ts < ?", (state.ts - 7 * 86400,))
        if self.state is not None:
            self.state.set("screen.active_app", state.active_app, confidence=Confidence.OBSERVED,
                           provenance=Provenance(ProvenanceKind.SCREEN_CAPTURE, state.id), ttl=300)
            self.state.set("screen.window", state.window_title, confidence=Confidence.OBSERVED,
                           provenance=Provenance(ProvenanceKind.SCREEN_CAPTURE, state.id), ttl=300)
            self.state.set("screen.detected", [d.kind for d in state.detections], confidence=Confidence.OBSERVED,
                           provenance=Provenance(ProvenanceKind.SCREEN_CAPTURE, state.id), ttl=300)
        if self.world is not None and state.active_app:
            self.world.upsert_entity("application", state.active_app, {"window": state.window_title[:120],
                                                                       "seen_at": state.ts}, source="screen")
        self._emit("SCREEN_CAPTURED", {"state_id": state.id, "observation_id": state.observation_id,
                                       "active_app": state.active_app, "detections": [d.kind for d in state.detections]})

    def history(self, limit: int = 10) -> list[ScreenState]:
        db = getattr(self.store, "db", None)
        if db is None:
            return [self.last] if self.last else []
        rows = db.query("SELECT data FROM screen_states ORDER BY ts DESC LIMIT ?", (limit,))
        return [ScreenState.from_dict(loads(r["data"], {})) for r in rows]

    def _audit(self, by: str, reason: str, ok: bool, summary: str) -> None:
        if self.audit is not None:
            self.audit.record(actor=by, action="screen_capture", ok=ok, summary=summary,
                              params={"reason": reason[:200], "mode": self.mode.value})

    def _emit(self, etype: str, payload: dict[str, Any], severity: Severity = Severity.INFO) -> None:
        if self.bus is not None:
            from jarvis.events.types import Event
            self.bus.emit(Event(etype, "screen", payload, severity=severity))

    def _now(self) -> float:
        return self.clock.now() if self.clock else __import__("time").time()


def changes(before: ScreenState | None, after: ScreenState) -> list[tuple[str, dict[str, Any]]]:
    """Meaningful differences between two screen states, as events (new problems only, never repeats)."""
    events: list[tuple[str, dict[str, Any]]] = []
    old = {(d.kind, d.evidence) for d in (before.detections if before else [])}
    old_kinds = {d.kind for d in (before.detections if before else [])}
    base = {"state_id": after.id, "active_app": after.active_app, "window": after.window_title,
            "observation_id": after.observation_id}
    for d in after.detections:
        if (d.kind, d.evidence) in old:
            continue
        payload = {**base, "evidence": d.evidence, "confidence": d.confidence, "source": d.source}
        if d.kind == "build_failed":
            events.append(("BUILD_FAILED", payload))
        elif d.kind == "build_completed" and "build_completed" not in old_kinds:
            events.append(("BUILD_COMPLETED", {**payload, "source": "screen"}))
        elif d.kind in ("permission_prompt", "dialog") and d.kind not in old_kinds:
            events.append(("DIALOG_APPEARED", {**payload, "permission": d.kind == "permission_prompt"}))
        elif d.kind == "error" and "error" not in old_kinds:
            events.append(("SCREEN_ERROR_DETECTED", payload))
    unique: dict[str, tuple[str, dict[str, Any]]] = {}
    for etype, payload in events:
        unique.setdefault(etype, (etype, payload))        # one event per kind per change (the first evidence)
    events = list(unique.values())
    if before is not None and after.pixel_change is not None and after.pixel_change >= 0.5 and \
            before.text_hash != after.text_hash and not events:
        events.append(("IMPORTANT_UI_CHANGE", {**base, "changed_fraction": after.pixel_change}))
    if events:
        events.append(("SCREEN_STATE_CHANGED", {**base, "events": [e[0] for e in events]}))
    return events
