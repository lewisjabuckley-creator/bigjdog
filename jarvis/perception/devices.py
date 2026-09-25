"""Which input devices this computer has (Phase 4 §28): for awareness only — nothing is ever switched on.

Displays, cameras, microphones, keyboards, mice and USB devices are listed from what the operating system reports
(Windows: one CIM query through PowerShell; Linux: /sys and /proc; macOS: system_profiler). Discovery runs in a
thread with a timeout and is cached. "Not detected" is a normal, honest answer: in particular a missing microphone
changes nothing, because JARVIS is used by typing.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from jarvis.platforms import hidden_window_kwargs

KINDS = ("display", "camera", "microphone", "keyboard", "mouse", "usb")


@dataclass
class InputDevice:
    kind: str
    name: str
    detail: str = ""
    status: str = "detected"         # detected | problem

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "detail": self.detail, "status": self.status, "active": False}


@dataclass
class DeviceReport:
    devices: list[InputDevice] = field(default_factory=list)
    checked_at: float = 0.0
    method: str = ""
    errors: list[str] = field(default_factory=list)

    def of(self, kind: str) -> list[InputDevice]:
        return [d for d in self.devices if d.kind == kind]

    def to_dict(self) -> dict[str, Any]:
        return {"checked_at": self.checked_at, "method": self.method, "errors": self.errors,
                "devices": [d.to_dict() for d in self.devices],
                "counts": {k: len(self.of(k)) for k in KINDS}}


class DeviceDiscovery:
    def __init__(self, *, ttl_s: float = 300.0, timeout_s: float = 20.0, platform: str | None = None,
                 simulated: DeviceReport | None = None) -> None:
        self.ttl_s = ttl_s
        self.timeout_s = timeout_s
        self.platform = platform or sys.platform
        self.simulated = simulated
        self._cache: DeviceReport | None = None

    async def discover(self, *, refresh: bool = False) -> DeviceReport:
        if self.simulated is not None:
            return self.simulated
        now = time.time()
        if self._cache is not None and not refresh and now - self._cache.checked_at < self.ttl_s:
            return self._cache
        try:
            report = await asyncio.wait_for(asyncio.to_thread(self._discover), timeout=self.timeout_s)
        except asyncio.TimeoutError:
            report = DeviceReport(errors=["device discovery took too long"])
        except Exception as exc:
            report = DeviceReport(errors=[f"device discovery failed: {exc}"])
        report.checked_at = now
        self._cache = report
        return report

    def cached(self) -> DeviceReport | None:
        return self.simulated or self._cache

    def _discover(self) -> DeviceReport:
        if self.platform == "win32":
            return _windows()
        if self.platform == "darwin":
            return _macos()
        return _linux()


# -- Windows ----------------------------------------------------------------------------------------------------------

_PS = ("Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -in "
       "@('Camera','Image','AudioEndpoint','Keyboard','Mouse','Monitor','USB','HIDClass') } | "
       "Select-Object Name,PNPClass,Status | ConvertTo-Json -Compress")


def _windows() -> DeviceReport:
    report = DeviceReport(method="Windows device manager (CIM)")
    try:
        proc = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS], capture_output=True,
                              timeout=18, **hidden_window_kwargs())
        entries = json.loads(proc.stdout.decode("utf-8", errors="replace") or "[]")
    except Exception as exc:
        report.errors.append(f"PowerShell device query failed: {exc}")
        entries = []
    if isinstance(entries, dict):
        entries = [entries]
    seen = set()
    for e in entries:
        name, cls, status = str(e.get("Name") or ""), str(e.get("PNPClass") or ""), str(e.get("Status") or "")
        kind = {"Camera": "camera", "Image": "camera", "Keyboard": "keyboard", "Mouse": "mouse", "Monitor": "display",
                "USB": "usb"}.get(cls)
        if cls == "AudioEndpoint":
            kind = "microphone" if re.search(r"micro|mic\b|line in|headset|array", name, re.I) else None
        if kind is None or (kind, name) in seen:
            continue
        seen.add((kind, name))
        report.devices.append(InputDevice(kind, name, cls, "detected" if status in ("OK", "") else "problem"))
    try:
        import ctypes
        monitors = ctypes.windll.user32.GetSystemMetrics(80)       # SM_CMONITORS
        width, height = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
        if not report.of("display") and monitors:
            for i in range(monitors):
                report.devices.append(InputDevice("display", f"Display {i + 1}",
                                                  f"primary {width}×{height}" if i == 0 else ""))
    except Exception:
        pass
    return report


# -- Linux ------------------------------------------------------------------------------------------------------------

def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _linux() -> DeviceReport:
    report = DeviceReport(method="/sys and /proc")
    for status_file in sorted(glob.glob("/sys/class/drm/card*-*/status")):
        if _read(status_file) == "connected":
            connector = os.path.basename(os.path.dirname(status_file)).split("-", 1)[-1]
            report.devices.append(InputDevice("display", connector, "connected"))
    for dev in sorted(glob.glob("/sys/class/video4linux/video*")):
        name = _read(os.path.join(dev, "name")) or os.path.basename(dev)
        if name not in {d.name for d in report.of("camera")}:
            report.devices.append(InputDevice("camera", name, f"/dev/{os.path.basename(dev)}"))
    cards = _read("/proc/asound/cards")
    pcm = glob.glob("/proc/asound/card*/pcm*c")          # capture devices
    for path in sorted(pcm):
        card = re.search(r"card(\d+)", path)
        label = re.search(rf"^\s*{card.group(1)} \[[^\]]*\]:\s*(.+)$", cards, re.M) if card else None
        report.devices.append(InputDevice("microphone", (label.group(1).strip() if label else path)[:80],
                                          "audio capture device"))
    blocks = _read("/proc/bus/input/devices").split("\n\n")
    for block in blocks:
        name = re.search(r'N: Name="([^"]+)"', block)
        handlers = re.search(r"H: Handlers=(.+)", block)
        if not name or not handlers:
            continue
        h = handlers.group(1)
        if "kbd" in h and re.search(r"keyboard", name.group(1), re.I):
            report.devices.append(InputDevice("keyboard", name.group(1)))
        elif "mouse" in h:
            report.devices.append(InputDevice("mouse", name.group(1)))
    for dev in sorted(glob.glob("/sys/bus/usb/devices/*/product")):
        product = _read(dev)
        if product and "hub" not in product.lower() and "host controller" not in product.lower():
            report.devices.append(InputDevice("usb", product))
    return report


# -- macOS ------------------------------------------------------------------------------------------------------------

def _macos() -> DeviceReport:
    report = DeviceReport(method="system_profiler")
    try:
        proc = subprocess.run(["system_profiler", "-json", "SPDisplaysDataType", "SPCameraDataType",
                               "SPAudioDataType", "SPUSBDataType"], capture_output=True, timeout=18)
        data = json.loads(proc.stdout or b"{}")
    except Exception as exc:
        report.errors.append(f"system_profiler failed: {exc}")
        return report
    for gpu in data.get("SPDisplaysDataType", []):
        for d in gpu.get("spdisplays_ndrvs", []):
            report.devices.append(InputDevice("display", d.get("_name", "display"), d.get("_spdisplays_resolution", "")))
    for cam in data.get("SPCameraDataType", []):
        report.devices.append(InputDevice("camera", cam.get("_name", "camera")))
    for audio in data.get("SPAudioDataType", []):
        for item in audio.get("_items", []):
            if item.get("coreaudio_device_input"):
                report.devices.append(InputDevice("microphone", item.get("_name", "microphone")))

    def usb(items: list[dict[str, Any]]) -> None:
        for item in items:
            name = item.get("_name", "")
            if name and "hub" not in name.lower():
                report.devices.append(InputDevice("usb", name))
            usb(item.get("_items", []))
    for bus in data.get("SPUSBDataType", []):
        usb(bus.get("_items", []))
    return report
