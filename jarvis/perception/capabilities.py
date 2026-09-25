"""What JARVIS can perceive right now, stated accurately (Phase 4 §27).

"What can you see?" and "What inputs do you have access to?" are answered from live checks, never from a fixed
list: the model inventory (is a vision model installed and allowed?), the OCR engines present, PDF support, the
screen-awareness switch and whether capture works here, and the devices the operating system reports. Nothing is
claimed that isn't there; nothing is switched on to find out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jarvis.perception.images import pillow_available

AVAILABLE, DISABLED, NOT_AVAILABLE, NOT_IMPLEMENTED, NOT_CONNECTED = \
    "available", "disabled", "not available", "not implemented", "not connected"


@dataclass
class CapabilityLine:
    name: str
    state: str
    detail: str = ""

    def render(self) -> str:
        return f"{self.name}: {self.state.upper()}" + (f" — {self.detail}" if self.detail else "")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "state": self.state, "detail": self.detail}


async def report(perception: Any, *, refresh_devices: bool = False) -> list[CapabilityLine]:
    svc = perception.svc
    lines: list[CapabilityLine] = [CapabilityLine("Text input", AVAILABLE, "typing (the main way to talk to me)")]
    router = svc.router
    conversation = router.status().get("conversation_model") if router.inventory else None
    lines.append(CapabilityLine("Text reasoning", AVAILABLE if conversation else NOT_AVAILABLE,
                                conversation or "no language model is reachable"))
    ok, detail = perception.vision.status()
    lines.append(CapabilityLine("Vision model", AVAILABLE if ok else NOT_AVAILABLE, detail))
    ok, detail = perception.ocr.status()
    lines.append(CapabilityLine("OCR (reading text in images)", AVAILABLE if ok else NOT_AVAILABLE, detail))
    from jarvis.perception.documents import pdf_backend
    pdf = pdf_backend()
    lines.append(CapabilityLine("Documents", AVAILABLE, "text, Markdown, code, JSON, CSV, logs, config files; PDFs "
                                + ({"pypdf": "with pypdf", "pdftotext": "with pdftotext"}.get(
                                    pdf, "with a basic reader (py -m pip install pypdf reads them better)"))))
    lines.append(CapabilityLine("Image shrinking", AVAILABLE if pillow_available() else NOT_AVAILABLE,
                                "Pillow" if pillow_available() else "large images can't be shrunk without Pillow "
                                                                    "(py -m pip install pillow)"))
    screen = perception.screen.status()
    if screen["mode"] == "off":
        lines.append(CapabilityLine("Screen capture", DISABLED, "off until you turn it on ('turn on screen "
                                                                "awareness')" + ("" if screen["capture_available"]
                                                                                  else f"; also: {screen['capture_method']}")))
    elif not screen["capture_available"]:
        lines.append(CapabilityLine("Screen capture", NOT_AVAILABLE, screen["capture_method"]))
    else:
        lines.append(CapabilityLine("Screen capture", AVAILABLE, "watching" if screen["mode"] == "watching"
                                    else "when you ask"))
    devices = await perception.devices.discover(refresh=refresh_devices)
    cameras = devices.of("camera")
    lines.append(CapabilityLine("Camera", NOT_IMPLEMENTED if cameras else NOT_CONNECTED,
                                f"{len(cameras)} detected ({cameras[0].name}); camera input isn't built yet"
                                if cameras else "none detected"))
    mics = devices.of("microphone")
    lines.append(CapabilityLine("Microphone", NOT_IMPLEMENTED if mics else NOT_AVAILABLE,
                                f"{mics[0].name} detected, but voice input isn't built yet; typing works" if mics else
                                "none set up; typing works for everything"))
    displays = devices.of("display")
    if displays:
        lines.append(CapabilityLine("Displays", AVAILABLE, ", ".join(d.name for d in displays[:3])))
    effective = svc.modes.effective()
    lines.append(CapabilityLine("Offline mode", "on" if effective.local_only else "off",
                                "local models only" if effective.local_only else
                                ("images stay local" if not perception.vision.allows_cloud() else
                                 "cloud vision allowed for non-private images")))
    return lines


def render(lines: list[CapabilityLine]) -> str:
    return "\n".join(line.render() for line in lines)
