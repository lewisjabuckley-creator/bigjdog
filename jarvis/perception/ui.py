"""User-interface elements, and the interface future computer control will use (Phase 4 §10, §36).

Detected elements are structured (type, label, position, state, where the detection came from and how sure it is),
so later phases can act on them. Acting is *not* part of Phase 4: :class:`UIController` defines click / type / scroll /
select / open / close, and :class:`UIActionTool` shows how each action will run — as a consequential tool, so every
action goes through the permission system (approval), is audited, and is verified afterwards by looking at the
screen again. No controller ships, so the tool is not registered and JARVIS cannot click anything.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from jarvis.core.types import RiskLevel
from jarvis.permissions.model import PermissionLevel
from jarvis.tools.base import Tool, ToolContext, ToolResult, ToolSpec, Verification

ELEMENT_TYPES = {"window", "button", "menu", "menu item", "text field", "input", "checkbox", "toggle", "switch",
                 "radio", "dialog", "icon", "tab", "list", "list item", "notification", "link", "dropdown", "slider",
                 "label", "error", "warning", "toolbar", "status bar", "table", "image", "progress bar", "control"}


@dataclass
class UIElement:
    type: str
    label: str
    position: str = ""        # e.g. "top left", "bottom right"
    state: str = ""           # e.g. "on", "selected", "disabled", "error"
    source: str = "vision"    # vision | ocr | accessibility (future)
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        return f"{self.label} {self.type}".strip() + (f" ({self.state})" if self.state else "") + \
            (f", {self.position}" if self.position else "")


def parse_elements(data: Any, *, source: str = "vision") -> tuple[str, str, list[UIElement]]:
    """(application, window, elements) from a vision model's JSON, dropping anything malformed."""
    if not isinstance(data, dict):
        return "", "", []
    out: list[UIElement] = []
    for raw in data.get("elements") or []:
        if not isinstance(raw, dict):
            continue
        kind = " ".join(str(raw.get("type") or "").lower().split())[:30]
        label = " ".join(str(raw.get("label") or "").split())[:80]
        if not label:
            continue
        if kind not in ELEMENT_TYPES:
            kind = "control"
        out.append(UIElement(kind, label, str(raw.get("position") or "")[:30], str(raw.get("state") or "")[:30],
                             source))
    return str(data.get("application") or "")[:60], str(data.get("window") or "")[:80], out[:40]


def describe_elements(application: str, window: str, elements: list[UIElement]) -> str:
    lines = []
    if application:
        lines.append(f"Application: {application}")
    lines.append(f"Window: {window or 'unknown'}")
    if elements:
        lines.append("Elements:")
        lines += [f"- {e.describe()}" for e in elements]
    return "\n".join(lines)


def find(elements: list[UIElement], phrase: str) -> list[UIElement]:
    """Elements matching a natural reference: "the button on the left", "the red warning", "the Wi-Fi toggle"."""
    words = [w for w in phrase.lower().replace("-", " ").split() if w not in ("the", "a", "an", "on", "at", "in",
                                                                             "that", "this")]
    positions = {"left", "right", "top", "bottom", "middle", "centre", "center"}
    wanted_pos = [w for w in words if w in positions]
    wanted_type = [w for w in words if w in {t.split()[0] for t in ELEMENT_TYPES}]
    rest = [w for w in words if w not in positions and w not in wanted_type]
    out = []
    for e in elements:
        text = f"{e.label} {e.type} {e.state}".lower().replace("-", " ")
        if wanted_type and not any(t in e.type for t in wanted_type):
            continue
        if wanted_pos and not all(p in e.position.lower() for p in wanted_pos):
            continue
        if rest and not any(r in text for r in rest):
            continue
        out.append(e)
    return out


class UIAction(StrEnum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    SELECT = "select"
    OPEN = "open"
    CLOSE = "close"


@dataclass
class UIActionRequest:
    action: UIAction
    target: str                  # an element label/description, or an application for open/close
    text: str = ""               # for TYPE
    expected: str = ""           # what the screen should show afterwards (checked by looking again)


class UIController(ABC):
    """Future: performs one UI action on the real desktop. Implementations must be driven only by UIActionTool."""

    @abstractmethod
    async def perform(self, request: UIActionRequest) -> tuple[bool, str]: ...


class UIActionTool(Tool):
    """How computer control will plug in: consequential (always approved), audited, verified by looking again."""
    spec = ToolSpec(
        name="ui_action",
        description="Click, type, scroll, select, open or close something on screen (needs the user's approval).",
        parameters={"type": "object", "properties": {
            "action": {"type": "string", "enum": [a.value for a in UIAction]},
            "target": {"type": "string"}, "text": {"type": "string", "default": ""},
            "expected": {"type": "string", "default": ""}}, "required": ["action", "target"]},
        level=PermissionLevel.EXECUTE_CONSEQUENTIAL, risk=RiskLevel.HIGH, reversible=False,
        side_effects=("desktop.input",), verification="the screen is looked at again and compared with 'expected'",
        category="perception")

    def __init__(self, controller: UIController, screen: Any) -> None:
        self.controller = controller
        self.screen = screen

    def preview(self, args: dict[str, Any]) -> str:
        return f"{args.get('action')} {args.get('target')}" + (f" (type \"{args['text']}\")" if args.get("text") else "")

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ok, detail = await self.controller.perform(UIActionRequest(UIAction(args["action"]), args["target"],
                                                                   args.get("text", ""), args.get("expected", "")))
        return ToolResult(ok, detail, {"action": args["action"], "target": args["target"]})

    async def verify(self, args: dict[str, Any], result: ToolResult, ctx: ToolContext) -> Verification:
        expected = args.get("expected") or ""
        if not expected or self.screen is None:
            return Verification.not_performed("nothing to compare the screen with")
        state, _obs = await self.screen.look(reason="verifying a UI action", by="system:verifier")
        seen = expected.lower() in (state.text_excerpt or "").lower()
        return Verification(True, seen, "screen", "the expected state is visible" if seen else
                            "the expected state isn't visible on screen")
