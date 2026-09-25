"""Perception as tools, so the planner, agents and the chat model can use it through the normal pipeline (Phase 4 §21,
§34, §38): request → authorization → permission → execution → verification → audit.

All are observe-level (they only look). Their output is external content: when a model has read it, anything more
than observing needs the user's explicit approval (see ``EXTERNAL_CONTENT_TOOLS``). ``screen_look`` and
``screen_check`` work only while the user has screen awareness switched on.
"""

from __future__ import annotations

import os
from typing import Any

from jarvis.core.types import Provenance, ProvenanceKind
from jarvis.perception.inputs import Attachment, InputOrigin
from jarvis.perception.screen import ScreenAccessDenied
from jarvis.perception.store import InputRejected
from jarvis.permissions.model import PermissionLevel
from jarvis.tools.base import Tool, ToolContext, ToolResult, ToolSpec, Verification

# tools whose results carry content from images, documents, files or the screen: data, never instructions
EXTERNAL_CONTENT_TOOLS = frozenset({"image_analyze", "image_read_text", "document_read", "screen_look", "file_read",
                                    "file_search"})


def _obs(perception: Any, obs_id: str) -> Any:
    obs = perception.store.get(str(obs_id or ""))
    if obs is None:
        raise InputRejected(f"there's no input called {obs_id}")
    return obs


class ImageAnalyzeTool(Tool):
    spec = ToolSpec(
        name="image_analyze",
        description="Look at an image or screenshot the user shared (by its id) and answer a question about it with a "
                    "vision model. Returns what is visible, the text read from it and any recognised problem.",
        parameters={"type": "object", "properties": {
            "observation_id": {"type": "string"}, "question": {"type": "string", "default": "What does it show?"}},
            "required": ["observation_id"]},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="read-only", timeout_s=300.0,
        category="perception", resources=("vision",))

    def __init__(self, perception: Any) -> None:
        self.perception = perception

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            obs = _obs(self.perception, args["observation_id"])
        except InputRejected as exc:
            return ToolResult(False, str(exc), error="not_found")
        if not args.get("question") and obs.derived.get("finding"):
            # what was already found (a plan's first step re-reads the analysis instead of repeating it)
            d = obs.derived
            return ToolResult(True, d["finding"][:200], {
                "answer": d["finding"], "basis": "earlier analysis", "finding": d["finding"],
                "problem": d.get("problem"), "observation_id": obs.id,
                "text": ((d.get("ocr") or {}).get("text") or "")[:3000] or None}, provenance=obs.provenance())
        seen = await self.perception.understand([obs], args.get("question") or "What does it show?",
                                                interactive=ctx.actor.interactive)
        data = {"answer": seen.text, "basis": seen.basis, "model": seen.model, "finding": seen.finding,
                "problem": seen.problem, "notes": seen.notes, "observation_id": obs.id,
                "text": seen.ocr.text[:3000] if seen.ocr else None}
        return ToolResult(not seen.failed, seen.finding or seen.text[:200], data,
                          error=None if not seen.failed else "unavailable",
                          provenance=seen.provenance[0] if seen.provenance else obs.provenance())

    async def verify(self, args: dict[str, Any], result: ToolResult, ctx: ToolContext) -> Verification:
        return Verification(True, result.ok, "read-only", "analysis recorded" if result.ok else "not analysed")


class ImageReadTextTool(Tool):
    spec = ToolSpec(
        name="image_read_text",
        description="Read the text in an image or screenshot the user shared (OCR), with confidence.",
        parameters={"type": "object", "properties": {"observation_id": {"type": "string"}},
                    "required": ["observation_id"]},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="read-only", timeout_s=180.0,
        category="perception")

    def __init__(self, perception: Any) -> None:
        self.perception = perception

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            obs = _obs(self.perception, args["observation_id"])
        except InputRejected as exc:
            return ToolResult(False, str(exc), error="not_found")
        seen = await self.perception.read_text(obs)
        if seen.failed or seen.ocr is None:
            return ToolResult(False, seen.text, error="unavailable")
        return ToolResult(True, f"read {len(seen.ocr.lines)} line(s) with {seen.ocr.engine}",
                          {"text": seen.ocr.marked_text(), "engine": seen.ocr.engine,
                           "mean_confidence": seen.ocr.mean_confidence, "measured": seen.ocr.measured,
                           "problem": None}, provenance=Provenance(ProvenanceKind.OCR, seen.ocr.engine, obs.id))


class DocumentReadTool(Tool):
    spec = ToolSpec(
        name="document_read",
        description="Read a document or file (text, Markdown, code, JSON, CSV, log, config, PDF) without loading all of "
                    "it: an outline, the parts relevant to a question, its requirements, or where a topic is mentioned. "
                    "Give a path, or the id of a document the user shared.",
        parameters={"type": "object", "properties": {
            "path": {"type": "string"}, "observation_id": {"type": "string"},
            "question": {"type": "string", "default": ""},
            "mode": {"type": "string", "enum": ["answer", "summary", "outline", "requirements", "references"],
                     "default": "outline"},
            "topic": {"type": "string", "default": ""}}},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="read-only", path_params=("path",),
        timeout_s=300.0, category="perception")

    def __init__(self, perception: Any) -> None:
        self.perception = perception

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            if args.get("observation_id"):
                obs = _obs(self.perception, args["observation_id"])
            elif args.get("path"):
                path = os.path.expanduser(args["path"])
                if not os.path.isabs(path):
                    path = os.path.join(ctx.cwd or os.getcwd(), path)
                obs = self.perception.store.ingest(Attachment(path=path, origin=InputOrigin.TOOL))
            else:
                return ToolResult(False, "give a path or an observation_id", error="invalid")
        except InputRejected as exc:
            return ToolResult(False, str(exc), error="rejected")
        seen = await self.perception.document(obs, args.get("question") or "", mode=args.get("mode") or "outline",
                                              topic=args.get("topic") or "")
        return ToolResult(not seen.failed, seen.finding or seen.text[:200],
                          {"text": seen.text[:6000], "observation_id": obs.id, "notes": seen.notes, **{
                              k: v for k, v in seen.data.items() if k in ("requirements", "references", "parts")}},
                          provenance=obs.provenance())


class ScreenLookTool(Tool):
    spec = ToolSpec(
        name="screen_look",
        description="Look at the user's screen now (only works while the user has screen awareness switched on): the "
                    "active application and window, the text on screen, and any error, dialog or build result.",
        parameters={"type": "object", "properties": {"question": {"type": "string", "default": ""},
                                                     "elements": {"type": "boolean", "default": False}}},
        level=PermissionLevel.OBSERVE, verification="read-only", timeout_s=120.0, category="perception")

    def __init__(self, perception: Any) -> None:
        self.perception = perception

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            state, obs = await self.perception.screen.look(reason=args.get("question") or "requested by a tool",
                                                           by=ctx.actor.subject, ui=bool(args.get("elements")))
        except ScreenAccessDenied as exc:
            return ToolResult(False, str(exc), error="screen_off")
        return ToolResult(True, f"{state.active_app or 'screen'}: {state.window_title}"[:200],
                          {"state": state.to_dict(), "description": state.describe(),
                           "observation_id": obs.id if obs else None},
                          provenance=Provenance(ProvenanceKind.SCREEN_CAPTURE, state.id))


class ScreenCheckTool(Tool):
    """Visual verification: is the error still on screen?"""
    spec = ToolSpec(
        name="screen_check",
        description="Check whether some text (for example an error message) is still visible on the screen.",
        parameters={"type": "object", "properties": {"absent": {"type": "array", "items": {"type": "string"}}},
                    "required": ["absent"]},
        level=PermissionLevel.OBSERVE, verification="read-only", timeout_s=120.0, category="perception")

    def __init__(self, perception: Any) -> None:
        self.perception = perception

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        passed, detail = await self.perception.screen.check_absent([str(t) for t in args.get("absent") or []],
                                                                   by=ctx.actor.subject)
        return ToolResult(True, detail, {"passed": passed, "detail": detail, "checked": passed is not None},
                          provenance=Provenance(ProvenanceKind.SCREEN_CAPTURE, "screen check"))


def register_perception_tools(registry: Any, perception: Any) -> None:
    for tool in (ImageAnalyzeTool(perception), ImageReadTextTool(perception), DocumentReadTool(perception),
                 ScreenLookTool(perception), ScreenCheckTool(perception)):
        registry.register(tool)
