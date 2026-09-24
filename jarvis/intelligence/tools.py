"""Tools the planning layer uses. Like every other tool, they only run through the tool registry, so
authorization, scope, verification and audit apply to them exactly as to anything else.

* ``plan_gate`` — an approval point. It changes nothing; it exists so that approving a plan's changes goes
  through the same permission check and "proceed?" flow as any single consequential action.
* ``plan_analyze`` / ``plan_verify`` — deterministic analysis of evidence, and independent comparison of a
  result against fresh measurements.
* ``disk_usage``, ``file_copy``, ``model_status``, ``model_unload`` — capabilities the playbooks need.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from jarvis.core.types import Provenance, ProvenanceKind, RiskLevel
from jarvis.intelligence.analysis import ANALYZERS, COMPARATORS, size_text
from jarvis.permissions.model import PermissionLevel
from jarvis.tools.base import Assessment, Tool, ToolContext, ToolResult, ToolSpec, Verification


class PlanGateTool(Tool):
    spec = ToolSpec(
        name="plan_gate",
        description="An approval point in a plan: approving it approves exactly the listed changes.",
        parameters={"type": "object", "properties": {
            "plan_id": {"type": "string"},
            "summary": {"type": "string"},
            "actions": {"type": "array", "items": {"type": "object"}},
        }, "required": ["plan_id", "summary", "actions"]},
        level=PermissionLevel.EXECUTE_CONSEQUENTIAL, risk=RiskLevel.MEDIUM, idempotent=True,
        verification="the approval was granted by the user", category="planning",
    )

    def assess(self, args: dict[str, Any]) -> Assessment:
        risky = any(not a.get("reversible", True) for a in args.get("actions", []))
        return Assessment(PermissionLevel.EXECUTE_CONSEQUENTIAL, RiskLevel.HIGH if risky else RiskLevel.MEDIUM,
                          reason="approves changes as part of a plan")

    def preview(self, args: dict[str, Any]) -> str:
        return str(args.get("summary") or "carry out the planned changes")

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult(True, f"approved: {args.get('summary')}", {"approved": True, "actions": args["actions"]})

    async def verify(self, args: dict[str, Any], result: ToolResult, ctx: ToolContext) -> Verification:
        # it only runs once the permission check passed, which for this level means the user approved
        return Verification(True, True, "approval", "approved through the permission system")


class PlanAnalyzeTool(Tool):
    spec = ToolSpec(
        name="plan_analyze",
        description="Deterministic analysis of evidence gathered by a plan (performance, disk, research).",
        parameters={"type": "object", "properties": {
            "analyzer": {"type": "string", "enum": sorted(ANALYZERS)},
            "evidence": {"type": "object"},
            "context": {"type": "object", "default": {}},
        }, "required": ["analyzer", "evidence"]},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="inference from recorded evidence",
        category="planning",
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        result = ANALYZERS[args["analyzer"]](args.get("evidence") or {}, args.get("context") or {})
        return ToolResult(True, result.get("cause") or "analysis complete", result,
                          provenance=Provenance(ProvenanceKind.INFERENCE, f"{args['analyzer']} analysis",
                                                "from evidence gathered by the plan"))

    async def verify(self, args: dict[str, Any], result: ToolResult, ctx: ToolContext) -> Verification:
        return Verification.not_performed("an analysis is inference; its conclusions are checked by verification")


class PlanVerifyTool(Tool):
    spec = ToolSpec(
        name="plan_verify",
        description="Independently check a plan's result against fresh measurements.",
        parameters={"type": "object", "properties": {"check": {"type": "string", "enum": sorted(COMPARATORS)}},
                    "required": ["check"], "additionalProperties": True},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="independent comparison", category="planning",
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        outcome = COMPARATORS[args["check"]](args)
        return ToolResult(True, f"{outcome['quality'].replace('_', ' ')}: {outcome['detail']}", outcome,
                          provenance=Provenance(ProvenanceKind.SYSTEM_STATE, "verification", args["check"]))

    async def verify(self, args: dict[str, Any], result: ToolResult, ctx: ToolContext) -> Verification:
        # this checks that the comparison itself was made; *what* it found (verified, failed, conflicting) is the
        # result, which the plan acts on
        data = result.data or {}
        return Verification(True, "quality" in data, "comparison", data.get("detail", ""))


class DiskUsageTool(Tool):
    spec = ToolSpec(
        name="disk_usage",
        description="Disk totals for a folder's drive, the size of its sub-folders, and its largest old files.",
        parameters={"type": "object", "properties": {
            "path": {"type": "string"},
            "top": {"type": "integer", "default": 10, "minimum": 1},
            "largest_files": {"type": "integer", "default": 0, "minimum": 0},
            "older_than_days": {"type": "number", "default": 30, "minimum": 0},
            "budget_s": {"type": "number", "default": 10, "minimum": 0.1},
        }, "required": ["path"]},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="read-only", path_params=("path",),
        category="filesystem",
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        import asyncio
        root = Path(os.path.expanduser(args["path"]))
        if not root.is_absolute():
            root = Path(ctx.cwd or os.getcwd()) / root
        if not root.exists():
            return ToolResult(False, f"{root} does not exist", error="not_found")
        data = await asyncio.to_thread(_disk_usage, root, args["top"], args["largest_files"],
                                       args["older_than_days"], args["budget_s"])
        summary = f"{root}: {data['percent']:.0f}% of the drive used, {size_text(data['free'])} free"
        return ToolResult(True, summary, data, provenance=Provenance(ProvenanceKind.LOCAL_FILE, str(root)))


def _disk_usage(root: Path, top: int, largest: int, older_than_days: float, budget_s: float) -> dict[str, Any]:
    started = time.monotonic()
    total, used, free = shutil.disk_usage(root)
    children: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    cutoff = time.time() - older_than_days * 86400
    truncated = False
    entries = list(root.iterdir()) if root.is_dir() else [root]
    for entry in entries:
        size = 0
        try:
            if entry.is_symlink():
                continue
            if entry.is_file():
                st = entry.stat()
                size = st.st_size
                if largest and st.st_mtime < cutoff:
                    files.append({"path": str(entry), "size": size, "age_days": round((time.time() - st.st_mtime) / 86400)})
            else:
                for dirpath, dirnames, filenames in os.walk(entry):
                    dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
                    for name in filenames:
                        p = os.path.join(dirpath, name)
                        try:
                            st = os.stat(p, follow_symlinks=False)
                        except OSError:
                            continue
                        size += st.st_size
                        if largest and st.st_mtime < cutoff:
                            files.append({"path": p, "size": st.st_size,
                                          "age_days": round((time.time() - st.st_mtime) / 86400)})
                    if time.monotonic() - started > budget_s:
                        truncated = True
                        break
        except OSError:
            continue
        children.append({"path": str(entry), "size": size, "type": "file" if entry.is_file() else "dir"})
        if time.monotonic() - started > budget_s:
            truncated = True
            break
    children.sort(key=lambda c: -c["size"])
    files.sort(key=lambda f: -f["size"])
    return {"path": str(root), "total": total, "used": used, "free": free, "percent": round(used / total * 100, 1)
            if total else 0.0, "children": children[:top], "largest_files": files[:largest] if largest else [],
            "truncated": truncated}


class FileCopyTool(Tool):
    """Copy a file or folder. Files that already exist at the destination with the same size and date are
    skipped, so running it again after an interruption just continues (safe to repeat)."""

    spec = ToolSpec(
        name="file_copy",
        description="Copy a file or folder to a destination (existing identical files are skipped).",
        parameters={"type": "object", "properties": {
            "source": {"type": "string"},
            "destination": {"type": "string"},
            "overwrite": {"type": "boolean", "default": False},
        }, "required": ["source", "destination"]},
        level=PermissionLevel.EXECUTE_REVERSIBLE, risk=RiskLevel.LOW, side_effects=("filesystem.write",),
        reversible=True, idempotent=True, verification="every source file present at the destination",
        path_params=("source", "destination"), category="filesystem", timeout_s=3600.0,
    )

    def assess(self, args: dict[str, Any]) -> Assessment:
        if args.get("overwrite"):
            return Assessment(PermissionLevel.EXECUTE_CONSEQUENTIAL, RiskLevel.MEDIUM,
                              reason="it may overwrite existing files", reversible=False)
        return Assessment(PermissionLevel.EXECUTE_REVERSIBLE, RiskLevel.LOW, reason="creates copies only")

    def preview(self, args: dict[str, Any]) -> str:
        return f"copy {args['source']} to {args['destination']}"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        import asyncio
        src = _abs(args["source"], ctx)
        dst = _abs(args["destination"], ctx)
        if not src.exists():
            return ToolResult(False, f"{src} does not exist", error="not_found")
        try:
            created, skipped, replaced = await asyncio.to_thread(_copy, src, dst, bool(args["overwrite"]), ctx)
        except OSError as exc:
            return ToolResult(False, f"copy failed: {exc}", error=str(exc))
        if ctx.cancel.is_set():
            return ToolResult(False, "copy cancelled", error="cancelled")
        summary = f"copied {len(created)} file(s) to {dst}" + (f", {skipped} already there" if skipped else "")
        return ToolResult(True, summary, {"source": str(src), "destination": str(dst), "copied": len(created),
                                          "skipped": skipped, "replaced": replaced, "files": created[:50]},
                          rollback={"created": created} if created and not replaced else None,
                          provenance=Provenance(ProvenanceKind.LOCAL_FILE, str(dst)))

    async def verify(self, args: dict[str, Any], result: ToolResult, ctx: ToolContext) -> Verification:
        from jarvis.intelligence.analysis import verify_backup
        data = result.data or {}
        outcome = verify_backup({"source": data.get("source"), "destination": data.get("destination")})
        return Verification(True, outcome["quality"] == "verified", "size_check", outcome["detail"])

    async def rollback(self, info: dict[str, Any], ctx: ToolContext) -> ToolResult:
        removed = 0
        for path in info.get("created", []):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
        return ToolResult(True, f"removed {removed} copied file(s)")


def _abs(path: str, ctx: ToolContext) -> Path:
    p = Path(os.path.expanduser(path))
    return p if p.is_absolute() else Path(ctx.cwd or os.getcwd()) / p


def _copy(src: Path, dst: Path, overwrite: bool, ctx: ToolContext) -> tuple[list[str], int, int]:
    created: list[str] = []
    skipped = replaced = 0
    pairs: list[tuple[Path, Path]] = []
    if src.is_file():
        target = dst / src.name if dst.is_dir() else dst
        pairs.append((src, target))
    else:
        for f in src.rglob("*"):
            if f.is_file() and not f.is_symlink():
                pairs.append((f, dst / f.relative_to(src)))
    for s, d in pairs:
        if ctx.cancel.is_set():
            break
        if d.exists():
            ss, ds = s.stat(), d.stat()
            if ss.st_size == ds.st_size and int(ss.st_mtime) <= int(ds.st_mtime):
                skipped += 1
                continue
            if not overwrite:
                skipped += 1
                continue
            replaced += 1
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_name(f".{d.name}.jarvis-partial")
        shutil.copy2(s, tmp)            # a copy interrupted half-way never looks complete
        os.replace(tmp, d)
        created.append(str(d))
    return created, skipped, replaced


class ModelStatusTool(Tool):
    spec = ToolSpec(
        name="model_status",
        description="Which language models are installed and loaded, and which are in use right now.",
        parameters={"type": "object", "properties": {}},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="read-only", category="models",
    )

    def __init__(self, router: Any) -> None:
        self.router = router

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            await self.router.refresh()
        except Exception:
            pass
        status = self.router.status()
        in_use = sorted({r["model"] for r in status.get("active_requests", [])})
        data = {"installed": status.get("models", []), "loaded": status.get("loaded", []), "in_use": in_use,
                "providers": status.get("providers", {})}
        loaded = ", ".join(data["loaded"]) or "none"
        return ToolResult(True, f"{len(data['installed'])} model(s) installed; loaded: {loaded}", data,
                          provenance=Provenance(ProvenanceKind.SYSTEM_STATE, "model router"))


class ModelUnloadTool(Tool):
    spec = ToolSpec(
        name="model_unload",
        description="Unload a language model from memory. It loads again automatically when next needed.",
        parameters={"type": "object", "properties": {"model": {"type": "string"}}, "required": ["model"]},
        level=PermissionLevel.EXECUTE_REVERSIBLE, risk=RiskLevel.LOW, side_effects=("model.memory",),
        reversible=True, idempotent=True, verification="the model is no longer loaded", category="models",
    )

    def __init__(self, router: Any) -> None:
        self.router = router

    def preview(self, args: dict[str, Any]) -> str:
        return f"unload the model {args['model']}"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        from jarvis.models.base import ModelError
        try:
            info = await self.router.unload(args["model"])
        except ModelError as exc:
            return ToolResult(False, f"couldn't unload {args['model']}: {exc}", error="model_error")
        return ToolResult(True, f"unloaded {info.name}", {"model": info.name},
                          provenance=Provenance(ProvenanceKind.SYSTEM_STATE, "model router"))

    async def verify(self, args: dict[str, Any], result: ToolResult, ctx: ToolContext) -> Verification:
        try:
            inventory = await self.router.refresh()
        except Exception as exc:
            return Verification(False, None, "none", f"couldn't re-read the model list: {exc}")
        name = (result.data or {}).get("model", args["model"])
        loaded = any(m.name == name and m.loaded for m in inventory)
        return Verification(True, not loaded, "model_list", "no longer loaded" if not loaded else "still loaded")


def register_intelligence_tools(registry: Any, router: Any) -> None:
    for tool in (PlanGateTool(), PlanAnalyzeTool(), PlanVerifyTool(), DiskUsageTool(), FileCopyTool(),
                 ModelStatusTool(router), ModelUnloadTool(router)):
        if registry.get(tool.spec.name) is None:
            registry.register(tool)


def temp_dir() -> str:
    return tempfile.gettempdir()
