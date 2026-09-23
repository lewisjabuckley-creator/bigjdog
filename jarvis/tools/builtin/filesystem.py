"""Filesystem tools.

Writes are reversible: an existing file is backed up before being overwritten
and the backup location is recorded as rollback information. Deletion moves the
file to JARVIS's trash rather than unlinking it, and still requires
consequential authority. Every write is verified by reading the result back.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Any

from jarvis.core.types import Provenance, ProvenanceKind, RiskLevel
from jarvis.permissions.model import PermissionLevel
from jarvis.tools.base import Tool, ToolContext, ToolResult, ToolSpec, Verification


def _resolve(path: str, ctx: ToolContext) -> Path:
    p = Path(os.path.expanduser(path))
    if not p.is_absolute():
        p = Path(ctx.cwd or os.getcwd()) / p
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _data_dir(ctx: ToolContext) -> Path:
    return Path(os.path.expanduser(ctx.data_dir or "~/.jarvis"))


def _prov(path: Path) -> Provenance:
    return Provenance(ProvenanceKind.LOCAL_FILE, str(path))


class FileReadTool(Tool):
    spec = ToolSpec(
        name="file_read",
        description="Read a text file and return its contents (truncated to max_bytes).",
        parameters={"type": "object", "properties": {
            "path": {"type": "string"},
            "max_bytes": {"type": "integer", "default": 200_000, "minimum": 1},
        }, "required": ["path"]},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="read-only", path_params=("path",),
        category="filesystem",
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = _resolve(args["path"], ctx)
        if not path.is_file():
            return ToolResult(False, f"{path} does not exist or is not a file", error="not_found",
                              provenance=_prov(path))
        size = path.stat().st_size
        with open(path, "rb") as fh:
            raw = fh.read(args["max_bytes"])
        text = raw.decode(errors="replace")
        truncated = size > len(raw)
        return ToolResult(True, f"read {path.name} ({size} bytes{', truncated' if truncated else ''})",
                          {"path": str(path), "content": text, "size": size, "truncated": truncated},
                          provenance=_prov(path))


class FileListTool(Tool):
    spec = ToolSpec(
        name="file_list",
        description="List directory entries, optionally recursively and filtered by a glob pattern.",
        parameters={"type": "object", "properties": {
            "path": {"type": "string"},
            "pattern": {"type": "string", "default": "*"},
            "recursive": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "default": 500, "minimum": 1},
        }, "required": ["path"]},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="read-only", path_params=("path",),
        category="filesystem",
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = _resolve(args["path"], ctx)
        if not root.is_dir():
            return ToolResult(False, f"{root} is not a directory", error="not_found", provenance=_prov(root))
        entries = []
        iterator = root.rglob(args["pattern"]) if args["recursive"] else root.glob(args["pattern"])
        for p in iterator:
            if any(part.startswith(".") and part not in (".", "..") for part in p.relative_to(root).parts[:-1]):
                continue  # skip contents of hidden directories (.git, .venv...)
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append({"path": str(p.relative_to(root)), "type": "dir" if p.is_dir() else "file",
                            "size": st.st_size, "modified": st.st_mtime})
            if len(entries) >= args["limit"]:
                break
        entries.sort(key=lambda e: e["path"])
        return ToolResult(True, f"{len(entries)} entries in {root}", {"root": str(root), "entries": entries},
                          provenance=_prov(root))


class FileSearchTool(Tool):
    spec = ToolSpec(
        name="file_search",
        description="Search text files under a directory for a substring; returns matching lines.",
        parameters={"type": "object", "properties": {
            "path": {"type": "string"},
            "query": {"type": "string"},
            "glob": {"type": "string", "default": "*"},
            "max_results": {"type": "integer", "default": 100, "minimum": 1},
        }, "required": ["path", "query"]},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="read-only", path_params=("path",),
        category="filesystem",
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = _resolve(args["path"], ctx)
        needle = args["query"].lower()
        matches = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("node_modules", "__pycache__")]
            for name in filenames:
                if not fnmatch.fnmatch(name, args["glob"]):
                    continue
                path = Path(dirpath) / name
                try:
                    if path.stat().st_size > 2_000_000:
                        continue
                    with open(path, encoding="utf-8", errors="strict") as fh:
                        for lineno, line in enumerate(fh, 1):
                            if needle in line.lower():
                                matches.append({"path": str(path.relative_to(root)), "line": lineno,
                                                "text": line.rstrip()[:300]})
                                if len(matches) >= args["max_results"]:
                                    break
                except (UnicodeDecodeError, OSError):
                    continue
                if len(matches) >= args["max_results"]:
                    break
            if len(matches) >= args["max_results"]:
                break
        return ToolResult(True, f"{len(matches)} match(es) for {args['query']!r}", {"matches": matches},
                          provenance=_prov(root))


class FileWriteTool(Tool):
    spec = ToolSpec(
        name="file_write",
        description="Write text to a file (overwrite, append or create-only). Existing files are backed up first.",
        parameters={"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "mode": {"type": "string", "enum": ["overwrite", "append", "create"], "default": "overwrite"},
        }, "required": ["path", "content"]},
        level=PermissionLevel.EXECUTE_REVERSIBLE, risk=RiskLevel.LOW, side_effects=("filesystem.write",),
        reversible=True, verification="file read back and content hash compared", path_params=("path",),
        category="filesystem",
    )

    def preview(self, args: dict[str, Any]) -> str:
        return f"{args.get('mode', 'overwrite')} {args['path']} ({len(args['content'])} characters)"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = _resolve(args["path"], ctx)
        mode = args["mode"]
        if mode == "create" and path.exists():
            return ToolResult(False, f"{path} already exists", error="exists", provenance=_prov(path))
        rollback: dict[str, Any] = {"path": str(path), "existed": path.exists()}
        if path.exists():
            backup_dir = _data_dir(ctx) / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / f"{int(time.time() * 1000)}_{path.name}"
            shutil.copy2(path, backup)
            rollback["backup"] = str(backup)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a" if mode == "append" else "w", encoding="utf-8") as fh:
            fh.write(args["content"])
        return ToolResult(True, f"wrote {path}", {"path": str(path), "sha256": _sha256(path),
                                                   "bytes": path.stat().st_size},
                          rollback=rollback, provenance=_prov(path))

    async def verify(self, args: dict[str, Any], result: ToolResult, ctx: ToolContext) -> Verification:
        path = _resolve(args["path"], ctx)
        if not path.is_file():
            return Verification(True, False, "read_back", f"{path} does not exist after write")
        text = path.read_text(encoding="utf-8", errors="replace")
        if args["mode"] == "append":
            passed = text.endswith(args["content"])
        else:
            passed = text == args["content"]
        return Verification(True, passed, "read_back",
                            "content on disk matches" if passed else "content on disk differs from what was written")

    async def rollback(self, info: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = Path(info["path"])
        if info.get("backup"):
            shutil.copy2(info["backup"], path)
            return ToolResult(True, f"restored {path} from backup")
        if not info.get("existed") and path.exists():
            path.unlink()
            return ToolResult(True, f"removed {path} (it did not exist before)")
        return ToolResult(False, "nothing to roll back", error="noop")


class FileDeleteTool(Tool):
    spec = ToolSpec(
        name="file_delete",
        description="Delete a file by moving it to JARVIS's trash (recoverable). Requires authorization.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        level=PermissionLevel.EXECUTE_CONSEQUENTIAL, risk=RiskLevel.MEDIUM, side_effects=("filesystem.delete",),
        reversible=True, verification="path absent and trash copy present", path_params=("path",),
        category="filesystem",
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = _resolve(args["path"], ctx)
        if not path.exists():
            return ToolResult(False, f"{path} does not exist", error="not_found", provenance=_prov(path))
        if path.is_dir():
            return ToolResult(False, f"{path} is a directory; refusing to delete directories", error="is_dir")
        trash = _data_dir(ctx) / "trash"
        trash.mkdir(parents=True, exist_ok=True)
        target = trash / f"{int(time.time() * 1000)}_{path.name}"
        shutil.move(str(path), target)
        return ToolResult(True, f"moved {path} to trash", {"path": str(path), "trash": str(target)},
                          rollback={"path": str(path), "trash": str(target)}, provenance=_prov(path))

    async def verify(self, args: dict[str, Any], result: ToolResult, ctx: ToolContext) -> Verification:
        path = _resolve(args["path"], ctx)
        trash_ok = bool(result.data) and Path(result.data["trash"]).exists()
        passed = not path.exists() and trash_ok
        return Verification(True, passed, "exists_check", "removed; recoverable from trash" if passed
                            else "file still present or trash copy missing")

    async def rollback(self, info: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path, trash = Path(info["path"]), Path(info["trash"])
        if path.exists():
            return ToolResult(False, f"{path} exists again; not overwriting", error="exists")
        shutil.move(str(trash), path)
        return ToolResult(True, f"restored {path} from trash")
