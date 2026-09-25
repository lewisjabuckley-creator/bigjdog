"""Project scan: measured facts about a codebase, gathered without running any of its code.

The scan is read-only and deterministic (file counts, languages, size, markers,
test and build commands, README, git branch). It is the evidence a language
model later summarises, so an analysis never rests on the model's guesses about
files it has not seen.
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from pathlib import Path
from typing import Any

from jarvis.core.types import Provenance, ProvenanceKind
from jarvis.permissions.model import PermissionLevel
from jarvis.tools.base import Tool, ToolContext, ToolResult, ToolSpec, Verification

_SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env", ".tox", ".mypy_cache",
              ".pytest_cache", ".ruff_cache", "dist", "build", "target", ".idea", ".vscode", ".next", ".cache",
              "site-packages", ".eggs"}
_LANGUAGES = {".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
              ".jsx": "JavaScript", ".rs": "Rust", ".go": "Go", ".java": "Java", ".kt": "Kotlin", ".c": "C",
              ".h": "C", ".cpp": "C++", ".cc": "C++", ".hpp": "C++", ".cs": "C#", ".rb": "Ruby", ".php": "PHP",
              ".swift": "Swift", ".m": "Objective-C", ".scala": "Scala", ".sh": "Shell", ".ps1": "PowerShell",
              ".lua": "Lua", ".dart": "Dart", ".sql": "SQL", ".html": "HTML", ".css": "CSS", ".scss": "CSS",
              ".vue": "Vue", ".svelte": "Svelte", ".md": "Markdown", ".toml": "TOML", ".yaml": "YAML",
              ".yml": "YAML", ".json": "JSON"}
_CODE = {"Python", "JavaScript", "TypeScript", "Rust", "Go", "Java", "Kotlin", "C", "C++", "C#", "Ruby", "PHP",
         "Swift", "Objective-C", "Scala", "Shell", "PowerShell", "Lua", "Dart", "Vue", "Svelte"}


def scan_project(root: Path, *, max_files: int = 20000, cancel: asyncio.Event | None = None) -> dict[str, Any]:
    from jarvis.planner.templates import probe_project
    from jarvis.projects.manager import git_branch

    files = 0
    by_lang: Counter[str] = Counter()
    lines_by_lang: Counter[str] = Counter()
    sizes: list[tuple[int, str]] = []
    tests = 0
    todo = 0
    top_dirs: Counter[str] = Counter()
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        if cancel is not None and cancel.is_set():
            break
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.endswith(".egg-info"))
        rel_dir = os.path.relpath(dirpath, root)
        for name in filenames:
            files += 1
            if files > max_files:
                truncated = True
                break
            path = os.path.join(dirpath, name)
            ext = os.path.splitext(name)[1].lower()
            lang = _LANGUAGES.get(ext)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            rel = os.path.relpath(path, root)
            sizes.append((size, rel))
            top = rel_dir.split(os.sep)[0] if rel_dir != "." else "(root)"
            top_dirs[top] += 1
            if "test" in name.lower() or f"{os.sep}tests{os.sep}" in f"{os.sep}{rel}":
                tests += 1
            if lang:
                by_lang[lang] += 1
                if lang in _CODE and size <= 2_000_000:
                    try:
                        with open(path, "rb") as fh:
                            data = fh.read()
                        lines_by_lang[lang] += data.count(b"\n")
                        todo += data.count(b"TODO") + data.count(b"FIXME")
                    except OSError:
                        pass
        if truncated:
            break
    probe = probe_project(root)
    readme = next((root / n for n in ("README.md", "README.rst", "README.txt", "README") if (root / n).is_file()),
                  None)
    excerpt = ""
    if readme is not None:
        try:
            excerpt = readme.read_text(errors="replace")[:1500]
        except OSError:
            pass
    sizes.sort(reverse=True)
    return {
        "root": str(root), "name": root.name, "files": min(files, max_files), "truncated": truncated,
        "languages": dict(by_lang.most_common(12)), "code_lines": dict(lines_by_lang.most_common(12)),
        "total_code_lines": sum(lines_by_lang.values()), "test_files": tests, "todo_fixme": todo,
        "top_directories": dict(top_dirs.most_common(12)),
        "largest_files": [{"path": p, "kb": round(s / 1024, 1)} for s, p in sizes[:8]],
        "markers": probe.markers, "test_command": probe.test_command, "build_command": probe.build_command,
        "git_branch": git_branch(str(root)), "readme": readme.name if readme else None, "readme_excerpt": excerpt,
    }


def describe_scan(scan: dict[str, Any]) -> str:
    """A factual one-paragraph summary of a scan (used when no language model is available)."""
    langs = ", ".join(f"{k} ({v} files, {scan['code_lines'].get(k, 0):,} lines)" if k in scan["code_lines"]
                      else f"{k} ({v} files)" for k, v in list(scan["languages"].items())[:5])
    parts = [f"{scan['name']}: {scan['files']:,} files{' (scan capped)' if scan['truncated'] else ''}, "
             f"{scan['total_code_lines']:,} lines of code."]
    if langs:
        parts.append(f"Languages: {langs}.")
    parts.append(f"{scan['test_files']} test file(s)" + (f"; tests run with `{scan['test_command']}`."
                                                        if scan.get("test_command") else "; no test command found."))
    if scan.get("git_branch"):
        parts.append(f"Git branch: {scan['git_branch']}.")
    if scan.get("todo_fixme"):
        parts.append(f"{scan['todo_fixme']} TODO/FIXME marker(s).")
    return " ".join(parts)


class ProjectScanTool(Tool):
    spec = ToolSpec(
        name="project_scan",
        description=("Measure a codebase without running it: file and line counts per language, tests, markers, "
                     "test/build commands, largest files, README excerpt, git branch."),
        parameters={"type": "object", "properties": {
            "path": {"type": "string"},
            "max_files": {"type": "integer", "default": 20000, "minimum": 1},
        }, "required": ["path"]},
        level=PermissionLevel.OBSERVE, idempotent=True, verification="read-only", path_params=("path",),
        timeout_s=300.0, category="filesystem",
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = Path(os.path.expanduser(args["path"]))
        if not root.is_absolute():
            root = Path(ctx.cwd or os.getcwd()) / root
        if not root.is_dir():
            return ToolResult(False, f"{root} is not a folder", error="not_found")
        scan = await asyncio.to_thread(scan_project, root, max_files=args["max_files"], cancel=ctx.cancel)
        return ToolResult(True, describe_scan(scan), scan, provenance=Provenance(ProvenanceKind.LOCAL_FILE, str(root)))

    async def verify(self, args: dict[str, Any], result: ToolResult, ctx: ToolContext) -> Verification:
        data = result.data if isinstance(result.data, dict) else {}
        root = Path(data.get("root", ""))
        passed = root.is_dir() and isinstance(data.get("files"), int)
        return Verification(True, passed, "folder re-checked", f"{data.get('files', 0)} file(s) measured in {root}")
