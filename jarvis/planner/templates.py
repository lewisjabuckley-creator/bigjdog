"""Deterministic plan templates for common engineering intents.

Templates need no language model, so the most common requests ("run the tests")
work even when inference is unavailable, and their plans are predictable,
testable and auditable.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jarvis.tasks.models import Step


@dataclass
class ProjectProbe:
    root: str
    languages: list[str]
    test_command: str | None
    build_command: str | None
    markers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"root": self.root, "languages": self.languages, "test_command": self.test_command,
                "build_command": self.build_command, "markers": self.markers}


def probe_project(root: str | os.PathLike[str]) -> ProjectProbe:
    r = Path(root).expanduser()
    markers = [name for name in ("pyproject.toml", "setup.py", "setup.cfg", "pytest.ini", "tox.ini", "package.json",
                                 "Cargo.toml", "go.mod", "Makefile", "CMakeLists.txt", "pom.xml", "build.gradle",
                                 ".git") if (r / name).exists()]
    languages = []
    if any(m in markers for m in ("pyproject.toml", "setup.py", "setup.cfg", "pytest.ini", "tox.ini")) or \
            any(r.glob("*.py")):
        languages.append("python")
    if "package.json" in markers:
        languages.append("javascript")
    if "Cargo.toml" in markers:
        languages.append("rust")
    if "go.mod" in markers:
        languages.append("go")
    return ProjectProbe(str(r), languages, detect_test_command(r, markers), detect_build_command(r, markers), markers)


def _python(root: Path) -> str:
    for candidate in (".venv/bin/python", "venv/bin/python", ".venv/Scripts/python.exe"):
        if (root / candidate).exists():
            return candidate
    # Windows: the py launcher ships with python.org installs even when python isn't on PATH
    return "py" if sys.platform == "win32" else "python3"


def _make_targets(root: Path) -> set[str]:
    try:
        text = (root / "Makefile").read_text(errors="replace")
    except OSError:
        return set()
    return set(re.findall(r"^([A-Za-z0-9_.-]+):", text, re.MULTILINE))


def detect_test_command(root: Path, markers: list[str] | None = None) -> str | None:
    markers = markers if markers is not None else [p.name for p in root.iterdir()] if root.is_dir() else []
    if "package.json" in markers:
        try:
            scripts = json.loads((root / "package.json").read_text()).get("scripts", {})
        except (OSError, ValueError):
            scripts = {}
        if "test" in scripts and "no test specified" not in scripts["test"]:
            return "npm test --silent"
    if "Cargo.toml" in markers:
        return "cargo test"
    if "go.mod" in markers:
        return "go test ./..."
    python_markers = {"pyproject.toml", "setup.py", "setup.cfg", "pytest.ini", "tox.ini"}
    has_tests = (root / "tests").is_dir() or (root / "test").is_dir() or any(root.glob("test_*.py"))
    if python_markers & set(markers) or has_tests:
        return f"{_python(root)} -m pytest -q"
    if "Makefile" in markers and "test" in _make_targets(root):
        return "make test"
    return None


def detect_build_command(root: Path, markers: list[str] | None = None) -> str | None:
    markers = markers or []
    if "package.json" in markers:
        try:
            scripts = json.loads((root / "package.json").read_text()).get("scripts", {})
        except (OSError, ValueError):
            scripts = {}
        if "build" in scripts:
            return "npm run build --silent"
    if "Cargo.toml" in markers:
        return "cargo build"
    if "go.mod" in markers:
        return "go build ./..."
    if "Makefile" in markers:
        return "make"
    if "pyproject.toml" in markers:
        return f"{_python(root)} -m compileall -q ."
    return None


def run_tests_plan(root: str, command: str | None = None, timeout_s: float = 1800) -> tuple[list[Step], dict[str, Any]]:
    probe = probe_project(root)
    cmd = command or probe.test_command
    if not cmd:
        raise LookupError(f"I couldn't find a test suite in {root}")
    steps = [Step(f"run the test suite ({cmd})", "shell_execute",
                  {"command": cmd, "cwd": probe.root, "timeout_s": timeout_s}, allow_failure=True)]
    return steps, {"type": "tests"}


def build_plan(root: str, command: str | None = None, timeout_s: float = 1800) -> tuple[list[Step], dict[str, Any]]:
    probe = probe_project(root)
    cmd = command or probe.build_command
    if not cmd:
        raise LookupError(f"I couldn't find a build command for {root}")
    return [Step(f"build the project ({cmd})", "shell_execute",
                 {"command": cmd, "cwd": probe.root, "timeout_s": timeout_s})], {"type": "all_steps_ok"}
