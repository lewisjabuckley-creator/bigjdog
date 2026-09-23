"""Shell command execution with risk classification (spec §55).

Commands are logged, run in an explicit working directory with a timeout, and
their exit status and output are captured. Exit status 0 means "the command
completed" — not that the user's goal was achieved; goal verification belongs to
the task's success condition.

Risk is assessed per command: known read-only inspection is observation, known
build/test runners are reversible local execution, anything unrecognised or
destructive is consequential, and a short list of catastrophic patterns is
blocked outright.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import signal
import sys
import time
from typing import Any

from jarvis.core.types import Provenance, ProvenanceKind, RiskLevel
from jarvis.permissions.model import PermissionLevel
from jarvis.security.redaction import scrubbed_environment
from jarvis.tools.base import Assessment, Tool, ToolContext, ToolResult, ToolSpec, Verification

MAX_OUTPUT = 64_000

_CATASTROPHIC = [
    re.compile(r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+(--no-preserve-root\s+)?(/|/\*|~|~/|\$HOME)(\s|$)"),
    re.compile(r"\bmkfs(\.\w+)?\b"),
    re.compile(r"\bdd\b[^|;&]*\bof=/dev/(sd|nvme|hd|disk|mmcblk)"),
    re.compile(r">\s*/dev/(sd|nvme|hd|disk|mmcblk)"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:"),                       # fork bomb
    re.compile(r"\b(chmod|chown)\s+-R\s+\S+\s+/(\s|$)"),
    re.compile(r"\bformat\s+[a-zA-Z]:"),
]

_READ_ONLY = {
    "ls", "cat", "head", "tail", "less", "wc", "grep", "rg", "ag", "pwd", "echo", "which", "whereis", "type",
    "file", "stat", "du", "df", "free", "uptime", "uname", "whoami", "id", "date", "hostname", "ps", "top",
    "pgrep", "lsof", "tree", "diff", "cmp", "md5sum", "sha256sum", "sha1sum", "sort", "uniq", "cut", "tr",
    "basename", "dirname", "realpath", "readlink", "nproc", "lscpu", "lsblk", "lsusb", "lspci", "nvidia-smi",
    "sensors", "true", "false", "test", "jq", "awk", "column", "printf",
}
_GIT_READ_ONLY = {"status", "log", "diff", "show", "branch", "remote", "rev-parse", "describe", "blame",
                  "ls-files", "shortlog", "tag", "config", "stash"}
_RUNNERS = {  # local build/test/lint tools: reversible execution with local side effects
    "pytest", "python", "python3", "node", "npm", "pnpm", "yarn", "cargo", "go", "make", "cmake", "ruff",
    "mypy", "black", "flake8", "eslint", "tsc", "jest", "vitest", "gradle", "mvn", "dotnet", "swift", "rustc",
    "gcc", "g++", "clang", "javac", "java", "tox", "nox", "uv", "poetry", "bundle", "rake", "mix", "deno", "bun",
    "mkdir", "touch", "cp",
}
_CONSEQUENTIAL = {
    "rm", "rmdir", "mv", "dd", "shred", "kill", "killall", "pkill", "sudo", "su", "doas", "systemctl", "service",
    "launchctl", "reboot", "shutdown", "halt", "poweroff", "chmod", "chown", "chgrp", "mount", "umount",
    "iptables", "ufw", "firewall-cmd", "crontab", "useradd", "userdel", "passwd", "docker", "podman",
    "kubectl", "helm", "terraform", "ansible", "psql", "mysql", "mongo", "redis-cli", "sqlite3", "apt",
    "apt-get", "yum", "dnf", "brew", "pacman", "snap", "ssh", "scp", "rsync", "truncate", "ln",
}
_NETWORK = {"curl", "wget", "ssh", "scp", "rsync", "ping", "nc", "telnet", "ftp", "sftp", "dig", "nslookup",
            "traceroute", "http", "aria2c"}
_NETWORK_SUBCOMMANDS = {
    "git": {"clone", "fetch", "pull", "push", "ls-remote", "submodule"},
    "pip": {"install", "download"}, "pip3": {"install", "download"},
    "npm": {"install", "i", "ci", "publish", "update"}, "pnpm": {"install", "add", "publish"},
    "yarn": {"install", "add", "publish"}, "cargo": {"install", "publish", "update", "fetch"},
    "go": {"get", "install", "mod"}, "uv": {"pip", "sync", "add"}, "poetry": {"install", "add", "publish"},
    "docker": {"pull", "push", "login"},
}
_CONSEQUENTIAL_SUBCOMMANDS = {
    "git": {"push", "reset", "clean", "checkout", "rebase", "merge", "commit", "restore", "rm", "gc", "prune",
            "switch", "cherry-pick", "revert", "filter-branch", "am", "apply"},
    "npm": {"publish", "unpublish", "install", "i", "ci", "uninstall", "update"},
    "pip": {"install", "uninstall"}, "pip3": {"install", "uninstall"},
    "cargo": {"publish", "install"}, "yarn": {"publish", "add", "remove"}, "pnpm": {"publish", "add", "remove"},
    "poetry": {"publish", "add", "remove"}, "uv": {"pip", "add", "remove"},
}
_SEGMENT_SPLIT = re.compile(r"\|\||&&|;|\||\n")
_REDIRECT = re.compile(r"(?<![0-9&>])>{1,2}\s*([^\s&|;]+)")


def _segments(command: str) -> list[str]:
    return [s.strip() for s in _SEGMENT_SPLIT.split(command) if s.strip()]


def _words(segment: str) -> list[str]:
    try:
        words = shlex.split(segment)
    except ValueError:
        words = segment.split()
    while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):  # leading VAR=value assignments
        words = words[1:]
    return words


def classify_command(command: str) -> Assessment:
    text = command.strip()
    for pattern in _CATASTROPHIC:
        if pattern.search(text):
            return Assessment(PermissionLevel.AUTONOMOUS, RiskLevel.CRITICAL, "catastrophic command pattern",
                              blocked=True, reversible=False)
    level, risk, reasons, network, reversible = PermissionLevel.OBSERVE, RiskLevel.NONE, [], False, True
    if "$(" in text or "`" in text:
        level, risk, reversible = PermissionLevel.EXECUTE_CONSEQUENTIAL, RiskLevel.HIGH, False
        reasons.append("command substitution cannot be analysed")
    targets = [t for t in _REDIRECT.findall(text) if t != "/dev/null"]
    if targets:
        if level < PermissionLevel.EXECUTE_REVERSIBLE:
            level, risk = PermissionLevel.EXECUTE_REVERSIBLE, RiskLevel.LOW
        reasons.append("writes output to a file")
    for segment in _segments(text):
        words = _words(segment)
        if not words:
            continue
        prog = os.path.basename(words[0])
        sub = words[1] if len(words) > 1 else ""
        seg_level, seg_risk = PermissionLevel.EXECUTE_CONSEQUENTIAL, RiskLevel.MEDIUM
        why = f"unrecognised command {prog!r}"
        if prog in _NETWORK or sub in _NETWORK_SUBCOMMANDS.get(prog, set()):
            network = True
        if prog in _CONSEQUENTIAL or sub in _CONSEQUENTIAL_SUBCOMMANDS.get(prog, set()):
            seg_level, seg_risk, why = PermissionLevel.EXECUTE_CONSEQUENTIAL, RiskLevel.HIGH, f"{prog} {sub}".strip()
            reversible = False
        elif prog == "git" and sub in _GIT_READ_ONLY:
            seg_level, seg_risk, why = PermissionLevel.OBSERVE, RiskLevel.NONE, "read-only git"
        elif prog == "find":
            if any(w in ("-delete", "-exec", "-execdir", "-ok") for w in words):
                seg_level, seg_risk, why = PermissionLevel.EXECUTE_CONSEQUENTIAL, RiskLevel.HIGH, "find with actions"
                reversible = False
            else:
                seg_level, seg_risk, why = PermissionLevel.OBSERVE, RiskLevel.NONE, "read-only find"
        elif prog in ("sed", "perl") and any(w.startswith("-i") for w in words):
            seg_level, seg_risk, why = PermissionLevel.EXECUTE_REVERSIBLE, RiskLevel.MEDIUM, "in-place edit"
        elif prog in ("sed", "perl") or prog in _READ_ONLY:
            seg_level, seg_risk, why = PermissionLevel.OBSERVE, RiskLevel.NONE, "read-only inspection"
        elif prog in _RUNNERS:
            seg_level, seg_risk, why = PermissionLevel.EXECUTE_REVERSIBLE, RiskLevel.LOW, "local build/test tool"
        elif prog in _NETWORK:
            seg_level, seg_risk, why = PermissionLevel.EXECUTE_CONSEQUENTIAL, RiskLevel.MEDIUM, "network access"
        if seg_level > level or (seg_level == level and seg_risk > risk):
            level, risk = seg_level, seg_risk
        reasons.append(why)
    return Assessment(level, risk, "; ".join(dict.fromkeys(reasons)), requires_network=network,
                      reversible=reversible)


class ShellTool(Tool):
    spec = ToolSpec(
        name="shell_execute",
        description=("Run a shell command in a working directory and capture exit status, stdout and stderr. "
                     "Read-only inspection runs freely; destructive or unrecognised commands need authorization."),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command line to run."},
                "cwd": {"type": "string", "description": "Working directory (defaults to the active project)."},
                "timeout_s": {"type": "number", "description": "Timeout in seconds.", "default": 120, "minimum": 0.1},
            },
            "required": ["command"],
        },
        level=PermissionLevel.EXECUTE_CONSEQUENTIAL,
        risk=RiskLevel.MEDIUM,
        side_effects=("process.spawn", "filesystem", "network"),
        reversible=False,
        timeout_s=120,
        verification="exit status inspected; output captured",
        path_params=("cwd",),
        long_running=True,
        category="system",
    )

    def assess(self, args: dict[str, Any]) -> Assessment:
        return classify_command(args["command"])

    def preview(self, args: dict[str, Any]) -> str:
        where = f" (in {args['cwd']})" if args.get("cwd") else ""
        return f"$ {args['command']}{where}"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        cwd = os.path.expanduser(args.get("cwd") or ctx.cwd or os.getcwd())
        if not os.path.isdir(cwd):
            return ToolResult(False, f"working directory {cwd} does not exist", error="bad_cwd")
        started = time.monotonic()
        kwargs: dict[str, Any] = {}
        if sys.platform != "win32":
            kwargs["start_new_session"] = True  # own process group so timeouts kill children too
        proc = await asyncio.create_subprocess_shell(
            args["command"], cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL, env=scrubbed_environment(dict(os.environ)), **kwargs)
        try:
            stdout, stderr = await proc.communicate()
        except asyncio.CancelledError:
            _kill(proc)
            await proc.wait()
            raise
        duration = time.monotonic() - started
        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")
        code = proc.returncode
        data = {"command": args["command"], "cwd": cwd, "exit_code": code, "stdout": _clip(out),
                "stderr": _clip(err), "duration_s": round(duration, 3)}
        summary = f"`{_short(args['command'])}` exited with {code}"
        return ToolResult(code == 0, summary, data, error=None if code == 0 else (err.strip()[-500:] or f"exit {code}"),
                          exit_code=code, provenance=Provenance(ProvenanceKind.TOOL_OUTPUT, "shell_execute"))

    async def verify(self, args: dict[str, Any], result: ToolResult, ctx: ToolContext) -> Verification:
        # At this layer we can only confirm that the reported outcome matches the captured exit status.
        # Whether the user's goal was achieved is judged by the task's success condition.
        if result.exit_code is None:
            return Verification(True, False, "exit_code", "no exit status captured")
        consistent = result.ok == (result.exit_code == 0)
        return Verification(True, consistent, "exit_code", f"exit status {result.exit_code}")


def _kill(proc: asyncio.subprocess.Process) -> None:
    try:
        if sys.platform != "win32":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    half = MAX_OUTPUT // 2
    return text[:half] + f"\n…[{len(text) - MAX_OUTPUT} characters omitted]…\n" + text[-half:]


def _short(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
