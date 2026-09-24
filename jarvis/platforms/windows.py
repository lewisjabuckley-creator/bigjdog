"""Windows adapter: msvcrt byte-range lock, a background process with a hidden console, a Startup launcher.

Written against the documented Win32 behaviour but not yet run on Windows by the developers; see
docs/RUNTIME.md. Graceful stop goes through the local API; ``terminate`` is a hard stop
(TerminateProcess), after which restart recovery treats in-flight steps as having unknown outcomes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from jarvis.platforms import CREATE_NO_WINDOW
from jarvis.platforms.base import Platform, ServiceDefinition, popen_detached


def _sibling(name: str, beside: str | None = None) -> str | None:
    """python.exe / pythonw.exe next to the given interpreter (default: this one), if it exists."""
    candidate = Path(beside or sys.executable).with_name(name)
    return str(candidate) if candidate.exists() else None


class WindowsPlatform(Platform):
    name = "windows"
    tested = False

    def lock_file(self, fh: object) -> None:
        import msvcrt
        fh.seek(0)  # type: ignore[attr-defined]
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]

    def unlock_file(self, fh: object) -> None:
        import msvcrt
        fh.seek(0)  # type: ignore[attr-defined]
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]

    def python(self) -> str:
        # The runtime runs under python.exe with a hidden console (see spawn_detached), not pythonw.exe: a process
        # without any console makes every console program it starts (nvidia-smi, cmd, git) open a visible window.
        return _sibling("python.exe") or sys.executable

    def spawn_detached(self, argv: list[str], log_path: Path, env: dict[str, str] | None = None,
                       cwd: str | None = None) -> int:
        # CREATE_NO_WINDOW: its own console, never shown, which its children share (so nothing flashes); not the
        # terminal's console, so closing the terminal doesn't stop it. CREATE_NEW_PROCESS_GROUP: Ctrl+C in the
        # terminal doesn't reach it.
        flags = getattr(subprocess, "CREATE_NO_WINDOW", CREATE_NO_WINDOW) | \
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
        return popen_detached(argv, log_path, env=env, cwd=cwd, creationflags=flags)

    def service_definition(self, argv: list[str], data_dir: Path,
                           env: dict[str, str] | None = None) -> ServiceDefinition:
        startup = Path(os.environ.get("APPDATA", "~")).expanduser() / \
            "Microsoft/Windows/Start Menu/Programs/Startup/jarvis-runtime.cmd"
        # At sign-in, run the short-lived launcher (`runtime start`, under pythonw.exe so it needs no window);
        # it starts the runtime with a hidden console and exits.
        launcher = list(argv)
        if launcher[-2:] == ["runtime", "run"]:
            launcher[-1] = "start"
        launcher[0] = _sibling("pythonw.exe", launcher[0]) or launcher[0]
        command = subprocess.list2cmdline(launcher)
        variables = "".join(f'set "{k}={v}"\r\n' for k, v in (env or {}).items())
        content = f'@echo off\r\n{variables}start "" {command}\r\n'
        return ServiceDefinition(startup, content, "JARVIS will start the next time you sign in (or run the file "
                                                   "once now).", tested=False)
