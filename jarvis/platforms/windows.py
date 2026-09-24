"""Windows adapter: msvcrt byte-range lock, a detached process, a Startup-folder launcher.

Written against the documented Win32 behaviour but not yet run on Windows by the developers; see
docs/RUNTIME.md. Graceful stop goes through the local API; ``terminate`` is a hard stop
(TerminateProcess), after which restart recovery treats in-flight steps as having unknown outcomes.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from jarvis.platforms.base import Platform, ServiceDefinition, popen_detached


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
        # pythonw.exe has no console window; fall back to python.exe
        import sys
        candidate = Path(sys.executable).with_name("pythonw.exe")
        return str(candidate) if candidate.exists() else sys.executable

    def spawn_detached(self, argv: list[str], log_path: Path, env: dict[str, str] | None = None,
                       cwd: str | None = None) -> int:
        flags = getattr(subprocess, "DETACHED_PROCESS", 0x8) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
        return popen_detached(argv, log_path, env=env, cwd=cwd, creationflags=flags)

    def service_definition(self, argv: list[str], data_dir: Path,
                           env: dict[str, str] | None = None) -> ServiceDefinition:
        startup = Path(os.environ.get("APPDATA", "~")).expanduser() / \
            "Microsoft/Windows/Start Menu/Programs/Startup/jarvis-runtime.cmd"
        command = subprocess.list2cmdline(argv)
        variables = "".join(f'set "{k}={v}"\r\n' for k, v in (env or {}).items())
        content = f'@echo off\r\n{variables}start "" /B {command}\r\n'
        return ServiceDefinition(startup, content, "JARVIS will start the next time you sign in (or run the file "
                                                   "once now).", tested=False)
