"""Linux adapter: flock, a new session for the background process, a systemd user unit."""

from __future__ import annotations

import fcntl
import shlex
from pathlib import Path

from jarvis.platforms.base import Platform, ServiceDefinition, popen_detached


class PosixPlatform(Platform):
    def lock_file(self, fh: object) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]

    def unlock_file(self, fh: object) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]

    def spawn_detached(self, argv: list[str], log_path: Path, env: dict[str, str] | None = None,
                       cwd: str | None = None) -> int:
        # a new session: closing the terminal (SIGHUP to its process group) does not reach the runtime
        return popen_detached(argv, log_path, env=env, cwd=cwd, start_new_session=True)


class LinuxPlatform(PosixPlatform):
    name = "linux"
    tested = True

    def service_definition(self, argv: list[str], data_dir: Path,
                           env: dict[str, str] | None = None) -> ServiceDefinition:
        unit = Path("~/.config/systemd/user/jarvis.service").expanduser()
        environment = "".join(f'Environment="{k}={v}"\n' for k, v in (env or {}).items())
        content = (
            "[Unit]\n"
            "Description=JARVIS runtime\n\n"
            "[Service]\n"
            "Type=simple\n"
            f"{environment}"
            f"ExecStart={' '.join(shlex.quote(a) for a in argv)}\n"
            "Restart=on-failure\n"
            "RestartSec=5\n"
            # SIGTERM lets the runtime checkpoint running tasks before it exits
            "KillSignal=SIGTERM\n"
            "TimeoutStopSec=30\n\n"
            "[Install]\n"
            "WantedBy=default.target\n")
        return ServiceDefinition(unit, content, "systemctl --user daemon-reload && systemctl --user enable --now "
                                               "jarvis.service", tested=False)
