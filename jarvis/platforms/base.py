"""Operating-system adapter contract for the persistent runtime.

The runtime itself is plain asyncio and portable; what differs per OS is how a
background process is started and stopped, how a single-instance lock is
held, and how JARVIS is registered to start at login. Each adapter implements
only those pieces.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil


class LockHeld(RuntimeError):
    """Another JARVIS runtime already owns this data directory."""

    def __init__(self, path: Path, pid: int | None) -> None:
        self.path = path
        self.pid = pid
        who = f" (process {pid})" if pid else ""
        super().__init__(f"another JARVIS runtime is already running for this data directory{who}")


@dataclass
class ServiceDefinition:
    path: Path                  # where the definition file goes
    content: str
    enable_hint: str            # the command that activates it, shown to the user
    tested: bool                # whether this adapter has been exercised on its OS


class InstanceLock:
    """An OS-level exclusive lock on a file. The OS releases it when the process dies, so a crashed runtime
    never leaves a stale lock behind (unlike a bare pid file)."""

    def __init__(self, path: Path, platform: "Platform") -> None:
        self.path = Path(path)
        self.platform = platform
        self._fh: object | None = None

    @property
    def held(self) -> bool:
        return self._fh is not None

    def acquire(self) -> "InstanceLock":
        if self._fh is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            self.platform.lock_file(fh)
        except OSError:
            fh.close()
            raise LockHeld(self.path, read_pid(self.path.with_suffix(".pid"))) from None
        self._fh = fh
        self.path.with_suffix(".pid").write_text(str(os.getpid()))
        return self

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            pid_file = self.path.with_suffix(".pid")
            if read_pid(pid_file) == os.getpid():
                pid_file.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            self.platform.unlock_file(fh)
        except OSError:
            pass
        fh.close()  # type: ignore[attr-defined]


def read_pid(path: Path) -> int | None:
    try:
        return int(Path(path).read_text().strip() or 0) or None
    except (OSError, ValueError):
        return None


class Platform:
    name = "generic"
    tested = False

    # -- locks ----------------------------------------------------------------------------------
    def lock_file(self, fh: object) -> None:
        raise NotImplementedError

    def unlock_file(self, fh: object) -> None:
        raise NotImplementedError

    def instance_lock(self, path: Path) -> InstanceLock:
        return InstanceLock(path, self)

    # -- processes ----------------------------------------------------------------------------------
    def python(self) -> str:
        return sys.executable

    def spawn_detached(self, argv: list[str], log_path: Path, env: dict[str, str] | None = None,
                       cwd: str | None = None) -> int:
        """Start ``argv`` in the background, detached from this terminal. Returns its pid."""
        raise NotImplementedError

    def is_alive(self, pid: int | None) -> bool:
        if not pid:
            return False
        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def terminate(self, pid: int, timeout: float = 15.0) -> bool:
        """Ask a process to stop, escalating to a hard kill after ``timeout``. True once it is gone."""
        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return True
        try:
            proc.terminate()
            proc.wait(timeout)
            return True
        except psutil.NoSuchProcess:
            return True
        except psutil.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(5)
            except psutil.TimeoutExpired:
                return False
            return True

    def wait_gone(self, pid: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_alive(pid):
                return True
            time.sleep(0.1)
        return not self.is_alive(pid)

    # -- services -------------------------------------------------------------------------------------
    def service_definition(self, argv: list[str], data_dir: Path,
                           env: dict[str, str] | None = None) -> ServiceDefinition:
        """A start-at-login definition that runs ``argv`` (the foreground runtime) with ``env`` set."""
        raise NotImplementedError


def popen_detached(argv: list[str], log_path: Path, *, env: dict[str, str] | None, cwd: str | None,
                   **kwargs: object) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                env=env, cwd=cwd, close_fds=True, **kwargs)  # type: ignore[call-overload]
    return proc.pid
