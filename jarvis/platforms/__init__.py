"""Operating-system adapters for the persistent runtime (background start/stop, locks, start at login)."""

from __future__ import annotations

import sys

from jarvis.platforms.base import InstanceLock, LockHeld, Platform, ServiceDefinition, read_pid

__all__ = ["InstanceLock", "LockHeld", "Platform", "ServiceDefinition", "current", "hidden_window_kwargs", "read_pid"]

CREATE_NO_WINDOW = 0x08000000


def hidden_window_kwargs() -> dict[str, int]:
    """Keyword arguments for starting a child process without a console window.

    On Windows, a process with no console of its own (the background runtime) that starts a console program
    (nvidia-smi, cmd, git...) makes Windows open a new console window for it, which flashes on screen. Every
    child JARVIS starts passes these. Elsewhere they are empty.
    """
    if sys.platform == "win32":
        return {"creationflags": CREATE_NO_WINDOW}
    return {}


def current() -> Platform:
    if sys.platform == "win32":
        from jarvis.platforms.windows import WindowsPlatform
        return WindowsPlatform()
    if sys.platform == "darwin":
        from jarvis.platforms.macos import MacPlatform
        return MacPlatform()
    from jarvis.platforms.linux import LinuxPlatform
    return LinuxPlatform()
