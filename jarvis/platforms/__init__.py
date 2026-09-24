"""Operating-system adapters for the persistent runtime (background start/stop, locks, start at login)."""

from __future__ import annotations

import sys

from jarvis.platforms.base import InstanceLock, LockHeld, Platform, ServiceDefinition, read_pid

__all__ = ["InstanceLock", "LockHeld", "Platform", "ServiceDefinition", "current", "read_pid"]


def current() -> Platform:
    if sys.platform == "win32":
        from jarvis.platforms.windows import WindowsPlatform
        return WindowsPlatform()
    if sys.platform == "darwin":
        from jarvis.platforms.macos import MacPlatform
        return MacPlatform()
    from jarvis.platforms.linux import LinuxPlatform
    return LinuxPlatform()
