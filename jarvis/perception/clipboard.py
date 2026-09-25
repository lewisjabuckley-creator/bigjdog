"""An image on the clipboard, saved to a file for sending (``/paste``), run by the interface, not the runtime.

The clipboard belongs to the user's desktop session, so the interface reads it: Windows through PowerShell (built
in: take a snip with Win+Shift+S, then type /paste), macOS through ``pngpaste`` or AppleScript, Linux through
``wl-paste`` or ``xclip``. Nothing is read unless the user asks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

from jarvis.platforms import hidden_window_kwargs

_WIN = r"""
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$img = [System.Windows.Forms.Clipboard]::GetImage()
if ($img -eq $null) { exit 3 }
$img.Save('{path}', [System.Drawing.Imaging.ImageFormat]::Png)
"""


def grab_image() -> tuple[str | None, str]:
    """(path to a PNG of the clipboard image, "") or (None, why not)."""
    fd, path = tempfile.mkstemp(suffix=".png", prefix="clipboard-")
    os.close(fd)
    try:
        ok, why = _grab(path)
    except Exception as exc:
        ok, why = False, str(exc)
    if ok and os.path.getsize(path) > 0:
        return path, ""
    try:
        os.remove(path)
    except OSError:
        pass
    return None, why or "there's no image on the clipboard"


def _grab(path: str) -> tuple[bool, str]:
    if sys.platform == "win32":
        proc = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-STA", "-Command",
                               _WIN.replace("{path}", path.replace("'", "''"))], capture_output=True, timeout=20,
                              **hidden_window_kwargs())
        if proc.returncode == 3:
            return False, "there's no image on the clipboard (take one with Win+Shift+S first)"
        return proc.returncode == 0, (proc.stderr or b"").decode(errors="replace").strip()[:200]
    if sys.platform == "darwin":
        if shutil.which("pngpaste"):
            proc = subprocess.run(["pngpaste", path], capture_output=True, timeout=20)
            return proc.returncode == 0, "there's no image on the clipboard"
        script = (f'set f to (open for access POSIX file "{path}" with write permission)\n'
                  'try\nwrite (the clipboard as «class PNGf») to f\nend try\nclose access f')
        proc = subprocess.run(["osascript", "-e", script], capture_output=True, timeout=20)
        return proc.returncode == 0, "there's no image on the clipboard"
    for argv in (["wl-paste", "--type", "image/png"], ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]):
        if shutil.which(argv[0]):
            proc = subprocess.run(argv, capture_output=True, timeout=20)
            if proc.returncode == 0 and proc.stdout:
                with open(path, "wb") as fh:
                    fh.write(proc.stdout)
                return True, ""
            return False, "there's no image on the clipboard"
    return False, "reading the clipboard needs wl-paste or xclip here"
