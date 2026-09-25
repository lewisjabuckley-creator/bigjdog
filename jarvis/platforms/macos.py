"""macOS adapter: same process model as Linux, a launchd agent for start at login. Not tested on macOS."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from jarvis.platforms.base import ServiceDefinition
from jarvis.platforms.linux import PosixPlatform


class MacPlatform(PosixPlatform):
    name = "macos"
    tested = False

    def service_definition(self, argv: list[str], data_dir: Path,
                           env: dict[str, str] | None = None) -> ServiceDefinition:
        plist = Path("~/Library/LaunchAgents/com.jarvis.runtime.plist").expanduser()
        args = "\n".join(f"    <string>{escape(a)}</string>" for a in argv)
        variables = "".join(f"    <key>{escape(k)}</key><string>{escape(v)}</string>\n" for k, v in (env or {}).items())
        env_block = f"  <key>EnvironmentVariables</key>\n  <dict>\n{variables}  </dict>\n" if env else ""
        log = escape(str(data_dir / "logs" / "runtime-service.log"))
        content = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0">\n<dict>\n'
            "  <key>Label</key><string>com.jarvis.runtime</string>\n"
            f"  <key>ProgramArguments</key>\n  <array>\n{args}\n  </array>\n"
            f"{env_block}"
            "  <key>RunAtLoad</key><true/>\n"
            "  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>\n"
            f"  <key>StandardOutPath</key><string>{log}</string>\n"
            f"  <key>StandardErrorPath</key><string>{log}</string>\n"
            "</dict>\n</plist>\n")
        return ServiceDefinition(plist, content, f"launchctl load -w {plist}", tested=False)
