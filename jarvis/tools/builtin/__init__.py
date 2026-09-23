"""Built-in tools available on every installation."""

from __future__ import annotations

from typing import Any

from jarvis.tools.builtin.filesystem import (FileDeleteTool, FileListTool, FileReadTool, FileSearchTool,
                                             FileWriteTool)
from jarvis.tools.builtin.shell import ShellTool
from jarvis.tools.builtin.system import (ProcessInspectTool, ProcessListTool, ProcessStopTool, SystemInfoTool,
                                         TimeTool)
from jarvis.tools.registry import ToolRegistry


def register_builtin_tools(registry: ToolRegistry, *, metrics_source: Any | None = None) -> None:
    for tool in (FileReadTool(), FileListTool(), FileSearchTool(), FileWriteTool(), FileDeleteTool(), ShellTool(),
                 SystemInfoTool(metrics_source), ProcessListTool(), ProcessInspectTool(), ProcessStopTool(),
                 TimeTool()):
        registry.register(tool)
