"""Platform adapters. The Windows paths are exercised by simulating ``sys.platform`` — they are not run on Windows
here — to pin down the flags that keep console windows from flashing up."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from jarvis import platforms
from jarvis.monitoring import metrics as metrics_module
from jarvis.permissions.model import Actor
from jarvis.platforms import CREATE_NO_WINDOW, hidden_window_kwargs
from jarvis.platforms import windows as windows_module
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin import shell as shell_module


def test_child_processes_get_no_console_window_on_windows(monkeypatch):
    assert hidden_window_kwargs() == ({"creationflags": CREATE_NO_WINDOW} if sys.platform == "win32" else {})
    monkeypatch.setattr(sys, "platform", "win32")
    assert hidden_window_kwargs() == {"creationflags": 0x08000000}


def test_the_gpu_probe_runs_without_a_window(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(kwargs)

        class Done:
            stdout = "RTX 4070, 12, 3000, 12282, 45\n"
        return Done()

    source = metrics_module.PsutilMetrics()
    source._nvidia_smi = "nvidia-smi"
    monkeypatch.setattr(metrics_module.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "platform", "win32")
    gpus = source._gpus()
    monkeypatch.undo()
    assert gpus[0]["name"] == "RTX 4070" and calls[0]["creationflags"] == CREATE_NO_WINDOW


async def test_shell_commands_run_without_a_window(monkeypatch, tmp_path):
    seen = {}

    class FakeProc:
        pid = 1234
        returncode = 0

        async def communicate(self):
            return b"hi\n", b""

    async def fake_shell(command, **kwargs):
        seen.update(kwargs)
        return FakeProc()

    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_shell", fake_shell)
    monkeypatch.setattr(sys, "platform", "win32")
    result = await shell_module.ShellTool().run({"command": "echo hi"}, ToolContext(actor=Actor.user(),
                                                                                     cwd=str(tmp_path)))
    monkeypatch.undo()
    assert result.ok and seen["creationflags"] == CREATE_NO_WINDOW and "start_new_session" not in seen


def test_windows_runtime_gets_a_hidden_console_not_none(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(windows_module, "popen_detached",
                        lambda argv, log, **kw: captured.update(kw) or 4242)
    pid = windows_module.WindowsPlatform().spawn_detached(["python.exe", "-m", "jarvis", "runtime", "run"],
                                                          tmp_path / "out.log")
    flags = captured["creationflags"]
    assert pid == 4242
    assert flags & CREATE_NO_WINDOW and flags & 0x200          # hidden console, own process group
    assert not flags & 0x8                                      # not DETACHED_PROCESS (no console at all)


def test_windows_start_at_login_uses_the_short_launcher(tmp_path):
    definition = windows_module.WindowsPlatform().service_definition(
        ["C:/Python313/python.exe", "-m", "jarvis", "--data-dir", "C:/Users/me/.jarvis", "runtime", "run"],
        Path("C:/Users/me/.jarvis"), env={"PYTHONPATH": "D:/JARVIS/bigjdog"})
    assert definition.content.startswith("@echo off\r\n")
    assert 'set "PYTHONPATH=D:/JARVIS/bigjdog"' in definition.content
    assert definition.content.rstrip().endswith("runtime start")
    assert definition.path.name == "jarvis-runtime.cmd"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX adapter")
def test_posix_adapter_is_selected_here():
    assert platforms.current().name in ("linux", "macos")
