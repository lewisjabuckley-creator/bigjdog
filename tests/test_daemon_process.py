"""Phase 2, process level: the real background runtime started and stopped through the CLI, work that
continues after the interface closes, and recovery after the runtime is killed.

These start real processes (simulated model and hardware, so no Ollama is needed) and take a few seconds each.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from jarvis.service.client import Client, read_info

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals; Windows is covered by manual checks")


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    config = tmp_path / "jarvis.toml"
    config.write_text(
        f'[general]\ndata_dir = "{data}"\n\n'
        f'[permissions]\nallowed_roots = ["{tmp_path}"]\n\n'
        '[monitoring]\nenabled = false\n\n'
        '[tasks]\nscheduler_interval_s = 0.05\n\n'
        '[runtime]\nheartbeat_s = 0.5\n')
    return data, config


def jarvis(config: Path, *args: str, input: str | None = None, timeout: float = 60,
           cwd: Path | None = None) -> subprocess.CompletedProcess:
    # --simulate alone would use a temporary data directory; this test's directory is given explicitly
    data = config.parent / "data"
    return subprocess.run([sys.executable, "-m", "jarvis", "--config", str(config), "--data-dir", str(data),
                           "--simulate", *args],
                          input=input, capture_output=True, text=True, timeout=timeout, env=_env(),
                          cwd=str(cwd or ROOT))


def _client(data: Path) -> Client:
    info = read_info(data)
    assert info is not None, "runtime not running"
    return Client(info)


def _wait(predicate, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met in time")
        time.sleep(0.1)


def _slow_project(tmp_path: Path, seconds: int = 8) -> Path:
    project = tmp_path / "slowproj"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "test_slow.py").write_text(f"import time\n\ndef test_slow():\n    time.sleep({seconds})\n")
    return project


@pytest.fixture
def runtime_dirs(tmp_path):
    data, config = _setup(tmp_path)
    yield data, config
    info = read_info(data)
    if info is not None:           # never leave a runtime behind
        try:
            os.kill(info.pid, signal.SIGKILL)
        except OSError:
            pass


def test_start_status_single_instance_and_clean_stop(runtime_dirs):
    data, config = runtime_dirs
    started = jarvis(config, "runtime", "start")
    assert started.returncode == 0, started.stdout + started.stderr
    assert "Started the JARVIS runtime" in started.stdout
    info = read_info(data)
    assert info is not None and (data / "api.token").stat().st_mode & 0o777 == 0o600
    status = jarvis(config, "runtime", "status")
    assert status.returncode == 0 and status.stdout.startswith("running — pid")
    again = jarvis(config, "runtime", "start")
    assert "already running" in again.stdout
    second = jarvis(config, "runtime", "run", timeout=30)              # a second runtime refuses to start
    assert second.returncode == 3 and "already running" in second.stderr
    health = jarvis(config, "runtime", "health")
    assert health.returncode == 0 and "overall: healthy" in health.stdout
    stopped = jarvis(config, "runtime", "stop")
    assert stopped.returncode == 0 and "Stopped the JARVIS runtime" in stopped.stdout
    assert not (data / "runtime.json").exists() and not (data / "api.token").exists()
    after = jarvis(config, "runtime", "status")
    assert after.returncode == 3 and "stopped cleanly" in after.stdout
    logs = jarvis(config, "runtime", "logs", "-n", "5")
    assert "JARVIS runtime stopped." in logs.stdout


def test_closing_the_interface_does_not_stop_the_work(runtime_dirs, tmp_path):
    data, config = runtime_dirs
    project = tmp_path / "proj"
    project.mkdir()
    (project / "app.py").write_text("print('hello')\n")
    # the interface starts the runtime itself, starts the analysis, and is closed (EOF) straight away
    first = jarvis(config, input="Analyze this project.\n", cwd=project)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "Analyzing proj in the background" in first.stdout
    assert "keeps running in the background" in first.stdout
    with _client(data) as client:
        _wait(lambda: any(t["title"] == "Analyze proj" and t["status"] == "completed"
                          for t in client.get("/v1/tasks", status="all")["tasks"]))
        task = next(t for t in client.get("/v1/tasks", status="all")["tasks"] if t["title"] == "Analyze proj")
    assert task["request"] == "Analyze this project." and task["result"]
    # reopening: JARVIS knows it finished, and says what it found
    second = jarvis(config, input="What happened while I was away?\n", cwd=project)
    out = second.stdout
    assert "While you were away: Analyze proj completed." in out
    assert "Analyze proj — completed" in out and task["result"].splitlines()[0] in out
    assert len([t for t in _client(data).get("/v1/tasks", status="all")["tasks"]
                if t["title"] == "Analyze proj"]) == 1                                    # never duplicated
    assert jarvis(config, "runtime", "stop").returncode == 0


def test_kill_9_mid_step_is_detected_and_the_step_is_not_blindly_repeated(runtime_dirs, tmp_path):
    data, config = runtime_dirs
    project = _slow_project(tmp_path)
    assert jarvis(config, "runtime", "start").returncode == 0
    with _client(data) as client:
        created = client.post("/v1/tasks", {"objective": "run the slow tests", "cwd": str(project),
                                            "steps": [{"tool": "shell_execute", "description": "run tests",
                                                       "args": {"command": f"{sys.executable} -m pytest -q"}}]})
        task_id = created["task"]["id"]
        _wait(lambda: (client.get(f"/v1/tasks/{task_id}")["task"]["current_step"] or {}).get("status") == "running")
        pid = client.info.pid
    os.kill(pid, signal.SIGKILL)                         # no shutdown, no checkpoint: a crash
    _wait(lambda: read_info(data) is None, 10)
    down = jarvis(config, "runtime", "status")
    assert down.returncode == 3 and "stopped unexpectedly" in down.stdout
    assert jarvis(config, "runtime", "start").returncode == 0
    with _client(data) as client:
        task = client.get(f"/v1/tasks/{task_id}")["task"]
        assert task["status"] == "paused"
        assert task["current_step"]["outcome_unknown"] is True                 # UNKNOWN, not SUCCESS or FAILED
        assert "can't tell whether it finished" in task["recovery"]
        events = client.get("/v1/events", types="SYSTEM_RECOVERED,TASK_RECOVERED")["events"]
        kinds = {e["type"] for e in events}
        assert kinds == {"SYSTEM_RECOVERED", "TASK_RECOVERED"}
        decision = next(e for e in events if e["type"] == "TASK_RECOVERED")["payload"]
        assert decision["decision"] == "paused" and decision["crashed"] and decision["outcome_unknown"]
        away = client.get("/v1/away")["text"]
        assert "stopped unexpectedly" in away
        time.sleep(1.0)
        assert client.get(f"/v1/tasks/{task_id}")["task"]["status"] == "paused"      # still waiting for the user
        client.post(f"/v1/tasks/{task_id}/cancel")
    assert jarvis(config, "runtime", "stop").returncode == 0


def test_sigterm_checkpoints_running_work_and_stops_cleanly(runtime_dirs, tmp_path):
    data, config = runtime_dirs
    project = _slow_project(tmp_path, seconds=20)
    assert jarvis(config, "runtime", "start").returncode == 0
    with _client(data) as client:
        task_id = client.post("/v1/tasks", {"objective": "slow tests", "cwd": str(project),
                                            "steps": [{"tool": "shell_execute",
                                                       "args": {"command": f"{sys.executable} -m pytest -q"}}]}
                              )["task"]["id"]
        _wait(lambda: client.get(f"/v1/tasks/{task_id}")["task"]["status"] == "running")
        pid = client.info.pid
    os.kill(pid, signal.SIGTERM)
    _wait(lambda: read_info(data) is None, 30)
    assert "stopped cleanly" in jarvis(config, "runtime", "status").stdout
    assert jarvis(config, "runtime", "start").returncode == 0
    with _client(data) as client:
        task = client.get(f"/v1/tasks/{task_id}")["task"]
        assert task["status"] == "paused" and "interrupted" in task["recovery"]
        assert not client.get("/v1/events", types="SYSTEM_RECOVERED")["events"]    # a clean stop is not a crash
        client.post(f"/v1/tasks/{task_id}/cancel")
    assert jarvis(config, "runtime", "stop").returncode == 0


def test_install_service_prints_a_definition(runtime_dirs):
    data, config = runtime_dirs
    out = jarvis(config, "runtime", "install-service")
    assert out.returncode == 0 and "runtime run" in out.stdout
    if sys.platform.startswith("linux"):
        assert "[Service]" in out.stdout and "KillSignal=SIGTERM" in out.stdout
