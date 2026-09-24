"""Phase 2 against a real Ollama server: the persistent runtime's health and model state, a task that
waits through an Ollama outage, and the acceptance scenario end to end through the CLI and a background
runtime process.

Acceptance: talk to JARVIS through the real model → "Analyze this project." → a durable task → close the
interface → the task completes in the background → reopen → JARVIS knows it completed →
"What happened while I was away?" returns the actual result.

A deterministic puppet model (see puppet.py) stands in for an instruction model, so every assertion is exact.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest

from jarvis.config import config_from_dict
from jarvis.runtime import Runtime
from jarvis.service.client import Client, read_info
from jarvis.service.status import health_report, state_snapshot
from jarvis.tasks.models import Step, StepStatus, TaskStatus

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = "PUPPET-ANALYSIS: proj is a one-file Python script; it has no tests."


@pytest.fixture(scope="module")
def analyst(ollama_url):
    pytest.importorskip("gguf", reason="pip install gguf numpy to build puppet models")
    from tests.integration.puppet import Script, create_puppet
    workdir = Path(tempfile.mkdtemp(prefix="jarvis-analyst-"))
    name = create_puppet(ollama_url, "jarvis-puppet-analyst:latest", Script(ANALYSIS, ANALYSIS), workdir)
    yield name
    with httpx.Client(base_url=ollama_url, timeout=30, trust_env=False) as client:
        client.request("DELETE", "/api/delete", json={"model": name})


def _config(tmp_path: Path, ollama_url: str, model: str) -> dict:
    return {"general": {"data_dir": str(tmp_path / "data")},
            "models": {"ollama": {"base_url": ollama_url, "num_ctx": 2048},
                       "profiles": {"conversation": [model], "summarization": [model], "planning": [model]}},
            "permissions": {"allowed_roots": [str(tmp_path)]}, "monitoring": {"enabled": False},
            "tasks": {"scheduler_interval_s": 0.05}, "runtime": {"heartbeat_s": 0.2}}


async def test_runtime_health_and_model_state_see_the_real_ollama(tmp_path, ollama_url, analyst):
    rt = Runtime(config_from_dict(_config(tmp_path, ollama_url, analyst)), mode="daemon")
    await rt.start()
    try:
        health = await health_report(rt)
        assert health["components"]["ollama"]["status"] == "healthy"
        assert health["components"]["ollama"]["url"] == ollama_url
        assert health["components"]["model_layer"]["status"] == "healthy"
        reply = await rt.orchestrator().handle("Hello JARVIS")
        assert reply.text == ANALYSIS and reply.model == analyst
        state = state_snapshot(rt)
        assert state["ollama"]["reachable"] is True and state["models"]["conversation_model"] == analyst
        assert rt.svc.router.status()["last_success"]["model"] == analyst
    finally:
        await rt.stop()


class _Switch:
    """A TCP relay in front of Ollama that can be taken down and brought back (an Ollama restart)."""

    def __init__(self, target: str) -> None:
        host, port = target.removeprefix("http://").split(":")
        self.target = (host, int(port))
        self.server: asyncio.base_events.Server | None = None
        self.port = 0
        self.open: set[asyncio.StreamWriter] = set()

    async def up(self) -> None:
        async def relay(reader, writer):
            try:
                up_reader, up_writer = await asyncio.open_connection(*self.target)
            except OSError:
                writer.close()
                return
            self.open.update((writer, up_writer))

            async def pipe(src, dst):
                try:
                    while data := await src.read(65536):
                        dst.write(data)
                        await dst.drain()
                except (ConnectionError, asyncio.CancelledError):
                    pass
                finally:
                    dst.close()

            await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))

        self.server = await asyncio.start_server(relay, "127.0.0.1", self.port)
        self.port = self.server.sockets[0].getsockname()[1]

    async def down(self) -> None:
        if self.server is not None:
            self.server.close()
            self.server = None
        for writer in list(self.open):      # kept-alive connections die with the server, as in a real restart
            writer.close()
        self.open.clear()
        await asyncio.sleep(0.05)


async def test_a_task_waits_through_an_ollama_outage(tmp_path, ollama_url, analyst):
    switch = _Switch(ollama_url)
    await switch.up()
    cfg = _config(tmp_path, f"http://127.0.0.1:{switch.port}", analyst)
    rt = Runtime(config_from_dict(cfg), mode="daemon")
    await rt.start()
    try:
        svc = rt.svc
        await switch.down()                                   # Ollama goes away
        await svc.router.refresh()
        assert svc.router.provider_status["ollama"] is False
        task = svc.tasks.create_task("report on the folder", created_by="user:owner", steps=[
            Step("scan", "project_scan", {"path": str(tmp_path)}),
            Step("report", "model_report", {"instruction": "Report", "material": {"$from_step": 0}})])
        waiting = await svc.pool.wait_for(task.id, [TaskStatus.WAITING], timeout=30)
        assert waiting.checkpoint["waiting_for"] == "model" and waiting.plan[0].status == StepStatus.DONE
        health = await health_report(rt)
        assert health["components"]["ollama"]["status"] == "offline"
        assert health["components"]["model_layer"]["status"] == "degraded"
        await switch.up()                                     # Ollama is back on the same address
        done = await svc.pool.wait_for(task.id, timeout=120)
        assert done.status == TaskStatus.COMPLETED and done.result == ANALYSIS
        types = {e.type for e in svc.events.query(types=["MODEL_UNAVAILABLE", "MODEL_RECOVERED"])}
        assert types == {"MODEL_UNAVAILABLE", "MODEL_RECOVERED"}
    finally:
        await rt.stop()
        await switch.down()


# -- the acceptance scenario, through real processes ------------------------------------------------------

def _jarvis(config: Path, data: Path, *args: str, input: str | None = None, cwd: Path | None = None,
            timeout: float = 600) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", "jarvis", "--config", str(config), "--data-dir", str(data), *args],
                          input=input, capture_output=True, text=True, timeout=timeout, env=env,
                          cwd=str(cwd or ROOT))


def test_acceptance_close_the_interface_and_ask_what_happened(tmp_path, ollama_url, analyst):
    data = tmp_path / "data"
    config = tmp_path / "jarvis.toml"
    config.write_text(
        f'[models.ollama]\nbase_url = "{ollama_url}"\nnum_ctx = 2048\n\n'
        f'[models.profiles]\nconversation = ["{analyst}"]\nsummarization = ["{analyst}"]\n\n'
        f'[permissions]\nallowed_roots = ["{tmp_path}"]\n\n'
        '[monitoring]\nenabled = false\n')
    project = tmp_path / "proj"
    project.mkdir()
    (project / "main.py").write_text("print('hello')\n")
    try:
        # 1. talk to JARVIS through the real model, ask for the analysis, then close the window
        first = _jarvis(config, data, input="Hello JARVIS\nAnalyze this project.\n", cwd=project)
        out = first.stdout
        assert first.returncode == 0, out + first.stderr
        assert f"Talking through {analyst}" in out
        assert ANALYSIS in out                                           # the conversation reply, from Ollama
        assert "Analyzing proj in the background" in out
        assert "keeps running in the background" in out
        info = read_info(data)
        assert info is not None                                          # the runtime outlived the interface
        # 2. the task completes with nobody attached
        with Client(info) as client:
            deadline = time.monotonic() + 300
            while True:
                tasks = [t for t in client.get("/v1/tasks", status="all")["tasks"] if t["title"] == "Analyze proj"]
                if tasks and tasks[0]["status"] in ("completed", "failed", "blocked", "waiting"):
                    break
                assert time.monotonic() < deadline, "the analysis did not finish"
                time.sleep(0.5)
            task = tasks[0]
            assert task["status"] == "completed" and task["result"] == ANALYSIS
            assert task["request"] == "Analyze this project." and len(tasks) == 1
            report_step = next(s for s in task["completed_steps"] if s["tool"] == "model_report")
            assert report_step["status"] == "done"
            state = client.get("/v1/state")
            assert state["presence"]["away"] is True                    # nobody was attached
        # 3. reopen: JARVIS knows it completed, and reports the actual result
        second = _jarvis(config, data, input="What happened while I was away?\n", cwd=project)
        out = second.stdout
        assert "While you were away: Analyze proj completed." in out
        assert "Analyze proj — completed" in out and ANALYSIS in out
        assert "kept running the whole time" in out
    finally:
        stopped = _jarvis(config, data, "runtime", "stop", timeout=60)
        assert stopped.returncode == 0, stopped.stdout + stopped.stderr
