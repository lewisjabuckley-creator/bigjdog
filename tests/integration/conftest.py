"""Live integration tests against a real Ollama server.

Opt-in, because they need a running Ollama and can be slow on CPU:

    JARVIS_OLLAMA_TESTS=1 python -m pytest tests/integration          # Linux/macOS
    set JARVIS_OLLAMA_TESTS=1 && py -m pytest tests/integration       # Windows (cmd)

Environment:
    JARVIS_OLLAMA_URL   server to test (default http://127.0.0.1:11434)
    JARVIS_TEST_MODEL   an installed instruction model (e.g. llama3.1:8b) for the capability tests;
                        without it those tests are skipped

The plumbing tests build tiny deterministic "puppet" models (see ``puppet.py``), so they need no model
download — only ``pip install gguf numpy`` — and remove them afterwards.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

OLLAMA_URL = os.environ.get("JARVIS_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def pytest_collection_modifyitems(config, items):
    here = Path(__file__).parent
    for item in items:
        if Path(str(item.fspath)).parent == here:
            item.add_marker(pytest.mark.ollama)
            if os.environ.get("JARVIS_OLLAMA_TESTS") != "1":
                item.add_marker(pytest.mark.skip(reason="live Ollama tests are opt-in: set JARVIS_OLLAMA_TESTS=1"))


@pytest.fixture(scope="session")
def ollama_url() -> str:
    try:
        httpx.get(f"{OLLAMA_URL}/api/version", timeout=5, trust_env=False).raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"no Ollama server at {OLLAMA_URL}: {exc}")
    return OLLAMA_URL


@dataclass
class Puppets:
    dir: Path
    names: dict[str, str]
    write_target: Path
    delete_target: Path

    def __getitem__(self, key: str) -> str:
        return self.names[key]


@pytest.fixture(scope="session")
def puppets(ollama_url: str):
    pytest.importorskip("gguf", reason="pip install gguf numpy to build puppet models")
    pytest.importorskip("numpy")
    from tests.integration.puppet import Script, create_puppet, tool_call

    workdir = Path(tempfile.mkdtemp(prefix="jarvis-puppets-"))
    write_target = workdir / "files" / "written-by-puppet.txt"
    delete_target = workdir / "files" / "delete-me.txt"
    scripts = {
        "text": Script("Plain answer from the model.", "Plain answer from the model."),
        "sysinfo": Script(tool_call("system_info", {}), "SYSINFO-ANSWER: the machine looks fine."),
        "write": Script(tool_call("file_write", {"path": str(write_target), "content": "written through ollama"}),
                        "WRITE-ANSWER: the file is written."),
        "delete": Script(tool_call("file_delete", {"path": str(delete_target)}), "DELETE-ANSWER: deleted."),
        "background": Script(tool_call("start_background_task", {
            "objective": "echo a marker in the background",
            "steps": [{"tool": "shell_execute", "description": "echo marker",
                       "args": {"command": "echo puppet-background-marker"}}]}),
            "BACKGROUND-ANSWER: started."),
        "shell": Script(tool_call("shell_execute", {"command": "echo puppet-shell-marker"}),
                        "SHELL-ANSWER: the command ran."),
        "hallucinate": Script(tool_call("file_read", {"path": str(workdir / "files" / "nope.txt"), "bogus": 1}),
                              "HALLUCINATE-ANSWER: handled."),
    }
    names = {key: create_puppet(ollama_url, f"jarvis-puppet-{key}:latest", script, workdir)
             for key, script in scripts.items()}
    yield Puppets(workdir, names, write_target, delete_target)
    with httpx.Client(base_url=ollama_url, timeout=30, trust_env=False) as client:
        for name in names.values():
            client.request("DELETE", "/api/delete", json={"model": name})
