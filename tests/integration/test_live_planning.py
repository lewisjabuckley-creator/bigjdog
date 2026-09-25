"""Phase 3 against a real Ollama server: plans proposed by a model, research agents and a model-written summary,
and advice that must not act — with deterministic "puppet" models (see puppet.py), so every assertion is exact.

What this proves is the plumbing and the safety properties through a real inference server: a model's plan is
validated (its destructive step never runs), agents' claims are checked against the sources, and advice never
changes anything even when the model tries to.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import httpx
import pytest

from jarvis.config import config_from_dict
from jarvis.intelligence.plans import NodeKind, NodeStatus, PlanStatus, Quality
from jarvis.runtime import Runtime
from tests.helpers import wait_plan

WORK = Path(tempfile.mkdtemp(prefix="jarvis-live-planning-"))
NOTES = WORK / "notes"


def _notes() -> None:
    NOTES.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        (NOTES / f"arm{i}.md").write_text(f"Robot arm notes, part {chr(65 + i)}.\n"
                                          "The robot arm wrist uses a brushless servo.\n")
    (WORK / "keep.txt").write_text("do not delete me")


@pytest.fixture(scope="module")
def planning_puppets(ollama_url):
    pytest.importorskip("gguf", reason="pip install gguf numpy to build puppet models")
    from tests.integration.puppet import Script, create_puppet, tool_call
    _notes()
    plan_json = json.dumps({"nodes": [
        {"id": "clock", "title": "Read the clock", "tool": "time_now", "args": {}},
        {"id": "system", "title": "Read the system", "tool": "system_info", "args": {}},
        {"id": "wipe", "title": "Tidy up", "tool": "shell_execute", "args": {"command": "rm -rf /"}},
        {"id": "files", "title": "List the folder", "tool": "file_list", "args": {"path": str(WORK)},
         "depends_on": ["clock"]}], "notes": "puppet plan"})
    findings = json.dumps({"status": "completed", "summary": "read them", "confidence": 0.8, "findings": [
        {"text": "brushless wrist servo", "quote": "The robot arm wrist uses a brushless servo.",
         "source": str(NOTES / "arm0.md"), "line": 2},
        {"text": "invented", "quote": "The robot arm runs on steam.", "source": str(NOTES / "arm0.md"), "line": 1}]})
    scripts = {
        "planner": Script(plan_json, plan_json),
        "agent": Script(findings, findings),
        "writer": Script("PUPPET-SUMMARY: the wrist uses a brushless servo [1].",
                         "PUPPET-SUMMARY: the wrist uses a brushless servo [1]."),
        "advisor": Script(tool_call("file_delete", {"path": str(WORK / "keep.txt")}),
                          "ADVICE: keep the file; nothing was changed."),
    }
    names = {key: create_puppet(ollama_url, f"jarvis-puppet-p3-{key}:latest", script, WORK)
             for key, script in scripts.items()}
    yield names
    with httpx.Client(base_url=ollama_url, timeout=30, trust_env=False) as client:
        for name in names.values():
            client.request("DELETE", "/api/delete", json={"model": name})


def _runtime(tmp_path: Path, ollama_url: str, profiles: dict[str, list[str]]) -> Runtime:
    config = {"general": {"data_dir": str(tmp_path / "data")},
              "models": {"ollama": {"base_url": ollama_url, "num_ctx": 2048}, "profiles": profiles},
              "permissions": {"allowed_roots": [str(tmp_path), str(WORK)]}, "monitoring": {"enabled": False},
              "tasks": {"scheduler_interval_s": 0.05}, "scheduler": {"tick_s": 0.1},
              "runtime": {"heartbeat_s": 0.2}, "intelligence": {"process_sample_s": 0.3}}
    return Runtime(config_from_dict(config), mode="embedded")


async def test_a_model_proposed_plan_is_validated_and_its_destructive_step_never_runs(tmp_path, ollama_url,
                                                                                      planning_puppets):
    rt = _runtime(tmp_path, ollama_url, {"planning": [planning_puppets["planner"]],
                                         "conversation": [planning_puppets["writer"]]})
    await rt.start()
    try:
        intel = rt.svc.intelligence
        goal = intel.understand("Work out step by step what time it is and how busy the system is")
        started = await intel.run(goal, cwd=str(WORK))
        assert started.plan is not None, started.problems
        plan = await wait_plan(intel, started.plan.id, timeout=300)
        assert plan.source == "model" and plan.status == PlanStatus.COMPLETED, plan.status_reason
        assert [n.id for n in plan.nodes if n.kind != NodeKind.REPORT] == ["clock", "system", "files"]
        assert plan.facts["files"]["out"]["entries"]                     # the listing really ran
        executed = {e.tool for e in rt.svc.audit.query(action="tool_execute", limit=200)}
        assert "shell_execute" not in executed and {"time_now", "system_info", "file_list"} <= executed
        assert rt.svc.router.status()["last_success"]["model"] == planning_puppets["planner"]
    finally:
        await rt.stop()


async def test_research_agents_and_a_model_summary_through_ollama(tmp_path, ollama_url, planning_puppets):
    rt = _runtime(tmp_path, ollama_url, {"reasoning": [planning_puppets["agent"]],
                                         "summarization": [planning_puppets["writer"]],
                                         "conversation": [planning_puppets["writer"]]})
    await rt.start()
    try:
        intel = rt.svc.intelligence
        started = await intel.run(intel.understand(f"Research what my notes say about the robot arm in {NOTES}"),
                                  cwd=str(WORK))
        plan = await wait_plan(intel, started.plan.id, timeout=600)
        agents = [n for n in plan.nodes if n.kind == NodeKind.AGENT]
        assert agents and all(n.status == NodeStatus.DONE for n in agents), [n.note for n in agents]
        verify = next(n for n in plan.nodes if n.kind == NodeKind.VERIFY)
        assert verify.quality == Quality.PARTIALLY_VERIFIED              # "runs on steam" isn't in the file
        assert "brushless servo" in plan.result and "Left out" in plan.result
        assert "steam" not in plan.result.split("Left out")[0]
        assert f"written by {planning_puppets['writer']}" in plan.result and "PUPPET-SUMMARY" in plan.result
    finally:
        await rt.stop()


async def test_advice_never_acts_even_when_the_model_tries_to(tmp_path, ollama_url, planning_puppets):
    rt = _runtime(tmp_path, ollama_url, {"conversation": [planning_puppets["advisor"]]})
    await rt.start()
    offered: list[list[str]] = []
    real_chat = rt.svc.router.chat

    async def recording_chat(profile, messages, **kw):
        offered.append([t["function"]["name"] for t in kw.get("tools") or []])
        return await real_chat(profile, messages, **kw)
    rt.svc.router.chat = recording_chat                                   # type: ignore[method-assign]
    try:
        await rt.orchestrator().handle("What should I do about keep.txt?", cwd=str(WORK))
        assert (WORK / "keep.txt").exists()                               # the model tried; nothing was deleted
        assert offered and all(offered)
        levels = {rt.svc.registry.get(name).spec.level for names in offered for name in names}
        assert levels == {0}                                              # advice only ever offers observing tools
        assert not [e for e in rt.svc.audit.query(limit=100, include_dry_runs=True) if e.tool == "file_delete"]
    finally:
        await rt.stop()
