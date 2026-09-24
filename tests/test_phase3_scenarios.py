"""Phase 3 definition of done: the ten scenarios, through the conversation exactly as a user would type them.

1. a simple question gets a direct answer (no planning)       6. nothing is called done until it is verified
2. a complex investigation and fix                            7. a restart in the middle, then "continue the backup"
3. recovery from a broken tool                                8. a multi-agent task
4. a user correction                                          9. "why did you do that?"
5. resource pressure                                          10. "what should I do?" (advice, no action)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import psutil

from jarvis.intelligence.plans import NodeKind, NodeStatus, PlanStatus, Quality
from jarvis.models.base import Purpose
from jarvis.models.router import TaskProfile
from jarvis.tools.base import ToolResult
from tests.helpers import kill_pid, make_runtime, spawn_cpu_hog, wait_plan, wait_until
from tests.test_planning import _agent_model, _research_files, alive, cpu_drops_when_stopped, runtime

P = PlanStatus


def _plans(rt: Any) -> list[Any]:
    return rt.svc.intelligence.store.list(limit=20)


async def test_1_a_simple_question_gets_a_direct_answer_without_planning(tmp_path):
    async with runtime(tmp_path, cpu=23.0) as (rt, sim):
        o = rt.orchestrator()
        started = time.monotonic()
        reply = await o.handle("Check my CPU temperature", cwd=str(tmp_path))
        assert time.monotonic() - started < 2.0
        assert reply.intent.value == "system_query"
        assert "can't read the CPU temperature" in reply.text and "23%" in reply.text     # honest: no sensor here
        sim.metrics.set(cpu_temp_c=61.0)
        assert (await o.handle("what's my cpu temperature?")).text == "The CPU is at 61°C."
        assert not _plans(rt) and not rt.svc.tasks.list_tasks(limit=10)                 # no plan, no task
        assert rt.svc.audit.query(action="tool_execute")[0].tool == "system_info"        # still audited


async def test_2_and_9_investigate_and_fix_then_explain_why(tmp_path):
    hog = spawn_cpu_hog()
    try:
        async with runtime(tmp_path, cpu=97.0) as (rt, sim):
            cpu_drops_when_stopped(rt, sim)
            o = rt.orchestrator()
            ask = await o.handle("My computer is slow — find out why and fix it", cwd=str(tmp_path))
            assert ask.kind == "question" and ask.approval_id
            assert f"PID {hog}" in ask.text and "CPU usage is 97%" in ask.text and ask.text.endswith("Proceed?")
            status = await o.handle("what are you doing?")
            assert "waiting" in status.text and "approval" in status.text
            done = await o.handle("yes")
            assert "Checked independently" in done.text and "Result: verified independently." in done.text
            assert not alive(hog)
            why = await o.handle("Why did you do that?")
            assert f"I decided to stop python (PID {hog})" in why.text or "I decided to stop" in why.text
            assert "The evidence:" in why.text and "You approved it at" in why.text
            assert "an independent check found" in why.text
            plan = _plans(rt)[0]
            assert plan.status == P.COMPLETED and plan.quality == Quality.VERIFIED
    finally:
        kill_pid(hog)


async def test_3_a_broken_tool_is_worked_around(tmp_path):
    hog = spawn_cpu_hog()
    try:
        async with runtime(tmp_path, cpu=97.0) as (rt, sim):
            async def broken(args: dict, ctx: Any) -> ToolResult:
                raise RuntimeError("process table unavailable")
            rt.svc.registry.get("process_list").run = broken           # type: ignore[method-assign]
            o = rt.orchestrator()
            ask = await o.handle("My computer is slow, find out why and fix it", cwd=str(tmp_path))
            if ask.kind != "question":                                   # the retry and fallback take a moment
                plan = await wait_plan(rt.svc.intelligence, _plans(rt)[0].id, [P.WAITING], timeout=20)
                ask = await o.handle("show me the plan")
            plan = _plans(rt)[0]
            assert plan.status == P.WAITING and f"PID {hog}" in json.dumps(plan.facts.get("analyze", {}))
            changes = await o.handle("what did you change in the plan?")
            assert "tool failure" in changes.text and "shell_execute" in changes.text
            await o.handle("cancel it")
    finally:
        kill_pid(hog)


async def test_4_a_correction_changes_the_plan_while_it_runs(tmp_path):
    hog = spawn_cpu_hog()
    try:
        async with runtime(tmp_path, cpu=97.0) as (rt, sim):
            o = rt.orchestrator()
            ask = await o.handle("My computer is slow, fix it", cwd=str(tmp_path))
            assert ask.kind == "question"
            name = psutil.Process(hog).name()
            fixed = await o.handle(f"leave {name} alone")
            assert fixed.kind == "action" and "dropped" in fixed.text and "Completed work is kept" in fixed.text
            plan = await wait_plan(rt.svc.intelligence, _plans(rt)[0].id)
            assert alive(hog) and not rt.svc.approvals.pending()
            assert f"I left {name} alone, as you asked." in plan.result
    finally:
        kill_pid(hog)


async def test_5_background_work_yields_to_resource_pressure(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    _research_files(notes, 3)
    async with runtime(tmp_path) as (rt, sim):
        _agent_model(sim)
        svc = rt.svc
        sim.metrics.set(memory_percent=97.0)
        await svc.extra["system_monitor"].sample_once()
        o = rt.orchestrator()
        await o.handle(f"Research what my notes say about the robot arm in {notes}, in the background", cwd=str(tmp_path))
        plan = await wait_plan(svc.intelligence, _plans(rt)[0].id, [P.WAITING], timeout=15)
        assert plan.priority.value == "background" and "waiting for resources" in plan.status_reason
        assert plan.node("summary").task_id is None                     # the model-heavy step is held back
        status = await o.handle("what are you doing?")
        assert "waiting for resources" in status.text
        # background model requests move to a smaller model while memory is short
        profile = svc.router.profile_hook(TaskProfile(purpose=Purpose.SUMMARIZATION, complexity="high",
                                                      interactive=False))
        assert svc.router.select(profile).model == "sim-small:3b"
        sim.metrics.set(memory_percent=40.0)
        await svc.extra["system_monitor"].sample_once()
        plan = await wait_plan(svc.intelligence, plan.id, timeout=20)
        assert plan.status == P.COMPLETED and plan.node("summary").status == NodeStatus.DONE


async def test_6_nothing_is_called_done_until_it_is_verified(tmp_path):
    hog = spawn_cpu_hog()
    try:
        async with runtime(tmp_path, cpu=97.0) as (rt, sim):              # stopping it won't ease the CPU
            o = rt.orchestrator()
            await o.handle("My computer is slow, fix it", cwd=str(tmp_path))
            outcome = await o.handle("go ahead")
            if "Proceeding" in outcome.text:
                await wait_plan(rt.svc.intelligence, _plans(rt)[0].id)
                outcome = await o.handle("show me the plan")
            plan = _plans(rt)[0]
            assert plan.status == P.FAILED and plan.quality != Quality.VERIFIED
            text = outcome.text + plan.result
            assert "still under pressure" in text and "verified independently." not in text
            await rt.svc.bus.drain()
            titles = [n["text"] for n in rt.svc.notifications.list(limit=20)]
            assert not any("done" in t.split(".")[0].lower() for t in titles if plan.title.lower() in t.lower())
    finally:
        kill_pid(hog)


async def test_7_restart_in_the_middle_then_continue_the_backup(tmp_path):
    src, dst = tmp_path / "Documents", tmp_path / "Backup"
    src.mkdir()
    for i in range(4):
        (src / f"report{i}.txt").write_text(f"report {i}\n" * 100)
    rt, sim = make_runtime(str(tmp_path))
    await rt.start()
    tool = rt.svc.registry.get("file_copy")
    real = tool.run

    async def slow(args: dict, ctx: Any) -> ToolResult:
        await asyncio.sleep(20)
        return await real(args, ctx)
    tool.run = slow                                                      # type: ignore[method-assign]
    o = rt.orchestrator()
    first = await o.handle(f"Back up {src} to {dst}", cwd=str(tmp_path))
    assert first.kind == "action" and "keeps going if you close this window" in first.text
    plan_id = _plans(rt)[0].id
    await wait_until(lambda: rt.svc.intelligence.get(plan_id).node("copy").status == NodeStatus.RUNNING)
    stopped = await o.handle("stop the backup")
    assert "Stopped" in stopped.text and "checkpointed" in stopped.text
    await rt.stop()                                                      # JARVIS restarts
    rt2, _ = make_runtime(str(tmp_path))
    await rt2.start()
    try:
        assert rt2.svc.intelligence.get(plan_id).status == P.PAUSED
        reply = await rt2.orchestrator().handle("Continue the backup", cwd=str(tmp_path))
        assert reply.text.startswith("Resuming back up Documents") and "Already done: Look at" in reply.text
        plan = await wait_plan(rt2.svc.intelligence, plan_id)
        assert plan.status == P.COMPLETED and plan.quality == Quality.VERIFIED
        assert sorted(p.name for p in (dst / "Documents").iterdir()) == [f"report{i}.txt" for i in range(4)]
        assert len(plan.node("check").task_ids) == 1                    # completed work was not repeated
    finally:
        await rt2.stop()


async def test_8_a_multi_agent_research_task(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    _research_files(notes, 10)
    async with runtime(tmp_path) as (rt, sim):
        calls = _agent_model(sim)
        o = rt.orchestrator()
        reply = await o.handle(f"Research what my notes say about the robot arm in {notes}", cwd=str(tmp_path))
        plan = await wait_plan(rt.svc.intelligence, _plans(rt)[0].id, timeout=20)
        agents = [n for n in plan.nodes if n.kind == NodeKind.AGENT]
        assert len(agents) >= 2 and calls["agents"] == len(agents)
        text = plan.result
        assert "What the sources say about the robot arm" in text and "[1]" in text
        assert "Left out 1 statement(s)" in text or "Left out" in text      # the invented claim is not passed on
        assert "fusion reactor" not in text.split("Left out")[0]
        assert reply.kind in ("answer", "action")


async def test_10_what_should_i_do_advises_without_acting_then_fix_it(tmp_path):
    hog = spawn_cpu_hog()
    try:
        async with runtime(tmp_path, cpu=97.0) as (rt, sim):
            o = rt.orchestrator()
            advice = await o.handle("What should I do about my computer being slow?", cwd=str(tmp_path))
            assert "Recommended:" in advice.text and f"PID {hog}" in advice.text
            assert "I haven't changed anything" in advice.text
            assert alive(hog) and not rt.svc.approvals.pending()
            assert not [e for e in rt.svc.audit.query(action="tool_execute", limit=500) if e.tool == "process_stop"]
            then = await o.handle("fix it")
            assert then.kind == "question" and then.approval_id and f"PID {hog}" in then.text
            await o.handle("no")
    finally:
        kill_pid(hog)


# -- the local API --------------------------------------------------------------------------------------------------

async def test_goals_plans_and_intelligence_over_the_local_api(tmp_path):
    import httpx

    from jarvis.service.api import ApiServer
    src, dst = tmp_path / "docs", tmp_path / "copy"
    src.mkdir()
    (src / "a.txt").write_text("a")
    rt, sim = make_runtime(str(tmp_path), mode="daemon")
    await rt.start()
    server = ApiServer(rt, "tok", sim=sim)
    port = await server.start()
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", headers={"Authorization": "Bearer tok"},
                                 trust_env=False, timeout=20) as client:
        try:
            simple = (await client.post("/v1/goals", json={"text": "check my cpu temperature"})).json()
            assert simple["plan"] is None and "no plan needed" in simple["note"]
            made = await client.post("/v1/goals", json={"text": f"back up {src} to {dst}", "cwd": str(tmp_path)})
            assert made.status_code == 201
            plan_id = made.json()["plan"]["id"]
            await wait_plan(rt.svc.intelligence, plan_id)
            detail = (await client.get(f"/v1/plans/{plan_id}")).json()
            assert detail["plan"]["status"] == "completed" and detail["plan"]["quality"] == "verified"
            assert "Plan:" in detail["preview"] and detail["revisions"] == []
            listed = (await client.get("/v1/plans", params={"status": "all"})).json()["plans"]
            assert [p["id"] for p in listed] == [plan_id]
            state = (await client.get("/v1/intelligence")).json()
            assert state["autonomy"]["level"] == "normal" and state["agents"]["contracts"]
            assert state["inference"]["limit"] >= 1
            assert (await client.post("/v1/intelligence/autonomy", json={"level": "high"})).status_code == 200
            assert (await client.post("/v1/intelligence/autonomy", json={"level": "reckless"})).status_code == 400
            assert (await client.get("/v1/goals")).json()["goals"]
            assert (await client.get("/v1/plans/plan-nope")).status_code == 404
        finally:
            await server.stop()
            await rt.stop()
