import asyncio
import json
from typing import Any

import pytest

from jarvis.core.types import Outcome, Priority
from jarvis.models.base import ChatResponse
from jarvis.permissions.hierarchy import InstructionSource
from jarvis.permissions.model import PermissionLevel
from jarvis.planner.templates import probe_project, run_tests_plan
from jarvis.tasks.manager import TaskManager
from jarvis.tasks.models import InvalidTransition, MonitorSpec, Step, StepStatus, TaskKind, TaskPolicy, TaskStatus
from jarvis.tools.base import Tool, ToolContext, ToolResult, ToolSpec, Verification
from tests.helpers import build_engine

S = TaskStatus


@pytest.fixture
def engine(tmp_path):
    return build_engine(str(tmp_path))


async def started(engine):
    await engine.pool.start()
    return engine


def test_state_machine_is_enforced(db, clock):
    manager = TaskManager(db, clock=clock)
    task = manager.create_task("x")
    with pytest.raises(InvalidTransition):
        manager.transition(task, S.COMPLETED)   # must go through RUNNING and VERIFYING
    manager.transition(task, S.RUNNING)
    manager.transition(task, S.VERIFYING)
    manager.transition(task, S.COMPLETED)
    with pytest.raises(InvalidTransition):
        manager.transition(task, S.RUNNING)
    assert [h["to"] for h in manager.get_task(task.id).history] == ["queued", "running", "verifying", "completed"]


async def test_plan_runs_verifies_and_checkpoints(engine, tmp_path):
    await started(engine)
    target = tmp_path / "out.txt"
    task = engine.tasks.create_task("write and read a file", steps=[
        Step("write the file", "file_write", {"path": str(target), "content": "hello"}),
        Step("read it back", "file_read", {"path": str(target)}),
    ], cwd=str(tmp_path))
    done = await engine.pool.wait_for(task.id)
    assert done.status == S.COMPLETED and done.outcome == Outcome.COMPLETE
    assert done.progress == 1.0
    assert done.plan[0].verification["passed"] is True
    assert set(done.checkpoint["completed_steps"]) == {s.id for s in done.plan}
    assert "verified" in done.outputs["summary"] or "without independent verification" in done.outputs["summary"]
    await engine.pool.stop()


async def test_run_tests_template_reports_failures(engine, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0'\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_demo.py").write_text(
        "def test_ok():\n    assert True\n\ndef test_broken():\n    assert 1 == 2\n")
    probe = probe_project(tmp_path)
    assert probe.test_command == "python3 -m pytest -q" and "python" in probe.languages
    steps, cond = run_tests_plan(str(tmp_path))
    events = []
    engine.bus.subscribe("TEST_FAILED", lambda e: events.append(e.payload))
    await started(engine)
    task = engine.tasks.create_task("run the tests", steps=steps, success_condition=cond, cwd=str(tmp_path))
    done = await engine.pool.wait_for(task.id, timeout=60)
    assert done.status == S.COMPLETED, done.status_reason
    tests = done.outputs["tests"]
    assert tests["failed"] == 1 and tests["passed"] == 1
    assert any("test_broken" in f for f in tests["failures"])
    await engine.bus.drain()
    assert events and events[0]["summary"] == "1 failed, 1 passed"
    await engine.pool.stop()


async def test_consequential_step_waits_for_approval(engine, tmp_path):
    await started(engine)
    (tmp_path / "old.log").write_text("x")
    task = engine.tasks.create_task("clean up", steps=[
        Step("delete old.log", "file_delete", {"path": str(tmp_path / "old.log")})], cwd=str(tmp_path))
    waiting = await engine.pool.wait_for(task.id, [S.WAITING])
    assert "approval" in waiting.status_reason
    assert (tmp_path / "old.log").exists()       # nothing happened without authority
    pending = engine.approvals.pending()
    assert len(pending) == 1 and pending[0].task_id == task.id
    engine.approvals.approve(pending[0].id)
    engine.tasks.resume_task(task.id)
    done = await engine.pool.wait_for(task.id)
    assert done.status == S.COMPLETED and not (tmp_path / "old.log").exists()
    await engine.pool.stop()


async def test_cancel_stops_running_command_quickly(engine, tmp_path):
    engine.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
    await started(engine)
    task = engine.tasks.create_task("long job", steps=[Step("sleep", "shell_execute", {"command": "sleep 20"}),
                                                       Step("after", "time_now", {})], cwd=str(tmp_path))
    await engine.pool.wait_for(task.id, [S.RUNNING])
    await asyncio.sleep(0.1)
    result = engine.tasks.cancel_task(task.id)
    assert result.ok
    done = await engine.pool.wait_for(task.id, timeout=5)
    assert done.status == S.CANCELLED
    assert [s.status for s in done.plan] == [StepStatus.SKIPPED, StepStatus.SKIPPED]
    await engine.pool.stop()


async def test_pause_and_resume_from_checkpoint(engine, tmp_path):
    engine.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
    await started(engine)
    marker = tmp_path / "first.txt"
    task = engine.tasks.create_task("two steps", steps=[
        Step("write marker", "file_write", {"path": str(marker), "content": "1"}),
        Step("wait", "shell_execute", {"command": "sleep 0.4"}),
    ], cwd=str(tmp_path))
    await engine.pool.wait_for(task.id, [S.RUNNING])
    for _ in range(100):
        t = engine.tasks.get_task(task.id)
        if t.plan[1].status == StepStatus.RUNNING:
            break
        await asyncio.sleep(0.01)
    engine.tasks.pause_task(task.id)
    paused = await engine.pool.wait_for(task.id, [S.PAUSED])
    assert paused.plan[0].status == StepStatus.DONE and paused.plan[1].status == StepStatus.PENDING
    # an automation cannot override the user's pause
    denied = engine.tasks.resume_task(task.id, by="automation:nightly", source=InstructionSource.AUTOMATION)
    assert not denied.ok
    engine.tasks.resume_task(task.id)
    done = await engine.pool.wait_for(task.id)
    assert done.status == S.COMPLETED
    writes = [e for e in engine.audit.query(task_id=task.id) if e.tool == "file_write"]
    assert len(writes) == 1   # completed work was not redone
    await engine.pool.stop()


async def test_failed_dependency_blocks_dependent(engine, tmp_path):
    await started(engine)
    build = engine.tasks.create_task("build", steps=[Step("read missing", "file_read",
                                                          {"path": str(tmp_path / "nope")})],
                                     policy=TaskPolicy(on_step_failure="fail"))
    deploy = engine.tasks.create_task("deploy", steps=[Step("t", "time_now", {})], dependencies=[build.id])
    assert (await engine.pool.wait_for(build.id)).status == S.FAILED
    blocked = await engine.pool.wait_for(deploy.id, [S.BLOCKED])
    assert "build" in blocked.status_reason
    await engine.pool.stop()


async def test_recovery_after_crash_does_not_rerun_completed_steps(tmp_path, db, clock):
    engine = build_engine(str(tmp_path), db=db, clock=clock)
    target = tmp_path / "a.txt"
    task = engine.tasks.create_task("migration", steps=[
        Step("schema migration", "file_write", {"path": str(target), "content": "v2"}),
        Step("validate", "file_read", {"path": str(target)}),
    ], cwd=str(tmp_path))
    # simulate a crash mid-task: step 1 done, step 2 running, process gone
    task.plan[0].status = StepStatus.DONE
    target.write_text("v2")
    task.plan[1].status = StepStatus.RUNNING
    engine.tasks.transition(task, S.RUNNING)
    engine.tasks.save(task)

    restarted = build_engine(str(tmp_path), db=db, clock=clock)
    reports = restarted.tasks.recover_interrupted(lambda tool: False)
    assert len(reports) == 1 and not reports[0].resumed
    assert "during 'validate'" in reports[0].summary and "Completed: schema migration" in reports[0].summary
    recovered = restarted.tasks.get_task(task.id)
    assert recovered.status == S.PAUSED
    await restarted.pool.start()
    restarted.tasks.resume_task(task.id)    # "continue"
    done = await restarted.pool.wait_for(task.id)
    assert done.status == S.COMPLETED
    assert not [e for e in restarted.audit.query(task_id=task.id) if e.tool == "file_write"]
    await restarted.pool.stop()


async def test_monitors_resume_automatically_after_restart(tmp_path, db, clock):
    engine = build_engine(str(tmp_path), db=db, clock=clock)
    mon = engine.tasks.create_task("watch folder", kind=TaskKind.MONITOR,
                                   monitor=MonitorSpec({"type": "path", "path": str(tmp_path)}, interval_s=0.02))
    engine.tasks.transition(mon, S.RUNNING)
    restarted = build_engine(str(tmp_path), db=db, clock=clock)
    reports = restarted.tasks.recover_interrupted(lambda tool: False)
    assert reports[0].resumed and restarted.tasks.get_task(mon.id).status == S.QUEUED


async def test_graceful_shutdown_marks_interrupted(engine, tmp_path):
    engine.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
    await started(engine)
    task = engine.tasks.create_task("long", steps=[Step("sleep", "shell_execute", {"command": "sleep 20"})])
    await engine.pool.wait_for(task.id, [S.RUNNING])
    await asyncio.sleep(0.05)
    await engine.pool.stop()
    t = engine.tasks.get_task(task.id)
    assert t.status == S.PAUSED and t.checkpoint.get("interrupted") is True
    reports = engine.tasks.recover_interrupted(lambda tool: False)
    assert reports and "interrupted during 'sleep'" in reports[0].summary


async def test_resource_pressure_throttles_low_priority_work(engine, tmp_path):
    engine.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
    await started(engine)
    task = engine.tasks.create_task("index files", priority=Priority.P3, created_by="automation:indexer",
                                    authority={"interactive": False},
                                    steps=[Step("slow", "shell_execute", {"command": "sleep 20"})])
    await engine.pool.wait_for(task.id, [S.RUNNING])
    engine.state.set("resources.memory_percent", 96.0, ttl=60)
    paused = await engine.pool.wait_for(task.id, [S.PAUSED], timeout=5)
    assert "memory usage is 96%" in paused.status_reason
    explanation = engine.resources.explain(task.id)
    assert explanation and "P3" in explanation.rule
    assert engine.audit.last_decision().action == "pause_task"
    # a new P3 task is deferred, a user task (P1) still starts
    deferred = engine.tasks.create_task("opportunistic", priority=Priority.P4, steps=[Step("t", "time_now", {})])
    urgent = engine.tasks.create_task("user request", priority=Priority.P1, steps=[Step("t", "time_now", {})])
    assert (await engine.pool.wait_for(urgent.id)).status == S.COMPLETED
    assert "deferred" in engine.tasks.get_task(deferred.id).status_reason
    engine.state.set("resources.memory_percent", 50.0, ttl=60)
    resumed = await engine.pool.wait_for(task.id, [S.RUNNING, S.QUEUED], timeout=5)
    assert resumed.status in (S.RUNNING, S.QUEUED)
    assert (await engine.pool.wait_for(deferred.id)).status == S.COMPLETED
    engine.tasks.cancel_task(task.id)
    await engine.pool.stop()


async def test_monitor_watches_task_until_done(engine, tmp_path):
    engine.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
    triggered = []
    engine.bus.subscribe("MONITOR_TRIGGERED", lambda e: triggered.append(e.payload["detail"]))
    await started(engine)
    job = engine.tasks.create_task("build", steps=[Step("compile", "shell_execute", {"command": "sleep 0.2"})])
    mon = engine.tasks.create_task("keep an eye on the build", kind=TaskKind.MONITOR,
                                   monitor=MonitorSpec({"type": "task", "task_id": job.id}, interval_s=0.02))
    done = await engine.pool.wait_for(mon.id, timeout=10)
    assert done.status == S.COMPLETED and done.outputs["stop_reason"] == "resolved"
    await engine.bus.drain()
    assert triggered and "build completed" in triggered[-1]
    await engine.pool.stop()


async def test_path_monitor_detects_changes(engine, tmp_path):
    watched = tmp_path / "inbox"
    watched.mkdir()
    events = []
    engine.bus.subscribe("FILE_CREATED", lambda e: events.append(e.payload["path"]))
    await started(engine)
    mon = engine.tasks.create_task("watch inbox", kind=TaskKind.MONITOR,
                                   monitor=MonitorSpec({"type": "path", "path": str(watched)}, interval_s=0.02,
                                                       stop_when=["triggered_once"]))
    await engine.pool.wait_for(mon.id, [S.RUNNING])
    await asyncio.sleep(0.1)
    (watched / "report.pdf").write_text("x")
    done = await engine.pool.wait_for(mon.id, timeout=5)
    assert "1 created" in done.outputs["summary"]
    await engine.bus.drain()
    assert events == [str(watched / "report.pdf")]
    await engine.pool.stop()


async def test_llm_planning_with_validation(engine, tmp_path):
    (tmp_path / "notes.md").write_text("remember the milk")
    plan = {"steps": [{"description": "list files", "tool": "file_list", "args": {"path": str(tmp_path)}},
                      {"description": "bogus", "tool": "launch_rockets", "args": {}},
                      {"description": "read notes", "tool": "file_read", "args": {"path": str(tmp_path / "notes.md")}}]}
    engine.provider.when("Objective", json.dumps(plan))
    await engine.router.refresh()
    await started(engine)
    task = engine.tasks.create_task("summarize my notes", cwd=str(tmp_path))
    done = await engine.pool.wait_for(task.id)
    assert done.status == S.COMPLETED, done.status_reason
    assert [s.tool for s in done.plan] == ["file_list", "file_read"]     # unknown tool rejected
    assert done.outputs["plan_source"] == "model"
    await engine.pool.stop()


async def test_no_model_means_honest_planning_failure(engine, tmp_path):
    engine.provider.online = False
    await engine.router.refresh()
    await started(engine)
    task = engine.tasks.create_task("do something vague")
    done = await engine.pool.wait_for(task.id)
    assert done.status == S.FAILED and "no language model" in done.status_reason
    await engine.pool.stop()


class FlakyTool(Tool):
    spec = ToolSpec("flaky_read", "fails once", {"type": "object", "properties": {}}, level=PermissionLevel.OBSERVE,
                    idempotent=True)

    def __init__(self):
        self.calls = 0

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.calls += 1
        if self.calls == 1:
            raise ConnectionError("transient glitch")
        return ToolResult(True, "read ok")


class LyingTool(Tool):
    spec = ToolSpec("lying_deploy", "claims success", {"type": "object", "properties": {}},
                    level=PermissionLevel.OBSERVE)

    async def run(self, args, ctx):
        return ToolResult(True, "deployment succeeded")

    async def verify(self, args, result, ctx):
        return Verification(True, False, "health_check", "health endpoint returned 503")


async def test_idempotent_step_retried_after_transient_failure(engine):
    flaky = FlakyTool()
    engine.registry.register(flaky)
    await started(engine)
    task = engine.tasks.create_task("flaky", steps=[Step("read", "flaky_read", {})],
                                    policy=TaskPolicy(retry_backoff_s=0.01))
    done = await engine.pool.wait_for(task.id)
    assert done.status == S.COMPLETED and flaky.calls == 2
    assert engine.audit.query(task_id=task.id, action="recovery_retry")
    await engine.pool.stop()


async def test_task_fails_when_verification_contradicts_tool(engine):
    engine.registry.register(LyingTool())
    engine.provider.online = False   # no replanning possible
    await engine.router.refresh()
    await started(engine)
    task = engine.tasks.create_task("deploy", steps=[Step("deploy", "lying_deploy", {})])
    done = await engine.pool.wait_for(task.id)
    assert done.status == S.FAILED
    assert "503" in done.plan[0].verification["detail"]
    await engine.pool.stop()


async def test_dry_run_task_changes_nothing(engine, tmp_path):
    await started(engine)
    task = engine.tasks.create_task("clean", dry_run=True, steps=[
        Step("write", "file_write", {"path": str(tmp_path / "x.txt"), "content": "x"}),
        Step("delete", "file_delete", {"path": str(tmp_path / "y.txt")})])
    done = await engine.pool.wait_for(task.id)
    assert done.status == S.COMPLETED and done.outcome == Outcome.UNKNOWN
    assert len(done.outputs["dry_run"]) == 2 and not (tmp_path / "x.txt").exists()
    await engine.pool.stop()


async def test_skip_steps_modifies_plan_without_restart(engine, tmp_path):
    engine.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
    await started(engine)
    task = engine.tasks.create_task("release", steps=[
        Step("run tests", "shell_execute", {"command": "sleep 0.3"}),
        Step("deploy to production", "time_now", {})])
    await engine.pool.wait_for(task.id, [S.RUNNING])
    result = engine.tasks.skip_steps(task.id, lambda s: "deploy" in s.description, reason="user: don't deploy yet")
    assert result.ok
    done = await engine.pool.wait_for(task.id)
    assert done.status == S.COMPLETED and done.outcome == Outcome.PARTIAL
    assert done.plan[1].status == StepStatus.SKIPPED
    await engine.pool.stop()


async def test_deadline_at_risk_is_reported(engine, tmp_path, clock):
    engine.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
    events = []
    engine.bus.subscribe("DEADLINE_AT_RISK", lambda e: events.append(e.payload))
    await started(engine)
    import time
    task = engine.tasks.create_task("render", deadline=time.time() + 0.05, steps=[
        Step("a", "shell_execute", {"command": "sleep 0.3"}), Step("b", "shell_execute", {"command": "sleep 0.3"})])
    await engine.pool.wait_for(task.id, timeout=10)
    await engine.bus.drain()
    assert events and events[0]["title"] == "render"
    await engine.pool.stop()


async def test_concurrency_limit_and_priority_order(tmp_path):
    engine = build_engine(str(tmp_path), max_concurrent=1)
    engine.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
    low = engine.tasks.create_task("low", priority=Priority.P3, steps=[Step("s", "shell_execute", {"command": "sleep 0.1"})])
    high = engine.tasks.create_task("high", priority=Priority.P1, steps=[Step("s", "shell_execute", {"command": "sleep 0.1"})])
    await engine.pool.start()
    await engine.pool.wait_idle(timeout=10)
    lo, hi = engine.tasks.get_task(low.id), engine.tasks.get_task(high.id)
    assert hi.started_at <= lo.started_at
    await engine.pool.stop()
