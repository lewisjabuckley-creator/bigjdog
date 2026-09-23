"""Acceptance tests: the twenty "final test" scenarios from the specification (§200).

Each test drives the real runtime (database, event bus, task engine, worker pool,
permissions, notifications, orchestrator) in simulation mode: the model,
metrics and network are simulated; filesystem, subprocesses and SQLite are real.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from jarvis.config import config_from_dict
from jarvis.core.intent import IntentKind
from jarvis.core.modes import Mode
from jarvis.core.types import NotificationPriority as NP
from jarvis.devices.registry import SimulatedFan
from jarvis.models.base import Capability, ModelInfo
from jarvis.models.fake import ScriptedProvider
from jarvis.monitoring.service import SystemMonitor
from jarvis.permissions.model import PermissionLevel
from jarvis.projects.manager import ProjectPolicy
from jarvis.runtime import Runtime
from jarvis.simulation.environment import SimulatedEnvironment
from jarvis.tasks.models import Step, StepStatus, TaskPolicy, TaskStatus

S = TaskStatus


def make_config(tmp: Path, **overrides: Any):
    data: dict[str, Any] = {
        "general": {"data_dir": str(tmp / "data")},
        "permissions": {"allowed_roots": [str(tmp)], "denied_paths": [str(tmp / "secrets")]},
        "monitoring": {"enabled": False},
        "tasks": {"scheduler_interval_s": 0.02},
        "memory": {"use_embeddings": True},
    }
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    return config_from_dict(data)


@asynccontextmanager
async def jarvis(tmp: Path, *, sim: SimulatedEnvironment | None = None, providers: list | None = None,
                 **overrides: Any) -> AsyncIterator[tuple[Runtime, SimulatedEnvironment]]:
    sim = sim or SimulatedEnvironment()
    runtime = Runtime(make_config(tmp, **overrides), providers=providers or [sim.provider], metrics=sim.metrics,
                      network_probe=sim.probe, simulated=True)
    await runtime.start(monitoring=False)
    try:
        yield runtime, sim
    finally:
        await runtime.stop()


def make_project(root: Path, *, failing: bool = False, slow: float = 0.0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.1'\n")
    (root / "tests").mkdir(exist_ok=True)
    body = "import time\n\ndef test_ok():\n"
    body += f"    time.sleep({slow})\n    assert True\n" if slow else "    assert True\n"
    if failing:
        body += "\ndef test_broken():\n    assert 1 + 1 == 3\n"
    (root / "tests" / "test_demo.py").write_text(body)
    return root


def sink_of(runtime: Runtime) -> list:
    delivered: list = []
    runtime.svc.notifications.sinks.append(delivered.append)
    return delivered


async def wait_until(predicate, timeout: float = 15.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise TimeoutError("condition not met")
        await asyncio.sleep(0.02)


# -- Scenario 1 (and the first milestone, spec §166) --------------------------------------------------

async def test_s01_long_running_task_continues_while_user_does_other_things(tmp_path):
    project = make_project(tmp_path / "demo", slow=0.6)
    async with jarvis(tmp_path) as (rt, _):
        orch = rt.orchestrator()
        opened = await orch.handle(f"open the project {project}")
        assert "Opened demo" in opened.text
        started = await orch.handle("run the tests")
        assert started.intent == IntentKind.RUN_TESTS and started.task_id
        # the conversation is not blocked by the running task
        other = await orch.handle("what time is it?")
        assert other.intent == IntentKind.TIME
        assert rt.svc.tasks.get_task(started.task_id).status in (S.QUEUED, S.RUNNING, S.VERIFYING)
        done = await rt.svc.pool.wait_for(started.task_id, timeout=60)
        assert done.status == S.COMPLETED and done.outputs["tests"]["passed"] == 1   # verified via parsed output
        await rt.svc.bus.drain()
        follow_up = await orch.handle("what are you doing?")
        assert any("demo tests finished" in n.text().lower() for n in follow_up.notifications)  # reported


# -- Scenario 2 ---------------------------------------------------------------------------------------

async def test_s02_failure_detected_without_being_asked(tmp_path):
    async with jarvis(tmp_path) as (rt, _):
        delivered = sink_of(rt)
        task = rt.svc.tasks.create_task("nightly report", steps=[
            Step("read the input data", "file_read", {"path": str(tmp_path / "missing.csv")})],
            policy=TaskPolicy(on_step_failure="fail"), created_by="user:owner")
        await rt.svc.pool.wait_for(task.id)
        await rt.svc.bus.drain()
        failed = [n for n in delivered if "failed" in n.title]
        assert failed and failed[0].priority == NP.URGENT and "does not exist" in failed[0].text()


# -- Scenario 3 ---------------------------------------------------------------------------------------

async def test_s03_diagnosis_gathers_evidence(tmp_path):
    async with jarvis(tmp_path) as (rt, _):
        orch = rt.orchestrator()
        task = rt.svc.tasks.create_task("import job", steps=[
            Step("load config", "file_read", {"path": str(tmp_path / "config.yaml")})],
            policy=TaskPolicy(on_step_failure="fail"), created_by="user:owner")
        orch.focus.touch_task(task.id)
        await rt.svc.pool.wait_for(task.id)
        response = await orch.handle("Check why it failed.")
        assert response.intent == IntentKind.DIAGNOSE
        assert "Observed:" in response.text and "config.yaml" in response.text
        assert "Likely cause:" in response.text and "(inferred)" in response.text
        assert response.data["observed"]


# -- Scenario 4 + 9 -----------------------------------------------------------------------------------

async def test_s04_s09_resource_constraint_adapts_and_is_explained(tmp_path):
    async with jarvis(tmp_path) as (rt, sim):
        svc = rt.svc
        svc.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
        task = svc.tasks.create_task("reindex documents", priority=3, created_by="automation:indexer",
                                     authority={"interactive": False},
                                     steps=[Step("index", "shell_execute", {"command": "sleep 30"})])
        await svc.pool.wait_for(task.id, [S.RUNNING])
        monitor = SystemMonitor(sim.metrics, svc.state, svc.config.monitoring, bus=svc.bus, clock=svc.clock)
        sim.metrics.set(memory_percent=96.0)
        await monitor.sample_once()
        paused = await svc.pool.wait_for(task.id, [S.PAUSED], timeout=5)
        assert "memory usage is 96%" in paused.status_reason
        orch = rt.orchestrator()
        why = await orch.handle("Why did you do that?")
        assert "paused 'reindex documents'" in why.text and "memory usage is 96%" in why.text and "P3" in why.text
        slow = await orch.handle("why is the reindex task slow?")
        assert "memory" in slow.text
        sim.metrics.set(memory_percent=40.0)
        await monitor.sample_once()
        await svc.pool.wait_for(task.id, [S.RUNNING, S.QUEUED], timeout=5)
        svc.tasks.cancel_task(task.id)


# -- Scenario 5 ---------------------------------------------------------------------------------------

async def test_s05_dependency_disappearance_detected(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        async with jarvis(tmp_path) as (rt, _):
            delivered = sink_of(rt)
            orch = rt.orchestrator()
            question = await orch.handle("Keep an eye on the server.")
            assert question.kind == "question"
            answer = await orch.handle(str(proc.pid))
            assert answer.task_id and "Watching process" in answer.text
            await rt.svc.pool.wait_for(answer.task_id, [S.RUNNING])
            proc.terminate()
            proc.wait(5)
            await rt.svc.pool.wait_for(answer.task_id, timeout=10)
            await rt.svc.bus.drain()
            assert any("has exited" in n.text() for n in delivered + rt.svc.notifications.pending())
    finally:
        if proc.poll() is None:
            proc.kill()


# -- Scenario 6 ---------------------------------------------------------------------------------------

async def test_s06_user_changes_priorities_mid_task(tmp_path):
    async with jarvis(tmp_path) as (rt, _):
        svc = rt.svc
        svc.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
        task = svc.tasks.create_task("release 1.8.2", created_by="user:owner", steps=[
            Step("run the test suite", "shell_execute", {"command": "sleep 0.5"}),
            Step("build the artifact", "file_write", {"path": str(tmp_path / "artifact.txt"), "content": "v1.8.2"}),
            Step("deploy to production", "shell_execute", {"command": "echo deploying"})])
        orch = rt.orchestrator()
        orch.focus.touch_task(task.id)
        await svc.pool.wait_for(task.id, [S.RUNNING])
        response = await orch.handle("Actually, don't deploy yet.")
        assert "deploy to production" in response.text and "Completed work is kept" in response.text
        done = await svc.pool.wait_for(task.id)
        assert done.status == S.COMPLETED and done.outcome.value == "partial"
        assert [s.status for s in done.plan] == [StepStatus.DONE, StepStatus.DONE, StepStatus.SKIPPED]
        assert (tmp_path / "artifact.txt").read_text() == "v1.8.2"


# -- Scenario 7 ---------------------------------------------------------------------------------------

async def test_s07_restart_recovers_persistent_tasks(tmp_path):
    marker = tmp_path / "schema.sql"
    async with jarvis(tmp_path) as (rt, _):
        svc = rt.svc
        svc.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
        task = svc.tasks.create_task("repository migration", created_by="user:owner", steps=[
            Step("schema migration", "file_write", {"path": str(marker), "content": "ALTER TABLE ..."}),
            Step("database validation", "shell_execute", {"command": "sleep 30"})])
        await wait_until(lambda: svc.tasks.get_task(task.id).plan[1].status == StepStatus.RUNNING)
    # process "restarts"
    async with jarvis(tmp_path) as (rt2, _):
        report = rt2.svc.extra["recovery_reports"]
        assert report and "interrupted during 'database validation'" in report[0].summary
        assert "Completed: schema migration" in report[0].summary
        orch = rt2.orchestrator()
        where = await orch.handle("Where were we?")
        assert "database validation" in where.text
        rt2.svc.tasks.get_task(task.id)
        t = rt2.svc.tasks.get_task(task.id)
        t.plan[1].args["command"] = "true"      # make the resumed step quick for the test
        rt2.svc.tasks.save(t)
        cont = await orch.handle("Continue.")
        assert "Already done: schema migration" in cont.text
        done = await rt2.svc.pool.wait_for(task.id)
        assert done.status == S.COMPLETED
        writes = [e for e in rt2.svc.audit.query(task_id=task.id, limit=50) if e.tool == "file_write"]
        assert len(writes) == 1   # the completed step was not re-run after the restart


# -- Scenario 8 ---------------------------------------------------------------------------------------

async def test_s08_what_are_you_doing_is_accurate(tmp_path):
    async with jarvis(tmp_path) as (rt, _):
        svc = rt.svc
        svc.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
        orch = rt.orchestrator()
        idle = await orch.handle("What are you doing?")
        assert idle.text.startswith("Nothing is running")
        job = svc.tasks.create_task("project tests", created_by="user:owner",
                                    steps=[Step("run", "shell_execute", {"command": "sleep 30"})])
        await svc.pool.wait_for(job.id, [S.RUNNING])
        watch = await orch.handle(f"watch folder {tmp_path}")
        await svc.pool.wait_for(watch.task_id, [S.RUNNING])
        busy = await orch.handle("What are you doing?")
        assert "I'm working on project tests" in busy.text
        assert "monitoring" in busy.text and "Nothing is blocked" in busy.text
        await orch.handle("cancel everything")


# -- Scenario 10 + 11 ---------------------------------------------------------------------------------

async def test_s10_s11_stop_and_continue_from_checkpoint(tmp_path):
    async with jarvis(tmp_path) as (rt, _):
        svc = rt.svc
        svc.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
        out = tmp_path / "step1.txt"
        task = svc.tasks.create_task("data export", created_by="user:owner", steps=[
            Step("write header", "file_write", {"path": str(out), "content": "id,name\n"}),
            Step("export rows", "shell_execute", {"command": "sleep 30"})])
        orch = rt.orchestrator()
        orch.focus.touch_task(task.id)
        await wait_until(lambda: svc.tasks.get_task(task.id).plan[1].status == StepStatus.RUNNING)
        loop = asyncio.get_running_loop()
        began = loop.time()
        stop = await orch.handle("Stop.")
        paused = await svc.pool.wait_for(task.id, [S.PAUSED], timeout=5)
        assert loop.time() - began < 3, "stop must be prompt"
        assert "checkpointed" in stop.text
        assert paused.plan[0].status == StepStatus.DONE and paused.plan[1].status == StepStatus.PENDING
        t = svc.tasks.get_task(task.id)
        t.plan[1].args["command"] = "true"
        svc.tasks.save(t)
        cont = await orch.handle("Continue.")
        assert cont.intent == IntentKind.RESUME
        done = await svc.pool.wait_for(task.id)
        assert done.status == S.COMPLETED
        assert len([e for e in svc.audit.query(task_id=task.id) if e.tool == "file_write"]) == 1


# -- Scenario 12 --------------------------------------------------------------------------------------

async def test_s12_keep_an_eye_on_it(tmp_path):
    async with jarvis(tmp_path) as (rt, _):
        svc = rt.svc
        svc.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
        delivered = sink_of(rt)
        orch = rt.orchestrator()
        build = orch._user_task("build the firmware", title="firmware build",
                                steps=[Step("compile", "shell_execute", {"command": "sleep 0.4"})])
        await svc.pool.wait_for(build.id, [S.RUNNING])
        response = await orch.handle("Tell me when it's done.")
        assert response.intent == IntentKind.MONITOR and "firmware build" in response.text
        monitor = await svc.pool.wait_for(response.task_id, timeout=10)
        assert monitor.status == S.COMPLETED and monitor.outputs["stop_reason"] == "resolved"
        await svc.bus.drain()
        assert any("firmware build completed" in n.text().lower() for n in delivered)


# -- Scenario 13 --------------------------------------------------------------------------------------

async def test_s13_consequential_actions_require_authorization(tmp_path):
    work = tmp_path / "work"
    build = work / "build"
    build.mkdir(parents=True)
    (build / "app.o").write_text("binary")
    async with jarvis(tmp_path) as (rt, _):
        orch = rt.orchestrator()
        await orch.handle(f"open the project {work}")
        ask = await orch.handle("$ rm -r build")
        assert ask.kind == "question" and "approval" in ask.text
        assert build.exists()                               # nothing happened yet
        refused = await orch.handle("$ rm -rf /")
        assert "safety" in refused.text or "won't" in refused.text
        approved = await orch.handle("Proceed.")
        assert "Proceeding" in approved.text
        await rt.svc.pool.wait_for(ask.task_id)
        assert not build.exists()
        entries = rt.svc.audit.query(task_id=ask.task_id)
        executed = [e for e in entries if e.action == "tool_execute"]
        assert executed and executed[0].authorization["basis"] == "grant"   # authority came from the approval


# -- Scenario 14 --------------------------------------------------------------------------------------

async def test_s14_tool_results_are_verified(tmp_path):
    async with jarvis(tmp_path) as (rt, _):
        svc = rt.svc
        fan = SimulatedFan()
        fan.stuck = True                        # accepts commands but does nothing
        svc.devices.register(fan)
        svc.permissions.grant("user:owner", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["device_command"])
        svc.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["device_command"])
        sim_provider: ScriptedProvider = rt._providers[0]
        sim_provider.online = False             # no replanning: the failure must surface
        await svc.router.refresh()
        task = svc.tasks.create_task("cool the lab", created_by="user:owner", steps=[
            Step("set fan to 80%", "device_command", {"device": "lab fan", "command": "set_speed",
                                                      "args": {"speed": 80}, "settle_s": 0.01})])
        done = await svc.pool.wait_for(task.id)
        assert done.status == S.FAILED
        assert "telemetry shows 0" in done.plan[0].verification["detail"]
        response = await rt.orchestrator().handle("what's wrong with cool the lab?")
        assert "verification" in response.text


# -- Scenario 15 --------------------------------------------------------------------------------------

async def test_s15_local_model_offline_rest_keeps_working(tmp_path):
    project = make_project(tmp_path / "svc")
    async with jarvis(tmp_path) as (rt, sim):
        sim.provider.online = False
        await rt.svc.router.refresh()
        orch = rt.orchestrator()
        chat = await orch.handle("Summarize the architecture of this project.")
        assert chat.kind == "error" and "unavailable" in chat.text and "Everything else still works" in chat.text
        assert (await orch.handle("What are you doing?")).text
        await orch.handle(f"open the project {project}")
        tests = await orch.handle("Run the tests")
        done = await rt.svc.pool.wait_for(tests.task_id, timeout=60)
        assert done.status == S.COMPLETED                      # deterministic templates need no model
        status = await orch.handle("status")
        assert "Model: unavailable" in status.text
        diag = await orch.handle("what's wrong?")
        assert "model" in diag.text.lower()


# -- Scenario 16 --------------------------------------------------------------------------------------

async def test_s16_state_questions_use_live_state(tmp_path):
    async with jarvis(tmp_path) as (rt, sim):
        monitor = SystemMonitor(sim.metrics, rt.svc.state, rt.svc.config.monitoring, bus=rt.svc.bus,
                                clock=rt.svc.clock)
        sim.metrics.set(cpu_percent=73.0)
        await monitor.sample_once()
        orch = rt.orchestrator()
        status = await orch.handle("system status")
        assert "CPU: 73%" in status.text
        sim.metrics.set(cpu_percent=12.0)
        await monitor.sample_once()
        assert "CPU: 12%" in (await orch.handle("status")).text
        # the model is given live state, labelled as observed, in its context
        assembled = await orch.context.build("how busy is the machine?", [])
        assert "LIVE STATE (observed just now)" in assembled.messages[0].content
        assert "cpu_percent=12.0" in assembled.messages[0].content


# -- Scenario 17 --------------------------------------------------------------------------------------

async def test_s17_past_events_come_from_memory(tmp_path):
    async with jarvis(tmp_path) as (rt, _):
        await rt.orchestrator().handle("Remember that the staging server listens on port 8443")
    async with jarvis(tmp_path) as (rt2, _):                  # a later session
        orch = rt2.orchestrator()
        recall = await orch.handle("What do you remember about the staging server?")
        assert "8443" in recall.text
        source = await orch.handle("Where did you get that?")
        assert "memory" in source.text
        forgot = await orch.handle("forget about the staging server")
        assert "Deleted 1" in forgot.text


# -- Scenario 18 --------------------------------------------------------------------------------------

async def test_s18_decisions_from_weeks_ago(tmp_path):
    async with jarvis(tmp_path) as (rt, _):
        rec = rt.svc.decisions.record("Primary datastore", "PostgreSQL",
                                      reason="we need transactional guarantees across the order and billing services",
                                      alternatives=["MongoDB", "DynamoDB"], user_involvement="approved by you")
        rt.svc.db.execute("UPDATE decisions SET ts=? WHERE id=?", (rt.svc.clock.now() - 21 * 86400, rec.id))
        answer = await rt.orchestrator().handle("Why did we choose PostgreSQL?")
        assert "transactional guarantees" in answer.text and "MongoDB" in answer.text
        assert answer.provenance and "decision history" in answer.provenance[0].describe()


# -- Scenario 19 --------------------------------------------------------------------------------------

async def test_s19_sensitive_project_applies_its_policy(tmp_path):
    cloud = ScriptedProvider("cloud", local=False, models=[
        ModelInfo("cloud-large", "cloud", False, frozenset({Capability.CHAT, Capability.TOOLS}), parameter_size="400B")])
    sim = SimulatedEnvironment()
    secret_root = tmp_path / "medical"
    secret_root.mkdir()
    (secret_root / "notes.md").write_text("patient cohort analysis")
    (tmp_path / "elsewhere.txt").write_text("unrelated")
    async with jarvis(tmp_path, sim=sim, providers=[sim.provider, cloud], privacy={"allow_cloud": True}) as (rt, _):
        svc = rt.svc
        svc.config.models.profiles["conversation"] = ["cloud-large"]
        from jarvis.models.router import TaskProfile
        assert svc.router.select(TaskProfile()).model == "cloud-large"      # cloud is allowed in general
        svc.projects.create("Medical", str(secret_root),
                            policy=ProjectPolicy(sensitive=True, network=False, allowed_tools=["file_*", "time_now"]))
        orch = rt.orchestrator()
        opened = await orch.handle("Open the medical project")
        assert "sensitive" in opened.text
        assert svc.router.select(TaskProfile()).local                       # local models only now
        blocked = await orch.handle("$ ls")
        assert "not allowed in the active project" in blocked.text
        from jarvis.permissions.model import Actor
        from jarvis.tools.base import ToolContext
        ctx = ToolContext(actor=Actor.user(), cwd=str(secret_root), data_dir=str(tmp_path / "data"))
        outside = await svc.registry.execute("file_read", {"path": str(tmp_path / "elsewhere.txt")}, ctx)
        assert outside.status.value == "denied"                             # isolation: other dirs are off limits
        inside = await svc.registry.execute("file_read", {"path": str(secret_root / "notes.md")}, ctx)
        assert inside.ok


# -- Scenario 20 --------------------------------------------------------------------------------------

async def test_s20_knows_when_not_to_interrupt(tmp_path):
    async with jarvis(tmp_path) as (rt, _):
        svc = rt.svc
        delivered = sink_of(rt)
        orch = rt.orchestrator()
        focus = await orch.handle("Focus mode.")
        assert svc.modes.current == Mode.FOCUS and "hold" in focus.text
        svc.notifications.notify(NP.IMPORTANT, "Research summary is ready")
        svc.notifications.notify(NP.INFORMATIONAL, "Backup completed")
        assert delivered == []                                             # held while focused
        chatty = await orch.handle("what time is it?")
        assert chatty.notifications == []                                   # not even at the next reply
        svc.notifications.notify(NP.CRITICAL, "Disk will be full in 5 minutes")
        assert [n.title for n in delivered] == ["Disk will be full in 5 minutes"]   # critical always gets through
        back = await orch.handle("exit focus mode")
        assert "Research summary is ready" in back.text                    # delivered when focus ends
        assert "Backup completed" not in back.text                          # informational stays in history


# -- extra behaviours from the spec ------------------------------------------------------------------------

async def test_ambiguity_asks_minimum_question_and_resolves_answer(tmp_path):
    async with jarvis(tmp_path) as (rt, _):
        svc = rt.svc
        svc.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
        a = svc.tasks.create_task("frontend build", created_by="user:owner",
                                  steps=[Step("s", "shell_execute", {"command": "sleep 30"})])
        b = svc.tasks.create_task("backend build", created_by="user:owner",
                                  steps=[Step("s", "shell_execute", {"command": "sleep 30"})])
        await svc.pool.wait_for(a.id, [S.RUNNING])
        await svc.pool.wait_for(b.id, [S.RUNNING])
        orch = rt.orchestrator()
        question = await orch.handle("cancel the build")
        assert question.kind == "question" and "frontend build" in question.text and "backend build" in question.text
        answer = await orch.handle("the backend one")
        assert "Cancelled backend build" in answer.text
        assert (await svc.pool.wait_for(b.id)).status == S.CANCELLED
        assert svc.tasks.get_task(a.id).status == S.RUNNING
        await orch.handle("cancel everything")


async def test_open_the_other_one_and_repeat_for_other_project(tmp_path):
    alpha = make_project(tmp_path / "alpha")
    beta = make_project(tmp_path / "beta", failing=True)
    async with jarvis(tmp_path) as (rt, _):
        svc = rt.svc
        svc.projects.create("alpha", str(alpha))
        svc.projects.create("beta", str(beta))
        orch = rt.orchestrator()
        await orch.handle("open the alpha project")
        run = await orch.handle("run the tests")
        await svc.pool.wait_for(run.task_id, timeout=60)
        again = await orch.handle("Do the same thing for the other project.")
        assert "beta" in again.text
        other = await svc.pool.wait_for(again.task_id, timeout=60)
        assert other.cwd == str(beta.resolve()) and other.outputs["tests"]["failed"] == 1
        switched = await orch.handle("Open the other one.")
        assert "Opened beta" in switched.text


async def test_dry_run_changes_nothing(tmp_path):
    project = make_project(tmp_path / "p")
    async with jarvis(tmp_path) as (rt, _):
        orch = rt.orchestrator()
        await orch.handle(f"open the project {project}")
        response = await orch.handle("dry run: run the tests")
        assert response.text.startswith("Dry run")
        done = await rt.svc.pool.wait_for(response.task_id)
        assert done.outcome.value == "unknown" and done.outputs["dry_run"]
        assert not [e for e in rt.svc.audit.query(task_id=done.id) if e.action == "tool_execute"]


async def test_private_mode_blocks_network_and_says_so(tmp_path):
    async with jarvis(tmp_path) as (rt, _):
        rt.svc.permissions.grant("user:owner", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
        rt.svc.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
        orch = rt.orchestrator()
        await orch.handle("private mode")
        response = await orch.handle("$ curl https://example.com")
        assert "Private mode is active" in response.text


async def test_model_tool_loop_uses_tools_and_cites_them(tmp_path):
    async with jarvis(tmp_path) as (rt, sim):
        orch = rt.orchestrator()
        response = await orch.handle("How much memory is the system using?")
        assert "system_info tool reports" in response.text
        assert any(p.describe().startswith("system state") or "system_info" in p.describe()
                   for p in response.provenance)
        audit = rt.svc.audit.query(action="tool_execute")
        assert audit and audit[0].tool == "system_info" and audit[0].actor == "user:owner"


async def test_morning_briefing_and_reentry(tmp_path):
    async with jarvis(tmp_path) as (rt, _):
        orch = rt.orchestrator()
        brief = await orch.handle("Morning.")
        assert brief.intent == IntentKind.BRIEFING and "It's" in brief.text and "healthy" in brief.text
        fresh = await orch.handle("Where were we?")
        assert fresh.intent == IntentKind.REENTRY


async def test_conditional_automation_when_build_finishes_run_tests(tmp_path):
    async with jarvis(tmp_path) as (rt, _):
        svc = rt.svc
        svc.automations.create("test after build", "rule", {
            "when": "TASK_COMPLETED", "if": {"payload.title": {"contains": "build"}},
            "do": {"type": "task", "objective": "verify {title}", "priority": "P2",
                   "steps": [{"tool": "time_now", "description": "post-build check"}]}})
        build = svc.tasks.create_task("nightly build", created_by="user:owner", steps=[Step("t", "time_now", {})])
        await svc.pool.wait_for(build.id)
        await svc.bus.drain()
        await wait_until(lambda: any(t.title == "verify nightly build" for t in svc.tasks.list_tasks()))
        follow = next(t for t in svc.tasks.list_tasks() if t.title == "verify nightly build")
        assert follow.created_by.startswith("automation:") and follow.authority == {"interactive": False}
        assert (await svc.pool.wait_for(follow.id)).status == S.COMPLETED
