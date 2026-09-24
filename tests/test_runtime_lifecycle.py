"""Phase 2: the persistent runtime — single instance, run records, restart recovery, unknown outcomes,
the task record, health, live state, presence, notifications, resource policy and model waiting."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import time

import pytest

from jarvis.core.awareness import away_report, latest_briefing, prepare_briefing
from jarvis.core.intent import IntentKind, parse
from jarvis.core.types import HealthStatus, NotificationPriority, Outcome, Priority
from jarvis.database.db import Database
from jarvis.database.schema import MIGRATIONS
from jarvis.events.bus import EventBus
from jarvis.events.store import EventStore
from jarvis.permissions.model import PermissionLevel
from jarvis.platforms import LockHeld
from jarvis.service.status import health_report, state_snapshot
from jarvis.state.health import HealthRegistry
from jarvis.tasks.manager import TaskManager
from jarvis.tasks.models import Step, StepStatus, TaskPolicy, TaskStatus
from jarvis.tasks.resources import ResourceManager
from tests.helpers import build_engine, make_runtime, wait_until

S = TaskStatus


# -- one runtime, run records --------------------------------------------------------------------------

async def test_one_runtime_per_data_directory(tmp_path):
    first, _ = make_runtime(str(tmp_path))
    await first.start()
    second, _ = make_runtime(str(tmp_path))
    with pytest.raises(LockHeld) as held:
        await second.start()
    assert held.value.pid == os.getpid()
    await first.stop()
    await second.start()        # the lock is released on stop
    await second.stop()


async def test_clean_stop_is_recorded_and_an_unclean_one_is_detected(tmp_path):
    rt, _ = make_runtime(str(tmp_path))
    await rt.start()
    run_id = rt.run_id
    await rt.stop()
    db = Database(rt.config.db_path)
    row = db.query_one("SELECT clean, stopped_at FROM runtime_runs WHERE id=?", (run_id,))
    assert row["clean"] == 1 and row["stopped_at"]
    now = time.time()
    # a run that died without recording a clean stop (kill -9, power loss)
    db.execute("INSERT INTO runtime_runs(id, pid, mode, started_at, heartbeat_at, clean) VALUES(?,?,?,?,?,0)",
               ("run-crashed", 999_999, "daemon", now, now))
    db.close()
    rt2, _ = make_runtime(str(tmp_path), mode="daemon")
    report = await rt2.start()
    try:
        assert report.unclean_previous_stop
        recovered = rt2.svc.events.query(types=["SYSTEM_RECOVERED"])
        assert recovered and "stopped without shutting down" in recovered[0].payload["summary"]
        assert rt2.svc.audit.query(action="unclean_stop_detected")
        # nobody is attached to a daemon yet: the news waits for the user
        assert any("unexpected stop" in n.title for n in rt2.svc.notifications.pending())
    finally:
        await rt2.stop()


def test_a_v1_database_is_migrated_in_place(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    for statement in [s for s in MIGRATIONS[0][1].split(";") if s.strip()]:
        conn.execute(statement)
    conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', '1')")
    conn.execute("INSERT INTO tasks(id, kind, title, objective, owner, created_by, priority, status, data, created_at, "
                 "updated_at) VALUES('task-old', 'oneshot', 'old', 'old task', 'owner', 'user', 1, 'queued', "
                 "'{\"plan\": []}', 1, 1)")
    conn.commit()
    conn.close()
    db = Database(path)
    assert db.schema_version() == 2
    task = TaskManager(db).get_task("task-old")
    assert task.title == "old" and task.idempotency_key is None and task.request == "" and task.artifacts == []
    db.close()


# -- the task record ------------------------------------------------------------------------------------

async def test_an_idempotency_key_never_creates_a_second_task(db):
    bus = EventBus(EventStore(db))
    created = []
    bus.subscribe("TASK_CREATED", lambda e: created.append(e.task_id))
    tasks = TaskManager(db, bus=bus)
    first = tasks.create_task("nightly backup", idempotency_key="schedule:auto-1:1700000000")
    again = tasks.create_task("nightly backup", idempotency_key="schedule:auto-1:1700000000")
    other = tasks.create_task("nightly backup", idempotency_key="schedule:auto-1:1700086400")
    await bus.drain()
    assert first.id == again.id != other.id
    assert created == [first.id, other.id]


async def test_the_task_record_has_every_field_an_interface_needs(tmp_path):
    rt, _ = make_runtime(str(tmp_path))
    await rt.start()
    try:
        target = tmp_path / "note.txt"
        orch = rt.orchestrator()
        orch._current_text = "write the note"
        task = orch._user_task("write a note", steps=[Step("write note", "file_write",
                                                           {"path": str(target), "content": "hi"})])
        done = await rt.svc.pool.wait_for(task.id)
        api = done.to_api()
        for key in ("id", "request", "goal", "status", "current_step", "checkpoint", "completed_steps",
                    "failed_steps", "retry_count", "dependencies", "permissions", "artifacts", "result", "error",
                    "created_at", "updated_at", "started_at", "finished_at", "origin"):
            assert key in api, key
        assert api["status"] == "completed" and api["request"] == "write the note"
        assert api["origin"] == "conversation:default"
        assert api["artifacts"] == [{"type": "file", "path": str(target), "step": "write note"}]
        assert api["result"] and api["error"] is None and api["permissions"]["interactive"] is True
    finally:
        await rt.stop()


async def test_a_worker_crash_leaves_the_step_outcome_unknown_not_failed(tmp_path):
    engine = build_engine(str(tmp_path))
    await engine.pool.start()
    real_execute = engine.registry.execute

    async def broken(*args, **kwargs):
        raise RuntimeError("executor bug")

    engine.executor.registry.execute = broken
    task = engine.tasks.create_task("append a line", steps=[Step("write", "file_write",
                                                                 {"path": str(tmp_path / "x"), "content": "1"})])
    blocked = await engine.pool.wait_for(task.id, [S.BLOCKED])
    assert blocked.outcome == Outcome.UNKNOWN and "outcome is unknown" in blocked.status_reason
    assert blocked.plan[0].outcome_unknown and blocked.plan[0].status == StepStatus.PENDING
    assert engine.audit.query(action="task_crashed")
    engine.executor.registry.execute = real_execute
    engine.tasks.resume_task(task.id)             # the user decides to run it again
    done = await engine.pool.wait_for(task.id)
    assert done.status == S.COMPLETED and not done.plan[0].outcome_unknown
    await engine.pool.stop()


# -- restart recovery ------------------------------------------------------------------------------------

def _interrupted(engine, tmp_path, *, tool: str = "file_read", cwd: str | None = None, **create):
    args = {"path": str(tmp_path / "a.txt")} if tool == "file_read" else {"path": str(tmp_path / "a.txt"),
                                                                            "content": "x"}
    task = engine.tasks.create_task("job", steps=[Step("done first", "time_now", {}), Step("in flight", tool, args)],
                                    cwd=cwd, **create)
    task.plan[0].status = StepStatus.DONE
    task.plan[1].status = StepStatus.RUNNING       # the process died here
    engine.tasks.transition(task, S.RUNNING)
    engine.tasks.save(task)
    return task


async def test_recovery_resumes_when_the_step_in_flight_is_safe_to_repeat(tmp_path, db):
    (tmp_path / "a.txt").write_text("a")
    engine = build_engine(str(tmp_path), db=db)
    task = _interrupted(engine, tmp_path)
    restarted = build_engine(str(tmp_path), db=db)
    events = []
    restarted.bus.subscribe("TASK_RECOVERED", events.append)
    reports = restarted.tasks.recover_interrupted(lambda tool: tool == "file_read", audit=restarted.audit)
    await restarted.bus.drain()
    assert reports[0].decision == "resumed" and not reports[0].outcome_unknown
    assert restarted.tasks.get_task(task.id).status == S.QUEUED
    assert events[0].payload["decision"] == "resumed" and events[0].payload["crashed"] is True
    decision = restarted.audit.query(action="recovery_decision")[0]
    assert "safe to repeat" in decision.summary
    await restarted.pool.start()
    done = await restarted.pool.wait_for(task.id)
    assert done.status == S.COMPLETED and [s.status for s in done.plan] == [StepStatus.DONE, StepStatus.DONE]
    await restarted.pool.stop()


async def test_recovery_never_repeats_a_step_whose_outcome_is_unknown(tmp_path, db):
    engine = build_engine(str(tmp_path), db=db)
    task = _interrupted(engine, tmp_path, tool="file_write")
    restarted = build_engine(str(tmp_path), db=db)
    reports = restarted.tasks.recover_interrupted(lambda tool: tool == "file_read", audit=restarted.audit)
    held = restarted.tasks.get_task(task.id)
    assert reports[0].decision == "paused" and reports[0].outcome_unknown
    assert held.status == S.PAUSED and held.plan[1].outcome_unknown
    assert "can't tell whether it finished" in held.recovery and "Say 'continue'" in held.recovery
    assert restarted.audit.query(action="recovery_decision")[0].outcome == "unknown"
    await restarted.pool.start()
    await asyncio.sleep(0.1)
    assert not restarted.audit.query(action="tool_execute")      # nothing ran on its own


@pytest.mark.parametrize("case", ["missing_cwd", "disabled_automation", "stale", "ask_policy", "failed_dependency"])
async def test_recovery_validates_before_resuming(tmp_path, db, case):
    (tmp_path / "a.txt").write_text("a")
    engine = build_engine(str(tmp_path), db=db)
    create: dict = {}
    if case == "missing_cwd":
        create["cwd"] = str(tmp_path / "gone")
    if case == "disabled_automation":
        create.update(created_by="automation:auto-x", authority={"interactive": False})
    if case == "ask_policy":
        create["policy"] = TaskPolicy(resume_after_restart="ask")
    if case == "failed_dependency":
        dep = engine.tasks.create_task("dep", steps=[Step("t", "time_now", {})])
        engine.tasks.transition(dep, S.RUNNING)
        engine.tasks.transition(dep, S.FAILED, "boom")
        create["dependencies"] = [dep.id]
    task = _interrupted(engine, tmp_path, **create)
    if case == "stale":
        db.execute("UPDATE tasks SET updated_at=? WHERE id=?", (time.time() - 3 * 86400, task.id))
    restarted = build_engine(str(tmp_path), db=db)
    reports = restarted.tasks.recover_interrupted(lambda tool: True, max_age_s=86400,
                                                  automation_state=lambda aid: False)
    report = next(r for r in reports if r.task_id == task.id)
    status = restarted.tasks.get_task(task.id).status
    expected = {"missing_cwd": ("blocked", "no longer exists"), "failed_dependency": ("blocked", "dependency"),
                "disabled_automation": ("paused", "automation that created it is disabled"),
                "stale": ("paused", "hours ago"), "ask_policy": ("paused", "policy is to ask")}[case]
    assert report.decision == expected[0] and expected[1] in report.summary
    assert status == (S.BLOCKED if expected[0] == "blocked" else S.PAUSED)


# -- health, live state ---------------------------------------------------------------------------------

async def test_health_changed_is_published_on_transitions_only(db):
    bus = EventBus(EventStore(db))
    changes = []
    bus.subscribe("HEALTH_CHANGED", changes.append)
    health = HealthRegistry(bus)
    health.register("database", critical=True)
    health.report("database", HealthStatus.HEALTHY)          # first observation: not news
    for _ in range(5):
        health.report("database", HealthStatus.HEALTHY)
    health.report("database", HealthStatus.CRITICAL, "disk I/O error")
    health.report("database", HealthStatus.CRITICAL, "disk I/O error")
    health.report("database", HealthStatus.HEALTHY)
    await bus.drain()
    assert [(e.payload["from"], e.payload["to"]) for e in changes] == [("healthy", "critical"), ("critical", "healthy")]


async def test_one_health_report_covers_every_subsystem(tmp_path):
    rt, sim = make_runtime(str(tmp_path))
    await rt.start()
    try:
        await wait_until(lambda: "scheduler" in rt.loop_ticks())
        report = await health_report(rt)
        assert set(report["components"]) == {"runtime", "database", "task_engine", "workers", "scheduler",
                                             "event_bus", "monitoring", "model_layer", "ollama"}
        assert report["overall"] == "healthy"
        assert report["components"]["monitoring"]["status"] == "disabled"
        assert report["components"]["ollama"]["status"] == "disabled"     # only the simulated provider here
        sim.provider.online = False
        await rt.svc.router.refresh()
        degraded = await health_report(rt)
        assert degraded["components"]["model_layer"]["status"] == "degraded" and degraded["overall"] == "degraded"
    finally:
        await rt.stop()


async def test_live_state_includes_jarvis_own_cost(tmp_path):
    rt, sim = make_runtime(str(tmp_path))
    sim.metrics.set(battery_percent=81, battery_plugged=False)
    await rt.start()
    try:
        await wait_until(lambda: rt.svc.state.value("runtime.self.memory_mb") is not None)
        snap = state_snapshot(rt)
        for key in ("runtime", "health", "resources", "battery", "network", "ollama", "models", "tasks", "workers",
                    "alerts", "presence", "schedule"):
            assert key in snap, key
        assert snap["runtime"]["self"]["memory_mb"] > 0 and snap["resources"]["cpu_percent"] == 10.0
        assert snap["battery"]["percent"] == 81 and snap["models"]["conversation_model"] == "sim-chat:8b"
    finally:
        await rt.stop()


# -- presence, notifications, away -------------------------------------------------------------------------

async def test_notifications_wait_for_the_user_and_survive_a_restart(tmp_path):
    rt, _ = make_runtime(str(tmp_path), mode="daemon")
    await rt.start()
    created, acked = [], []
    rt.svc.bus.subscribe("NOTIFICATION_CREATED", created.append)
    rt.svc.bus.subscribe("NOTIFICATION_ACKNOWLEDGED", acked.append)
    shown = []
    rt.svc.notifications.sinks.append(shown.append)
    n = rt.svc.notifications.notify(NotificationPriority.URGENT, "Backup failed", source="test")
    await rt.svc.bus.drain()
    assert n.state == "queued" and not shown and created[0].payload["id"] == n.id   # nobody attached: queued
    await rt.stop()

    rt2, _ = make_runtime(str(tmp_path), mode="daemon")
    await rt2.start()
    try:
        assert [p.id for p in rt2.svc.notifications.pending()] == [n.id]
        shown2 = []
        rt2.svc.notifications.sinks.append(shown2.append)
        client = rt2.svc.presence.attach("cli")
        live = rt2.svc.notifications.notify(NotificationPriority.URGENT, "Disk almost full", source="test")
        assert live.state == "delivered" and shown2 == [live]
        acked2 = []
        rt2.svc.bus.subscribe("NOTIFICATION_ACKNOWLEDGED", acked2.append)
        assert rt2.svc.notifications.acknowledge(n.id) == 1
        await rt2.svc.bus.drain()
        assert acked2 and acked2[0].payload["ids"] == [n.id]
        rt2.svc.presence.detach(client.client_id)
        assert rt2.svc.presence.away
    finally:
        await rt2.stop()


async def test_what_happened_while_i_was_away_reports_the_actual_result(tmp_path):
    project = tmp_path / "proj"
    (project / "src").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='proj'\n")
    (project / "src" / "app.py").write_text("print('hi')\n" * 10)
    rt, sim = make_runtime(str(tmp_path), mode="daemon")
    sim.provider.when("Analyze the software project", "REPORT: proj is a small Python app with no tests.")
    await rt.start()
    try:
        svc = rt.svc
        me = svc.presence.attach("cli")
        orch = rt.orchestrator()
        reply = await orch.handle("Analyze this project.", cwd=str(project))
        assert reply.intent == IntentKind.ANALYZE_PROJECT and reply.task_id
        svc.presence.detach(me.client_id)                      # the window closes; the task keeps going
        done = await svc.pool.wait_for(reply.task_id)
        assert done.status == S.COMPLETED and done.request == "Analyze this project."
        assert done.result == "REPORT: proj is a small Python app with no tests."
        scan = done.plan[0].result["data"]
        assert scan["languages"]["Python"] == 1 and scan["code_lines"]["Python"] == 10
        await svc.bus.drain()
        assert any(done.id == n.task_id for n in svc.notifications.pending())     # waiting for the user
        back = svc.presence.attach("cli")
        assert back.first and back.away_since is not None
        answer = await orch.handle("What happened while I was away?")
        assert answer.intent == IntentKind.AWAY
        assert "Analyze proj — completed" in answer.text
        assert "REPORT: proj is a small Python app with no tests." in answer.text
        assert "kept running the whole time" in answer.text
        assert not any(done.id == n.task_id for n in svc.notifications.pending())   # reported, so acknowledged
    finally:
        await rt.stop()


async def test_the_away_report_shows_when_jarvis_was_not_running(tmp_path):
    rt, _ = make_runtime(str(tmp_path))
    await rt.start()
    try:
        now = rt.svc.clock.now()
        db = rt.svc.db
        db.execute("INSERT INTO runtime_runs(id, pid, mode, started_at, heartbeat_at, stopped_at, clean) "
                   "VALUES('r1', 1, 'daemon', ?, ?, ?, 1)", (now - 5000, now - 3000, now - 3000))
        db.execute("INSERT INTO runtime_runs(id, pid, mode, started_at, heartbeat_at, clean) "
                   "VALUES('r2', 2, 'daemon', ?, ?, 0)", (now - 2500, now - 2000))
        db.execute("UPDATE runtime_runs SET started_at=? WHERE id=?", (now - 60, rt.run_id))
        report = away_report(rt.svc, since=now - 3600, until=now)
        assert not report.kept_running
        assert [round(now - g["from"]) for g in report.gaps] == [3000, 2000]
        assert report.unclean and report.unclean[0]["run"] == "r2"
        text = report.text()
        assert "stopped unexpectedly" in text and "was not running from" in text
    finally:
        await rt.stop()


async def test_conversation_continues_after_a_restart(tmp_path):
    rt, sim = make_runtime(str(tmp_path))
    await rt.start()
    await rt.orchestrator().handle("Remember that the launch code is 1234")
    await rt.stop()
    rt2, _ = make_runtime(str(tmp_path), sim=sim)
    await rt2.start()
    try:
        history = rt2.orchestrator().history
        assert history and history[0].content == "Remember that the launch code is 1234"
    finally:
        await rt2.stop()


def test_away_and_analysis_requests_are_understood():
    for text in ("What happened while I was away?", "what did I miss", "What did you do while I was away?",
                 "I'm back"):
        assert parse(text).kind == IntentKind.AWAY, text
    for text in ("Analyze this project.", "review this codebase", "analyse the jarvis project"):
        assert parse(text).kind == IntentKind.ANALYZE_PROJECT, text
    assert parse("Summarize the architecture of this project.").kind == IntentKind.CHAT


# -- models ------------------------------------------------------------------------------------------------

async def test_a_task_waits_for_a_model_and_resumes_when_one_returns(tmp_path):
    rt, sim = make_runtime(str(tmp_path))
    sim.provider.when("Summarize the scan", "SUMMARY: all good.")
    await rt.start()
    try:
        sim.provider.online = False
        await rt.svc.router.refresh()
        task = rt.svc.tasks.create_task("summarize", steps=[
            Step("scan", "project_scan", {"path": str(tmp_path)}),
            Step("write it up", "model_report", {"instruction": "Summarize the scan", "material": {"$from_step": 0}})],
            created_by="user:owner")
        waiting = await rt.svc.pool.wait_for(task.id, [S.WAITING])
        assert waiting.checkpoint.get("waiting_for") == "model" and "language model" in waiting.status_reason
        assert waiting.plan[0].status == StepStatus.DONE             # the finished step is kept
        assert rt.svc.router.status()["providers"] == {"simulated": False}
        sim.provider.online = True                                    # Ollama comes back
        done = await rt.svc.pool.wait_for(task.id, timeout=10)
        assert done.status == S.COMPLETED and done.result == "SUMMARY: all good."
        assert [e.tool for e in rt.svc.audit.query(task_id=task.id, action="tool_execute")].count("project_scan") == 1
    finally:
        await rt.stop()


async def test_failed_requests_and_fallbacks_are_recorded(tmp_path):
    rt, sim = make_runtime(str(tmp_path))
    await rt.start()
    try:
        sim.provider.failing_models = {"sim-chat:8b"}
        from jarvis.models.base import ChatMessage
        from jarvis.models.router import TaskProfile
        # high complexity prefers the larger model first, which fails; the router falls back
        routed = await rt.svc.router.chat(TaskProfile(complexity="high"), [ChatMessage("user", "hi")])
        await rt.svc.bus.drain()
        status = rt.svc.router.status()
        assert routed.used_fallback and routed.response.model == "sim-small:3b"
        assert status["last_fallback"]["intended"] == "sim-chat:8b" and status["last_fallback"]["used"] == "sim-small:3b"
        assert rt.svc.events.query(types=["MODEL_REQUEST_FAILED"])[0].payload["model"] == "sim-chat:8b"
        assert status["last_failure"]["model"] == "sim-chat:8b"
    finally:
        await rt.stop()


async def test_the_router_reports_the_active_request(tmp_path):
    rt, sim = make_runtime(str(tmp_path))
    await rt.start()
    try:
        seen = []
        original = sim.provider.chat

        async def slow_chat(*args, **kwargs):
            seen.append(list(rt.svc.router.status()["active_requests"]))
            return await original(*args, **kwargs)

        sim.provider.chat = slow_chat
        await rt.orchestrator().handle("Tell me a joke about disks")
        assert seen and seen[0][0]["model"] == "sim-chat:8b" and seen[0][0]["purpose"] == "conversation"
        assert rt.svc.router.status()["active_requests"] == []
        assert rt.svc.router.status()["last_success"]["model"] == "sim-chat:8b"
    finally:
        await rt.stop()


# -- resources ---------------------------------------------------------------------------------------------

def _pressure_engine(tmp_path, policy):
    engine = build_engine(str(tmp_path))
    engine.resources = ResourceManager(engine.state, low_priority_policy=policy)
    engine.pool.resources = engine.resources
    engine.permissions.grant("*", PermissionLevel.EXECUTE_CONSEQUENTIAL, tools=["shell_execute"])
    return engine


async def test_policy_wait_lets_the_step_in_flight_finish_before_pausing(tmp_path):
    engine = _pressure_engine(tmp_path, "wait")
    await engine.pool.start()
    task = engine.tasks.create_task("index", priority=Priority.P3, created_by="automation:x",
                                    authority={"interactive": False},
                                    steps=[Step("slow", "shell_execute", {"command": "sleep 0.6"}),
                                           Step("next", "time_now", {})])
    await engine.pool.wait_for(task.id, [S.RUNNING])
    engine.state.set("resources.memory_percent", 97.0, ttl=60)
    await asyncio.sleep(0.2)
    assert engine.tasks.get_task(task.id).status == S.RUNNING         # not cut short
    paused = await engine.pool.wait_for(task.id, [S.PAUSED], timeout=5)
    assert paused.plan[0].status == StepStatus.DONE and paused.plan[1].status == StepStatus.PENDING
    engine.state.set("resources.memory_percent", 40.0, ttl=60)
    done = await engine.pool.wait_for(task.id, timeout=5)
    assert done.status == S.COMPLETED
    await engine.pool.stop()


async def test_policy_continue_and_slow(tmp_path):
    engine = _pressure_engine(tmp_path, "continue")
    engine.state.set("resources.memory_percent", 97.0, ttl=60)
    await engine.pool.start()
    task = engine.tasks.create_task("opportunistic", priority=Priority.P4, steps=[Step("t", "time_now", {})])
    assert (await engine.pool.wait_for(task.id)).status == S.COMPLETED
    await engine.pool.stop()

    slow = _pressure_engine(tmp_path / "b", "slow")
    (tmp_path / "b").mkdir()
    await slow.pool.start()
    jobs = [slow.tasks.create_task(f"job {i}", priority=Priority.P3, created_by="automation:x",
                                   authority={"interactive": False},
                                   steps=[Step("s", "shell_execute", {"command": "sleep 5"})]) for i in range(2)]
    for job in jobs:
        await slow.pool.wait_for(job.id, [S.RUNNING])
    slow.state.set("resources.memory_percent", 97.0, ttl=60)
    await wait_until(lambda: sum(slow.tasks.get_task(j.id).status == S.PAUSED for j in jobs) == 1, timeout=5)
    assert sum(slow.tasks.get_task(j.id).status == S.RUNNING for j in jobs) == 1
    for job in jobs:
        slow.tasks.cancel_task(job.id)
    await slow.pool.stop()


# -- briefing ------------------------------------------------------------------------------------------------

async def test_the_briefing_pipeline_stores_and_announces_data(tmp_path):
    rt, _ = make_runtime(str(tmp_path), mode="daemon", briefing={"enabled": True, "time": "07:30"})
    await rt.start()
    try:
        schedule = rt.svc.automations.find("Morning briefing", owner="system")
        assert schedule and schedule.schedule == {"type": "daily", "at": "07:30"} and schedule.next_run
        data = prepare_briefing(rt.svc, briefing_id="brief:test")
        prepare_briefing(rt.svc, briefing_id="brief:test")              # idempotent per id
        await rt.svc.bus.drain()
        stored = latest_briefing(rt.svc)
        assert stored["id"] == "brief:test" and stored["text"].startswith("It's ")
        assert data["health"]["overall"] == "healthy" and "headline" in data
        assert [e.payload["id"] for e in rt.svc.events.query(types=["BRIEFING_READY"])] == ["brief:test"]
        assert any(n.title == "Your briefing is ready" for n in rt.svc.notifications.pending())
    finally:
        await rt.stop()


async def test_an_unseen_result_is_reported_even_if_it_finished_before_the_away_window(tmp_path):
    """Seen on a real PC: the analysis finished, then the runtime restarted while an interface was attached, so
    the absence was measured from the restart and "what happened while I was away?" left the result out."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "app.py").write_text("print(1)\n")
    rt, sim = make_runtime(str(tmp_path), mode="daemon")
    sim.provider.when("Analyze the software project", "REPORT: one Python file.")
    await rt.start()
    reply = await rt.orchestrator().handle("Analyze this project.", cwd=str(project))
    await rt.svc.pool.wait_for(reply.task_id)
    me = rt.svc.presence.attach("cli")               # the user is here when the runtime goes down
    await rt.stop()
    rt2, _ = make_runtime(str(tmp_path), mode="daemon", sim=sim)
    await rt2.start()
    try:
        await asyncio.sleep(0.05)
        back = rt2.svc.presence.attach("cli")
        assert back.away_since is not None and back.away_since > rt2.svc.tasks.get_task(reply.task_id).finished_at
        answer = await rt2.orchestrator().handle("What happened while I was away?")
        assert "Analyze proj — completed" in answer.text and "REPORT: one Python file." in answer.text
        again = await rt2.orchestrator().handle("What happened while I was away?")
        assert "REPORT: one Python file." not in again.text          # reported once, not forever
        assert me.client_id
    finally:
        await rt2.stop()


# -- found on a real PC after the first Phase 2 build ---------------------------------------------------------

async def test_a_jarvis_command_typed_into_the_chat_is_explained_not_run(tmp_path):
    from jarvis.models.base import ToolCall
    rt, sim = make_runtime(str(tmp_path), mode="daemon")
    await rt.start()
    try:
        svc = rt.svc
        reply = await rt.orchestrator().handle("py -m jarvis runtime stop")
        assert reply.intent == IntentKind.CLI_COMMAND and "Command Prompt" in reply.text
        assert "Closing this window doesn't stop me" in reply.text
        assert not svc.tasks.list_tasks(limit=10) and not svc.audit.query(action="tool_execute")
        # and if the model tries to run one itself, the shell tool refuses without asking for approval
        sim.provider.when("stop yourself", tool_calls=[ToolCall("shell_execute",
                                                                {"command": "py -m jarvis runtime stop"})], once=True)
        sim.provider.when("controls JARVIS itself", "I can't stop myself from here.")
        await rt.orchestrator().handle("please stop yourself")
        runs = [e for e in svc.audit.query(limit=20) if e.tool == "shell_execute"]
        assert runs and not runs[0].ok and "controls JARVIS itself" in runs[0].summary
        assert not svc.approvals.pending() and rt.started
    finally:
        await rt.stop()


async def test_a_safety_blocked_step_fails_the_task_instead_of_waiting_forever(tmp_path):
    engine = build_engine(str(tmp_path))
    await engine.pool.start()
    engine.permissions.grant("*", PermissionLevel.AUTONOMOUS, tools=["shell_execute"])     # even with a grant
    task = engine.tasks.create_task("wipe", steps=[Step("wipe", "shell_execute", {"command": "rm -rf /"})])
    done = await engine.pool.wait_for(task.id)
    assert done.status == S.FAILED and "safety policy" in done.status_reason
    await engine.pool.stop()


async def test_completion_notifications_show_the_result_not_the_verification(tmp_path):
    rt, sim = make_runtime(str(tmp_path), mode="daemon")
    sim.provider.when("Summarize the scan", "The folder holds one small Python script and nothing else.")
    await rt.start()
    try:
        task = rt.svc.tasks.create_task("summarize", created_by="user:owner", steps=[
            Step("scan", "project_scan", {"path": str(tmp_path)}),
            Step("write it up", "model_report", {"instruction": "Summarize the scan", "material": {"$from_step": 0}})])
        done = await rt.svc.pool.wait_for(task.id)
        await rt.svc.bus.drain()
        assert done.status_reason == "all steps completed and verified"
        note = next(n for n in rt.svc.notifications.pending() if n.task_id == task.id)
        assert note.text() == "Summarize finished. The folder holds one small Python script and nothing else."
    finally:
        await rt.stop()


def test_recovery_messages_say_continue_once():
    from jarvis.runtime import StartupReport
    from jarvis.tasks.manager import RecoveryReport
    summary = "job was interrupted during 'x'. I haven't resumed it: it's unknown. Say 'continue' to resume."
    greeting = StartupReport(recovered=[RecoveryReport("t1", "job", summary, False, "paused", True)]).greeting()
    assert greeting.count("Say 'continue' to resume.") == 1


def test_control_characters_never_reach_jarvis():
    from jarvis.cli import _clean_input
    assert _clean_input("\x01") == "" and _clean_input(" what's up?\x01 ") == "what's up?"


async def test_the_model_is_told_which_operating_system_it_works_on(tmp_path, monkeypatch):
    from jarvis.core.context import environment_note
    rt, sim = make_runtime(str(tmp_path))
    await rt.start()
    try:
        await rt.orchestrator().handle("What's using the most memory right now?")
        system = sim.provider.calls[0]["messages"][0].content
        assert "COMPUTER: " in system and "system_info and process_list" in system
    finally:
        await rt.stop()
    monkeypatch.setattr(sys, "platform", "win32")
    note = environment_note()
    assert "cmd.exe" in note and "tasklist" in note and "not Unix ones" in note


async def test_the_away_answer_shows_the_newest_results_and_labels_the_catch_up(tmp_path):
    rt, _ = make_runtime(str(tmp_path), mode="daemon")
    await rt.start()
    try:
        svc = rt.svc
        ids = []
        for i in range(5):
            t = svc.tasks.create_task(f"job {i}", created_by="user:owner", steps=[Step("t", "time_now", {})])
            done = await svc.pool.wait_for(t.id)
            done.result = f"RESULT {i}"
            svc.tasks.save(done)
            ids.append(done.id)
        report = away_report(svc, since=svc.clock.now() + 1, until=svc.clock.now() + 60)   # all finished earlier
        text = report.text()
        assert "Finished earlier, not reported to you until now:" in text
        assert "RESULT 4" in text and "RESULT 3" in text and "RESULT 2" in text          # the newest three in full
        assert "RESULT 0" not in text and f"(full result: /task {ids[0]})" in text
    finally:
        await rt.stop()


def test_reports_are_plain_text_for_the_terminal():
    from jarvis.tools.internal import plain_text
    assert plain_text("**Overview**\n## Next\n* a **b** c\n- d\n2*3 and a_b") == "Overview\nNext\n• a b c\n• d\n2*3 and a_b"


async def test_queued_news_reaches_an_open_window_once_the_user_is_idle(tmp_path):
    rt, _ = make_runtime(str(tmp_path), mode="daemon")
    await rt.start()
    try:
        svc = rt.svc
        shown = []
        svc.notifications.sinks.append(shown.append)
        svc.presence.attach("cli")
        n = svc.notifications.notify(NotificationPriority.IMPORTANT, "Hello from JARVIS", source="test")
        assert n.state == "queued" and not shown          # below the interrupt threshold: not pushed straight away
        await wait_until(lambda: shown and shown[0].id == n.id, timeout=5)   # the heartbeat delivers it
        assert n.state == "delivered"
    finally:
        await rt.stop()
