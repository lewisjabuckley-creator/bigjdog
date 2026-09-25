"""Phase 2: the durable scheduler — schedule types, one run per slot, catch-up and missed runs, restarts,
independence from monitoring and interfaces, and automation authority for scheduled work."""

from __future__ import annotations

from datetime import datetime

import pytest

from jarvis.automation.engine import AutomationEngine, describe_schedule, next_run_for, validate_schedule
from jarvis.clock import FakeClock
from jarvis.events.bus import EventBus
from jarvis.events.store import EventStore
from jarvis.permissions.model import PermissionLevel
from jarvis.tasks.manager import TaskManager
from jarvis.tasks.models import TaskStatus
from tests.helpers import make_runtime, wait_until

S = TaskStatus


def _local(year, month, day, hour, minute) -> float:
    return datetime(year, month, day, hour, minute).timestamp()


# -- schedule arithmetic -----------------------------------------------------------------------------------

def test_next_run_for_each_schedule_type():
    base = _local(2026, 9, 24, 10, 0)          # a Thursday
    assert next_run_for({"schedule": {"type": "once", "at": base + 60}}, base) == base + 60
    assert next_run_for({"schedule": {"type": "once", "at": base - 60}}, base) is None
    assert next_run_for({"schedule": {"type": "daily", "at": "09:30"}}, base) == _local(2026, 9, 25, 9, 30)
    assert next_run_for({"schedule": {"type": "daily", "at": "10:30"}}, base) == _local(2026, 9, 24, 10, 30)
    weekly = {"schedule": {"type": "weekly", "days": ["mon", "thu"], "at": "08:00"}}
    assert next_run_for(weekly, base) == _local(2026, 9, 28, 8, 0)      # Thursday 08:00 has passed -> Monday
    assert next_run_for(weekly, _local(2026, 9, 28, 8, 0)) == _local(2026, 10, 1, 8, 0)
    assert next_run_for({"schedule": {"type": "interval", "every_s": 3600}}, base) == base + 3600
    anchored = {"schedule": {"type": "interval", "every_s": 3600, "start": base - 5400}}
    assert next_run_for(anchored, base) == base + 1800                   # stays on the anchor's grid
    assert next_run_for({"every_s": 60}, base) == base + 60              # older spec format still works
    assert next_run_for({"daily_at": "11:00"}, base) == _local(2026, 9, 24, 11, 0)


def test_schedules_are_validated_and_described():
    assert validate_schedule({"type": "weekly", "days": ["Thursday", "mon"], "at": "7:05"}) == \
        {"type": "weekly", "at": "07:05", "days": ["mon", "thu"]}
    for bad in ({"type": "daily", "at": "25:00"}, {"type": "weekly", "days": ["funday"], "at": "08:00"},
                {"type": "interval", "every_s": 0}, {"type": "hourly"}):
        with pytest.raises(ValueError):
            validate_schedule(bad)
    assert describe_schedule({"type": "interval", "every_s": 7200}) == "every 2 hours"
    assert describe_schedule({"type": "weekly", "days": ["mon", "thu"], "at": "08:00"}) == "every Mon, Thu at 08:00"


# -- one run per slot, catch-up, missed ----------------------------------------------------------------------

def _engine(db, clock, **kw):
    bus = EventBus(EventStore(db), clock)
    tasks = TaskManager(db, bus=bus, clock=clock)
    return AutomationEngine(db, tasks, bus=bus, clock=clock, **kw), tasks, bus


async def test_each_slot_runs_exactly_once_even_across_a_crash(db):
    clock = FakeClock()
    engine, tasks, bus = _engine(db, clock)
    auto = engine.schedule("hourly sync", {"type": "interval", "every_s": 3600},
                           {"type": "task", "objective": "sync", "steps": [{"tool": "time_now"}]})
    assert await engine.tick() == 0
    clock.advance(3601)
    assert await engine.tick() == 1
    assert await engine.tick() == 0                    # same slot, second tick: nothing
    first = tasks.list_tasks()
    assert len(first) == 1 and first[0].origin == f"schedule:{auto.id}"
    # a crash after the task was created but before the run was recorded: the slot is still "due"
    slot = first[0].idempotency_key.rsplit(":", 1)[1]
    db.execute("UPDATE automations SET next_run=? WHERE id=?", (float(slot), auto.id))
    engine2, tasks2, _ = _engine(db, clock)
    assert await engine2.tick() == 1                   # the slot is handled again...
    assert len(tasks2.list_tasks()) == 1               # ...but its idempotency key returns the same task
    await bus.drain()


async def test_runs_missed_while_down_are_made_up_once_or_recorded_as_missed(db):
    clock = FakeClock()
    engine, tasks, bus = _engine(db, clock, catch_up_window_s=6 * 3600)
    missed = []
    bus.subscribe("SCHEDULE_MISSED", missed.append)
    every = engine.schedule("every 10 minutes", {"type": "interval", "every_s": 600},
                            {"type": "task", "objective": "poll", "steps": [{"tool": "time_now"}]})
    at = datetime.fromtimestamp(clock.now() + 4 * 3600).strftime("%H:%M")      # not due in the first 3 hours
    daily = engine.schedule("report", {"type": "daily", "at": at},
                            {"type": "task", "objective": "report", "steps": [{"tool": "time_now"}]})
    clock.advance(3 * 3600)             # JARVIS was down for 3 hours: 18 interval slots passed
    await engine.tick()
    polls = [t for t in tasks.list_tasks() if t.objective == "poll"]
    assert len(polls) == 1                                  # made up once, not 18 times
    assert engine.get(every.id).next_run > clock.now()
    clock.advance(3 * 86400)            # down for three days: far outside the catch-up window
    await engine.tick()
    await bus.drain()
    reports = [t for t in tasks.list_tasks() if t.objective == "report"]
    assert not reports
    record = engine.get(daily.id)
    assert record.last_status == "missed" and record.missed == 1 and record.next_run > clock.now()
    report_missed = [e for e in missed if e.payload["name"] == "report"]
    assert report_missed and "not running" in report_missed[0].payload["reason"]


async def test_catch_up_skip_and_re_enabling_never_replay_old_slots(db):
    clock = FakeClock()
    engine, tasks, _ = _engine(db, clock)
    auto = engine.schedule("strict", {"type": "interval", "every_s": 600},
                           {"type": "task", "objective": "strict", "steps": [{"tool": "time_now"}]}, catch_up="skip")
    clock.advance(1800)
    await engine.tick()
    assert not tasks.list_tasks() and engine.get(auto.id).last_status == "missed"
    engine.set_enabled(auto.id, False)
    clock.advance(7200)
    engine.set_enabled(auto.id, True)
    assert engine.get(auto.id).next_run > clock.now()


# -- in the runtime -----------------------------------------------------------------------------------------

async def test_schedules_run_without_monitoring_or_any_interface_and_survive_restarts(tmp_path):
    rt, _ = make_runtime(str(tmp_path), mode="daemon")          # monitoring disabled, nobody attached
    await rt.start()
    auto = rt.svc.automations.schedule("tick", {"type": "interval", "every_s": 1},
                                       {"type": "task", "objective": "scheduled check",
                                        "steps": [{"tool": "time_now"}]})
    await wait_until(lambda: any(t.objective == "scheduled check" and t.status == S.COMPLETED
                                 for t in rt.svc.tasks.list_tasks(limit=500)), timeout=10)
    assert rt.svc.automations.get(auto.id).last_status == "ok"
    await rt.stop()
    rt2, _ = make_runtime(str(tmp_path), mode="daemon")
    await rt2.start()
    try:
        again = rt2.svc.automations.get(auto.id)
        assert again is not None and again.enabled and again.run_count >= 1
    finally:
        await rt2.stop()


async def test_a_scheduled_job_gains_no_privileges_by_being_scheduled(tmp_path):
    target = tmp_path / "report.txt"
    rt, _ = make_runtime(str(tmp_path), mode="daemon")
    await rt.start()
    try:
        svc = rt.svc
        write = {"type": "task", "objective": "write the report",
                 "steps": [{"tool": "file_write", "args": {"path": str(target), "content": "numbers"}}]}
        auto = svc.automations.schedule("report", {"type": "interval", "every_s": 1}, write)
        await wait_until(lambda: any(t.objective == "write the report" and t.status in (S.BLOCKED, S.FAILED)
                                     for t in svc.tasks.list_tasks(limit=100)), timeout=10)
        job = next(t for t in svc.tasks.list_tasks(limit=100) if t.objective == "write the report")
        assert job.authority == {"interactive": False} and job.created_by == f"automation:{auto.id}"
        assert not target.exists()
        denied = [e for e in svc.audit.query(task_id=job.id, limit=20) if e.action == "tool_denied"]
        assert denied and denied[0].tool == "file_write"       # the same registry pipeline refused it
        # an explicit grant, scoped to one automation, is what gives that automation's work authority
        again = svc.automations.schedule("report 2", {"type": "interval", "every_s": 1}, write)
        svc.permissions.grant(f"automation:{again.id}", PermissionLevel.EXECUTE_REVERSIBLE, tools=["file_write"],
                              paths=[str(tmp_path)])
        def writes_so_far():   # the audit record lands just after the file itself
            return [e for e in svc.audit.query(action="tool_execute", limit=50) if e.tool == "file_write"]

        await wait_until(lambda: target.exists() and writes_so_far(), timeout=10)
        writes = writes_so_far()
        granted = {t.id for t in svc.tasks.list_tasks(limit=500) if t.created_by == f"automation:{again.id}"}
        assert writes and all(e.task_id in granted for e in writes) and writes[0].ok
        svc.automations.set_enabled(auto.id, False)
        svc.automations.set_enabled(again.id, False)
        await svc.pool.wait_idle(timeout=10)
        ungranted = [t for t in svc.tasks.list_tasks(limit=500) if t.created_by == f"automation:{auto.id}"]
        assert ungranted and all(t.status != S.COMPLETED for t in ungranted)   # the other automation gained nothing
    finally:
        await rt.stop()
