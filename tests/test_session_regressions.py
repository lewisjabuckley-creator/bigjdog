"""Regressions from a real Windows session (Phase 3, first use).

What went wrong, and what each test pins down:
- dry-run findings were only a one-line notification; "what are these files?" went to the model, which guessed
- "yes" to something the model asked started a deletion plan nobody offered
- "sure proceed jarvis" wasn't understood as yes, so the model tried to delete a file itself
- "cancel all deletes…" made the model claim "all delete processes have been cancelled" (nothing was)
- "cancel the free up disk space process", `cancel task "free up disk space"` and "pause free up disk space"
  each started a new disk clean-up plan instead of stopping the open one
- "cancel that process" reached the model, which called file_delete
- asking for the same thing twice started the same plan twice
- a file that disappeared before approval was still asked about, and then failed
- the approval question read "file_delete(path='…')"
- a hung runtime blocked startup with advice that couldn't help
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from jarvis.core.intent import IntentKind, parse
from jarvis.intelligence.plans import PlanStatus
from jarvis.models.base import ToolCall
from tests.helpers import wait_plan
from tests.test_planning import runtime

P = PlanStatus
MB = 1024 * 1024


@pytest.fixture
def downloads(tmp_path, monkeypatch) -> list[Path]:
    """A home folder whose Downloads holds three large files nobody has touched in months."""
    home = tmp_path / "home"
    folder = home / "Downloads"
    folder.mkdir(parents=True)
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    old = time.time() - 120 * 86400
    files = []
    for name, size in (("wctEF27.tmp", 3 * MB), ("v_clouds.rpf", 2 * MB), ("vc_redist.x64.exe", 1 * MB)):
        path = folder / name
        path.write_bytes(b"x" * size)
        os.utime(path, (old, old))
        files.append(path)
    return files


def _plans(rt: Any) -> list[Any]:
    return rt.svc.intelligence.store.list(limit=50)


def _loose_deletes(rt: Any) -> list[Any]:
    """Deletion tasks outside any plan: the model acting on its own."""
    return [t for t in rt.svc.tasks.list_tasks(limit=100) if not t.outputs.get("plan_id")
            and any(s.tool == "file_delete" for s in t.plan)]


def _eager_model(sim: Any, target: Path) -> None:
    """A model that deletes whenever it gets the chance: if control language ever reaches it, the test sees it."""
    sim.provider.when(r"cancel|pause|stop|delete|process|proceed|files",
                      tool_calls=[ToolCall("file_delete", {"path": str(target)})])


def test_the_phrases_from_the_session_parse_as_control_not_chat():
    cases = {
        "sure proceed jarvis": (IntentKind.APPROVE, None),
        "yes proceed": (IntentKind.APPROVE, None),
        "ok go ahead jarvis": (IntentKind.APPROVE, None),
        "cancel all deletes for me please jarvis": (IntentKind.STOP, "all deletes"),
        "pause free up disk space": (IntentKind.PAUSE, "free up disk space"),
        "okay jarvis what are these files": (IntentKind.PLAN_DETAILS, None),
        "what did you find?": (IntentKind.PLAN_DETAILS, None),
        "all processes": (IntentKind.STATUS, None),
    }
    for text, (kind, target) in cases.items():
        intent = parse(text)
        assert intent.kind == kind, text
        if target is not None:
            assert intent.target == target, text
    assert parse("cancel that process").kind == IntentKind.STOP
    from jarvis.core.intent import is_pronoun
    assert is_pronoun(parse("cancel that process").target)
    assert parse("yes but not chrome").kind != IntentKind.APPROVE      # a condition is not a plain yes


async def test_dry_run_details_then_an_offered_fix_asks_only_about_files_still_there(tmp_path, downloads):
    async with runtime(tmp_path) as (rt, sim):
        _eager_model(sim, downloads[2])
        o = rt.orchestrator()
        dry = await o.handle("dry run: free up disk space", cwd=str(tmp_path))
        assert not dry.text.startswith("Assuming")
        plan = await wait_plan(rt.svc.intelligence, dry.data["plan_id"], [P.COMPLETED])
        assert plan.title.startswith("Dry run:")
        assert all(f.exists() for f in downloads)                          # a dry run changes nothing

        details = await o.handle("okay jarvis what are these files")
        assert details.intent == IntentKind.PLAN_DETAILS
        for f in downloads:
            assert str(f) in details.text                                  # from the record, not a guess
        assert "Nothing was changed." in details.text and "Say 'go ahead'" in details.text
        assert not sim.provider.calls                                      # the model was never asked

        downloads[0].unlink()                                              # gone before anyone approves
        ask = await o.handle("sure proceed jarvis")
        assert ask.kind == "question" and ask.approval_id and ask.text.endswith("Proceed?")
        assert "file_delete(" not in ask.text and "trash" in ask.text
        assert downloads[0].name not in ask.text                          # not asked about a vanished file
        assert downloads[1].name in ask.text
        assert not _loose_deletes(rt)

        done = await o.handle("yes")
        fix = rt.svc.intelligence.get(done.data["plan_id"])
        fix = await wait_plan(rt.svc.intelligence, fix.id, [P.COMPLETED, P.FAILED])
        assert fix.status == P.COMPLETED, fix.status_reason
        assert not downloads[1].exists() and not downloads[2].exists()
        trash = Path(rt.svc.config.data_path) / "trash"
        assert len(list(trash.iterdir())) == 2                             # recoverable
        assert not _loose_deletes(rt) and not sim.provider.calls


async def test_yes_without_an_offer_never_starts_a_fix(tmp_path, downloads):
    async with runtime(tmp_path) as (rt, sim):
        o = rt.orchestrator()
        dry = await o.handle("dry run: free up disk space", cwd=str(tmp_path))
        await wait_plan(rt.svc.intelligence, dry.data["plan_id"], [P.COMPLETED])
        sim.provider.when("hello", "Would you like me to look at the D: drive too?")
        await o.handle("hello")                                            # the model asks something
        before = len(_plans(rt))
        await o.handle("yes")
        await asyncio.sleep(0.3)
        assert len(_plans(rt)) == before                                   # "yes" answered the model, nothing more
        assert not rt.svc.approvals.pending() and all(f.exists() for f in downloads)


async def test_cancel_and_pause_stop_the_open_plan_and_never_start_another(tmp_path, downloads):
    async with runtime(tmp_path) as (rt, sim):
        _eager_model(sim, downloads[2])
        o = rt.orchestrator()

        def open_plans() -> list[Any]:
            return [p for p in _plans(rt) if not p.terminal]

        first = await o.handle("free up disk space", cwd=str(tmp_path))
        assert first.kind == "question" and first.approval_id
        again = await o.handle("free up disk space", cwd=str(tmp_path))
        assert again.text.startswith("I'm already on that") and len(_plans(rt)) == 1

        paused = await o.handle("pause free up disk space")
        assert paused.text.startswith("Paused free up disk space"), paused.text
        assert _plans(rt)[0].status == P.PAUSED and not rt.svc.approvals.pending()
        assert len(_plans(rt)) == 1

        cancelled = await o.handle('cancel task "free up disk space"')
        assert cancelled.text.startswith("Cancelled free up disk space"), cancelled.text
        assert not open_plans() and len(_plans(rt)) == 1

        # the same again with the other phrasings from the session, each on a fresh plan
        for phrase in ("cancel the free up disk space process please jarvis", "cancel all deletes for me please jarvis"):
            started = await o.handle("free up disk space", cwd=str(tmp_path))
            assert started.approval_id
            count = len(_plans(rt))
            reply = await o.handle(phrase)
            assert reply.text.startswith("Cancelled"), (phrase, reply.text)
            assert not open_plans() and len(_plans(rt)) == count, phrase

        nothing = await o.handle("cancel that process")
        assert nothing.text == "Nothing is running, so there's nothing to cancel."
        assert not sim.provider.calls and not _loose_deletes(rt)           # the model never saw any of it
        assert all(f.exists() for f in downloads)


async def test_a_paused_plan_asks_again_when_resumed(tmp_path, downloads):
    async with runtime(tmp_path) as (rt, sim):
        o = rt.orchestrator()
        first = await o.handle("free up disk space", cwd=str(tmp_path))
        assert first.approval_id
        await o.handle("pause free up disk space")
        assert not rt.svc.approvals.pending()                              # a paused plan asks nothing
        assert (await o.handle("yes")).text == "There's nothing waiting for approval."
        resumed = await o.handle("continue")
        plan = rt.svc.intelligence.get(first.data["plan_id"])
        if resumed.kind != "question":                                     # the question comes back shortly
            plan = await wait_plan(rt.svc.intelligence, plan.id, [P.WAITING])
            await asyncio.sleep(0.3)
        pending = rt.svc.approvals.pending()
        assert len(pending) == 1 and pending[0].id != first.approval_id and "trash" in pending[0].summary
        done = await o.handle("yes")
        plan = await wait_plan(rt.svc.intelligence, plan.id, [P.COMPLETED, P.FAILED])
        assert plan.status == P.COMPLETED and not any(f.exists() for f in downloads), done.text


async def test_an_unmatched_cancel_lists_what_is_open_and_changes_nothing(tmp_path, downloads):
    async with runtime(tmp_path) as (rt, sim):
        _eager_model(sim, downloads[2])
        o = rt.orchestrator()
        await o.handle("free up disk space", cwd=str(tmp_path))
        reply = await o.handle("cancel the video render")
        assert "couldn't find anything matching 'video render'" in reply.text and "haven't cancelled" in reply.text
        assert "'Free up disk space'" in reply.text
        assert [p.status for p in _plans(rt)] == [P.WAITING] and not sim.provider.calls


async def test_stopping_a_program_still_reaches_the_model_and_finished_plans_are_named(tmp_path, downloads):
    async with runtime(tmp_path) as (rt, sim):
        o = rt.orchestrator()
        first = await o.handle("free up disk space", cwd=str(tmp_path))
        sim.provider.when("python", "Several Python processes are running; which one do you mean?")
        program = await o.handle("stop python")            # a program, not JARVIS's work: the model can help
        assert sim.provider.calls and "which one" in program.text
        assert len(_plans(rt)) == 1 and rt.svc.intelligence.get(first.data["plan_id"]).status == P.WAITING
        await o.handle("cancel free up disk space")
        calls = len(sim.provider.calls)
        late = await o.handle("cancel the free up disk space process")
        assert late.text.startswith("Free up disk space was already cancelled") and "nothing to stop" in late.text
        assert len(sim.provider.calls) == calls and len(_plans(rt)) == 1


async def test_a_claimed_action_without_a_tool_is_flagged(tmp_path):
    async with runtime(tmp_path) as (rt, sim):
        sim.provider.when("tidy up", "All delete processes have been cancelled.")
        reply = await rt.orchestrator().handle("tidy up the old stuff for me")
        assert "Nothing was actually changed" in reply.footnote
        sim.provider.when("weather", "I can't check the weather from here.")
        honest = await rt.orchestrator().handle("what's the weather like")
        assert "Nothing was actually changed" not in (honest.footnote or "")


async def test_plan_results_are_in_the_model_context(tmp_path, downloads):
    async with runtime(tmp_path) as (rt, sim):
        o = rt.orchestrator()
        dry = await o.handle("dry run: free up disk space", cwd=str(tmp_path))
        await wait_plan(rt.svc.intelligence, dry.data["plan_id"], [P.COMPLETED])
        block = o.context.plans_block()
        assert "Most recent finished plan" in block and str(downloads[0]) in block


async def test_all_processes_lists_the_work_without_the_model(tmp_path, downloads):
    async with runtime(tmp_path) as (rt, sim):
        o = rt.orchestrator()
        await o.handle("free up disk space", cwd=str(tmp_path))
        reply = await o.handle("all processes")
        assert reply.intent == IntentKind.STATUS and "Free up disk space" in reply.text
        assert not sim.provider.calls


# -- the runtime process ------------------------------------------------------------------------------------------

def test_a_hung_runtime_is_reported_as_such_and_can_be_restarted(tmp_path):
    import json

    from jarvis.cli import _stop_runtime
    from jarvis.platforms import current as current_platform
    from jarvis.service.client import RuntimeUnresponsive, connect
    data = tmp_path / "data"
    data.mkdir()
    # a process that looks like a runtime and holds its record, but never answers
    hung = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "jarvis", "runtime", "run"])
    try:
        (data / "runtime.json").write_text(json.dumps({"pid": hung.pid, "host": "127.0.0.1", "port": 9,
                                                       "version": "0.2.0"}))
        (data / "api.token").write_text("x")
        with pytest.raises(RuntimeUnresponsive) as err:
            connect(data_dir=data, auto_start=True)
        assert err.value.pid == hung.pid and "runtime restart" in str(err.value)
        assert _stop_runtime(data)
        assert current_platform().wait_gone(hung.pid, 10)
    finally:
        hung.kill()
        hung.wait()


def test_a_runtime_from_another_version_or_folder_is_noticed():
    from jarvis import __version__
    from jarvis.service.client import RuntimeInfo, mismatch, source_root
    same = RuntimeInfo(1, "127.0.0.1", 1, "t", "d", version=__version__, source=source_root())
    assert mismatch(same) == ""
    older = RuntimeInfo(1, "127.0.0.1", 1, "t", "d", version="0.2.0")
    assert "version 0.2.0" in mismatch(older) and __version__ in mismatch(older)
    elsewhere = RuntimeInfo(1, "127.0.0.1", 1, "t", "d", version=__version__, source="/somewhere/else")
    assert "/somewhere/else" in mismatch(elsewhere)


async def test_the_stall_watchdog_records_a_blocked_event_loop(tmp_path):
    from jarvis.service.daemon import Daemon
    from tests.helpers import runtime_config
    daemon = Daemon(runtime_config(str(tmp_path)), simulate=True)
    watch = asyncio.create_task(daemon._watch_for_stalls(beat_s=0.1, dump_after_s=0.5))
    await asyncio.sleep(0.15)
    time.sleep(1.0)                              # the loop is blocked: nothing else can run
    await asyncio.sleep(0.3)
    watch.cancel()
    try:
        await watch
    except asyncio.CancelledError:
        pass
    text = (Path(daemon.data_dir) / "logs" / "stall-traces.log").read_text()
    assert "event loop was blocked" in text and "Thread" in text     # a line with a timestamp, and the stacks


# -- second session: leftover plans from the day before -----------------------------------------------------------

def _count(rt: Any, *types: str) -> dict[str, int]:
    seen = {t: 0 for t in types}

    def on(e: Any) -> None:
        seen[str(e.type)] = seen.get(str(e.type), 0) + 1
    for t in types:
        rt.svc.bus.subscribe(t, on)
    return seen


def _age(rt: Any, plan_id: str, hours: float) -> None:
    """Make a plan look as if it started (and was last touched by the user) ``hours`` ago."""
    intel = rt.svc.intelligence
    plan = intel.get(plan_id)
    past = time.time() - hours * 3600
    plan.created_at = plan.started_at = past
    for h in plan.history:
        h["ts"] = past
    intel.store.save(plan)


async def test_an_unanswered_question_expires_into_one_clear_hold_not_a_stream_of_notices(tmp_path, downloads):
    async with runtime(tmp_path) as (rt, sim):
        o = rt.orchestrator()
        seen = _count(rt, "PLAN_BLOCKED", "PLAN_PAUSED", "LOOP_LIMIT_REACHED")
        first = await o.handle("free up disk space", cwd=str(tmp_path))
        assert first.approval_id
        _age(rt, first.data["plan_id"], 20)
        rt.svc.db.execute("UPDATE approvals SET expires_at=? WHERE id=?", (time.time() - 60, first.approval_id))
        for _ in range(6):                                   # the periodic check, many times over
            await rt.svc.intelligence.engine.tick()
        await rt.svc.bus.drain()
        plan = rt.svc.intelligence.get(first.data["plan_id"])
        assert plan.status == P.PAUSED and "went unanswered" in plan.status_reason and "on hold" in plan.status_reason
        assert "continue free up disk space" in plan.status_reason
        assert seen == {"PLAN_BLOCKED": 0, "PLAN_PAUSED": 1, "LOOP_LIMIT_REACHED": 0}
        held = rt.svc.db.query("SELECT count, title FROM notifications WHERE title LIKE '%on hold%'")
        assert len(held) == 1 and held[0]["count"] == 1              # one notice, once

        # asking again replaces the stalled plan with a fresh one that asks now
        again = await o.handle("free up disk space", cwd=str(tmp_path))
        assert again.text.startswith("(I've set aside an older, stalled attempt") and again.approval_id
        assert rt.svc.intelligence.get(first.data["plan_id"]).status == P.CANCELLED
        assert len([p for p in _plans(rt) if not p.terminal]) == 1


async def test_continue_after_an_expired_question_asks_again(tmp_path, downloads):
    async with runtime(tmp_path) as (rt, sim):
        o = rt.orchestrator()
        first = await o.handle("free up disk space", cwd=str(tmp_path))
        rt.svc.db.execute("UPDATE approvals SET expires_at=? WHERE id=?", (time.time() - 60, first.approval_id))
        await rt.svc.intelligence.engine.tick()
        assert rt.svc.intelligence.get(first.data["plan_id"]).status == P.PAUSED
        await o.handle("continue free up disk space")
        await wait_plan(rt.svc.intelligence, first.data["plan_id"], [P.WAITING])
        await asyncio.sleep(0.3)
        pending = rt.svc.approvals.pending()
        assert len(pending) == 1 and pending[0].id != first.approval_id


async def test_a_plan_left_on_hold_by_the_old_version_settles_once(tmp_path, downloads):
    async with runtime(tmp_path) as (rt, sim):
        o = rt.orchestrator()
        first = await o.handle("free up disk space", cwd=str(tmp_path))
        intel, svc = rt.svc.intelligence, rt.svc
        await svc.bus.drain()
        # what 0.3.0 and earlier left behind, written directly (no events): the approval step paused by the
        # planner, the plan blocked, no hold marker
        from jarvis.permissions.hierarchy import Directive, InstructionSource
        from jarvis.tasks.models import TaskStatus
        task = svc.tasks.get_task(first.task_id)
        task.status = TaskStatus.PAUSED
        task.status_reason = "the plan has been open for more than 12 hours"
        task.control = Directive(InstructionSource.DEFAULT, "pause", time.time(), "system:planner").to_dict()
        task.control_dirty = True
        svc.tasks.save(task)
        svc.approvals.cancel_for_task(task.id)
        plan = intel.get(first.data["plan_id"])
        plan.status = P.BLOCKED
        plan.status_reason = "I stopped because the plan has been open for more than 12 hours"
        intel.store.save(plan)
        _age(rt, plan.id, 20)
        seen = _count(rt, "PLAN_BLOCKED", "PLAN_WAITING", "PLAN_PAUSED")
        for _ in range(6):
            await intel.engine.tick()
        await svc.bus.drain()
        plan = intel.get(plan.id)
        assert plan.status == P.PAUSED and "continue free up disk space" in plan.status_reason
        assert seen == {"PLAN_BLOCKED": 0, "PLAN_WAITING": 0, "PLAN_PAUSED": 1}


async def test_a_safety_limit_blocks_once_and_stays_blocked_until_the_user_decides(tmp_path, downloads):
    async with runtime(tmp_path) as (rt, sim):
        o = rt.orchestrator()
        intel = rt.svc.intelligence
        seen = _count(rt, "PLAN_BLOCKED", "LOOP_LIMIT_REACHED")
        first = await o.handle("free up disk space", cwd=str(tmp_path))
        intel.engine._working_alone = lambda plan: True       # as if a step were running on its own
        _age(rt, first.data["plan_id"], 20)
        for _ in range(6):
            await intel.engine.tick()
        await rt.svc.bus.drain()
        plan = intel.get(first.data["plan_id"])
        assert plan.status == P.BLOCKED and plan.facts.get("_limit") == "duration"
        assert "working on its own" in plan.status_reason
        assert seen == {"PLAN_BLOCKED": 1, "LOOP_LIMIT_REACHED": 1}      # said once, not every few seconds
        await o.handle("continue free up disk space")
        plan = intel.get(plan.id)
        assert "_limit" not in plan.facts and plan.status != P.BLOCKED     # the user lifted it


async def test_waiting_for_the_user_is_not_counted_against_the_plan(tmp_path, downloads):
    async with runtime(tmp_path) as (rt, sim):
        o = rt.orchestrator()
        first = await o.handle("free up disk space", cwd=str(tmp_path))
        _age(rt, first.data["plan_id"], 20)                  # asked long ago; the question is still open
        for _ in range(3):
            await rt.svc.intelligence.engine.tick()
        plan = rt.svc.intelligence.get(first.data["plan_id"])
        assert plan.status == P.WAITING and rt.svc.approvals.pending()


async def test_cancel_all_processes_means_all_of_jarviss_work(tmp_path, downloads):
    async with runtime(tmp_path) as (rt, sim):
        o = rt.orchestrator()
        await o.handle("free up disk space", cwd=str(tmp_path))
        await o.handle("$ sleep 30", cwd=str(tmp_path))
        reply = await o.handle("cancel all processes")
        assert reply.text.startswith("Cancelled free up disk space and 1 task"), reply.text
        assert not [p for p in _plans(rt) if not p.terminal]
        assert not [t for t in rt.svc.tasks.open_tasks() if t.kind.value != "monitor"]
        assert (await o.handle("cancel everything")).text == "Nothing is running, so there's nothing to cancel."


async def test_copies_of_the_same_plan_are_named_once(tmp_path, downloads):
    from tests.test_planning import start
    async with runtime(tmp_path) as (rt, sim):
        for _ in range(3):
            await start(rt, "free up disk space", str(tmp_path))
        reply = await rt.orchestrator().handle("cancel free up disk space")
        assert reply.text.startswith("Cancelled free up disk space (3 copies)."), reply.text


async def test_declining_a_direct_action_ends_it_as_declined_not_completed(tmp_path, downloads):
    async with runtime(tmp_path) as (rt, sim):
        sim.provider.when("tidy", tool_calls=[ToolCall("file_delete", {"path": str(downloads[2])})], once=True)
        ask = await rt.orchestrator().handle("tidy up that installer", cwd=str(tmp_path))
        assert ask.kind == "question" and f"move {downloads[2]} to JARVIS's trash" in ask.text   # exactly which file
        task = rt.svc.tasks.get_task(ask.task_id)
        assert task.title == "move vc_redist.x64.exe to JARVIS's trash"      # short, readable
        await rt.orchestrator().handle("no")
        task = rt.svc.tasks.get_task(ask.task_id)
        assert task.status.value == "cancelled" and "declined" in task.status_reason
        assert downloads[2].exists()


async def test_files_inside_unpacked_folders_in_downloads_are_not_suggested(tmp_path, downloads):
    deep = downloads[0].parent / "x64" / "levels" / "gta5" / "cloudhats" / "v_clouds_mod.rpf"
    deep.parent.mkdir(parents=True)
    deep.write_bytes(b"x" * 5 * MB)
    old = time.time() - 500 * 86400
    os.utime(deep, (old, old))
    async with runtime(tmp_path) as (rt, sim):
        o = rt.orchestrator()
        dry = await o.handle("dry run: free up disk space", cwd=str(tmp_path))
        await wait_plan(rt.svc.intelligence, dry.data["plan_id"], [P.COMPLETED])
        details = await o.handle("what are these files")
        assert str(downloads[0]) in details.text and "v_clouds_mod.rpf" not in details.text


def test_repeats_are_counted_not_announced_again_and_identical_news_is_merged(db, clock):
    from jarvis.notifications.manager import NotificationPriority as NP
    from tests.test_memory_projects_modes import _notifier
    nm, modes, delivered = _notifier(db, clock)
    for _ in range(2):
        nm.notify(NP.URGENT, "Backup needs you", "disk missing", dedupe_key="plan-blocked:1")
        clock.advance(10)
    assert len(delivered) == 1 and delivered[0].count == 2                 # counted, shown once
    for i in range(3):
        nm.notify(NP.URGENT, "Free up disk space is on hold", "say continue", dedupe_key=f"plan-held:{i}")
    held = [n for n in delivered if n.title.startswith("Free up disk space is on hold")]
    assert len(held) == 1 and held[0].count == 3 and held[0].title.endswith("(3 of them)")
