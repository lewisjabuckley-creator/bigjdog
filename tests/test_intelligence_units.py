"""Phase 3 building blocks: goal understanding, plans as data, failure classification, loop protection,
analysis and independent verification, context packing, inference scheduling and routing. No runtime needed."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

from jarvis.clock import FakeClock
from jarvis.config import IntelligenceConfig
from jarvis.intelligence.analysis import (analyze_disk, analyze_performance, analyze_research, parse_process_table,
                                          verify_backup, verify_citations, verify_performance)
from jarvis.intelligence.context import ContextPacker
from jarvis.intelligence.failures import FailureCategory, classify
from jarvis.intelligence.goals import (AmbiguityClass, Complexity, Constraint, ExecutionMode, GoalParser,
                                       PlanPriority)
from jarvis.intelligence.guard import LoopGuard
from jarvis.intelligence.plans import (InvalidPlanTransition, NodeKind, NodeStatus, Plan, PlanNode, PlanStatus,
                                       Quality, check_plan_transition)
from jarvis.intelligence.quality import plan_quality
from jarvis.intelligence.store import PlanStore
from jarvis.models.base import Capability, ModelInfo, Purpose
from jarvis.models.router import ModelRouter, TaskProfile
from jarvis.models.scheduler import InferenceScheduler
from jarvis.tasks.models import Step, StepStatus, Task, TaskStatus

parser = GoalParser(FakeClock())


# -- goals --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,kind,mode,complexity", [
    ("Check my CPU temperature", "generic", ExecutionMode.EXECUTE, Complexity.SIMPLE),
    ("tell me about disks", "generic", ExecutionMode.EXECUTE, Complexity.SIMPLE),
    ("My computer is slow, find out why and fix it", "performance", ExecutionMode.EXECUTE, Complexity.COMPLEX),
    ("why is my laptop so slow", "performance", ExecutionMode.EXECUTE, Complexity.MODERATE),
    ("What should I do about my computer being slow?", "performance", ExecutionMode.ADVISE, Complexity.MODERATE),
    ("dry run: free up disk space", "disk_cleanup", ExecutionMode.DRY_RUN, Complexity.MODERATE),
    ("what would happen if I stopped chrome?", "generic", ExecutionMode.SIMULATE, Complexity.SIMPLE),
    ("how long will the tests take?", "generic", ExecutionMode.PREDICT, Complexity.SIMPLE),
    ("Research what my notes say about the robot arm", "research", ExecutionMode.EXECUTE, Complexity.COMPLEX),
    ("Run the tests, and if they pass build it and then tell me", "compound", ExecutionMode.EXECUTE,
     Complexity.COMPLEX),
])
def test_goal_kind_mode_and_complexity(text, kind, mode, complexity):
    goal = parser.parse(text)
    assert (goal.kind, goal.mode, goal.complexity) == (kind, mode, complexity)


def test_constraints_priority_and_deadline_are_understood():
    goal = parser.parse("urgent: my computer is really slow, speed it up, but don't close Chrome and don't delete "
                        "anything")
    assert goal.priority == PlanPriority.HIGH and goal.wants_fix
    kinds = {c.kind: c.value for c in goal.constraints}
    assert kinds["protect_process"].lower() == "chrome" and "no_delete" in kinds
    assert goal.forbids("process_stop", {"pid": 1}, {"target": "chrome.exe"}, level=4)
    assert goal.forbids("file_delete", {"path": "/tmp/x"}, level=4)
    assert goal.forbids("process_stop", {"pid": 1}, {"target": "spotify.exe"}, level=4) is None
    backup = parser.parse("Back up ~/Documents to /mnt/backup by 5pm")
    assert backup.target == "~/Documents -> /mnt/backup" and backup.deadline is not None
    assert parser.parse("clean up my downloads in the background").priority == PlanPriority.BACKGROUND
    assert parser.parse("just look at why the pc is slow").mode == ExecutionMode.ADVISE     # read-only constraint


def test_compound_requests_become_a_dependency_graph_with_conditions():
    goal = parser.parse("Run the tests, and if they pass build it and then tell me")
    parts = [(s.text, s.after, s.condition) for s in goal.subgoals]
    assert parts == [("Run the tests", [], None), ("build it", [0], "success"), ("tell me", [1], None)]
    parallel = parser.parse("check disk space and clean my downloads folder then empty the recycle bin")
    assert [s.after for s in parallel.subgoals] == [[], [], [0, 1]]          # two in parallel, then one


def test_ambiguity_is_classified_by_what_a_wrong_guess_would_cost():
    assert parser.parse("delete the old files").ambiguity.klass == AmbiguityClass.CONSEQUENTIAL
    assert parser.parse("wipe everything").ambiguity.klass == AmbiguityClass.DANGEROUS
    assert parser.parse("kill it").ambiguity.question == "Which program should I kill?"
    backup = parser.parse("back up my photos")
    assert backup.ambiguity.must_ask and "Where" in backup.ambiguity.question
    cleanup = parser.parse("free up disk space")
    assert cleanup.ambiguity.klass == AmbiguityClass.RECOVERABLE and not cleanup.ambiguity.must_ask
    assert parser.parse("My computer is slow, fix it").ambiguity is None


def test_constraints_rule_out_actions():
    assert Constraint("read_only").forbids("file_write", {"path": "x"}, level=3)
    assert Constraint("read_only").forbids("file_read", {"path": "x"}, level=0) is None
    assert Constraint("no_restart").forbids("shell_execute", {"command": "shutdown /r /t 0"})
    assert Constraint("no_network").forbids("web", {}, requires_network=True)
    assert Constraint("protect_path", "/data").forbids("file_delete", {"path": "/data/a.txt"}, level=4)
    assert Constraint("only_path", "/work").forbids("file_write", {"path": "/etc/x"}, level=3)


# -- plans ----------------------------------------------------------------------------------------------------------

def _plan(*nodes: PlanNode) -> Plan:
    return Plan(parser.parse("do things step by step"), "test", list(nodes))


def test_plan_graph_cycles_waves_and_readiness():
    a, b, c = PlanNode("a", "A", NodeKind.GATHER), PlanNode("b", "B", NodeKind.GATHER), \
        PlanNode("c", "C", NodeKind.REPORT, depends_on=["a", "b"])
    plan = _plan(a, b, c)
    assert plan.problems() == []
    assert [[n.id for n in w] for w in plan.waves()] == [["a", "b"], ["c"]]          # a and b in parallel
    assert [n.id for n in plan.ready_nodes()] == ["a", "b"]
    a.depends_on = ["c"]
    assert "circular dependency" in " ".join(plan.problems())
    assert LoopGuard().cycle(plan) is not None
    bad = _plan(PlanNode("x", "X", NodeKind.GATHER, depends_on=["missing"]))
    assert "unknown node 'missing'" in bad.problems()[0]


def test_conditions_are_evaluated_against_facts_and_outcomes():
    tests = PlanNode("tests", "Run the tests", NodeKind.ACTION, status=NodeStatus.DONE)
    plan = _plan(tests, PlanNode("build", "Build", NodeKind.ACTION, depends_on=["tests"]))
    plan.facts["tests"] = {"ok": False}
    assert plan.evaluate({"node": "tests", "outcome": "success"}) is False      # the tests ran but failed
    assert plan.evaluate({"node": "tests", "outcome": "failure"}) is True
    plan.facts["decide"] = {"actions": ["act1"]}
    assert plan.evaluate({"fact": "decide.actions", "op": "nonempty"}) is True
    assert plan.evaluate({"all": [{"fact": "decide.actions", "op": "nonempty"}, {"node": "build"}]}) is None
    assert plan.evaluate({"not": {"fact": "decide.missing", "op": "true"}}) is True


def test_plan_lifecycle_transitions_are_enforced():
    check_plan_transition(PlanStatus.CREATED, PlanStatus.VALIDATING)
    check_plan_transition(PlanStatus.RUNNING, PlanStatus.REPLANNING)
    with pytest.raises(InvalidPlanTransition):
        check_plan_transition(PlanStatus.COMPLETED, PlanStatus.RUNNING)
    with pytest.raises(InvalidPlanTransition):
        check_plan_transition(PlanStatus.CREATED, PlanStatus.RUNNING)       # never skip validation


def test_plans_goals_and_revisions_persist(db):
    store = PlanStore(db, FakeClock())
    plan = _plan(PlanNode("a", "A", NodeKind.GATHER), PlanNode("b", "B", NodeKind.REPORT, depends_on=["a"]))
    plan.facts["a"] = {"ok": True, "system": {"cpu_percent": 50}}
    store.save_goal(plan.goal)
    store.save(plan)
    store.record_revision(plan, "failure", "a failed")
    again = store.get(plan.id)
    assert again.to_dict() == plan.to_dict()
    assert store.get_goal(plan.goal.id).text == "do things step by step"
    assert store.revisions(plan.id)[0]["trigger"] == "failure"
    assert store.find("test")[0].id == plan.id


def test_plan_quality_comes_from_independent_verification():
    act = PlanNode("act", "Stop X", NodeKind.ACTION, status=NodeStatus.DONE, quality=Quality.UNVERIFIED)
    verify = PlanNode("verify", "Verify", NodeKind.VERIFY, depends_on=["act"], status=NodeStatus.DONE,
                      quality=Quality.VERIFIED)
    plan = _plan(act, verify)
    assert plan_quality(plan) == Quality.VERIFIED
    verify.quality = Quality.CONFLICTING
    assert plan_quality(plan) == Quality.CONFLICTING
    alone = _plan(PlanNode("act", "Write", NodeKind.ACTION, status=NodeStatus.DONE, quality=Quality.UNVERIFIED))
    assert plan_quality(alone) == Quality.UNVERIFIED                     # nobody checked it: never "verified"
    assert plan_quality(_plan(PlanNode("g", "Look", NodeKind.GATHER, status=NodeStatus.DONE))) is None


# -- failures and loop protection --------------------------------------------------------------------------------

def _failed_task(error: str, status: TaskStatus = TaskStatus.FAILED, **step: object) -> Task:
    s = Step("do it", "shell_execute", status=StepStatus.FAILED, error=error, **step)
    return Task("x", status=status, status_reason=error, plan=[s])


@pytest.mark.parametrize("error,category", [
    ("`x` timed out", FailureCategory.TRANSIENT),
    ("not authorized (requires execute consequential authorization)", FailureCategory.PERMISSION),
    ("No space left on device", FailureCategory.RESOURCE),
    ("'foo' is not recognized as an internal or external command", FailureCategory.TOOL),
    ("/tmp/x does not exist", FailureCategory.DATA),
    ("invalid arguments: missing required argument(s): path", FailureCategory.PLANNING),
    ("something odd", FailureCategory.UNKNOWN),
])
def test_failures_are_classified_before_any_retry(error, category):
    assert classify(_failed_task(error)).category == category


def test_an_unknown_outcome_is_never_classified_as_retryable():
    task = _failed_task("interrupted", status=TaskStatus.PAUSED, outcome_unknown=True)
    task.plan[0].status = StepStatus.PENDING
    assert classify(task).category == FailureCategory.UNKNOWN
    assert classify(None, verification_failed=True).category == FailureCategory.VERIFICATION


def test_loop_guard_limits_attempts_replans_identical_failures_and_growth():
    guard = LoopGuard(IntelligenceConfig(max_node_attempts=2, max_replans=1, max_identical_failures=1, max_nodes=3))
    plan = _plan(PlanNode("a", "A", NodeKind.GATHER, attempts=2))
    assert guard.node_attempt(plan, plan.nodes[0]).name == "node_attempts"
    plan.replans.append({"version": 1})
    assert guard.replan(plan).name == "replans"
    failure = classify(_failed_task("`x` timed out"))
    assert guard.failure(plan, failure) is None
    assert guard.failure(plan, failure).name == "identical_failures"     # the same failure again: stop
    assert guard.nodes(plan, extra=3).name == "nodes"
    loop = PlanNode("t", "Tests", NodeKind.ACTION, loop={"max": 2}, iterations=2)
    assert guard.loop_iteration(loop).name == "loop_iterations"


# -- analysis and verification ---------------------------------------------------------------------------------

def test_performance_analysis_finds_the_hog_and_never_proposes_protected_processes():
    evidence = {"system": {"cpu_percent": 97, "memory_percent": 40, "cpu_count": 4},
                "by_cpu": {"processes": [{"pid": 4242, "name": "miner.exe", "cpu_percent": 180.0, "memory_percent": 2},
                                         {"pid": 5, "name": "svchost.exe", "cpu_percent": 90.0, "memory_percent": 1},
                                         {"pid": os.getpid(), "name": "python", "cpu_percent": 95.0}]}}
    result = analyze_performance(evidence, {"constraints": [], "history": []})
    usable = [c for c in result["candidates"] if not c["blocked"]]
    assert result["bottlenecks"] == ["cpu"] and usable[0]["args"] == {"pid": 4242}
    blocked = {c["meta"]["target"]: c["blocked"] for c in result["candidates"] if c["blocked"]}
    assert "operating system" in blocked["svchost.exe"] and "JARVIS itself" in blocked["python"]
    # the user's constraint and an approach that failed before both count
    constrained = analyze_performance(evidence, {"constraints": [{"kind": "protect_process", "value": "miner"}],
                                                 "history": []})
    assert all(c["blocked"] for c in constrained["candidates"] if c["meta"].get("target") == "miner.exe")
    remembered = analyze_performance(evidence, {"history": [{"key": "process_stop:miner.exe", "outcome": "no_effect",
                                                             "when": "on Tuesday"}]})
    miner = next(c for c in remembered["candidates"] if c["meta"].get("target") == "miner.exe")
    assert miner["confidence"] < usable[0]["confidence"] and "already tried" in miner["history"]


def test_nothing_overloaded_is_reported_honestly():
    result = analyze_performance({"system": {"cpu_percent": 12, "memory_percent": 30}, "by_cpu": {"processes": []}}, {})
    assert result["bottlenecks"] == [] and "nothing is overloaded" in result["cause"]
    assert result["confidence"] <= 0.3


def test_fallback_process_tables_are_parsed():
    ps = "  PID %CPU %MEM COMMAND\n 4242 180.0  2.0 /usr/bin/miner\n   12  0.1  0.3 sshd\n"
    assert parse_process_table(ps)[0] == {"pid": 4242, "name": "miner", "cpu_percent": 180.0, "memory_percent": 2.0}
    tasklist = '"chrome.exe","1234","Console","1","250,000 K"\n'
    assert parse_process_table(tasklist)[0]["pid"] == 1234


def test_performance_verification_states():
    targets = [{"node": "act", "tool": "process_stop", "pid": 999_999, "target": "miner.exe"}]
    ok = verify_performance({"before": {"cpu_percent": 97}, "after": {"cpu_percent": 30}, "targets": targets})
    assert ok["quality"] == Quality.VERIFIED.value
    not_eased = verify_performance({"before": {"cpu_percent": 97}, "after": {"cpu_percent": 96}, "targets": targets})
    assert not_eased["quality"] == Quality.PARTIALLY_VERIFIED.value and not not_eased.get("resolved")
    me = [{"node": "act", "tool": "process_stop", "pid": os.getpid(), "target": "python"}]
    conflicting = verify_performance({"before": {"cpu_percent": 97}, "after": {"cpu_percent": 30}, "targets": me,
                                      "claimed": {"act": {"ok": True}}})
    assert conflicting["quality"] == Quality.CONFLICTING.value           # reported done, still running
    failed = verify_performance({"before": {"cpu_percent": 97}, "after": {"cpu_percent": 97}, "targets": me})
    assert failed["quality"] == Quality.FAILED.value


def test_disk_analysis_and_backup_verification(tmp_path):
    big = tmp_path / "Downloads" / "old.iso"
    big.parent.mkdir()
    big.write_bytes(b"x" * 2048)
    usage = {"path": str(tmp_path), "percent": 95.0, "free": 1000, "children": [{"path": str(big.parent), "size": 2048}],
             "largest_files": [{"path": str(big), "size": 2048, "age_days": 400}]}
    result = analyze_disk({"usage": usage}, {"constraints": [{"kind": "no_delete"}]})
    assert result["candidates"][0]["blocked"] == "you asked me not to delete anything"
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "a").mkdir(parents=True)
    (src / "a" / "one.txt").write_text("one")
    assert verify_backup({"source": str(src), "destination": str(dst)})["quality"] == Quality.FAILED.value
    (dst / "a").mkdir(parents=True)
    (dst / "a" / "one.txt").write_text("one")
    assert verify_backup({"source": str(src), "destination": str(dst)})["quality"] == Quality.VERIFIED.value


def test_research_claims_are_cross_checked_and_citations_verified(tmp_path):
    (tmp_path / "a.md").write_text("The robot arm has 6 joints and a payload of 2 kg.\nUnrelated line.\n")
    (tmp_path / "b.md").write_text("Our robot arm has 5 joints and a payload of 2 kg.\n")
    docs = [{"path": str(tmp_path / "a.md"), "content": (tmp_path / "a.md").read_text()},
            {"path": str(tmp_path / "b.md"), "content": (tmp_path / "b.md").read_text()}]
    result = analyze_research({"documents": docs}, {"topic": "the robot arm"})
    assert result["conflicts"], "6 joints vs 5 joints must be reported as a disagreement"
    assert {c["source"] for c in result["claims"]} == {str(tmp_path / "a.md"), str(tmp_path / "b.md")}
    fabricated = {"text": "The arm is made of titanium", "quote": "made of titanium", "source": str(tmp_path / "a.md")}
    verdict = verify_citations({"claims": result["claims"] + [fabricated], "conflicts": []})
    assert verdict["quality"] == Quality.PARTIALLY_VERIFIED.value
    assert verdict["unsupported"][0]["text"] == "The arm is made of titanium"
    assert verify_citations({"claims": result["claims"], "conflicts": result["conflicts"]})["quality"] == \
        Quality.CONFLICTING.value


# -- context, scheduling, routing ---------------------------------------------------------------------------------

def test_context_packing_respects_the_budget_and_marks_truncation():
    packer = ContextPacker(max_tokens=600)
    text = packer.pack([("Goal", "fix it"), ("Evidence", "x" * 10_000), ("Memory", "later")], context_length=2048)
    assert text.startswith("Goal:\nfix it") and "[truncated]" in text and len(text) <= 600 * 4 + 50


async def test_inference_scheduler_caps_concurrency_and_serves_the_user_first():
    pressure = {"on": False}
    sched = InferenceScheduler(1, pressure=lambda: (pressure["on"], "memory"))
    order: list[str] = []
    await sched.acquire(2)                           # a background request holds the only slot

    async def request(name: str, priority: int) -> None:
        async with sched.slot(priority):
            order.append(name)

    waiting = [asyncio.create_task(request("background", 2)), asyncio.create_task(request("user", 0))]
    await asyncio.sleep(0.01)
    assert sched.status()["queued"] == 2 and order == []
    sched.release()
    await asyncio.gather(*waiting)
    assert order == ["user", "background"]           # the person waiting goes first
    pressure["on"] = True
    assert InferenceScheduler(3, pressure=lambda: (True, "memory")).limit == 1


def test_router_moves_background_work_to_a_smaller_model_under_pressure():
    from jarvis.intelligence.routing import RoutingPolicy
    router = ModelRouter([])
    router.inventory = [ModelInfo("big:70b", "p", True, frozenset({Capability.CHAT}), parameter_size="70B"),
                        ModelInfo("small:3b", "p", True, frozenset({Capability.CHAT}), parameter_size="3B")]
    router.provider_status = {"p": True}
    pressure = {"on": False}
    resources = SimpleNamespace(pressure=lambda: (pressure["on"], "memory at 96%"), gpu_pressure=lambda: (False, ""))
    policy = RoutingPolicy(resources=resources)
    background = TaskProfile(purpose=Purpose.REASONING, complexity="high", interactive=False)
    assert router.select(policy.adjust(background)).model == "big:70b"
    pressure["on"] = True
    assert router.select(policy.adjust(background)).model == "small:3b"
    interactive = TaskProfile(purpose=Purpose.REASONING, complexity="high", interactive=True)
    assert router.select(policy.adjust(interactive)).model == "big:70b"     # the user's answer isn't degraded
    profile, why = policy.profile(Purpose.PLANNING, complexity="high")
    assert profile.max_params_b == 8.0 and "resources are short" in why


def test_agents_cannot_grant_permissions(db):
    from jarvis.permissions.manager import PermissionManager
    permissions = PermissionManager(db)
    with pytest.raises(PermissionError):
        permissions.grant("agent:research", 4, created_by="agent:research")
    with pytest.raises(PermissionError):
        permissions.grant("task:t1", 4, created_by="agent:system")


def test_agent_contracts_are_declared():
    from jarvis.agents.base import BUILTIN_AGENTS
    for spec in BUILTIN_AGENTS:
        contract = spec.contract()
        assert contract["permission_ceiling"] and contract["timeout_s"] > 0 and contract["outputs"]
        assert "delegate_to_agent" not in contract["tools"]
