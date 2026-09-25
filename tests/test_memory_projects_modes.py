import pytest

from jarvis.config import JarvisConfig, NotificationsConfig
from jarvis.core.modes import Mode, ModeManager
from jarvis.core.types import NetworkState, NotificationPriority as NP
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.memory.decisions import DecisionLog
from jarvis.memory.store import MemoryKind, MemoryStore
from jarvis.models.fake import ScriptedProvider
from jarvis.models.router import ModelRouter
from jarvis.notifications.manager import NotificationManager
from jarvis.projects.manager import ProjectManager, ProjectPolicy, git_branch
from jarvis.state.engine import StateEngine
from jarvis.world.model import WorldModel


# -- memory -------------------------------------------------------------------------------

async def test_remember_retrieve_forget(db, clock):
    mem = MemoryStore(db, clock=clock)
    a = await mem.remember("The robotics project uses ROS 2 Humble on a Jetson Orin", kind=MemoryKind.PROJECT,
                           project_id="p-robotics")
    await mem.remember("User prefers concise answers", kind=MemoryKind.PREFERENCE)
    await mem.remember("Backups run nightly at 02:00", kind=MemoryKind.SEMANTIC)
    hits = await mem.retrieve("what does robotics run on?", project_id="p-robotics")
    assert hits and hits[0].item.id == a.id
    prefs = await mem.retrieve("concise", kinds=[MemoryKind.PREFERENCE])
    assert prefs[0].item.kind == MemoryKind.PREFERENCE
    assert mem.forget(project_id="p-robotics") == 1
    assert not await mem.retrieve("robotics Jetson")
    with pytest.raises(ValueError):
        mem.forget()   # never forget everything without a scope


async def test_dedupe_and_suppression(db, clock):
    mem = MemoryStore(db, clock=clock)
    first = await mem.remember("the API key lives in the vault")
    again = await mem.remember("the API key lives in the vault")
    assert first.id == again.id and mem.count() == 1
    mem.suppressed = True    # "don't remember this"
    assert await mem.remember("secret plans") is None
    assert mem.count() == 1


async def test_embedding_retrieval_finds_semantic_matches(db, clock):
    provider = ScriptedProvider()
    router = ModelRouter([provider])
    await router.refresh()
    mem = MemoryStore(db, clock=clock, embedder=router.embed)
    await mem.remember("database connection pool exhausted during peak load")
    await mem.remember("weekly team meeting on thursdays")
    hits = await mem.retrieve("pool exhausted")
    assert hits[0].item.content.startswith("database") and hits[0].matched == "both"


def test_decision_history(db, clock):
    log = DecisionLog(db, clock)
    log.record("Primary datastore", "PostgreSQL", reason="we need transactional guarantees across services",
               alternatives=["MongoDB", "SQLite"], user_involvement="approved by the user", project_id="p1")
    log.record("Queue", "Redis streams", reason="low operational overhead", project_id="p1")
    found = log.search("why did we choose PostgreSQL?", project_id="p1")
    assert found[0].decision == "PostgreSQL"
    text = found[0].explain()
    assert "transactional guarantees" in text and "MongoDB" in text


# -- projects -------------------------------------------------------------------------------

def test_projects_open_and_policy(db, clock, tmp_path):
    root = tmp_path / "robotics"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/feature/arm\n")
    (root / "pyproject.toml").write_text("[project]\nname='r'\n")
    (root / "tests").mkdir()
    state = StateEngine(db, clock)
    events = []
    bus = EventBus(clock=clock)
    bus.subscribe("PROJECT_CHANGED", lambda e: events.append(e))
    pm = ProjectManager(db, state, world=WorldModel(db, clock), clock=clock, bus=bus)
    proj = pm.create("Robotics", str(root), policy=ProjectPolicy(sensitive=True, network=False))
    assert pm.find("the robotics project")[0].id == proj.id
    assert pm.for_path(str(root / "src" / "x.py")).id == proj.id
    ctx = pm.open(proj)
    assert ctx["branch"] == "feature/arm" and ctx["test_command"].endswith("pytest -q")
    assert pm.active().id == proj.id
    assert git_branch(str(root)) == "feature/arm"
    assert pm.discover(str(root)).id == proj.id


# -- modes ------------------------------------------------------------------------------------

def test_mode_policies_and_overlays(db, clock):
    state = StateEngine(db, clock)
    sensitive = {"on": False}
    modes = ModeManager(state, JarvisConfig(), clock=clock, project_sensitive=lambda: sensitive["on"])
    assert modes.current == Mode.NORMAL
    eff = modes.effective()
    assert eff.network_allowed and not eff.local_only
    modes.set_private(True)
    eff = modes.effective()
    assert not eff.network_allowed and eff.local_only and "Private mode" in eff.network_block_reason
    modes.set_private(False)
    state.set("network.state", NetworkState.OFFLINE.value)
    assert modes.effective().network_block_reason == "the network is offline"
    state.set("network.state", NetworkState.ONLINE.value)
    sensitive["on"] = True
    assert modes.effective().local_only
    previous = modes.set_mode(Mode.FOCUS)
    assert previous == Mode.NORMAL and modes.verbosity() == "short"
    assert modes.execution_policy().network_allowed


# -- notifications -------------------------------------------------------------------------------

def _notifier(db, clock, **cfg):
    state = StateEngine(db, clock)
    modes = ModeManager(state, JarvisConfig(), clock=clock)
    nm = NotificationManager(db, modes, config=NotificationsConfig(**cfg), clock=clock)
    delivered = []
    nm.sinks.append(delivered.append)
    return nm, modes, delivered


def test_interrupt_policy_by_priority_and_mode(db, clock):
    nm, modes, delivered = _notifier(db, clock)
    assert nm.notify(NP.CRITICAL, "Disk nearly full").state == "delivered"
    assert nm.notify(NP.URGENT, "Build failed").state == "delivered"
    assert nm.notify(NP.IMPORTANT, "Research finished").state == "queued"
    assert nm.notify(NP.INFORMATIONAL, "Backup complete").state == "logged"
    assert nm.notify(NP.DEBUG, "tick").state == "logged"
    nm.set_activity("typing")                     # busy user: only critical interrupts
    assert nm.notify(NP.URGENT, "Service degraded").state == "queued"
    nm.set_activity("idle")
    modes.set_mode(Mode.PRESENTATION)
    assert nm.notify(NP.URGENT, "Another failure").state == "queued"
    assert nm.notify(NP.CRITICAL, "Security incident").state == "delivered"
    modes.set_mode(Mode.NORMAL)
    modes.set_quiet(True)
    assert nm.notify(NP.URGENT, "quiet please").state == "queued"
    assert [n.title for n in delivered] == ["Disk nearly full", "Build failed", "Security incident"]
    drained = nm.drain()
    assert [n.priority for n in drained][:1] == [NP.URGENT]


def test_dedupe_counts_repeats_and_escalates(db, clock):
    nm, modes, delivered = _notifier(db, clock)
    for _ in range(3):
        n = nm.notify(NP.IMPORTANT, "api crashed", dedupe_key="crash:api")
        clock.advance(300)
    assert n.count == 3 and "3 times in 10 minutes" in n.title
    assert n.priority == NP.URGENT and n.state == "delivered"


def test_rate_limit_queues_excess_interrupts(db, clock):
    nm, modes, delivered = _notifier(db, clock, max_interrupts_per_window=2)
    states = [nm.notify(NP.URGENT, f"alert {i}").state for i in range(4)]
    assert states == ["delivered", "delivered", "queued", "queued"]
    assert nm.notify(NP.CRITICAL, "critical still gets through").state == "delivered"


async def test_event_rules_produce_useful_notifications(db, clock):
    nm, modes, delivered = _notifier(db, clock)
    bus = EventBus(clock=clock)
    nm.attach(bus)
    await bus.publish(Event(EventType.TASK_FAILED, "tasks", {"title": "deployment", "reason": "health check returned 503",
                                                              "created_by": "user"}, task_id="t1"))
    await bus.publish(Event(EventType.TASK_COMPLETED, "tasks", {"title": "nightly backup", "reason": "done",
                                                                 "created_by": "automation:backup"}))
    assert delivered[0].text() == "Deployment failed. health check returned 503"
    assert nm.history(min_priority=NP.INFORMATIONAL)[0]["title"] == "Nightly backup finished"
