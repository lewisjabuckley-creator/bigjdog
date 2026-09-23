from jarvis.core.types import Confidence, HealthStatus, Provenance, ProvenanceKind
from jarvis.events.bus import EventBus
from jarvis.state.engine import StateEngine
from jarvis.state.health import HealthRegistry
from jarvis.world.model import DEPENDS_ON, WorldModel


def test_state_facts_have_provenance_and_staleness(db, clock):
    state = StateEngine(db, clock)
    state.set("resources.cpu_percent", 82.0, provenance=Provenance(ProvenanceKind.SYSTEM_STATE, "psutil"), ttl=30)
    fact = state.get("resources.cpu_percent")
    assert fact.confidence is Confidence.OBSERVED
    assert fact.provenance.source == "psutil"
    assert state.value("resources.cpu_percent", allow_stale=False) == 82.0
    clock.advance(31)
    assert state.is_stale("resources.cpu_percent")
    assert state.value("resources.cpu_percent", allow_stale=False) is None


def test_state_survives_restart(db, clock):
    StateEngine(db, clock).set("deployment.status", "RUNNING")
    StateEngine(db, clock).set("volatile", 1, persist=False)
    restored = StateEngine(db, clock)
    assert restored.restore() == 1
    assert restored.value("deployment.status") == "RUNNING"
    assert restored.value("volatile") is None


def test_state_change_callbacks_only_on_change(clock):
    state = StateEngine(None, clock)
    seen = []
    state.on_change("mode.", lambda k, old, new: seen.append((k, new.value if new else None)))
    state.set("mode.current", "focus")
    state.set("mode.current", "focus")
    state.set("mode.current", "normal")
    state.set("other", 1)
    assert seen == [("mode.current", "focus"), ("mode.current", "normal")]


def test_world_model_relations_and_chain(db, clock):
    world = WorldModel(db, clock)
    svc = world.upsert_entity("service", "api", {"port": 8080})
    dbe = world.upsert_entity("service", "postgres")
    pool = world.upsert_entity("resource", "connection-pool", {"used": 100, "max": 100})
    world.relate(svc.id, DEPENDS_ON, dbe.id)
    world.relate(dbe.id, DEPENDS_ON, pool.id)
    world.relate(pool.id, DEPENDS_ON, svc.id)  # cycle must not loop forever
    assert [e.name for e in world.chain(svc.id)] == ["postgres", "connection-pool"]
    related = world.find_related(dbe.id, direction="in")
    assert related[0][1].name == "api"
    world.upsert_entity("service", "api", {"healthy": False})
    assert world.get_entity(svc.id).attrs == {"port": 8080, "healthy": False}
    world.delete_entity(dbe.id)
    assert world.relations(svc.id) == []


def test_world_model_name_resolution(db, clock):
    world = WorldModel(db, clock)
    world.upsert_entity("project", "Robotics")
    world.upsert_entity("project", "Local AI Assistant")
    assert [e.name for e in world.resolve_name("robotics", ["project"])] == ["Robotics"]
    assert [e.name for e in world.resolve_name("local ai", ["project"])] == ["Local AI Assistant"]
    assert [e.name for e in world.resolve_name("robotcs", ["project"])] == ["Robotics"]


async def test_health_outages_and_overall(clock):
    bus = EventBus(clock=clock)
    events = []
    bus.subscribe("SUBSYSTEM_*", lambda e: events.append((e.type, e.payload.get("duration"))))
    health = HealthRegistry(bus, clock)
    health.report("database", HealthStatus.HEALTHY, critical=True)
    health.report("voice", HealthStatus.HEALTHY)
    assert health.overall() == HealthStatus.HEALTHY
    health.report("voice", HealthStatus.OFFLINE, "microphone lost")
    assert health.overall() == HealthStatus.DEGRADED  # optional component only degrades
    health.report("model:ollama", HealthStatus.OFFLINE, "connection refused", critical=True)
    assert health.overall() == HealthStatus.OFFLINE
    clock.advance(18)
    health.report("model:ollama", HealthStatus.HEALTHY)
    await bus.drain()
    outage = health.recent_outages(component="model:ollama")[0]
    assert outage.duration == 18
    assert ("SUBSYSTEM_RECOVERED", 18) in events


def test_heartbeat_timeout(clock):
    health = HealthRegistry(None, clock)
    health.register("worker", heartbeat_timeout_s=10)
    health.report("worker", HealthStatus.HEALTHY)
    clock.advance(11)
    assert health.check_heartbeats() == ["worker"]
    assert health.components["worker"].status == HealthStatus.OFFLINE
