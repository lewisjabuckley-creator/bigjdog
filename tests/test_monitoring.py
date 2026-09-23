from jarvis.audit.log import AuditLog
from jarvis.config import JarvisConfig, MonitoringConfig, Threshold
from jarvis.core.emergency import EmergencyController
from jarvis.core.modes import Mode, ModeManager
from jarvis.core.types import HealthStatus, NetworkState
from jarvis.events.bus import EventBus
from jarvis.events.store import EventStore
from jarvis.models.fake import ScriptedProvider
from jarvis.models.router import ModelRouter
from jarvis.monitoring.analysis import ThresholdDetector, TrendTracker
from jarvis.monitoring.metrics import PsutilMetrics, StaticMetrics
from jarvis.monitoring.service import ModelMonitor, NetworkMonitor, SelfMonitor, SystemMonitor
from jarvis.state.engine import StateEngine
from jarvis.state.health import HealthRegistry
from jarvis.world.model import WorldModel


def test_sustained_threshold_ignores_transient_spikes():
    det = ThresholdDetector({"cpu_percent": Threshold(90, sustain_s=300)})
    assert det.evaluate({"cpu_percent": 99}, 0) == []
    assert det.evaluate({"cpu_percent": 40}, 60) == []        # spike ended: nothing reported
    assert det.evaluate({"cpu_percent": 95}, 100) == []
    events = det.evaluate({"cpu_percent": 96}, 401)            # sustained 5 minutes
    assert events and events[0].exceeded and "for 5 minutes" in events[0].message
    assert det.evaluate({"cpu_percent": 88}, 460) == []        # hysteresis: not cleared yet
    cleared = det.evaluate({"cpu_percent": 50}, 500)
    assert cleared and not cleared[0].exceeded


def test_battery_threshold_respects_power():
    det = ThresholdDetector({"battery_low_percent": Threshold(15)})
    assert det.evaluate({"battery_percent": 10, "battery_plugged": True}, 0) == []
    assert det.evaluate({"battery_percent": 10, "battery_plugged": False}, 1)[0].exceeded


def test_trend_and_prediction():
    tr = TrendTracker()
    for i in range(60):     # disk grows 1% per hour, sampled every minute
        tr.add("disk_percent", i * 60.0, 80 + i / 60)
    trend = tr.trend("disk_percent")
    assert abs(trend.slope_per_hour - 1.0) < 0.01 and trend.r2 > 0.99
    eta = tr.time_to_threshold("disk_percent", 90)
    assert 8.9 * 3600 < eta < 9.1 * 3600
    assert tr.time_to_threshold("disk_percent", 50) is None    # moving away from it


def test_psutil_metrics_real_sample():
    sample = PsutilMetrics().sample()
    assert 0 <= sample["memory_percent"] <= 100 and sample["memory_total_gb"] > 0


async def test_system_monitor_updates_state_and_emits(db, clock):
    bus = EventBus(EventStore(db), clock)
    events = []
    bus.subscribe("RESOURCE_*", lambda e: events.append(e))
    state = StateEngine(db, clock)
    metrics = StaticMetrics(disk_percent=98.0)
    cfg = MonitoringConfig(thresholds={"disk_critical_percent": Threshold(97, severity="critical")})
    world = WorldModel(db, clock)
    mon = SystemMonitor(metrics, state, cfg, bus=bus, clock=clock, world=world)
    await mon.sample_once()
    await bus.drain()
    assert state.value("resources.disk_percent") == 98.0
    assert state.get("resources.disk_percent").provenance.source == "system monitor"
    assert events and events[0].payload["severity"] == "critical"
    assert world.find_entities("machine")


async def test_predictive_warning_is_labelled_estimate(db, clock):
    bus = EventBus(clock=clock)
    got = []
    bus.subscribe("PREDICTIVE_WARNING", lambda e: got.append(e.payload))
    state = StateEngine(db, clock)
    metrics = StaticMetrics()
    mon = SystemMonitor(metrics, state, MonitoringConfig(), bus=bus, clock=clock)
    for i in range(30):
        metrics.set(disk_percent=70 + i * 0.5)
        await mon.sample_once()
        clock.advance(600)
    await bus.drain()
    assert got and got[0]["confidence"] == "estimated" and "approximately" in got[0]["message"]


async def test_network_monitor_states(db, clock):
    bus = EventBus(clock=clock)
    changes = []
    bus.subscribe("NETWORK_CHANGED", lambda e: changes.append(e.payload["state"]))
    state = StateEngine(db, clock)
    result = {"ok": True, "latency": 20.0}

    async def probe():
        return result["ok"], result["latency"] if result["ok"] else None

    mon = NetworkMonitor(state, MonitoringConfig(), bus=bus, clock=clock, probe=probe, interfaces=lambda: False)
    assert await mon.check_once() == NetworkState.ONLINE
    result.update(ok=False)
    assert await mon.check_once() == NetworkState.OFFLINE
    result.update(ok=True, latency=900.0)
    assert await mon.check_once() == NetworkState.HIGH_LATENCY
    await bus.drain()
    assert changes == ["offline", "high_latency"]


async def test_model_monitor_populates_state_and_world(db, clock):
    router = ModelRouter([ScriptedProvider()], clock=clock)
    state = StateEngine(db, clock)
    world = WorldModel(db, clock)
    await ModelMonitor(router, state, world=world, host="box").check_once()
    assert "sim-chat:8b" in state.value("models.available")
    assert world.get_entity("model:sim-chat:8b") is not None


def test_self_monitor(db, clock):
    bus = EventBus(clock=clock)
    health = HealthRegistry(bus, clock)
    SelfMonitor(db, bus, health).check_once()
    assert health.components["database"].status == HealthStatus.HEALTHY


async def test_emergency_enters_on_critical_and_exits_when_cleared(db, clock, tmp_path):
    bus = EventBus(EventStore(db), clock)
    state = StateEngine(db, clock)
    modes = ModeManager(state, JarvisConfig(), bus=bus, clock=clock)
    modes.set_mode(Mode.FOCUS)
    audit = AuditLog(db, clock)
    ctl = EmergencyController(modes, state, bus, audit, events=EventStore(db), clock=clock,
                              evidence_dir=str(tmp_path / "evidence"))
    ctl.attach()
    metrics = StaticMetrics(disk_percent=98.5)
    cfg = MonitoringConfig(thresholds={"disk_critical_percent": Threshold(97, severity="critical")})
    mon = SystemMonitor(metrics, state, cfg, bus=bus, clock=clock)
    await mon.sample_once()
    await bus.drain()
    assert modes.current == Mode.EMERGENCY
    assert list((tmp_path / "evidence").glob("emergency-*.json"))
    assert audit.last_decision().action == "enter_emergency"
    metrics.set(disk_percent=80.0)
    await mon.sample_once()
    await bus.drain()
    assert modes.current == Mode.FOCUS     # restored to the mode before the emergency
