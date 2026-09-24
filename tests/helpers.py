"""Assembles a task engine against an in-memory database for tests."""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.audit.log import AuditLog
from jarvis.clock import Clock, SystemClock
from jarvis.config import PermissionsConfig, TasksConfig
from jarvis.database.db import Database
from jarvis.events.bus import EventBus
from jarvis.events.store import EventStore
from jarvis.models.fake import ScriptedProvider
from jarvis.models.router import ModelRouter
from jarvis.monitoring.watchers import MonitorChecker
from jarvis.permissions.manager import ApprovalManager, PermissionManager
from jarvis.planner.planner import Planner
from jarvis.state.engine import StateEngine
from jarvis.state.health import HealthRegistry
from jarvis.tasks.executor import TaskExecutor
from jarvis.tasks.manager import TaskManager
from jarvis.tasks.resources import ResourceManager
from jarvis.tasks.workers import WorkerPool
from jarvis.tools.builtin import register_builtin_tools
from jarvis.tools.registry import ToolRegistry
from jarvis.world.model import WorldModel


@dataclass
class Engine:
    db: Database
    bus: EventBus
    state: StateEngine
    permissions: PermissionManager
    approvals: ApprovalManager
    audit: AuditLog
    registry: ToolRegistry
    tasks: TaskManager
    resources: ResourceManager
    executor: TaskExecutor
    pool: WorkerPool
    provider: ScriptedProvider
    router: ModelRouter
    health: HealthRegistry


def build_engine(root: str, db: Database | None = None, clock: Clock | None = None, *,
                 max_concurrent: int = 4) -> Engine:
    db = db or Database(":memory:")
    clock = clock or SystemClock()
    bus = EventBus(EventStore(db), clock)
    state = StateEngine(db, clock)
    health = HealthRegistry(bus, clock)
    permissions = PermissionManager(db, PermissionsConfig(allowed_roots=[root], denied_paths=[]), bus=bus, clock=clock)
    approvals = ApprovalManager(db, permissions, bus=bus, clock=clock)
    audit = AuditLog(db, clock)
    registry = ToolRegistry(permissions, audit, bus=bus)
    register_builtin_tools(registry)
    world = WorldModel(db, clock)
    tasks = TaskManager(db, bus=bus, clock=clock, world=world)
    provider = ScriptedProvider()
    router = ModelRouter([provider], bus=bus, clock=clock, health=health)
    planner = Planner(registry, router)
    resources = ResourceManager(state, clock=clock)
    executor = TaskExecutor(tasks, registry, approvals, permissions, audit, planner=planner,
                            monitor_checker=MonitorChecker(tasks, registry, state, bus), bus=bus, clock=clock,
                            data_dir=f"{root}/.jarvis")
    pool = WorkerPool(tasks, executor, resources, config=TasksConfig(max_concurrent=max_concurrent,
                                                                     scheduler_interval_s=0.02),
                      bus=bus, clock=clock, health=health, audit=audit)
    return Engine(db, bus, state, permissions, approvals, audit, registry, tasks, resources, executor, pool,
                  provider, router, health)


# -- a full runtime (Phase 2 tests) -------------------------------------------------------------------

def runtime_config(tmp: str, **overrides):
    from jarvis.config import config_from_dict
    data = {"general": {"data_dir": f"{tmp}/data"}, "monitoring": {"enabled": False},
            "permissions": {"allowed_roots": [tmp]}, "tasks": {"scheduler_interval_s": 0.02},
            "scheduler": {"tick_s": 0.05}, "runtime": {"heartbeat_s": 0.1}}
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    return config_from_dict(data)


def make_runtime(tmp: str, *, mode: str = "embedded", sim=None, clock=None, **overrides):
    from jarvis.runtime import Runtime
    from jarvis.simulation.environment import SimulatedEnvironment
    sim = sim or SimulatedEnvironment()
    rt = Runtime(runtime_config(tmp, **overrides), providers=[sim.provider], metrics=sim.metrics,
                 network_probe=sim.probe, simulated=True, mode=mode, clock=clock)
    return rt, sim


async def wait_until(predicate, timeout: float = 10.0, interval: float = 0.02):
    import asyncio
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise TimeoutError("condition not met in time")
        await asyncio.sleep(interval)
