"""Runtime: wiring plus the startup and shutdown sequences (spec §152-155).

Startup: initialise core → configuration → database → events → restore state →
recover tasks → detect models → start workers and monitors → verify subsystems
→ report only what matters → ready. Shutdown: checkpoint running tasks, stop
workers and monitors safely, persist state, close resources.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from jarvis.audit.log import AuditLog
from jarvis.automation.engine import AutomationEngine, notify_action
from jarvis.clock import Clock, SystemClock
from jarvis.config import JarvisConfig
from jarvis.core.emergency import EmergencyController
from jarvis.core.modes import ModeManager
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.services import Services
from jarvis.core.types import HealthStatus, Severity
from jarvis.database.db import Database
from jarvis.devices.registry import DeviceCommandTool, DeviceRegistry
from jarvis.events.bus import EventBus
from jarvis.events.store import EventStore
from jarvis.events.types import Event, EventType
from jarvis.log import configure_logging, get_logger
from jarvis.memory.decisions import DecisionLog
from jarvis.memory.store import MemoryStore
from jarvis.models.base import ModelProvider
from jarvis.models.ollama import OllamaProvider
from jarvis.models.openai_compat import OpenAICompatibleProvider
from jarvis.models.router import ModelRouter
from jarvis.monitoring.metrics import MetricsSource, PsutilMetrics
from jarvis.monitoring.service import (ModelMonitor, MonitoringService, NetworkMonitor, NetworkProbe, SelfMonitor,
                                       SystemMonitor)
from jarvis.monitoring.watchers import MonitorChecker
from jarvis.notifications.manager import NotificationManager
from jarvis.permissions.manager import ApprovalManager, PermissionManager
from jarvis.permissions.model import Actor
from jarvis.planner.planner import Planner
from jarvis.projects.manager import ProjectManager
from jarvis.security.secrets import SecretStore
from jarvis.state.engine import StateEngine
from jarvis.state.health import HealthRegistry
from jarvis.tasks.executor import TaskExecutor
from jarvis.tasks.manager import RecoveryReport, TaskManager
from jarvis.tasks.resources import ResourceManager
from jarvis.tasks.workers import WorkerPool
from jarvis.tools.base import ExecutionPolicy, ToolContext
from jarvis.tools.builtin import register_builtin_tools
from jarvis.tools.registry import ExecStatus, ToolRegistry
from jarvis.verification.engine import Verifier
from jarvis.world.model import WorldModel

log = get_logger("runtime")


@dataclass
class StartupReport:
    issues: list[str] = field(default_factory=list)
    recovered: list[RecoveryReport] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    def greeting(self) -> str:
        parts = ["Ready."]
        if not self.models:
            parts.append("No language model is reachable, so I'm running on deterministic capabilities only.")
        interrupted = [r for r in self.recovered if not r.resumed]
        resumed = [r for r in self.recovered if r.resumed]
        if interrupted:
            parts.append(" ".join(r.summary for r in interrupted[:2]) + " Say 'continue' to resume.")
        if resumed:
            parts.append(f"Resumed {len(resumed)} monitor{'s' if len(resumed) != 1 else ''} after the restart.")
        parts += self.issues[:3]
        return " ".join(parts)


class Runtime:
    def __init__(self, config: JarvisConfig, *, providers: list[ModelProvider] | None = None,
                 metrics: MetricsSource | None = None, clock: Clock | None = None,
                 network_probe: NetworkProbe | None = None, db_path: str | None = None,
                 simulated: bool = False, log_to_stderr: bool = False) -> None:
        self.config = config
        self.clock = clock or SystemClock()
        self._providers = providers
        self._metrics = metrics
        self._network_probe = network_probe
        self._db_path = db_path
        self.simulated = simulated
        self.log_to_stderr = log_to_stderr
        self.svc: Services | None = None
        self.started = False
        self._orchestrator: Orchestrator | None = None

    # -- wiring ---------------------------------------------------------------------------------
    def _build(self) -> Services:
        cfg = self.config
        clock = self.clock
        data = cfg.data_path
        data.mkdir(parents=True, exist_ok=True)
        configure_logging(data / "logs", logging.INFO, stderr=self.log_to_stderr)
        db = Database(self._db_path or cfg.db_path)
        events = EventStore(db, cfg.events)
        bus = EventBus(events, clock, recent_size=cfg.events.recent_buffer)
        state = StateEngine(db, clock)
        state.restore()
        world = WorldModel(db, clock)
        world.upsert_entity("user", cfg.general.user, {"role": "owner"}, id=f"user:{cfg.general.user}")
        health = HealthRegistry(bus, clock)
        for name in ("database", "event_bus", "workers"):
            health.register(name, critical=True)
        permissions = PermissionManager(db, cfg.permissions, bus=bus, clock=clock)
        approvals = ApprovalManager(db, permissions, bus=bus, clock=clock)
        audit = AuditLog(db, clock)
        tasks = TaskManager(db, bus=bus, clock=clock, world=world)
        projects = ProjectManager(db, state, world=world, bus=bus, clock=clock, owner=cfg.general.user)
        permissions.paths.extra_roots = lambda: [p.root for p in projects.list() if p.root]

        def project_sensitive() -> bool:
            p = projects.active()
            return bool(p and (p.policy.sensitive or p.policy.local_models_only))

        modes = ModeManager(state, cfg, bus=bus, clock=clock, project_sensitive=project_sensitive)

        def execution_policy() -> ExecutionPolicy:
            p = projects.active()
            base = ExecutionPolicy()
            if p is not None:
                base = ExecutionPolicy(network_allowed=p.policy.network,
                                       network_block_reason=f"network access is disabled for {p.name}",
                                       allowed_tools=p.policy.allowed_tools,
                                       allowed_dirs=p.dirs if (p.policy.sensitive or p.policy.allowed_dirs) else None,
                                       project_id=p.id)
            return modes.execution_policy(base)

        registry = ToolRegistry(permissions, audit, bus=bus, policy_provider=execution_policy)
        metrics = self._metrics or PsutilMetrics()
        register_builtin_tools(registry, metrics_source=metrics)
        devices = DeviceRegistry(bus=bus, clock=clock)
        registry.register(DeviceCommandTool(devices))

        providers = self._providers if self._providers is not None else self._default_providers()
        router = ModelRouter(providers, cfg.models, bus=bus, clock=clock, health=health,
                             local_only=lambda: modes.effective().local_only or bool(state.value("models.prefer_local")),
                             allow_cloud=lambda: cfg.privacy.allow_cloud and not modes.private)
        planner = Planner(registry, router)

        async def verify_command(command: str, cwd: str | None) -> tuple[int | None, str]:
            ctx = ToolContext(actor=Actor("system", "verifier"), cwd=cwd, clock=clock, data_dir=str(data))
            ex = await registry.execute("shell_execute", {"command": command, "cwd": cwd, "timeout_s": 60}, ctx)
            if ex.status != ExecStatus.EXECUTED or ex.result is None:
                return None, ex.message
            out = ex.result.data or {}
            return ex.result.exit_code, f"{out.get('stdout', '')}{out.get('stderr', '')}"

        resources = ResourceManager(state, clock=clock, mode_provider=lambda: modes.current.value)
        executor = TaskExecutor(tasks, registry, approvals, permissions, audit, planner=planner,
                                verifier=Verifier(verify_command),
                                monitor_checker=MonitorChecker(tasks, registry, state, bus), bus=bus, clock=clock,
                                data_dir=str(data), context_provider=self._planning_context(projects))
        pool = WorkerPool(tasks, executor, resources, config=cfg.tasks, bus=bus, clock=clock, health=health, audit=audit)
        embedder = router.embed if cfg.memory.use_embeddings else None
        memory = MemoryStore(db, clock=clock, embedder=embedder, bus=bus)
        decisions = DecisionLog(db, clock)
        notifications = NotificationManager(db, modes, config=cfg.notifications, bus=bus, clock=clock)
        notifications.attach(bus)
        emergency = EmergencyController(modes, state, bus, audit, events=events, clock=clock,
                                        evidence_dir=str(data / "emergency"))
        emergency.attach()
        automations = AutomationEngine(db, tasks, bus=bus, clock=clock)
        automations.handlers["notify"] = notify_action(notifications)
        automations.attach()
        tasks.on_cancel.append(lambda t: approvals.cancel_for_task(t.id))
        return Services(cfg, clock, db, bus, events, state, world, health, permissions, approvals, audit, registry,
                        router, tasks, resources, pool, memory, decisions, projects, modes, notifications, emergency,
                        automations, devices, metrics, user=cfg.general.user, simulated=self.simulated)

    def _default_providers(self) -> list[ModelProvider]:
        cfg = self.config.models
        providers: list[ModelProvider] = []
        if cfg.ollama.enabled:
            providers.append(OllamaProvider(cfg.ollama.base_url, keep_alive=cfg.ollama.keep_alive,
                                            timeout=cfg.ollama.request_timeout_s))
        oc = cfg.openai_compatible
        if oc.enabled and oc.base_url:
            key = SecretStore().get(oc.api_key_env) if oc.api_key_env else None
            providers.append(OpenAICompatibleProvider(oc.base_url, api_key=key, local=oc.local, models=oc.models,
                                                      name=oc.name, timeout=oc.request_timeout_s))
        return providers

    @staticmethod
    def _planning_context(projects: ProjectManager) -> Any:
        def ctx(task: Any) -> str:
            project = projects.get(task.project_id) if task.project_id else projects.active()
            if project is None:
                return ""
            info = projects.context(project)
            return "Project: " + ", ".join(f"{k}={v}" for k, v in info.items() if v)
        return ctx

    # -- lifecycle ----------------------------------------------------------------------------------
    async def start(self, *, monitoring: bool | None = None) -> StartupReport:
        started = self.clock.monotonic()
        report = StartupReport()
        svc = self._build()
        self.svc = svc
        svc.health.report("database", HealthStatus.HEALTHY if svc.db.healthy() else HealthStatus.CRITICAL)
        svc.health.report("event_bus", HealthStatus.HEALTHY)
        idempotent = lambda tool: bool(tool and (t := svc.registry.get(tool)) and t.spec.idempotent)  # noqa: E731
        report.recovered = svc.tasks.recover_interrupted(idempotent)
        svc.extra["recovery_reports"] = report.recovered
        try:
            inventory = await asyncio.wait_for(svc.router.refresh(), timeout=10)
            report.models = [m.name for m in inventory]
        except Exception as exc:
            report.issues.append(f"Model discovery failed: {exc}.")
        await svc.pool.start()
        use_monitoring = self.config.monitoring.enabled if monitoring is None else monitoring
        system = SystemMonitor(svc.metrics, svc.state, self.config.monitoring, bus=svc.bus, clock=svc.clock,
                               world=svc.world, health=svc.health)
        try:
            await system.sample_once()
        except Exception as exc:
            report.issues.append(f"System metrics are unavailable ({exc}).")
        if use_monitoring:
            svc.monitoring = MonitoringService(
                self.config.monitoring, system=system,
                network=NetworkMonitor(svc.state, self.config.monitoring, bus=svc.bus, clock=svc.clock,
                                       probe=self._network_probe),
                models=ModelMonitor(svc.router, svc.state, world=svc.world),
                self_monitor=SelfMonitor(svc.db, svc.bus, svc.health), health=svc.health,
                extra=[("automation", 30.0, svc.automations.tick), ("retention", 6 * 3600.0, self._maintenance)])
            await svc.monitoring.start()
        for comp in svc.health.unhealthy():
            if comp.critical:
                report.issues.append(f"{comp.name} is {comp.status.label}: {comp.detail}.")
        svc.bus.emit(Event(EventType.SYSTEM_STARTED, "runtime",
                           {"models": len(report.models), "recovered": len(report.recovered),
                            "simulated": self.simulated}))
        await svc.bus.drain()
        self.started = True
        report.duration_s = round(self.clock.monotonic() - started, 3)
        log.info("started", duration_s=report.duration_s, models=len(report.models))
        return report

    async def _maintenance(self) -> None:
        assert self.svc is not None
        pruned = self.svc.events.prune(self.svc.clock.now())
        expired = self.svc.memory.purge_expired()
        log.info("maintenance", events_pruned=pruned, memories_expired=expired)

    def orchestrator(self) -> Orchestrator:
        assert self.svc is not None, "runtime not started"
        if self._orchestrator is None:
            self._orchestrator = Orchestrator(self.svc)
        return self._orchestrator

    async def stop(self) -> None:
        if not self.svc:
            return
        svc = self.svc
        svc.bus.emit(Event(EventType.SYSTEM_STOPPING, "runtime", {}, severity=Severity.INFO))
        if svc.monitoring:
            await svc.monitoring.stop()
        await svc.pool.stop()
        await svc.bus.drain()
        await svc.router.close()
        svc.db.close()
        self.started = False
        log.info("stopped")

    async def __aenter__(self) -> "Runtime":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()
