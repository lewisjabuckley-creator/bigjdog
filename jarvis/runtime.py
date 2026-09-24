"""Runtime: wiring plus the startup and shutdown sequences (spec §152-155).

Startup: take the single-instance lock → initialise core → configuration →
database → events → restore state → record this run (and notice if the last
one never shut down) → recover tasks → detect models → start workers, the
scheduler, the heartbeat and monitors → verify subsystems → report only what
matters → ready. Shutdown: checkpoint running tasks, stop workers and loops
safely, persist state, record a clean stop, close resources, release the lock.

The same Runtime runs in-process (``jarvis --embedded``, tests) or as the
persistent background runtime (``jarvis runtime start``, see
:mod:`jarvis.service`); interfaces are clients of it, never owners.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import psutil

from jarvis import __version__
from jarvis import platforms as platform_adapters
from jarvis.agents.base import AgentRegistry, AgentRunner, DelegateToAgentTool
from jarvis.audit.log import AuditLog
from jarvis.automation.engine import AutomationEngine, notify_action, validate_schedule
from jarvis.clock import Clock, SystemClock, format_datetime, format_duration
from jarvis.config import JarvisConfig
from jarvis.core.emergency import EmergencyController
from jarvis.core.modes import ModeManager
from jarvis.core.orchestrator import Orchestrator
from jarvis.core.presence import Presence
from jarvis.core.services import Services
from jarvis.core.types import HealthStatus, Severity, new_id
from jarvis.database.db import Database, dumps
from jarvis.devices.registry import DeviceCommandTool, DeviceRegistry
from jarvis.events.bus import EventBus
from jarvis.events.store import EventStore
from jarvis.events.types import Event, EventType
from jarvis.intelligence.service import IntelligenceService
from jarvis.log import configure_logging, get_logger
from jarvis.memory.decisions import DecisionLog
from jarvis.memory.store import MemoryStore
from jarvis.models.base import ModelProvider
from jarvis.models.ollama import OllamaProvider
from jarvis.models.openai_compat import OpenAICompatibleProvider
from jarvis.models.readiness import ModelReadiness
from jarvis.models.readiness import assess as assess_models
from jarvis.models.router import ModelRouter
from jarvis.models.scheduler import InferenceScheduler
from jarvis.monitoring.metrics import MetricsSource, PsutilMetrics
from jarvis.monitoring.service import (ModelMonitor, MonitoringService, NetworkMonitor, NetworkProbe, SelfMonitor,
                                       SystemMonitor)
from jarvis.monitoring.watchers import MonitorChecker
from jarvis.notifications.manager import NotificationManager
from jarvis.permissions.manager import ApprovalManager, PermissionManager
from jarvis.permissions.model import Actor, PermissionLevel
from jarvis.planner.planner import Planner
from jarvis.projects.manager import ProjectManager
from jarvis.security.secrets import SecretStore
from jarvis.state.engine import StateEngine
from jarvis.state.health import HealthRegistry
from jarvis.tasks.executor import TaskExecutor
from jarvis.tasks.manager import RecoveryReport, TaskManager
from jarvis.tasks.models import TaskStatus
from jarvis.tasks.resources import ResourceManager
from jarvis.tasks.workers import WorkerPool
from jarvis.tools.base import ExecutionPolicy, ToolContext
from jarvis.tools.builtin import register_builtin_tools
from jarvis.tools.internal import ModelReportTool
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
    readiness: ModelReadiness | None = None
    unclean_previous_stop: bool = False
    plans: list[dict[str, Any]] = field(default_factory=list)

    def greeting(self) -> str:
        parts = ["Ready."]
        r = self.readiness
        if r is not None and r.can_converse:
            parts.append(f"Talking through {r.summary()}.")
        else:
            parts.append("No language model is available, so I'm running on deterministic capabilities only "
                         "(status, tasks, monitoring, memory, commands).")
        if r is not None:
            parts += r.issues[:2]
        interrupted = [r for r in self.recovered if not r.resumed]
        resumed = [r for r in self.recovered if r.resumed]
        if interrupted:
            text = " ".join(r.summary for r in interrupted[:2])
            parts.append(text if "'continue'" in text else text + " Say 'continue' to resume.")
        if resumed:
            parts.append(f"Resumed {len(resumed)} task{'s' if len(resumed) != 1 else ''} after the restart.")
        held_plans = [p for p in self.plans if p["status"] in ("paused", "blocked")]
        if held_plans:
            parts.append(" ".join(p["summary"].rstrip(".") + "." for p in held_plans[:2]))
        parts += self.issues[:3]
        return " ".join(parts)


class Runtime:
    def __init__(self, config: JarvisConfig, *, providers: list[ModelProvider] | None = None,
                 metrics: MetricsSource | None = None, clock: Clock | None = None,
                 network_probe: NetworkProbe | None = None, db_path: str | None = None,
                 simulated: bool = False, log_to_stderr: bool = False, mode: str = "embedded",
                 single_instance: bool = True, passive: bool = False) -> None:
        self.config = config
        self.clock = clock or SystemClock()
        self._providers = providers
        self._metrics = metrics
        self._network_probe = network_probe
        self._db_path = db_path
        self.simulated = simulated
        self.log_to_stderr = log_to_stderr
        self.mode = mode                        # embedded | daemon
        self.single_instance = single_instance
        # passive: read and report only (one-shot CLI commands while no runtime is running). No workers,
        # scheduler, recovery or run record, so a quick `jarvis tasks` never starts, interrupts or re-labels work
        # and never hides an earlier crash from the next real start.
        self.passive = passive
        self.platform = platform_adapters.current()
        self.svc: Services | None = None
        self.started = False
        self.run_id: str | None = None
        self.started_at: float | None = None
        # presence decides notification delivery only where interfaces attach and detach (the daemon)
        self.track_presence = mode == "daemon"
        self._lock: platform_adapters.InstanceLock | None = None
        self._previous_run: dict[str, Any] | None = None
        self._loops: list[asyncio.Task[None]] = []
        self._loop_ticks: dict[str, float] = {}
        self._orchestrators: dict[str, Orchestrator] = {}
        self._process = psutil.Process(os.getpid())
        self._self_memory_high = False

    # -- wiring ---------------------------------------------------------------------------------
    def _build(self) -> Services:
        cfg = self.config
        clock = self.clock
        data = cfg.data_path
        data.mkdir(parents=True, exist_ok=True)
        configure_logging(data / "logs", logging.INFO, stderr=self.log_to_stderr)
        db = Database(self._db_path or cfg.db_path)
        previous = db.query_one("SELECT * FROM runtime_runs ORDER BY started_at DESC, rowid DESC LIMIT 1")
        self._previous_run = dict(previous) if previous else None
        events = EventStore(db, cfg.events)
        bus = EventBus(events, clock, recent_size=cfg.events.recent_buffer)
        state = StateEngine(db, clock)
        state.restore()
        last_alive = None
        if self._previous_run:
            last_alive = self._previous_run.get("stopped_at") or self._previous_run.get("heartbeat_at")
        presence = Presence(state, bus=bus, clock=clock, timeout_s=cfg.runtime.presence_timeout_s,
                            default_since=last_alive)
        world = WorldModel(db, clock)
        world.upsert_entity("user", cfg.general.user, {"role": "owner"}, id=f"user:{cfg.general.user}")
        health = HealthRegistry(bus, clock)
        for name in ("database", "event_bus", "workers", "runtime", "scheduler"):
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
        registry.register(ModelReportTool(router))
        planner = Planner(registry, router)
        agents = AgentRegistry()
        agent_runner = AgentRunner(registry, router, max_concurrent=cfg.intelligence.agents_max_concurrent)
        registry.register(DelegateToAgentTool(agents, agent_runner))

        async def verify_command(command: str, cwd: str | None) -> tuple[int | None, str]:
            ctx = ToolContext(actor=Actor("system", "verifier"), cwd=cwd, clock=clock, data_dir=str(data))
            ex = await registry.execute("shell_execute", {"command": command, "cwd": cwd, "timeout_s": 60}, ctx)
            if ex.status != ExecStatus.EXECUTED or ex.result is None:
                return None, ex.message
            out = ex.result.data or {}
            return ex.result.exit_code, f"{out.get('stdout', '')}{out.get('stderr', '')}"

        resources = ResourceManager(state, clock=clock, mode_provider=lambda: modes.current.value,
                                    memory_critical=cfg.resources.memory_critical,
                                    cpu_critical=cfg.resources.cpu_critical, vram_critical=cfg.resources.vram_critical,
                                    low_priority_policy=cfg.resources.low_priority_policy)
        # model requests queue by priority (the user first); under pressure one at a time
        router.scheduler = InferenceScheduler(cfg.intelligence.max_concurrent_inference, pressure=resources.pressure)
        executor = TaskExecutor(tasks, registry, approvals, permissions, audit, planner=planner,
                                verifier=Verifier(verify_command),
                                monitor_checker=MonitorChecker(tasks, registry, state, bus), bus=bus, clock=clock,
                                data_dir=str(data), context_provider=self._planning_context(projects))
        pool = WorkerPool(tasks, executor, resources, config=cfg.tasks, bus=bus, clock=clock, health=health, audit=audit)
        embedder = router.embed if cfg.memory.use_embeddings else None
        memory = MemoryStore(db, clock=clock, embedder=embedder, bus=bus)
        decisions = DecisionLog(db, clock)
        notifications = NotificationManager(db, modes, config=cfg.notifications, bus=bus, clock=clock)
        notifications.restore()
        notifications.away = lambda: self.track_presence and presence.away
        notifications.attach(bus)
        emergency = EmergencyController(modes, state, bus, audit, events=events, clock=clock,
                                        evidence_dir=str(data / "emergency"))
        emergency.attach()
        automations = AutomationEngine(db, tasks, bus=bus, clock=clock,
                                       catch_up_window_s=cfg.scheduler.catch_up_window_s)
        automations.handlers["notify"] = notify_action(notifications)
        automations.handlers["briefing"] = self._briefing_action
        automations.attach()
        bus.subscribe(str(EventType.MODEL_RECOVERED), lambda e: self._resume_model_waiters(), name="model-waiters")
        tasks.on_cancel.append(lambda t: approvals.cancel_for_task(t.id))
        svc = Services(cfg, clock, db, bus, events, state, world, health, permissions, approvals, audit, registry,
                       router, tasks, resources, pool, memory, decisions, projects, modes, notifications, emergency,
                       automations, devices, metrics, presence=presence, user=cfg.general.user,
                       simulated=self.simulated, extra={"agents": agents, "agent_runner": agent_runner})
        svc.intelligence = IntelligenceService(svc, agents=agents, runner=agent_runner)
        return svc

    def _default_providers(self) -> list[ModelProvider]:
        cfg = self.config.models
        providers: list[ModelProvider] = []
        if cfg.ollama.enabled:
            options: dict[str, Any] = {"num_ctx": cfg.ollama.num_ctx}
            options.update({k: (int(v) if k in ("seed", "num_ctx", "num_predict", "top_k") else v)
                            for k, v in cfg.ollama.options.items()})
            providers.append(OllamaProvider(cfg.ollama.base_url, keep_alive=cfg.ollama.keep_alive,
                                            timeout=cfg.ollama.request_timeout_s, default_options=options))
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
        if self.single_instance and self._lock is None:
            # one runtime per data directory: two would both schedule the same queued tasks
            self._lock = self.platform.instance_lock(self.config.data_path / "runtime.lock").acquire()
        try:
            return await self._start(monitoring)
        except BaseException:
            self._release()
            raise

    async def _start(self, monitoring: bool | None) -> StartupReport:
        started = self.clock.monotonic()
        report = StartupReport()
        svc = self._build()
        self.svc = svc
        now = self.clock.now()
        self.started_at = now
        if self.passive:
            return await self._start_passive(svc, report, started)
        self.run_id = new_id("run")
        previous = self._previous_run
        report.unclean_previous_stop = bool(previous and not previous["clean"])
        svc.db.execute("INSERT INTO runtime_runs(id, pid, mode, version, host, started_at, heartbeat_at, info) "
                       "VALUES(?,?,?,?,?,?,?,?)", (self.run_id, os.getpid(), self.mode, __version__,
                                                   socket.gethostname(), now, now,
                                                   dumps({"simulated": self.simulated})))
        svc.extra["run"] = {"id": self.run_id, "pid": os.getpid(), "mode": self.mode, "started_at": now,
                            "previous": previous, "unclean_previous_stop": report.unclean_previous_stop}
        svc.health.report("database", HealthStatus.HEALTHY if svc.db.healthy() else HealthStatus.CRITICAL)
        svc.health.report("event_bus", HealthStatus.HEALTHY)
        svc.health.report("runtime", HealthStatus.HEALTHY, "starting")
        report.recovered = self._recover(svc)
        svc.extra["recovery_reports"] = report.recovered
        if svc.intelligence is not None:
            # plans follow their tasks: after task recovery, bring every open plan up to date
            svc.intelligence.attach()
            report.plans = await svc.intelligence.recover()
        if report.unclean_previous_stop and previous is not None:
            self._report_unclean_stop(svc, previous, report.recovered)
        try:
            inventory = await asyncio.wait_for(svc.router.refresh(), timeout=10)
            report.models = [m.name for m in inventory]
        except Exception as exc:
            report.issues.append(f"Model discovery failed: {exc}.")
        ollama_url = self.config.models.ollama.base_url if self.config.models.ollama.enabled else None
        report.readiness = assess_models(svc.router, ollama_url)
        svc.extra["model_readiness"] = report.readiness
        self._ensure_briefing_schedule(svc)
        await svc.pool.start()
        use_monitoring = self.config.monitoring.enabled if monitoring is None else monitoring
        system = SystemMonitor(svc.metrics, svc.state, self.config.monitoring, bus=svc.bus, clock=svc.clock,
                               world=svc.world, health=svc.health)
        svc.extra["system_monitor"] = system
        try:
            await system.sample_once()
        except Exception as exc:
            report.issues.append(f"System metrics are unavailable ({exc}).")
        # the scheduler, heartbeat and maintenance run whether or not monitoring is enabled or an interface
        # is attached: scheduled work must not depend on either
        self._periodic("scheduler", self.config.scheduler.tick_s, self._schedule_tick)
        self._periodic("runtime", self.config.runtime.heartbeat_s, self._heartbeat)
        self._periodic("maintenance", 6 * 3600.0, self._maintenance)
        if svc.intelligence is not None:
            self._periodic("planner", max(0.05, self.config.scheduler.tick_s), svc.intelligence.tick)
        if use_monitoring:
            svc.monitoring = MonitoringService(
                self.config.monitoring, system=system,
                network=NetworkMonitor(svc.state, self.config.monitoring, bus=svc.bus, clock=svc.clock,
                                       probe=self._network_probe),
                models=ModelMonitor(svc.router, svc.state, world=svc.world),
                self_monitor=SelfMonitor(svc.db, svc.bus, svc.health), health=svc.health)
            await svc.monitoring.start()
        for comp in svc.health.unhealthy():
            if comp.critical:
                report.issues.append(f"{comp.name} is {comp.status.label}: {comp.detail}.")
        svc.bus.emit(Event(EventType.SYSTEM_STARTED, "runtime",
                           {"models": len(report.models), "recovered": len(report.recovered),
                            "simulated": self.simulated, "run": self.run_id, "mode": self.mode,
                            "pid": os.getpid()}))
        await svc.bus.drain()
        self.started = True
        report.duration_s = round(self.clock.monotonic() - started, 3)
        log.info("started", duration_s=report.duration_s, models=len(report.models), mode=self.mode,
                 run=self.run_id)
        return report

    async def _start_passive(self, svc: Services, report: StartupReport, started: float) -> StartupReport:
        svc.extra["run"] = {"id": None, "pid": os.getpid(), "mode": "passive", "started_at": self.started_at,
                            "previous": self._previous_run, "unclean_previous_stop": False}
        svc.health.report("database", HealthStatus.HEALTHY if svc.db.healthy() else HealthStatus.CRITICAL)
        try:
            inventory = await asyncio.wait_for(svc.router.refresh(), timeout=10)
            report.models = [m.name for m in inventory]
        except Exception as exc:
            report.issues.append(f"Model discovery failed: {exc}.")
        ollama_url = self.config.models.ollama.base_url if self.config.models.ollama.enabled else None
        report.readiness = assess_models(svc.router, ollama_url)
        svc.extra["model_readiness"] = report.readiness
        try:
            await SystemMonitor(svc.metrics, svc.state, self.config.monitoring, clock=svc.clock).sample_once()
        except Exception as exc:
            report.issues.append(f"System metrics are unavailable ({exc}).")
        self.started = True
        report.duration_s = round(self.clock.monotonic() - started, 3)
        return report

    def _recover(self, svc: Services) -> list[RecoveryReport]:
        def idempotent(tool: str | None) -> bool:
            spec = svc.registry.get(tool) if tool else None
            return bool(spec and spec.spec.idempotent)

        def safe_to_repeat(step: Any) -> bool:
            """Idempotent tools, and calls that only observe (e.g. `git status`), may run again."""
            tool = svc.registry.get(step.tool) if step.tool else None
            if tool is None:
                return False
            if tool.spec.idempotent:
                return True
            try:
                return tool.assess(step.args).level == PermissionLevel.OBSERVE
            except Exception:
                return False

        def automation_state(automation_id: str) -> bool | None:
            auto = next((a for a in svc.automations.list() if a.id == automation_id), None)
            return None if auto is None else auto.enabled

        return svc.tasks.recover_interrupted(idempotent, safe_to_repeat=safe_to_repeat,
                                             max_age_s=self.config.runtime.recovery_max_age_s,
                                             automation_state=automation_state, audit=svc.audit)

    def _report_unclean_stop(self, svc: Services, previous: dict[str, Any],
                             recovered: list[RecoveryReport]) -> None:
        last = previous.get("heartbeat_at") or previous.get("started_at")
        summary = f"The previous run stopped without shutting down; it was last seen alive at " \
                  f"{format_datetime(last)}."
        if recovered:
            resumed = [r for r in recovered if r.resumed]
            held = [r for r in recovered if not r.resumed]
            if resumed:
                summary += f" Resumed: {', '.join(r.title for r in resumed[:3])}."
            if held:
                summary += f" Waiting for you: {', '.join(r.title for r in held[:3])}."
        else:
            summary += " No task was running."
        svc.audit.record(actor="system:runtime", action="unclean_stop_detected", summary=summary,
                         reason={"condition": "the previous run has no clean-stop record",
                                 "rule": "an unclean stop means in-flight work has unknown outcomes",
                                 "action": "recovered interrupted tasks and recorded the decision"})
        svc.bus.emit(Event(EventType.SYSTEM_RECOVERED, "runtime",
                           {"previous_run": previous.get("id"), "previous_pid": previous.get("pid"),
                            "last_heartbeat": last, "interrupted": len(recovered), "summary": summary},
                           severity=Severity.WARNING))

    def _periodic(self, name: str, interval: float, fn: Callable[[], Awaitable[Any]]) -> None:
        async def loop() -> None:
            while True:
                try:
                    await fn()
                    self._loop_ticks[name] = self.clock.now()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:   # a failing loop degrades, never kills, the runtime
                    log.error("loop_failed", loop=name, error=repr(exc))
                    if self.svc is not None and name in self.svc.health.components:
                        self.svc.health.report(name, HealthStatus.DEGRADED, f"{name} failed: {exc}")
                await asyncio.sleep(interval)
        self._loops.append(asyncio.create_task(loop(), name=f"runtime-{name}"))

    async def _schedule_tick(self) -> None:
        assert self.svc is not None
        await self.svc.automations.tick()
        upcoming = self.svc.automations.upcoming(limit=1)
        detail = f"next: {upcoming[0].name}" if upcoming else "nothing scheduled"
        self.svc.health.report("scheduler", HealthStatus.HEALTHY, detail)

    async def _briefing_action(self, action: dict[str, Any], auto: Any, event: Any) -> None:
        from jarvis.core.awareness import prepare_briefing
        assert self.svc is not None
        key = self.svc.automations.idempotency_key(auto, event)
        prepare_briefing(self.svc, kind="scheduled", briefing_id=f"brief:{key}")

    def _ensure_briefing_schedule(self, svc: Services) -> None:
        """Keep the system's morning-briefing schedule in line with the [briefing] configuration."""
        cfg = self.config.briefing
        days = [d.lower()[:3] for d in cfg.days]
        when = {"type": "daily", "at": cfg.time} if len(set(days)) == 7 else \
            {"type": "weekly", "days": days, "at": cfg.time}
        existing = svc.automations.find("Morning briefing", owner="system")
        if existing is not None and (not cfg.enabled or existing.schedule != validate_schedule(when)):
            if cfg.enabled:
                svc.automations.delete(existing.id)
                existing = None
            elif existing.enabled:
                svc.automations.set_enabled(existing.id, False)
        if cfg.enabled and existing is None:
            svc.automations.schedule("Morning briefing", when, {"type": "briefing"}, owner="system")
        elif cfg.enabled and existing is not None and not existing.enabled:
            svc.automations.set_enabled(existing.id, True)

    async def _resume_model_waiters(self) -> None:
        """Tasks that stopped because no language model was reachable continue once one is."""
        svc = self.svc
        if svc is None:
            return
        waiting = [t for t in svc.tasks.list_tasks([TaskStatus.WAITING]) if t.checkpoint.get("waiting_for") == "model"]
        if not waiting:
            return
        if not svc.router.available():
            try:
                await asyncio.wait_for(svc.router.refresh(), timeout=5)
            except Exception:
                return
        if not svc.router.available():
            return
        for task in waiting:
            task.checkpoint.pop("waiting_for", None)
            svc.tasks.transition(task, TaskStatus.QUEUED, "a language model is available again", by="system:models")

    async def _heartbeat(self) -> None:
        svc = self.svc
        assert svc is not None
        now = self.clock.now()
        svc.db.execute("UPDATE runtime_runs SET heartbeat_at=? WHERE id=?", (now, self.run_id))
        svc.health.report("runtime", HealthStatus.HEALTHY, f"up {format_duration(now - (self.started_at or now))}")
        if svc.monitoring is None:
            SelfMonitor(svc.db, svc.bus, svc.health).check_once()
        self._sample_self()
        if svc.presence is not None:
            svc.presence.expire()
            if self.track_presence and not svc.presence.away:
                # an interface is open and the user has been quiet for a while: a good moment for queued news
                # (the interrupt policy holds back anything below its threshold until now)
                svc.notifications.drain_if_idle(30.0)
        await self._resume_model_waiters()

    def _sample_self(self) -> None:
        """JARVIS's own cost: it is one of the processes competing for this machine."""
        svc = self.svc
        assert svc is not None
        try:
            with self._process.oneshot():
                cpu = self._process.cpu_percent(None)
                rss_mb = self._process.memory_info().rss / 2**20
                threads = self._process.num_threads()
        except (psutil.Error, OSError):
            return
        ttl = max(3 * self.config.runtime.heartbeat_s, 30.0)
        for key, value in (("cpu_percent", round(cpu, 1)), ("memory_mb", round(rss_mb, 1)), ("threads", threads),
                           ("asyncio_tasks", len(asyncio.all_tasks()))):
            svc.state.set(f"runtime.self.{key}", value, ttl=ttl, persist=False)
        limit = self.config.resources.self_memory_warn_mb
        if rss_mb >= limit and not self._self_memory_high:
            self._self_memory_high = True
            svc.bus.emit(Event(EventType.RESOURCE_THRESHOLD_EXCEEDED, "runtime",
                               {"name": "jarvis_memory_mb", "value": round(rss_mb), "threshold": limit,
                                "severity": "warning",
                                "message": f"JARVIS itself is using {rss_mb:.0f} MB of memory (limit {limit:.0f} MB)"},
                               severity=Severity.WARNING))
        elif rss_mb < limit * 0.9 and self._self_memory_high:
            self._self_memory_high = False
            svc.bus.emit(Event(EventType.RESOURCE_THRESHOLD_CLEARED, "runtime",
                               {"name": "jarvis_memory_mb", "value": round(rss_mb),
                                "message": f"JARVIS's own memory use is back to {rss_mb:.0f} MB"}))

    async def _maintenance(self) -> None:
        assert self.svc is not None
        now = self.svc.clock.now()
        pruned = self.svc.events.prune(now)
        expired = self.svc.memory.purge_expired()
        # conversation request records only need to outlive a client's retries
        requests = self.svc.db.execute("DELETE FROM requests WHERE status='done' AND created_at < ?",
                                       (now - 7 * 86400,))
        log.info("maintenance", events_pruned=pruned, memories_expired=expired, requests_pruned=requests)

    def loop_ticks(self) -> dict[str, float]:
        return dict(self._loop_ticks)

    def orchestrator(self, session_id: str = "default") -> Orchestrator:
        """The conversation for a session. One per session, restored from the conversation log, shared by
        every interface attached to that session."""
        assert self.svc is not None, "runtime not started"
        orch = self._orchestrators.get(session_id)
        if orch is None:
            orch = Orchestrator(self.svc, session_id=session_id)
            orch.restore_history()
            self._orchestrators[session_id] = orch
        return orch

    async def stop(self) -> None:
        if not self.svc or not self.started:
            self._release()
            return
        svc = self.svc
        if self.passive:
            await svc.bus.drain()
            await svc.router.close()
            svc.db.close()
            self.started = False
            self._release()
            return
        svc.bus.emit(Event(EventType.SYSTEM_STOPPING, "runtime", {}, severity=Severity.INFO))
        for task in self._loops:
            task.cancel()
        for task in self._loops:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._loops.clear()
        if svc.monitoring:
            await svc.monitoring.stop()
        if svc.intelligence is not None:
            svc.intelligence.engine.enabled = False
            await svc.intelligence.engine.settle(2.0)
        await svc.pool.stop()
        now = self.clock.now()
        svc.bus.emit(Event(EventType.SYSTEM_STOPPED, "runtime",
                           {"run": self.run_id, "uptime_s": round(now - (self.started_at or now), 1)}))
        await svc.bus.drain()
        try:
            svc.db.execute("UPDATE runtime_runs SET stopped_at=?, heartbeat_at=?, clean=1 WHERE id=?",
                           (now, now, self.run_id))
        except Exception as exc:
            log.error("run_record_failed", error=repr(exc))
        await svc.router.close()
        svc.db.close()
        self.started = False
        self._orchestrators.clear()
        self._release()
        log.info("stopped")

    def _release(self) -> None:
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    async def __aenter__(self) -> "Runtime":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()
