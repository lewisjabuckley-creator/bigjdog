"""The intelligence layer as one service (Phase 3 §47-49).

Wires goals, planning, the plan engine, replanning, agents, routing, autonomy, simulation and reactions onto
the existing runtime services, and gives the conversation, the local API and the CLI one place to ask:
understand this request, plan it, run it, advise, simulate, correct, explain, and report the intelligence
state (what is planned, running, waiting, which agents are working, the inference queue, autonomy, limits).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from jarvis.intelligence.agents import AgentCoordinator
from jarvis.intelligence.autonomy import Autonomy
from jarvis.intelligence.builder import BuildResult, PlanBuilder
from jarvis.intelligence.context import ContextPacker
from jarvis.intelligence.engine import PlanEngine
from jarvis.intelligence.goals import Complexity, ExecutionMode, Goal, GoalParser
from jarvis.intelligence.memory import PlanMemory
from jarvis.intelligence.plans import PLAN_OPEN, Plan, PlanStatus
from jarvis.intelligence.reactions import AttentionModel, ReactionEngine
from jarvis.intelligence.replanning import Replanner, parse_correction
from jarvis.intelligence.routing import RoutingPolicy
from jarvis.intelligence.simulation import Estimate, Simulator
from jarvis.intelligence.store import PlanStore
from jarvis.intelligence.tools import register_intelligence_tools


@dataclass
class Started:
    plan: Plan | None
    preview: bool = False
    problems: list[str] | None = None
    notes: list[str] | None = None


class IntelligenceService:
    def __init__(self, svc: Any, *, agents: Any = None, runner: Any = None) -> None:
        self.svc = svc
        cfg = svc.config.intelligence
        self.config = cfg
        self.parser = GoalParser(svc.clock)
        self.store = PlanStore(svc.db, svc.clock)
        self.memory = PlanMemory(self.store, svc.decisions, svc.memory, svc.clock)
        self.packer = ContextPacker()
        self.routing = RoutingPolicy(resources=svc.resources,
                                     local_only=lambda: svc.modes.effective().local_only)
        register_intelligence_tools(svc.registry, svc.router)
        self.builder = PlanBuilder(svc.registry, svc.router, svc.permissions, self.memory, config=cfg,
                                   projects=svc.projects, resources=svc.resources, parser=self.parser,
                                   packer=self.packer)
        self.coordinator = AgentCoordinator(agents, router=svc.router, resources=svc.resources, runner=runner,
                                            bus=svc.bus, packer=ContextPacker(max_tokens=1500),
                                            max_concurrent=cfg.agents_max_concurrent) if agents is not None else None
        if runner is not None:
            runner.profile_hook = self.routing.agent_profile
        svc.router.profile_hook = self.routing.adjust
        self.autonomy = Autonomy(svc.state, cfg, modes=svc.modes, bus=svc.bus, audit=svc.audit)
        self.engine = PlanEngine(store=self.store, tasks=svc.tasks, registry=svc.registry, permissions=svc.permissions,
                                 approvals=svc.approvals, audit=svc.audit, bus=svc.bus, builder=self.builder,
                                 memory=self.memory, clock=svc.clock, config=cfg, resources=svc.resources,
                                 router=svc.router, coordinator=self.coordinator,
                                 data_dir=str(svc.config.data_path), autonomy=lambda: self.autonomy.level.value)
        self.engine.replanner = Replanner(self.engine)
        self.simulator = Simulator(svc.registry, svc.tasks, router=svc.router, clock=svc.clock, owner=svc.user,
                                   data_dir=str(svc.config.data_path))
        self.attention = AttentionModel(svc.state, svc.modes, svc.clock)
        self.reactions = ReactionEngine(self, bus=svc.bus, notifications=svc.notifications, autonomy=self.autonomy,
                                        attention=self.attention, config=cfg, clock=svc.clock, audit=svc.audit)
        self.engine.enabled = cfg.enabled

    # -- lifecycle ---------------------------------------------------------------------------------------------
    def attach(self) -> None:
        self.engine.attach()
        if self.config.enabled:
            self.reactions.attach()

    async def recover(self) -> list[dict[str, Any]]:
        return await self.engine.recover()

    async def tick(self) -> None:
        await self.engine.tick()

    def home(self) -> str:
        return os.path.expanduser("~")

    # -- understanding ------------------------------------------------------------------------------------------
    def understand(self, text: str, *, dry_run: bool = False, context: dict[str, Any] | None = None) -> Goal:
        return self.parser.parse(text, dry_run=dry_run, context=context)

    def should_plan(self, goal: Goal) -> bool:
        """Complexity decides how much machinery a request gets. Simple requests keep the direct paths."""
        if not self.config.enabled:
            return False
        if goal.mode in (ExecutionMode.SIMULATE, ExecutionMode.PREDICT):
            return False
        if goal.kind in ("performance", "disk_cleanup", "backup", "research"):
            return True
        if goal.kind == "compound":
            # several questions in one sentence are still a conversation; a plan needs something to do
            from jarvis.intelligence.playbooks import actionable
            return goal.complexity >= Complexity.MODERATE and any(actionable(s.text) for s in goal.subgoals)
        return goal.complexity >= Complexity.COMPLEX

    # -- planning and running ------------------------------------------------------------------------------------
    async def plan_goal(self, goal: Goal, *, cwd: str, interactive: bool = True, created_by: str | None = None,
                        origin: str = "", session_id: str | None = None, project: Any = None) -> BuildResult:
        project = project if project is not None else self.svc.projects.active()
        return await self.builder.build(goal, cwd=cwd, interactive=interactive,
                                        created_by=created_by or f"user:{self.svc.user}", owner=self.svc.user,
                                        origin=origin, session_id=session_id, project=project,
                                        autonomy=self.autonomy.level.value)

    async def run(self, goal: Goal, *, cwd: str, session_id: str | None = None, origin: str = "",
                  interactive: bool = True, created_by: str | None = None) -> Started:
        built = await self.plan_goal(goal, cwd=cwd, interactive=interactive, created_by=created_by, origin=origin,
                                     session_id=session_id)
        if built.plan is None:
            return Started(None, problems=built.problems, notes=built.notes)
        preview = self.builder.needs_preview(built.plan, self.autonomy.level.value) and interactive
        plan = self.engine.submit(built.plan, preview=preview)
        if plan.status == PlanStatus.FAILED:
            return Started(plan, problems=[plan.status_reason])
        if not preview:
            plan = await self.engine.start(plan.id, by=created_by or self.svc.user) or plan
        return Started(plan, preview=preview, notes=built.notes)

    async def confirm(self, plan_id: str) -> Plan | None:
        return await self.engine.start(plan_id, by=self.svc.user)

    async def correct(self, text: str, plan: Plan) -> tuple[bool, str]:
        correction = parse_correction(text)
        if correction is None:
            return False, ""
        async with self.engine._lock(plan.id):
            fresh = self.store.get(plan.id)
            if fresh is None or fresh.terminal:
                return False, "that plan has already finished"
            fresh.corrections.append({"ts": self.svc.clock.now(), "text": text, "correction": correction})
            applied = await self.engine.replanner.replan(fresh, "correction", f"you said: {text}",
                                                         correction=correction)
            self.store.save(fresh)
        if applied:
            from jarvis.events.types import Event, EventType
            self.svc.bus.emit(Event(EventType.CORRECTION_APPLIED, "planner",
                                    {"plan_id": plan.id, "title": fresh.title, "correction": correction,
                                     "changed": fresh.replans[-1]["changed"]}, entity_id=f"plan:{plan.id}"))
            self.engine.schedule(plan.id)
            return True, "; ".join(fresh.replans[-1]["changed"])
        return False, "that doesn't change anything in the current plan"

    async def simulate(self, text: str, *, cwd: str | None = None) -> Estimate:
        return await self.simulator.simulate(text, cwd=cwd)

    def predict(self, text: str) -> Estimate:
        project = self.svc.projects.active()
        return self.simulator.predict(text, project_id=project.id if project else None)

    # -- reading ---------------------------------------------------------------------------------------------------
    def get(self, plan_id: str) -> Plan | None:
        return self.store.get(plan_id)

    def open_plans(self, session_id: str | None = None) -> list[Plan]:
        plans = self.store.list(PLAN_OPEN, limit=50)
        return [p for p in plans if session_id is None or p.session_id in (session_id, None)]

    def recent(self, limit: int = 10) -> list[Plan]:
        return self.store.list(limit=limit)

    def latest(self, *, statuses: Any = None, session_id: str | None = None) -> Plan | None:
        plans = self.store.list(statuses, limit=20)
        for plan in plans:
            if session_id is None or plan.session_id in (session_id, None):
                return plan
        return None

    def plan_for_task(self, task_id: str) -> Plan | None:
        task = self.svc.tasks.get_task(task_id)
        plan_id = task.outputs.get("plan_id") if task else None
        return self.store.get(plan_id) if plan_id else None

    def state(self) -> dict[str, Any]:
        """Intelligence state (Phase 3 §48)."""
        open_plans = self.store.open_plans()
        recent = self.store.list(limit=10)
        router_status = self.svc.router.status()
        return {
            "enabled": self.config.enabled,
            "autonomy": {"level": self.autonomy.level.value, "reactions": self.autonomy.reactions(),
                         "description": self.autonomy.describe()},
            "plans": {"open": [p.to_api() for p in open_plans],
                      "recent": [{"id": p.id, "title": p.title, "status": p.status.value,
                                  "quality": p.quality.value if p.quality else None,
                                  "updated_at": p.updated_at} for p in recent]},
            "waiting_for_you": [{"plan_id": p.id, "title": p.title, "reason": p.status_reason}
                                for p in open_plans if p.status in (PlanStatus.WAITING, PlanStatus.BLOCKED,
                                                                    PlanStatus.PAUSED) or p.awaiting_confirmation],
            "agents": {"running": self.coordinator.status() if self.coordinator else [],
                       "contracts": self.coordinator.contracts() if self.coordinator else []},
            "inference": router_status.get("inference"),
            "active_model_requests": router_status.get("active_requests", []),
            "limits": {"max_parallel_nodes": self.config.max_parallel_nodes, "max_replans": self.config.max_replans,
                       "max_node_attempts": self.config.max_node_attempts,
                       "max_model_calls": self.config.max_model_calls, "max_nodes": self.config.max_nodes},
            "resources": {"pressure": self.svc.resources.pressure()[1] or None},
        }
