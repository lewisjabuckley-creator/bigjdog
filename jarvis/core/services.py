"""The container of wired subsystems handed to the orchestrator and interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jarvis.audit.log import AuditLog
    from jarvis.automation.engine import AutomationEngine
    from jarvis.clock import Clock
    from jarvis.config import JarvisConfig
    from jarvis.core.emergency import EmergencyController
    from jarvis.core.modes import ModeManager
    from jarvis.core.presence import Presence
    from jarvis.database.db import Database
    from jarvis.devices.registry import DeviceRegistry
    from jarvis.events.bus import EventBus
    from jarvis.events.store import EventStore
    from jarvis.intelligence.service import IntelligenceService
    from jarvis.memory.decisions import DecisionLog
    from jarvis.memory.store import MemoryStore
    from jarvis.models.router import ModelRouter
    from jarvis.monitoring.metrics import MetricsSource
    from jarvis.monitoring.service import MonitoringService
    from jarvis.notifications.manager import NotificationManager
    from jarvis.permissions.manager import ApprovalManager, PermissionManager
    from jarvis.projects.manager import ProjectManager
    from jarvis.state.engine import StateEngine
    from jarvis.state.health import HealthRegistry
    from jarvis.tasks.manager import TaskManager
    from jarvis.tasks.resources import ResourceManager
    from jarvis.tasks.workers import WorkerPool
    from jarvis.tools.registry import ToolRegistry
    from jarvis.world.model import WorldModel


@dataclass
class Services:
    config: "JarvisConfig"
    clock: "Clock"
    db: "Database"
    bus: "EventBus"
    events: "EventStore"
    state: "StateEngine"
    world: "WorldModel"
    health: "HealthRegistry"
    permissions: "PermissionManager"
    approvals: "ApprovalManager"
    audit: "AuditLog"
    registry: "ToolRegistry"
    router: "ModelRouter"
    tasks: "TaskManager"
    resources: "ResourceManager"
    pool: "WorkerPool"
    memory: "MemoryStore"
    decisions: "DecisionLog"
    projects: "ProjectManager"
    modes: "ModeManager"
    notifications: "NotificationManager"
    emergency: "EmergencyController"
    automations: "AutomationEngine"
    devices: "DeviceRegistry"
    metrics: "MetricsSource"
    monitoring: "MonitoringService | None" = None
    presence: "Presence | None" = None
    intelligence: "IntelligenceService | None" = None
    user: str = "owner"
    simulated: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
