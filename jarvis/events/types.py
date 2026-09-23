"""Structured events: the nervous system of JARVIS (spec §9, §171)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from jarvis.core.types import Severity, new_id


class EventType(StrEnum):
    # lifecycle
    SYSTEM_STARTED = "SYSTEM_STARTED"
    SYSTEM_STOPPING = "SYSTEM_STOPPING"
    SUBSYSTEM_DEGRADED = "SUBSYSTEM_DEGRADED"
    SUBSYSTEM_RECOVERED = "SUBSYSTEM_RECOVERED"
    HEALTH_CHANGED = "HEALTH_CHANGED"
    MODE_CHANGED = "MODE_CHANGED"
    EMERGENCY_ENTERED = "EMERGENCY_ENTERED"
    EMERGENCY_EXITED = "EMERGENCY_EXITED"
    # models
    MODEL_LOADED = "MODEL_LOADED"
    MODEL_UNLOADED = "MODEL_UNLOADED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_RECOVERED = "MODEL_RECOVERED"
    MODEL_FALLBACK = "MODEL_FALLBACK"
    MODEL_RESPONSE_READY = "MODEL_RESPONSE_READY"
    # filesystem / processes / devices / network
    FILE_CREATED = "FILE_CREATED"
    FILE_CHANGED = "FILE_CHANGED"
    FILE_DELETED = "FILE_DELETED"
    PROCESS_STARTED = "PROCESS_STARTED"
    PROCESS_CRASHED = "PROCESS_CRASHED"
    PROCESS_FINISHED = "PROCESS_FINISHED"
    DEVICE_CONNECTED = "DEVICE_CONNECTED"
    DEVICE_DISCONNECTED = "DEVICE_DISCONNECTED"
    DEVICE_FAILURE = "DEVICE_FAILURE"
    TELEMETRY = "TELEMETRY"
    NETWORK_CHANGED = "NETWORK_CHANGED"
    # resources
    SYSTEM_METRICS = "SYSTEM_METRICS"
    RESOURCE_THRESHOLD_EXCEEDED = "RESOURCE_THRESHOLD_EXCEEDED"
    RESOURCE_THRESHOLD_CLEARED = "RESOURCE_THRESHOLD_CLEARED"
    TREND_DETECTED = "TREND_DETECTED"
    PREDICTIVE_WARNING = "PREDICTIVE_WARNING"
    RESOURCE_THROTTLED = "RESOURCE_THROTTLED"
    # tasks
    TASK_CREATED = "TASK_CREATED"
    TASK_STARTED = "TASK_STARTED"
    TASK_PROGRESS = "TASK_PROGRESS"
    TASK_WAITING = "TASK_WAITING"
    TASK_BLOCKED = "TASK_BLOCKED"
    TASK_PAUSED = "TASK_PAUSED"
    TASK_RESUMED = "TASK_RESUMED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    TASK_CANCELLED = "TASK_CANCELLED"
    TASK_INTERRUPTED = "TASK_INTERRUPTED"
    TASK_STATUS_CHANGED = "TASK_STATUS_CHANGED"
    DEADLINE_AT_RISK = "DEADLINE_AT_RISK"
    MONITOR_TRIGGERED = "MONITOR_TRIGGERED"
    # tools / permissions
    TOOL_EXECUTED = "TOOL_EXECUTED"
    TOOL_FAILED = "TOOL_FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    APPROVAL_REQUESTED = "APPROVAL_REQUESTED"
    APPROVAL_DECIDED = "APPROVAL_DECIDED"
    PERMISSION_GRANTED = "PERMISSION_GRANTED"
    PERMISSION_REVOKED = "PERMISSION_REVOKED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    SECURITY_EVENT = "SECURITY_EVENT"
    ERROR_DETECTED = "ERROR_DETECTED"
    # user / conversation
    USER_MESSAGE = "USER_MESSAGE"
    USER_SPEECH = "USER_SPEECH"
    USER_INTERRUPTED = "USER_INTERRUPTED"
    PROJECT_CHANGED = "PROJECT_CHANGED"
    MEMORY_STORED = "MEMORY_STORED"
    MEMORY_FORGOTTEN = "MEMORY_FORGOTTEN"
    # automation / communications
    TIMER_EXPIRED = "TIMER_EXPIRED"
    AUTOMATION_TRIGGERED = "AUTOMATION_TRIGGERED"
    CALL_RECEIVED = "CALL_RECEIVED"
    MESSAGE_RECEIVED = "MESSAGE_RECEIVED"
    BUILD_COMPLETED = "BUILD_COMPLETED"
    DEPLOYMENT_COMPLETED = "DEPLOYMENT_COMPLETED"
    TEST_FAILED = "TEST_FAILED"
    NOTIFICATION = "NOTIFICATION"


# High-frequency events that are useful live but not worth persisting by default.
NOISY_EVENTS = frozenset({EventType.SYSTEM_METRICS, EventType.TELEMETRY, EventType.TASK_PROGRESS,
                          EventType.MODEL_RESPONSE_READY})


@dataclass
class Event:
    type: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.INFO
    entity_id: str | None = None
    task_id: str | None = None
    ts: float = 0.0
    id: str = field(default_factory=lambda: new_id("evt"))
    persist: bool | None = None  # None = decide by type/severity

    def should_persist(self) -> bool:
        if self.persist is not None:
            return self.persist
        return self.severity >= Severity.INFO and self.type not in NOISY_EVENTS

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": str(self.type),
            "source": self.source,
            "severity": self.severity.name.lower(),
            "entity_id": self.entity_id,
            "task_id": self.task_id,
            "ts": self.ts,
            "payload": self.payload,
        }
