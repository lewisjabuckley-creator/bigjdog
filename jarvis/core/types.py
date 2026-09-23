"""Shared vocabulary used across every JARVIS subsystem.

These enums are deliberately small and explicit. Internal components exchange
structured values built from them; natural language is produced only at the
edge (the personality/response layer).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any


def new_id(prefix: str) -> str:
    """Short, prefixed, collision-resistant identifier (e.g. ``task-3f9a2c1b7d``)."""
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


class Severity(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class Confidence(StrEnum):
    """How JARVIS knows something (spec §34). Preserved internally, exposed when useful."""

    KNOWN = "known"          # configured / definitional
    OBSERVED = "observed"    # measured directly from live state or a tool
    RETRIEVED = "retrieved"  # read back from memory / database
    INFERRED = "inferred"    # reasoned from observations
    ESTIMATED = "estimated"  # projection / prediction
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"


class ProvenanceKind(StrEnum):
    """Where a piece of information came from (spec §35)."""

    SYSTEM_STATE = "system_state"
    LOCAL_FILE = "local_file"
    USER_STATEMENT = "user_statement"
    MEMORY = "memory"
    DATABASE = "database"
    TOOL_OUTPUT = "tool_output"
    WEB = "web"
    EXTERNAL_API = "external_api"
    MODEL_KNOWLEDGE = "model_knowledge"
    INFERENCE = "inference"
    CONFIGURATION = "configuration"


@dataclass(frozen=True)
class Provenance:
    kind: ProvenanceKind
    source: str = ""          # e.g. tool name, file path, memory id
    detail: str = ""

    def describe(self) -> str:
        parts = [self.kind.value.replace("_", " ")]
        if self.source:
            parts.append(self.source)
        if self.detail:
            parts.append(f"({self.detail})")
        return " ".join(parts)

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "source": self.source, "detail": self.detail}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Provenance":
        return cls(ProvenanceKind(data["kind"]), data.get("source", ""), data.get("detail", ""))


@dataclass
class Fact:
    """A value together with how and when JARVIS came to believe it."""

    value: Any
    confidence: Confidence
    provenance: Provenance
    observed_at: float
    ttl: float | None = None  # seconds before the value should be considered stale

    def is_stale(self, now: float) -> bool:
        return self.ttl is not None and (now - self.observed_at) > self.ttl

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "confidence": self.confidence.value,
            "provenance": self.provenance.to_dict(),
            "observed_at": self.observed_at,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fact":
        return cls(
            value=data["value"],
            confidence=Confidence(data["confidence"]),
            provenance=Provenance.from_dict(data["provenance"]),
            observed_at=float(data["observed_at"]),
            ttl=data.get("ttl"),
        )


class Outcome(StrEnum):
    """Result quality of an operation (spec §148). Never flatten partial into success."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class HealthStatus(IntEnum):
    """Ordered so that ``max()`` yields the worst status."""

    HEALTHY = 0
    UNKNOWN = 1
    DEGRADED = 2
    WARNING = 3
    CRITICAL = 4
    OFFLINE = 5

    @property
    def label(self) -> str:
        return self.name.lower()


class NetworkState(StrEnum):
    ONLINE = "online"
    OFFLINE = "offline"
    LIMITED = "limited"
    HIGH_LATENCY = "high_latency"
    UNSTABLE = "unstable"
    PRIVATE_NETWORK = "private_network"
    UNKNOWN = "unknown"


class Priority(IntEnum):
    """Work priority for resource allocation (spec §30). Lower value = more important."""

    P0 = 0  # critical
    P1 = 1  # user-active
    P2 = 2  # important background
    P3 = 3  # routine automation
    P4 = 4  # opportunistic


class NotificationPriority(IntEnum):
    """Interrupt policy classes (spec §11). Higher value = more interrupting."""

    DEBUG = 0
    INFORMATIONAL = 1
    IMPORTANT = 2
    URGENT = 3
    CRITICAL = 4


class RiskLevel(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class OperationalReason:
    """Explanation of an autonomous decision without exposing chain-of-thought (spec §49)."""

    condition: str            # observed condition
    rule: str                 # rule / goal / policy applied
    action: str               # action taken
    expected: str = ""        # expected consequence
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "rule": self.rule,
            "action": self.action,
            "expected": self.expected,
            **({"extra": self.extra} if self.extra else {}),
        }

    def sentence(self) -> str:
        text = f"I {self.action} because {self.condition.rstrip('.')}"
        if self.rule:
            text += f" ({self.rule})"
        text += "."
        if self.expected:
            expected = self.expected.strip()
            text += f" {expected}" + ("" if expected.endswith((".", "!", "?")) else ".")
        return text
