"""Hardware abstraction: STATE / COMMAND / RESULT / TELEMETRY (spec §123-125, §186).

The core does not care whether a device is physical or simulated. Every
controllable device exposes state and telemetry; a command is never assumed to
have worked because it was sent — the resulting telemetry is read back and
compared with the expected state.

No physical integrations ship yet. This module provides the contract, a
registry, a permission-checked command tool, and a simulated device used in
tests and simulation mode.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from jarvis.clock import Clock, SystemClock
from jarvis.core.types import Provenance, ProvenanceKind, RiskLevel, Severity
from jarvis.events.bus import EventBus
from jarvis.events.types import Event, EventType
from jarvis.permissions.model import PermissionLevel
from jarvis.tools.base import Tool, ToolContext, ToolResult, ToolSpec, Verification


@dataclass
class CommandResult:
    accepted: bool
    detail: str = ""
    expected: dict[str, Any] = field(default_factory=dict)   # state the command should produce


class Device(ABC):
    id: str
    name: str
    kind: str                       # sensor | actuator | robot | printer | smart_home | camera | ...
    simulated: bool = False

    @abstractmethod
    async def state(self) -> dict[str, Any]: ...

    @abstractmethod
    async def telemetry(self) -> dict[str, Any]: ...

    @abstractmethod
    async def command(self, name: str, args: dict[str, Any]) -> CommandResult: ...

    def commands(self) -> dict[str, dict[str, Any]]:
        """Supported command names with their risk level (e.g. {"set_speed": {"risk": "low"}})."""
        return {}

    @property
    def connected(self) -> bool:
        return True


class DeviceRegistry:
    def __init__(self, *, bus: EventBus | None = None, clock: Clock | None = None) -> None:
        self.bus = bus
        self.clock = clock or SystemClock()
        self.devices: dict[str, Device] = {}

    def register(self, device: Device) -> None:
        self.devices[device.id] = device
        self._emit(EventType.DEVICE_CONNECTED, device, {})

    def unregister(self, device_id: str, *, failed: bool = False) -> None:
        device = self.devices.pop(device_id, None)
        if device:
            self._emit(EventType.DEVICE_FAILURE if failed else EventType.DEVICE_DISCONNECTED, device, {},
                       Severity.WARNING)

    def get(self, device_id: str) -> Device | None:
        return self.devices.get(device_id) or next((d for d in self.devices.values()
                                                    if d.name.lower() == device_id.lower()), None)

    def list(self) -> list[Device]:
        return list(self.devices.values())

    def _emit(self, etype: EventType, device: Device, payload: dict[str, Any], severity: Severity = Severity.INFO) -> None:
        if self.bus:
            self.bus.emit(Event(etype, "devices", {"device": device.id, "name": device.name, "kind": device.kind,
                                                   "simulated": device.simulated, **payload},
                                severity=severity, entity_id=f"device:{device.id}"))


class DeviceCommandTool(Tool):
    spec = ToolSpec(
        name="device_command",
        description="Send a command to a registered device and verify the result from its telemetry.",
        parameters={"type": "object", "properties": {
            "device": {"type": "string"}, "command": {"type": "string"},
            "args": {"type": "object", "default": {}},
            "settle_s": {"type": "number", "default": 0.5, "minimum": 0}},
            "required": ["device", "command"]},
        level=PermissionLevel.EXECUTE_CONSEQUENTIAL, risk=RiskLevel.HIGH, side_effects=("physical",),
        reversible=False, verification="telemetry read back and compared with the expected state",
        category="devices",
    )

    def __init__(self, registry: DeviceRegistry) -> None:
        self.registry = registry

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        device = self.registry.get(args["device"])
        if device is None:
            return ToolResult(False, f"no device named {args['device']!r}", error="not_found")
        if not device.connected:
            return ToolResult(False, f"{device.name} is not connected", error="disconnected")
        result = await device.command(args["command"], args.get("args") or {})
        if not result.accepted:
            return ToolResult(False, f"{device.name} rejected {args['command']}: {result.detail}", error="rejected")
        return ToolResult(True, f"sent {args['command']} to {device.name}",
                          {"device": device.id, "expected": result.expected, "detail": result.detail},
                          provenance=Provenance(ProvenanceKind.TOOL_OUTPUT, f"device:{device.id}"))

    async def verify(self, args: dict[str, Any], result: ToolResult, ctx: ToolContext) -> Verification:
        if not result.ok:
            return Verification.not_performed("command not accepted")
        device = self.registry.get(args["device"])
        expected = (result.data or {}).get("expected") or {}
        if device is None or not expected:
            return Verification(False, None, "telemetry", "no expected state declared; outcome unverified")
        await asyncio.sleep(args.get("settle_s", 0.5))
        telemetry = await device.telemetry()
        mismatched = {k: (v, telemetry.get(k)) for k, v in expected.items() if telemetry.get(k) != v}
        if mismatched:
            detail = ", ".join(f"{k}: expected {e}, telemetry shows {a}" for k, (e, a) in mismatched.items())
            return Verification(True, False, "telemetry", detail)
        return Verification(True, True, "telemetry", "telemetry matches the expected state")


class SimulatedFan(Device):
    """A simulated actuator. ``stuck=True`` makes it accept commands without acting (a lying device)."""

    kind = "actuator"
    simulated = True

    def __init__(self, device_id: str = "sim-fan", name: str = "lab fan") -> None:
        self.id = device_id
        self.name = name
        self.speed = 0
        self.stuck = False
        self.temperature_c = 30.0

    async def state(self) -> dict[str, Any]:
        return {"speed": self.speed}

    async def telemetry(self) -> dict[str, Any]:
        return {"speed": self.speed, "temperature_c": round(self.temperature_c - self.speed * 0.05, 1)}

    def commands(self) -> dict[str, dict[str, Any]]:
        return {"set_speed": {"risk": "low", "args": {"speed": "0-100"}}}

    async def command(self, name: str, args: dict[str, Any]) -> CommandResult:
        if name != "set_speed":
            return CommandResult(False, f"unsupported command {name}")
        speed = int(args.get("speed", 0))
        if not 0 <= speed <= 100:
            return CommandResult(False, "speed must be 0-100")
        if not self.stuck:
            self.speed = speed
        return CommandResult(True, "ok", {"speed": speed})
