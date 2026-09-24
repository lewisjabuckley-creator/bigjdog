"""Typed configuration.

Everything important is configurable (spec §158). Configuration is loaded from a
TOML file (``jarvis.toml`` in the working directory, ``$JARVIS_CONFIG``, or
``~/.jarvis/jarvis.toml``) and merged over safe defaults. Unknown keys are an
error: a silently ignored typo in a permission setting is a reliability bug.

Secrets never live here (spec §159). Adapters reference environment variable
*names* (e.g. ``api_key_env``) and resolve them through :mod:`jarvis.security`.
"""

from __future__ import annotations

import dataclasses
import os
import re
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class ConfigError(ValueError):
    pass


@dataclass
class GeneralConfig:
    data_dir: str = "~/.jarvis"
    user: str = "owner"
    default_mode: str = "normal"


@dataclass
class OllamaConfig:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:11434"
    keep_alive: str = "5m"
    request_timeout_s: float = 300.0
    # Context window requested from Ollama. Its default is small; the system prompt, live state, memory and
    # tool definitions need room, and Ollama silently drops the start of an over-long prompt.
    num_ctx: int = 8192
    # Extra sampling options passed to every request, e.g. {temperature = 0.2, seed = 7}.
    options: dict[str, float] = field(default_factory=dict)


@dataclass
class OpenAICompatibleConfig:
    """Optional adapter for any OpenAI-compatible endpoint (llama.cpp server, vLLM, LM Studio, cloud)."""

    enabled: bool = False
    name: str = "openai_compatible"
    base_url: str = ""
    api_key_env: str = ""
    local: bool = False  # true when the endpoint runs on this machine / private network
    models: list[str] = field(default_factory=list)
    request_timeout_s: float = 120.0


def _default_profiles() -> dict[str, list[str]]:
    # Ordered preferences per purpose. Missing models are skipped; if none match,
    # the router auto-selects any installed model with the required capability.
    return {
        "conversation": ["llama3.1:8b", "qwen2.5:7b", "llama3.2:3b"],
        "reasoning": ["qwen2.5:32b", "qwen2.5:14b", "llama3.1:8b"],
        "planning": ["qwen2.5:14b", "llama3.1:8b"],
        "coding": ["qwen2.5-coder:14b", "qwen2.5-coder:7b", "llama3.1:8b"],
        "vision": ["llama3.2-vision", "llava:13b", "llava"],
        "summarization": ["llama3.2:3b", "llama3.1:8b"],
        "classification": ["llama3.2:3b", "llama3.1:8b"],
        "background": ["llama3.2:3b", "llama3.1:8b"],
        "embedding": ["nomic-embed-text", "mxbai-embed-large"],
    }


@dataclass
class ModelsConfig:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openai_compatible: OpenAICompatibleConfig = field(default_factory=OpenAICompatibleConfig)
    profiles: dict[str, list[str]] = field(default_factory=_default_profiles)
    failure_threshold: int = 3        # consecutive failures before a model is marked unhealthy
    unhealthy_cooldown_s: float = 30.0


def _default_denied_paths() -> list[str]:
    return ["~/.ssh", "~/.gnupg", "~/.aws", "~/.azure", "~/.kube", "~/.config/gcloud", "~/.jarvis/secrets"]


@dataclass
class PermissionsConfig:
    # Highest level auto-approved for actions that directly serve an explicit user
    # request (spec §145). Level 4 (consequential) always needs approval or a grant.
    interactive_level: int = 3
    # Baseline for automations and background agents without explicit grants.
    automation_level: int = 1
    allowed_roots: list[str] = field(default_factory=lambda: ["~"])
    denied_paths: list[str] = field(default_factory=_default_denied_paths)
    approval_timeout_s: float = 3600.0


@dataclass
class TasksConfig:
    max_concurrent: int = 4
    default_step_timeout_s: float = 1800.0
    max_retries: int = 2
    scheduler_interval_s: float = 0.25


@dataclass
class Threshold:
    value: float
    sustain_s: float = 0.0
    severity: str = "warning"  # warning | critical
    clear_below: float | None = None  # hysteresis; defaults to value


def _default_thresholds() -> dict[str, Threshold]:
    return {
        "cpu_percent": Threshold(90.0, sustain_s=300.0),
        "memory_percent": Threshold(90.0, sustain_s=120.0),
        "disk_percent": Threshold(90.0),
        "disk_critical_percent": Threshold(97.0, severity="critical"),
        "gpu_temp_c": Threshold(85.0, sustain_s=60.0),
        "cpu_temp_c": Threshold(90.0, sustain_s=60.0),
        "battery_low_percent": Threshold(15.0),
    }


@dataclass
class MonitoringConfig:
    enabled: bool = True
    system_interval_s: float = 10.0
    model_interval_s: float = 30.0
    network_interval_s: float = 30.0
    network_probe_host: str = "1.1.1.1"
    network_probe_port: int = 443
    network_high_latency_ms: float = 400.0
    thresholds: dict[str, Threshold] = field(default_factory=_default_thresholds)
    trend_window: int = 360  # samples retained per metric


@dataclass
class NotificationsConfig:
    max_interrupts_per_window: int = 5
    interrupt_window_s: float = 600.0
    dedupe_window_s: float = 1200.0
    user_active_window_s: float = 20.0  # recent input => user considered busy typing


@dataclass
class MemoryConfig:
    enabled: bool = True
    use_embeddings: bool = True
    max_context_items: int = 6


@dataclass
class EventsConfig:
    retention_days_debug: float = 1.0
    retention_days_info: float = 30.0
    retention_days_warning: float = 365.0
    recent_buffer: int = 500


@dataclass
class PrivacyConfig:
    private_mode: bool = False
    allow_cloud: bool = False  # optional cloud adapters are opt-in


@dataclass
class UIConfig:
    verbosity: str = "normal"  # short | normal | detailed
    use_sir: bool = False
    humor: bool = True


@dataclass
class RuntimeConfig:
    """The persistent runtime and its local API (docs/RUNTIME.md)."""
    api_host: str = "127.0.0.1"         # loopback only: the API is never exposed to the network
    api_port: int = 0                   # 0 = any free port; the chosen port is written to runtime.json
    heartbeat_s: float = 10.0
    presence_timeout_s: float = 30.0    # an interface silent for this long counts as closed
    auto_start: bool = True             # the CLI starts the runtime in the background when it isn't running
    recovery_max_age_s: float = 86400.0 # interrupted work older than this waits for the user instead of resuming


@dataclass
class SchedulerConfig:
    tick_s: float = 5.0
    catch_up_window_s: float = 21600.0  # a run missed while JARVIS was down is made up if it is this recent


@dataclass
class ResourcesConfig:
    memory_critical: float = 92.0       # percent; above this only P0-P2 work starts
    cpu_critical: float = 97.0
    vram_critical: float = 90.0
    # what happens to running P3+ work under pressure: pause (checkpoint, resume automatically), slow (keep
    # running but start nothing new below P2), wait (finish the current step, then pause), continue
    low_priority_policy: str = "pause"
    self_memory_warn_mb: float = 1500.0 # JARVIS's own resident memory that is worth a warning


@dataclass
class BriefingConfig:
    enabled: bool = False               # prepare a morning briefing on a schedule
    time: str = "07:30"
    days: list[str] = field(default_factory=lambda: ["mon", "tue", "wed", "thu", "fri", "sat", "sun"])


@dataclass
class JarvisConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    tasks: TasksConfig = field(default_factory=TasksConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    briefing: BriefingConfig = field(default_factory=BriefingConfig)
    source: str = "defaults"

    @property
    def data_path(self) -> Path:
        return Path(os.path.expanduser(self.general.data_dir))

    @property
    def db_path(self) -> Path:
        return self.data_path / "jarvis.db"


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a table")
        return _build(tp, value, where)
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a table")
        return {str(k): _coerce(args[1], v, f"{where}.{k}") for k, v in value.items()}
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list")
        return [_coerce(args[0], v, f"{where}[]") for v in value]
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        if value is None and type(None) in args:
            return None
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(arg, value, where)
            except ConfigError:
                continue
        raise ConfigError(f"{where}: invalid value {value!r}")
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string")
        return value
    return value


def _build(cls: Any, data: dict[str, Any], where: str) -> Any:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{where or 'config'}: unknown key(s): {', '.join(sorted(unknown))}")
    kwargs = {}
    for name, value in data.items():
        kwargs[name] = _coerce(hints[name], value, f"{where}.{name}" if where else name)
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{where}: {exc}") from exc


def _merge(defaults: Any, data: dict[str, Any], where: str = "") -> Any:
    """Overlay ``data`` onto an existing dataclass instance (so partial tables keep defaults)."""
    hints = typing.get_type_hints(type(defaults))
    known = {f.name for f in dataclasses.fields(defaults)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{where or 'config'}: unknown key(s): {', '.join(sorted(unknown))}")
    for name, value in data.items():
        path = f"{where}.{name}" if where else name
        current = getattr(defaults, name)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, path)
        elif isinstance(current, dict) and isinstance(value, dict) and name == "thresholds":
            merged = dict(current)
            for key, raw in value.items():
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    raw = {"value": raw}
                base = dataclasses.asdict(merged[key]) if key in merged else {}
                base.update(raw)
                merged[key] = _build(Threshold, base, f"{path}.{key}")
            setattr(defaults, name, merged)
        elif isinstance(current, dict) and isinstance(value, dict):
            merged = dict(current)
            merged.update(_coerce(hints[name], value, path))
            setattr(defaults, name, merged)
        else:
            setattr(defaults, name, _coerce(hints[name], value, path))
    return defaults


_ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "JARVIS_DATA_DIR": ("general", "data_dir"),
    "JARVIS_OLLAMA_URL": ("models", "ollama", "base_url"),
    "JARVIS_PRIVATE_MODE": ("privacy", "private_mode"),
}


def _apply_env(cfg: JarvisConfig, env: dict[str, str]) -> None:
    for var, path in _ENV_OVERRIDES.items():
        if var not in env:
            continue
        target: Any = cfg
        for part in path[:-1]:
            target = getattr(target, part)
        current = getattr(target, path[-1])
        raw = env[var]
        value: Any = raw.lower() in ("1", "true", "yes", "on") if isinstance(current, bool) else raw
        setattr(target, path[-1], value)


def config_from_dict(data: dict[str, Any], source: str = "dict") -> JarvisConfig:
    cfg = JarvisConfig()
    _merge(cfg, data)
    cfg.source = source
    validate(cfg)
    return cfg


def validate(cfg: JarvisConfig) -> None:
    p = cfg.permissions
    if not 0 <= p.interactive_level <= 5 or not 0 <= p.automation_level <= 5:
        raise ConfigError("permissions levels must be between 0 and 5")
    if p.interactive_level >= 4:
        # Capability is not authority: consequential actions must never be blanket-approved.
        raise ConfigError("permissions.interactive_level must be <= 3; delegate level 4+ through scoped grants")
    if cfg.ui.verbosity not in ("short", "normal", "detailed"):
        raise ConfigError("ui.verbosity must be short, normal or detailed")
    if cfg.tasks.max_concurrent < 1:
        raise ConfigError("tasks.max_concurrent must be >= 1")
    if cfg.runtime.api_host not in ("127.0.0.1", "localhost", "::1"):
        raise ConfigError("runtime.api_host must be a loopback address (127.0.0.1, localhost or ::1)")
    if cfg.resources.low_priority_policy not in ("pause", "slow", "wait", "continue"):
        raise ConfigError("resources.low_priority_policy must be pause, slow, wait or continue")
    if not re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", cfg.briefing.time):
        raise ConfigError("briefing.time must be HH:MM")
    bad_days = [d for d in cfg.briefing.days if d.lower()[:3] not in WEEKDAYS]
    if bad_days:
        raise ConfigError(f"briefing.days: unknown day(s) {', '.join(bad_days)}")


def find_config_file(env: dict[str, str] | None = None) -> Path | None:
    env = dict(os.environ) if env is None else env
    candidates = []
    if env.get("JARVIS_CONFIG"):
        candidates.append(Path(env["JARVIS_CONFIG"]).expanduser())
    candidates.append(Path.cwd() / "jarvis.toml")
    candidates.append(Path("~/.jarvis/jarvis.toml").expanduser())
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_config(path: str | Path | None = None, env: dict[str, str] | None = None) -> JarvisConfig:
    env = dict(os.environ) if env is None else env
    file_path = Path(path).expanduser() if path else find_config_file(env)
    data: dict[str, Any] = {}
    source = "defaults"
    if file_path is not None:
        try:
            with open(file_path, "rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{file_path}: {exc}") from exc
        source = str(file_path)
    cfg = JarvisConfig()
    _merge(cfg, data)
    _apply_env(cfg, env)
    cfg.source = source
    validate(cfg)
    return cfg
