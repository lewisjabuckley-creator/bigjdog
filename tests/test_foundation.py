import json
import logging

import pytest

from jarvis.config import ConfigError, config_from_dict, load_config
from jarvis.core.types import Confidence, Fact, Provenance, ProvenanceKind, Severity
from jarvis.database.db import Database
from jarvis.events.bus import EventBus
from jarvis.events.store import EventStore
from jarvis.events.types import Event, EventType
from jarvis.log import JsonFormatter
from jarvis.security.redaction import REDACTED, redact
from jarvis.security.secrets import EnvSecretBackend, SecretStore


# -- configuration ---------------------------------------------------------------

def test_defaults_are_safe():
    cfg = config_from_dict({})
    assert cfg.permissions.interactive_level == 3
    assert cfg.privacy.allow_cloud is False
    assert cfg.models.openai_compatible.enabled is False


def test_partial_tables_keep_defaults():
    cfg = config_from_dict({"monitoring": {"thresholds": {"cpu_percent": 75}}})
    assert cfg.monitoring.thresholds["cpu_percent"].value == 75
    assert cfg.monitoring.thresholds["cpu_percent"].sustain_s == 300
    assert "disk_percent" in cfg.monitoring.thresholds


def test_unknown_keys_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict({"models": {"olama": {}}})


def test_type_errors_rejected():
    with pytest.raises(ConfigError):
        config_from_dict({"tasks": {"max_concurrent": "four"}})


def test_blanket_consequential_authority_rejected():
    with pytest.raises(ConfigError):
        config_from_dict({"permissions": {"interactive_level": 5}})


def test_load_from_file_and_env(tmp_path):
    path = tmp_path / "jarvis.toml"
    path.write_text('[general]\ndata_dir = "/tmp/x"\n[ui]\nverbosity = "short"\n')
    cfg = load_config(path, env={"JARVIS_PRIVATE_MODE": "true"})
    assert cfg.general.data_dir == "/tmp/x"
    assert cfg.ui.verbosity == "short"
    assert cfg.privacy.private_mode is True
    assert cfg.source == str(path)


# -- database --------------------------------------------------------------------

def test_migrations_idempotent(tmp_path):
    path = tmp_path / "j.db"
    db1 = Database(path)
    v = db1.schema_version()
    db1.close()
    db2 = Database(path)
    assert db2.schema_version() == v >= 1
    db2.close()


def test_transaction_rollback(db):
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.execute("INSERT INTO meta(key, value) VALUES('a', '1')")
            raise RuntimeError("boom")
    assert db.query_one("SELECT * FROM meta WHERE key='a'") is None


# -- security --------------------------------------------------------------------

def test_redaction_by_key_and_pattern():
    data = {"api_key": "abc123", "nested": {"password": "hunter2"},
            "cmd": "curl -H 'Authorization: Bearer abcdefghijklmnopqrstuvwxyz'",
            "text": "export OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwx"}
    out = redact(data)
    assert out["api_key"] == REDACTED
    assert out["nested"]["password"] == REDACTED
    assert "abcdefghijklmnopqrstuvwxyz" not in out["cmd"]
    assert "sk-abc" not in out["text"]


def test_secret_references_resolve_only_on_demand():
    store = SecretStore(EnvSecretBackend({"MY_TOKEN": "t0k"}))
    assert store.resolve("secret://MY_TOKEN") == "t0k"
    assert store.resolve("plain") == "plain"
    with pytest.raises(KeyError):
        store.resolve("secret://MISSING")


def test_json_log_formatter_is_structured_and_redacted():
    record = logging.makeLogRecord({"name": "jarvis.tools", "msg": "tool_executed", "levelname": "INFO",
                                    "tool": "shell", "token": "secret-value"})
    payload = json.loads(JsonFormatter().format(record))
    assert payload["component"] == "tools"
    assert payload["event"] == "tool_executed"
    assert payload["tool"] == "shell"
    assert payload["token"] == REDACTED


# -- core types ------------------------------------------------------------------

def test_fact_staleness_and_roundtrip():
    fact = Fact(82, Confidence.OBSERVED, Provenance(ProvenanceKind.SYSTEM_STATE, "psutil"), 100.0, ttl=30)
    assert not fact.is_stale(120)
    assert fact.is_stale(131)
    assert Fact.from_dict(fact.to_dict()) == fact


# -- events ----------------------------------------------------------------------

async def test_bus_delivers_by_pattern_and_isolates_failures(db, clock):
    bus = EventBus(EventStore(db), clock)
    got: list[str] = []

    def broken(event):
        raise RuntimeError("subscriber bug")

    async def task_handler(event):
        got.append(f"task:{event.type}")

    bus.subscribe("*", broken)
    bus.subscribe("TASK_*", task_handler)
    bus.subscribe(EventType.FILE_CHANGED, lambda e: got.append("file"))

    await bus.publish(Event(EventType.TASK_FAILED, "test"))
    await bus.publish(Event(EventType.FILE_CHANGED, "test"))
    await bus.publish(Event(EventType.SYSTEM_METRICS, "test"))

    assert got == ["task:TASK_FAILED", "file"]
    assert bus.handler_errors == 3
    # noisy metrics are live-only; the rest persisted
    stored = EventStore(db).query(newest_first=False)
    assert [e.type for e in stored] == ["TASK_FAILED", "FILE_CHANGED"]


async def test_emit_and_drain(db, clock):
    bus = EventBus(EventStore(db), clock)
    got = []
    bus.subscribe("*", lambda e: got.append(e.type))
    bus.emit(Event(EventType.TIMER_EXPIRED, "t"))
    await bus.drain()
    assert got == ["TIMER_EXPIRED"]


def test_retention(db):
    store = EventStore(db)
    day = 86400
    now = 1000 * day
    store.append(Event("X", "t", severity=Severity.DEBUG, ts=now - 2 * day, persist=True))
    store.append(Event("X", "t", severity=Severity.INFO, ts=now - 2 * day))
    store.append(Event("X", "t", severity=Severity.INFO, ts=now - 40 * day))
    store.append(Event("X", "t", severity=Severity.ERROR, ts=now - 40 * day))
    assert store.prune(now) == 2
    assert sorted(e.severity for e in store.query()) == [Severity.INFO, Severity.ERROR]
