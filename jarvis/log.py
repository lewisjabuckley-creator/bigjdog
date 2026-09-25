"""Structured logging (spec §160).

Every record is a JSON object with ``timestamp``, ``component``, ``event`` and
``severity`` plus any contextual fields (``task_id``, ``tool``, ``result``...).
Human-readable output is derived from structure, never the other way round.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from jarvis.security.redaction import redact

_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": round(record.created, 3),
            "severity": record.levelname.lower(),
            "component": record.name.removeprefix("jarvis."),
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(redact(payload), default=str)


class ComponentLogger(logging.LoggerAdapter):
    """Logger whose calls take structured keyword fields: ``log.info("task_started", task_id=...)``."""

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        extra = dict(self.extra or {})
        for key in list(kwargs):
            if key not in ("exc_info", "stack_info", "stacklevel", "extra"):
                extra[key] = kwargs.pop(key)
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(component: str, **context: Any) -> ComponentLogger:
    return ComponentLogger(logging.getLogger(f"jarvis.{component}"), context)


def configure_logging(log_dir: Path | None = None, level: int = logging.INFO, stderr: bool = False) -> None:
    root = logging.getLogger("jarvis")
    root.setLevel(level)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = JsonFormatter()
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"jarvis-{time.strftime('%Y%m%d')}.jsonl", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    if stderr or log_dir is None:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        stream.setLevel(level if stderr else logging.ERROR)
        root.addHandler(stream)
