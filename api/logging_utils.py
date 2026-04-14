import contextvars
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional


_STANDARD_LOG_RECORD_KEYS = set(logging.makeLogRecord({}).__dict__.keys())
_LOG_CONTEXT: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "evomind_log_context",
    default={},
)
_KNOWN_CONTEXT_FIELDS = (
    "service",
    "component",
    "request_id",
    "tenant_id",
    "key_id",
    "job_id",
    "worker_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "route_template",
    "event",
)


def get_log_context() -> Dict[str, Any]:
    current = _LOG_CONTEXT.get()
    return dict(current) if isinstance(current, dict) else {}


def push_log_context(**fields: Any) -> contextvars.Token[Dict[str, Any]]:
    updated = get_log_context()
    for key, value in fields.items():
        if value is None:
            continue
        updated[str(key)] = value
    return _LOG_CONTEXT.set(updated)


def merge_log_context(**fields: Any) -> None:
    updated = get_log_context()
    for key, value in fields.items():
        if value is None:
            continue
        updated[str(key)] = value
    _LOG_CONTEXT.set(updated)


def reset_log_context(token: contextvars.Token[Dict[str, Any]]) -> None:
    _LOG_CONTEXT.reset(token)


class ContextFieldsFilter(logging.Filter):
    def __init__(self, static_fields: Optional[Mapping[str, Any]] = None):
        super().__init__()
        self.static_fields = dict(static_fields or {})

    def filter(self, record: logging.LogRecord) -> bool:
        merged = dict(self.static_fields)
        merged.update(get_log_context())
        for field in _KNOWN_CONTEXT_FIELDS:
            if not hasattr(record, field) and field in merged:
                setattr(record, field, merged[field])
        return True


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in _KNOWN_CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_RECORD_KEYS or key in payload or key.startswith("_"):
                continue
            if value is None:
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(service_name: str, component: Optional[str] = None) -> None:
    level_name = os.getenv("EVOMIND_LOG_LEVEL", "INFO").strip().upper() or "INFO"
    level = getattr(logging, level_name, logging.INFO)
    static_fields: Dict[str, Any] = {"service": service_name}
    if component:
        static_fields["component"] = component

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    handlers = list(root_logger.handlers)
    if not handlers:
        handlers = [logging.StreamHandler()]
        for handler in handlers:
            root_logger.addHandler(handler)

    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(JsonLogFormatter())
        has_context_filter = any(isinstance(existing, ContextFieldsFilter) for existing in handler.filters)
        if not has_context_filter:
            handler.addFilter(ContextFieldsFilter(static_fields))
