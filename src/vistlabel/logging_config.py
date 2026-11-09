# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Structured (JSON) logging setup.

The application uses structured logging so log lines can be correlated
with provenance events without pulling in a database. A minimal
stdlib-based JSON formatter is used instead of an extra dependency such as
structlog.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from vistlabel.settings import LoggingSettings

_RESERVED_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_ATTRS
        }
        if extra:
            payload["context"] = extra
        return json.dumps(payload, default=str)


def configure_logging(settings: LoggingSettings | None = None) -> None:
    settings = settings or LoggingSettings()
    root = logging.getLogger()
    root.setLevel(settings.level.upper())
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    if settings.json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
