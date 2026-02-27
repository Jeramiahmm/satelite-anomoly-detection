"""
Structured JSON logging for Astraea-1.

When ASTRAEA_LOG_FORMAT=json is set, all log output switches to
single-line JSON objects for easy ingestion by log aggregators
(ELK, Loki, CloudWatch, etc.).

Usage:
    from astraea.observability.logging import configure_logging
    configure_logging()  # reads ASTRAEA_LOG_FORMAT from env

Environment Variables:
    ASTRAEA_LOG_FORMAT  — "text" (default) or "json"
    ASTRAEA_LOG_LEVEL   — "DEBUG", "INFO" (default), "WARNING", "ERROR"
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        # Include extra fields if attached
        for key in ("node_id", "event", "metric", "value"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        return json.dumps(entry, default=str)


class TextFormatter(logging.Formatter):
    """Clean text formatter with timestamps."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )


def configure_logging(
    log_format: Optional[str] = None,
    log_level: Optional[str] = None,
) -> None:
    """
    Configure logging based on environment variables or explicit args.

    Args:
        log_format: "json" or "text". Defaults to ASTRAEA_LOG_FORMAT env var.
        log_level: Standard Python level. Defaults to ASTRAEA_LOG_LEVEL env var.
    """
    fmt = log_format or os.environ.get("ASTRAEA_LOG_FORMAT", "text").lower()
    level_str = log_level or os.environ.get("ASTRAEA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if fmt == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root.addHandler(handler)
