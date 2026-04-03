"""Minimal logging utilities for Cosmic Origins."""

from __future__ import annotations

import datetime as _dt
from typing import Any


def log(message: str, *args: Any) -> None:
    """Print a timestamped log message."""
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args:
        message = message.format(*args)
    print(f"[{timestamp}] {message}")

