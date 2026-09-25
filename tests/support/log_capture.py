"""Log capture context manager and utilities for privacy/security assertions."""

from __future__ import annotations

import io
import logging
import sys
from contextlib import contextmanager
from typing import Generator


class LogCaptureHandler(logging.Handler):
    """Memory handler to capture logging records."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        self.messages.append(self.format(record))


@contextmanager
def capture_logs_and_stdout(
    logger_name: str = "pyhue2d",
) -> Generator[tuple[LogCaptureHandler, io.StringIO], None, None]:
    """Capture log records and stdout simultaneously."""
    logger = logging.getLogger(logger_name)
    handler = LogCaptureHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(logging.DEBUG)

    old_stdout = sys.stdout
    captured_stdout = io.StringIO()
    sys.stdout = captured_stdout

    try:
        yield handler, captured_stdout
    finally:
        sys.stdout = old_stdout
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
