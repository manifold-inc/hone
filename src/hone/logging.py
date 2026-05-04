"""Logging utilities for Hone."""

import logging
import time

import bittensor as bt
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler


def T() -> float:
    return time.time()


def P(window: int, duration: float) -> str:
    return f"[steel_blue]{window}[/steel_blue] ([grey63]{duration:.2f}s[/grey63])"


FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(
            markup=True,
            rich_tracebacks=True,
            highlighter=NullHighlighter(),
            show_level=False,
            show_time=True,
            show_path=False,
        )
    ],
)

logger = logging.getLogger("hone")
logger.setLevel(logging.INFO)


class NoSubtensorWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return (
            "Verify your local subtensor is running on port" not in record.getMessage()
        )


logging.getLogger().addFilter(NoSubtensorWarning())
logger.addFilter(NoSubtensorWarning())
for handler in logging.getLogger().handlers:
    handler.addFilter(NoSubtensorWarning())


def debug() -> None:
    logger.setLevel(logging.DEBUG)


def trace() -> None:
    TRACE_LEVEL_NUM = 5
    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

    def trace_method(self, message, *args, **kws) -> None:
        if self.isEnabledFor(TRACE_LEVEL_NUM):
            self._log(TRACE_LEVEL_NUM, message, args, **kws)

    logging.Logger.trace = trace_method
    logger.setLevel(TRACE_LEVEL_NUM)


bt.logging.off()

logger.setLevel(logging.INFO)
logger.propagate = True
logger.handlers.clear()
logger.addHandler(
    RichHandler(
        markup=True,
        rich_tracebacks=True,
        highlighter=NullHighlighter(),
        show_level=False,
        show_time=True,
        show_path=False,
    )
)


def setup_loki_logger(
    service: str,
    uid: str,
    version: str,
    environment: str = "finney",
    url: str | None = None,
) -> logging.Logger:
    """Return the process logger with ``log_with_context`` attached.

    (historical name; Loki backend removed 2026-05-03. Kept for import
    stability. Returns the process logger with ``log_with_context``
    attached.)

    The ``service``, ``uid``, ``version``, ``environment`` and ``url``
    parameters are accepted for backwards compatibility with callers
    such as ``hone/neurons/validator.py`` that pass them positionally
    or by keyword. They are intentionally unused: shipping logs to
    ``logs.tplr.ai`` was retired after repeated Cloudflare 5xx upstream
    failures, and the local ``RichHandler`` console output already
    captures the full record stream that pm2 retains on disk.
    """
    del service, uid, version, environment, url

    log = logging.getLogger("hone")

    def _log_with_context(level: str, message: str, **context) -> None:
        record = logging.LogRecord(
            name=log.name,
            level=getattr(logging, level.upper()),
            pathname=__file__,
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        record.extra_data = context
        for handler in log.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    log.log_with_context = _log_with_context
    log.propagate = False
    return log


def log_with_context(level, message, **context):
    if not hasattr(logger, "log_with_context"):
        getattr(logger, level.lower())(message)
        return
    logger.log_with_context(level, message, **context)


__all__ = [
    "logger",
    "debug",
    "trace",
    "P",
    "T",
    "setup_loki_logger",
    "log_with_context",
]
