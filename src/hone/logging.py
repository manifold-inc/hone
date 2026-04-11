"""Logging utilities for Hone."""

import json
import logging
import logging.handlers
import os
import socket
import time
import uuid
from datetime import datetime
from queue import Queue
from typing import Final

import bittensor as bt
import logging_loki
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler

LOKI_URL: Final[str] = os.environ.get(
    "LOKI_URL", "https://logs.tplr.ai/loki/api/v1/push"
)
TRACE_ID: Final[str] = str(uuid.uuid4())


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
    environment="finney",
    url=LOKI_URL,
) -> logging.Logger:
    host = socket.gethostname()
    pid = os.getpid()
    tags = {
        "service": service,
        "host": host,
        "pid": pid,
        "environment": environment,
        "version": version,
        "uid": uid,
        "trace_id": TRACE_ID,
    }

    class StructuredLogFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            log_data = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "host": host,
                "pid": pid,
                "service": service,
                "environment": environment,
                "version": version,
                "uid": uid,
                "trace_id": TRACE_ID,
            }
            if hasattr(record, "extra_data") and record.extra_data:
                log_data.update(record.extra_data)
            return json.dumps(log_data)

    def _log_with_context(logger, level, message, **context):
        record = logging.LogRecord(
            name=logger.name,
            level=getattr(logging, level.upper()),
            pathname=__file__,
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        record.extra_data = context
        for handler in logger.handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    try:
        log = logging.getLogger("hone")
        log_queue = Queue(-1)
        queue_handler = logging.handlers.QueueHandler(log_queue)
        listener = logging.handlers.QueueListener(
            log_queue, respect_handler_level=True
        )
        loki_handler = logging_loki.LokiHandler(
            url=url, tags=tags, auth=None, version="1"
        )
        listener.handlers = [loki_handler]
        console_handler = RichHandler(
            markup=True,
            rich_tracebacks=True,
            highlighter=NullHighlighter(),
            show_level=False,
            show_time=True,
            show_path=False,
        )
        loki_handler.setFormatter(StructuredLogFormatter())
        log.setLevel(logging.INFO)
        log.handlers.clear()
        listener.start()
        log.addHandler(queue_handler)
        log.addHandler(console_handler)
        log.log_with_context = lambda level, message, **kwargs: _log_with_context(
            log, level, message, **kwargs
        )
        log.propagate = False
        log._listener = listener
        return log
    except Exception as e:
        log = logging.getLogger("hone")
        log.error(f"Failed to add Loki logging: {e}")
        if not log.handlers:
            log.addHandler(
                RichHandler(
                    markup=True,
                    rich_tracebacks=True,
                    highlighter=NullHighlighter(),
                    show_level=False,
                    show_time=True,
                    show_path=False,
                )
            )
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
