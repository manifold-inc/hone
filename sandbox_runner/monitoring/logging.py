"""
Structured Logging Module

Provides structured JSON logging for the sandbox runner with:
- Consistent log format across all modules
- Contextual information (request ID, job ID, etc.)
- Log levels and filtering
- File and console output
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    Each log entry is formatted as a JSON object with:
    - timestamp: ISO format timestamp
    - level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - logger: Logger name (module path)
    - message: Log message
    - extra: Any additional context fields
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add any extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs", "message",
                "pathname", "process", "processName", "relativeCreated", "thread",
                "threadName", "exc_info", "exc_text", "stack_info"
            ]:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        # Add source location for errors
        if record.levelno >= logging.ERROR:
            log_entry["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName
            }
        
        return json.dumps(log_entry)


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter for console output.
    
    Format: [TIMESTAMP] LEVEL - LOGGER - MESSAGE
    """
    
    def __init__(self):
        super().__init__(
            fmt="[%(asctime)s] %(levelname)-8s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def setup_logging(log_level: str = "INFO", log_file: Path = None) -> logging.Logger:
    """
    Setup structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file for JSON output
        
    Returns:
        Root logger instance
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove any existing handlers
    root_logger.handlers.clear()
    
    # Console handler with text formatting (for human readability)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(TextFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler with JSON formatting (for log aggregation)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    
    # Log startup message
    root_logger.info(
        "Logging initialized",
        extra={
            "log_level": log_level,
            "log_file": str(log_file) if log_file else None
        }
    )
    
    return root_logger


class LogContext:
    """
    Context manager for adding contextual information to log messages.
    
    Usage:
        with LogContext(job_id="job_123", validator="5ABC..."):
            logger.info("Processing job")  # Will include job_id and validator
    """
    
    _context: Dict[str, Any] = {}
    
    def __init__(self, **kwargs):
        """
        Initialize log context with key-value pairs.
        
        Args:
            **kwargs: Context fields to add to log messages
        """
        self.context = kwargs
        self.old_context = None
    
    def __enter__(self):
        """Enter context manager - save old context and apply new."""
        self.old_context = LogContext._context.copy()
        LogContext._context.update(self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - restore old context."""
        LogContext._context = self.old_context
    
    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """Get current log context."""
        return cls._context.copy()


# Custom log filter to inject context
class ContextFilter(logging.Filter):
    """Filter that injects context fields into log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add context fields to log record.
        
        Args:
            record: Log record to modify
            
        Returns:
            True (always allow record through)
        """
        # Inject context fields
        for key, value in LogContext.get_context().items():
            setattr(record, key, value)
        
        return True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with context filtering enabled.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance with context support
    """
    logger = logging.getLogger(name)
    logger.addFilter(ContextFilter())
    return logger