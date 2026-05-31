"""
Structured logging configuration using structlog.

In development (APP_ENV=development):
    Outputs colored, human-readable logs to console.

In production (APP_ENV=production):
    Outputs JSON lines (one JSON object per log entry), suitable for
    shipping to Datadog, Elasticsearch, GCP Logging, or Render's log drain.

Usage in any module:
    import structlog
    logger = structlog.get_logger(__name__)
    logger.info("event.name", user_id="123", tool="bash", duration_ms=45.2)
"""

import logging
import os
import sys
import structlog


def configure_logging() -> None:
    """
    Configure structlog and the standard library logging bridge.
    Call once at application startup in main.py.
    """
    app_env = os.getenv("APP_ENV", "development")
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Processors run on every log event, in order
    shared_processors = [
        structlog.contextvars.merge_contextvars,        # Merges bound context (request_id, user_id)
        structlog.stdlib.add_log_level,                 # Adds "level" field
        structlog.stdlib.add_logger_name,               # Adds "logger" field
        structlog.processors.TimeStamper(fmt="iso"),    # ISO 8601 timestamp
        structlog.processors.StackInfoRenderer(),       # Stack info if present
        structlog.processors.format_exc_info,           # Exception tracebacks
    ]

    if app_env == "production":
        # JSON output for log aggregation tools
        renderer = structlog.processors.JSONRenderer()
    else:
        # Colored console output for development
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Root logger — catches everything including uvicorn, motor, etc.
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("motor").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
