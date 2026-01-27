"""Logging configuration with OpenTelemetry integration.

Configures Python logging to automatically include trace context.
"""

from __future__ import annotations

import logging


def configure_otel_logging(log_level: str = "INFO") -> None:
    """Configure logging with OpenTelemetry integration.

    Sets up Python logging to include trace_id and span_id in log records
    when available. Only adds OTel handler, does not remove existing handlers.

    Args:
        log_level: Logging level (default: INFO).
    """
    # Set up basic logging format that includes trace context when available
    log_format = (
        "%(asctime)s %(levelname)s [%(name)s] "
        "[trace_id=%(otelTraceID)s span_id=%(otelSpanID)s] "
        "%(message)s"
    )

    # Create a custom formatter that handles missing trace context
    class OTelFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            # Ensure OTel fields exist
            if not hasattr(record, "otelTraceID"):
                record.otelTraceID = "0" * 32
            if not hasattr(record, "otelSpanID"):
                record.otelSpanID = "0" * 16
            return super().format(record)

    handler = logging.StreamHandler()
    handler.setFormatter(OTelFormatter(log_format))

    # Configure root logger level and add the OTel handler
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.addHandler(handler)
