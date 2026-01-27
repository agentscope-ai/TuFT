"""OpenTelemetry Provider initialization.

Initializes TracerProvider, MeterProvider, and LoggerProvider with
OTLP or Console exporters based on configuration.
"""

from __future__ import annotations

import logging
import os

from opentelemetry import metrics, trace
from opentelemetry._logs import get_logger_provider, set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from tuft.config import TelemetryConfig

from .metrics import ResourceMetricsCollector, clear_meters
from .tracing import clear_tracers


logger = logging.getLogger(__name__)

# Global state to track initialization
_tel_initialized = False


def _is_debug_mode() -> bool:
    """Check if debug mode is enabled via OTEL_DEBUG environment variable."""
    return os.getenv("OTEL_DEBUG", "0") == "1"


def _get_otlp_endpoint(config: TelemetryConfig) -> str | None:
    """Get OTLP endpoint from config or environment variable."""
    if config.otlp_endpoint:
        return config.otlp_endpoint
    return os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")


def init_telemetry(config: TelemetryConfig) -> None:
    """Initialize OpenTelemetry providers.

    Sets up TracerProvider, MeterProvider, and LoggerProvider with appropriate
    exporters based on configuration and environment variables.

    Args:
        config: Telemetry configuration from tuft.config.TelemetryConfig.
    """
    global _tel_initialized

    if not config.enabled:
        logger.debug("Telemetry is disabled")
        return

    if _tel_initialized:
        logger.warning("Telemetry already initialized, skipping")
        return

    # Build resource with service info
    resource_attrs = {
        "service.name": config.service_name,
    }
    resource_attrs.update(config.resource_attributes)
    resource = Resource.create(resource_attrs)

    # Determine exporter mode
    debug_mode = _is_debug_mode()
    otlp_endpoint = _get_otlp_endpoint(config)

    if debug_mode:
        logger.info("Initializing telemetry with Console exporters (debug mode)")
        _init_console_exporters(resource)
    elif otlp_endpoint:
        logger.info("Initializing telemetry with OTLP exporters to %s", otlp_endpoint)
        _init_otlp_exporters(resource, otlp_endpoint)
    else:
        logger.info("Initializing telemetry with OTLP exporters (using default endpoint)")
        _init_otlp_exporters(resource, None)

    # Configure logging integration
    _configure_logging_integration()

    _tel_initialized = True
    logger.info("Telemetry initialized successfully")


def _init_console_exporters(resource: Resource) -> None:
    """Initialize Console exporters for debugging."""
    # Trace
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    # Metrics
    reader = PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=10000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)

    # Logs
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(ConsoleLogExporter()))
    set_logger_provider(logger_provider)


def _init_otlp_exporters(resource: Resource, endpoint: str | None) -> None:
    """Initialize OTLP exporters."""
    # Build exporter kwargs
    exporter_kwargs = {}
    if endpoint:
        exporter_kwargs["endpoint"] = endpoint

    # Trace
    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(**exporter_kwargs)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    # Metrics
    metric_exporter = OTLPMetricExporter(**exporter_kwargs)
    reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=10000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)

    # Logs
    log_exporter = OTLPLogExporter(**exporter_kwargs)
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    set_logger_provider(logger_provider)


def _configure_logging_integration() -> None:
    """Configure Python logging to integrate with OpenTelemetry."""
    LoggingInstrumentor().instrument(set_logging_format=True)

    # Bridge Python logging to OTel LoggerProvider
    # Get the configured logger provider
    otel_logger_provider = get_logger_provider()

    # Create a handler that sends logs to OTel
    otel_handler = LoggingHandler(
        level=logging.DEBUG,
        logger_provider=otel_logger_provider,
    )

    # Add the handler to the root logger (don't remove existing handlers)
    root_logger = logging.getLogger()
    root_logger.addHandler(otel_handler)

    logger.debug("Python logging bridged to OTel LoggerProvider")


def shutdown_telemetry() -> None:
    """Shutdown OpenTelemetry providers gracefully."""
    global _tel_initialized

    if not _tel_initialized:
        return

    # Shutdown trace provider
    tracer_provider = trace.get_tracer_provider()
    shutdown_fn = getattr(tracer_provider, "shutdown", None)
    if shutdown_fn is not None:
        shutdown_fn()

    # Shutdown meter provider
    meter_provider = metrics.get_meter_provider()
    shutdown_fn = getattr(meter_provider, "shutdown", None)
    if shutdown_fn is not None:
        shutdown_fn()

    # Shutdown logger provider
    logger_provider = get_logger_provider()
    shutdown_fn = getattr(logger_provider, "shutdown", None)
    if shutdown_fn is not None:
        shutdown_fn()

    # Clear cached tracers and meters
    clear_tracers()
    clear_meters()

    # Shutdown resource metrics collector
    ResourceMetricsCollector.shutdown()

    _tel_initialized = False
    logger.info("Telemetry shutdown complete")
