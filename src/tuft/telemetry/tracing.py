"""Tracing utilities for TuFT.

Provides tracer access and context propagation utilities.
"""

from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.propagate import extract, inject


# Module-level tracer cache
_tracers: dict[str, Any] = {}

# Re-export for convenience
inject_context = inject
extract_context = extract
get_current_span = trace.get_current_span


def get_tracer(name: str = "tuft"):
    """Get a tracer instance by name.

    Args:
        name: Name for the tracer (typically module name).

    Returns:
        A Tracer instance. When no TracerProvider is configured,
        OpenTelemetry automatically returns a NoOpTracer.
    """
    if name in _tracers:
        return _tracers[name]

    tracer = trace.get_tracer(name)
    _tracers[name] = tracer
    return tracer


def clear_tracers() -> None:
    """Clear the tracer cache. Used during shutdown."""
    _tracers.clear()
