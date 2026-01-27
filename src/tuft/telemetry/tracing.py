"""Tracing utilities for TuFT.

Provides tracer access and context propagation utilities.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from opentelemetry import trace
from opentelemetry.propagate import extract, inject
from opentelemetry.trace import StatusCode


F = TypeVar("F", bound=Callable[..., Any])

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


def traced(
    name: str | None = None,
    tracer_name: str = "tuft",
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorator to trace a function.

    Args:
        name: Span name. Defaults to function name.
        tracer_name: Name of the tracer to use.
        attributes: Static attributes to add to the span.

    Returns:
        Decorated function.
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__
        tracer = get_tracer(tracer_name)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    span.set_attributes(attributes)
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(StatusCode.ERROR)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    span.set_attributes(attributes)
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(StatusCode.ERROR)
                    raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
