"""Tracing utilities for TuFT.

Provides tracer access and context propagation utilities.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Module-level tracer cache
_tracers: dict[str, Any] = {}


def get_tracer(name: str = "tuft"):
    """Get a tracer instance by name.

    Args:
        name: Name for the tracer (typically module name).

    Returns:
        A Tracer instance, or a NoOpTracer if OTel is not available.
    """
    if name in _tracers:
        return _tracers[name]

    try:
        from opentelemetry import trace

        tracer = trace.get_tracer(name)
    except ImportError:
        tracer = _NoOpTracer()

    _tracers[name] = tracer
    return tracer


class _NoOpTracer:
    """No-op tracer for when OpenTelemetry is not available."""

    def start_span(self, name: str, **kwargs):
        return _NoOpSpan()

    def start_as_current_span(self, name: str, **kwargs):
        return _NoOpContextManager()


class _NoOpSpan:
    """No-op span."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _NoOpContextManager:
    """No-op context manager."""

    def __enter__(self):
        return _NoOpSpan()

    def __exit__(self, *args):
        pass


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
                    _set_error_status(span)
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
                    _set_error_status(span)
                    raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def _set_error_status(span) -> None:
    """Set error status on a span."""
    try:
        from opentelemetry.trace import StatusCode

        span.set_status(StatusCode.ERROR)
    except ImportError:
        pass


def inject_context(carrier: dict[str, str]) -> None:
    """Inject current trace context into a carrier dict.

    Useful for propagating context across process boundaries (e.g., to Ray actors).

    Args:
        carrier: Dictionary to inject context into.
    """
    try:
        from opentelemetry.propagate import inject

        inject(carrier)
    except ImportError:
        pass


def extract_context(carrier: dict[str, str]):
    """Extract trace context from a carrier dict.

    Args:
        carrier: Dictionary containing trace context.

    Returns:
        Context object, or None if extraction fails.
    """
    try:
        from opentelemetry.propagate import extract

        return extract(carrier)
    except ImportError:
        return None


def get_current_span():
    """Get the current active span.

    Returns:
        Current span, or a NoOpSpan if not available.
    """
    try:
        from opentelemetry import trace

        return trace.get_current_span()
    except ImportError:
        return _NoOpSpan()
