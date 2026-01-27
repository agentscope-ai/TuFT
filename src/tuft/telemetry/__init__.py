"""OpenTelemetry integration for TuFT.

This module provides observability through Traces, Metrics, and Logs.
"""

from __future__ import annotations

from tuft.config import TelemetryConfig

from .provider import init_telemetry, shutdown_telemetry


__all__ = [
    "TelemetryConfig",
    "init_telemetry",
    "shutdown_telemetry",
]
