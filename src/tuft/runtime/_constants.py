"""Constants for TuFT runtime: environment variable names, default paths, and ports."""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------

ENV_TUFT_ADDRESS = "TUFT_ADDRESS"
"""Service address, e.g. http://127.0.0.1:10610"""

ENV_TUFT_API_KEY = "TUFT_API_KEY"  # pragma: allowlist secret
"""API authentication key"""

ENV_TUFT_CONFIG = "TUFT_CONFIG"
"""Path to config file"""

ENV_TUFT_MODEL_PATH = "TUFT_MODEL_PATH"
"""Model path for auto-generating minimal config"""

ENV_TUFT_HOME = "TUFT_HOME"
"""TuFT home directory, defaults to ~/.tuft"""

ENV_TUFT_HOST = "TUFT_HOST"
"""Server bind address"""

ENV_TUFT_PORT = "TUFT_PORT"
"""Server bind port"""

ENV_TUFT_ENABLE_AUTO_CONNECT = "TUFT_ENABLE_AUTO_CONNECT"
"""Whether to enable auto-connect, defaults to "1" (enabled)"""

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 10610

# ---------------------------------------------------------------------------
# Derived paths (resolved at import time, but functions allow override)
# ---------------------------------------------------------------------------


def get_tuft_home() -> Path:
    """Return the TUFT_HOME directory, defaulting to ~/.tuft."""
    return Path(os.environ.get(ENV_TUFT_HOME, Path.home() / ".tuft"))


def get_address_file() -> Path:
    """Return the path to the server address file."""
    return get_tuft_home() / "tuft_current_server"


def get_default_config_path() -> Path:
    """Return the default config file path."""
    return get_tuft_home() / "configs" / "tuft_config.yaml"


def get_default_checkpoint_dir() -> Path:
    """Return the default checkpoint directory."""
    return get_tuft_home() / "checkpoints"


def get_credentials_file() -> Path:
    """Return the path to the auto-generated credentials file."""
    return get_tuft_home() / "credentials"


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------

HEALTHZ_PATH = "/api/v1/healthz"
"""Health check endpoint path used for service discovery."""

HEALTHZ_TIMEOUT = 2.0
"""Timeout in seconds for a single health check request."""

STARTUP_TIMEOUT = 120.0
"""Maximum seconds to wait for an embedded service to become healthy."""

STARTUP_POLL_INTERVAL = 0.5
"""Seconds between health check polls during startup."""
