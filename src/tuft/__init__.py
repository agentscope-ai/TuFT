"""TuFT service package."""

# Runtime API (embedded mode & service management)
from .runtime import (  # noqa: E402
    create_sampling_client,
    create_training_client,
    get_service_client,
    init,
    is_initialized,
    shutdown,
)
from .server import create_root_app


__all__ = [
    "create_root_app",
    # Runtime API
    "init",
    "shutdown",
    "is_initialized",
    "get_service_client",
    "create_training_client",
    "create_sampling_client",
]
