"""TuFT Runtime — public API for embedded mode and service management.

Usage:
    import tuft

    tuft.init(model="/path/to/model")   # auto-discover or start embedded server
    client = tuft.get_service_client()  # returns tinker.ServiceClient
    tuft.shutdown()                     # stop embedded server if any
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import tinker

from ._config_gen import generate_api_key, generate_config_file
from ._constants import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    ENV_TUFT_API_KEY,
    ENV_TUFT_CONFIG,
    ENV_TUFT_ENABLE_AUTO_CONNECT,
    ENV_TUFT_HOST,
    ENV_TUFT_MODEL_PATH,
    ENV_TUFT_PORT,
    get_credentials_file,
    get_default_config_path,
)
from ._discovery import discover
from ._launcher import EmbeddedServer


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (singleton, thread-safe)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_initialized = False
_mode: Optional[str] = None  # "connected" | "embedded"
_service_client: Optional[tinker.ServiceClient] = None
_embedded_server: Optional[EmbeddedServer] = None
_api_key: Optional[str] = None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "init",
    "shutdown",
    "is_initialized",
    "get_service_client",
    "create_training_client",
    "create_sampling_client",
    "generate_api_key",
]


def init(
    *,
    address: Optional[str] = None,
    model: Optional[str | Path] = None,
    config: Optional[str | Path] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    api_key: Optional[str] = None,
    ignore_reinit_error: bool = True,
) -> None:
    """Initialize TuFT: discover an existing service or start an embedded one.

    This function is idempotent. Calling it multiple times is safe when
    ignore_reinit_error=True (default).

    Args:
        address: Explicit service address to connect to.
        model: Model path for auto-generating config and starting embedded server.
        config: Path to a YAML config file for the embedded server.
        host: Host to bind the embedded server (default: 127.0.0.1).
        port: Port to bind the embedded server (default: 10610).
        api_key: API key for authentication. Auto-generated if not provided.
        ignore_reinit_error: If True, silently skip if already initialized.

    Raises:
        RuntimeError: If already initialized and ignore_reinit_error is False.
        RuntimeError: If no service found and cannot start embedded server.
    """
    global _initialized, _mode, _service_client, _embedded_server, _api_key

    if _initialized:
        if ignore_reinit_error:
            return
        raise RuntimeError(
            "TuFT is already initialized. Call tuft.shutdown() first, "
            "or use ignore_reinit_error=True."
        )

    with _lock:
        # Double-check after acquiring lock
        if _initialized:
            if ignore_reinit_error:
                return
            raise RuntimeError("TuFT is already initialized.")

        resolved_host = host or os.environ.get(ENV_TUFT_HOST, DEFAULT_HOST)
        resolved_port = port or int(os.environ.get(ENV_TUFT_PORT, str(DEFAULT_PORT)))

        # Phase 1: Try to discover an existing service
        discovered = discover(explicit_address=address)
        if discovered:
            _api_key = api_key or os.environ.get(ENV_TUFT_API_KEY)
            _service_client = tinker.ServiceClient(
                base_url=discovered,
                api_key=_api_key or "",
            )
            _mode = "connected"
            _initialized = True
            logger.info("TuFT initialized in connected mode: %s", discovered)
            return

        # If explicit address was given but not healthy, fail
        if address:
            raise RuntimeError(
                f"Cannot connect to TuFT at {address}. "
                "Ensure the server is running or remove the address parameter."
            )

        # Phase 2: Auto-start embedded server
        # Determine config source
        config_path = _resolve_config_for_launch(config, model, resolved_host, resolved_port)
        if config_path is None:
            raise RuntimeError(
                "Cannot start TuFT: no service found and no configuration available.\n"
                "Please provide one of:\n"
                "  - tuft.init(address='http://...')  to connect to existing service\n"
                "  - tuft.init(model='/path/to/model')  to auto-start\n"
                "  - tuft.init(config='/path/to/config.yaml')  to auto-start\n"
                "  - Set TUFT_ADDRESS, TUFT_MODEL_PATH, or TUFT_CONFIG env var\n"
                "  - Create ~/.tuft/configs/tuft_config.yaml"
            )

        _embedded_server = EmbeddedServer(
            config_path=config_path,
            host=resolved_host,
            port=resolved_port,
        )
        server_address = _embedded_server.start()

        # Resolve API key
        if _api_key is None:
            _api_key = api_key or os.environ.get(ENV_TUFT_API_KEY) or ""

        _service_client = tinker.ServiceClient(
            base_url=server_address,
            api_key=_api_key,
        )
        _mode = "embedded"
        _initialized = True
        logger.info("TuFT initialized in embedded mode: %s", server_address)


def shutdown() -> None:
    """Shutdown TuFT: disconnect and stop embedded server if running."""
    global _initialized, _mode, _service_client, _embedded_server, _api_key

    with _lock:
        if _embedded_server is not None:
            _embedded_server.shutdown()
            _embedded_server = None
        _service_client = None
        _api_key = None
        _mode = None
        _initialized = False
        logger.info("TuFT shut down.")


def is_initialized() -> bool:
    """Return True if TuFT has been initialized."""
    return _initialized


def get_service_client() -> tinker.ServiceClient:
    """Return the global ServiceClient, auto-initializing if needed.

    Returns:
        A connected tinker.ServiceClient instance.

    Raises:
        RuntimeError: If auto-initialization fails.
    """
    global _service_client
    if not _initialized:
        # Lazy init: check if auto-connect is enabled
        auto_connect = os.environ.get(ENV_TUFT_ENABLE_AUTO_CONNECT, "1")
        if auto_connect != "1":
            raise RuntimeError(
                "TuFT is not initialized and auto-connect is disabled "
                f"(TUFT_ENABLE_AUTO_CONNECT={auto_connect}). "
                "Call tuft.init() explicitly."
            )
        init()
    if _service_client is None:
        raise RuntimeError("TuFT initialization failed: no service client available.")
    return _service_client


def create_training_client(
    base_model: str,
    rank: int | None = 16,
    **kwargs,
):
    """Convenience: create a training client via the global ServiceClient.

    Args:
        base_model: The base model name/path registered on the server.
            If a full path is given, it will be resolved to the model directory name.
        rank: LoRA rank. Set to None for full-parameter training (no LoRA).
        **kwargs: Additional arguments passed to create_lora_training_client.

    Returns:
        A training client (LoRA or full-param depending on rank).
    """
    # If base_model looks like an absolute path, extract the directory name
    # since the server registers models by directory name (e.g. "Qwen2.5-0.5B-Instruct")
    if os.path.sep in base_model or base_model.startswith("/"):
        base_model = Path(base_model).name

    client = get_service_client()

    if rank is None:
        # Full-parameter training path
        return _create_full_param_training_client(client, base_model, **kwargs)

    return client.create_lora_training_client(
        base_model=base_model,
        rank=rank,
        **kwargs,
    )


def create_sampling_client(
    base_model: Optional[str] = None,
    model_path: Optional[str] = None,
    **kwargs,
):
    """Convenience: create a sampling client via the global ServiceClient.

    Args:
        base_model: The base model name (for base model sampling).
            If a full path is given, it will be resolved to the model directory name.
        model_path: A specific model path (e.g., LoRA checkpoint).
        **kwargs: Additional arguments passed to create_sampling_client.

    Returns:
        A sampling client.
    """
    # Resolve absolute path to model name
    if base_model and (os.path.sep in base_model or base_model.startswith("/")):
        base_model = Path(base_model).name

    client = get_service_client()
    if model_path:
        return client.create_sampling_client(model_path=model_path, **kwargs)
    return client.create_sampling_client(base_model=base_model, **kwargs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _create_full_param_training_client(
    service_client: tinker.ServiceClient,
    base_model: str,
    user_metadata: dict[str, str] | None = None,
    **kwargs,
):
    """Create a TrainingClient for full-parameter training (lora_config=None).

    Uses the Tinker SDK's internal mechanisms to send a CreateModelRequest
    with lora_config=None, then wraps the response into a TrainingClient.
    """
    import time

    from tinker import types as tinker_types
    from tinker.lib.api_future_impl import _APIFuture
    from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
    from tinker.lib.public_interfaces.training_client import TrainingClient
    from tinker.lib.queue_state_logger import QueueStateLogger

    holder = service_client.holder
    session_id = holder.get_session_id()
    model_seq_id = holder.get_training_client_id()

    async def _create_async():
        start_time = time.time()
        with holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            request = tinker_types.CreateModelRequest(
                session_id=session_id,
                model_seq_id=model_seq_id,
                base_model=base_model,
                lora_config=None,
                user_metadata=user_metadata,
            )
            future = await client.models.create(request=request)
        create_model_response = await _APIFuture(
            tinker_types.CreateModelResponse,
            holder,
            future,
            request_start_time=start_time,
            request_type="CreateModel",
            queue_state_observer=QueueStateLogger(base_model, "Model creation"),
        ).result_async()
        model_id = create_model_response.model_id
        training_client = TrainingClient(holder, model_seq_id=model_seq_id, model_id=model_id)
        logger.info("Full-param TrainingClient initialized for model %s", model_id)
        return training_client

    return holder.run_coroutine_threadsafe(_create_async()).result()


def _resolve_config_for_launch(
    config: Optional[str | Path],
    model: Optional[str | Path],
    host: str,
    port: int,
) -> Optional[Path]:
    """Resolve config file path for launching embedded server.

    Priority:
    1. Explicit config argument
    2. TUFT_CONFIG env var
    3. model argument -> auto-generate config
    4. TUFT_MODEL_PATH env var -> auto-generate config
    5. Default config file (~/.tuft/configs/tuft_config.yaml)
    """
    global _api_key

    # 1. Explicit config
    if config is not None:
        path = Path(config)
        if not path.exists():
            raise RuntimeError(f"Config file not found: {path}")
        return path

    # 2. TUFT_CONFIG env var
    env_config = os.environ.get(ENV_TUFT_CONFIG)
    if env_config:
        path = Path(env_config)
        if path.exists():
            return path
        logger.warning("TUFT_CONFIG=%s does not exist, skipping", env_config)

    # 3. model argument -> auto-generate
    if model is not None:
        config_path, api_key = generate_config_file(model, host=host, port=port)
        _api_key = api_key
        _save_credentials(api_key)
        return config_path

    # 4. TUFT_MODEL_PATH env var
    env_model = os.environ.get(ENV_TUFT_MODEL_PATH)
    if env_model:
        config_path, api_key = generate_config_file(env_model, host=host, port=port)
        _api_key = api_key
        _save_credentials(api_key)
        return config_path

    # 5. Default config file
    default_config = get_default_config_path()
    if default_config.exists():
        return default_config

    return None


def _save_credentials(api_key: str) -> None:
    """Save auto-generated API key to credentials file."""
    creds_file = get_credentials_file()
    try:
        creds_file.parent.mkdir(parents=True, exist_ok=True)
        creds_file.write_text(api_key)
        # Set restrictive permissions
        creds_file.chmod(0o600)
        logger.debug("Saved credentials to %s", creds_file)
    except OSError as e:
        logger.warning("Failed to save credentials: %s", e)
