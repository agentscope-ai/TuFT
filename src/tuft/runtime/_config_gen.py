"""Auto-generate a minimal AppConfig from a model path.

Used by the embedded mode to create a working configuration without user
intervention. Detects GPU count/memory and infers reasonable defaults.
"""

from __future__ import annotations

import json
import logging
import secrets
import tempfile
from pathlib import Path
from typing import Optional

from ._constants import get_default_checkpoint_dir


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU detection helpers
# ---------------------------------------------------------------------------


def _detect_gpu_count() -> int:
    """Return the number of available NVIDIA GPUs, or 0 if none."""
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return count
    except Exception:
        return 0


def _detect_gpu_memory_gb() -> float:
    """Return the memory (GB) of the first GPU, or 0.0 if unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.total / (1024**3)  # type: ignore[operator]
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Model metadata helpers
# ---------------------------------------------------------------------------


def _read_max_position_embeddings(model_path: Path) -> Optional[int]:
    """Try to read max_position_embeddings from config.json in the model directory."""
    config_file = model_path / "config.json"
    if not config_file.exists():
        return None
    try:
        with open(config_file) as f:
            data = json.load(f)
        # Common keys in HuggingFace model configs
        for key in ("max_position_embeddings", "n_positions", "seq_length"):
            if key in data:
                return int(data[key])
    except (json.JSONDecodeError, OSError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def _infer_tensor_parallel_size(gpu_count: int, gpu_memory_gb: float) -> int:
    """Infer a reasonable tensor_parallel_size.

    Heuristic: use all GPUs if model likely needs multiple (>40GB models),
    otherwise default to 1.
    """
    if gpu_count <= 1:
        return 1
    # For now, default to 1 for simplicity; users can override via config
    return 1


def _infer_max_model_len(model_path: Path) -> int:
    """Infer max_model_len from model config or use a conservative default."""
    max_pos = _read_max_position_embeddings(model_path)
    if max_pos is not None:
        # Cap at 32768 to avoid OOM on smaller GPUs
        return min(max_pos, 32768)
    return 4096  # conservative default


def generate_api_key() -> str:
    """Generate a random local API key."""
    return f"tml-{secrets.token_hex(16)}"


def generate_config_dict(
    model_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 10610,
    checkpoint_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
) -> dict:
    """Generate a minimal AppConfig dict suitable for YAML serialization.

    Args:
        model_path: Path to the model directory or HuggingFace model ID.
        host: Server bind address.
        port: Server bind port.
        checkpoint_dir: Where to store checkpoints.
        api_key: API key for auth; auto-generated if None.

    Returns:
        A dict that can be passed to AppConfig.model_validate().
    """
    model_path = Path(model_path)
    model_name = model_path.name  # Use directory name as model_name

    gpu_count = _detect_gpu_count()
    gpu_memory_gb = _detect_gpu_memory_gb()
    tp_size = _infer_tensor_parallel_size(gpu_count, gpu_memory_gb)
    max_model_len = _infer_max_model_len(model_path)

    if checkpoint_dir is None:
        checkpoint_dir = get_default_checkpoint_dir()

    if api_key is None:
        api_key = generate_api_key()

    config = {
        "checkpoint_dir": str(checkpoint_dir),
        "authorized_users": {api_key: "local"},  # pragma: allowlist secret
        "supported_models": [
            {
                "model_name": model_name,
                "model_path": str(model_path),
                "max_model_len": max_model_len,
                "tensor_parallel_size": tp_size,
            }
        ],
    }

    logger.info(
        "Generated minimal config: model=%s, tp_size=%d, max_model_len=%d, gpus_detected=%d",
        model_name,
        tp_size,
        max_model_len,
        gpu_count,
    )
    return config


def generate_config_file(
    model_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 10610,
    checkpoint_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
) -> tuple[Path, str]:
    """Generate a temporary YAML config file.

    Returns:
        A tuple of (config_file_path, api_key).
    """
    from omegaconf import OmegaConf

    if api_key is None:
        api_key = generate_api_key()

    config_dict = generate_config_dict(
        model_path,
        host=host,
        port=port,
        checkpoint_dir=checkpoint_dir,
        api_key=api_key,
    )

    # Write to a temp file that persists until process exit
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="tuft_auto_",
        delete=False,
    )
    conf = OmegaConf.create(config_dict)
    tmp.write(OmegaConf.to_yaml(conf))
    tmp.close()

    logger.info("Generated temporary config file: %s", tmp.name)
    return Path(tmp.name), api_key
