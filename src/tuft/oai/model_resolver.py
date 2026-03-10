"""Resolve the 'model' field from OpenAI requests to backend routing info."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..checkpoints import CheckpointRecord
from ..config import AppConfig


logger = logging.getLogger(__name__)


@dataclass
class ResolvedModel:
    """Result of resolving the 'model' field in an OpenAI request."""

    base_model: str
    """Backend key matching a supported_models entry (e.g. "Qwen/Qwen3-8B")."""

    backend_model_name: str
    """Model name to send to the vLLM OpenAI server (base model path or LoRA name)."""

    lora_adapter_path: Optional[Path] = None
    """Filesystem path to the LoRA adapter, if applicable."""

    lora_id: Optional[str] = None
    """Identifier for the LoRA adapter (sampling session id)."""


def resolve_model(
    model_field: str,
    config: AppConfig,
) -> ResolvedModel:
    """Resolve the ``model`` field from an OpenAI API request.

    Supports two formats:
    - ``tinker://uuid:train:0/sampler_weights/000080`` → checkpoint-based resolution
    - ``Qwen/Qwen3-8B`` → direct base model match

    Raises:
        ValueError: If the model cannot be resolved.
    """
    supported_names = {m.model_name for m in config.supported_models}
    model_paths = {str(m.model_path): m.model_name for m in config.supported_models}

    # --- Direct base model match ---
    if model_field in supported_names:
        model_cfg = next(m for m in config.supported_models if m.model_name == model_field)
        return ResolvedModel(
            base_model=model_field,
            backend_model_name=str(model_cfg.model_path),
        )

    # --- tinker:// path resolution ---
    if model_field.startswith("tinker://"):
        if config.checkpoint_dir is None:
            raise ValueError("Cannot resolve tinker:// path: checkpoint_dir is not configured.")
        try:
            parsed_checkpoint = CheckpointRecord.from_tinker_path(
                model_field,
                config.checkpoint_dir,
            )
        except FileNotFoundError as exc:
            raise ValueError(f"Checkpoint not found for model: {model_field}") from exc

        metadata = parsed_checkpoint.metadata
        base_model_ref = metadata.base_model

        if base_model_ref not in supported_names:
            raise ValueError(
                f"Base model '{base_model_ref}' from checkpoint is not in supported_models."
            )

        adapter_path = parsed_checkpoint.adapter_path
        lora_id = parsed_checkpoint.training_run_id

        if adapter_path.exists():
            # vLLM expects the lora_name (lora_id) as the model field,
            # not the filesystem path. The LoRA is registered under lora_id
            # via the OAI router's _ensure_lora_loaded().
            return ResolvedModel(
                base_model=base_model_ref,
                backend_model_name=lora_id,
                lora_adapter_path=adapter_path,
                lora_id=lora_id,
            )
        else:
            return ResolvedModel(
                base_model=base_model_ref,
                backend_model_name=str(
                    next(
                        m.model_path
                        for m in config.supported_models
                        if m.model_name == base_model_ref
                    )
                ),
            )

    # --- Try matching against model paths ---
    if model_field in model_paths:
        base_model = model_paths[model_field]
        return ResolvedModel(
            base_model=base_model,
            backend_model_name=model_field,
        )

    raise ValueError(
        f"Unknown model: '{model_field}'. "
        f"Supported models: {sorted(supported_names)}. "
        "Or use a tinker:// checkpoint path."
    )
