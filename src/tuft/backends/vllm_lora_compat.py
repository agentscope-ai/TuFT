"""Shared helpers for writing LoRA checkpoints that vLLM can load.

Both the HF and FSDP training backends must emit adapter keys that match
the vLLM module tree of the served model. For most architectures the peft
keys (`base_model.model.model.layers.*`) match directly, but some vLLM
implementations (e.g. Qwen3.5's ``Qwen3_5ForConditionalGeneration``) wrap
the text backbone under ``language_model``, and vLLM's hf_to_vllm mapper only
rewrites keys prefixed with ``model.language_model.`` — without matching
keys, vLLM silently ignores ALL adapter weights and serves the base model.

The decision is made from the base model's ``config.json`` ``architectures``
field rather than path substrings, and the helpers here are shared by both
backends so the two key layouts never drift apart.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch


logger = logging.getLogger(__name__)

_NATIVE_PREFIX = "base_model.model.model.layers."
_ALIASED_PREFIX = "base_model.model.model.language_model.layers."


def load_model_config_json(model_path: str | Path) -> dict | None:
    """Read the base model's ``config.json``; return None if missing/unreadable."""
    try:
        with open(Path(model_path) / "config.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def resolve_model_series(model_path: str | Path) -> str | None:
    """Resolve the model series ('qwen'/'llama') for module mapping.

    Prefer ``model_type`` from ``config.json`` so directories not named after
    the model family (e.g. ``checkpoint-1000``, ``sft-final``) resolve
    correctly. When a config exists it is authoritative (an unknown
    ``model_type`` resolves to None instead of falling back to unreliable
    path substrings); the path-substring match is only used when no config
    is available.
    """
    config = load_model_config_json(model_path)
    if config is not None:
        model_type = str(config.get("model_type", "")).lower()
        if model_type.startswith("qwen"):
            return "qwen"
        if model_type.startswith("llama"):
            return "llama"
        return None
    path_lower = str(model_path).lower()
    if "qwen" in path_lower:
        return "qwen"
    if "llama" in path_lower:
        return "llama"
    return None


def resolve_model_architecture(model_path: str | Path) -> str | None:
    """Resolve architecture-specific behavior that a broad model series cannot express.

    Qwen3.5 and Qwen3.8 use the same Transformers architecture (``qwen3_5``),
    including Gated DeltaNet blocks whose projection names differ from standard
    attention. Prefer ``model_type`` for local checkpoints, and retain a narrow
    model-ID fallback for configured Hugging Face IDs whose config has not been
    downloaded locally yet. As with :func:`resolve_model_series`, an existing
    config is authoritative.
    """

    config = load_model_config_json(model_path)
    if config is not None:
        model_type = str(config.get("model_type", "")).lower()
        if model_type.startswith("qwen3_5"):
            return "qwen3_5"
        return None

    path_lower = str(model_path).lower()
    if any(marker in path_lower for marker in ("qwen3.5", "qwen3_5", "qwen3.8", "qwen3_8")):
        return "qwen3_5"
    return None


def vllm_nests_language_model(model_path: str | Path) -> bool:
    """True when the vLLM implementation nests the text backbone under ``language_model``.

    Detected from ``architectures`` in ``config.json``: ``*ForConditionalGeneration``
    classes (e.g. ``Qwen3_5ForConditionalGeneration``) expose
    ``language_model.model.layers.*`` in vLLM, while HF maps the same
    ``model_type`` to a ``*ForCausalLM`` class exposing ``model.layers.*``.
    """
    config = load_model_config_json(model_path)
    if config is None:
        return False
    architectures = config.get("architectures") or []
    return any(str(arch).endswith("ForConditionalGeneration") for arch in architectures)


def add_language_model_aliases(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return a copy of ``state`` with ``language_model.``-prefixed alias keys.

    Alias tensors are cloned: ``safetensors.save_file`` rejects dictionaries
    whose keys share storage, which is exactly what a plain dual-key export
    produces.
    """
    aliased: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        aliased[key] = tensor
        alias = key.replace(_NATIVE_PREFIX, _ALIASED_PREFIX, 1)
        if alias != key:
            aliased[alias] = tensor.clone()
    return aliased
