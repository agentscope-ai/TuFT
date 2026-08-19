"""CPU unit tests for vLLM LoRA checkpoint compatibility helpers.

Covers the shared helpers in tuft.backends.vllm_lora_compat, the FSDP
adapter-weight writer, and the HF alias export (key layout, failure
propagation, and atomic rewrite).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


NATIVE_KEY = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
ALIASED_KEY = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"


def _write_model_config(model_dir: Path, model_type: str, architectures: list[str]) -> None:
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": architectures}),
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# vllm_lora_compat helpers
# -----------------------------------------------------------------------------


def test_resolve_model_series_prefers_config_json_over_path(tmp_path):
    from tuft.backends.vllm_lora_compat import resolve_model_series

    # Directory NOT named after the model family resolves via config.json.
    qwen_dir = tmp_path / "checkpoint-1000"
    qwen_dir.mkdir()
    _write_model_config(qwen_dir, "qwen3_5", ["Qwen3_5ForConditionalGeneration"])
    assert resolve_model_series(qwen_dir) == "qwen"

    # config.json wins over a misleading path substring.
    mistral_dir = tmp_path / "qwen_models" / "mistral-7b"
    mistral_dir.mkdir(parents=True)
    _write_model_config(mistral_dir, "mistral", ["MistralForCausalLM"])
    assert resolve_model_series(mistral_dir) is None

    llama_dir = tmp_path / "sft-final"
    llama_dir.mkdir()
    _write_model_config(llama_dir, "llama", ["LlamaForCausalLM"])
    assert resolve_model_series(llama_dir) == "llama"


def test_resolve_model_series_falls_back_to_path_substring(tmp_path):
    from tuft.backends.vllm_lora_compat import resolve_model_series

    # No config.json: fall back to the path substring.
    assert resolve_model_series(tmp_path / "Qwen3-4B") == "qwen"
    assert resolve_model_series(tmp_path / "Llama-3.1-8B") == "llama"
    assert resolve_model_series(tmp_path / "gemma-2b") is None


def test_vllm_nests_language_model_from_architectures(tmp_path):
    from tuft.backends.vllm_lora_compat import vllm_nests_language_model

    nested = tmp_path / "qwen3_5"
    nested.mkdir()
    _write_model_config(nested, "qwen3_5", ["Qwen3_5ForConditionalGeneration"])
    assert vllm_nests_language_model(nested) is True

    flat = tmp_path / "qwen3"
    flat.mkdir()
    _write_model_config(flat, "qwen3", ["Qwen3ForCausalLM"])
    assert vllm_nests_language_model(flat) is False


def test_vllm_nests_language_model_falls_back_to_qwen35_model_ids(tmp_path):
    """Hub IDs without a local config nest exactly when target resolution says qwen3_5.

    Alias export and target resolution must agree, or a hub-ID Qwen3.5 model
    would train linear_attn adapters whose checkpoints vLLM silently ignores.
    """
    from tuft.backends.vllm_lora_compat import vllm_nests_language_model

    assert vllm_nests_language_model("Qwen/Qwen3.5-4B") is True
    assert vllm_nests_language_model("Qwen/Qwen3.6-27B") is True
    assert vllm_nests_language_model("Qwen/Qwen3.8-27B") is True
    assert vllm_nests_language_model("Qwen/Qwen3-8B") is False
    assert vllm_nests_language_model("someorg/qwen3_8b-sft") is False

    # No config.json and no recognized ID -> no aliases (safe default).
    assert vllm_nests_language_model(tmp_path / "missing") is False


def test_add_language_model_aliases_clones_tensors():
    from tuft.backends.vllm_lora_compat import add_language_model_aliases

    tensor = torch.randn(4, 4)
    state = {NATIVE_KEY: tensor, "base_model.model.other.weight": torch.randn(2)}
    aliased = add_language_model_aliases(state)

    assert set(aliased) == {NATIVE_KEY, ALIASED_KEY, "base_model.model.other.weight"}
    assert torch.equal(aliased[ALIASED_KEY], tensor)
    # Clone is mandatory: safetensors rejects keys sharing storage.
    assert aliased[ALIASED_KEY].data_ptr() != tensor.data_ptr()


# -----------------------------------------------------------------------------
# FSDP adapter-weight writer (extracted from MultiAdapterFSDPWorker.save_checkpoint)
# -----------------------------------------------------------------------------


def test_write_adapter_weights_file_prefers_safetensors_and_removes_stale_bin(tmp_path):
    from tuft.backends.fsdp_training_backend import _write_adapter_weights_file

    logger = logging.getLogger("test")
    peft_state = {NATIVE_KEY: torch.randn(4, 4), ALIASED_KEY: torch.randn(4, 4)}
    # Simulate a stale .bin from an older save at the same checkpoint path.
    (tmp_path / "adapter_model.bin").write_bytes(b"stale")

    _write_adapter_weights_file(logger, "adapter_r8_0", tmp_path, peft_state)

    saved = tmp_path / "adapter_model.safetensors"
    assert saved.exists()
    assert not (tmp_path / "adapter_model.bin").exists()
    loaded = load_file(str(saved))
    assert set(loaded) == set(peft_state)


def test_write_adapter_weights_file_falls_back_to_bin_without_shared_storage(tmp_path):
    from tuft.backends.fsdp_training_backend import _write_adapter_weights_file

    logger = logging.getLogger("test")
    tensor = torch.randn(4, 4)
    # Shared-storage keys make safetensors.save_file raise; the writer must
    # log and fall back to .bin, and must not leave a stale safetensors file.
    peft_state = {NATIVE_KEY: tensor, ALIASED_KEY: tensor}
    (tmp_path / "adapter_model.safetensors").write_bytes(b"stale")

    _write_adapter_weights_file(logger, "adapter_r8_0", tmp_path, peft_state)

    assert (tmp_path / "adapter_model.bin").exists()
    assert not (tmp_path / "adapter_model.safetensors").exists()
    loaded = torch.load(tmp_path / "adapter_model.bin", weights_only=True)
    assert set(loaded) == set(peft_state)


# -----------------------------------------------------------------------------
# HF alias export
# -----------------------------------------------------------------------------


def _make_adapter_dir(tmp_path: Path) -> Path:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    save_file({NATIVE_KEY: torch.randn(4, 4)}, str(adapter_dir / "adapter_model.safetensors"))
    return adapter_dir


def test_hf_alias_export_adds_aliases_for_nested_architecture(tmp_path):
    from tuft.backends.hf_training_model import _export_vllm_compatible_lora_aliases

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_model_config(model_dir, "qwen3_5", ["Qwen3_5ForConditionalGeneration"])
    adapter_dir = _make_adapter_dir(tmp_path)

    _export_vllm_compatible_lora_aliases(adapter_dir, str(model_dir))

    loaded = load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert set(loaded) == {NATIVE_KEY, ALIASED_KEY}
    # No leftover temp file from the atomic rewrite.
    assert not (adapter_dir / "adapter_model.safetensors.tmp").exists()


def test_hf_alias_export_is_noop_for_flat_architecture(tmp_path):
    from tuft.backends.hf_training_model import _export_vllm_compatible_lora_aliases

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_model_config(model_dir, "qwen3", ["Qwen3ForCausalLM"])
    adapter_dir = _make_adapter_dir(tmp_path)

    _export_vllm_compatible_lora_aliases(adapter_dir, str(model_dir))

    # Non-nested architectures keep native keys only (no redundant aliases).
    loaded = load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert set(loaded) == {NATIVE_KEY}


def test_hf_alias_export_failure_propagates_and_keeps_original(tmp_path, monkeypatch):
    import safetensors.torch as safetensors_torch

    from tuft.backends.hf_training_model import _export_vllm_compatible_lora_aliases

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_model_config(model_dir, "qwen3_5", ["Qwen3_5ForConditionalGeneration"])
    adapter_dir = _make_adapter_dir(tmp_path)

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(safetensors_torch, "save_file", _boom)

    # Failure must propagate (save_state fails loudly) and the original
    # checkpoint must remain intact and loadable.
    with pytest.raises(OSError, match="disk full"):
        _export_vllm_compatible_lora_aliases(adapter_dir, str(model_dir))

    loaded = load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert set(loaded) == {NATIVE_KEY}
