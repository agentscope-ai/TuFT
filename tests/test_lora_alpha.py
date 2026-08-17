"""Tests for the shared LoRA alpha ratio (issue #142).

The Tinker ``LoraConfig`` carries only a rank, so ``lora_alpha`` comes from
``ModelConfig.lora_alpha_ratio``. These tests pin the two properties that
setting exists for: the "hf" and "fsdp" training backends must derive the same
alpha from the same rank, and a checkpoint trained at one alpha must not be
loaded into an adapter built with another.

All tests here run on CPU without a model.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tinker import types

from tuft.checkpoints import CheckpointRecord
from tuft.config import (
    DEFAULT_LORA_ALPHA_RATIO,
    AppConfig,
    ModelConfig,
    compute_lora_alpha,
    load_yaml_config,
)
from tuft.exceptions import CheckpointIncompatibleException


RANKS = [1, 4, 8, 16, 32, 64]
RATIOS = [1, 2, 4]


def _model_config(**overrides) -> ModelConfig:
    kwargs = {
        "model_name": "Qwen/Qwen3-4B",
        "model_path": Path("Qwen/Qwen3-4B"),
        "max_model_len": 1024,
    }
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


# -----------------------------------------------------------------------------
# Shared helper and configuration
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("ratio", RATIOS)
def test_compute_lora_alpha_scales_rank_by_ratio(rank: int, ratio: int):
    assert compute_lora_alpha(rank, ratio) == rank * ratio


def test_compute_lora_alpha_defaults_to_shared_ratio():
    assert compute_lora_alpha(8) == 8 * DEFAULT_LORA_ALPHA_RATIO


def test_model_config_defaults_to_shared_ratio():
    """The default is 2 for every backend, including "hf", which used 1 before."""
    assert _model_config().lora_alpha_ratio == DEFAULT_LORA_ALPHA_RATIO == 2


def test_model_config_accepts_ratio_one_for_legacy_hf_deployments():
    assert _model_config(lora_alpha_ratio=1).lora_alpha_ratio == 1


@pytest.mark.parametrize("ratio", [0, -1])
def test_model_config_rejects_ratio_below_one(ratio: int):
    with pytest.raises(ValueError, match="lora_alpha_ratio must be >= 1"):
        _model_config(lora_alpha_ratio=ratio)


def test_lora_alpha_ratio_round_trips_through_yaml(tmp_path):
    """The setting survives the real YAML load path and persistence dump."""
    config_path = tmp_path / "tuft_config.yaml"
    config_path.write_text(
        "supported_models:\n"
        "  - model_name: Qwen/Qwen3-4B\n"
        "    model_path: Qwen/Qwen3-4B\n"
        "    max_model_len: 1024\n"
        "    lora_alpha_ratio: 4\n",
        encoding="utf-8",
    )

    loaded = load_yaml_config(config_path)
    assert loaded.supported_models[0].lora_alpha_ratio == 4

    dumped = loaded.get_config_for_persistence()
    assert dumped["supported_models"][0]["lora_alpha_ratio"] == 4
    assert AppConfig.model_validate(dumped).supported_models[0].lora_alpha_ratio == 4


def test_yaml_without_the_setting_gets_the_default(tmp_path):
    config_path = tmp_path / "tuft_config.yaml"
    config_path.write_text(
        "supported_models:\n"
        "  - model_name: Qwen/Qwen3-4B\n"
        "    model_path: Qwen/Qwen3-4B\n"
        "    max_model_len: 1024\n",
        encoding="utf-8",
    )

    loaded = load_yaml_config(config_path)
    assert loaded.supported_models[0].lora_alpha_ratio == DEFAULT_LORA_ALPHA_RATIO


# -----------------------------------------------------------------------------
# Backend parity: the reason the setting is shared
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("ratio", RATIOS)
def test_hf_and_fsdp_derive_the_same_alpha(rank: int, ratio: int):
    """Selecting a backend must not change LoRA update scaling."""
    from tuft.backends.fsdp_training_backend import SlotPoolConfig
    from tuft.backends.hf_training_model import build_peft_lora_config

    hf_config = build_peft_lora_config(
        "Qwen/Qwen3-4B", types.LoraConfig(rank=rank), lora_alpha_ratio=ratio
    )
    fsdp_alpha = SlotPoolConfig(rank_slots={rank: 1}, lora_alpha_ratio=ratio).get_lora_alpha(rank)

    assert hf_config.lora_alpha == fsdp_alpha == rank * ratio
    assert hf_config.r == rank


def test_hf_peft_config_defaults_to_shared_ratio():
    from tuft.backends.hf_training_model import build_peft_lora_config

    config = build_peft_lora_config("Qwen/Qwen3-4B", types.LoraConfig(rank=16))
    assert config.lora_alpha == 16 * DEFAULT_LORA_ALPHA_RATIO


@pytest.mark.parametrize("ratio", RATIOS)
def test_config_to_worker_dict_propagates_lora_alpha_ratio(ratio: int):
    """The ratio must reach the Ray worker; it is not re-derived there."""
    from tuft.backends.fsdp_training_backend import (
        _config_to_worker_dict,
        _worker_dict_to_configs,
    )

    worker_dict = _config_to_worker_dict(
        _model_config(model_path=Path("/tmp/model"), lora_alpha_ratio=ratio, max_lora_rank=8)
    )
    assert worker_dict["slot_config"]["lora_alpha_ratio"] == ratio

    _, slot_config = _worker_dict_to_configs(worker_dict)
    assert slot_config.lora_alpha_ratio == ratio
    assert slot_config.get_lora_alpha(8) == 8 * ratio


@pytest.mark.parametrize("ratio", RATIOS)
def test_fsdp_backend_slot_pool_honours_configured_ratio(ratio: int):
    """The single-process (TUFT_FSDP_NO_RAY=1) path builds slots from the same dict.

    It used to construct its own SlotPoolConfig, which silently kept the dataclass
    default ratio and ignored the model configuration.
    """
    from tuft.backends.fsdp_training_backend import (
        FSDPTrainingBackend,
        _worker_dict_to_configs,
    )

    backend = FSDPTrainingBackend(
        _model_config(
            model_path=Path("/tmp/model"),
            training_backend="fsdp",
            lora_alpha_ratio=ratio,
            max_lora_rank=8,
        )
    )

    _, slot_config = _worker_dict_to_configs(backend._config_dict)
    assert slot_config.lora_alpha_ratio == ratio
    assert slot_config.get_lora_alpha(8) == 8 * ratio


# -----------------------------------------------------------------------------
# Checkpoint metadata and load validation
# -----------------------------------------------------------------------------


def _checkpoint(tmp_path: Path) -> CheckpointRecord:
    return CheckpointRecord.from_training_run(
        training_run_id="run-1",
        checkpoint_name="checkpoint-0001",
        owner_name="tester",
        checkpoint_type="training",
        checkpoint_root_dir=tmp_path,
    )


def _write_adapter_config(record: CheckpointRecord, lora_alpha: int) -> None:
    """Write the peft file a real save_state produces, with only what we read."""
    record.adapter_path.mkdir(parents=True, exist_ok=True)
    (record.adapter_path / "adapter_config.json").write_text(
        f'{{"peft_type": "LORA", "r": 8, "lora_alpha": {lora_alpha}}}', encoding="utf-8"
    )


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("ratio", RATIOS)
def test_save_metadata_records_effective_alpha(tmp_path, rank: int, ratio: int):
    record = _checkpoint(tmp_path)
    record.save_metadata(
        base_model="Qwen/Qwen3-4B",
        session_id="session-1",
        lora_rank=rank,
        lora_alpha=compute_lora_alpha(rank, ratio),
    )

    assert record.metadata.lora_rank == rank
    assert record.metadata.lora_alpha == rank * ratio
    assert record.saved_lora_alpha == rank * ratio


def test_set_visibility_preserves_recorded_alpha(tmp_path):
    """set_visibility rewrites metadata.json and must not drop the alpha."""
    record = _checkpoint(tmp_path)
    record.save_metadata(
        base_model="Qwen/Qwen3-4B", session_id="session-1", lora_rank=8, lora_alpha=16
    )

    record.set_visibility(True)

    assert record.metadata.public is True
    assert record.metadata.lora_alpha == 16


def test_saved_alpha_falls_back_to_adapter_config(tmp_path):
    """Checkpoints written before metadata.lora_alpha existed stay readable."""
    record = _checkpoint(tmp_path)
    record.save_metadata(base_model="Qwen/Qwen3-4B", session_id="session-1", lora_rank=8)
    _write_adapter_config(record, lora_alpha=8)

    assert record.metadata.lora_alpha is None
    assert record.saved_lora_alpha == 8


def test_metadata_alpha_wins_over_adapter_config(tmp_path):
    record = _checkpoint(tmp_path)
    record.save_metadata(
        base_model="Qwen/Qwen3-4B", session_id="session-1", lora_rank=8, lora_alpha=16
    )
    _write_adapter_config(record, lora_alpha=8)

    assert record.saved_lora_alpha == 16


def test_saved_alpha_is_none_when_nothing_records_it(tmp_path):
    record = _checkpoint(tmp_path)

    assert record.saved_lora_alpha is None


def test_validate_accepts_matching_alpha(tmp_path):
    record = _checkpoint(tmp_path)
    record.save_metadata(
        base_model="Qwen/Qwen3-4B", session_id="session-1", lora_rank=8, lora_alpha=16
    )

    record.validate_lora_alpha(16)


def test_validate_accepts_unknown_alpha(tmp_path):
    """Nothing to compare against, so the load is allowed to proceed."""
    record = _checkpoint(tmp_path)

    record.validate_lora_alpha(16)


def test_validate_rejects_mismatched_alpha(tmp_path):
    """A ratio-1 'hf' checkpoint must not silently load under the ratio-2 default."""
    record = _checkpoint(tmp_path)
    record.save_metadata(
        base_model="Qwen/Qwen3-4B", session_id="session-1", lora_rank=8, lora_alpha=8
    )

    with pytest.raises(CheckpointIncompatibleException) as excinfo:
        record.validate_lora_alpha(compute_lora_alpha(8, DEFAULT_LORA_ALPHA_RATIO))

    assert excinfo.value.status_code == 400
    assert excinfo.value.checkpoint_id == "checkpoint-0001"
    assert "lora_alpha=8" in excinfo.value.detail
    assert "lora_alpha=16" in excinfo.value.detail
    assert "lora_alpha_ratio" in excinfo.value.detail


def test_validate_rejects_mismatch_found_only_in_adapter_config(tmp_path):
    """The legacy path is validated too, not just checkpoints with new metadata."""
    record = _checkpoint(tmp_path)
    record.save_metadata(base_model="Qwen/Qwen3-4B", session_id="session-1", lora_rank=8)
    _write_adapter_config(record, lora_alpha=8)

    with pytest.raises(CheckpointIncompatibleException):
        record.validate_lora_alpha(16)


def test_validate_accepts_legacy_checkpoint_under_ratio_one(tmp_path):
    """Setting lora_alpha_ratio: 1 restores loading of pre-change 'hf' checkpoints."""
    record = _checkpoint(tmp_path)
    record.save_metadata(base_model="Qwen/Qwen3-4B", session_id="session-1", lora_rank=8)
    _write_adapter_config(record, lora_alpha=8)

    legacy_config = _model_config(lora_alpha_ratio=1)
    record.validate_lora_alpha(compute_lora_alpha(8, legacy_config.lora_alpha_ratio))
