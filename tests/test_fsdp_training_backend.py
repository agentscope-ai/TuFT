"""
Unit and integration tests for FSDP training backend.

Unit tests (no GPU):
  - Config/slot helpers and async_init validation.
  - Torch-native batching, log-prob extraction, and gradient accumulation on CPU.

Integration tests (GPU, optional TUFT_TEST_MODEL):
  - FSDPTrainingBackend single-process: create_adapter, forward, optim_step, save/load.

Run:
  pytest tests/test_fsdp_training_backend.py                    # unit only
  pytest tests/test_fsdp_training_backend.py --gpu -m gpu      # include GPU/integration
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from tinker import types

from tuft.config import ModelConfig


# -----------------------------------------------------------------------------
# Unit tests: config and slot (no GPU)
# -----------------------------------------------------------------------------


def test_config_to_worker_dict():
    """_config_to_worker_dict returns a serializable dict with expected keys and slot_config."""
    from tuft.backends.fsdp_training_backend import _config_to_worker_dict

    config = ModelConfig(
        model_name="test",
        model_path=Path("/tmp/qwen-model"),
        max_model_len=1024,
        max_lora_rank=8,
    )
    d = _config_to_worker_dict(config)
    assert d["model_path"] == "/tmp/qwen-model"
    assert d["max_model_len"] == 1024
    assert "slot_config" in d
    assert d["slot_config"]["rank_slots"] == {8: 16}
    assert d["slot_config"]["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert d["slot_config"]["lora_alpha_ratio"] == config.lora_alpha_ratio
    assert "fsdp_override_config" in d
    assert isinstance(d["fsdp_override_config"], dict)


@pytest.mark.parametrize(
    ("train_mlp", "expected"),
    [
        (False, ["q_proj", "k_proj", "v_proj", "o_proj"]),
        (
            True,
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    ],
)
def test_config_to_worker_dict_resolves_fsdp_modifiers(train_mlp, expected):
    """Attention-only and attention+MLP pools share HF's model-series map."""
    from tuft.backends.fsdp_training_backend import FSDPTrainingBackend, _config_to_worker_dict

    config = ModelConfig(
        model_name="test",
        model_path=Path("/tmp/qwen-model"),
        max_model_len=1024,
        fsdp_train_attn=True,
        fsdp_train_mlp=train_mlp,
        fsdp_train_unembed=False,
    )

    assert _config_to_worker_dict(config)["slot_config"]["target_modules"] == expected
    FSDPTrainingBackend(config)._validate_lora_config(
        types.LoraConfig(
            rank=8,
            train_attn=True,
            train_mlp=train_mlp,
            train_unembed=False,
        )
    )


def test_config_to_worker_dict_honors_explicit_targets_and_rank_slots():
    """Custom rank pools retain the configured broader target geometry."""
    from tuft.backends.fsdp_training_backend import FSDPTrainingBackend, _config_to_worker_dict

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    config = ModelConfig(
        model_name="custom",
        model_path=Path("/tmp/custom-model"),
        max_model_len=1024,
        max_lora_rank=64,
        training_backend="fsdp",
        fsdp_rank_slots={64: 4},
        fsdp_target_modules=targets,
    )

    slot_config = _config_to_worker_dict(config)["slot_config"]
    assert slot_config["rank_slots"] == {64: 4}
    assert slot_config["target_modules"] == targets
    # Explicit geometry is authoritative even though this custom set cannot be
    # represented by the public modifier flags or resolved from the model path.
    FSDPTrainingBackend(config)._validate_lora_config(types.LoraConfig(rank=64))


def test_default_rank_slots_do_not_vary_with_target_geometry():
    """Widening the geometry must not silently cost concurrent adapter capacity."""
    from tuft.backends.fsdp_training_backend import _config_to_worker_dict

    broad = ModelConfig(
        model_name="broad",
        model_path=Path("/tmp/qwen-model"),
        max_model_len=1024,
        max_lora_rank=16,
        training_backend="fsdp",
    )
    assert _config_to_worker_dict(broad)["slot_config"]["rank_slots"] == {8: 16, 16: 8}

    # The released Q/V pool, a single module, and a two-module set that is far
    # more expensive than Q/V all keep the same capacity: per-slot adapter
    # memory is small next to the base model the slot is attached to.
    narrow = broad.model_copy(update={"fsdp_target_modules": ["q_proj", "v_proj"]})
    q_only = broad.model_copy(update={"fsdp_target_modules": ["q_proj"]})
    wide_pair = broad.model_copy(update={"fsdp_target_modules": ["gate_proj", "up_proj"]})
    for config in (narrow, q_only, wide_pair):
        assert _config_to_worker_dict(config)["slot_config"]["rank_slots"] == {8: 16, 16: 8}

    # A max rank at or below 8 collapses to a single pool at that rank, so no
    # slots are preallocated above the rank the server advertises.
    low_max = ModelConfig(
        model_name="low-max",
        model_path=Path("/tmp/qwen-model"),
        max_model_len=1024,
        max_lora_rank=4,
        training_backend="fsdp",
    )
    assert _config_to_worker_dict(low_max)["slot_config"]["rank_slots"] == {4: 16}


def test_explicit_rank_slots_must_cover_advertised_max_rank():
    with pytest.raises(ValueError, match="must define.*max_lora_rank=16.*configured ranks.*8"):
        ModelConfig(
            model_name="incoherent",
            model_path=Path("/tmp/qwen-model"),
            max_model_len=1024,
            max_lora_rank=16,
            training_backend="fsdp",
            fsdp_rank_slots={8: 1},
        )

    with pytest.raises(ValueError, match="slot counts must be at least 1"):
        ModelConfig(
            model_name="empty-rank",
            model_path=Path("/tmp/qwen-model"),
            max_model_len=1024,
            max_lora_rank=16,
            training_backend="fsdp",
            fsdp_rank_slots={8: 1, 16: 0},
        )


def test_unknown_series_requires_actionable_explicit_geometry():
    """Unsupported implicit geometry fails with the configuration escape hatch."""
    from tuft.backends.fsdp_training_backend import FSDPTrainingBackend

    config = ModelConfig(
        model_name="unsupported",
        model_path=Path("/tmp/mistral-model"),
        max_model_len=1024,
        training_backend="fsdp",
    )
    with pytest.raises(ValueError, match="Cannot infer.*configure fsdp_target_modules explicitly"):
        FSDPTrainingBackend(config)


def test_fsdp_target_modules_are_stripped_before_validation():
    config = ModelConfig(
        model_name="normalized",
        model_path=Path("/tmp/custom-model"),
        max_model_len=1024,
        fsdp_target_modules=[" q_proj ", "v_proj"],
    )
    assert config.fsdp_target_modules == ["q_proj", "v_proj"]

    with pytest.raises(ValueError, match="cannot contain duplicates"):
        ModelConfig(
            model_name="duplicates",
            model_path=Path("/tmp/custom-model"),
            max_model_len=1024,
            fsdp_target_modules=[" q_proj", "q_proj ", "v_proj"],
        )


@pytest.mark.asyncio
async def test_explicit_legacy_geometry_accepts_default_client_on_unknown_series():
    """An explicit Q/V pool can create runs and load released checkpoints."""
    from tuft.backends.fsdp_training_backend import FSDPTrainingBackend

    config = ModelConfig(
        model_name="legacy",
        model_path=Path("/tmp/mistral-model"),
        max_model_len=1024,
        training_backend="fsdp",
        fsdp_target_modules=["q_proj", "v_proj"],
    )
    backend = FSDPTrainingBackend(config)
    backend._worker = MagicMock()
    backend._worker.allocate_slot.return_value = "adapter_r8_0"

    await backend.create_adapter("legacy-run", types.LoraConfig(rank=8))

    assert backend._lora_id_to_adapter_name["legacy-run"] == "adapter_r8_0"


@pytest.mark.asyncio
async def test_explicit_geometry_warns_when_resolvable_client_modifiers_differ(caplog):
    """Explicit migration geometry wins visibly instead of discarding modifiers silently."""
    from tuft.backends.fsdp_training_backend import FSDPTrainingBackend

    config = ModelConfig(
        model_name="legacy",
        model_path=Path("/tmp/qwen-model"),
        max_model_len=1024,
        training_backend="fsdp",
        fsdp_target_modules=["q_proj", "v_proj"],
    )
    backend = FSDPTrainingBackend(config)
    backend._worker = MagicMock()
    backend._worker.allocate_slot.return_value = "adapter_r8_0"

    with caplog.at_level(logging.WARNING, logger=backend.logger.name):
        await backend.create_adapter("legacy-run", types.LoraConfig(rank=8))

    assert backend._lora_id_to_adapter_name["legacy-run"] == "adapter_r8_0"
    assert "explicit fsdp_target_modules" in caplog.text
    assert "train_mlp=True" in caplog.text
    assert "this training run will target ['q_proj', 'v_proj']" in caplog.text


@pytest.mark.asyncio
async def test_create_adapter_rejects_incompatible_client_modifiers_before_init():
    """A request cannot silently allocate a slot with different target geometry."""
    from tuft.backends.fsdp_training_backend import FSDPTrainingBackend
    from tuft.exceptions import InvalidRequestException

    config = ModelConfig(
        model_name="test",
        model_path=Path("/tmp/qwen-model"),
        max_model_len=1024,
        training_backend="fsdp",
        fsdp_train_attn=True,
        fsdp_train_mlp=False,
        fsdp_train_unembed=False,
    )
    backend = FSDPTrainingBackend(config)
    backend.async_init = MagicMock(side_effect=AssertionError("must validate before init"))

    with pytest.raises(InvalidRequestException, match="target-module mismatch.*train_mlp=True"):
        await backend.create_adapter(
            "incompatible",
            types.LoraConfig(rank=8, train_attn=True, train_mlp=True, train_unembed=False),
        )


@pytest.mark.asyncio
async def test_create_adapter_rejects_unconfigured_rank_before_init():
    """A discrete FSDP pool reports its configured ranks as a request error."""
    from tuft.backends.fsdp_training_backend import FSDPTrainingBackend
    from tuft.exceptions import InvalidRequestException

    config = ModelConfig(
        model_name="test",
        model_path=Path("/tmp/qwen-model"),
        max_model_len=1024,
        max_lora_rank=16,
        training_backend="fsdp",
        fsdp_rank_slots={8: 1, 16: 1},
    )
    backend = FSDPTrainingBackend(config)
    backend.async_init = MagicMock(side_effect=AssertionError("must reject before init"))

    with pytest.raises(InvalidRequestException, match=r"rank 4 is not configured.*\[8, 16\]"):
        await backend.create_adapter("unsupported-rank", types.LoraConfig(rank=4))
    backend.async_init.assert_not_called()


@pytest.mark.asyncio
async def test_create_adapter_reports_local_slot_exhaustion_as_429():
    from tuft.backends.fsdp_training_backend import FSDPTrainingBackend
    from tuft.exceptions import ResourceExhaustedException

    config = ModelConfig(
        model_name="test",
        model_path=Path("/tmp/qwen-model"),
        max_model_len=1024,
        max_lora_rank=8,
        training_backend="fsdp",
        fsdp_rank_slots={8: 1},
    )
    backend = FSDPTrainingBackend(config)
    backend._worker = MagicMock()
    backend._worker.allocate_slot.return_value = None

    with pytest.raises(ResourceExhaustedException, match="All 1.*rank 8.*in use") as exc_info:
        await backend.create_adapter("exhausted", types.LoraConfig(rank=8))
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_create_adapter_reports_ray_slot_exhaustion_as_429(monkeypatch):
    """The Ray leader's None result uses the same typed exhaustion path."""
    import ray

    from tuft.backends.fsdp_training_backend import FSDPTrainingBackend
    from tuft.exceptions import ResourceExhaustedException

    config = ModelConfig(
        model_name="test",
        model_path=Path("/tmp/qwen-model"),
        max_model_len=1024,
        max_lora_rank=8,
        training_backend="fsdp",
        fsdp_rank_slots={8: 1},
    )
    backend = FSDPTrainingBackend(config)
    actor = MagicMock()
    actor.allocate_slot.remote.return_value = object()
    backend._actors = [actor]
    backend._world_size = 1
    monkeypatch.setattr(ray, "get", lambda _ref: None)

    with pytest.raises(ResourceExhaustedException, match="All 1.*rank 8.*in use") as exc_info:
        await backend.create_adapter("exhausted", types.LoraConfig(rank=8))
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_load_state_validates_checkpoint_target_geometry(tmp_path):
    """FSDP load_state rejects a partial match instead of leaving random modules."""
    from tuft.backends.fsdp_training_backend import FSDPTrainingBackend
    from tuft.checkpoints import CheckpointRecord
    from tuft.exceptions import InvalidRequestException

    config = ModelConfig(
        model_name="test",
        model_path=Path("/tmp/qwen-model"),
        max_model_len=1024,
        training_backend="fsdp",
    )
    backend = FSDPTrainingBackend(config)
    backend._lora_id_to_adapter_name["run"] = "adapter_r8_0"
    backend._worker = MagicMock()
    checkpoint = CheckpointRecord(
        checkpoint_id="checkpoint-0001",
        owner_name="tester",
        checkpoint_type="training",
        training_run_id="source",
        path=tmp_path / "checkpoint-0001",
    )
    checkpoint.adapter_path.mkdir(parents=True)
    (checkpoint.adapter_path / "adapter_config.json").write_text(
        json.dumps({"target_modules": ["q_proj", "v_proj"]}),
        encoding="utf-8",
    )

    with pytest.raises(InvalidRequestException, match="targeting modules.*q_proj.*v_proj"):
        await backend.load_state("run", checkpoint, optimizer=False)
    backend._worker.load_checkpoint.assert_not_called()

    (checkpoint.adapter_path / "adapter_config.json").write_text(
        json.dumps({"target_modules": backend._slot_config.target_modules}),
        encoding="utf-8",
    )
    await backend.load_state("run", checkpoint, optimizer=False)
    backend._worker.load_checkpoint.assert_called_once_with(
        "adapter_r8_0",
        checkpoint.adapter_path,
        backend._slot_config.target_modules,
        False,
    )

    regex_checkpoint = CheckpointRecord(
        checkpoint_id="checkpoint-regex",
        owner_name="tester",
        checkpoint_type="training",
        training_run_id="source",
        path=tmp_path / "checkpoint-regex",
    )
    regex_checkpoint.adapter_path.mkdir(parents=True)
    (regex_checkpoint.adapter_path / "adapter_config.json").write_text(
        json.dumps({"target_modules": ".*\\.q_proj"}),
        encoding="utf-8",
    )
    with pytest.raises(InvalidRequestException, match="regex-string targets"):
        await backend.load_state("run", regex_checkpoint, optimizer=False)

    # Current metadata is a functional fallback when adapter_config.json is
    # missing; the validated module list is handed to the worker instead of
    # making the worker reread the absent file.
    backend._worker.load_checkpoint.reset_mock()
    (checkpoint.adapter_path / "adapter_config.json").unlink()
    checkpoint.save_metadata(
        base_model="test",
        session_id="session",
        lora_rank=8,
        target_modules=backend._slot_config.target_modules,
    )
    await backend.load_state("run", checkpoint, optimizer=False)
    backend._worker.load_checkpoint.assert_called_once_with(
        "adapter_r8_0",
        checkpoint.adapter_path,
        backend._slot_config.target_modules,
        False,
    )


def test_worker_load_checkpoint_validates_supplied_geometry_before_weights(tmp_path):
    """The worker independently rejects a backend/slot geometry disagreement."""
    from tuft.backends.fsdp_engine import FSDPModelConfig
    from tuft.backends.fsdp_training_backend import (
        AdapterInfo,
        MultiAdapterFSDPWorker,
        SlotPoolConfig,
    )

    worker = MultiAdapterFSDPWorker(
        FSDPModelConfig(path="/tmp/model", max_model_len=1024),
        SlotPoolConfig(rank_slots={8: 1}, target_modules=["q_proj", "v_proj"]),
    )
    worker._adapters["adapter_r8_0"] = AdapterInfo(
        name="adapter_r8_0",
        rank=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
    )

    with pytest.raises(RuntimeError, match="targeting modules.*q_proj"):
        worker.load_checkpoint(
            "adapter_r8_0",
            tmp_path,
            ["q_proj"],
            optimizer=False,
        )


def test_slot_pool_config_get_lora_alpha():
    """SlotPoolConfig.get_lora_alpha returns rank * lora_alpha_ratio."""
    from tuft.backends.fsdp_training_backend import SlotPoolConfig

    cfg = SlotPoolConfig(rank_slots={8: 2}, lora_alpha_ratio=2)
    assert cfg.get_lora_alpha(8) == 16
    assert cfg.get_lora_alpha(16) == 32

    legacy = SlotPoolConfig(rank_slots={8: 2}, lora_alpha_ratio=1)
    assert legacy.get_lora_alpha(8) == 8
    assert legacy.get_lora_alpha(16) == 16


@pytest.mark.asyncio
async def test_async_init_raises_when_no_ray_and_multi_gpu():
    """async_init raises ValueError when TUFT_FSDP_NO_RAY=1 and fsdp_num_gpus != 1."""
    from tuft.backends.fsdp_training_backend import FSDPTrainingBackend

    prev = os.environ.get("TUFT_FSDP_NO_RAY")
    os.environ["TUFT_FSDP_NO_RAY"] = "1"
    try:
        config = ModelConfig(
            model_name="test",
            model_path=Path("/tmp/qwen-model"),
            max_model_len=1024,
            training_backend="fsdp",
            fsdp_num_gpus=2,
        )
        backend = FSDPTrainingBackend(config)
        with pytest.raises(ValueError, match="TUFT_FSDP_NO_RAY=1.*fsdp_num_gpus=1"):
            await backend.async_init()
    finally:
        if prev is None:
            os.environ.pop("TUFT_FSDP_NO_RAY", None)
        else:
            os.environ["TUFT_FSDP_NO_RAY"] = prev


def test_fsdp_port_allocation_by_index():
    """Ports 29500, 29501, ... by FSDP model order; backends get correct fsdp_index."""
    from tuft.backends.base_backend import BaseTrainingBackend

    # Need real backend path so _fsdp_index is set; restore env after
    saved = os.environ.pop("TUFT_CPU_TEST", None)
    try:
        model_configs = [
            ModelConfig(
                model_name="model_a",
                model_path=Path("/tmp/qwen-a"),
                max_model_len=1024,
                training_backend="fsdp",
            ),
            ModelConfig(
                model_name="model_b",
                model_path=Path("/tmp/b"),
                max_model_len=1024,
                training_backend="hf",
            ),
            ModelConfig(
                model_name="model_c",
                model_path=Path("/tmp/qwen-c"),
                max_model_len=1024,
                training_backend="fsdp",
            ),
        ]
        fsdp_names = [
            c.model_name for c in model_configs if getattr(c, "training_backend", "hf") == "fsdp"
        ]
        backends = {}
        for config in model_configs:
            fsdp_index = (
                fsdp_names.index(config.model_name) if config.model_name in fsdp_names else None
            )
            backends[config.model_name] = BaseTrainingBackend.create_backend(
                config, fsdp_index=fsdp_index
            )
        assert getattr(backends["model_a"], "_fsdp_index", None) == 0
        assert getattr(backends["model_b"], "_fsdp_index", None) is None
        assert getattr(backends["model_c"], "_fsdp_index", None) == 1
    finally:
        if saved is not None:
            os.environ["TUFT_CPU_TEST"] = saved


# -----------------------------------------------------------------------------
# Unit tests: torch-native FSDP engine (CPU)
# -----------------------------------------------------------------------------


def test_prepare_micro_batch_uses_length_masks_and_flat_rolled_labels():
    from tuft.backends.fsdp_engine import _prepare_micro_batch

    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[0, 2, 3]),
            loss_fn_inputs={},
        ),
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[4, 5]),
            loss_fn_inputs={},
        ),
    ]
    batch = _prepare_micro_batch(data, "cpu")
    assert batch.input_ids.tolist() == [[0, 2, 3], [4, 5, 0]]
    # Token id 0 is real in row 0; padding is derived from lengths, not token values.
    assert batch.attention_mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch.position_ids.tolist() == [[0, 1, 2], [0, 1, 2]]
    # roll([0, 2, 3, 4, 5], -1) -> [2, 3, 4, 5, 0]
    assert batch.labels.tolist() == [[2, 3, 4], [5, 0, 0]]
    assert batch.lengths == [3, 2]


def test_prepare_loss_inputs_pads_weights_and_defaults_missing_rows():
    import torch

    from tuft.backends.fsdp_engine import _prepare_loss_fn_inputs, _prepare_micro_batch

    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
            loss_fn_inputs={"weights": types.TensorData(data=[0.0, 1.0, 0.0], dtype="float32")},
        ),
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[4, 5]),
            loss_fn_inputs={},
        ),
    ]
    target_logprobs = torch.randn(2, 3, requires_grad=True)
    inputs = _prepare_loss_fn_inputs(
        data,
        target_logprobs,
        "cross_entropy",
        prepared_target_tokens=_prepare_micro_batch(data, "cpu").labels,
    )
    assert inputs["target_logprobs"] is target_logprobs
    # Row 1 has no weights and no target_tokens -> all-ones fallback, but the
    # flat-roll garbage label at the final position must be zeroed.
    assert inputs["weights"].tolist() == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]


def test_prepare_loss_inputs_uses_prepared_targets_for_mixed_rows():
    import torch

    from tuft.backends.fsdp_engine import _prepare_loss_fn_inputs, _prepare_micro_batch

    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=[9, 8, 7], dtype="int64"),
            },
        ),
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[4, 5]),
            loss_fn_inputs={},
        ),
    ]
    batch = _prepare_micro_batch(data, "cpu")
    target_logprobs = torch.randn(2, 3, requires_grad=True)

    inputs = _prepare_loss_fn_inputs(
        data,
        target_logprobs,
        "cross_entropy",
        prepared_target_tokens=batch.labels,
    )

    assert inputs["target_tokens"] is batch.labels
    assert inputs["target_tokens"].tolist() == [[9, 8, 7], [5, 1, 0]]


def test_prepare_loss_inputs_matches_hf_for_extra_fields_and_overlays_target_logprobs():
    from types import SimpleNamespace
    from typing import cast

    import torch

    from tuft.backends.fsdp_engine import _prepare_loss_fn_inputs, _prepare_micro_batch
    from tuft.backends.hf_training_model import HFTrainingModel

    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=[2, 3, 4], dtype="int64"),
                "target_logprobs": types.TensorData(data=[99.0, 99.0, 99.0], dtype="float32"),
                "reference_logprobs": types.TensorData(data=[-0.2, -0.4, -0.8], dtype="float32"),
                "custom_matrix": types.TensorData(
                    data=[1.0, 2.0, 3.0, 4.0], dtype="float32", shape=[2, 2]
                ),
            },
        ),
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[4, 5]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=[5, 6], dtype="int64"),
                "target_logprobs": types.TensorData(data=[88.0, 88.0], dtype="float32"),
                "reference_logprobs": types.TensorData(data=[-0.1, -0.3], dtype="float32"),
                "custom_matrix": types.TensorData(
                    data=[5.0, 6.0, 7.0], dtype="float32", shape=[1, 3]
                ),
            },
        ),
    ]
    target_logprobs = torch.randn(2, 3, requires_grad=True)
    prepared_target_tokens = _prepare_micro_batch(data, "cpu").labels

    inputs = _prepare_loss_fn_inputs(
        data,
        target_logprobs,
        "dro",
        prepared_target_tokens=prepared_target_tokens,
    )
    hf_model = cast(HFTrainingModel, SimpleNamespace(model=torch.nn.Linear(1, 1)))
    hf_inputs = HFTrainingModel._prepare_loss_fn_inputs(hf_model, data)
    hf_inputs["target_logprobs"] = target_logprobs

    # target_logprobs is asserted by identity above; hf_inputs' copy is written by
    # this test, so comparing the two would assert a tensor against itself.
    assert inputs["target_logprobs"] is target_logprobs
    for key in ("target_tokens", "reference_logprobs", "custom_matrix"):
        torch.testing.assert_close(inputs[key], hf_inputs[key])
    assert inputs["custom_matrix"].tolist() == [
        [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]],
        [[5.0, 6.0, 7.0], [0.0, 0.0, 0.0]],
    ]
    torch.testing.assert_close(inputs["logprobs"], target_logprobs.detach())
    assert inputs["advantages"].tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_prepare_loss_inputs_rejects_rows_missing_a_generic_client_field():
    """Arbitrary fields have no safe default, so every row must supply them."""
    from types import SimpleNamespace
    from typing import cast

    import torch

    from tuft.backends.fsdp_engine import _prepare_loss_fn_inputs, _prepare_micro_batch
    from tuft.backends.hf_training_model import HFTrainingModel

    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[4, 5]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=[5, 1], dtype="int64"),
            },
        ),
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=[2, 3, 4], dtype="int64"),
                "reference_logprobs": types.TensorData(data=[-0.2, -0.4, -0.8], dtype="float32"),
            },
        ),
    ]
    with pytest.raises(ValueError, match="'reference_logprobs' must be present for every datum"):
        _prepare_loss_fn_inputs(
            data,
            torch.randn(2, 3),
            "cross_entropy",
            prepared_target_tokens=_prepare_micro_batch(data, "cpu").labels,
        )

    hf_model = cast(HFTrainingModel, SimpleNamespace(model=torch.nn.Linear(1, 1)))
    with pytest.raises(ValueError, match="'reference_logprobs' must be present for every datum"):
        HFTrainingModel._prepare_loss_fn_inputs(hf_model, data)


def test_hf_prepare_loss_inputs_rejects_mixed_target_tokens():
    """HF must not turn an omitted target into token id zero."""
    from types import SimpleNamespace
    from typing import cast

    import torch

    from tuft.backends.hf_training_model import HFTrainingModel

    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[1, 2]),
            loss_fn_inputs={},
        ),
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[3, 4, 5]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=[4, 5, 6], dtype="int64"),
            },
        ),
    ]
    hf_model = cast(HFTrainingModel, SimpleNamespace(model=torch.nn.Linear(1, 1)))

    with pytest.raises(ValueError, match="'target_tokens' must be present for every datum"):
        HFTrainingModel._prepare_loss_fn_inputs(hf_model, data)


def test_validate_client_loss_fn_inputs_allows_an_empty_request():
    """An empty batch is a no-op for both backends, not a missing-field error."""
    from tuft.backends.loss_inputs import (
        MODEL_DERIVED_LOSS_INPUTS,
        validate_client_loss_fn_inputs,
    )

    assert validate_client_loss_fn_inputs([]) == []
    # HF requires target_tokens per row, but an empty request has no rows; it
    # must still no-op the way the FSDP backend's empty-data guard does.
    assert (
        validate_client_loss_fn_inputs(
            [],
            ignored_keys=MODEL_DERIVED_LOSS_INPUTS,
            required_keys=frozenset({"target_tokens"}),
        )
        == []
    )


def test_prepare_loss_inputs_rejects_inconsistent_client_field_rank_and_dtype():
    import pytest
    import torch

    from tuft.backends.fsdp_engine import _prepare_loss_fn_inputs, _prepare_micro_batch

    def prepare(first, second):
        data = [
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
                loss_fn_inputs={"custom": first},
            ),
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens=[4, 5]),
                loss_fn_inputs={"custom": second},
            ),
        ]
        return _prepare_loss_fn_inputs(
            data,
            torch.randn(2, 3),
            "cross_entropy",
            prepared_target_tokens=_prepare_micro_batch(data, "cpu").labels,
        )

    with pytest.raises(ValueError, match="'custom' must have the same rank"):
        prepare(
            types.TensorData(data=[1.0, 2.0], dtype="float32"),
            types.TensorData(data=[1.0, 2.0, 3.0, 4.0], dtype="float32", shape=[2, 2]),
        )

    # Mixed dtypes must name the offending key rather than surfacing as a bare
    # torch.stack error (or being silently promoted by pad_sequence).
    with pytest.raises(ValueError, match="'custom' must have the same dtype"):
        prepare(
            types.TensorData(data=[1.0, 2.0], dtype="float32"),
            types.TensorData(data=[1, 2], dtype="int64"),
        )


def test_prepare_loss_inputs_key_set_is_stable_across_micro_batches():
    """The emitted key set must depend on the request, not on micro_batch_size."""
    import torch

    from tuft.backends.fsdp_engine import (
        _prepare_loss_fn_inputs,
        _prepare_micro_batch,
    )
    from tuft.backends.loss_inputs import client_loss_fn_input_keys

    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=[2, 3, 4], dtype="int64"),
                "logprobs": types.TensorData(data=[-0.1, -0.2, -0.3], dtype="float32"),
                "reference_logprobs": types.TensorData(data=[-0.2, -0.4, -0.8], dtype="float32"),
            },
        ),
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[4, 5]),
            loss_fn_inputs={
                "reference_logprobs": types.TensorData(data=[-0.1, -0.3], dtype="float32"),
            },
        ),
    ]
    client_keys = client_loss_fn_input_keys(data)

    key_sets = []
    for micro_data in ([data[0]], [data[1]]):
        micro_batch = _prepare_micro_batch(micro_data, "cpu")
        inputs = _prepare_loss_fn_inputs(
            micro_data,
            torch.randn(1, len(micro_data[0].model_input.to_ints())),
            "cross_entropy",
            prepared_target_tokens=micro_batch.labels,
            client_keys=client_keys,
        )
        key_sets.append(sorted(inputs))

    assert key_sets[0] == key_sets[1]
    assert key_sets[0] == [
        "logprobs",
        "reference_logprobs",
        "target_logprobs",
        "target_tokens",
        "weights",
    ]


def test_prepare_loss_inputs_does_not_supervise_rlhf_rows_missing_weights():
    """Under an RLHF loss, weights exists only because a client asked for it."""
    import torch

    from tuft.backends.fsdp_engine import _prepare_loss_fn_inputs, _prepare_micro_batch

    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
            loss_fn_inputs={"weights": types.TensorData(data=[1.0, 1.0, 0.0], dtype="float32")},
        ),
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[4, 5]),
            loss_fn_inputs={},
        ),
    ]
    inputs = _prepare_loss_fn_inputs(
        data,
        torch.randn(2, 3),
        "ppo",
        prepared_target_tokens=_prepare_micro_batch(data, "cpu").labels,
    )
    # Row 1 omitted weights, so it is masked out rather than fully supervised.
    assert inputs["weights"].tolist() == [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]


def test_compute_target_logprobs_matches_log_softmax():
    import torch

    from tuft.backends.fsdp_engine import _compute_target_logprobs

    torch.manual_seed(7)
    logits = torch.randn(2, 4, 11)
    labels = torch.randint(0, 11, (2, 4))
    expected = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    actual = _compute_target_logprobs(logits, labels)
    torch.testing.assert_close(actual, expected)


def test_forward_backward_micro_batches_preserve_summed_gradients():
    import copy
    from types import SimpleNamespace

    import torch

    from tuft.backends.fsdp_engine import forward_backward

    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.lm_head = torch.nn.Linear(8, 16, bias=False)

        def forward(self, input_ids, **_kwargs):
            return SimpleNamespace(logits=self.lm_head(self.embed(input_ids)))

    data = []
    for tokens in ([1, 2, 3, 4], [5, 6], [7, 8, 9], [10, 11, 12, 13]):
        weights = [1.0] * (len(tokens) - 1) + [0.0]
        data.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens=list(tokens)),
                loss_fn_inputs={"weights": types.TensorData(data=weights, dtype="float32")},
            )
        )

    torch.manual_seed(11)
    full_batch_model = TinyCausalLM()
    micro_batch_model = copy.deepcopy(full_batch_model)
    full = forward_backward(
        full_batch_model,
        data,
        "cross_entropy",
        None,
        micro_batch_size=4,
    )
    micro = forward_backward(
        micro_batch_model,
        data,
        "cross_entropy",
        None,
        micro_batch_size=2,
    )

    assert micro["metrics"]["loss:sum"] == pytest.approx(full["metrics"]["loss:sum"])
    for full_param, micro_param in zip(
        full_batch_model.parameters(), micro_batch_model.parameters(), strict=True
    ):
        torch.testing.assert_close(micro_param.grad, full_param.grad)
    assert [len(row) for row in micro["model_output"]["log_probs"]] == [4, 2, 3, 4]

    forward_only_model = copy.deepcopy(full_batch_model)
    for parameter in forward_only_model.parameters():
        parameter.grad = None
    forward_backward(
        forward_only_model,
        data,
        "cross_entropy",
        None,
        micro_batch_size=2,
        forward_only=True,
    )
    assert all(parameter.grad is None for parameter in forward_only_model.parameters())


def test_fsdp_engine_rl_loss_gradients_are_nonzero_on_cpu():
    """Regression test: sampling_logprobs must be detached from the computation graph.

    Without .detach(), clone() preserves the autograd connection and the gradients
    of target_logprobs cancel to zero in prob_ratio = exp(A - B) when B is a clone
    of A. This causes zero weight updates and reward never grows in RL training.
    """
    from types import SimpleNamespace

    import torch

    from tuft.backends.fsdp_engine import forward_backward

    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.lm_head = torch.nn.Linear(8, 16, bias=False)

        def forward(self, input_ids, **_kwargs):
            return SimpleNamespace(logits=self.lm_head(self.embed(input_ids)))

    torch.manual_seed(42)
    model = TinyCausalLM()
    # Construct RL-style data: logprobs and advantages in loss_fn_inputs.
    # target_tokens align with model_input length; logprobs are "old policy" values;
    # advantages are non-zero to ensure a non-trivial RL signal.
    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[1, 2, 3, 4]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=[2, 3, 4, 5], dtype="int64"),
                "logprobs": types.TensorData(data=[-0.5, -0.8, -1.2, -0.3], dtype="float32"),
                "advantages": types.TensorData(data=[1.0, 1.0, -1.0, 0.5], dtype="float32"),
            },
        ),
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[5, 6, 7]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=[6, 7, 8], dtype="int64"),
                "logprobs": types.TensorData(data=[-0.4, -1.0, -0.6], dtype="float32"),
                "advantages": types.TensorData(data=[-1.0, 0.5, 1.0], dtype="float32"),
            },
        ),
    ]
    for loss_fn_name in ("importance_sampling", "ppo", "cispo"):
        # Reset gradients for each loss function check
        for p in model.parameters():
            p.grad = None
        forward_backward(model, data, loss_fn_name, {}, micro_batch_size=2)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, f"No gradients were computed for loss_fn={loss_fn_name!r}"
        total_grad_norm = sum(g.abs().sum().item() for g in grads)
        assert total_grad_norm > 1e-8, (
            f"Gradients are effectively zero for loss_fn={loss_fn_name!r} "
            f"(total_grad_norm={total_grad_norm:.2e}). "
            "This indicates sampling_logprobs is connected to the computation graph "
            "and gradients cancel out (the clone-without-detach bug)."
        )


def test_fsdp_engine_rl_prepare_loss_inputs_sampling_logprobs_is_constant_on_cpu():
    """sampling_logprobs must equal the datum's logprobs (not target_logprobs) after prepare."""
    import torch

    from tuft.backends.fsdp_engine import _prepare_loss_fn_inputs, _prepare_micro_batch

    torch.manual_seed(0)
    old_lp = torch.tensor([-0.3, -0.7, -1.1], dtype=torch.float32)
    adv = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32)
    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
            loss_fn_inputs={
                "logprobs": types.TensorData.from_torch(old_lp),
                "advantages": types.TensorData.from_torch(adv),
            },
        )
    ]
    target_logprobs = torch.randn(1, 3, requires_grad=True)
    inputs = _prepare_loss_fn_inputs(
        data,
        target_logprobs,
        "importance_sampling",
        prepared_target_tokens=_prepare_micro_batch(data, "cpu").labels,
    )

    # sampling_logprobs must equal the datum's logprobs (the old policy values)
    torch.testing.assert_close(
        inputs["logprobs"][0, :3],
        old_lp,
        msg="sampling_logprobs must equal datum logprobs, not target_logprobs",
    )
    # sampling_logprobs must NOT require grad (must be detached from computation graph)
    assert not inputs["logprobs"].requires_grad, (
        "sampling_logprobs must not require grad; if it does, gradients cancel in "
        "prob_ratio = exp(target_logprobs - sampling_logprobs) and RL training stalls."
    )
    # advantages must match datum values
    torch.testing.assert_close(
        inputs["advantages"][0, :3],
        adv,
        msg="advantages must match datum values",
    )


@pytest.mark.asyncio
async def test_fsdp_engine_matches_hf_target_tokens_on_cpu():
    from types import SimpleNamespace

    import torch

    from tuft.backends.fsdp_engine import forward_backward
    from tuft.backends.hf_training_model import HFTrainingModel
    from tuft.loss_fn import get_loss_fn

    class PositionIndependentLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids, **_kwargs):
            batch, seq_len = input_ids.shape
            vocab_logits = self.scale * torch.arange(16, dtype=torch.float32)
            logits = vocab_logits.expand(batch, seq_len, -1).clone()
            return SimpleNamespace(logits=logits)

    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[10, 11]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=[11, 12], dtype="int64"),
                "weights": types.TensorData(data=[1.0, 1.0], dtype="float32"),
            },
        )
    ]

    hf_model = HFTrainingModel.__new__(HFTrainingModel)
    hf_model.model = PositionIndependentLM()  # type: ignore[assignment]
    hf_loss, hf_metrics, hf_outputs = await hf_model._forward_micro_batch(
        data,
        get_loss_fn("cross_entropy"),
        loss_fn_config=None,
        backward=False,
    )

    fsdp_model = PositionIndependentLM()
    fsdp_out = forward_backward(
        fsdp_model,
        data,
        "cross_entropy",
        None,
        micro_batch_size=1,
        forward_only=True,
    )

    torch.testing.assert_close(
        fsdp_out["model_output"]["log_probs"][0],
        hf_outputs[0]["logprobs"].to_torch(),
    )
    assert fsdp_out["metrics"]["loss:sum"] == pytest.approx(hf_metrics["loss:sum"])
    assert fsdp_out["metrics"]["loss:sum"] == pytest.approx(hf_loss)


# -----------------------------------------------------------------------------
# Integration tests: FSDP backend single-process (GPU, TUFT_TEST_MODEL)
# -----------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_backend_single_process_create_forward_optim_save_load():
    """
    FSDPTrainingBackend with TUFT_FSDP_NO_RAY=1 and 1 GPU: async_init, create_adapter,
    forward, optim_step, save_state, load_state. Requires TUFT_TEST_MODEL and CUDA.
    """
    if "TUFT_TEST_MODEL" not in os.environ:
        pytest.skip("Set TUFT_TEST_MODEL for FSDP integration test")

    prev_no_ray = os.environ.get("TUFT_FSDP_NO_RAY")
    os.environ["TUFT_FSDP_NO_RAY"] = "1"
    try:
        await _run_fsdp_single_process_flow()
    finally:
        if prev_no_ray is None:
            os.environ.pop("TUFT_FSDP_NO_RAY", None)
        else:
            os.environ["TUFT_FSDP_NO_RAY"] = prev_no_ray


async def _run_fsdp_single_process_flow() -> None:
    import transformers

    from tuft.backends.fsdp_training_backend import FSDPTrainingBackend
    from tuft.checkpoints import CheckpointRecord

    model_path = Path(os.environ["TUFT_TEST_MODEL"])
    config = ModelConfig(
        model_name="fsdp-test",
        model_path=model_path,
        max_model_len=512,
        max_lora_rank=8,  # match LoraConfig(rank=8) so slot pool has slots for rank 8
        training_backend="fsdp",
        fsdp_num_gpus=1,
    )
    backend = FSDPTrainingBackend(config)
    await backend.async_init()

    await backend.create_adapter("lora_1", types.LoraConfig(rank=8, seed=42))
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokens = tokenizer.encode("Hello world", add_special_tokens=True)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = [1.0] * len(target_tokens)
    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs=dict(
                weights=types.TensorData(data=weights, dtype="float32"),
                target_tokens=types.TensorData(data=target_tokens, dtype="int64"),
            ),
        ),
    ]
    out = await backend.forward(
        data=data,
        lora_id="lora_1",
        loss_fn="cross_entropy",
        loss_fn_config=None,
        backward=True,
    )
    assert "loss" in out.metrics or "loss:sum" in out.metrics
    await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="lora_1")

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = CheckpointRecord(
            checkpoint_id="lora_1",
            owner_name="default",
            checkpoint_type="training",
            training_run_id="run1",
            path=Path(tmp) / "lora_1",
            size_bytes=0,
        )
        await backend.save_state(lora_id="lora_1", checkpoint_record=ckpt, optimizer=False)
        # CheckpointRecord.adapter_path is path / "adapter", so adapter.pt is under that dir
        assert (Path(tmp) / "lora_1" / "adapter" / "adapter.pt").exists()
        # PEFT format for sampling (VLLM)
        assert (Path(tmp) / "lora_1" / "adapter" / "adapter_config.json").exists()

        await backend.create_adapter("lora_2", types.LoraConfig(rank=8, seed=43))
        await backend.load_state(lora_id="lora_2", checkpoint_record=ckpt, optimizer=False)
        out2 = await backend.forward(
            data=data,
            lora_id="lora_2",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=False,
        )
        assert "loss" in out2.metrics or "loss:sum" in out2.metrics


# -----------------------------------------------------------------------------
# Unit tests: _shard_list order-preserving guarantees
# -----------------------------------------------------------------------------


def test_shard_list_preserves_order_even_split():
    """_shard_list with even split preserves original order."""
    from tuft.backends.fsdp_training_backend import _shard_list

    data = list(range(10))
    shards = _shard_list(data, 2)
    assert shards == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    # Concatenating shards in order reconstructs original
    reconstructed = [x for shard in shards for x in shard]
    assert reconstructed == data


def test_shard_list_preserves_order_uneven_split():
    """_shard_list with uneven split (remainder) preserves original order."""
    from tuft.backends.fsdp_training_backend import _shard_list

    data = list(range(7))
    shards = _shard_list(data, 3)
    # 7 / 3 = 2 rem 1 → first shard gets 3, rest get 2
    assert shards == [[0, 1, 2], [3, 4], [5, 6]]
    reconstructed = [x for shard in shards for x in shard]
    assert reconstructed == data


def test_shard_list_preserves_order_single_shard():
    """_shard_list with n_shards=1 returns the full list."""
    from tuft.backends.fsdp_training_backend import _shard_list

    data = list(range(5))
    shards = _shard_list(data, 1)
    assert shards == [data]


def test_shard_list_raises_on_zero_shards():
    """_shard_list raises ValueError when n_shards <= 0."""
    from tuft.backends.fsdp_training_backend import _shard_list

    with pytest.raises(ValueError):
        _shard_list([1, 2, 3], 0)


def test_shard_list_more_shards_than_elements():
    """_shard_list with more shards than elements produces empty shards at end."""
    from tuft.backends.fsdp_training_backend import _shard_list

    data = [10, 20]
    shards = _shard_list(data, 4)
    # 2 / 4 = 0 rem 2 → first 2 shards get 1 each, last 2 empty
    assert shards == [[10], [20], [], []]
    reconstructed = [x for shard in shards for x in shard]
    assert reconstructed == data


def test_shard_list_batch_order_contract_with_variable_length_data():
    """Verify the batch-order contract: after shard+merge, zip(data, outputs) aligns.

    This simulates the multi-actor forward path:
    1. Split data into N shards
    2. Each shard produces outputs in shard-local order
    3. Extending outputs in shard order reconstructs original order

    This is the critical property that was broken by the old token-balanced
    sharding and is now restored with the simple contiguous _shard_list.
    """
    from tuft.backends.fsdp_training_backend import _shard_list

    # Simulate variable-length sequences (like real training data)
    data = [
        {"id": i, "tokens": list(range(length))}
        for i, length in enumerate([128, 256, 64, 512, 32, 1024, 100, 200])
    ]
    n_actors = 3
    shards = _shard_list(data, n_actors)

    # Simulate each actor processing its shard and returning logprobs
    # with lengths matching their input sequence lengths
    all_outputs = []
    for shard in shards:
        shard_outputs = []
        for datum in shard:
            # Each actor returns logprob of length == len(tokens)
            # Simulate the FSDP engine returning per-token logprobs.
            shard_outputs.append({"logprobs_len": len(datum["tokens"]), "datum_id": datum["id"]})
        all_outputs.extend(shard_outputs)

    # The critical assertion: after extend, outputs[i] corresponds to data[i]
    assert len(all_outputs) == len(data)
    for i, (datum, output) in enumerate(zip(data, all_outputs, strict=True)):
        assert output["datum_id"] == datum["id"], (
            f"Order mismatch at index {i}: expected datum_id={datum['id']}, "
            f"got {output['datum_id']}. This would cause silent logprob corruption "
            f"in training (the [-am:] slicing hides shape mismatches)."
        )
        assert output["logprobs_len"] == len(datum["tokens"]), (
            f"Length mismatch at index {i}: logprobs_len={output['logprobs_len']} "
            f"!= tokens_len={len(datum['tokens'])}. This is the shape mismatch "
            f"that Python's [-am:] slicing would hide."
        )


@pytest.mark.asyncio
async def test_forward_raises_when_data_fewer_than_actors():
    """forward() raises ValueError when len(data) < world_size (NCCL deadlock guard)."""
    from unittest.mock import MagicMock

    from tuft.backends.fsdp_training_backend import FSDPTrainingBackend

    config = ModelConfig(
        model_name="test",
        model_path=Path("/tmp/qwen-model"),
        max_model_len=1024,
        training_backend="fsdp",
        fsdp_num_gpus=2,
    )
    backend = FSDPTrainingBackend(config)
    # Simulate multi-actor path: _worker is None, _actors has 2 stubs
    backend._worker = None
    backend._actors = [MagicMock(), MagicMock()]
    backend._lora_id_to_adapter_name = {"lora1": "adapter_0"}
    backend._adapter_name_to_lora_id = {"adapter_0": "lora1"}

    single_datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
        loss_fn_inputs={},
    )

    with pytest.raises(ValueError, match=r"len\(data\)=1, world_size=2"):
        await backend.forward(
            data=[single_datum],
            lora_id="lora1",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )
