"""
Unit and integration tests for FSDP training backend.

Unit tests (no GPU):
  - Config/slot helpers and async_init validation (no torch/verl).
  - Tensordict and loss adapters (torch/tensordict on CPU).

Integration tests (GPU, optional TUFT_TEST_MODEL):
  - FSDPTrainingBackend single-process: create_adapter, forward, optim_step, save/load.

Run:
  pytest tests/test_fsdp_training_backend.py                    # unit only
  pytest tests/test_fsdp_training_backend.py --gpu -m gpu      # include GPU/integration
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from tinker import types

from tuft.config import ModelConfig


# -----------------------------------------------------------------------------
# Unit tests: config and slot (no GPU, no torch/verl)
# -----------------------------------------------------------------------------


def test_config_to_worker_dict():
    """_config_to_worker_dict returns a serializable dict with expected keys and slot_config."""
    from tuft.backends.fsdp_training_backend import _config_to_worker_dict

    config = ModelConfig(
        model_name="test",
        model_path=Path("/tmp/model"),
        max_model_len=1024,
        max_lora_rank=8,
    )
    d = _config_to_worker_dict(config)
    assert d["model_path"] == "/tmp/model"
    assert d["max_model_len"] == 1024
    assert "slot_config" in d
    assert d["slot_config"]["rank_slots"] == {8: 16}  # default for max_lora_rank=8
    assert d["slot_config"]["target_modules"] == ["q_proj", "v_proj"]
    assert "fsdp_override_config" in d
    assert isinstance(d["fsdp_override_config"], dict)


def test_slot_pool_config_get_lora_alpha():
    """SlotPoolConfig.get_lora_alpha returns rank * lora_alpha_ratio."""
    from tuft.backends.fsdp_training_backend import SlotPoolConfig

    cfg = SlotPoolConfig(rank_slots={8: 2}, lora_alpha_ratio=2)
    assert cfg.get_lora_alpha(8) == 16
    assert cfg.get_lora_alpha(16) == 32


@pytest.mark.asyncio
async def test_async_init_raises_when_no_ray_and_multi_gpu():
    """async_init raises ValueError when TUFT_FSDP_NO_RAY=1 and fsdp_num_gpus != 1."""
    from tuft.backends.fsdp_training_backend import FSDPTrainingBackend

    prev = os.environ.get("TUFT_FSDP_NO_RAY")
    os.environ["TUFT_FSDP_NO_RAY"] = "1"
    try:
        config = ModelConfig(
            model_name="test",
            model_path=Path("/tmp/model"),
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
                model_path=Path("/tmp/a"),
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
                model_path=Path("/tmp/c"),
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
# Unit tests: tensordict and loss adapters (torch/tensordict, CPU)
# -----------------------------------------------------------------------------


def test_chunk_tensordict_allow_2d_nested():
    """_chunk_tensordict_allow_2d_nested splits TensorDict into N chunks; 2D nested use unbind."""
    import torch
    from tensordict import TensorDict

    from tuft.backends.fsdp_training_backend import _chunk_tensordict_allow_2d_nested

    # Build a small TensorDict: 4 rows, one regular key, one 2D nested
    batch_size = 4
    a = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    nested = torch.nested.nested_tensor(
        [
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.0]),
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0]),
        ],
        dtype=torch.float32,
    )
    td = TensorDict({"a": a, "n": nested}, batch_size=[batch_size])
    chunks = _chunk_tensordict_allow_2d_nested(td, chunks=2)
    assert len(chunks) == 2
    assert chunks[0]["a"].shape == (2, 2)
    assert chunks[1]["a"].shape == (2, 2)
    assert chunks[0]["n"].is_nested and chunks[1]["n"].is_nested


def test_datum_list_to_tensordict_keys_and_shapes():
    """_datum_list_to_tensordict yields TensorDict with expected keys and nested ids."""
    from tuft.backends.fsdp_training_backend import _datum_list_to_tensordict

    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
            loss_fn_inputs=dict(
                weights=types.TensorData(data=[0.0, 1.0, 1.0], dtype="float32"),
                target_tokens=types.TensorData(data=[2, 3, 4], dtype="int64"),
            ),
        ),
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=[5, 6]),
            loss_fn_inputs=dict(
                weights=types.TensorData(data=[1.0, 1.0], dtype="float32"),
                target_tokens=types.TensorData(data=[6, 7], dtype="int64"),
            ),
        ),
    ]
    td = _datum_list_to_tensordict(data, adapter_id="a1", device="cpu")
    assert td.batch_size[0] == 2
    assert "input_ids" in td
    assert "position_ids" in td
    assert "weights" in td
    assert "target_tokens" in td
    assert "adapter_id" in td
    assert getattr(td["input_ids"], "is_nested", False) or td["input_ids"].dim() >= 1
    assert td["weights"].shape[0] == 2


def test_make_verl_loss_fn_signature():
    """_make_verl_loss_fn returns (model_output, data) -> (loss, metrics) callable."""
    import torch
    from tensordict import TensorDict

    from tuft.backends.fsdp_training_backend import _make_verl_loss_fn

    loss_fn = _make_verl_loss_fn("cross_entropy", {})
    batch_size, max_len = 2, 3
    # Engine passes nested log_probs (B, seq) as 2D nested; each row is 1D variable-length
    log_probs = torch.randn(batch_size, max_len)
    log_probs_nt = torch.nested.as_nested_tensor(
        [log_probs[i] for i in range(batch_size)],
        layout=torch.jagged,
    )
    weights = torch.ones(batch_size, max_len)
    data = TensorDict({"weights": weights}, batch_size=[batch_size])
    loss, metrics = loss_fn(model_output={"log_probs": log_probs_nt}, data=data)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0
    assert isinstance(metrics, dict)
    assert "loss:sum" in metrics or "loss" in str(metrics)


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
            # (simulates verl engine returning per-token logprobs)
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
        model_path=Path("/tmp/model"),
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
