"""Tests for FSDP2 multi-GPU training backend.

Mirrors the structure of test_training_backend.py but exercises the
FSDPTrainingBackend with multiple GPUs.

Requires:
    - ``--gpu`` pytest flag
    - ``TUFT_TEST_MODEL`` environment variable pointing to a HuggingFace model
    - At least 2 GPUs available (FSDP_TEST_GPUS, default 2)
"""

import pytest

pytest.importorskip("torch", reason="FSDP tests require PyTorch")

import os
import tempfile
from pathlib import Path
from typing import Any, List

import numpy as np
import ray
import transformers
from tinker import types

from tuft.backends.fsdp_training_backend import FSDPTrainingBackend
from tuft.checkpoints import CheckpointRecord
from tuft.config import ModelConfig

from .helpers import (
    PIG_LATIN_EXAMPLES,
    PIG_LATIN_EXAMPLES_EXTENDED,
    clear_ray_state,
)


NUM_FSDP_GPUS = max(2, int(os.environ.get("FSDP_TEST_GPUS", "2")))
MODEL_PATH = Path(os.environ["TUFT_TEST_MODEL"]) if "TUFT_TEST_MODEL" in os.environ else None


def _skip_if_no_model():
    if MODEL_PATH is None:
        pytest.skip("TUFT_TEST_MODEL not set")


@pytest.fixture(scope="function")
def fsdp_ray_cluster(request):
    if not request.config.getoption("--gpu"):
        yield
        return
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < NUM_FSDP_GPUS:
        pytest.skip(
            f"Need at least {NUM_FSDP_GPUS} GPUs for FSDP, found {torch.cuda.device_count()}"
        )
    _skip_if_no_model()
    clear_ray_state()
    ray.init(ignore_reinit_error=True)
    yield
    clear_ray_state()


def _construct_data(name: str = "extended") -> List[types.Datum]:
    assert MODEL_PATH is not None, "TUFT_TEST_MODEL must be set for FSDP tests."
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    examples = PIG_LATIN_EXAMPLES_EXTENDED if name == "extended" else PIG_LATIN_EXAMPLES
    data = []
    for example in examples:
        prompt = f"English: {example['input']}\nPig Latin:"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_weights = [0] * len(prompt_tokens)
        completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
        completion_weights = [1] * len(completion_tokens)

        tokens = prompt_tokens + completion_tokens
        weights = prompt_weights + completion_weights
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        weights = weights[1:]

        data.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs=dict(
                    weights=types.TensorData(data=weights, dtype="float32"),
                    target_tokens=types.TensorData(data=target_tokens, dtype="int64"),
                ),
            )
        )
    return data


def _make_model_config(**overrides) -> ModelConfig:
    assert MODEL_PATH is not None
    defaults: dict[str, Any] = dict(
        model_name="test-fsdp",
        model_path=MODEL_PATH,
        max_model_len=2048,
        tensor_parallel_size=1,
        training_backend="fsdp",
        num_gpus_per_node=NUM_FSDP_GPUS,
        num_nodes=1,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


# ------------------------------------------------------------------
# Test 1: Basic FSDP2 training — loss should decrease
# ------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_training_loss_decrease(fsdp_ray_cluster):
    """Create one adapter, train for a few steps, assert loss decreases."""
    config = _make_model_config()
    backend = FSDPTrainingBackend(config)
    await backend.async_init()

    await backend.create_adapter("lora_a", types.LoraConfig(rank=8, seed=42))

    data = _construct_data()
    weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in data])

    losses = []
    for step in range(4):
        output = await backend.forward(
            data=data,
            lora_id="lora_a",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )
        await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="lora_a")
        logprobs = np.concatenate([o["logprobs"].tolist() for o in output.loss_fn_outputs])
        loss = -np.dot(logprobs, weights) / weights.sum()
        losses.append(loss)
        print(f"[FSDP] step {step} loss={loss:.4f}")

    for i in range(1, len(losses)):
        assert losses[i] < losses[i - 1], (
            f"Loss did not decrease at step {i}: {losses[i]:.4f} >= {losses[i - 1]:.4f}"
        )


# ------------------------------------------------------------------
# Test 2: Multi-tenant — two adapters trained concurrently
# ------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_multi_tenant(fsdp_ray_cluster):
    """Two adapters trained in interleaved fashion; both losses should decrease."""
    config = _make_model_config()
    backend = FSDPTrainingBackend(config)
    await backend.async_init()

    await backend.create_adapter("lora_1", types.LoraConfig(rank=8, seed=42))
    await backend.create_adapter("lora_2", types.LoraConfig(rank=8, seed=42))

    data = _construct_data()
    weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in data])

    losses_1, losses_2 = [], []
    for step in range(3):
        out1 = await backend.forward(
            data=data,
            lora_id="lora_1",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )
        out2 = await backend.forward(
            data=data,
            lora_id="lora_2",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )
        await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="lora_1")
        await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="lora_2")

        lp1 = np.concatenate([o["logprobs"].tolist() for o in out1.loss_fn_outputs])
        lp2 = np.concatenate([o["logprobs"].tolist() for o in out2.loss_fn_outputs])
        loss1 = -np.dot(lp1, weights) / weights.sum()
        loss2 = -np.dot(lp2, weights) / weights.sum()
        losses_1.append(loss1)
        losses_2.append(loss2)
        print(f"[FSDP multi-tenant] step {step}  lora_1={loss1:.4f}  lora_2={loss2:.4f}")

    for i in range(1, len(losses_1)):
        assert losses_1[i] < losses_1[i - 1], "lora_1 loss did not decrease"
        assert losses_2[i] < losses_2[i - 1], "lora_2 loss did not decrease"

    assert abs(losses_1[-1] - losses_2[-1]) < 0.5, (
        "Two identically-seeded LoRAs should have similar loss"
    )


# ------------------------------------------------------------------
# Test 3: Checkpoint save & load
# ------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_checkpoint_save_load(fsdp_ray_cluster):
    """Train, save checkpoint, load into new adapter, verify continuity."""
    config = _make_model_config()
    backend = FSDPTrainingBackend(config)
    await backend.async_init()

    await backend.create_adapter("lora_ckpt", types.LoraConfig(rank=8, seed=42))
    data = _construct_data()
    weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in data])

    pre_losses = []
    for step in range(3):
        output = await backend.forward(
            data=data,
            lora_id="lora_ckpt",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )
        await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="lora_ckpt")
        lp = np.concatenate([o["logprobs"].tolist() for o in output.loss_fn_outputs])
        loss = -np.dot(lp, weights) / weights.sum()
        pre_losses.append(loss)
        print(f"[FSDP ckpt] pre-save step {step} loss={loss:.4f}")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = CheckpointRecord(
            checkpoint_id="lora_ckpt",
            owner_name="default",
            checkpoint_type="training",
            path=Path(tmpdir) / "lora_ckpt",
            training_run_id="test_run",
            size_bytes=0,
        )
        await backend.save_state(lora_id="lora_ckpt", checkpoint_record=ckpt, optimizer=True)

        await backend.load_state(lora_id="lora_loaded", checkpoint_record=ckpt, optimizer=True)

        post_losses = []
        for step in range(3, 6):
            output = await backend.forward(
                data=data,
                lora_id="lora_loaded",
                loss_fn="cross_entropy",
                loss_fn_config=None,
                backward=True,
            )
            await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="lora_loaded")
            lp = np.concatenate([o["logprobs"].tolist() for o in output.loss_fn_outputs])
            loss = -np.dot(lp, weights) / weights.sum()
            post_losses.append(loss)
            print(f"[FSDP ckpt] post-load step {step} loss={loss:.4f}")

    assert post_losses[0] < pre_losses[-1] + 0.1, (
        "Post-load loss should be close to or below pre-save loss"
    )
    for i in range(1, len(post_losses)):
        assert post_losses[i] < post_losses[i - 1], "Post-load loss should keep decreasing"


# ------------------------------------------------------------------
# Test 4: Gradient accumulation isolation across adapter switches
# ------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_gradient_isolation(fsdp_ray_cluster):
    """Verify that gradient accumulation is properly isolated when
    switching adapters between forward_backward calls.

    Train lora_x with 2 forward_backward calls (gradient accumulation)
    with an adapter switch in between, vs lora_y with 2 consecutive
    forward_backward calls (no switch). Both should converge similarly.
    """
    config = _make_model_config()
    backend = FSDPTrainingBackend(config)
    await backend.async_init()

    await backend.create_adapter("lora_x", types.LoraConfig(rank=8, seed=42))
    await backend.create_adapter("lora_y", types.LoraConfig(rank=8, seed=42))
    await backend.create_adapter("lora_dummy", types.LoraConfig(rank=8, seed=99))

    data = _construct_data(name="default")
    weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in data])

    losses_x, losses_y = [], []
    for step in range(3):
        # lora_x: forward_backward → switch to dummy → switch back → forward_backward → optim
        await backend.forward(
            data=data,
            lora_id="lora_x",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )
        # interleave with dummy adapter
        await backend.forward(
            data=data,
            lora_id="lora_dummy",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )
        await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="lora_dummy")
        out_x = await backend.forward(
            data=data,
            lora_id="lora_x",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )
        await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="lora_x")

        # lora_y: two consecutive forward_backward → optim (no switch)
        await backend.forward(
            data=data,
            lora_id="lora_y",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )
        out_y = await backend.forward(
            data=data,
            lora_id="lora_y",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )
        await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="lora_y")

        lp_x = np.concatenate([o["logprobs"].tolist() for o in out_x.loss_fn_outputs])
        lp_y = np.concatenate([o["logprobs"].tolist() for o in out_y.loss_fn_outputs])
        loss_x = -np.dot(lp_x, weights) / weights.sum()
        loss_y = -np.dot(lp_y, weights) / weights.sum()
        losses_x.append(loss_x)
        losses_y.append(loss_y)
        print(f"[grad iso] step {step}  lora_x={loss_x:.4f}  lora_y={loss_y:.4f}")

    for i in range(1, len(losses_x)):
        assert losses_x[i] < losses_x[i - 1], "lora_x loss should decrease"
        assert losses_y[i] < losses_y[i - 1], "lora_y loss should decrease"
