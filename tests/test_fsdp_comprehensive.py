"""Comprehensive FSDP2 multi-GPU training test suite.

Covers:
    1. Single base model with multiple adapters of different LoRA ranks
    2. Training-state preservation across adapter switches
       (weights + optimizer momentum are not lost)
    3. Sequential training on two different base models
    4. Deploying a trained LoRA adapter to vLLM inference
    5. GPU memory monitoring throughout every phase

Requires:
    - ``--gpu`` pytest flag
    - At least 2 GPUs for FSDP multi-GPU training (FSDP_TEST_GPUS, default 2)
    - Inference test trains first (2 GPUs), then shuts down and starts vLLM (1 GPU)
    - ``TUFT_TEST_MODEL`` (or ``FSDP_TEST_MODEL_A``) env var pointing to a model
    - ``FSDP_TEST_MODEL_B`` env var for sequential multi-model tests (optional)
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="FSDP tests require PyTorch")

import os
import tempfile
import time
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
    TEST_PROMPTS,
    _normalize_text,
    clear_ray_state,
)


# ---------------------------------------------------------------------------
# Model paths — resolved from environment variables only
# ---------------------------------------------------------------------------
def _resolve_model_path(env_key: str) -> Path | None:
    val = os.environ.get(env_key)
    if val:
        return Path(val)
    val = os.environ.get("TUFT_TEST_MODEL")
    if val:
        return Path(val)
    return None


MODEL_A_PATH = _resolve_model_path("FSDP_TEST_MODEL_A")
MODEL_B_PATH = _resolve_model_path("FSDP_TEST_MODEL_B")
NUM_FSDP_GPUS = max(2, int(os.environ.get("FSDP_TEST_GPUS", "2")))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
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
    if MODEL_A_PATH is None:
        pytest.skip("TUFT_TEST_MODEL / FSDP_TEST_MODEL_A not set")
    clear_ray_state()
    ray.init(ignore_reinit_error=True)
    yield
    clear_ray_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_config(model_path: Path | None, **overrides) -> ModelConfig:
    assert model_path is not None, "Model path must be set via environment variable"
    defaults: dict[str, Any] = dict(
        model_name=model_path.name,
        model_path=model_path,
        max_model_len=2048,
        tensor_parallel_size=1,
        training_backend="fsdp",
        num_gpus_per_node=NUM_FSDP_GPUS,
        num_nodes=1,
        max_lora_rank=32,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _construct_data(model_path: Path | None, name: str = "extended") -> List[types.Datum]:
    assert model_path is not None, "Model path must be set via environment variable"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
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


def _compute_loss(output: types.ForwardBackwardOutput, data) -> float:
    weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in data])
    logprobs = np.concatenate([o["logprobs"].tolist() for o in output.loss_fn_outputs])
    return float(-np.dot(logprobs, weights) / weights.sum())


async def _train_step(
    backend: FSDPTrainingBackend,
    data,
    lora_id: str,
    lr: float = 1e-4,
) -> float:
    """One forward-backward + optim-step; returns the loss."""
    output = await backend.forward(
        data=data,
        lora_id=lora_id,
        loss_fn="cross_entropy",
        loss_fn_config=None,
        backward=True,
    )
    await backend.optim_step(types.AdamParams(learning_rate=lr), lora_id=lora_id)
    return _compute_loss(output, data)


async def _forward_only(
    backend: FSDPTrainingBackend,
    data,
    lora_id: str,
) -> float:
    """Forward pass without backward — returns the loss."""
    output = await backend.forward(
        data=data,
        lora_id=lora_id,
        loss_fn="cross_entropy",
        loss_fn_config=None,
        backward=False,
    )
    return _compute_loss(output, data)


def _print_memory(stats: list[dict], tag: str) -> None:
    for s in stats:
        print(
            f"  [{tag}] rank {s['rank']}  "
            f"alloc={s['allocated_mb']:.0f}MB  "
            f"reserved={s['reserved_mb']:.0f}MB  "
            f"peak={s['max_allocated_mb']:.0f}MB",
            flush=True,
        )


# ===================================================================
# Test 1 — Multi-adapter training with different LoRA ranks
# ===================================================================
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_multi_adapter_different_ranks(fsdp_ray_cluster):
    """Create adapters with rank 4, 8, 16 on one model.
    Train each and verify loss decreases.  Monitor GPU memory growth.
    """
    config = _make_config(MODEL_A_PATH)
    backend = FSDPTrainingBackend(config)
    await backend.async_init()
    data = _construct_data(MODEL_A_PATH)

    mem_init = await backend.get_memory_stats()
    _print_memory(mem_init, "after init")

    rank_losses: dict[int, list[float]] = {}
    for rank_val in (4, 8, 16):
        lora_id = f"lora_r{rank_val}"
        await backend.create_adapter(lora_id, types.LoraConfig(rank=rank_val, seed=42))

        mem_after = await backend.get_memory_stats()
        _print_memory(mem_after, f"after create {lora_id}")

        losses: list[float] = []
        for step in range(4):
            loss = await _train_step(backend, data, lora_id)
            losses.append(loss)
            print(f"  [{lora_id}] step {step}  loss={loss:.4f}", flush=True)

        rank_losses[rank_val] = losses

        for i in range(1, len(losses)):
            assert losses[i] < losses[i - 1], (
                f"{lora_id}: loss did not decrease at step {i}: "
                f"{losses[i]:.4f} >= {losses[i - 1]:.4f}"
            )

    mem_end = await backend.get_memory_stats()
    _print_memory(mem_end, "end of multi-adapter test")

    for s in mem_end:
        assert s["allocated_mb"] > 0, "Expected non-zero GPU allocation"
        assert s["reserved_mb"] < 50000, "Unexpected memory explosion"


# ===================================================================
# Test 2 — Training-state preservation across adapter switches
# ===================================================================
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_training_state_preservation(fsdp_ray_cluster):
    """Verify that weights and optimizer state survive adapter switches.

    1. Train adapter_x for 3 steps → get eval loss
    2. Switch to adapter_y → train 2 steps
    3. Switch back to adapter_x → get eval loss (should match step 1 end)
    4. Continue training adapter_x for 3 more steps → loss keeps decreasing
    """
    config = _make_config(MODEL_A_PATH)
    backend = FSDPTrainingBackend(config)
    await backend.async_init()
    data = _construct_data(MODEL_A_PATH, name="default")

    await backend.create_adapter("adapter_x", types.LoraConfig(rank=8, seed=42))
    await backend.create_adapter("adapter_y", types.LoraConfig(rank=4, seed=99))

    # --- Phase 1: train adapter_x for 3 steps ---
    losses_x_phase1: list[float] = []
    for step in range(3):
        loss = await _train_step(backend, data, "adapter_x")
        losses_x_phase1.append(loss)
        print(f"  [phase1] adapter_x step {step}  loss={loss:.4f}", flush=True)

    eval_loss_before_switch = await _forward_only(backend, data, "adapter_x")
    print(
        f"  [phase1] adapter_x eval loss before switch = {eval_loss_before_switch:.4f}",
        flush=True,
    )

    # --- Phase 2: switch to adapter_y, train 2 steps ---
    for step in range(2):
        loss = await _train_step(backend, data, "adapter_y")
        print(f"  [phase2] adapter_y step {step}  loss={loss:.4f}", flush=True)

    # --- Phase 3: switch back to adapter_x ---
    eval_loss_after_switch = await _forward_only(backend, data, "adapter_x")
    print(
        f"  [phase3] adapter_x eval loss after switch  = {eval_loss_after_switch:.4f}",
        flush=True,
    )

    assert abs(eval_loss_after_switch - eval_loss_before_switch) < 0.01, (
        f"adapter_x weights changed during adapter_y training: "
        f"{eval_loss_before_switch:.4f} → {eval_loss_after_switch:.4f}"
    )

    # --- Phase 4: continue training adapter_x ---
    losses_x_phase2: list[float] = []
    for step in range(3):
        loss = await _train_step(backend, data, "adapter_x")
        losses_x_phase2.append(loss)
        print(f"  [phase4] adapter_x step {step + 3}  loss={loss:.4f}", flush=True)

    assert losses_x_phase2[0] < losses_x_phase1[-1] + 0.1, (
        "adapter_x should continue improving from where it left off"
    )
    all_x_losses = losses_x_phase1 + losses_x_phase2
    for i in range(1, len(all_x_losses)):
        assert all_x_losses[i] < all_x_losses[i - 1] + 0.05, (
            f"adapter_x non-monotonic at step {i}: "
            f"{all_x_losses[i]:.4f} >= {all_x_losses[i - 1]:.4f}"
        )

    mem = await backend.get_memory_stats()
    _print_memory(mem, "end of preservation test")


# ===================================================================
# Test 3 — Sequential training on two base models
# ===================================================================
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_multi_model_sequential(fsdp_ray_cluster):
    """Train on model A, shut down, train on model B.
    Save checkpoints from both; verify checkpoint files exist.
    """
    if MODEL_B_PATH is None:
        pytest.skip("FSDP_TEST_MODEL_B not set — need two distinct models")

    ckpt_paths: dict[str, Path] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # --- Model A ---
        print("\n=== Model A: training ===", flush=True)
        cfg_a = _make_config(MODEL_A_PATH)
        backend_a = FSDPTrainingBackend(cfg_a)
        await backend_a.async_init()
        data_a = _construct_data(MODEL_A_PATH, name="default")

        await backend_a.create_adapter("lora_a", types.LoraConfig(rank=8, seed=42))
        losses_a: list[float] = []
        for step in range(4):
            loss = await _train_step(backend_a, data_a, "lora_a")
            losses_a.append(loss)
            print(f"  [model_a] step {step}  loss={loss:.4f}", flush=True)

        ckpt_a = CheckpointRecord(
            checkpoint_id="lora_a",
            owner_name="default",
            checkpoint_type="training",
            path=tmpdir / "ckpt_model_a",
            training_run_id="test_multi_model",
            size_bytes=0,
        )
        await backend_a.save_state(lora_id="lora_a", checkpoint_record=ckpt_a, optimizer=True)
        ckpt_paths["model_a"] = ckpt_a.adapter_path

        mem_a = await backend_a.get_memory_stats()
        _print_memory(mem_a, "model_a before shutdown")
        backend_a.shutdown()
        print("  [model_a] shut down", flush=True)

        clear_ray_state()
        time.sleep(5)
        ray.init(ignore_reinit_error=True)

        # --- Model B ---
        print("\n=== Model B: training ===", flush=True)
        cfg_b = _make_config(MODEL_B_PATH)
        backend_b = FSDPTrainingBackend(cfg_b)
        await backend_b.async_init()
        data_b = _construct_data(MODEL_B_PATH, name="default")

        await backend_b.create_adapter("lora_b", types.LoraConfig(rank=8, seed=42))
        losses_b: list[float] = []
        for step in range(4):
            loss = await _train_step(backend_b, data_b, "lora_b")
            losses_b.append(loss)
            print(f"  [model_b] step {step}  loss={loss:.4f}", flush=True)

        ckpt_b = CheckpointRecord(
            checkpoint_id="lora_b",
            owner_name="default",
            checkpoint_type="training",
            path=tmpdir / "ckpt_model_b",
            training_run_id="test_multi_model",
            size_bytes=0,
        )
        await backend_b.save_state(lora_id="lora_b", checkpoint_record=ckpt_b, optimizer=True)
        ckpt_paths["model_b"] = ckpt_b.adapter_path

        mem_b = await backend_b.get_memory_stats()
        _print_memory(mem_b, "model_b before shutdown")
        backend_b.shutdown()
        print("  [model_b] shut down", flush=True)

        # --- Verify losses decreased ---
        for i in range(1, len(losses_a)):
            assert losses_a[i] < losses_a[i - 1], "model_a loss did not decrease"
        for i in range(1, len(losses_b)):
            assert losses_b[i] < losses_b[i - 1], "model_b loss did not decrease"

        # --- Verify checkpoint files ---
        for tag, adapter_path in ckpt_paths.items():
            assert adapter_path.exists(), f"{tag} adapter_path missing"
            files = list(adapter_path.iterdir())
            assert len(files) > 0, f"{tag} adapter_path is empty"
            print(f"  [{tag}] checkpoint files: {[f.name for f in files]}", flush=True)


# ===================================================================
# Test 4 — Deploy trained adapter to vLLM inference
# ===================================================================
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_inference_deployment(fsdp_ray_cluster):
    """Train a LoRA adapter with FSDP2, save it, load into vLLM, and sample."""
    try:
        from tuft.backends.sampling_backend import VLLMSamplingBackend
    except ImportError:
        pytest.skip("VLLMSamplingBackend not available (missing vllm/trinity)")
        return
    model_path = MODEL_A_PATH
    assert model_path is not None
    config = _make_config(model_path)
    backend = FSDPTrainingBackend(config)
    await backend.async_init()
    data = _construct_data(model_path, name="default")

    await backend.create_adapter("lora_infer", types.LoraConfig(rank=8, seed=42))

    print("\n=== Training for inference deployment ===", flush=True)
    for step in range(40):
        loss = await _train_step(backend, data, "lora_infer")
        if step % 10 == 0:
            print(f"  step {step}  loss={loss:.4f}", flush=True)

    mem_train = await backend.get_memory_stats()
    _print_memory(mem_train, "after training")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = CheckpointRecord(
            checkpoint_id="lora_infer",
            owner_name="default",
            checkpoint_type="training",
            path=Path(tmpdir) / "lora_infer",
            training_run_id="test_inference",
            size_bytes=0,
        )
        await backend.save_state(lora_id="lora_infer", checkpoint_record=ckpt, optimizer=False)
        backend.shutdown()
        time.sleep(3)

        print("\n=== Deploying to vLLM ===", flush=True)
        sampling_config = ModelConfig(
            model_name=model_path.name,
            model_path=model_path,
            max_model_len=2048,
            tensor_parallel_size=1,
            colocate=False,
            max_lora_rank=16,
        )
        sampling_backend = VLLMSamplingBackend(sampling_config)
        await sampling_backend.async_init()

        await sampling_backend.add_adapter("lora_infer", ckpt.adapter_path)
        print("  adapter loaded into vLLM", flush=True)

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        correct = 0
        total = len(PIG_LATIN_EXAMPLES)
        for prompt_text, example in zip(TEST_PROMPTS, PIG_LATIN_EXAMPLES, strict=True):
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            sample_res = await sampling_backend.sample(
                prompt=types.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=16, temperature=0.1, top_p=1.0, stop=["\n"]
                ),
                lora_id="lora_infer",
            )
            assert sample_res.sequences and sample_res.sequences[0].tokens
            output_text = tokenizer.decode(sample_res.sequences[0].tokens, skip_special_tokens=True)
            expected = _normalize_text(example["output"])
            got = _normalize_text(output_text)
            match = got == expected
            correct += int(match)
            print(
                f"  prompt='{prompt_text.strip()}'  "
                f"expected='{expected}'  got='{got}'  {'OK' if match else 'MISMATCH'}",
                flush=True,
            )

        print(f"\n  inference accuracy: {correct}/{total}", flush=True)
        assert correct >= total - 1, (
            f"Only {correct}/{total} samples matched Pig Latin after FSDP training. "
            f"At least {total - 1}/{total} expected."
        )


# ===================================================================
# Test 5 — GPU memory lifecycle
# ===================================================================
@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_gpu_memory_lifecycle(fsdp_ray_cluster):
    """Track GPU memory at each lifecycle stage — init, adapter creation,
    forward, backward, optim, and after adapter removal.
    Verify no memory explosion or leaks.
    """
    config = _make_config(MODEL_A_PATH)
    backend = FSDPTrainingBackend(config)
    await backend.async_init()
    data = _construct_data(MODEL_A_PATH, name="default")

    mem_init = await backend.get_memory_stats()
    _print_memory(mem_init, "after model init")

    await backend.create_adapter("mem_lora", types.LoraConfig(rank=8, seed=42))
    mem_adapter = await backend.get_memory_stats()
    _print_memory(mem_adapter, "after create_adapter")

    await backend.forward(
        data=data,
        lora_id="mem_lora",
        loss_fn="cross_entropy",
        loss_fn_config=None,
        backward=False,
    )
    mem_fwd = await backend.get_memory_stats()
    _print_memory(mem_fwd, "after forward (no backward)")

    await backend.forward(
        data=data,
        lora_id="mem_lora",
        loss_fn="cross_entropy",
        loss_fn_config=None,
        backward=True,
    )
    mem_bwd = await backend.get_memory_stats()
    _print_memory(mem_bwd, "after forward+backward")

    await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="mem_lora")
    mem_optim = await backend.get_memory_stats()
    _print_memory(mem_optim, "after optim_step")

    await backend.remove_adapter("mem_lora")
    mem_removed = await backend.get_memory_stats()
    _print_memory(mem_removed, "after remove_adapter")

    for r0_init, r0_removed in zip(mem_init, mem_removed, strict=True):
        delta = r0_removed["allocated_mb"] - r0_init["allocated_mb"]
        assert delta < 500, f"rank {r0_init['rank']}: {delta:.0f}MB leaked after remove_adapter"

    for s in mem_bwd:
        assert s["allocated_mb"] < 30000, (
            f"rank {s['rank']}: unexpectedly high memory "
            f"after forward+backward: {s['allocated_mb']:.0f}MB"
        )
