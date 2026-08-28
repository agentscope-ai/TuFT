"""GPU integration test: FlexBackend single-backend RL mode with real Qwen3-0.6B.

Verifies the complete RL loop with a real model on a single GPU:
  training (forward/backward/optim) -> sampling (vLLM generate) -> training again

Requires: 1 GPU, Qwen3-0.6B model at TUFT_QWEN3_MODEL_ROOT.
Run: TUFT_FLEX_GPU_RL_LOOP=1 pytest tests/test_flex_gpu_rl_loop.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from tinker import types

from tuft.backends.flex.flex_backend import FlexBackendMode
from tuft.backends.flex.torchtp import FusedTorchTPVLLMFlexBackend
from tuft.config import ModelConfig


pytestmark = [pytest.mark.gpu, pytest.mark.integration]

QWEN3_ROOT = Path(os.getenv("TUFT_QWEN3_MODEL_ROOT", "/mnt/cpfs/shared/checkpoints/qwen/qwen3"))
MODEL_PATH = QWEN3_ROOT / "Qwen3-0.6B"


def _gpu_available() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() >= 1
    except Exception:
        return False


def _skip_unless_gpu_and_model() -> None:
    if os.getenv("TUFT_FLEX_GPU_RL_LOOP", "0") != "1":
        pytest.skip("set TUFT_FLEX_GPU_RL_LOOP=1 to run GPU RL loop test")
    if not _gpu_available():
        pytest.skip("no GPU available")
    if not MODEL_PATH.exists():
        pytest.skip(f"model not found: {MODEL_PATH}")
    # Required for vLLM collective_rpc with function arguments
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@pytest.fixture()
def flex_config() -> ModelConfig:
    # Uses the optimal FlexBackend defaults (enforce_eager=False + pre-capture
    # alias + CUDA Graph sampling, keep-runtime Flex sleep/wake switching) so
    # this test validates the production high-performance path end-to-end on a
    # real model. Only test-specific knobs (memory fraction, context len, batch)
    # are overridden here.
    return ModelConfig(
        model_name="qwen3-0.6b",
        model_path=MODEL_PATH,
        max_model_len=512,
        tensor_parallel_size=1,
        training_backend="flex",
        sampling_memory_fraction=0.45,
        sampling_max_model_len=512,
        micro_batch_size=2,
    )


def _make_training_batch(device: torch.device, batch_size: int = 2, seq_len: int = 64):
    """Create a synthetic training batch with target tokens and weights."""
    vocab_size = 151936  # Qwen3 vocab
    input_ids = torch.randint(100, vocab_size - 100, (batch_size, seq_len), device=device)
    # Target tokens: shifted input (next-token prediction)
    target_tokens = input_ids[:, 1:].contiguous()
    weights = torch.ones(batch_size, seq_len - 1, device=device)
    data = []
    for i in range(batch_size):
        data.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(input_ids[i].tolist()),
                loss_fn_inputs={
                    "target_tokens": types.TensorData.from_torch(target_tokens[i].cpu()),
                    "weights": types.TensorData.from_torch(weights[i].cpu()),
                },
            )
        )
    return data


@pytest.mark.asyncio
async def test_flex_backend_gpu_rl_loop(flex_config: ModelConfig) -> None:
    """End-to-end RL loop with the correct flow: sample -> train -> sample -> train.

    Uses a real Qwen3-0.6B on a single GPU with the optimal FlexBackend switching
    config (enforce_eager=False + pre-capture alias + CUDA Graph sampling and
    keep-runtime Flex sleep/wake). Each round rolls out first (sampling) and then
    updates (training), mirroring a real single-GPU RL training loop.

    Verifies per round:
    1. Transform to sampling works (vLLM create on round 1, warm wake afterwards)
    2. Sampling generates real tokens
    3. Transform back to training works (Flex sleep) and adapter is auto-restored
    4. Training produces finite, positive loss and does not explode
    Across rounds:
    5. Adapter state persists across multiple sampling/training round-trips
    """
    _skip_unless_gpu_and_model()

    backend = FusedTorchTPVLLMFlexBackend(flex_config)
    await backend.async_init()
    assert backend.training_model is not None, "Training model should be loaded"
    assert backend.mode == FlexBackendMode.TRAINING
    # Warmup should have pre-created and slept the vLLM sampling runtime so the
    # first user-facing transform_to_sampling hits the fast wake path.
    assert backend._warmup_done, "async_init should have warmed up the sampling runtime"
    assert backend._sampling_runtime_asleep, "Sampling runtime should be slept after warmup"

    # --- Create LoRA adapter (the training run) ---
    await backend.create_adapter("rl_test", types.LoraConfig(rank=8, seed=42))
    assert "rl_test" in backend._adapter_optimizers

    device = next(backend.training_model.parameters()).device
    prompt = types.ModelInput.from_ints([1, 2, 3, 4, 5, 100, 200, 300])
    sampling_params = types.SamplingParams(
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=42,
    )

    num_rl_rounds = 2
    for round_idx in range(num_rl_rounds):
        # === Sampling phase first (rollout) ===
        result = await backend.transform_to_sampling()
        assert result.supported, f"Transform to sampling should be supported: {result.message}"
        assert backend.mode == FlexBackendMode.SAMPLING
        assert backend.vllm_engine is not None, "vLLM engine should be present"
        transform_ms = result.metrics.get("transform_ms", 0.0) if result.metrics else 0.0
        print(f"\n[GPU_RL_LOOP] Round {round_idx} transform to sampling: {result.metrics}")
        # With warmup, ALL rounds (including round 0) should hit the fast wake
        # path, not the ~100s cold start. Allow generous margin for the wake.
        assert transform_ms < 10_000, (
            f"Round {round_idx}: transform_to_sampling took {transform_ms:.0f}ms, "
            "expected <10s (warmup should have covered the cold start)"
        )

        sample_response = await backend.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=sampling_params,
        )
        assert len(sample_response.sequences) >= 1, "Should generate at least 1 sequence"
        generated_tokens = sample_response.sequences[0].tokens
        assert len(generated_tokens) > 0, "Should generate at least 1 token"
        print(f"[GPU_RL_LOOP] Round {round_idx} generated {len(generated_tokens)} tokens: {generated_tokens[:10]}...")

        # === Training phase (update) ===
        result = await backend.transform_to_training()
        assert result.supported, f"Transform to training should be supported: {result.message}"
        assert backend.mode == FlexBackendMode.TRAINING
        # Adapter must be automatically restored after the round-trip
        assert "rl_test" in backend._adapter_optimizers, (
            f"Round {round_idx}: adapter should be auto-restored after transform_to_training"
        )
        print(f"[GPU_RL_LOOP] Round {round_idx} transform to training: {result.metrics}")

        round_losses = []
        for step in range(3):
            batch = _make_training_batch(device)
            output = await backend.forward(
                data=batch,
                lora_id="rl_test",
                loss_fn="cross_entropy",
                loss_fn_config=None,
                backward=True,
            )
            loss = output.metrics.get("loss:sum", output.metrics.get("loss:mean", 0.0))
            assert loss > 0, f"Round {round_idx} step {step}: loss should be positive"
            assert torch.isfinite(torch.tensor(loss)), (
                f"Round {round_idx} step {step}: loss should be finite"
            )
            round_losses.append(loss)

            await backend.optim_step(
                adam_params=types.AdamParams(
                    learning_rate=1e-4,
                    beta1=0.9,
                    beta2=0.999,
                    eps=1e-8,
                    weight_decay=0.0,
                    grad_clip_norm=1.0,
                ),
                lora_id="rl_test",
            )

        print(f"[GPU_RL_LOOP] Round {round_idx} training losses: {round_losses}")
        # Loss should not explode within a round
        assert round_losses[-1] <= round_losses[0] * 1.5, (
            f"Round {round_idx}: loss should not explode: {round_losses}"
        )

    print("[GPU_RL_LOOP] PASSED: Full sampling->training RL loop completed successfully")
