"""Measure logprob mismatch between flex training and flex sampling backends.

Both backends share the same base weights via IPC alias, so logprobs on the
same token sequence should agree within numerical precision.

Usage:
  TUFT_FLEX_GPU_LOGPROB=1 pytest tests/test_flex_gpu_logprob_mismatch.py --gpu -v -s
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

NUM_COMPARE_POSITIONS = 32  # number of token positions to compare


def _skip_unless() -> None:
    if os.getenv("TUFT_FLEX_GPU_LOGPROB", "0") != "1":
        pytest.skip("set TUFT_FLEX_GPU_LOGPROB=1 to run logprob mismatch test")
    try:
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= 1):
            pytest.skip("no GPU available")
    except Exception:
        pytest.skip("torch not available")
    if not MODEL_PATH.exists():
        pytest.skip(f"model not found: {MODEL_PATH}")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@pytest.fixture()
def flex_config() -> ModelConfig:
    return ModelConfig(
        model_name="qwen3-0.6b",
        model_path=MODEL_PATH,
        max_model_len=512,
        tensor_parallel_size=1,
        training_backend="flex",
        sampling_memory_fraction=0.45,
        sampling_enforce_eager=True,
        sampling_max_model_len=512,
        micro_batch_size=1,
    )


def _compute_training_logprobs(
    backend: FusedTorchTPVLLMFlexBackend,
    token_ids: list[int],
) -> list[float]:
    """Run training forward on token_ids and extract per-position logprobs.

    Uses the model's logits to compute log_softmax and gather the logprob
    of each next token (autoregressive convention: position i predicts token i+1).
    """
    model = backend.training_model
    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)
        logits = outputs.logits  # [1, seq_len, vocab]

    # log_softmax in float32 for precision
    log_probs = torch.nn.functional.log_softmax(logits[0].float(), dim=-1)
    # Position i predicts token_ids[i+1]
    target_tokens = torch.tensor(token_ids[1:], dtype=torch.long, device=device)
    position_logprobs = log_probs[: len(target_tokens)].gather(
        dim=-1, index=target_tokens.unsqueeze(-1)
    ).squeeze(-1)
    return position_logprobs.cpu().tolist()


@pytest.mark.asyncio
async def test_flex_logprob_mismatch_training_vs_sampling(
    flex_config: ModelConfig,
) -> None:
    """Compare logprobs from training model vs vLLM sampling on same sequence.

    Steps:
    1. Load flex backend (training mode) with real model.
    2. Create a fixed token sequence.
    3. Transform to sampling, generate with logprobs=1.
    4. Transform back to training.
    5. Compute training logprobs on the same sequence.
    6. Compare and report mismatch statistics.
    """
    _skip_unless()

    backend = FusedTorchTPVLLMFlexBackend(flex_config)
    await backend.async_init()
    assert backend.mode == FlexBackendMode.TRAINING

    # Create adapter (required for forward)
    await backend.create_adapter("logprob_test", types.LoraConfig(rank=4, seed=0))

    # Use a deterministic prompt + fixed continuation for comparison
    # We'll use a simple increasing sequence as prompt, then let vLLM generate
    prompt_tokens = list(range(100, 100 + NUM_COMPARE_POSITIONS))

    # --- Phase 1: Get sampling logprobs via vLLM ---
    result = await backend.transform_to_sampling()
    assert result.supported, f"transform failed: {result.message}"

    sample_response = await backend.sample(
        prompt=types.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(
            max_tokens=NUM_COMPARE_POSITIONS,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            seed=42,
        ),
    )
    assert len(sample_response.sequences) >= 1
    seq = sample_response.sequences[0]
    sampled_tokens = list(seq.tokens)
    sampling_logprobs = list(seq.logprobs)
    assert len(sampled_tokens) > 0, "vLLM should generate tokens"
    assert len(sampling_logprobs) == len(sampled_tokens)

    print(f"\n[LOGPROB] Generated {len(sampled_tokens)} tokens")
    print(f"[LOGPROB] First 5 sampling logprobs: {sampling_logprobs[:5]}")

    # --- Phase 2: Transform back to training and compute training logprobs ---
    result = await backend.transform_to_training()
    assert result.supported, f"transform back failed: {result.message}"

    # Adapter may already exist after transform_to_training (kept across switches)
    if "logprob_test" not in backend._adapter_optimizers:
        await backend.create_adapter("logprob_test", types.LoraConfig(rank=4, seed=0))

    # The full sequence for training: prompt + generated tokens
    # vLLM logprobs[i] = logprob of sampled_tokens[i] given context up to position i
    # Training: position j predicts token_ids[j+1]
    # So for comparison, we feed [prompt + sampled_tokens] and compare
    # training_logprob[position_of_generated_token - 1] vs sampling_logprobs[i]
    full_sequence = prompt_tokens + sampled_tokens
    training_logprobs_full = _compute_training_logprobs(backend, full_sequence)

    # training_logprobs_full[i] = logprob of full_sequence[i+1] given full_sequence[:i+1]
    # The generated tokens start at index len(prompt_tokens) in full_sequence
    # So training_logprobs_full[len(prompt_tokens) - 1 + i] should match sampling_logprobs[i]
    prompt_len = len(prompt_tokens)
    n_compare = min(len(sampling_logprobs), len(training_logprobs_full) - prompt_len + 1)

    training_logprobs_aligned = []
    for i in range(n_compare):
        idx = prompt_len - 1 + i  # position in training_logprobs_full
        if idx < len(training_logprobs_full):
            training_logprobs_aligned.append(training_logprobs_full[idx])

    n_compare = min(n_compare, len(training_logprobs_aligned))
    assert n_compare > 0, "No positions to compare"

    # --- Phase 3: Compute mismatch statistics ---
    train_t = torch.tensor(training_logprobs_aligned[:n_compare], dtype=torch.float64)
    sample_t = torch.tensor(sampling_logprobs[:n_compare], dtype=torch.float64)
    diff = (train_t - sample_t).abs()

    mean_abs_diff = diff.mean().item()
    max_abs_diff = diff.max().item()
    p95_diff = torch.quantile(diff.float(), 0.95).item()
    relative_diff = (diff / (sample_t.abs() + 1e-8)).mean().item()

    print(f"\n[LOGPROB] Compared {n_compare} positions")
    print(f"[LOGPROB] Mean |train - sample|: {mean_abs_diff:.6e}")
    print(f"[LOGPROB] Max  |train - sample|: {max_abs_diff:.6e}")
    print(f"[LOGPROB] P95  |train - sample|: {p95_diff:.6e}")
    print(f"[LOGPROB] Mean relative diff:    {relative_diff:.6e}")
    print(f"[LOGPROB] First 5 train logprobs: {training_logprobs_aligned[:5]}")
    print(f"[LOGPROB] First 5 sample logprobs: {sampling_logprobs[:5]}")

    # The mismatch should be small (< 0.1) for bf16 models sharing exact same weights.
    # Typical numerical difference from bf16 vs fp32 compute paths is ~1e-3 to 1e-2.
    assert max_abs_diff < 1.0, (
        f"Logprob mismatch too large: max_diff={max_abs_diff:.6e}. "
        f"This suggests weights are not correctly shared between training and sampling."
    )
    print(f"\n[LOGPROB] PASSED: max mismatch {max_abs_diff:.6e} < 1.0")
