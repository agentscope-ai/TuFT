"""Real HF/vLLM integration; run with --gpu on an otherwise idle two-GPU host."""

import os

import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.integration]


async def test_multi_tenant_training_publication_and_sampling():
    import torch
    from huggingface_hub import snapshot_download

    from scripts.benchmark_loopweave import run

    if torch.cuda.device_count() < 2:
        pytest.skip("LoopWeave hybrid integration requires two CUDA GPUs")
    path = os.environ.get("TUFT_TEST_MODEL") or snapshot_download(
        "Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",  # pragma: allowlist secret
    )
    result = await run(
        "loopweave",
        path,
        os.environ.get("TUFT_TEST_MODEL_NAME", "Qwen/Qwen3-0.6B"),
        rounds=2,
        tenants=2,
        samples_per_tenant=8,
        output_tokens=128,
    )
    assert result["training_steps"] == 4
    assert result["published_updates"] == 2
    assert result["output_tokens"] == 4096
    assert result["correctness"]["adapter_changed"]
    assert result["correctness"]["other_tenant_unchanged"]
    assert result["correctness"]["checkpoint_roundtrip"]
    assert result["correctness"]["zero_snapshot_max_logprob_error"] < 0.005
    assert result["scheduler"]["active"] == [0, 0]
    assert result["scheduler"]["replica_0_completed"] > 0
    assert result["scheduler"]["replica_1_completed"] > 0
