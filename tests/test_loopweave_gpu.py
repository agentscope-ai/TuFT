"""Real HF/vLLM integration; run with --gpu on an otherwise idle two-GPU host."""

import importlib.util
import os
from pathlib import Path

import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.integration]


async def test_multi_tenant_training_publication_and_sampling():
    import torch
    from huggingface_hub import snapshot_download

    # Load by checkout path: the published container also has a top-level
    # `scripts` package, which shadows this repository's namespace directory.
    spec = importlib.util.spec_from_file_location(
        "tuft_loopweave_validation",
        Path(__file__).resolve().parents[1] / "scripts" / "benchmark_loopweave.py",
    )
    assert spec is not None and spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    run = benchmark.run

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
        rounds=5,
        tenants=2,
        samples_per_tenant=8,
        output_tokens=128,
    )
    assert result["training_steps"] == 10
    assert result["published_updates"] == 8
    assert result["output_tokens"] == 10240
    assert result["correctness"]["adapter_changed"]
    assert result["correctness"]["other_tenant_unchanged"]
    assert result["correctness"]["checkpoint_roundtrip"]
    assert result["correctness"]["zero_snapshot_max_logprob_error"] < 0.005
    assert result["scheduler"]["active"] == [0, 0]
    assert result["scheduler"]["replica_0_completed"] > 0
    assert result["scheduler"]["replica_1_completed"] > 0
