"""Real HF/vLLM integration; run with --gpu on an otherwise idle two-GPU host."""

import importlib.util
import json
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
        "tuft_shared_weight_validation",
        Path(__file__).resolve().parents[1] / "scripts" / "benchmark_shared_weight.py",
    )
    assert spec is not None and spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    run = benchmark.run

    backend = os.environ.get("TUFT_TEST_TRAINING_BACKEND", "hf")
    training_gpus = int(os.environ.get("TUFT_TEST_TRAINING_GPUS", "1"))
    if torch.cuda.device_count() < 2 * training_gpus:
        pytest.skip("Shared-weight borrowing hybrid integration requires two CUDA GPUs")
    path = os.environ.get("TUFT_TEST_MODEL") or snapshot_download(
        "Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",  # pragma: allowlist secret
    )
    result = await run(
        "shared_weight",
        path,
        os.environ.get("TUFT_TEST_MODEL_NAME", "Qwen/Qwen3-0.6B"),
        rounds=5,
        tenants=2,
        samples_per_tenant=8,
        output_tokens=128,
        training_backend=backend,
        training_gpus=training_gpus,
        kv_cache_bytes=int(os.environ.get("TUFT_TEST_KV_CACHE_GIB", "16")) * 1024**3,
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
    assert result["shared_weights"]["base_switch_max_logprob_error"] == 0
    switch = result["shared_weights"]["trained_switch"]
    assert switch["gpu_adapter_weights_unchanged"]
    if result["batch_invariant"]:
        assert switch["repeat_max_logprob_error"] == 0
        assert switch["switch_max_logprob_error"] == 0
    assert result["shared_weights_final"]["base_weights_unchanged"]
    assert result["shared_weights"]["reclaimed_bytes"] > 1024**3
    assert result["shared_weights_final"]["verified_alias_bytes"] > 0
    assert result["shared_weights_final"]["sleep_count"] >= 5
    assert result["shared_weights_final"]["wake_count"] >= 5
    if destination := os.environ.get("TUFT_GPU_RESULT_DIR"):
        directory = Path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        (
            directory / f"integration-shared_weight-invariant-{result['batch_invariant']}.json"
        ).write_text(json.dumps(result, indent=2) + "\n")
