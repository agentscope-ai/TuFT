from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


pytestmark = [pytest.mark.gpu, pytest.mark.integration]


BENCH_ROOT = Path(os.getenv("TUFT_FLEX_BENCH_ROOT", "/mnt/nas/hanzhang.yhz/lora_rl_bench"))
QWEN3_ROOT = Path(os.getenv("TUFT_QWEN3_MODEL_ROOT", "/mnt/cpfs/shared/checkpoints/qwen/qwen3"))
MODEL_CASES = [
    pytest.param("qwen3-4b", QWEN3_ROOT / "Qwen3-4B-Base", id="qwen3-4b"),
    pytest.param("qwen3-32b", QWEN3_ROOT / "qwen3-32B", id="qwen3-32b"),
]
TP_CASES = [1, 2, 4]
SUMMARY_PATTERNS = {
    "mean_train_ms": r"mean_train_ms:\s*([0-9.]+)",
    "mean_ipc_desc_ms": r"mean_ipc_desc_ms:\s*([0-9.]+)",
    "mean_runtime_release_ms": r"mean_runtime_release_ms:\s*([0-9.]+)",
    "inject_alias_ms": r"inject_alias_ms:\s*([0-9.]+)",
    "switch_core_ms": r"switch_core_ms:\s*([0-9.]+)",
    "dummy_load_ms": r"dummy_load_ms\(one-time\):\s*([0-9.]+)",
    "sampling_throughput_tok_s": r"sampling_throughput:\s*([0-9.]+)",
    "sampling_latency_ms": r"sampling_latency_ms:\s*([0-9.]+)",
}
ROUNDTRIP_PATTERNS = {
    "rounds": r"round\s+([0-9]+):",
    "sampling_tput": r"sampling_tput=([0-9.]+)",
    "steady_t2s_ms": r"steady_t2s_ms=([0-9.]+)",
    "alloc_after_release_max": r"alloc_after_release_max=([0-9.]+)GB",
    "free_after_release_min": r"free_after_release_min=([0-9.]+)GB",
}


def _gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _parse_summary(output: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, pattern in SUMMARY_PATTERNS.items():
        match = re.search(pattern, output)
        if match:
            metrics[name] = float(match.group(1))
    readback = re.search(r"readback_exact:\s*(True|False)", output)
    if readback:
        metrics["readback_exact"] = readback.group(1) == "True"
    source_released = re.search(r"source_runtime_released:\s*(True|False)", output)
    if source_released:
        metrics["source_runtime_released"] = source_released.group(1) == "True"
    metrics["done"] = "DONE" in output
    return metrics


def _benchmark_defaults(model_name: str, tp_size: int) -> dict[str, str]:
    if model_name == "qwen3-32b":
        gpu_memory_utilization = "0.45" if tp_size == 2 else "0.60"
        return {
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": "512",
            "max_tokens": "32",
            "sample_rounds": "1",
        }
    return {
        "gpu_memory_utilization": "0.25",
        "max_model_len": "1024",
        "max_tokens": "128",
        "sample_rounds": "3",
    }


def _parse_roundtrip_summary(output: str) -> dict[str, Any]:
    rounds = [int(item) for item in re.findall(ROUNDTRIP_PATTERNS["rounds"], output)]
    tputs = [float(item) for item in re.findall(ROUNDTRIP_PATTERNS["sampling_tput"], output)]
    steady = [float(item) for item in re.findall(ROUNDTRIP_PATTERNS["steady_t2s_ms"], output)]
    allocs = [
        float(item) for item in re.findall(ROUNDTRIP_PATTERNS["alloc_after_release_max"], output)
    ]
    frees = [
        float(item) for item in re.findall(ROUNDTRIP_PATTERNS["free_after_release_min"], output)
    ]
    return {
        "round_count": len(rounds),
        "min_sampling_tput": min(tputs) if tputs else 0.0,
        "max_steady_t2s_ms": max(steady) if steady else 0.0,
        "max_alloc_after_release_gb": max(allocs) if allocs else 0.0,
        "min_free_after_release_gb": min(frees) if frees else 0.0,
        "done": "DONE" in output,
    }


def _run_fused_benchmark(
    model_name: str,
    tp_size: int,
    timeout_s: int,
) -> tuple[str, dict[str, Any]]:
    script = Path(__file__).resolve().parent / "flexgpu_release_runner.py"
    if not script.exists():
        pytest.skip(f"benchmark script not found: {script}")

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
    env["PATH"] = f"/usr/local/cuda-12.9/bin:{env.get('PATH', '')}"

    defaults = _benchmark_defaults(model_name, tp_size)
    command = [
        sys.executable,
        str(script),
        "--model",
        model_name,
        "--tp-size",
        str(tp_size),
        "--train-batch",
        os.getenv("TUFT_FLEX_GPU_TRAIN_BATCH", "2"),
        "--train-seq-len",
        os.getenv("TUFT_FLEX_GPU_TRAIN_SEQ_LEN", "256"),
        "--gpu-memory-utilization",
        os.getenv(
            "TUFT_FLEX_GPU_MEMORY_UTILIZATION",
            defaults["gpu_memory_utilization"],
        ),
        "--max-model-len",
        os.getenv("TUFT_FLEX_GPU_MAX_MODEL_LEN", defaults["max_model_len"]),
        "--max-tokens",
        os.getenv("TUFT_FLEX_GPU_MAX_TOKENS", defaults["max_tokens"]),
        "--sample-rounds",
        os.getenv("TUFT_FLEX_GPU_SAMPLE_ROUNDS", defaults["sample_rounds"]),
        "--verify-inject",
    ]
    completed = subprocess.run(
        command,
        cwd=str(BENCH_ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    output = completed.stdout + "\n" + completed.stderr
    metrics = _parse_summary(output)
    if completed.returncode != 0:
        raise AssertionError(
            f"benchmark failed for model={model_name}, tp={tp_size}, "
            f"returncode={completed.returncode}\n{output[-8000:]}"
        )
    return output, metrics


@pytest.mark.parametrize(("model_name", "model_path"), MODEL_CASES)
@pytest.mark.parametrize("tp_size", TP_CASES)
def test_fused_torchtp_vllm_zero_copy_gpu_benchmark(
    model_name: str,
    model_path: Path,
    tp_size: int,
    tmp_path: Path,
    record_property: pytest.RecordProperty,  # pyright: ignore[reportAttributeAccessIssue]
) -> None:
    if not model_path.exists():
        pytest.skip(f"model path does not exist: {model_path}")
    available_gpus = _gpu_count()
    if available_gpus < tp_size:
        pytest.skip(f"requires {tp_size} GPUs, only {available_gpus} visible")

    if model_name == "qwen3-32b" and tp_size == 1:
        pytest.skip(
            "Qwen3-32B FlexGPU zero-copy TP=1 does not fit on a single A100-80GB: "
            "base storage keepalive leaves too little free memory for vLLM."
        )
    timeout_s = int(os.getenv("TUFT_FLEX_GPU_TIMEOUT_S", "3600"))
    output, metrics = _run_fused_benchmark(model_name, tp_size, timeout_s)

    artifact = tmp_path / f"{model_name}-tp{tp_size}-fused-zero-copy.json"
    artifact.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "model_path": str(model_path),
                "tp_size": tp_size,
                "metrics": metrics,
                "output_tail": output[-12000:],
            },
            indent=2,
            sort_keys=True,
        )
    )

    print(f"\n[FLEX_GPU_BENCH] model={model_name} tp={tp_size}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"[FLEX_GPU_BENCH] artifact={artifact}")
    record_property("model_name", model_name)
    record_property("model_path", str(model_path))
    record_property("tp_size", tp_size)
    for key, value in metrics.items():
        record_property(key, value)

    required_metrics = {
        "mean_train_ms",
        "mean_ipc_desc_ms",
        "mean_runtime_release_ms",
        "inject_alias_ms",
        "switch_core_ms",
        "sampling_throughput_tok_s",
        "sampling_latency_ms",
        "readback_exact",
        "source_runtime_released",
        "done",
    }
    missing = required_metrics.difference(metrics)
    assert not missing, f"missing benchmark metrics: {sorted(missing)}\n{output[-4000:]}"
    assert metrics["done"] is True
    assert metrics["readback_exact"] is True
    assert metrics["source_runtime_released"] is True
    assert metrics["mean_train_ms"] > 0
    assert metrics["mean_ipc_desc_ms"] >= 0
    assert metrics["mean_runtime_release_ms"] >= 0
    assert metrics["inject_alias_ms"] >= 0
    assert metrics["switch_core_ms"] >= metrics["inject_alias_ms"]
    assert metrics["sampling_throughput_tok_s"] > 0
    assert metrics["sampling_latency_ms"] > 0


def test_fused_torchtp_vllm_roundtrip_gpu_benchmark(
    tmp_path: Path,
    record_property: pytest.RecordProperty,  # pyright: ignore[reportAttributeAccessIssue]
) -> None:
    if os.getenv("TUFT_FLEX_GPU_ROUNDTRIP", "0") != "1":
        pytest.skip("set TUFT_FLEX_GPU_ROUNDTRIP=1 to run round-trip GPU benchmark")
    model_name = os.getenv("TUFT_FLEX_ROUNDTRIP_MODEL", "qwen3-4b")
    tp_size = int(os.getenv("TUFT_FLEX_ROUNDTRIP_TP", "2"))
    available_gpus = _gpu_count()
    if available_gpus < tp_size:
        pytest.skip(f"requires {tp_size} GPUs, only {available_gpus} visible")

    script = Path(__file__).resolve().parent / "flexgpu_roundtrip_runner.py"
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
    env["PATH"] = f"/usr/local/cuda-12.9/bin:{env.get('PATH', '')}"
    command = [
        sys.executable,
        str(script),
        "--model",
        model_name,
        "--tp-size",
        str(tp_size),
        "--rounds",
        os.getenv("TUFT_FLEX_ROUNDTRIP_ROUNDS", "2"),
        "--gpu-memory-utilization",
        os.getenv("TUFT_FLEX_ROUNDTRIP_GPU_MEMORY_UTILIZATION", "0.70"),
        "--max-model-len",
        os.getenv("TUFT_FLEX_ROUNDTRIP_MAX_MODEL_LEN", "1024"),
        "--max-tokens",
        os.getenv("TUFT_FLEX_ROUNDTRIP_MAX_TOKENS", "64"),
        "--num-prompts",
        os.getenv("TUFT_FLEX_ROUNDTRIP_NUM_PROMPTS", "4"),
        "--sample-rounds",
        os.getenv("TUFT_FLEX_ROUNDTRIP_SAMPLE_ROUNDS", "1"),
        "--descriptor-mode",
        "reuse",
    ]
    completed = subprocess.run(
        command,
        cwd=str(BENCH_ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=int(os.getenv("TUFT_FLEX_ROUNDTRIP_TIMEOUT_S", "5400")),
    )
    output = completed.stdout + "\n" + completed.stderr
    metrics = _parse_roundtrip_summary(output)
    artifact = tmp_path / f"{model_name}-tp{tp_size}-roundtrip.json"
    artifact.write_text(
        json.dumps({"metrics": metrics, "output_tail": output[-20000:]}, indent=2),
    )
    if completed.returncode != 0:
        raise AssertionError(f"round-trip benchmark failed\n{output[-12000:]}")
    assert metrics["done"]
    expected_rounds = int(os.getenv("TUFT_FLEX_ROUNDTRIP_ROUNDS", "2"))
    assert metrics["round_count"] >= expected_rounds
    assert metrics["min_sampling_tput"] > 0.0
    assert metrics["min_free_after_release_gb"] > 1.0
    print(f"\n[FLEX_GPU_ROUNDTRIP] model={model_name} tp={tp_size}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"[FLEX_GPU_ROUNDTRIP] artifact={artifact}")
    record_property("roundtrip_metrics", json.dumps(metrics, sort_keys=True))
