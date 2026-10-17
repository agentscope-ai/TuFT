"""One bounded GPU job at a time, with partial results and no automatic retries.

modal run scripts/validate_loopweave_budget_modal.py --label <unique-label>
modal run scripts/validate_loopweave_budget_modal.py --label <unique-label> --fsdp

HF runs one equal-work static/LoopWeave pair with eight rounds and eight tenants.
FSDP runs the existing real-engine integration once, preserving two-rank sharding.
Neither mode runs batch-invariant numerical controls or large-model sweeps.
"""

import json
import os
import string
import sys
import time
from pathlib import Path

import modal


if Path("/validation/scripts").exists():
    sys.path.insert(0, "/validation/scripts")

from benchmark_loopweave_modal import ROOT, image


app = modal.App("tuft-loopweave-budget-validation")


def execute(fsdp: bool):
    import subprocess

    started = time.monotonic()
    libs = sorted(Path("/root/.tuft/venv/lib/python3.12/site-packages/nvidia").glob("*/lib"))
    env = {
        **os.environ,
        "PYTHONPATH": "/validation:/validation/src",
        "TUFT_CPU_TEST": "0",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "VLLM_BATCH_INVARIANT": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PATH": "/root/.tuft/venv/bin:" + os.environ.get("PATH", ""),
        "LD_LIBRARY_PATH": ":".join(map(str, libs)) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
        "TUFT_TEST_TRAINING_BACKEND": "fsdp",
        "TUFT_TEST_TRAINING_GPUS": "2",
        "TUFT_TEST_KV_CACHE_GIB": "4",
        "TUFT_GPU_RESULT_DIR": "/tmp/gpu-results",
    }
    if fsdp:
        command = [
            "/root/.tuft/venv/bin/python",
            "-m",
            "pytest",
            "/validation/tests/test_loopweave_gpu.py",
            "--gpu",
            "--no-persistence",
            "-o",
            "addopts=",
            "-q",
            "-s",
            "--junitxml=/tmp/junit.xml",
        ]
    else:
        command = [
            "/root/.tuft/venv/bin/python",
            "/validation/scripts/benchmark_loopweave.py",
            "--repeats",
            "1",
            "--rounds",
            "8",
            "--tenants",
            "8",
            "--samples-per-tenant",
            "16",
            "--output-tokens",
            "128",
            "--kv-cache-gib",
            "4",
            "--output",
            "/tmp/results.json",
        ]
    print("BOUNDED_JOB_START", json.dumps({"fsdp": fsdp, "command": command}), flush=True)
    timed_out = False
    try:
        completed = subprocess.run(command, env=env, cwd="/validation", timeout=1140, check=False)
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        returncode = 124
        timed_out = True
    result = {
        "backend": "fsdp" if fsdp else "hf",
        "returncode": returncode,
        "timed_out": timed_out,
        "function_elapsed_s": time.monotonic() - started,
        "requested_resources": {"h100": 4 if fsdp else 2, "cpu": 16, "memory_gib": 64},
        "subprocess_timeout_s": 1140,
        "function_timeout_s": 1200,
    }
    if Path("/tmp/results.json").exists():
        result["benchmark"] = json.loads(Path("/tmp/results.json").read_text())
    if Path("/tmp/junit.xml").exists():
        result["junit"] = Path("/tmp/junit.xml").read_text()
    result["gpu_test_results"] = [
        json.loads(path.read_text()) for path in sorted(Path("/tmp/gpu-results").glob("*.json"))
    ]
    print("BOUNDED_JOB_END", result["function_elapsed_s"], returncode, flush=True)
    return result


@app.function(image=image, gpu="H100:2", cpu=16, memory=65536, timeout=1200, retries=0)
def hf_job():
    return execute(False)


@app.function(image=image, gpu="H100:4", cpu=16, memory=65536, timeout=1200, retries=0)
def fsdp_job():
    return execute(True)


@app.local_entrypoint()
def main(label: str, fsdp: bool = False):
    if not label or any(c not in string.ascii_lowercase + string.digits + "-_" for c in label):
        raise ValueError("Use a unique lowercase filename label")
    destination = ROOT / "benchmarks/loopweave" / f"budget-{label}.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    started = time.monotonic()
    result = fsdp_job.remote() if fsdp else hf_job.remote()  # type: ignore[attr-defined]
    result["client_elapsed_s"] = time.monotonic() - started
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {destination}")
    if result["returncode"]:
        raise RuntimeError("GPU validation failed; partial artifacts saved. Do not retry blindly.")
