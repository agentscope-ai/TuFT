"""Focused multi-rank storage validation on an ephemeral Modal GPU host."""

import os
import sys
from pathlib import Path

import modal


if Path("/validation/scripts").exists():
    sys.path.insert(0, "/validation/scripts")

from benchmark_loopweave_modal import image


app = modal.App("tuft-fsdp-shared-validation")


@app.function(image=image, gpu="H100:2", cpu=8, timeout=1200)
def verify():
    import subprocess

    result = subprocess.run(
        [
            "/root/.tuft/venv/bin/python",
            "-m",
            "pytest",
            "/validation/tests/test_fsdp_shared_gpu.py",
            "--gpu",
            "--no-persistence",
            "-o",
            "addopts=",
            "-q",
            "-s",
            "--junitxml=/tmp/fsdp-shards.xml",
        ],
        env={
            **os.environ,
            "PYTHONPATH": "/validation:/validation/src",
            "TUFT_CPU_TEST": "0",
            "OMP_NUM_THREADS": "1",
        },
        cwd="/validation",
        check=False,
    )
    return {"returncode": result.returncode, "junit": Path("/tmp/fsdp-shards.xml").read_text()}


@app.function(image=image, gpu="H100:4", cpu=16, memory=65536, timeout=3600)
def integration(repeats: int, model: str, validate_tests: bool):
    import json
    import subprocess

    libs = sorted(Path("/root/.tuft/venv/lib/python3.12/site-packages/nvidia").glob("*/lib"))
    env = {
        **os.environ,
        "PYTHONPATH": "/validation:/validation/src",
        "TUFT_CPU_TEST": "0",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PATH": "/root/.tuft/venv/bin:" + os.environ.get("PATH", ""),
        "LD_LIBRARY_PATH": ":".join(map(str, libs)) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    }
    command = [
        "/root/.tuft/venv/bin/python",
        "/validation/scripts/benchmark_loopweave.py",
        "--training-backend",
        "fsdp",
        "--training-gpus",
        "2",
        "--model",
        model,
        "--repeats",
        str(repeats),
        "--output",
        "/tmp/fsdp-results.json",
    ]
    subprocess.run(command, env=env, check=True, cwd="/validation")
    result = json.loads(Path("/tmp/fsdp-results.json").read_text())
    if validate_tests:
        config = json.loads((Path(result["environment"]["model_path"]) / "config.json").read_text())
        env.update(
            {
                "TUFT_TEST_TRAINING_BACKEND": "fsdp",
                "TUFT_TEST_TRAINING_GPUS": "2",
                "TUFT_TEST_MODEL": result["environment"]["model_path"],
                "TUFT_TEST_MODEL_NAME": model,
                "TUFT_GPU_RESULT_DIR": "/tmp/fsdp-gpu-results",
                "VLLM_BATCH_INVARIANT": "1" if config["model_type"] == "qwen3" else "0",
            }
        )
        subprocess.run(
            [
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
                "--junitxml=/tmp/fsdp-integration.xml",
            ],
            env=env,
            check=True,
            cwd="/validation",
        )
        result["gpu_tests_junit"] = Path("/tmp/fsdp-integration.xml").read_text()
        result["gpu_test_results"] = [
            json.loads(p.read_text()) for p in Path("/tmp/fsdp-gpu-results").glob("*.json")
        ]
    return result


@app.local_entrypoint()
def main(
    full: bool = False,
    repeats: int = 0,
    model: str = "Qwen/Qwen3-0.6B",
    validate_tests: bool = True,
):
    import json

    result = (
        integration.remote(repeats, model, validate_tests)  # type: ignore[attr-defined]
        if full
        else verify.remote()  # type: ignore[attr-defined]
    )
    filename = (
        f"fsdp-loopweave-static-{model.rsplit('/', 1)[-1]}-repeats-{repeats}.json"
        if full
        else "fsdp-shards-validation.json"
    )
    destination = Path(__file__).resolve().parents[1] / "benchmarks/loopweave" / filename
    destination.write_text(json.dumps(result, indent=2) + "\n")
    if result.get("returncode", 0):
        raise RuntimeError("FSDP storage verification failed")
