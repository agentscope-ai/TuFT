"""Run focused, ephemeral two-GPU TuFT validation on Modal.

Usage: .venv/bin/modal run scripts/benchmark_loopweave_modal.py --repeats 3
Results are saved locally; no service is deployed.
"""

import json
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
app = modal.App("tuft-loopweave-validation")
image = (
    modal.Image.from_registry("ghcr.io/agentscope-ai/tuft:latest", add_python="3.11")
    .entrypoint([])
    .add_local_file(ROOT / "pyproject.toml", "/validation/pyproject.toml", copy=True)
    .run_commands(
        "/root/.local/bin/uv pip install --python /root/.tuft/venv/bin/python "
        "-r /validation/pyproject.toml zstandard pytest pytest-asyncio",
        # The published image can carry a flash-attn wheel built for old torch.
        # This validation uses PyTorch SDPA and vLLM's own attention kernels.
        "/root/.local/bin/uv pip uninstall --python /root/.tuft/venv/bin/python flash-attn",
    )
    .add_local_dir(ROOT / "src", "/validation/src")
    .add_local_dir(ROOT / "scripts", "/validation/scripts")
    .add_local_dir(ROOT / "tests", "/validation/tests")
)


@app.function(image=image, gpu="H100:2", cpu=16, memory=65536, timeout=3600)
def benchmark(repeats: int, model: str, validate_tests: bool, output_tokens: int):
    import os
    import subprocess

    env = {
        **os.environ,
        "PYTHONPATH": "/validation:/validation/src",
        "TUFT_CPU_TEST": "0",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    # The registry image may have an older system CUDA toolkit. The pinned
    # torch/vLLM wheels carry CUDA 13 libraries; cuDNN/FlashInfer dlopen NVRTC
    # directly and require these directories in the loader search path.
    wheel_libs = sorted(Path("/root/.tuft/venv/lib/python3.12/site-packages/nvidia").glob("*/lib"))
    env["LD_LIBRARY_PATH"] = ":".join(map(str, wheel_libs)) + ":" + env.get("LD_LIBRARY_PATH", "")
    command = [
        "/root/.tuft/venv/bin/python",
        "/validation/scripts/benchmark_loopweave.py",
        "--repeats",
        str(repeats),
        "--model",
        model,
        "--output-tokens",
        str(output_tokens),
        "--output",
        "/tmp/results.json",
    ]
    subprocess.run(command, env=env, check=True)
    result = json.loads(Path("/tmp/results.json").read_text())
    if validate_tests:
        env["TUFT_TEST_MODEL"] = result["environment"]["model_path"]
        env["TUFT_TEST_MODEL_NAME"] = model
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
                "--junitxml=/tmp/gpu-tests.xml",
            ],
            cwd="/validation",
            env=env,
            check=True,
        )
        result["gpu_tests_junit"] = Path("/tmp/gpu-tests.xml").read_text()
    return result


@app.local_entrypoint()
def main(
    repeats: int = 3,
    model: str = "Qwen/Qwen3-0.6B",
    validate_tests: bool = False,
    output_tokens: int = 512,
):
    result = benchmark.remote(repeats, model, validate_tests, output_tokens)  # type: ignore[attr-defined]
    output = (
        ROOT / "benchmarks" / "loopweave" / f"gpu-{model.rsplit('/', 1)[-1]}-{output_tokens}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output}")
