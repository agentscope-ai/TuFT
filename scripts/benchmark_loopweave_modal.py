"""Run focused, ephemeral two-GPU TuFT validation on Modal.

Usage: .venv/bin/modal run scripts/benchmark_loopweave_modal.py --repeats 3
Results are saved locally; no service is deployed.
"""

import json
import os
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
app = modal.App("tuft-loopweave-validation")
cache_name = os.environ.get("TUFT_MODAL_CACHE_VOLUME")
compiler_cache = modal.Volume.from_name(cache_name, create_if_missing=True) if cache_name else None
image = (
    modal.Image.from_registry("ghcr.io/agentscope-ai/tuft:latest", add_python="3.11")
    .entrypoint([])
    .env({"TUFT_MODAL_CACHE_VOLUME": cache_name or ""})
    .add_local_file(ROOT / "pyproject.toml", "/validation/pyproject.toml", copy=True)
    .run_commands(
        "/root/.local/bin/uv pip install --python /root/.tuft/venv/bin/python "
        "-r /validation/pyproject.toml zstandard pytest pytest-asyncio "
        "flash-linear-attention==0.5.2",
        # The published image can carry a flash-attn wheel built for old torch.
        # This validation uses PyTorch SDPA and vLLM's own attention kernels.
        "/root/.local/bin/uv pip uninstall --python /root/.tuft/venv/bin/python flash-attn",
    )
    .add_local_dir(ROOT / "src", "/validation/src", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(
        ROOT / "scripts", "/validation/scripts", ignore=["**/__pycache__/**", "**/*.pyc"]
    )
    .add_local_dir(ROOT / "tests", "/validation/tests", ignore=["**/__pycache__/**", "**/*.pyc"])
)


@app.function(
    image=image,
    gpu="H100:2",
    cpu=16,
    memory=65536,
    timeout=3600,
    volumes={"/compiler-cache": compiler_cache} if compiler_cache else {},
)
def benchmark(
    repeats: int,
    model: str,
    validate_tests: bool,
    output_tokens: int,
    diagnostics: bool,
    enforce_eager: bool,
):
    import subprocess

    env = {
        **os.environ,
        "PYTHONPATH": "/validation:/validation/src",
        "TUFT_CPU_TEST": "0",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PATH": "/root/.tuft/venv/bin:" + os.environ.get("PATH", ""),
    }
    if compiler_cache is not None:
        env.update(
            {
                "VLLM_CACHE_ROOT": "/compiler-cache/vllm",
                "TORCHINDUCTOR_CACHE_DIR": "/compiler-cache/inductor",
                "TRITON_CACHE_DIR": "/compiler-cache/triton",
            }
        )

    def invoke(command, **kwargs):
        try:
            subprocess.run(command, env=env, check=True, **kwargs)
        finally:
            if compiler_cache is not None:
                compiler_cache.commit()

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
    if diagnostics:
        command.append("--diagnostics")
    if enforce_eager:
        command.append("--enforce-eager")
    invoke(command)
    result = json.loads(Path("/tmp/results.json").read_text())
    if validate_tests:
        env["TUFT_TEST_MODEL"] = result["environment"]["model_path"]
        env["TUFT_TEST_MODEL_NAME"] = model
        env["TUFT_GPU_RESULT_DIR"] = "/tmp/gpu-test-results"
        config = json.loads((Path(result["environment"]["model_path"]) / "config.json").read_text())
        is_hybrid = config["model_type"] in {
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_5_text",
            "qwen3_5_moe_text",
        }
        for invariant in [True, False] if not is_hybrid else [False]:
            env["VLLM_BATCH_INVARIANT"] = "1" if invariant else "0"
            junit = f"/tmp/gpu-tests-{invariant}.xml"
            invoke(
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
                    f"--junitxml={junit}",
                ],
                cwd="/validation",
            )
            key = "gpu_deterministic_tests_junit" if invariant else "gpu_tests_junit"
            result[key] = Path(junit).read_text()
        result["gpu_test_results"] = [
            json.loads(path.read_text())
            for path in sorted(Path("/tmp/gpu-test-results").glob("*.json"))
        ]
    return result


@app.local_entrypoint()
def main(
    repeats: int = 3,
    model: str = "Qwen/Qwen3-0.6B",
    validate_tests: bool = False,
    output_tokens: int = 512,
    diagnostics: bool = False,
    enforce_eager: bool = False,
):
    result = benchmark.remote(  # type: ignore[attr-defined]
        repeats, model, validate_tests, output_tokens, diagnostics, enforce_eager
    )
    name = (
        f"gpu-integration-{model.rsplit('/', 1)[-1]}.json"
        if repeats == 0
        else f"gpu-{model.rsplit('/', 1)[-1]}-{output_tokens}.json"
    )
    name = "loopweave-static-" + name
    if diagnostics:
        name = "diagnostic-" + name
    if enforce_eager:
        name = "eager-" + name
    output = ROOT / "benchmarks" / "loopweave" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output}")
