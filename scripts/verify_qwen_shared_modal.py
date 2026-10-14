"""Bounded Qwen hybrid checkpoint validation and performance on Modal.

Run HF with two B200s or FSDP with four H200s. Saves raw results locally;
GPU processes and checkpoint downloads live only for this invocation.
"""

import json
import os
import sys
from pathlib import Path

import modal


if Path("/validation/scripts").exists():
    sys.path.insert(0, "/validation/scripts")
from benchmark_loopweave_modal import image as base_image


ROOT = Path(__file__).resolve().parents[1]
image = base_image
app = modal.App("tuft-qwen-shared-validation")


@app.function(image=image, gpu="H100", cpu=8, timeout=1200)
def layout_probe():
    import subprocess

    libs = sorted(Path("/root/.tuft/venv/lib/python3.12/site-packages/nvidia").glob("*/lib"))
    result = subprocess.run(
        [
            "/root/.tuft/venv/bin/python",
            "-m",
            "pytest",
            "/validation/tests/test_qwen_vllm_shared_gpu.py",
            "--gpu",
            "--no-persistence",
            "-o",
            "addopts=",
            "-q",
            "-s",
            "--junitxml=/tmp/layout.xml",
        ],
        env={
            **os.environ,
            "PYTHONPATH": "/validation:/validation/src",
            "TUFT_CPU_TEST": "0",
            "LD_LIBRARY_PATH": ":".join(map(str, libs)),
            "OMP_NUM_THREADS": "1",
            "PATH": "/root/.tuft/venv/bin:" + os.environ.get("PATH", ""),
        },
        cwd="/validation",
        check=False,
    )
    return {"returncode": result.returncode, "junit": Path("/tmp/layout.xml").read_text()}


def execute(model, backend, repeats, validate_tests, single_split_lora=False, revision=""):
    import subprocess

    libs = sorted(Path("/root/.tuft/venv/lib/python3.12/site-packages/nvidia").glob("*/lib"))
    env = {
        **os.environ,
        "PYTHONPATH": "/validation:/validation/src",
        "TUFT_CPU_TEST": "0",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        # Large hybrid graphs spend minutes serializing AOT artifacts with 0.24.
        # Keep torch.compile/CUDA graphs, but use its non-AOT compilation path.
        "VLLM_USE_AOT_COMPILE": "0",
        "PATH": "/root/.tuft/venv/bin:" + os.environ.get("PATH", ""),
        "LD_LIBRARY_PATH": ":".join(map(str, libs)) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    }
    lora_configs = {}
    if single_split_lora:
        folder = Path("/tmp/tuft-lora-kernel-configs")
        folder.mkdir(exist_ok=True)
        # Published vLLM tuning interface; no kernel monkey patch. Keep the
        # upstream tile sizes and change only the shrink reduction partition.
        batches = {}
        for batch in (1, 64, 127, 128, 256, 512, 2048):
            batches[str(batch)] = {
                "4096": {
                    "8": {
                        "block_m": 32,
                        "block_n": 16,
                        "block_k": 256 if batch < 128 else 32,
                        "split_k": 1,
                        "num_warps": 4,
                        "num_ctas": 1,
                        "group_size_m": 8,
                        "num_stages": 2,
                        "max_nreg": None,
                    }
                }
            }
        config = {"4": {str(slices): batches for slices in range(1, 5)}}
        gpu_names = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).splitlines()
        for gpu_name in gpu_names:
            name = gpu_name.strip().replace(" ", "_").replace("-", "_")
            filename = f"{name}_SHRINK.json"
            (folder / filename).write_text(json.dumps(config))
            lora_configs[filename] = config
        env["VLLM_TUNED_CONFIG_FOLDER"] = str(folder)
        env["TUFT_TEST_SINGLE_SPLIT_LORA"] = "1"
    training_gpus = 2 if backend == "fsdp" else 1
    completed = subprocess.run(
        [
            "/root/.tuft/venv/bin/python",
            "/validation/scripts/benchmark_loopweave.py",
            "--model",
            model,
            *(["--revision", revision] if revision else []),
            "--training-backend",
            backend,
            "--training-gpus",
            str(training_gpus),
            "--repeats",
            str(repeats),
            "--kv-cache-gib",
            "4",
            "--output-tokens",
            "128",
            "--output",
            "/tmp/qwen-results.json",
        ],
        env=env,
        check=False,
        cwd="/validation",
    )
    result = json.loads(Path("/tmp/qwen-results.json").read_text())
    result["environment"]["lora_shrink_kernel_configs"] = lora_configs
    result["returncode"] = completed.returncode
    if completed.returncode:
        return result
    if validate_tests:
        env.update(
            {
                "TUFT_TEST_TRAINING_BACKEND": backend,
                "TUFT_TEST_TRAINING_GPUS": str(training_gpus),
                "TUFT_TEST_MODEL": result["environment"]["model_path"],
                "TUFT_TEST_MODEL_NAME": model,
                "TUFT_TEST_KV_CACHE_GIB": "4",
                "TUFT_GPU_RESULT_DIR": "/tmp/qwen-integration",
                # vLLM 0.24 explicitly rejects batch-invariant GDN attention.
                "VLLM_BATCH_INVARIANT": "0",
            }
        )
        completed = subprocess.run(
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
                "--junitxml=/tmp/qwen.xml",
            ],
            env=env,
            check=False,
            cwd="/validation",
        )
        result["returncode"] = completed.returncode
        result["gpu_tests_junit"] = Path("/tmp/qwen.xml").read_text()
        result["gpu_test_results"] = [
            json.loads(p.read_text()) for p in Path("/tmp/qwen-integration").glob("*.json")
        ]
    return result


@app.function(image=image, gpu="B200:2", cpu=16, memory=196608, timeout=7200)
def hf(
    model: str,
    repeats: int,
    validate_tests: bool,
    single_split_lora: bool = False,
    revision: str = "",
):
    return execute(model, "hf", repeats, validate_tests, single_split_lora, revision)


@app.function(image=image, gpu=["B200:2", "H200:2"], cpu=16, memory=196608, timeout=7200)
def hf_dense_control(
    model: str, repeats: int, validate_tests: bool, single_split_lora: bool, revision: str = ""
):
    # The dense 27B control fits either pair; MoE retains the larger B200 budget.
    return execute(model, "hf", repeats, validate_tests, single_split_lora, revision)


@app.function(image=image, gpu="H200:4", cpu=24, memory=262144, timeout=7200)
def fsdp(
    model: str,
    repeats: int,
    validate_tests: bool,
    single_split_lora: bool = False,
    revision: str = "",
):
    return execute(model, "fsdp", repeats, validate_tests, single_split_lora, revision)


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3.5-0.8B",
    backend: str = "hf",
    repeats: int = 0,
    validate_tests: bool = True,
    probe: bool = False,
    label: str = "",
    single_split_lora: bool = False,
    revision: str = "",
):
    if backend not in {"hf", "fsdp"}:
        raise ValueError("backend must be hf or fsdp")
    if label and not label.replace("-", "").isalnum():
        raise ValueError("label must contain only letters, numbers and hyphens")
    runner = fsdp if backend == "fsdp" else hf
    if backend == "hf" and model == "Qwen/Qwen3.8-27B" and single_split_lora:
        runner = hf_dense_control
    result = (
        layout_probe.remote()  # type: ignore[attr-defined]
        if probe
        else runner.remote(  # type: ignore[attr-defined]
            model, repeats, validate_tests, single_split_lora, revision
        )
    )
    destination = (
        ROOT
        / "benchmarks/loopweave"
        / (
            f"qwen-loopweave-static-{backend}-{model.rsplit('/', 1)[-1]}-repeats-{repeats}"
            f"{'-' + label if label else ''}.json"
        )
    )
    if probe:
        destination = ROOT / "benchmarks/loopweave/qwen-vllm-layout-validation.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {destination}")
    if result.get("returncode", 0):
        raise RuntimeError("Qwen integration failed; see saved JUnit result")
