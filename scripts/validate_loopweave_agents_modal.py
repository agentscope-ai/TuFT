"""One bounded agent-workload pair, no retries; preserve partial results."""

import json
import os
import string
import sys
import time
from pathlib import Path
from typing import Any

import modal


if Path("/validation/scripts").exists():
    sys.path.insert(0, "/validation/scripts")
from benchmark_loopweave_modal import ROOT, image


app = modal.App("tuft-loopweave-agent-validation")


@app.function(image=image, gpu="H100:2", cpu=16, memory=65536, timeout=1200, retries=0)
def compare():
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
    }
    python = "/root/.tuft/venv/bin/python"
    # Download in the pinned worker environment, once for both fresh arm processes.
    revision = "1cfa9a7208912126459214e8b04321603b3df60c"  # pragma: allowlist secret
    result: dict[str, Any] = dict(
        runs=[],
        requested_resources=dict(h100=2, cpu=16, memory_gib=64),
        function_timeout_s=1200,
        returncode=0,
        timed_out=False,
    )
    result["hardware"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], text=True
    )
    try:
        model_path = (
            subprocess.check_output(
                [
                    python,
                    "-c",
                    "from huggingface_hub import snapshot_download; "
                    f"print(snapshot_download('Qwen/Qwen3-4B', revision='{revision}'))",
                ],
                env=env,
                text=True,
                timeout=180,
            )
            .strip()
            .splitlines()[-1]
        )
        for policy in ("balanced", "loopweave"):
            destination = Path(f"/tmp/agents-{policy}.json")
            command = [
                python,
                "/validation/scripts/benchmark_loopweave_agents.py",
                "--policy",
                policy,
                "--model-path",
                model_path,
                "--output",
                str(destination),
            ]
            print("AGENT_ARM_START", policy, time.monotonic() - started, flush=True)
            completed = subprocess.run(
                command,
                env=env,
                cwd="/validation",
                timeout=max(1, 1140 - (time.monotonic() - started)),
                check=False,
            )
            result["returncode"] = completed.returncode
            if destination.exists():
                result["runs"].append(json.loads(destination.read_text()))
            if completed.returncode:
                break
    except subprocess.TimeoutExpired:
        result.update(returncode=124, timed_out=True)
    result["function_elapsed_s"] = time.monotonic() - started
    print("AGENT_PAIR_END", result["returncode"], result["function_elapsed_s"], flush=True)
    return result


@app.local_entrypoint()
def main(label: str):
    if not label or any(c not in string.ascii_lowercase + string.digits + "-_" for c in label):
        raise ValueError("Use a unique lowercase filename label")
    path = ROOT / "benchmarks/loopweave" / f"agents-{label}.json"
    if path.exists():
        raise FileExistsError(path)
    started = time.monotonic()
    result = compare.remote()  # type: ignore[attr-defined]
    result["client_elapsed_s"] = time.monotonic() - started
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {path}")
    if result["returncode"]:
        raise RuntimeError("Bounded run failed; partial results saved. Diagnose before any retry.")
