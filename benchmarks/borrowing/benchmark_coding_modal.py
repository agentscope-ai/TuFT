"""One static/shared-weight coding-agent comparison, with durable results and no dollar cap."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/validation")
BENCH = ROOT / "benchmarks/borrowing"
REMOTE = Path("/validation/benchmarks/borrowing")
PYTHON = "/root/.tuft/venv/bin/python"
MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # pragma: allowlist secret
app = modal.App("tuft-shared-weight-coding")
volume = modal.Volume.from_name("tuft-shared-weight-experiments", create_if_missing=True)
tool_image = modal.Image.debian_slim(python_version="3.12")
image = (
    modal.Image.from_registry("ghcr.io/agentscope-ai/tuft:latest", add_python="3.11")
    .entrypoint([])
    .add_local_file(ROOT / "pyproject.toml", "/validation/pyproject.toml", copy=True)
    .run_commands(
        f"/root/.local/bin/uv pip install --python {PYTHON} -r /validation/pyproject.toml "
        "zstandard flash-linear-attention==0.5.2 modal==1.6.1 transformers==5.19.0 tinker==0.25.0",
        f"/root/.local/bin/uv pip uninstall --python {PYTHON} flash-attn",
    )
    .add_local_dir(ROOT / "src", "/validation/src", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(BENCH, str(REMOTE), ignore=["**/__pycache__/**", "**/*.pyc", "runs/**"])
)


def environment():
    libs = sorted(Path("/root/.tuft/venv/lib").glob("python*/site-packages/nvidia/*/lib"))
    return {
        **os.environ,
        "PYTHONPATH": f"/validation/src:{REMOTE}",
        "TUFT_CPU_TEST": "0",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "VLLM_BATCH_INVARIANT": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PATH": "/root/.tuft/venv/bin:" + os.environ.get("PATH", ""),
        "LD_LIBRARY_PATH": ":".join(map(str, libs)) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    }


@app.function(image=image, cpu=4, memory=8192, timeout=3600, retries=0, volumes={"/data": volume})
def prepare(run_id: str, provenance: dict):
    """Download once and validate task contexts on CPU before any GPU allocation."""
    folder = Path("/data") / run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2))
    script = f"""
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from coding_tasks import initial_messages, next_messages
from benchmark_shared_weight_agents import encode_prompt
model = snapshot_download({MODEL!r}, revision={REVISION!r}, cache_dir="/data/models")
tokenizer = AutoTokenizer.from_pretrained(model)
lengths = []
for tenant in range(16):
    messages = initial_messages(tenant, 0, 0)
    for turn in range(4):
        lengths.append(len(encode_prompt(tokenizer, messages, 1536)))
        # Conservative context check: full output budget plus bounded test feedback.
        if turn < 3:
            messages = next_messages(tenant, 0, 0, turn, " code" * 1536,
                                     dict(error=" error" * 800))
result = dict(model_path=model, prompt_range=[min(lengths),max(lengths)], requests=len(lengths))
Path({str(folder / "preflight.json")!r}).write_text(json.dumps(result, indent=2))
print(result, flush=True)
"""
    try:
        subprocess.run([PYTHON, "-c", script], env=environment(), check=True)
        return json.loads((folder / "preflight.json").read_text())
    finally:
        volume.commit()


@app.function(image=image, cpu=1, memory=1024, timeout=300, retries=0, volumes={"/data": volume})
def resume_preflight(run_id: str, provenance: dict):
    """Reuse completed arms only with identical workload and implementation sources."""
    folder = Path("/data") / run_id
    old = json.loads((folder / "provenance.json").read_text())
    workload_files = {
        "benchmarks/borrowing/benchmark_shared_weight.py",
        "benchmarks/borrowing/benchmark_shared_weight_agents.py",
        "benchmarks/borrowing/coding_tasks.py",
    }
    for path, digest in old["source_sha256"].items():
        if path.startswith("src/") or path in workload_files:
            if provenance["source_sha256"].get(path) != digest:
                raise ValueError(f"Cannot reuse results after source change: {path}")
    for key in (
        "model",
        "model_revision",
        "gpu",
        "tenants",
        "trajectories",
        "updates_per_tenant",
        "milestones",
        "max_attempts",
        "max_output_tokens",
        "admission_timeout_s",
        "admitted_requests",
        "max_num_seqs",
        "max_loras",
        "kv_cache_gib",
        "flex_max_inflight",
    ):
        if old.get(key) != provenance.get(key):
            raise ValueError(f"Cannot reuse results after configuration change: {key}")
    return json.loads((folder / "preflight.json").read_text())


@app.function(image=image, timeout=3600, retries=0)
async def check_sandbox():
    sandbox = await modal.Sandbox.create.aio(
        app=app, image=tool_image, cpu=1, memory=1024, timeout=300, block_network=True
    )
    try:
        import asyncio

        script = """
import asyncio, inspect
from coding_tasks import SandboxTester, reference, cases, policy
async def main():
    tester = SandboxTester()
    await tester.start()
    for stage in range(4):
        code = inspect.getsource(reference) + (
            f"\\ndef build_report(orders):\\n"
            f"    return reference(orders, {stage}, {policy(0,0,0)!r})")
        result, _ = await tester.evaluate(code, cases(0,0,0,stage))
        assert result['passed'] == result['total'], result
    result, _ = await tester.evaluate('def build_report(x): return {}', cases(0,0,0,3))
    assert result['passed'] == 0, result
    print('Sandbox harness: four valid stages and invalid candidate verified')
asyncio.run(main())
"""
        env = environment()
        env["TUFT_CODING_SANDBOX_IDS"] = json.dumps([sandbox.object_id])
        process = await asyncio.create_subprocess_exec(PYTHON, "-c", script, env=env)
        assert await process.wait() == 0
        return "Sandbox harness verified"
    finally:
        await sandbox.terminate.aio()


@app.function(
    image=image,
    gpu="H100:2",
    cpu=16,
    memory=65536,
    timeout=86400,
    retries=0,
    volumes={"/data": volume},
)
async def compare(run_id: str, model_path: str, calibration: bool = False):
    import asyncio
    import sys

    await volume.reload.aio()
    folder = Path("/data") / run_id
    sandboxes = []
    # Restore to the same absolute path: Ninja build files embed source paths.
    kernel_cache = Path("/data/kernel-cache/flashinfer")
    local_cache = Path("/root/.cache/flashinfer")
    if kernel_cache.exists():
        shutil.copytree(kernel_cache, local_cache, dirs_exist_ok=True)
    try:
        # Bounded concurrency is part of the workload, not a spending cutoff.
        for _ in range(8):
            sandbox = await modal.Sandbox.create.aio(
                app=app, image=tool_image, cpu=1, memory=1024, timeout=86400, block_network=True
            )
            sandboxes.append(sandbox)
        env = environment()
        env["TUFT_CODING_SANDBOX_IDS"] = json.dumps([s.object_id for s in sandboxes])
        results = {}
        rounds, trajectories = (1, 3) if calibration else (4, 4)
        for policy in ("shared_weight",) if calibration else ("static", "shared_weight"):
            result_file = folder / f"{policy}.json"
            if result_file.exists():
                saved = json.loads(result_file.read_text())
                assert len(saved["updates"]) == 16 * rounds
                assert len(saved["episodes"]) == 16 * rounds * trajectories
                assert len(saved["requests"]) == sum(e.get("turns", 4) for e in saved["episodes"])
                assert saved["frozen_base_unchanged"]
                results[policy] = saved
                print("ARM_REUSED", policy, flush=True)
                continue
            # Preserve failed attempts before retrying the incomplete arm.
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            for suffix in ("log", "partial.json"):
                previous = folder / f"{policy}.{suffix}"
                if previous.exists():
                    previous.rename(folder / f"{policy}.failed-{stamp}.{suffix}")
            command = [
                PYTHON,
                str(REMOTE / "benchmark_shared_weight_agents.py"),
                "--policy",
                "borrow",
                "--model-path",
                model_path,
                "--workload",
                "coding",
                "--tenants",
                "16",
                "--trajectories",
                str(trajectories),
                "--rounds",
                str(rounds),
                "--output",
                str(folder / f"{policy}.json"),
            ]
            if policy == "static":
                command.append("--disable-borrowing")
            print("ARM_START", policy, flush=True)
            process = await asyncio.create_subprocess_exec(
                *command, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
            assert process.stdout is not None
            with (folder / f"{policy}.log").open("w") as log:
                async for line in process.stdout:
                    text = line.decode(errors="replace")
                    log.write(text)
                    log.flush()
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    if "UPDATE_DONE" in text or text.startswith("PROGRESS "):
                        await volume.commit.aio()
            code = await process.wait()
            await volume.commit.aio()
            if code:
                raise RuntimeError(f"{policy} failed with exit {code}; telemetry saved to {folder}")
            results[policy] = json.loads((folder / f"{policy}.json").read_text())
        (folder / ("calibration.json" if calibration else "comparison.json")).write_text(
            json.dumps(results)
        )
        return results
    finally:
        try:
            if local_cache.exists():
                shutil.copytree(local_cache, kernel_cache, dirs_exist_ok=True)
            await volume.commit.aio()
        finally:
            for sandbox in sandboxes:
                await sandbox.terminate.aio()


@app.local_entrypoint()
def main(resume_run: str = "", calibration: bool = False):
    import re

    if resume_run and not re.fullmatch(r"coding-\d{8}T\d{6}Z", resume_run):
        raise ValueError("Invalid run ID")
    prefix = "coding-calibration-" if calibration else "coding-"
    run_id = resume_run or prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = BENCH / "runs" / run_id
    output.mkdir(parents=True, exist_ok=bool(resume_run))
    sources = list((ROOT / "src").rglob("*.py")) + list(BENCH.glob("*.py"))
    provenance = dict(
        run_id=run_id,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        model=MODEL,
        model_revision=REVISION,
        gpu="H100:2",
        policy_order=["shared_weight"] if calibration else ["static", "shared_weight"],
        calibration=calibration,
        tenants=16,
        trajectories=3 if calibration else 4,
        updates_per_tenant=1 if calibration else 4,
        milestones=4,
        max_attempts=4,
        max_output_tokens=1536,
        admitted_requests=32,
        max_num_seqs=32,
        max_loras=16,
        kv_cache_gib=16,
        flex_max_inflight=4,
        dollar_cap=None,
        automatic_retries=0,
        admission_timeout_s=600,
        sandbox_pool_size=8,
    )
    if not resume_run:
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("RUN_ID", run_id, flush=True)
    if resume_run:
        preflight = resume_preflight.remote(run_id, provenance)  # type: ignore[attr-defined]
    else:
        print(check_sandbox.remote())  # type: ignore[attr-defined]
        preflight = prepare.remote(run_id, provenance)  # type: ignore[attr-defined]
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2))
    result = compare.remote(run_id, preflight["model_path"], calibration)  # type: ignore[attr-defined]
    result_name = "calibration.json" if calibration else "comparison.json"
    (output / result_name).write_text(json.dumps(result))
    print("SAVED", output, flush=True)
    plot_python = Path("/tmp/tuft-shared-docs-venv/bin/python")
    if plot_python.exists() and not calibration:
        subprocess.run(
            [
                str(plot_python),
                str(BENCH / "report_coding_comparison.py"),
                str(output / "comparison.json"),
            ],
            check=True,
        )
