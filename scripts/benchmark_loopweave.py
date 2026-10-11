"""Real-engine multi-tenant comparison on the same two GPUs.

The baseline uses current static allocation: one HF GPU and one vLLM GPU.
LoopWeave adds a bounded vLLM sampler to the HF GPU. Both execute identical
sampling token budgets and LoRA training data. Initialization is excluded.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path


class GPUMonitor:
    """Sample NVML utilization and memory; keep the raw observations."""

    def __init__(self):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())
        ]
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        started = time.perf_counter()
        while not self.stop.is_set():
            self.samples.append(
                {
                    "elapsed_s": time.perf_counter() - started,
                    "gpu": [
                        {
                            "sm_percent": self.nvml.nvmlDeviceGetUtilizationRates(h).gpu,
                            "memory_bytes": self.nvml.nvmlDeviceGetMemoryInfo(h).used,
                        }
                        for h in self.handles
                    ],
                }
            )
            self.stop.wait(0.2)

    def finish(self):
        self.stop.set()
        self.thread.join()
        self.nvml.nvmlShutdown()
        return self.samples


async def run(
    policy: str,
    model_path: str,
    model_name: str,
    rounds: int = 3,
    tenants: int = 4,
    samples_per_tenant: int = 32,
    output_tokens: int = 512,
):
    import ray
    from tinker import types

    from tuft.auth import User
    from tuft.backends.sampling_backend import DPSamplingBackend
    from tuft.config import AppConfig, ModelConfig, SchedulingConfig
    from tuft.state import ServerState

    ray.init(num_gpus=2, include_dashboard=False, log_to_driver=True)
    model = ModelConfig(
        model_name=model_name,
        model_path=Path(model_path),
        max_model_len=1024,
        sampling_max_model_len=1024,
        max_lora_rank=8,
        max_loras=4,
        micro_batch_size=4,
        colocate=policy == "loopweave",
        data_parallel_size=2 if policy == "loopweave" else 1,
        sampling_memory_fraction=0.45,
        sampling_enforce_eager=False,
        min_response_tokens=output_tokens,
        scheduling=SchedulingConfig.model_validate(
            {
                "policy": policy,
                "flex_max_tokens": 32 + output_tokens,
                "flex_max_inflight": max(4, tenants * samples_per_tenant // 2),
                "max_inflight_per_replica": tenants * samples_per_tenant,
                "sampling_alignment_s": 0.25,
            }
        ),
    )
    state = ServerState(
        AppConfig(
            supported_models=[model], checkpoint_dir=Path(tempfile.mkdtemp(prefix="tuft-bench-"))
        )
    )
    records = []
    latencies = []
    train_latencies = []
    try:
        await state.async_init()
        for i in range(tenants):
            user = f"tenant-{i}"
            session = state.create_session(
                types.CreateSessionRequest(sdk_version="benchmark", tags=[], user_metadata=None),
                User(user_id=user),
            )
            run_record = await state.create_model(
                session.session_id,
                model.model_name,
                types.LoraConfig(rank=8, train_unembed=False),
                user,
                None,
            )
            checkpoint = await state.save_checkpoint(
                run_record.training_run_id, user, "initial", "sampler"
            )
            sampler = await state.create_sampling_session(
                session.session_id,
                model.model_name,
                checkpoint.tinker_path,
                user,
                session_seq_id=0,
            )
            records.append((user, run_record.training_run_id, sampler))

        prompt = types.ModelInput.from_ints([100 + i for i in range(32)])
        data = [
            types.Datum(
                model_input=types.ModelInput.from_ints(list(range(100, 164))),
                loss_fn_inputs={
                    "target_tokens": types.TensorData(data=list(range(101, 165)), dtype="int64"),
                    "weights": types.TensorData(data=[1.0] * 64, dtype="float32"),
                },
            )
            for _ in range(4)
        ]

        async def sample(user, sampler, seq):
            start = time.perf_counter()
            result = await state.run_sample(
                types.SampleRequest(
                    sampling_session_id=sampler,
                    seq_id=seq,
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=types.SamplingParams(
                        max_tokens=output_tokens, temperature=0.7, seed=seq
                    ),
                ),
                user,
            )
            latencies.append(time.perf_counter() - start)
            assert result.sequences and result.sequences[0].tokens
            assert all(
                s.logprobs is not None and len(s.logprobs) == len(s.tokens)
                for s in result.sequences
            )
            return sum(len(s.tokens) for s in result.sequences)

        # Warm both routes and training kernels, without changing optimizer state.
        await asyncio.gather(*(sample(user, sampler, 0) for user, _, sampler in records))
        for user, run_id, _ in records:
            await state.run_forward(run_id, user, data, "cross_entropy", None, None, backward=False)
        latencies.clear()
        zero_snapshot_logprob_error = None
        pool = state.sampling._base_backends[model.model_name]
        params = types.SamplingParams(max_tokens=output_tokens, temperature=0, seed=42)
        if isinstance(pool, DPSamplingBackend):
            # Compare both paths on the same engine at identical sampling settings.
            # No concurrent requests run during this verification.
            replica = pool._instances[-1]
            initial_id = records[0][2]
            assert initial_id in replica._zero_adapters
            merged = await replica.sample(prompt, 1, params, lora_id=initial_id)
            replica._zero_adapters.remove(initial_id)
            try:
                original = await replica.sample(prompt, 1, params, lora_id=initial_id)
            finally:
                replica._zero_adapters.add(initial_id)
            assert merged.sequences[0].tokens == original.sequences[0].tokens
            merged_logprobs = merged.sequences[0].logprobs
            original_logprobs = original.sequences[0].logprobs
            assert merged_logprobs is not None and original_logprobs is not None
            zero_snapshot_logprob_error = max(
                abs(a - b) for a, b in zip(merged_logprobs, original_logprobs, strict=True)
            )
            assert zero_snapshot_logprob_error < 0.005

        # Verify real tenant state and versioned publication before timing.
        # Restoring the initial optimizer checkpoint keeps measured work equal.
        from safetensors.torch import load_file

        alice, alice_id, _ = records[0]
        bob, bob_id, _ = records[1]
        alice_before = await state.save_checkpoint(alice_id, alice, "verify-before", "training")
        bob_before = await state.save_checkpoint(bob_id, bob, "verify-before", "training")
        before_a = load_file(str(alice_before.adapter_path / "adapter_model.safetensors"))
        before_b = load_file(str(bob_before.adapter_path / "adapter_model.safetensors"))
        await state.run_forward(alice_id, alice, data, "cross_entropy", None, None, backward=True)
        await state.run_optim_step(alice_id, alice, types.AdamParams(learning_rate=1e-3), None)
        alice_after = await state.save_checkpoint(alice_id, alice, "verify-after", "sampler")
        bob_after = await state.save_checkpoint(bob_id, bob, "verify-after", "training")
        after_a = load_file(str(alice_after.adapter_path / "adapter_model.safetensors"))
        after_b = load_file(str(bob_after.adapter_path / "adapter_model.safetensors"))
        assert any(not before_a[k].equal(after_a[k]) for k in before_a)
        assert all(before_b[k].equal(after_b[k]) for k in before_b)
        new_sampler = await state.create_sampling_session(
            state.training.training_runs[alice_id].session_id,
            model.model_name,
            alice_after.tinker_path,
            alice,
            session_seq_id=1,
        )
        await sample(alice, new_sampler, 0)
        if isinstance(pool, DPSamplingBackend):
            assert all(new_sampler not in r._zero_adapters for r in pool._instances)
        # Warm the nonzero-LoRA kernels on every engine as well as the base
        # path. Zero-adapter merging otherwise leaves flex LoRA JIT compilation
        # inside the measured second round. There is no concurrent training here.
        replicas = pool._instances if isinstance(pool, DPSamplingBackend) else [pool]
        for replica in replicas:
            await asyncio.gather(
                *(replica.sample(prompt, 1, params, lora_id=new_sampler) for _ in range(8))
            )
        await state.load_checkpoint(alice_id, alice, alice_before.tinker_path, optimizer=True)
        restored = await state.save_checkpoint(alice_id, alice, "verify-restored", "training")
        restored_a = load_file(str(restored.adapter_path / "adapter_model.safetensors"))
        assert all(before_a[k].equal(restored_a[k]) for k in before_a)
        latencies.clear()

        async def tenant(user, run_id, sampler):
            tokens = 0
            losses = []
            sequence = 1
            for step in range(rounds):
                tokens += sum(
                    await asyncio.gather(
                        *(
                            sample(user, sampler, 1 + step * samples_per_tenant + j)
                            for j in range(samples_per_tenant)
                        )
                    )
                )
                started = time.perf_counter()
                result = await state.run_forward(
                    run_id, user, data, "cross_entropy", None, sequence, backward=True
                )
                sequence += 1
                await state.run_optim_step(
                    run_id, user, types.AdamParams(learning_rate=1e-4), sequence
                )
                sequence += 1
                train_latencies.append(time.perf_counter() - started)
                losses.append(result.metrics)
                if step + 1 < rounds:
                    checkpoint = await state.save_checkpoint(
                        run_id, user, f"round-{step}", "sampler", seq_id=sequence
                    )
                    sequence += 1
                    sampler = await state.create_sampling_session(
                        state.training.training_runs[run_id].session_id,
                        model.model_name,
                        checkpoint.tinker_path,
                        user,
                        session_seq_id=step + 2,
                    )
                assert state.training.training_runs[run_id].next_seq_id == sequence
            return {"tokens": tokens, "losses": losses}

        scheduler = getattr(pool, "scheduler", None)
        scheduler_before = scheduler.snapshot() if scheduler else {}
        monitor = GPUMonitor()
        monitor.thread.start()
        started = time.perf_counter()
        try:
            outputs = await asyncio.gather(*(tenant(*record) for record in records))
        finally:
            gpu_samples = monitor.finish()
        elapsed = time.perf_counter() - started
        scheduler_after = scheduler.snapshot() if scheduler else {}
        return {
            "policy": policy,
            "config": model.model_dump(mode="json"),
            "workload": {
                "tenants": tenants,
                "rounds": rounds,
                "samples_per_tenant": samples_per_tenant,
                "output_tokens_per_sample": output_tokens,
            },
            "elapsed_s": elapsed,
            "gpu_seconds": elapsed * 2,
            "training_steps": tenants * rounds,
            "published_updates": tenants * (rounds - 1),
            "steps_per_s": tenants * rounds / elapsed,
            "output_tokens": sum(o["tokens"] for o in outputs),
            "sample_latency_mean_s": statistics.mean(latencies),
            "sample_latency_p95_s": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
            "train_latency_mean_s": statistics.mean(train_latencies),
            "scheduler": scheduler_after or None,
            "scheduler_measured_counters": {
                k: v - scheduler_before.get(k, 0)
                for k, v in scheduler_after.items()
                if isinstance(v, float) and k != "peak_queued"
            },
            "tenants": outputs,
            "gpu_samples": gpu_samples,
            "correctness": {
                "adapter_changed": True,
                "other_tenant_unchanged": True,
                "checkpoint_roundtrip": True,
                "new_snapshot_sampled": True,
                "zero_snapshot_max_logprob_error": zero_snapshot_logprob_error,
            },
        }
    finally:
        await state.shutdown()
        ray.shutdown()


def main():
    import torch
    import transformers
    import vllm
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=512, choices=[128, 512])
    parser.add_argument(
        "--model", default="Qwen/Qwen3-0.6B", choices=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revisions = {
        "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",  # pragma: allowlist secret
        "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",  # pragma: allowlist secret
    }
    model_path = snapshot_download(args.model, revision=revisions[args.model])
    result = {
        "environment": {
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "model_path": model_path,
            "model_name": args.model,
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(
                b"".join(
                    p.relative_to(Path(__file__).resolve().parents[1]).as_posix().encode()
                    + p.read_bytes()
                    for p in sorted((Path(__file__).resolve().parents[1] / "src").rglob("*.py"))
                )
            ).hexdigest(),
            "gpus": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                text=True,
            ),
        },
        "runs": [],
    }
    for repeat in range(args.repeats):
        # Alternate ordering to reduce systematic warm-cache bias.
        policies = ["round_robin", "loopweave"] if repeat % 2 == 0 else ["loopweave", "round_robin"]
        for policy in policies:
            print(f"BENCHMARK repeat={repeat} policy={policy}", flush=True)
            row = asyncio.run(run(policy, model_path, args.model, output_tokens=args.output_tokens))
            row["repeat"] = repeat
            result["runs"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    os.environ["TUFT_CPU_TEST"] = "0"
    main()
