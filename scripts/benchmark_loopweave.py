"""Real-engine multi-tenant comparison with equal GPU reservations.

The baseline uses one trainer group and one dedicated vLLM group. LoopWeave
adds a bounded sampler to the training group. All policies execute identical
sampling token budgets and LoRA training data. Initialization is excluded.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal


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
    kv_cache_bytes: int | None = None,
    numerical_diagnostics: bool = False,
    enforce_eager: bool = False,
    training_backend: Literal["hf", "fsdp"] = "hf",
    training_gpus: int = 1,
):
    import ray
    from tinker import types

    from tuft.auth import User
    from tuft.backends.sampling_backend import DPSamplingBackend
    from tuft.config import AppConfig, ModelConfig, SchedulingConfig
    from tuft.state import ServerState

    if policy not in {"round_robin", "loopweave"}:
        raise ValueError("Compare static round_robin with shared-weight loopweave")
    share_base_weights = policy == "loopweave"
    gpu_count = 2 * training_gpus
    ray.init(num_gpus=gpu_count, include_dashboard=False, log_to_driver=True)
    model = ModelConfig(
        model_name=model_name,
        model_path=Path(model_path),
        training_backend=training_backend,
        fsdp_num_gpus=training_gpus,
        fsdp_rank_slots={8: tenants} if training_backend == "fsdp" else None,
        tensor_parallel_size=training_gpus,
        max_model_len=1024,
        sampling_max_model_len=1024,
        sampling_language_model_only=True,
        sampling_moe_backend="triton",
        sampling_gdn_prefill_backend="triton",
        sampling_max_num_seqs=64,
        max_lora_rank=8,
        max_loras=4,
        micro_batch_size=4,
        colocate=policy == "loopweave",
        data_parallel_size=2 if policy == "loopweave" else 1,
        sampling_memory_fraction=0.45,
        sampling_kv_cache_memory_bytes=(16 * 1024**3 if kv_cache_bytes is None else kv_cache_bytes),
        sampling_enforce_eager=enforce_eager,
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
    initial_hashes: list[dict] = []
    reference_continuation: list[int] = []
    diagnostic_data = {}
    try:
        await state.async_init()
        initial_pool = state.sampling._base_backends[model.model_name]

        async def constant_probe():
            assert isinstance(initial_pool, DPSamplingBackend)
            outputs = []
            for replica in initial_pool._instances:
                output = await replica.sample(
                    types.ModelInput.from_ints(list(range(100, 260))),
                    1,
                    types.SamplingParams(max_tokens=output_tokens, temperature=1.0, seed=42),
                    include_prompt_logprobs=True,
                )
                outputs.append(output.prompt_logprobs)
            return outputs

        if numerical_diagnostics or share_base_weights:
            assert isinstance(initial_pool, DPSamplingBackend)
            diagnostic_data["before_scores"] = await constant_probe()
        if numerical_diagnostics:
            assert isinstance(initial_pool, DPSamplingBackend)
            diagnostic_data["before_buffers"] = await asyncio.gather(
                *(
                    replica.engine.model_buffer_fingerprints.remote()
                    for replica in initial_pool._instances
                )
            )
        if share_base_weights:
            initial_pool = state.sampling._base_backends[model.model_name]
            assert isinstance(initial_pool, DPSamplingBackend)
            initial_hashes = await asyncio.gather(
                *(
                    replica.engine.base_weight_fingerprints.remote()
                    for replica in initial_pool._instances
                )
            )
            mismatch = [
                key
                for key in initial_hashes[0]
                if initial_hashes[0][key] != initial_hashes[1].get(key)
            ]
            print("SHARED_INITIAL_WEIGHT_MISMATCH", mismatch, flush=True)
            assert not mismatch
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
        sharing_checks = None
        if share_base_weights:
            assert isinstance(pool, DPSamplingBackend) and pool.scheduler is not None
            engine = pool._instances[0].engine

            async def report():
                return await engine.shared_weight_report.remote()

            awake = await pool.scheduler.training(report)
            asleep = await pool.scheduler.training(report, reclaim_sampler_memory=True)
            restored = await pool.scheduler.training(report)
            assert awake["verified_alias_bytes"] == asleep["verified_alias_bytes"] > 0
            assert restored["verified_alias_bytes"] == awake["verified_alias_bytes"]
            reclaimed = asleep["gpu_free_bytes"] - awake["gpu_free_bytes"]
            assert reclaimed > 1024**3, "Expected at least 1 GiB of reclaimed sampler memory"
            sharing_checks = {
                "awake": awake,
                "asleep": asleep,
                "restored": restored,
                "reclaimed_bytes": reclaimed,
            }
            after_scores = await constant_probe()
            assert after_scores == diagnostic_data["before_scores"], (
                "Sleep/wake changed same-engine scores on identical inputs"
            )
            sharing_checks["base_switch_max_logprob_error"] = 0.0
            # Same initialized adapter on independent and aliased storage. These
            # calls run serially outside the measured interval.
            compare_params = types.SamplingParams(max_tokens=output_tokens, temperature=0, seed=42)
            left = await pool._instances[0].sample(prompt, 1, compare_params, lora_id=records[0][2])
            right = await pool._instances[1].sample(
                prompt, 1, compare_params, lora_id=records[0][2]
            )
            hashes = await asyncio.gather(
                *(replica.engine.base_weight_fingerprints.remote() for replica in pool._instances)
            )
            mismatch = [key for key in hashes[0] if hashes[0][key] != hashes[1].get(key)]
            first_difference = next(
                (
                    i
                    for i, (a, b) in enumerate(
                        zip(left.sequences[0].tokens, right.sequences[0].tokens, strict=True)
                    )
                    if a != b
                ),
                None,
            )
            print(
                "SHARED_AFTER_SWITCH",
                json.dumps(
                    {
                        "weight_mismatches": mismatch,
                        "first_token_difference": first_difference,
                        "left_tokens": left.sequences[0].tokens[:32],
                        "right_tokens": right.sequences[0].tokens[:32],
                        "memory": sharing_checks,
                    }
                ),
                flush=True,
            )
            assert not mismatch
            assert hashes == initial_hashes
            reference_continuation = left.sequences[0].tokens
            sharing_checks["first_greedy_token_difference"] = first_difference
            assert left.sequences[0].tokens[0] == right.sequences[0].tokens[0]

        async def compare_fixed_continuation(adapter, continuation):
            # Teacher-force identical histories. Tiny roundoff can change a late
            # greedy choice; subsequent autoregressive logprobs would then refer
            # to different inputs. Keep this probe outside the measured loop.
            assert isinstance(pool, DPSamplingBackend) and pool.scheduler is not None
            forced = types.ModelInput.from_ints(prompt.to_ints() + continuation[:128])
            scores = []
            for replica in pool._instances:
                result = await replica.sample(
                    forced,
                    1,
                    types.SamplingParams(max_tokens=output_tokens, temperature=1.0, seed=42),
                    include_prompt_logprobs=True,
                    lora_id=adapter,
                )
                scores.append(result.prompt_logprobs)
            assert scores[0] is not None and scores[1] is not None
            errors = [
                abs(a - b)
                for a, b in zip(scores[0], scores[1], strict=True)
                if a is not None and b is not None
            ]
            assert len(errors) == len(forced.to_ints()) - 1
            assert all(math.isfinite(error) for error in errors)
            maximum = max(errors)
            print("SHARED_FORCED_LOGPROB_ERROR", maximum, flush=True)
            if numerical_diagnostics:
                print(
                    "SHARED_PROBE_ERRORS",
                    json.dumps({"scores": scores, "errors": errors}),
                    flush=True,
                )
            # Cross-engine BF16 execution variation also occurs between two
            # ordinary engines (see saved numerical controls). Acceptance uses
            # exact frozen-weight hashes and same-engine switching invariance.
            return maximum

        async def verify_adapter_switch(adapter, continuation):
            assert isinstance(pool, DPSamplingBackend) and pool.scheduler is not None
            replica = pool._instances[0]
            forced = types.ModelInput.from_ints(prompt.to_ints() + continuation[:128])

            async def probe(target=replica):
                return await target.sample(
                    forced,
                    1,
                    types.SamplingParams(max_tokens=output_tokens, temperature=1.0, seed=42),
                    include_prompt_logprobs=True,
                    lora_id=adapter,
                )

            before = await pool.scheduler.training(probe)
            repeated = await pool.scheduler.training(probe)
            ordinary_controls = [await probe(pool._instances[-1]) for _ in range(3)]
            await pool.scheduler.training(replica.engine.reset_prefix_cache.remote)
            cache_reset = await pool.scheduler.training(probe)
            cache_repeated = await pool.scheduler.training(probe)
            adapter_before = await replica.engine.adapter_weight_fingerprints.remote()
            await pool.scheduler.training(
                replica.engine.shared_weight_report.remote, reclaim_sampler_memory=True
            )
            await pool.scheduler.training(replica.engine.shared_weight_report.remote)
            adapter_after = await replica.engine.adapter_weight_fingerprints.remote()
            after = await pool.scheduler.training(probe)
            additional_switch_errors = []
            # Compare identical cold-cache paths over repeated transitions.
            # A single favorable switch is insufficient for recurrent state.
            for _ in range(2):
                await pool.scheduler.training(replica.engine.reset_prefix_cache.remote)
                cold_before = await pool.scheduler.training(probe)
                await pool.scheduler.training(
                    replica.engine.shared_weight_report.remote, reclaim_sampler_memory=True
                )
                await pool.scheduler.training(replica.engine.shared_weight_report.remote)
                cold_after = await pool.scheduler.training(probe)
                assert cold_before.prompt_logprobs is not None
                assert cold_after.prompt_logprobs is not None
                additional_switch_errors.append(
                    max(
                        abs(a - b)
                        for a, b in zip(
                            cold_before.prompt_logprobs, cold_after.prompt_logprobs, strict=True
                        )
                        if a is not None and b is not None
                    )
                )
            adapter_after = await replica.engine.adapter_weight_fingerprints.remote()
            assert before.prompt_logprobs is not None
            assert repeated.prompt_logprobs is not None and after.prompt_logprobs is not None
            assert cache_reset.prompt_logprobs is not None
            assert cache_repeated.prompt_logprobs is not None
            cache_reset_error = max(
                abs(a - b)
                for a, b in zip(repeated.prompt_logprobs, cache_reset.prompt_logprobs, strict=True)
                if a is not None and b is not None
            )
            reset_to_switch_error = max(
                abs(a - b)
                for a, b in zip(cache_reset.prompt_logprobs, after.prompt_logprobs, strict=True)
                if a is not None and b is not None
            )
            repeat_error = max(
                abs(a - b)
                for a, b in zip(before.prompt_logprobs, repeated.prompt_logprobs, strict=True)
                if a is not None and b is not None
            )
            switch_error = max(
                abs(a - b)
                for a, b in zip(repeated.prompt_logprobs, after.prompt_logprobs, strict=True)
                if a is not None and b is not None
            )
            from itertools import combinations

            def spread(outputs):
                errors = []
                for left, right in combinations(outputs, 2):
                    assert left.prompt_logprobs is not None and right.prompt_logprobs is not None
                    errors.extend(
                        abs(a - b)
                        for a, b in zip(left.prompt_logprobs, right.prompt_logprobs, strict=True)
                        if a is not None and b is not None
                    )
                return max(errors)

            ordinary_spread = spread(ordinary_controls)
            shared_control_spread = spread([before, repeated, cache_reset, cache_repeated])
            print(
                "TRAINED_SWITCH_DIAGNOSTIC",
                json.dumps(
                    {
                        "repeat_error": repeat_error,
                        "switch_error": switch_error,
                        "cache_reset_error": cache_reset_error,
                        "reset_to_switch_error": reset_to_switch_error,
                        "additional_switch_errors": additional_switch_errors,
                        "ordinary_repeat_spread": ordinary_spread,
                        "shared_control_spread": shared_control_spread,
                        "adapter_mismatches": [
                            k for k in adapter_before if adapter_before[k] != adapter_after.get(k)
                        ],
                        "scores": [
                            before.prompt_logprobs,
                            repeated.prompt_logprobs,
                            after.prompt_logprobs,
                        ],
                    }
                ),
                flush=True,
            )
            assert adapter_before == adapter_after, "Sleep/wake changed GPU LoRA weights"
            assert math.isfinite(repeat_error) and math.isfinite(switch_error)
            deterministic = os.environ.get("VLLM_BATCH_INVARIANT") == "1"
            if deterministic:
                assert before.prompt_logprobs == repeated.prompt_logprobs == after.prompt_logprobs
                assert all(error == 0 for error in additional_switch_errors)
            checkpoint_config = json.loads((Path(model_path) / "config.json").read_text())
            hybrid = checkpoint_config["model_type"].startswith("qwen3_5")
            # Large models amplify ordinary BF16/LoRA reduction variation.
            # Single-split controls isolate it without requiring batch-invariant
            # GDN support (which vLLM 0.24 explicitly rejects).
            single_split = os.environ.get("TUFT_TEST_SINGLE_SPLIT_LORA") == "1"
            tolerance = 0.001
            if single_split:
                assert max(ordinary_spread, shared_control_spread) < tolerance
            else:
                tolerance += max(ordinary_spread, shared_control_spread)
            if hybrid:
                assert max(reset_to_switch_error, *additional_switch_errors) <= tolerance, (
                    "Hybrid sleep/wake exceeds the cold-cache log-probability error bound"
                )
            return {
                "repeat_max_logprob_error": repeat_error,
                "switch_max_logprob_error": switch_error,
                "cache_reset_max_logprob_error": cache_reset_error,
                "reset_to_switch_max_logprob_error": reset_to_switch_error,
                "additional_switch_max_logprob_errors": additional_switch_errors,
                "ordinary_repeat_max_logprob_spread": ordinary_spread,
                "shared_control_max_logprob_spread": shared_control_spread,
                "switch_logprob_tolerance": tolerance if hybrid else None,
                "single_split_lora": single_split,
                "ordinary_control_scores": [r.prompt_logprobs for r in ordinary_controls],
                "cache_reset_scores": [cache_reset.prompt_logprobs, cache_repeated.prompt_logprobs],
                "gpu_adapter_weights_unchanged": True,
                "deterministic": deterministic,
                "scores": [before.prompt_logprobs, repeated.prompt_logprobs, after.prompt_logprobs],
            }

        if share_base_weights:
            assert sharing_checks is not None
            sharing_checks["max_logprob_error"] = await compare_fixed_continuation(
                records[0][2], reference_continuation
            )
        if numerical_diagnostics:
            assert isinstance(initial_pool, DPSamplingBackend)
            diagnostic_data["after_scores"] = await constant_probe()
            diagnostic_data["after_buffers"] = await asyncio.gather(
                *(
                    replica.engine.model_buffer_fingerprints.remote()
                    for replica in initial_pool._instances
                )
            )
            print("NUMERICAL_DIAGNOSTICS", json.dumps(diagnostic_data), flush=True)
            # Diagnostics are deliberately separate from acceptance/performance
            # runs. Preserve their evidence even when a later, unrelated
            # autoregressive equivalence check would fail.
            return {
                "policy": policy,
                "diagnostic_only": True,
                "numerical_diagnostics": diagnostic_data,
                "shared_weights": sharing_checks,
            }
        params = types.SamplingParams(max_tokens=output_tokens, temperature=0, seed=42)
        if isinstance(pool, DPSamplingBackend):
            # Compare both paths on the same engine at identical sampling settings.
            # No concurrent requests run during this verification.
            replica = pool._instances[-1]
            initial_id = records[0][2]
            # Expert-parameter snapshots intentionally bypass the conservative
            # zero-adapter optimization. Still verify their initialized delta
            # against the base engine before training.
            zero_optimized = initial_id in replica._zero_adapters
            # Disable prefix-cache reads and score the same complete history;
            # a long greedy continuation can amplify a rounding difference.
            zero_probe = types.ModelInput.from_ints(list(range(100, 260)))
            probe_params = types.SamplingParams(max_tokens=output_tokens, temperature=1.0, seed=42)
            merged = await replica.sample(
                zero_probe, 1, probe_params, include_prompt_logprobs=True, lora_id=None
            )
            replica._zero_adapters.discard(initial_id)
            try:
                original = await replica.sample(
                    zero_probe, 1, probe_params, include_prompt_logprobs=True, lora_id=initial_id
                )
            finally:
                if zero_optimized:
                    replica._zero_adapters.add(initial_id)
            merged_logprobs = merged.prompt_logprobs
            original_logprobs = original.prompt_logprobs
            assert merged_logprobs is not None and original_logprobs is not None
            zero_snapshot_logprob_error = max(
                abs(a - b)
                for a, b in zip(merged_logprobs, original_logprobs, strict=True)
                if a is not None and b is not None
            )
            print("ZERO_ADAPTER_FORCED_LOGPROB_ERROR", zero_snapshot_logprob_error, flush=True)
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
        if share_base_weights:
            assert sharing_checks is not None
            sharing_checks["trained_max_logprob_error"] = await compare_fixed_continuation(
                new_sampler, reference_continuation
            )
            sharing_checks["trained_switch"] = await verify_adapter_switch(
                new_sampler, reference_continuation
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
        phase_before = {}
        if share_base_weights:
            assert isinstance(pool, DPSamplingBackend)
            phase_before = await pool._instances[0].engine.shared_weight_report.remote()
        monitor = GPUMonitor()
        monitor.thread.start()
        started = time.perf_counter()
        try:
            outputs = await asyncio.gather(*(tenant(*record) for record in records))
        finally:
            gpu_samples = monitor.finish()
        elapsed = time.perf_counter() - started
        scheduler_after = scheduler.snapshot() if scheduler else {}
        final_sharing = None
        phase_counters = None
        if share_base_weights:
            assert isinstance(pool, DPSamplingBackend) and pool.scheduler is not None
            phase_after = await pool._instances[0].engine.shared_weight_report.remote()
            phase_counters = {
                key: phase_after.get(key, 0) - phase_before.get(key, 0)
                for key in ("sleep_count", "wake_count", "sleep_s", "wake_s")
            }

            async def final_report():
                # Wake private vocabulary allocations too before hashing. Only
                # base parameters are hashed; LoRA buffers must have changed.
                final_hashes = await asyncio.gather(
                    *(
                        replica.engine.base_weight_fingerprints.remote()
                        for replica in pool._instances
                    )
                )
                assert final_hashes == initial_hashes
                return await pool._instances[0].engine.shared_weight_report.remote()

            final_sharing = await pool.scheduler.training(final_report)
            final_sharing["base_weights_unchanged"] = True
        return {
            "policy": policy,
            "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT") == "1",
            "shared_weights": sharing_checks,
            "shared_weights_final": final_sharing,
            "shared_weights_measured_counters": phase_counters,
            "numerical_diagnostics": diagnostic_data or None,
            "config": model.model_dump(mode="json"),
            "workload": {
                "tenants": tenants,
                "rounds": rounds,
                "samples_per_tenant": samples_per_tenant,
                "output_tokens_per_sample": output_tokens,
            },
            "elapsed_s": elapsed,
            "gpu_seconds": elapsed * gpu_count,
            "allocated_gpus": gpu_count,
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
    from importlib.metadata import PackageNotFoundError, version

    import torch
    import transformers
    import vllm
    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=512, choices=[128, 512])
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--training-backend", choices=["hf", "fsdp"], default="hf")
    parser.add_argument("--training-gpus", type=int, default=1)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--kv-cache-gib", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revisions = {
        "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",  # pragma: allowlist secret
        "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",  # pragma: allowlist secret
    }
    revision = args.revision or revisions.get(args.model) or HfApi().model_info(args.model).sha
    model_path = snapshot_download(args.model, revision=revision)
    result = {
        "environment": {
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "model_path": model_path,
            "model_name": args.model,
            "model_revision": revision,
            "vllm_use_aot_compile": os.environ.get("VLLM_USE_AOT_COMPILE", "default"),
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
    for package in ("peft", "flash-linear-attention", "causal-conv1d"):
        try:
            result["environment"][package] = version(package)
        except PackageNotFoundError:
            result["environment"][package] = None
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for repeat in range(args.repeats):
        # Alternate ordering to reduce systematic warm-cache bias.
        policies = ["round_robin", "loopweave"] if repeat % 2 == 0 else ["loopweave", "round_robin"]
        if args.diagnostics:
            policies = ["round_robin", "loopweave"]
        for policy in policies:
            print(f"BENCHMARK repeat={repeat} policy={policy}", flush=True)
            row = asyncio.run(
                run(
                    policy,
                    model_path,
                    args.model,
                    output_tokens=args.output_tokens,
                    kv_cache_bytes=args.kv_cache_gib * 1024**3,
                    numerical_diagnostics=args.diagnostics,
                    enforce_eager=args.enforce_eager,
                    training_backend=args.training_backend,
                    training_gpus=args.training_gpus,
                    rounds=1 if args.diagnostics else 3,
                    tenants=2 if args.diagnostics else 4,
                    samples_per_tenant=8 if args.diagnostics else 32,
                )
            )
            row["repeat"] = repeat
            result["runs"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    os.environ["TUFT_CPU_TEST"] = "0"
    main()
