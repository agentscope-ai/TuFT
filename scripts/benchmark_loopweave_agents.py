"""Bounded causal agent workload; actual generated trajectories drive LoRA updates.

Two lookup turns followed by a decision, with long reference context and tool
latency. This is a systems workload, not a policy-convergence benchmark. Run each
arm in a fresh process with identical workload seeds and admission limits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import tempfile
import time
from pathlib import Path


if __package__:
    from .benchmark_loopweave import GPUMonitor
else:
    from benchmark_loopweave import GPUMonitor


def scheduling(policy: str) -> dict:
    return dict(
        policy=policy,
        max_inflight_per_replica=16,
        flex_max_inflight=4,
        flex_max_tokens=8192,
        borrow_probe_tokens=8192,
        sampling_alignment_s=0,
        scheduling_trace_capacity=4096,
    )


def task_spec(tenant: int, step: int, trajectory: int) -> dict:
    index = tenant * 100 + step * 10 + trajectory
    return dict(
        policy=f"policy-{index}",
        stock=f"stock-{index}",
        units=2 + index % 5,
        requested=3,
        delay_s=(0.05, 0.2, 0.8, 1.6)[index % 4],
    )


def parse_action(text: str) -> dict:
    decoder = json.JSONDecoder()
    for i, character in enumerate(text):
        if character == "{":
            try:
                value, _ = decoder.raw_decode(text[i:])
                if isinstance(value, dict):
                    return value
            except ValueError:
                pass
    return {}


def tool_reply(spec: dict, action: dict) -> tuple[str, bool]:
    key = action.get("key")
    if action.get("tool") != "lookup":
        return 'Error: use {"tool":"lookup","key":"..."}.', False
    if key == spec["policy"]:
        return json.dumps(
            dict(stock_key=spec["stock"], rule="Ship only if stock >= requested.")
        ), True
    if key == spec["stock"]:
        return json.dumps(dict(available=spec["units"], requested=spec["requested"])), True
    return (
        "Error: unknown key; use the policy key from the task or the stock key returned by lookup.",
        False,
    )


def reward(spec: dict, action: dict) -> float:
    expected = "ship" if spec["units"] >= spec["requested"] else "hold"
    return float(action.get("decision") == expected)


def training_datum(prompt: list[int], tokens: list[int], logprobs: list[float], score: float):
    from tinker import types

    assert tokens and len(tokens) == len(logprobs)
    # Targets are next-token aligned. Prompt tokens carry zero policy gradient.
    prefix = len(prompt) - 1
    sequence = prompt + tokens
    return types.Datum(
        model_input=types.ModelInput.from_ints(sequence[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=sequence[1:], dtype="int64"),
            "logprobs": types.TensorData(data=[0.0] * prefix + logprobs, dtype="float32"),
            "advantages": types.TensorData(
                data=[0.0] * prefix + [(score - 0.5) / len(tokens)] * len(tokens), dtype="float32"
            ),
        },
    )


async def run(policy: str, model_path: str, rounds: int, tenants: int, trajectories: int):
    import ray
    from tinker import types
    from transformers import AutoTokenizer

    from tuft.auth import User
    from tuft.backends.sampling_backend import DPSamplingBackend
    from tuft.backends.training_backend import HFTrainingBackend
    from tuft.config import AppConfig, ModelConfig, SchedulingConfig
    from tuft.state import ServerState

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ray.init(num_gpus=2, include_dashboard=False, log_to_driver=True)
    model = ModelConfig(
        model_name="Qwen/Qwen3-4B",
        model_path=Path(model_path),
        training_backend="hf",
        max_model_len=8192,
        sampling_max_model_len=8192,
        sampling_language_model_only=True,
        sampling_max_num_seqs=16,
        max_lora_rank=8,
        max_loras=4,
        max_cpu_loras=128,
        micro_batch_size=1,
        colocate=policy == "loopweave",
        data_parallel_size=2 if policy == "loopweave" else 1,
        sampling_memory_fraction=0.45,
        sampling_kv_cache_memory_bytes=4 * 1024**3,
        min_response_tokens=0,
        scheduling=SchedulingConfig(**scheduling(policy)),
    )
    state = ServerState(
        AppConfig(supported_models=[model], checkpoint_dir=Path(tempfile.mkdtemp()))
    )
    requests, episodes, updates, publications = [], [], [], []
    monitor = None
    try:
        await state.async_init()
        pool = state.sampling._base_backends[model.model_name]
        assert isinstance(pool, DPSamplingBackend)
        scheduler = pool.scheduler
        assert scheduler is not None
        trainer = state.training.training_backends[model.model_name]
        assert isinstance(trainer, HFTrainingBackend)
        trainer.scheduler = scheduler
        records = []
        for i in range(tenants):
            user = f"agent-{i}"
            session = state.create_session(
                types.CreateSessionRequest(
                    sdk_version="agent-benchmark", tags=[], user_metadata=None
                ),
                User(user_id=user),
            )
            run_record = await state.create_model(
                session.session_id,
                model.model_name,
                types.LoraConfig(rank=8, train_unembed=False),
                user,
                None,
            )
            records.append((user, session.session_id, run_record.training_run_id))

        async def publish(record, name, seq):
            user, session, run_id = record
            before = time.perf_counter()
            checkpoint = await state.save_checkpoint(run_id, user, name, "sampler")
            sampler = await state.create_sampling_session(
                session, model.model_name, checkpoint.tinker_path, user, session_seq_id=seq
            )
            publications.append(dict(tenant=user, duration_s=time.perf_counter() - before))
            return sampler

        # Warm nonzero-adapter sampling and training shapes outside measurement.
        warm_prompt = tokenizer.encode("Read the policy. Return a JSON decision to ship.")
        warm_data = [training_datum(warm_prompt, [100, 101, 102], [-5.0] * 3, 1)]
        samplers = []
        for record in records:
            user, _, run_id = record
            await state.run_forward(
                run_id, user, warm_data, "importance_sampling", None, None, backward=True
            )
            await state.run_optim_step(run_id, user, types.AdamParams(learning_rate=1e-5), None)
            samplers.append(await publish(record, "warm", 0))
        for replica in pool._instances:
            for count in (1, 4):
                await asyncio.gather(
                    *(
                        replica.sample(
                            types.ModelInput.from_ints(warm_prompt * 50),
                            1,
                            types.SamplingParams(max_tokens=96, temperature=0, seed=42),
                            lora_id=samplers[0],
                        )
                        for _ in range(count)
                    )
                )
        initial_hashes = await asyncio.gather(
            *(r.engine.base_weight_fingerprints.remote() for r in pool._instances)
        )
        shared = None
        if policy == "loopweave":
            shared = await scheduler.training(pool._instances[0].engine.shared_weight_report.remote)
            assert shared["verified_alias_bytes"] > 0 and shared["copied_parameters"] == 0
        # Discard warmup loop history, retaining measured service/switch calibration.
        for _, _, run_id in records:
            scheduler.profiler.forget(run_id)
        publications.clear()
        before = scheduler.snapshot()
        monitor = GPUMonitor()
        monitor.thread.start()
        started = time.perf_counter()

        async def episode(tenant, step, trajectory, sampler):
            spec = task_spec(tenant, step, trajectory)
            # Reference documents vary by tenant and contain meaningful tool
            # instructions among distractor warehouse records (~2k/~4k tokens).
            reference = "\n".join(
                f"Archive {j}: warehouse region {j % 9}; reserve {j % 7}; "
                "archived stock is not current. Use live lookup for decisions."
                for j in range(65 if tenant % 2 == 0 else 130)
            )
            messages = [
                dict(
                    role="system",
                    content="You are a warehouse agent. Use lookup tools to inspect live records. "
                    "Output only one JSON object per turn. "
                    'Tool syntax: {"tool":"lookup","key":"..."}. '
                    'Final syntax: {"decision":"ship"} or {"decision":"hold"}. '
                    "Do not guess live stock.\n" + reference,
                ),
                dict(
                    role="user",
                    content=f"Fulfill an order for {spec['requested']} units. "
                    f"First look up {spec['policy']}, then look up its stock_key, "
                    "then give the final decision.",
                ),
            ]
            outputs, valid = [], 0
            action = {}
            episode_start = time.perf_counter()
            for turn in range(3):
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
                assert len(prompt) + 96 <= 8192
                begin = time.perf_counter()
                result = await state.run_sample(
                    types.SampleRequest(
                        sampling_session_id=sampler,
                        seq_id=step * trajectories * 3 + trajectory * 3 + turn,
                        prompt=types.ModelInput.from_ints(prompt),
                        num_samples=1,
                        sampling_params=types.SamplingParams(
                            max_tokens=96, temperature=0, seed=tenant * 1000 + trajectory
                        ),
                    ),
                    records[tenant][0],
                )
                seq = result.sequences[0]
                assert (
                    seq.tokens
                    and seq.logprobs is not None
                    and all(math.isfinite(v) for v in seq.logprobs)
                )
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                outputs.append((prompt, seq.tokens, seq.logprobs))
                requests.append(
                    dict(
                        tenant=tenant,
                        step=step,
                        trajectory=trajectory,
                        turn=turn,
                        start_s=begin - started,
                        duration_s=time.perf_counter() - begin,
                        prompt_tokens=len(prompt),
                        output_tokens=len(seq.tokens),
                        snapshot=sampler,
                        response=text,
                    )
                )
                messages.append(dict(role="assistant", content=text))
                action = parse_action(text)
                if turn < 2:
                    reply, ok = tool_reply(spec, action)
                    valid += int(ok)
                    await asyncio.sleep(spec["delay_s"])
                    messages.append(
                        dict(
                            role="user",
                            content="Tool observation: "
                            + reply
                            + (
                                " Now give the final decision."
                                if turn == 1
                                else " Now look up the returned stock_key."
                            ),
                        )
                    )
            score = reward(spec, action)
            episodes.append(
                dict(
                    tenant=tenant,
                    step=step,
                    trajectory=trajectory,
                    duration_s=time.perf_counter() - episode_start,
                    reward=score,
                    valid_tools=valid,
                    tool_wait_s=2 * spec["delay_s"],
                )
            )
            return [training_datum(p, t, lp, score) for p, t, lp in outputs]

        async def tenant_loop(i):
            record, sampler = records[i], samplers[i]
            for step in range(rounds):
                round_start = time.perf_counter()
                groups = await asyncio.gather(
                    *(episode(i, step, j, sampler) for j in range(trajectories))
                )
                begin = time.perf_counter()
                result = await state.run_forward(
                    record[2],
                    record[0],
                    [d for g in groups for d in g],
                    "importance_sampling",
                    None,
                    None,
                    backward=True,
                )
                await state.run_optim_step(
                    record[2], record[0], types.AdamParams(learning_rate=1e-5), None
                )
                assert all(
                    math.isfinite(v) for v in result.metrics.values() if isinstance(v, (int, float))
                )
                updates.append(
                    dict(
                        tenant=i,
                        step=step,
                        duration_s=time.perf_counter() - begin,
                        loop_s=time.perf_counter() - round_start,
                        finished_s=time.perf_counter() - started,
                    )
                )
                if step + 1 < rounds:
                    sampler = await publish(record, f"step-{step}", step + 1)
            print("TENANT_DONE", policy, i, time.perf_counter() - started, flush=True)

        await asyncio.gather(*(tenant_loop(i) for i in range(tenants)))
        elapsed = time.perf_counter() - started
        gpu = monitor.finish()
        monitor = None
        after = scheduler.snapshot()
        final_hashes = await asyncio.gather(
            *(r.engine.base_weight_fingerprints.remote() for r in pool._instances)
        )
        assert initial_hashes == final_hashes, "Frozen base changed"
        return dict(
            policy=policy,
            elapsed_s=elapsed,
            rounds=rounds,
            tenants=tenants,
            trajectories=trajectories,
            requests=requests,
            episodes=episodes,
            updates=updates,
            publications=publications,
            scheduler_before=before,
            scheduler_after=after,
            gpu=gpu,
            shared=shared,
            frozen_base_unchanged=True,
            model_config=model.model_dump(mode="json"),
        )
    finally:
        if monitor is not None:
            monitor.finish()
        await state.shutdown()
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["balanced", "loopweave"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--tenants", type=int, default=8)
    parser.add_argument("--trajectories", type=int, default=4)
    args = parser.parse_args()
    result = asyncio.run(
        run(args.policy, args.model_path, args.rounds, args.tenants, args.trajectories)
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
