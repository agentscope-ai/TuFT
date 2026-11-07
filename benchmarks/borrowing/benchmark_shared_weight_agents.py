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
import os
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from benchmark_shared_weight import GPUMonitor


def scheduling(policy: str, workload: str = "warehouse") -> dict:
    return dict(
        borrowing=dict(
            enabled=policy in {"borrow", "shared_weight"},
            max_inflight=4,
            max_request_tokens=8192,
        ),
        queue_timeout_s=600 if workload == "coding" else 120,
        max_inflight_per_replica=32 if workload == "coding" else 16,
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


def initial_messages(spec: dict, tenant: int) -> list[dict]:
    # Reference documents vary by tenant and contain meaningful tool
    # instructions among distractor warehouse records (~2k/~4k tokens).
    reference = "\n".join(
        f"Archive {j}: warehouse region {j % 9}; reserve {j % 7}; "
        "archived stock is not current. Use live lookup for decisions."
        for j in range(65 if tenant % 2 == 0 else 130)
    )
    return [
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


def encode_prompt(tokenizer, messages: list[dict], max_tokens: int = 96) -> list[int]:
    # Transformers 5 defaults to a BatchEncoding; Tinker needs flat token IDs.
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
        raise TypeError("Chat template must produce a flat list of integer token IDs")
    if len(tokens) + max_tokens > 8192:
        raise ValueError("Agent context exceeds configured model capacity")
    return tokens


def preflight(model_path: str) -> dict:
    from tinker import types
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    lengths = []
    for tenant in range(8):
        spec = task_spec(tenant, 0, 0)
        messages = initial_messages(spec, tenant)
        for turn in range(3):
            prompt = encode_prompt(tokenizer, messages)
            lengths.append(len(prompt))
            types.SampleRequest(
                sampling_session_id="preflight",
                seq_id=turn,
                prompt=types.ModelInput.from_ints(prompt),
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=96, temperature=0),
            )
            action = {"tool": "lookup", "key": spec["policy" if turn == 0 else "stock"]}
            response = json.dumps(action)
            tokens = tokenizer.encode(response, add_special_tokens=False)
            datum = training_datum(prompt, tokens, [-1.0] * len(tokens), 1.0)
            assert len(datum.model_input.to_ints()) == len(
                datum.loss_fn_inputs["target_tokens"].data
            )
            messages.extend(
                [
                    dict(role="assistant", content=response),
                    dict(role="user", content="Tool observation: " + tool_reply(spec, action)[0]),
                ]
            )
    return dict(
        validated_requests=len(lengths),
        prompt_range=[min(lengths), max(lengths)],
        gpu_allocated=False,
    )


def capture_benchmark_adapters(model, optimizers, run_ids):
    """Keep warmup numerics out of the measured starting state (benchmark only)."""
    import copy

    import torch

    weights = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if any(f".{run_id}." in name for run_id in run_ids):
                # Nonzero B keeps all tenants on the real LoRA path even before
                # their first update; A is already initialized with a fixed seed.
                if "lora_B." in name:
                    parameter.fill_(1e-5)
                weights[name] = parameter.detach().cpu().clone()
    assert weights
    return weights, {run: copy.deepcopy(optimizers[run].state_dict()) for run in run_ids}


def restore_benchmark_adapters(model, optimizers, saved):
    import torch

    weights, states = saved
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in weights:
                parameter.copy_(weights[name])
                parameter.grad = None
    for run, state in states.items():
        optimizers[run].load_state_dict(state)


async def run(
    policy: str,
    model_path: str,
    rounds: int,
    tenants: int,
    trajectories: int,
    workload: str = "warehouse",
    checkpoint: Path | None = None,
    adapter_seed: int | None = None,
    representative_warmup: bool = False,
    borrowing_enabled: bool = True,
    tenant_stagger_s: float = 0.0,
    gdn_prefill_backend: Literal["flashinfer", "triton", "cutedsl"] | None = None,
    independent_clients=None,
):
    if policy == "borrow":
        policy = "shared_weight"
    if policy != "shared_weight":
        raise ValueError("Use policy='borrow' and borrowing_enabled=False for the static arm")
    import ray
    from tinker import types
    from transformers import AutoTokenizer

    from tuft.auth import User
    from tuft.backends.sampling_backend import DPSamplingBackend
    from tuft.backends.training_backend import HFTrainingBackend
    from tuft.config import AppConfig, ModelConfig, SamplingConfig
    from tuft.state import ServerState

    if representative_warmup and adapter_seed is None:
        raise ValueError("Representative warmup requires seeded adapter restoration")
    if adapter_seed is not None:
        # Benchmark-only actor: initialize each tenant identically across arms.
        # Changing routing can still change floating-point batching outcomes.
        import tuft.backends.hf_training_model as hf_module

        class SeededTrainingModel(hf_module.HFTrainingModel):
            async def capture_initial_state(self, run_ids):
                self._benchmark_initial_state = capture_benchmark_adapters(
                    self.model, self.adapter_optimizer, run_ids
                )

            async def restore_initial_state(self):
                restore_benchmark_adapters(
                    self.model, self.adapter_optimizer, self._benchmark_initial_state
                )
                del self._benchmark_initial_state

            async def create_adapter(self, *args, **kwargs):
                import torch

                index = getattr(self, "_benchmark_adapter_index", 0)
                torch.manual_seed(adapter_seed + index)
                self._benchmark_adapter_index = index + 1
                return await super().create_adapter(*args, **kwargs)

            async def adapter_fingerprints(self, run_ids):
                import hashlib

                import torch

                hashes = []
                for run_id in run_ids:
                    digest = hashlib.sha256()
                    for name, parameter in self.model.named_parameters():
                        if f".{run_id}." in name:
                            digest.update(name.replace(run_id, "adapter").encode())
                            digest.update(
                                parameter.detach()
                                .contiguous()
                                .view(torch.uint8)
                                .cpu()
                                .numpy()
                                .tobytes()
                            )
                    assert digest.hexdigest() != hashlib.sha256().hexdigest()
                    hashes.append(digest.hexdigest())
                return hashes

        from tuft.backends.base_backend import BaseTrainingBackend

        original_factory = BaseTrainingBackend.create_backend

        def seeded_factory(config, fsdp_index=None, worker_venv_path=None):
            if config.training_backend != "hf":
                return original_factory(config, fsdp_index, worker_venv_path)
            backend = HFTrainingBackend.__new__(HFTrainingBackend)
            BaseTrainingBackend.__init__(backend, config)
            backend.scheduler = None
            backend.model = SeededTrainingModel.get_actor(config)
            return backend

        BaseTrainingBackend.create_backend = staticmethod(seeded_factory)

    tester = None
    if workload == "coding":
        from coding_tasks import SandboxTester

        tester = SandboxTester()
        await tester.start()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ray.init(
        num_gpus=2,
        include_dashboard=False,
        log_to_driver=True,
        object_store_memory=2 * 1024**3,
    )
    model = ModelConfig(
        model_name="Qwen/Qwen3.5-4B" if workload == "coding" else "Qwen/Qwen3-4B",
        model_path=Path(model_path),
        training_backend="hf",
        max_model_len=8192,
        sampling_max_model_len=8192,
        sampling_language_model_only=True,
        sampling_gdn_prefill_backend=gdn_prefill_backend,
        sampling_enable_metrics=workload == "coding",
        sampling_max_num_seqs=32 if workload == "coding" else 16,
        max_lora_rank=8,
        max_loras=16 if workload == "coding" else 4,
        max_cpu_loras=128,
        micro_batch_size=1,
        colocate=policy == "shared_weight",
        data_parallel_size=2 if policy == "shared_weight" else 1,
        sampling_memory_fraction=0.45,
        sampling_kv_cache_memory_bytes=(16 if workload == "coding" else 4) * 1024**3,
        min_response_tokens=0,
        sampling=SamplingConfig(**scheduling(policy, workload)),
    )
    if not borrowing_enabled:
        # Keep identical shared-base placement and engines. Every benchmark
        # request exceeds one token, making the shared lane ineligible.
        model.sampling.borrowing.max_request_tokens = 1
    state = ServerState(
        AppConfig(supported_models=[model], checkpoint_dir=Path(tempfile.mkdtemp()))
    )
    requests, episodes, updates, publications = [], [], [], []
    monitor = None
    heartbeat = None
    metrics_task = None
    engine_metrics = []
    scheduling_samples = []
    scheduling_task = None
    save_partial = None
    measurement_started = None
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
            publications.append(
                dict(
                    tenant=user,
                    duration_s=time.perf_counter() - before,
                    start_s=before - measurement_started
                    if measurement_started is not None
                    else None,
                    finished_s=time.perf_counter() - measurement_started
                    if measurement_started is not None
                    else None,
                )
            )
            return sampler

        # Warm nonzero-adapter sampling and training shapes outside measurement.
        initial_adapter_hashes = None
        if representative_warmup:
            await trainer.model.capture_initial_state.remote([r[2] for r in records])
            initial_adapter_hashes = await trainer.model.adapter_fingerprints.remote(
                [r[2] for r in records]
            )
        warm_prompt = tokenizer.encode("Read the policy. Return a JSON decision to ship.")
        warm_data = [training_datum(warm_prompt, [100, 101, 102], [-5.0] * 3, 1)]
        samplers = []
        for record_index, record in enumerate(records):
            user, _, run_id = record
            training_warm_data = warm_data
            if representative_warmup and record_index == 0:
                # Exercise forward/backward at the measured coding batch sizes,
                # outside measurement, in both fresh arm processes.
                training_warm_data = [
                    training_datum([100] * prompt, [101] * output, [-5.0] * output, 1)
                    for _ in range(trajectories)
                    for prompt, output in ((1510, 198), (1840, 526), (2600, 674), (2899, 923))
                ]
            await state.run_forward(
                run_id, user, training_warm_data, "importance_sampling", None, None, backward=True
            )
            await state.run_optim_step(run_id, user, types.AdamParams(learning_rate=1e-5), None)
            samplers.append(await publish(record, "warm", 0))
        for replica_index, replica in enumerate(pool._instances):
            counts = (1, 4, 32) if workload == "coding" else (1, 4)
            if replica_index == scheduler.flex_index:
                counts = (1, 4)
                # Warmup bypasses request routing; explicitly register its
                # snapshots under the shared gate before direct engine calls.
                for adapter in samplers:
                    await scheduler.training(
                        lambda adapter=adapter: pool._ensure_borrowed_adapter(adapter),
                        kind="publication",
                    )
            for count in counts:
                await asyncio.gather(
                    *(
                        replica.sample(
                            types.ModelInput.from_ints(warm_prompt * 50),
                            1,
                            types.SamplingParams(max_tokens=96, temperature=0, seed=42),
                            lora_id=samplers[j % len(samplers)]
                            if workload == "coding"
                            else samplers[0],
                        )
                        for j in range(count)
                    )
                )
        if representative_warmup:
            await scheduler.training(trainer.model.restore_initial_state.remote)
            restored = await trainer.model.adapter_fingerprints.remote([r[2] for r in records])
            assert restored == initial_adapter_hashes, "Warmup restoration failed"
            samplers = [await publish(record, "initial", 1) for record in records]
        initial_hashes = await asyncio.gather(
            *(r.engine.base_weight_fingerprints.remote() for r in pool._instances)
        )
        adapter_hashes = (
            await trainer.model.adapter_fingerprints.remote([r[2] for r in records])
            if adapter_seed is not None
            else None
        )
        expected_hashes = os.environ.get("TUFT_EXPECTED_ADAPTER_HASHES")
        if expected_hashes is not None:
            assert adapter_hashes == json.loads(expected_hashes), "Starting adapters differ"
        shared = None
        if policy == "shared_weight":
            shared = await scheduler.training(pool._instances[0].engine.shared_weight_report.remote)
            assert shared["verified_alias_bytes"] > 0 and shared["copied_parameters"] == 0
        publications.clear()
        before = scheduler.snapshot()
        monitor = GPUMonitor()
        import psutil

        gpu_processes = []
        for index, handle in enumerate(monitor.handles):
            processes = []
            for proc in monitor.nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                try:
                    name = psutil.Process(proc.pid).name()
                except psutil.Error:
                    name = "unavailable"
                processes.append(dict(pid=proc.pid, name=name, memory_bytes=proc.usedGpuMemory))
            gpu_processes.append(
                dict(index=index, uuid=monitor.nvml.nvmlDeviceGetUUID(handle), processes=processes)
            )
        monitor.thread.start()
        started = time.perf_counter()
        measurement_started = started

        def save_progress(error=None):
            if checkpoint is None:
                return
            partial = dict(
                policy=policy,
                workload=workload,
                complete=False,
                independent_clients=independent_clients.metadata(tenants)
                if independent_clients
                else None,
                error=error,
                elapsed_s=time.perf_counter() - started,
                requests=requests,
                episodes=episodes,
                updates=updates,
                publications=publications,
                tool_results=tester.records if tester is not None else [],
                scheduler=scheduler.snapshot(),
                gpu=list(monitor.samples) if monitor else [],
                gpu_processes=gpu_processes,
                engine_metrics=list(engine_metrics),
                scheduling_samples=list(scheduling_samples),
            )
            tmp = checkpoint.with_suffix(".tmp")
            tmp.write_text(json.dumps(partial))
            tmp.replace(checkpoint)

        save_partial = save_progress

        async def report_progress():
            while True:
                save_progress()
                snapshot = scheduler.snapshot()
                print(
                    "PROGRESS",
                    json.dumps(
                        dict(
                            policy=policy,
                            elapsed_s=round(time.perf_counter() - started, 1),
                            requests=len(requests),
                            tools=len(tester.records) if tester else 0,
                            updates=len(updates),
                            queued=snapshot["queued"],
                            active=snapshot["active"],
                            training_active=snapshot["training_active"],
                            training_pending=snapshot["training_pending"],
                            shared_completed=snapshot.get("replica_0_completed", 0),
                            borrow_profitable_candidates=snapshot.get(
                                "borrow_profitable_candidates", 0
                            ),
                            borrow_duration_rejections=snapshot.get(
                                "borrow_duration_rejections", 0
                            ),
                            gpu=monitor.samples[-1]["gpu"] if monitor and monitor.samples else [],
                        )
                    ),
                    flush=True,
                )
                await asyncio.sleep(15)

        async def collect_metrics():
            while True:
                rows = await asyncio.gather(
                    *(r.engine.sampling_metrics.remote() for r in pool._instances)
                )
                engine_metrics.append(dict(elapsed_s=time.perf_counter() - started, replicas=rows))
                await asyncio.sleep(1)

        heartbeat = asyncio.create_task(report_progress())

        async def collect_scheduling():
            while True:
                snapshot = scheduler.snapshot()
                scheduling_samples.append(
                    dict(
                        elapsed_s=time.perf_counter() - started,
                        **{
                            key: snapshot[key]
                            for key in ("queued", "active", "training_active", "training_pending")
                        },
                    )
                )
                await asyncio.sleep(0.2)

        scheduling_task = asyncio.create_task(collect_scheduling())
        if workload == "coding":
            metrics_task = asyncio.create_task(collect_metrics())

        async def episode(tenant, step, trajectory, sampler):
            spec = task_spec(tenant, step, trajectory)
            messages = initial_messages(spec, tenant)
            outputs, valid = [], 0
            action = {}
            episode_start = time.perf_counter()
            for turn in range(3):
                prompt = encode_prompt(tokenizer, messages)
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

        async def coding_episode(tenant, step, trajectory, sampler):
            from coding_tasks import (
                cases,
                constants,
                extract_code,
                initial_messages as coding_messages,
                next_messages,
            )

            assert tester is not None
            messages = coding_messages(tenant, step, trajectory)
            outputs = []
            scores = []
            tool_wait = 0.0
            episode_start = time.perf_counter()
            for turn in range(4):
                stage = turn
                prompt = encode_prompt(tokenizer, messages, max_tokens=1536)
                begin = time.perf_counter()
                result = await state.run_sample(
                    types.SampleRequest(
                        sampling_session_id=sampler,
                        seq_id=step * trajectories * 4 + trajectory * 4 + turn,
                        prompt=types.ModelInput.from_ints(prompt),
                        num_samples=1,
                        sampling_params=types.SamplingParams(
                            max_tokens=1536, temperature=0, seed=tenant * 1000 + trajectory
                        ),
                    ),
                    records[tenant][0],
                )
                seq = result.sequences[0]
                assert seq.tokens and seq.logprobs is not None
                assert all(math.isfinite(v) for v in seq.logprobs)
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                requests.append(
                    dict(
                        tenant=tenant,
                        step=step,
                        trajectory=trajectory,
                        turn=turn,
                        milestone=stage,
                        start_s=begin - started,
                        duration_s=time.perf_counter() - begin,
                        prompt_tokens=len(prompt),
                        output_tokens=len(seq.tokens),
                        hit_output_limit=len(seq.tokens) == 1536,
                        snapshot=sampler,
                        response=text,
                    )
                )
                observation, wait = await tester.evaluate(
                    extract_code(text),
                    cases(tenant, step, trajectory, stage),
                    constants=constants(tenant, step, trajectory),
                    tenant=tenant,
                    step=step,
                    trajectory=trajectory,
                    turn=turn,
                    milestone=stage,
                    start_s=time.perf_counter() - started,
                )
                tool_wait += wait
                if independent_clients and turn < 3:
                    delay = independent_clients.tool_delay(tenant, step, trajectory, turn)
                    await asyncio.sleep(delay)
                    tool_wait += delay
                score = observation["passed"] / observation["total"]
                scores.append(score)
                outputs.append(training_datum(prompt, seq.tokens, seq.logprobs, score))
                if turn == 3:
                    break
                messages = next_messages(
                    tenant,
                    step,
                    trajectory,
                    stage,
                    text,
                    observation,
                )
            episodes.append(
                dict(
                    tenant=tenant,
                    step=step,
                    trajectory=trajectory,
                    duration_s=time.perf_counter() - episode_start,
                    reward=scores[-1],
                    success=scores[-1] == 1,
                    turns=4,
                    completed_milestones=sum(score == 1 for score in scores),
                    stage_rewards=scores,
                    tool_wait_s=tool_wait,
                )
            )
            return outputs

        async def tenant_loop(i):
            record, sampler = records[i], samplers[i]
            await asyncio.sleep(
                independent_clients.start_delay(i) if independent_clients else i * tenant_stagger_s
            )
            for step in range(rounds):
                if independent_clients:
                    await asyncio.sleep(independent_clients.think_delay(i, step))
                    if not independent_clients.can_start_round(time.perf_counter() - started):
                        break
                round_start = time.perf_counter()
                groups = await asyncio.gather(
                    *(
                        (coding_episode if tester is not None else episode)(i, step, j, sampler)
                        for j in range(trajectories)
                    )
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
                if step + 1 < rounds and (
                    independent_clients is None
                    or independent_clients.can_start_round(time.perf_counter() - started)
                ):
                    sampler = await publish(
                        record, f"step-{step}", step + 1 + int(representative_warmup)
                    )
                save_progress()
                print("UPDATE_DONE", policy, i, step, time.perf_counter() - started, flush=True)
            print("TENANT_DONE", policy, i, time.perf_counter() - started, flush=True)

        await asyncio.gather(*(tenant_loop(i) for i in range(tenants)))
        elapsed = time.perf_counter() - started
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)
        heartbeat = None
        gpu = monitor.finish()
        monitor = None
        after = scheduler.snapshot()
        if not borrowing_enabled:
            assert after.get("replica_0_completed", 0) == before.get("replica_0_completed", 0)
        final_hashes = await asyncio.gather(
            *(r.engine.base_weight_fingerprints.remote() for r in pool._instances)
        )
        assert initial_hashes == final_hashes, "Frozen base changed"
        return dict(
            adapter_seed=adapter_seed,
            representative_warmup=representative_warmup,
            borrowing_enabled=borrowing_enabled,
            tenant_stagger_s=tenant_stagger_s,
            independent_clients=independent_clients.metadata(tenants)
            if independent_clients
            else None,
            scheduling_samples=scheduling_samples,
            initial_adapter_fingerprints=adapter_hashes,
            policy=policy,
            workload=workload,
            tool_results=tester.records if tester is not None else [],
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
            gpu_processes=gpu_processes,
            engine_metrics=engine_metrics,
            shared=shared,
            frozen_base_unchanged=True,
            model_config=model.model_dump(mode="json"),
            versions={
                name: version(name) for name in ("torch", "vllm", "transformers", "tinker", "peft")
            },
        )
    except BaseException as exc:
        if save_partial is not None:
            save_partial(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if scheduling_task is not None:
            scheduling_task.cancel()
            await asyncio.gather(scheduling_task, return_exceptions=True)
        if metrics_task is not None:
            metrics_task.cancel()
            await asyncio.gather(metrics_task, return_exceptions=True)
        if heartbeat is not None:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        if monitor is not None:
            monitor.finish()
        await state.shutdown()
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["borrow", "shared_weight"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", choices=["warehouse", "coding"], default="warehouse")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--tenants", type=int, default=8)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--adapter-seed", type=int, default=None)
    parser.add_argument("--representative-warmup", action="store_true")
    parser.add_argument("--disable-borrowing", action="store_true")
    parser.add_argument("--tenant-stagger-s", type=float, default=0.0)
    parser.add_argument("--gdn-prefill-backend", choices=["flashinfer", "triton", "cutedsl"])
    parser.add_argument("--independent-client-seed", type=int)
    parser.add_argument("--arrival-window-s", type=float, default=240)
    parser.add_argument("--observation-s", type=float, default=600)
    parser.add_argument("--client-think-max-s", type=float, default=20)
    parser.add_argument("--tool-delay-max-s", type=float, default=3)
    args = parser.parse_args()
    clients = None
    if args.independent_client_seed is not None:
        from independent_clients import IndependentClients

        clients = IndependentClients(
            args.independent_client_seed,
            args.arrival_window_s,
            args.observation_s,
            args.client_think_max_s,
            args.tool_delay_max_s,
        )
        if args.tenant_stagger_s:
            parser.error("Independent arrivals cannot be combined with a fixed stagger")
    result = (
        preflight(args.model_path)
        if args.preflight
        else asyncio.run(
            run(
                args.policy,
                args.model_path,
                args.rounds,
                args.tenants,
                args.trajectories,
                args.workload,
                args.output.with_suffix(".partial.json"),
                args.adapter_seed,
                args.representative_warmup,
                not args.disable_borrowing,
                args.tenant_stagger_s,
                args.gdn_prefill_backend,
                clients,
            )
        )
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
