"""Controller/backend integration with controllable Ray and sampling boundaries.

Real GPU engine coverage lives in scripts/benchmark_loopweave.py. These tests
exercise the production factories, shared gate, sequence guard and tenant routing
without CUDA, sockets, or a running service.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from tinker import types

from tuft.auth import User
from tuft.backends.base_backend import BaseSamplingBackend, BaseTrainingBackend
from tuft.backends.sampling_backend import DPSamplingBackend
from tuft.backends.training_backend import HFTrainingBackend
from tuft.config import AppConfig, ModelConfig, SchedulingConfig
from tuft.exceptions import SequenceConflictException, UserMismatchException
from tuft.state import ServerState


class Replica:
    instances = []

    def __init__(self, config, worker_venv_path=None, *, instance_index=0):
        self.config = config
        self.index = instance_index
        self.samples = []
        self.sample_release: asyncio.Event | None = None
        self.adapters = {}
        self.closed = False
        self.shared_weights = None
        self.phases = []
        self.engine = SimpleNamespace(
            get_device_assignment=SimpleNamespace(remote=self.get_device_assignment),
            configure_shared_weights=SimpleNamespace(remote=self.configure_shared_weights),
            set_training_phase=SimpleNamespace(remote=self.set_training_phase),
        )
        self.instances.append(self)

    async def async_init(self):
        assert (self.shared_weights is not None) == self.config.colocate

    async def configure_shared_weights(self, descriptors):
        assert self.config.colocate and not self.shared_weights
        self.shared_weights = descriptors

    def configure_shared_placement(self, assignments):
        assert len(assignments) == 1 and self.config.colocate

    async def set_training_phase(self, training):
        assert self.shared_weights is not None
        self.phases.append(training)

    async def get_device_assignment(self):
        offset = 2 if self.config.model_name.endswith("-second") else 0
        return {"node": "test-node", "gpus": [self.index + offset]}

    async def sample(self, **kwargs):
        self.samples.append(kwargs)
        if self.sample_release is not None:
            await self.sample_release.wait()
        return types.SampleResponse(
            sequences=[
                types.SampledSequence(
                    stop_reason="length", _tokens_list=[self.index + 10], _logprobs_list=[-0.5]
                )
            ]
        )

    async def add_adapter(self, lora_id, adapter_path):
        self.adapters[lora_id] = adapter_path

    async def remove_adapter(self, lora_id):
        self.adapters.pop(lora_id)

    def get_openai_api_url(self):
        return f"http://replica-{self.index}"

    async def shutdown(self):
        self.closed = True


@pytest.fixture
def state(tmp_path, monkeypatch, request):
    import tuft.backends.sampling_backend as sampling_module

    Replica.instances = []
    monkeypatch.setattr(sampling_module, "VLLMSamplingBackend", Replica)
    updates = {}
    entered = asyncio.Event()
    release = asyncio.Event()
    release.set()

    async def create_adapter(lora_id, *_):
        updates[lora_id] = 0

    async def optim_step(params, lora_id, *_):
        entered.set()
        await release.wait()
        updates[lora_id] += 1
        return types.OptimStepResponse(metrics={"step:max": float(updates[lora_id])})

    async def noop(*args, **kwargs):
        pass

    async def forward(*args, **kwargs):
        return types.ForwardBackwardOutput(
            loss_fn_output_type="cross_entropy", loss_fn_outputs=[], metrics={}
        )

    async def save_state(*args, **kwargs):
        kwargs["checkpoint_record"].adapter_path.mkdir(parents=True, exist_ok=True)

    async def placement():
        return {"node": "test-node", "gpus": [0]}

    async def export():
        return {"frozen_weight": "test-descriptor"}

    backend = HFTrainingBackend.__new__(HFTrainingBackend)
    backend.scheduler = None
    backend.model = SimpleNamespace(
        create_adapter=SimpleNamespace(remote=create_adapter),
        optim_step=SimpleNamespace(remote=optim_step),
        forward=SimpleNamespace(remote=forward),
        save_state=SimpleNamespace(remote=save_state),
        load_state=SimpleNamespace(remote=noop),
        async_init=SimpleNamespace(remote=noop),
        get_device_assignment=SimpleNamespace(remote=placement),
        export_shared_base_weights=SimpleNamespace(remote=export),
    )
    monkeypatch.setattr(backend, "shutdown", noop)
    backends = {"Qwen/Qwen3-0.6B": backend}
    monkeypatch.setattr(
        BaseTrainingBackend, "create_backend", lambda c, **kw: backends[c.model_name]
    )
    monkeypatch.setattr(BaseSamplingBackend, "create_backend", lambda c, **kw: DPSamplingBackend(c))
    model = ModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        model_path=Path("Qwen/Qwen3-0.6B"),
        max_model_len=512,
        colocate=True,
        data_parallel_size=2,
        sampling_kv_cache_memory_bytes=16 * 1024**3,
        scheduling=SchedulingConfig(policy="loopweave", training_quiet_s=0),
    )
    models = [model]
    if getattr(request, "param", 1) == 2:
        import copy

        other = model.model_copy(deep=True, update={"model_name": model.model_name + "-second"})
        models.append(other)
        other_backend = copy.copy(backend)
        other_backend.model = SimpleNamespace(**vars(backend.model))

        async def other_placement():
            return {"node": "test-node", "gpus": [2]}

        other_backend.model.get_device_assignment = SimpleNamespace(remote=other_placement)
        backends[other.model_name] = other_backend
    server = ServerState(AppConfig(supported_models=models, checkpoint_dir=tmp_path))
    return server, updates, entered, release


async def create_run(state, user, model="Qwen/Qwen3-0.6B"):
    session = state.create_session(
        types.CreateSessionRequest(sdk_version="test", tags=[], user_metadata=None),
        User(user_id=user),
    )
    run = await state.create_model(session.session_id, model, types.LoraConfig(rank=8), user, None)
    return session.session_id, run.training_run_id


@pytest.mark.integration
async def test_cancelled_mutation_commits_sequence_and_cannot_be_retried(state):
    server, updates, entered, release = state
    await server.async_init()
    _, run_id = await create_run(server, "alice")
    _, other_id = await create_run(server, "bob")
    release.clear()
    task = asyncio.create_task(server.run_optim_step(run_id, "alice", types.AdamParams(), 1))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    await task
    assert updates == {run_id: 1, other_id: 0}
    assert server.training.training_runs[run_id].next_seq_id == 2
    with pytest.raises(SequenceConflictException):
        await server.run_optim_step(run_id, "alice", types.AdamParams(), 1)
    with pytest.raises(UserMismatchException):
        await server.run_optim_step(run_id, "bob", types.AdamParams(), 2)
    await server.run_optim_step(other_id, "bob", types.AdamParams(), 1)
    assert updates == {run_id: 1, other_id: 1}
    await server.shutdown()
    assert all(r.closed for r in Replica.instances)


@pytest.mark.integration
async def test_hybrid_factory_routes_http_to_fixed_and_samples_through_gate(state):
    server, _, entered, release = state
    await server.async_init()
    session, run_id = await create_run(server, "alice")
    sampler = await server.create_sampling_session(
        session, "Qwen/Qwen3-0.6B", None, "alice", session_seq_id=0
    )
    pool = server.sampling._base_backends["Qwen/Qwen3-0.6B"]
    assert pool.scheduler is server.training.training_backends["Qwen/Qwen3-0.6B"].scheduler
    assert [r.config.colocate for r in Replica.instances] == [True, False]
    assert pool.get_openai_api_urls() == ["http://replica-1"]
    assert {pool.get_next_openai_api_url() for _ in range(10)} == {"http://replica-1"}
    assert pool.get_openai_api_url() == "http://replica-1"
    release.clear()
    train = asyncio.create_task(server.run_optim_step(run_id, "alice", types.AdamParams(), 1))
    await asyncio.wait_for(entered.wait(), 2)
    request = types.SampleRequest(
        sampling_session_id=sampler,
        seq_id=1,
        prompt=types.ModelInput.from_ints([1, 2]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=2),
    )
    response = await server.run_sample(request, "alice")
    assert response.sequences[0].tokens == [11]
    assert not Replica.instances[0].samples
    release.set()
    await train
    await server.shutdown()


@pytest.mark.integration
async def test_hybrid_rejects_unshared_physical_placement(state):
    server, _, _, _ = state
    Replica.instances[0].index = 2
    with pytest.raises(RuntimeError, match="same GPU"):
        await server.async_init()
    await server.shutdown()


@pytest.mark.integration
async def test_hf_complete_update_provenance_and_wave_planning(state):
    """Real controllers, checkpoint metadata, HF wrapper and DP scheduler."""
    import time

    server, _, _, _ = state
    await server.async_init()
    session, run_id = await create_run(server, "alice")
    _, other_id = await create_run(server, "alice")
    pool = server.sampling._base_backends["Qwen/Qwen3-0.6B"]
    scheduler = pool.scheduler
    for _ in range(2):
        await server.run_forward(run_id, "alice", [], "cross_entropy", None, None, backward=True)
    assert not scheduler.profiler.arrivals[run_id]
    await server.run_optim_step(run_id, "alice", types.AdamParams(), None)
    assert scheduler.profiler.states[run_id].observations == 1
    assert other_id not in scheduler.profiler.states
    checkpoint = await server.save_checkpoint(run_id, "alice", "published", "sampler")
    sampler = await server.create_sampling_session(
        session, None, checkpoint.tinker_path, "alice", session_seq_id=0
    )
    record = server.sampling.sampling_sessions[sampler]
    assert record.training_run_id == run_id
    assert record.model_id == sampler
    assert scheduler.profiler.states[run_id].observations == 1

    # Seed histories rather than waiting for four natural RL loops.
    scheduler.profiler.forget(run_id)
    now = time.monotonic()
    for run, offset in [(run_id, 0), (other_id, 0.045)]:
        for j in range(4):
            scheduler.profiler.training_ready(run, now - 3.8 + j + offset)
            scheduler.profiler.executed(run, 0.01, commit=True)
    request = types.SampleRequest(
        sampling_session_id=sampler,
        seq_id=1,
        prompt=types.ModelInput.from_ints([1, 2]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=2),
    )
    task = asyncio.create_task(server.run_sample(request, "alice"))
    async with asyncio.timeout(2):
        while not scheduler.snapshot().get("loop_held_requests"):
            await asyncio.sleep(0)
    assert not any(r.samples for r in Replica.instances)
    await task
    assert [sample["lora_id"] for r in Replica.instances for sample in r.samples] == [sampler]
    assert scheduler.profiler.states[run_id].sampling_started is not None
    assert scheduler.profiler.states[other_id].sampling_started is None
    from tuft.sampling_controller import SamplingSessionRecord

    assert (
        SamplingSessionRecord.model_validate_json(record.model_dump_json()).training_run_id
        == run_id
    )
    old = record.model_dump(exclude={"training_run_id"})
    assert SamplingSessionRecord.model_validate(old).training_run_id is None
    await server.shutdown()


@pytest.mark.integration
async def test_foreign_public_snapshot_does_not_join_owners_training_loop(state):
    server, _, _, _ = state
    await server.async_init()
    _, run_id = await create_run(server, "alice")
    session, _ = await create_run(server, "bob")
    checkpoint = await server.save_checkpoint(run_id, "alice", "public", "sampler")
    checkpoint.set_visibility(True)
    sampler = await server.create_sampling_session(
        session, None, checkpoint.tinker_path, "bob", session_seq_id=0
    )
    assert server.sampling.sampling_sessions[sampler].training_run_id is None
    await server.shutdown()


@pytest.mark.integration
@pytest.mark.parametrize("state", [2], indirect=True)
async def test_many_adapters_on_two_bases_keep_publications_and_training_isolated(
    state, record_property
):
    server, _, training_entered, training_release = state
    await server.async_init()
    pools = server.sampling._base_backends
    assert len(pools) == 2
    samplers = {}
    runs = []
    sampling_release = asyncio.Event()
    for model, pool in pools.items():
        pool.scheduler.config.cost_aware_borrowing = False
        pool.scheduler.config.max_inflight_per_replica = 16
        for replica in pool._instances:
            replica.sample_release = sampling_release
        for i in range(8):
            user = f"user-{i}"
            session, run = await create_run(server, user, model)
            runs.append((run, user))
            checkpoint = await server.save_checkpoint(run, user, "published-v1", "sampler")
            sampler = await server.create_sampling_session(
                session, None, checkpoint.tinker_path, user, session_seq_id=0
            )
            samplers[sampler] = (model, user, run)
    schedulers = [pool.scheduler for pool in pools.values()]
    assert schedulers[0] is not schedulers[1]
    training_entered.clear()
    training_release.clear()
    run, user = runs[0]
    training = asyncio.create_task(server.run_optim_step(run, user, types.AdamParams(), None))
    await training_entered.wait()
    assert schedulers[0].snapshot()["training_pending"] == 1
    assert schedulers[1].snapshot()["training_pending"] == 0
    tasks = []
    for repeat in range(4):
        for sampler, (_, owner, _) in samplers.items():
            tasks.append(
                asyncio.create_task(
                    server.run_sample(
                        types.SampleRequest(
                            sampling_session_id=sampler,
                            seq_id=repeat + 1,
                            prompt=types.ModelInput.from_ints([1, 2]),
                            num_samples=1,
                            sampling_params=types.SamplingParams(max_tokens=2),
                        ),
                        owner,
                    )
                )
            )
    async with asyncio.timeout(20):
        # Wait for all controller persistence/sequence work to reach admission;
        # thread timing must not determine whether a complete wave can group.
        while not all(
            s.snapshot().get("admitted", 0) + s.snapshot()["queued"] == 32 for s in schedulers
        ):
            await asyncio.sleep(0.001)
        sampling_release.set()
        await asyncio.gather(*tasks)
    first, second = pools.values()
    assert not first._instances[0].samples  # This base's training lane is occupied.
    assert second._instances[0].samples  # The other base can still borrow.
    for model, pool in pools.items():
        seen = [s["lora_id"] for replica in pool._instances for s in replica.samples]
        expected = {s for s, (base, _, _) in samplers.items() if base == model}
        assert set(seen) == expected
        assert all(seen.count(s) == 4 for s in expected)
        assert pool.scheduler.snapshot()["active"] == [0, 0]
    assert sum(s.snapshot()["adapter_grouped_requests"] for s in schedulers) > 0
    record_property("base_models", len(pools))
    record_property("published_adapters", len(samplers))
    record_property("completed_samples", len(tasks))
    training_release.set()
    await training
    await server.shutdown()
    assert all(replica.closed for replica in Replica.instances)


def test_host_adapter_cache_capacity_is_independent_of_gpu_slots():
    from pydantic import ValidationError

    from tuft.backends.sampling_backend import VLLMSamplingBackend

    model = ModelConfig(
        model_name="qwen",
        model_path=Path("qwen"),
        max_model_len=512,
        max_loras=4,
        max_cpu_loras=128,
    )
    backend = VLLMSamplingBackend.__new__(VLLMSamplingBackend)
    engine = backend._build_engine_config(model, tensor_parallel_size=1, gpu_memory_utilization=0.5)
    assert engine.max_loras == 4
    assert engine.max_cpu_loras == 128
    with pytest.raises(ValidationError, match="max_cpu_loras"):
        ModelConfig(
            model_name="qwen",
            model_path=Path("qwen"),
            max_model_len=512,
            max_loras=4,
            max_cpu_loras=2,
        )
