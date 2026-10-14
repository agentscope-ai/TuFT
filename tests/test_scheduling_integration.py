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

    async def set_training_phase(self, training):
        assert self.shared_weights is not None
        self.phases.append(training)

    async def get_device_assignment(self):
        return {"node": "test-node", "gpus": [self.index]}

    async def sample(self, **kwargs):
        self.samples.append(kwargs)
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
def state(tmp_path, monkeypatch):
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
    monkeypatch.setattr(BaseTrainingBackend, "create_backend", lambda *a, **kw: backend)
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
    server = ServerState(AppConfig(supported_models=[model], checkpoint_dir=tmp_path))
    return server, updates, entered, release


async def create_run(state, user):
    session = state.create_session(
        types.CreateSessionRequest(sdk_version="test", tags=[], user_metadata=None),
        User(user_id=user),
    )
    run = await state.create_model(
        session.session_id, "Qwen/Qwen3-0.6B", types.LoraConfig(rank=8), user, None
    )
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
