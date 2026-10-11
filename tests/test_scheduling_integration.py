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
        self.engine = SimpleNamespace(
            get_device_assignment=SimpleNamespace(remote=self.get_device_assignment)
        )
        self.instances.append(self)

    async def async_init(self):
        pass

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

    async def placement():
        return {"node": "test-node", "gpus": [0]}

    backend = HFTrainingBackend.__new__(HFTrainingBackend)
    backend.scheduler = None
    backend.model = SimpleNamespace(
        create_adapter=SimpleNamespace(remote=create_adapter),
        optim_step=SimpleNamespace(remote=optim_step),
        async_init=SimpleNamespace(remote=noop),
        get_device_assignment=SimpleNamespace(remote=placement),
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
