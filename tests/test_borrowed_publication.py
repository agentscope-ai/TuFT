"""Publication readiness and borrowed registration under the real scheduler."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from tinker import types

from tuft.backends.fsdp_training_backend import FSDPTrainingBackend
from tuft.backends.sampling_backend import DPSamplingBackend
from tuft.backends.training_backend import HFTrainingBackend
from tuft.config import ModelConfig, SamplingConfig
from tuft.scheduling import ResourceScheduler


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


class Replica:
    def __init__(self, config, worker_venv_path=None, instance_index=0):
        self.index = instance_index
        self.adapters = {}
        self.loads = []
        self.samples = []
        self.load_release: asyncio.Event | None = None
        self.sample_release: asyncio.Event | None = None
        self.load_error = False

    async def add_adapter(self, adapter, path):
        self.loads.append(adapter)
        if self.load_release is not None:
            await self.load_release.wait()
        if self.load_error:
            raise RuntimeError("registration failed")
        self.adapters[adapter] = path

    async def remove_adapter(self, adapter):
        self.adapters.pop(adapter, None)

    async def sample(self, **kwargs):
        adapter = kwargs.get("lora_id")
        assert adapter is None or adapter in self.adapters
        self.samples.append(adapter)
        if self.sample_release is not None:
            await self.sample_release.wait()
        return types.SampleResponse(sequences=[])

    async def shutdown(self):
        pass


@pytest.fixture
async def pools(monkeypatch):
    monkeypatch.setattr("tuft.backends.sampling_backend.VLLMSamplingBackend", Replica)
    created: list[Any] = []

    def make(size=2, borrowing=True) -> Any:
        config = ModelConfig(
            model_name="test",
            model_path=Path("test"),
            max_model_len=512,
            data_parallel_size=size,
            colocate=borrowing,
            sampling_kv_cache_memory_bytes=1024,
            sampling=SamplingConfig.model_validate(
                {
                    "max_inflight_per_replica": 1,
                    "borrowing": {"enabled": borrowing, "max_inflight": 2},
                }
            ),
        )
        pool = DPSamplingBackend(config)
        created.append(pool)
        return pool

    yield make
    for pool in created:
        for replica in pool._instances:
            if replica.load_release is not None:
                replica.load_release.set()
            if replica.sample_release is not None:
                replica.sample_release.set()
        await pool.shutdown()


def sample(pool, adapter="a"):
    return pool.sample(
        types.ModelInput.from_ints([1]), 1, types.SamplingParams(max_tokens=1), lora_id=adapter
    )


async def test_publication_and_dedicated_sampling_do_not_wait_for_training(pools, tmp_path):
    pool = pools()
    entered, release = asyncio.Event(), asyncio.Event()

    async def train():
        entered.set()
        await release.wait()

    task = asyncio.create_task(pool.scheduler.training(train, tenant="other"))
    await entered.wait()
    try:
        await asyncio.wait_for(pool.add_adapter("a", tmp_path), 2)
        await asyncio.wait_for(sample(pool), 2)
        assert pool._instances[0].loads == []
        assert pool._instances[1].samples == ["a"]
        assert not task.done()
    finally:
        release.set()
        await task


async def test_first_borrow_loads_once_and_training_waits_through_cancelled_request(
    pools, tmp_path
):
    pool = pools()
    await pool.add_adapter("a", tmp_path)
    borrowed, dedicated = pool._instances
    dedicated.sample_release = asyncio.Event()
    borrowed.load_release = asyncio.Event()
    borrowed.sample_release = asyncio.Event()
    fixed = asyncio.create_task(sample(pool))
    await until(lambda: dedicated.samples)
    requests = [asyncio.create_task(sample(pool)) for _ in range(2)]
    await until(lambda: borrowed.loads)
    trained = asyncio.Event()

    async def train():
        assert pool.scheduler.snapshot()["active"][0] == 0
        trained.set()

    training = asyncio.create_task(pool.scheduler.training(train, tenant="other"))
    requests[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await requests[0]
    assert not trained.is_set()
    borrowed.load_release.set()
    await until(lambda: len(borrowed.samples) == 2)
    assert borrowed.loads == ["a"] and not trained.is_set()
    borrowed.sample_release.set()
    await asyncio.wait_for(requests[1], 2)
    await asyncio.wait_for(training, 2)
    dedicated.sample_release.set()
    await fixed


async def test_borrowed_load_failure_retries_dedicated_without_holding_trainer(pools, tmp_path):
    pool = pools()
    await pool.add_adapter("a", tmp_path)
    borrowed, dedicated = pool._instances
    dedicated.sample_release = asyncio.Event()
    borrowed.load_error = True
    fixed = asyncio.create_task(sample(pool))
    await until(lambda: dedicated.samples)
    retry = asyncio.create_task(sample(pool))
    await until(lambda: "a" in pool._borrowed_failed)
    trained = asyncio.Event()

    async def train():
        trained.set()

    await asyncio.wait_for(pool.scheduler.training(train, tenant="other"), 2)
    assert trained.is_set() and not retry.done()
    dedicated.sample_release.set()
    await asyncio.gather(fixed, retry)
    await sample(pool)
    assert borrowed.loads == ["a"] and len(dedicated.samples) == 3


@pytest.mark.parametrize("borrowing", [False, True])
async def test_partial_dedicated_failure_rolls_back_without_touching_borrowed(
    pools, tmp_path, borrowing
):
    pool = pools(size=3 if borrowing else 2, borrowing=borrowing)
    pool._instances[-1].load_error = True
    with pytest.raises(RuntimeError, match="registration failed"):
        await pool.add_adapter("a", tmp_path)
    assert all(not r.adapters for r in pool._instances)
    assert "a" not in pool._adapter_paths
    if borrowing:
        assert not pool._instances[0].loads


async def test_removal_drains_pending_load_and_does_not_resurrect_adapter(pools, tmp_path):
    pool = pools()
    await pool.add_adapter("a", tmp_path)
    borrowed, dedicated = pool._instances
    dedicated.sample_release = asyncio.Event()
    borrowed.load_release = asyncio.Event()
    fixed = asyncio.create_task(sample(pool))
    await until(lambda: dedicated.samples)
    request = asyncio.create_task(sample(pool))
    await until(lambda: borrowed.loads)
    removal = asyncio.create_task(pool.remove_adapter("a"))
    await until(lambda: pool.scheduler.snapshot()["training_pending"])
    assert not removal.done()
    borrowed.load_release.set()
    await request
    await removal
    assert all("a" not in r.adapters for r in pool._instances)
    assert "a" not in pool._borrowed_ready and "a" not in pool._adapter_paths
    dedicated.sample_release.set()
    await fixed


@pytest.mark.parametrize("backend_kind", ["hf", "fsdp"])
async def test_checkpoint_identity_allows_priority_without_breaking_group_reservation(backend_kind):
    scheduler = ResourceScheduler(
        2, SamplingConfig.model_validate({"borrowing": {"enabled": True}}), flex_index=0
    )
    seen = []
    checkpoint = MagicMock()
    if backend_kind == "hf":
        backend: Any = HFTrainingBackend.__new__(HFTrainingBackend)

        async def save(**kwargs):
            assert kwargs["lora_id"] == "ready"
            assert scheduler.snapshot()["training_active"]
            seen.append("checkpoint")

        backend.model = SimpleNamespace(save_state=SimpleNamespace(remote=save))
    else:
        backend = FSDPTrainingBackend.__new__(FSDPTrainingBackend)
        backend._worker = None
        backend._lora_id_to_adapter_name = {"ready": "slot"}
        rank_calls = []

        def save_rank(rank, *args):
            assert scheduler.snapshot()["training_active"]
            rank_calls.append((rank, args))
            return rank

        backend._actors = [
            SimpleNamespace(
                save_checkpoint=SimpleNamespace(
                    remote=lambda *args, rank=rank: save_rank(rank, *args)
                )
            )
            for rank in range(2)
        ]

        async def workers(refs):
            assert refs == [0, 1] and len(rank_calls) == 2
            seen.append("checkpoint")

        backend._await_workers = workers
    backend.scheduler = scheduler
    entered, release = asyncio.Event(), asyncio.Event()

    async def owner():
        entered.set()
        await release.wait()

    async def backward():
        seen.append("backward")

    current = asyncio.create_task(scheduler.training(owner, tenant="owner"))
    await entered.wait()
    waiting = asyncio.create_task(scheduler.training(backward, tenant="other"))
    await until(lambda: scheduler.snapshot()["training_pending"] == 2)
    saved = asyncio.create_task(backend.save_state("ready", checkpoint, False))
    await until(lambda: scheduler.snapshot()["training_pending"] == 3)
    release.set()
    await asyncio.gather(current, waiting, saved)
    assert seen == ["checkpoint", "backward"]
    await scheduler.close()
