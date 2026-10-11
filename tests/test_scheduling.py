"""Adversarial admission tests using explicit barriers instead of GPU timing."""

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from tuft.config import ModelConfig, SchedulingConfig
from tuft.scheduling import LoopProfiler, ResourceScheduler, SchedulingOverloadedError


async def spin_until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


def config(**kwargs):
    kwargs.setdefault("sampling_alignment_s", 0)
    return SchedulingConfig(policy="balanced", training_quiet_s=0, **kwargs)


def test_loop_profiler_requires_consistent_recent_observations_and_bounds_memory():
    profiler = LoopProfiler(max_tenants=2)
    for t in [0, 10, 20]:
        profiler.observe("regular", t)
    assert profiler.horizon(21) is None
    profiler.observe("regular", 30)
    assert profiler.horizon(31) == 40
    assert profiler.horizon(41) is None
    for t in [0, 10, 50, 60]:
        profiler.observe("irregular", t)
    assert profiler.horizon(61) is None
    profiler.observe("new", 70)
    assert len(profiler.arrivals) == 2
    assert "regular" not in profiler.arrivals


async def test_token_load_balances_and_releases_on_failure():
    scheduler = ResourceScheduler(2, config(locality_slack=0))
    release = asyncio.Event()
    selected = []

    async def operation(index):
        selected.append(index)
        await release.wait()
        if len(selected) == 3:
            raise ValueError("backend failure")

    tasks = []
    for tenant, work in [("large", 1000), ("small", 10), ("small", 10)]:
        tasks.append(asyncio.create_task(scheduler.sample(tenant, tenant, work, True, operation)))
        await spin_until(lambda: len(selected) == len(tasks))
    assert selected == [0, 1, 1]
    release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(r, ValueError) for r in results)
    assert scheduler.snapshot()["active"] == [0, 0]
    assert scheduler.snapshot()["reserved_tokens"] == [0, 0]
    await scheduler.close()


async def test_flex_drains_before_training_fixed_continues():
    scheduler = ResourceScheduler(2, config(max_inflight_per_replica=1), flex_index=0)
    release_flex = asyncio.Event()
    release_train = asyncio.Event()
    events = []

    async def sample(index):
        events.append(f"sample-{index}")
        if index == 0:
            await release_flex.wait()

    async def train():
        events.append("train")
        assert scheduler.snapshot()["active"][0] == 0
        await release_train.wait()

    sampling = asyncio.create_task(scheduler.sample("a", "a", 10, True, sample))
    await spin_until(lambda: events == ["sample-0"])
    training = asyncio.create_task(scheduler.training(train))
    await spin_until(lambda: scheduler.snapshot()["training_pending"] == 1)
    await scheduler.sample("b", "b", 10, True, sample)
    assert events == ["sample-0", "sample-1"]
    release_flex.set()
    await spin_until(lambda: "train" in events)
    await scheduler.sample("c", "c", 10, True, sample)
    assert events[-1] == "sample-1"
    release_train.set()
    await asyncio.gather(sampling, training)
    await scheduler.close()


async def test_cancelled_client_does_not_release_running_gpu_work():
    scheduler = ResourceScheduler(2, config(), flex_index=0)
    release = asyncio.Event()
    started = asyncio.Event()
    trained = asyncio.Event()

    async def sample(index):
        assert index == 0
        started.set()
        await release.wait()

    task = asyncio.create_task(scheduler.sample("a", "a", 10, True, sample))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert scheduler.snapshot()["active"][0] == 1

    async def train():
        trained.set()

    training = asyncio.create_task(scheduler.training(train))
    await spin_until(lambda: scheduler.snapshot()["training_pending"] == 1)
    assert not trained.is_set()
    release.set()
    await training
    assert trained.is_set()
    await scheduler.close()


async def test_cancelled_training_still_owns_lane_until_backend_completes():
    scheduler = ResourceScheduler(2, config(), flex_index=0)
    release = asyncio.Event()
    started = asyncio.Event()

    async def train():
        started.set()
        await release.wait()

    task = asyncio.create_task(scheduler.training(train))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert scheduler.snapshot()["training_active"]
    indices = []

    async def sample(index):
        indices.append(index)

    await scheduler.sample("a", "a", 10, True, sample)
    assert indices == [1]
    release.set()
    await task
    await scheduler.close()
    assert not scheduler.snapshot()["training_active"]


async def test_queue_capacity_timeout_and_cancel_cleanup():
    scheduler = ResourceScheduler(
        1, config(max_inflight_per_replica=1, max_queued_requests=1, queue_timeout_s=0.03)
    )
    release = asyncio.Event()
    started = asyncio.Event()

    async def sample(index):
        started.set()
        await release.wait()

    active = asyncio.create_task(scheduler.sample("a", "a", 1, True, sample))
    await started.wait()
    queued = asyncio.create_task(scheduler.sample("b", "b", 1, True, sample))
    await spin_until(lambda: scheduler.snapshot()["queued"] == 1)
    with pytest.raises(SchedulingOverloadedError, match="full") as rejected:
        await scheduler.sample("c", "c", 1, True, sample)
    assert rejected.value.status_code == 429
    with pytest.raises(SchedulingOverloadedError, match="expired"):
        await queued
    assert scheduler.snapshot()["queued"] == 0
    assert scheduler.snapshot()["active"] == [1]
    queued = asyncio.create_task(scheduler.sample("b", "b", 1, True, sample))
    await spin_until(lambda: scheduler.snapshot()["queued"] == 1)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert scheduler.snapshot()["queued"] == 0
    release.set()
    await active
    await scheduler.close()


async def test_tenants_take_turns_and_preserve_fifo():
    scheduler = ResourceScheduler(1, config(max_inflight_per_replica=1))
    release = asyncio.Event()
    events = []

    async def operation(label, index):
        events.append(label)
        if label == "active":
            await release.wait()

    tasks = []
    for label in ["active", "a1", "a2", "a3", "b1", "b2"]:
        tasks.append(
            asyncio.create_task(
                scheduler.sample(
                    label[0], label[0], 10, True, lambda index, label=label: operation(label, index)
                )
            )
        )
    await spin_until(lambda: scheduler.snapshot()["queued"] == 5)
    release.set()
    await asyncio.gather(*tasks)
    assert events == ["active", "a1", "b1", "a2", "b2", "a3"]
    await scheduler.close()


async def test_long_work_never_uses_flex_and_close_rejects_queue():
    scheduler = ResourceScheduler(2, config(max_inflight_per_replica=1), flex_index=0)
    release = asyncio.Event()
    indices = []

    async def operation(index):
        indices.append(index)
        await release.wait()

    active = asyncio.create_task(scheduler.sample("a", "a", 1000, False, operation))
    await spin_until(lambda: indices == [1])
    queued = asyncio.create_task(scheduler.sample("b", "b", 1000, False, operation))
    await spin_until(lambda: scheduler.snapshot()["queued"] == 1)
    closing = asyncio.create_task(scheduler.close())
    with pytest.raises(RuntimeError, match="closed"):
        await queued
    assert not closing.done()
    release.set()
    await asyncio.gather(active, closing)
    with pytest.raises(RuntimeError, match="closed"):
        await scheduler.sample("c", None, 1, True, operation)
    assert indices == [1]


async def test_backend_timeout_is_not_reported_as_admission_timeout():
    scheduler = ResourceScheduler(1, config())

    async def fail(index):
        raise TimeoutError("engine timeout")

    with pytest.raises(TimeoutError, match="engine timeout"):
        await scheduler.sample("a", None, 1, True, fail)
    assert scheduler.snapshot().get("queue_timeouts", 0) == 0
    await scheduler.close()


async def test_close_cancellation_does_not_cancel_admitted_work():
    scheduler = ResourceScheduler(1, config())
    release = asyncio.Event()
    started = asyncio.Event()

    async def operation(index):
        started.set()
        await release.wait()
        return 42

    sampling = asyncio.create_task(scheduler.sample("a", None, 1, True, operation))
    await started.wait()
    closing = asyncio.create_task(scheduler.close())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    assert scheduler.snapshot()["active"] == [1]
    release.set()
    assert await sampling == 42
    await closing


async def test_predicted_training_horizon_keeps_long_estimate_on_fixed():
    import time

    scheduler = ResourceScheduler(2, config(), flex_index=0)
    now = time.monotonic()
    for offset in [-3.1, -2.1, -1.1, -0.1]:
        scheduler.profiler.observe("regular", now + offset)
    scheduler._seconds_per_token = 0.1
    indices = []

    async def operation(index):
        indices.append(index)

    await scheduler.sample("a", None, 10, True, operation)
    assert indices == [1]
    assert scheduler.snapshot()["window_rejections"] > 0
    await scheduler.close()


async def test_alignment_keeps_next_rollout_available_for_reclaimed_gpu():
    scheduler = ResourceScheduler(2, config(sampling_alignment_s=1), flex_index=0)
    training_started = asyncio.Event()
    release = asyncio.Event()
    indices = []

    async def train():
        training_started.set()
        await release.wait()

    async def sample(index):
        indices.append(index)

    training = asyncio.create_task(scheduler.training(train))
    await training_started.wait()
    sampling = asyncio.create_task(scheduler.sample("a", None, 1, True, sample))
    await spin_until(lambda: scheduler.snapshot()["queued"] == 1)
    assert not indices
    release.set()
    await asyncio.gather(training, sampling)
    assert indices == [0]
    await scheduler.close()


async def test_alignment_budget_expires_to_fixed_during_long_training():
    scheduler = ResourceScheduler(2, config(sampling_alignment_s=0.01), flex_index=0)
    training_started = asyncio.Event()
    release = asyncio.Event()

    async def train():
        training_started.set()
        await release.wait()

    async def sample(index):
        return index

    training = asyncio.create_task(scheduler.training(train))
    await training_started.wait()
    try:
        assert await asyncio.wait_for(scheduler.sample("a", None, 1, True, sample), 2) == 1
        assert scheduler.snapshot()["training_active"]
    finally:
        release.set()
        await training
        await scheduler.close()


@pytest.mark.parametrize(
    "override",
    [
        {"colocate": False},
        {"training_backend": "fsdp"},
        {"data_parallel_size": 1},
        {"tensor_parallel_size": 2},
        {"sampling_memory_fraction": 1},
        {"capabilities": ["sampling"]},
    ],
)
def test_reject_invalid_hybrid_layout(override):
    values = dict(
        model_name="qwen",
        model_path=Path("qwen"),
        max_model_len=512,
        colocate=True,
        data_parallel_size=2,
        scheduling={"policy": "loopweave"},
    )
    values.update(override)
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(values)
