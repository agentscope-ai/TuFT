"""Reactive overflow, sustained borrowing, and training-drain protection."""

import asyncio
import json

import pytest

from tuft.config import SamplingConfig
from tuft.scheduling import ResourceScheduler


async def wait_for(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


def scheduler():
    return ResourceScheduler(
        2,
        SamplingConfig.model_validate(
            {
                "borrowing": {"enabled": True},
                "max_inflight_per_replica": 1,
                "scheduling_trace_capacity": 64,
            }
        ),
        flex_index=0,
    )


async def test_only_overflow_borrows_and_completion_refills_without_calibration():
    s = scheduler()
    fixed_release = asyncio.Event()
    borrowed_release = asyncio.Event()
    seen = []

    async def operation(index):
        seen.append(index)
        await (fixed_release if index == 1 else borrowed_release).wait()

    tasks = []
    try:
        tasks.append(asyncio.create_task(s.sample("first", None, 8, True, operation)))
        await wait_for(lambda: seen == [1])
        tasks.append(asyncio.create_task(s.sample("second", None, 8, True, operation)))
        await wait_for(lambda: seen == [1, 0])
        tasks.append(asyncio.create_task(s.sample("third", None, 8, True, operation)))
        await wait_for(lambda: s.snapshot()["queued"] == 1)
        assert s.snapshot()["active"] == [1, 1]
        borrowed_release.set()
        await wait_for(lambda: len(seen) == 3)
        assert seen == [1, 0, 0]
        assert not tasks[0].done()
        trace = s.snapshot()["decision_trace"]
        assert len([e for e in trace if e["event"] == "admit"]) == 3
        assert not any("estimate" in key for e in trace for key in e)
    finally:
        fixed_release.set()
        borrowed_release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await s.close()


async def test_training_closes_borrowing_and_drains_disconnected_request():
    s = scheduler()
    fixed_release, borrowed_release, training_release = (asyncio.Event() for _ in range(3))
    seen, phases = [], []
    trained = asyncio.Event()

    async def phase(training):
        phases.append(training)
        if training:
            assert s.snapshot()["active"][0] == 0

    s.set_phase_transition(phase)

    async def sample(index):
        seen.append(index)
        await (fixed_release if index == 1 else borrowed_release).wait()

    async def train():
        trained.set()
        await training_release.wait()

    tasks = []
    try:
        fixed = asyncio.create_task(s.sample("fixed", None, 8, True, sample))
        tasks.append(fixed)
        await wait_for(lambda: seen == [1])
        borrowed = asyncio.create_task(s.sample("borrowed", None, 8, True, sample))
        tasks.append(borrowed)
        await wait_for(lambda: seen == [1, 0])
        borrowed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await borrowed
        training = asyncio.create_task(s.training(train, reclaim_sampler_memory=True))
        tasks.append(training)
        await wait_for(lambda: s.snapshot()["training_pending"] == 1)
        queued = asyncio.create_task(s.sample("later", None, 8, True, sample))
        tasks.append(queued)
        await wait_for(lambda: s.snapshot()["queued"] == 1)
        assert not trained.is_set()
        fixed_release.set()
        await queued
        assert seen == [1, 0, 1]
        assert not trained.is_set()
        borrowed_release.set()
        await asyncio.wait_for(trained.wait(), 2)
        assert phases == [False, True]
        await s.sample("during-training", None, 8, True, sample)
        assert seen[-1] == 1
    finally:
        fixed_release.set()
        borrowed_release.set()
        training_release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await s.close()
    snap = s.snapshot()
    assert snap["active"] == [0, 0] and snap["training_pending"] == 0
    assert snap["maintenance_drain_events"] == 1


async def test_rejection_trace_records_only_actual_overflow_without_private_ids():
    s = scheduler()
    release, entered = asyncio.Event(), asyncio.Event()

    async def operation(index):
        assert index == 1
        entered.set()
        await release.wait()

    fixed = asyncio.create_task(s.sample("private-tenant", "private-adapter", 8, True, operation))
    await asyncio.wait_for(entered.wait(), 2)
    assert not s.snapshot()["borrow_rejection_trace"]
    oversized = asyncio.create_task(s.sample("private-large", None, 10000, False, operation))
    try:
        await wait_for(lambda: bool(s.snapshot()["borrow_rejection_trace"]))
        snapshot = s.snapshot()
        assert snapshot["borrow_rejection_trace"][0]["reason"] == "ineligible"
        assert "private" not in json.dumps(snapshot)
    finally:
        release.set()
        await asyncio.gather(fixed, oversized)
        await s.close()


async def test_early_pump_does_not_lose_post_training_grace_period_wake():
    s = scheduler()
    release, borrowed = asyncio.Event(), asyncio.Event()

    async def noop():
        pass

    await s.training(noop, reclaim_sampler_memory=True)

    async def sample(index):
        if index == 0:
            borrowed.set()
        await release.wait()

    jobs = [asyncio.create_task(s.sample(str(i), None, 8, True, sample)) for i in range(2)]
    try:
        await wait_for(lambda: s.snapshot()["active"][1] == 1)
        # Simulate another completion scheduling an earlier admission pass.
        s._schedule_wake(0)
        await asyncio.wait_for(borrowed.wait(), 2)
        assert s.snapshot()["active"] == [1, 1]
    finally:
        release.set()
        await asyncio.gather(*jobs)
        await s.close()
