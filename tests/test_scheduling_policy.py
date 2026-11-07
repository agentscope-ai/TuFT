"""CPU regressions for bounded mutation ordering."""

import asyncio

import pytest

from tuft.config import SamplingConfig
from tuft.mutation_queue import MutationQueue
from tuft.scheduling import ResourceScheduler


async def spin_until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


async def test_priority_is_bounded_preserves_run_fifo_and_global_barriers():
    queue = MutationQueue(2)
    seen = []

    async def enter(name, tenant, priority):
        async with queue.reserve(tenant, priority):
            seen.append(name)

    async with queue.reserve("owner", False):
        tasks = []
        for name, tenant, priority in [
            ("backward-a", "a", False),
            ("optimizer-a", "a", True),
            ("optimizer-b", "b", True),
            ("checkpoint-c", "c", True),
            ("optimizer-d", "d", True),
            ("maintenance", None, False),
            ("optimizer-e", "e", True),
        ]:
            tasks.append(asyncio.create_task(enter(name, tenant, priority)))
            await asyncio.sleep(0)
    await asyncio.gather(*tasks)
    assert seen == [
        "optimizer-b",
        "checkpoint-c",
        "backward-a",
        "optimizer-a",
        "optimizer-d",
        "maintenance",
        "optimizer-e",
    ]


async def test_cancelled_waiter_and_failed_owner_do_not_strand_mutation_queue():
    queue = MutationQueue(2)
    seen = []

    async def enter(name, fail=False):
        async with queue.reserve(name, True):
            seen.append(name)
            if fail:
                raise ValueError("failed operation")

    async with queue.reserve("owner", False):
        cancelled = asyncio.create_task(enter("cancelled"))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        failed = asyncio.create_task(enter("failed", True))
        next_task = asyncio.create_task(enter("next"))
        await asyncio.sleep(0)
    with pytest.raises(ValueError):
        await failed
    await next_task
    assert seen == ["failed", "next"]


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("shared_weight", ["optimizer", "backward"]),
        ("round_robin", ["backward", "optimizer"]),
    ],
)
async def test_scheduler_priority_is_opt_in_and_cancellation_retains_committed_operation(
    policy, expected
):
    config = SamplingConfig.model_validate({"policy": policy})
    scheduler = ResourceScheduler(2, config, flex_index=0 if config.borrowing.enabled else None)
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def owner():
        entered.set()
        await release.wait()

    async def operation(name):
        seen.append(name)

    active = asyncio.create_task(scheduler.training(owner, tenant="owner"))
    await entered.wait()
    backward = asyncio.create_task(scheduler.training(lambda: operation("backward"), tenant="a"))
    await spin_until(lambda: scheduler._training_pending == 2)
    optimizer = asyncio.create_task(
        scheduler.training(lambda: operation("optimizer"), tenant="b", kind="optimizer")
    )
    await spin_until(lambda: scheduler._training_pending == 3)
    optimizer.cancel()
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(active, backward, optimizer)
    assert seen == expected
    assert scheduler._training_pending == 0
    await scheduler.close()
