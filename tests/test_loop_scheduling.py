"""CPU policy tests: no engines, GPU allocation, or performance claims."""

import asyncio
from types import SimpleNamespace

from tuft.config import SamplingConfig
from tuft.scheduling import ResourceScheduler


async def test_exclusive_trace_separates_drain_queue_phase_service(monkeypatch):
    import tuft.scheduling as module

    now = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    s = ResourceScheduler(
        2,
        SamplingConfig.model_validate(
            {"borrowing": {"enabled": True}, "scheduling_trace_capacity": 2}
        ),
        flex_index=0,
    )
    s._flex_drained.clear()

    async def phase(training):
        now[0] += 3

    async def operation():
        now[0] += 5

    s.set_phase_transition(phase)
    first = asyncio.create_task(
        s.training(operation, tenant="private", reclaim_sampler_memory=True)
    )
    second = asyncio.create_task(
        s.training(operation, kind="checkpoint", reclaim_sampler_memory=True)
    )
    for _ in range(5):
        await asyncio.sleep(0)
    now[0] += 2
    s._flex_drained.set()
    await asyncio.gather(first, second)
    snap = s.snapshot()
    assert snap["training_drain_wait_s"] == 2
    assert snap["training_phase_s"] == 3
    assert snap["training_execution_s"] == 5
    assert snap["maintenance_queue_wait_s"] == 10
    assert snap["maintenance_drain_wait_s"] == 0
    assert len(snap["decision_trace"]) == 2
    assert "private" not in str(snap)
    await s.close()


async def test_compound_mutation_drains_on_disconnect_and_shutdown():
    s = ResourceScheduler(1, SamplingConfig.model_validate({"borrowing": {"enabled": False}}))
    entered, release = asyncio.Event(), asyncio.Event()

    async def operation():
        entered.set()
        await release.wait()
        return "committed"

    mutation = asyncio.create_task(s.transaction(operation))
    await entered.wait()
    mutation.cancel()
    closing = asyncio.create_task(s.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    assert await mutation == "committed"
    await closing
