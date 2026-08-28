from __future__ import annotations

import pytest

from tuft.schedulers.switch_scheduler import SwitchPolicy, SwitchScheduler, SwitchSchedulerConfig


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.asyncio
async def test_switch_scheduler_drains_training_before_sampling() -> None:
    clock = FakeClock()
    order: list[str] = []
    scheduler = SwitchScheduler(
        SwitchSchedulerConfig(sampling_min_window_s=0.0, idle_sleep_s=0.0),
        switch_to_sampling=lambda: order.append("switch_to_sampling"),
        switch_to_training=lambda: order.append("switch_to_training"),
        clock=clock,
    )
    scheduler.enqueue_training(lambda: order.append("train-1"))
    scheduler.enqueue_training(lambda: order.append("train-2"))
    scheduler.enqueue_sampling(lambda: order.append("sample-1"))

    stats = await scheduler.run_until_idle()

    assert order == ["train-1", "train-2", "switch_to_sampling", "sample-1"]
    assert stats.training_jobs_completed == 2
    assert stats.sampling_jobs_completed == 1
    assert stats.switches_to_sampling == 1


@pytest.mark.asyncio
async def test_switch_scheduler_min_sampling_window_blocks_early_train_switch() -> None:
    clock = FakeClock()
    order: list[str] = []
    scheduler = SwitchScheduler(
        SwitchSchedulerConfig(sampling_min_window_s=5.0, idle_sleep_s=0.0),
        switch_to_sampling=lambda: order.append("switch_to_sampling"),
        switch_to_training=lambda: order.append("switch_to_training"),
        clock=clock,
    )

    def sample_then_enqueue_training() -> None:
        order.append("sample-1")
        scheduler.enqueue_training(lambda: order.append("train-late"))
        clock.advance(4.0)

    def sample_after_min_window() -> None:
        order.append("sample-2")
        clock.advance(1.1)

    scheduler.enqueue_training(lambda: order.append("train-initial"))
    scheduler.enqueue_sampling(sample_then_enqueue_training)
    scheduler.enqueue_sampling(sample_after_min_window)

    await scheduler.run_until_idle(max_cycles=2)

    assert order == [
        "train-initial",
        "switch_to_sampling",
        "sample-1",
        "sample-2",
        "switch_to_training",
        "train-late",
        "switch_to_sampling",
    ]


@pytest.mark.asyncio
async def test_switch_scheduler_keeps_sampling_when_no_training_pending() -> None:
    clock = FakeClock()
    order: list[str] = []
    scheduler = SwitchScheduler(
        SwitchSchedulerConfig(sampling_min_window_s=0.0, idle_sleep_s=0.0),
        switch_to_sampling=lambda: order.append("switch_to_sampling"),
        switch_to_training=lambda: order.append("switch_to_training"),
        clock=clock,
    )
    scheduler.enqueue_sampling(lambda: order.append("sample-only"))

    await scheduler.run_until_idle()

    assert order == ["switch_to_sampling", "sample-only"]
    assert scheduler.mode.value == "sampling"


@pytest.mark.asyncio
async def test_adaptive_policy_increases_sampling_window_when_switch_cost_exceeds_threshold() -> None:
    clock = FakeClock()
    order: list[str] = []

    def costly_switch(name: str):
        def _inner() -> None:
            order.append(name)
            clock.advance(1.0)
        return _inner

    scheduler = SwitchScheduler(
        SwitchSchedulerConfig(
            sampling_min_window_s=2.0,
            policy=SwitchPolicy.ADAPTIVE,
            max_switch_overhead_fraction=0.10,
            adaptive_growth_factor=2.0,
            max_sampling_min_window_s=16.0,
            adaptive_window_s=100.0,
            idle_sleep_s=0.0,
        ),
        switch_to_sampling=costly_switch("switch_to_sampling"),
        switch_to_training=costly_switch("switch_to_training"),
        clock=clock,
    )
    scheduler.enqueue_training(lambda: order.append("train-1"))
    scheduler.enqueue_sampling(lambda: (order.append("sample-1"), clock.advance(2.1)))
    scheduler.enqueue_training(lambda: order.append("train-2"))
    scheduler.enqueue_sampling(lambda: (order.append("sample-2"), clock.advance(4.1)))

    stats = await scheduler.run_until_idle(max_cycles=2)

    assert stats.adaptive_adjustments >= 1
    assert scheduler.current_sampling_min_window_s > 2.0
    assert any(event.kind == "adaptive_window_increase" for event in scheduler.events)


@pytest.mark.asyncio
async def test_fixed_policy_does_not_change_sampling_window() -> None:
    clock = FakeClock()
    scheduler = SwitchScheduler(
        SwitchSchedulerConfig(
            sampling_min_window_s=2.0,
            policy=SwitchPolicy.FIXED,
            idle_sleep_s=0.0,
        ),
        switch_to_sampling=lambda: clock.advance(1.0),
        switch_to_training=lambda: clock.advance(1.0),
        clock=clock,
    )
    scheduler.enqueue_training(lambda: None)
    scheduler.enqueue_sampling(lambda: clock.advance(2.1))
    scheduler.enqueue_training(lambda: None)

    await scheduler.run_until_idle(max_cycles=2)

    assert scheduler.current_sampling_min_window_s == 2.0
    assert scheduler.stats.adaptive_adjustments == 0
