"""CPU policy tests: no engines, GPU allocation, or performance claims."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from tuft.config import SchedulingConfig
from tuft.loop_scheduling import LoopProfile, LoopProfiler, plan_training
from tuft.scheduling import ResourceScheduler


def history(profiler, run, last, *, period=1.0, duration=0.01):
    for i in range(4):
        profiler.training_ready(run, last - (3 - i) * period)
        profiler.executed(run, duration, commit=True)


def test_planner_consolidates_three_bursts_with_total_delay_budgets():
    profiles = [LoopProfile("a", 10, 1, 3), LoopProfile("b", 12, 1, 2), LoopProfile("c", 14, 1, 0)]
    slots = plan_training(profiles, 3)
    assert slots is not None
    assert [(s.run, s.start, s.end, s.delay) for s in slots] == [
        ("a", 12, 13, 2),
        ("b", 13, 14, 1),
        ("c", 14, 15, 0),
    ]
    assert profiles[0].ready == 10  # Planner never changes observed inputs.
    assert plan_training([LoopProfile("a", 0, 3, 0), LoopProfile("b", 1, 1, 1)], 5) is None
    # A tight budget preserves natural execution rather than violating the SLO.
    tight = plan_training([LoopProfile("a", 10, 1, 0.5), LoopProfile("b", 12, 1, 0)], 3)
    assert tight is not None
    assert [s.delay for s in tight] == [0, 0]


def test_gradient_accumulation_closes_once_and_forget_removes_history():
    p = LoopProfiler()
    p.training_ready("run", 10)
    p.executed("run", 0.2, commit=False)
    p.training_ready("run", 12)
    p.executed("run", 0.3, commit=False)
    assert not p.arrivals["run"]
    p.executed("run", 0.1, commit=True)
    assert list(p.arrivals["run"]) == [10]
    assert list(p.states["run"].durations) == pytest.approx([0.6])
    assert p.states["run"].observations == 1
    p.executed("run", 0.1, commit=True)  # An empty optimizer call is not a new loop.
    assert p.states["run"].observations == 1
    p.forget("run")
    assert not p.states and not p.arrivals


def test_complete_loops_exclude_own_delay_and_expire_irregular_predictions():
    p = LoopProfiler()
    history(p, "a", 30, period=10)
    p.states["a"].planned_delay = 2
    p.training_ready("a", 42)
    p.executed("a", 0.2, commit=True)
    assert p.period("a") == 10
    assert p.horizon(43) == 52
    assert p.horizon(53) is None
    p.training_ready("a", 100)
    p.executed("a", 0.2, commit=True)
    assert p.horizon(101) is None


def test_one_wave_deadline_never_extends_or_delays_already_started_wave():
    p = LoopProfiler()
    history(p, "a", 9, duration=0.1)
    history(p, "b", 9.4, duration=0.1)
    deadline, bad = p.sampling_gate("a", 9.5, max_delay=0.5, fraction=0.5, lookahead=1)
    assert not bad and deadline == pytest.approx(9.8)
    again, _ = p.sampling_gate("a", 9.7, max_delay=0.5, fraction=0.5, lookahead=1)
    assert again == deadline
    profiles = {r.run: r for r in p.profiles(9.7, 0.5, 0.5)}
    assert profiles["a"].budget == 0
    assert profiles["a"].ready == pytest.approx(10.3)
    assert p.sampling_gate("unknown", 9.8, max_delay=1, fraction=1, lookahead=1)[0] == 9.8


async def test_scheduler_wave_hold_cancellation_and_snapshot_identity():
    cfg = SchedulingConfig(
        policy="loopweave",
        loop_max_delay_s=0.2,
        loop_delay_fraction=0.25,
        training_quiet_s=0,
        sampling_alignment_s=0,
    )
    s = ResourceScheduler(2, cfg, flex_index=0)
    now = time.monotonic()
    history(s.profiler, "a", now - 0.8)
    history(s.profiler, "b", now - 0.7)
    seen = []

    async def sample(index):
        seen.append(index)
        return "immutable-snapshot"

    tasks = [
        asyncio.create_task(s.sample("same-user", "snapshot-v1", 4, True, sample, loop_id="a"))
        for _ in range(3)
    ]
    await asyncio.sleep(0)
    assert not seen and s.snapshot()["queued"] == 3
    deadline = s.profiler.states["a"].hold_until
    tasks[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await tasks[0]
    assert await asyncio.gather(*tasks[1:]) == ["immutable-snapshot"] * 2
    assert s.profiler.states["a"].hold_until == deadline
    assert s.snapshot()["queued"] == 0
    assert s._adapter[1] == "snapshot-v1"
    await s.close()


async def test_execution_telemetry_excludes_switch_cost_and_failure_invalidates(monkeypatch):
    import tuft.scheduling as module

    now = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    s = ResourceScheduler(2, SchedulingConfig(), flex_index=0)

    async def phase(training):
        now[0] += 10

    async def backward():
        now[0] += 2

    async def optimizer():
        now[0] += 1

    s.set_phase_transition(phase)
    await s.training(backward, tenant="a", reclaim_sampler_memory=True)
    await s.training(backward, tenant="a", reclaim_sampler_memory=True)
    await s.training(optimizer, tenant="a", kind="optimizer", reclaim_sampler_memory=True)
    assert list(s.profiler.arrivals["a"]) == [100]
    assert list(s.profiler.states["a"].durations) == [5]
    assert s.snapshot()["training_execution_s"] == 5
    assert s.snapshot()["switch_to_training_s"] == 10
    await s.training(optimizer)  # Maintenance without a loop identity.
    assert s.snapshot()["loop_observations"] == 1

    async def fail():
        raise RuntimeError("failed update")

    with pytest.raises(RuntimeError):
        await s.training(fail, tenant="a")
    assert "a" not in s.profiler.states
    await s.close()


async def test_cost_gate_uses_fixed_backlog_and_round_trip_not_just_idle_gpu():
    s = ResourceScheduler(
        2, SchedulingConfig(policy="loopweave", sampling_alignment_s=0), flex_index=0
    )
    s._phase_state = True
    s._switch_cost = {True: 0.2, False: 0.3}
    s._rates[1] = 0.01
    s._seconds_per_token = 0.005
    seen = []

    async def sample(index):
        seen.append(index)

    await s.sample("user", None, 10, True, sample)
    assert seen == [1]  # Fixed takes 0.1s; waking costs >=0.5s.
    s._rates[1] = 0.01
    release = asyncio.Event()
    started = asyncio.Event()

    async def busy(index):
        assert index == 1
        started.set()
        await release.wait()

    fixed = asyncio.create_task(s.sample("busy", None, 1000, False, busy))
    await started.wait()
    await s.sample("user", None, 10, True, sample)
    assert seen == [1, 0]  # The same rollout now benefits from borrowing.
    assert s.snapshot()["borrow_cost_rejections"] >= 1
    release.set()
    await fixed
    await s.close()


async def test_cold_probe_is_bounded_and_only_runs_with_fixed_backlog():
    cfg = SchedulingConfig(
        policy="loopweave",
        sampling_alignment_s=0,
        max_inflight_per_replica=1,
        borrow_min_backlog_tokens=20,
        borrow_probe_tokens=10,
        flex_max_inflight=8,
    )
    s = ResourceScheduler(2, cfg, flex_index=0)
    fixed_release, flex_release = asyncio.Event(), asyncio.Event()
    fixed_started, flex_started = asyncio.Event(), asyncio.Event()
    seen = []

    async def fixed(index):
        assert index == 1
        fixed_started.set()
        await fixed_release.wait()

    async def sample(index):
        seen.append(index)
        if index == 0:
            flex_started.set()
            await flex_release.wait()

    active = asyncio.create_task(s.sample("fixed", None, 100, False, fixed))
    await fixed_started.wait()
    jobs = [asyncio.create_task(s.sample(str(i), None, 10, True, sample)) for i in range(3)]
    await asyncio.wait_for(flex_started.wait(), 2)
    assert seen == [0] and s.snapshot()["queued"] == 2
    assert s.snapshot()["borrow_probes"] == 1
    fixed_release.set()
    flex_release.set()
    await asyncio.gather(active, *jobs)
    await s.close()


async def test_switch_back_reserve_rejects_deadline_crossing():
    s = ResourceScheduler(
        2, SchedulingConfig(cost_aware_borrowing=False, adapter_batch_window_s=0), flex_index=0
    )
    now = time.monotonic()
    history(s.profiler, "a", now - 0.8)
    s._seconds_per_token = 0.001
    s._switch_cost = {True: 0.2, False: 0.01}
    seen = []

    async def sample(index):
        seen.append(index)

    await s.sample("u", None, 1, True, sample)
    assert seen == [1]
    assert s.snapshot()["window_rejections"] > 0
    await s.close()


async def test_early_update_releases_obsolete_wave_hold():
    cfg = SchedulingConfig(
        policy="loopweave",
        loop_max_delay_s=1,
        loop_delay_fraction=0.25,
        sampling_alignment_s=0,
        training_quiet_s=0,
    )
    s = ResourceScheduler(2, cfg, flex_index=0)
    now = time.monotonic()
    history(s.profiler, "a", now - 0.8)
    history(s.profiler, "b", now - 0.6)
    entered = asyncio.Event()

    async def sample(index):
        entered.set()

    async def update():
        pass

    task = asyncio.create_task(s.sample("u", "v1", 10, True, sample, loop_id="a"))
    await asyncio.sleep(0)
    assert s.snapshot()["queued"] == 1 and not entered.is_set()
    await s.training(update, tenant="a", reclaim_sampler_memory=True)
    assert s.profiler.states["a"].planned_delay == 0
    await asyncio.wait_for(task, 0.1)  # Original hold would last ~0.19 seconds.
    await s.close()


async def test_shutdown_rejects_held_waves_without_dispatch():
    s = ResourceScheduler(
        2, SchedulingConfig(loop_max_delay_s=1, loop_delay_fraction=0.25), flex_index=0
    )
    now = time.monotonic()
    history(s.profiler, "a", now - 0.8)
    history(s.profiler, "b", now - 0.6)

    async def unexpected(index):
        pytest.fail("Held work must not dispatch on shutdown")

    task = asyncio.create_task(s.sample("u", None, 1, True, unexpected, loop_id="a"))
    await asyncio.sleep(0)
    await s.close()
    with pytest.raises(RuntimeError, match="closed"):
        await task
    assert s._wake_handle is None or s._wake_handle.cancelled()


async def test_probe_can_calibrate_under_a_known_long_horizon():
    s = ResourceScheduler(
        2,
        SchedulingConfig(
            policy="loopweave",
            sampling_alignment_s=0,
            max_inflight_per_replica=1,
            borrow_min_backlog_tokens=1,
        ),
        flex_index=0,
    )
    s._rates[1] = 0.01
    s._switch_cost = {True: 0.02, False: 0.02}
    s._phase_state = True
    now = time.monotonic()
    history(s.profiler, "run", now - 0.1)
    release = asyncio.Event()
    entered = asyncio.Event()
    indices = []

    async def busy(index):
        entered.set()
        await release.wait()

    async def sample(index):
        indices.append(index)

    task = asyncio.create_task(s.sample("fixed", None, 100, False, busy))
    await entered.wait()
    await s.sample("u", None, 1, True, sample)
    assert indices == [0] and s.snapshot()["borrow_probes"] == 1
    release.set()
    await task
    await s.close()


async def test_publication_wake_does_not_bypass_borrowing_decision():
    s = ResourceScheduler(
        2,
        SchedulingConfig(
            policy="loopweave", sampling_alignment_s=0, borrow_min_dwell_s=0, training_quiet_s=0
        ),
        flex_index=0,
    )
    s._switch_cost = {True: 0.2, False: 0.3}
    s._rates[1] = 0.01
    s._seconds_per_token = 0.005
    phases = []

    async def phase(training):
        phases.append(training)

    async def publication():
        pass

    s.set_phase_transition(phase)
    await s.training(publication)  # Adapter registration wakes the sampler.
    assert phases == [False] and s._phase_state is False
    selected = []

    async def sample(index):
        selected.append(index)

    await s.sample("u", None, 10, True, sample)
    assert selected == [1]  # Awake for maintenance is not a granted borrowing window.
    assert s.snapshot()["borrow_cost_rejections"] == 1
    await s.close()


async def test_held_work_is_not_counted_as_ready_probe_backlog():
    cfg = SchedulingConfig(
        policy="loopweave",
        sampling_alignment_s=0,
        loop_max_delay_s=1,
        loop_delay_fraction=0.25,
        max_inflight_per_replica=1,
        borrow_min_backlog_tokens=20,
    )
    s = ResourceScheduler(2, cfg, flex_index=0)
    now = time.monotonic()
    history(s.profiler, "a", now - 0.8)
    history(s.profiler, "b", now - 0.6)
    s._rates[1] = 0.001
    s._switch_cost = {True: 0.001, False: 0.001}
    release, entered = asyncio.Event(), asyncio.Event()

    async def fixed(index):
        entered.set()
        await release.wait()

    async def sample(index):
        pytest.fail("The only ready backlog is too small to justify a probe")

    busy = asyncio.create_task(s.sample("fixed", None, 100, False, fixed))
    await entered.wait()
    held = asyncio.create_task(s.sample("held", None, 20, True, sample, loop_id="a"))
    ready = asyncio.create_task(s.sample("ready", None, 10, True, sample))
    await asyncio.sleep(0)
    assert s.snapshot()["active"] == [0, 1]
    for task in [held, ready]:
        task.cancel()
    await asyncio.gather(held, ready, return_exceptions=True)
    release.set()
    await busy
    await s.close()


async def test_sampling_service_rate_excludes_wakeup(monkeypatch):
    import tuft.scheduling as module

    now = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    s = ResourceScheduler(
        2, SchedulingConfig(cost_aware_borrowing=False, adapter_batch_window_s=0), flex_index=0
    )

    async def phase(training):
        now[0] += 10

    async def sample(index):
        assert index == 0
        now[0] += 2

    s.set_phase_transition(phase)
    await s.sample("u", None, 10, True, sample)
    assert s._seconds_per_token == pytest.approx(0.2)
    assert s._rates[0] == pytest.approx(0.2)
    assert s._switch_cost[False] == 10
    await s.close()
