"""CPU policy tests: no engines, GPU allocation, or performance claims."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from tuft.config import SchedulingConfig
from tuft.loop_scheduling import LoopProfiler
from tuft.scheduling import ResourceScheduler


def history(profiler, run, last, *, period=1.0, duration=0.01):
    for i in range(4):
        profiler.training_ready(run, last - (3 - i) * period)
        profiler.executed(run, duration, commit=True)


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
        2, SchedulingConfig(policy="shared_weight", sampling_alignment_s=0), flex_index=0
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
        policy="shared_weight",
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


async def test_probe_can_calibrate_under_a_known_long_horizon():
    s = ResourceScheduler(
        2,
        SchedulingConfig(
            policy="shared_weight",
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
            policy="shared_weight", sampling_alignment_s=0, borrow_min_dwell_s=0, training_quiet_s=0
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


def test_profile_status():
    p = LoopProfiler()
    history(p, "regular", 9, duration=0.1)
    p.observe("new", 9)
    history(p, "late", 5)
    history(p, "pending", 9)
    p.training_ready("pending", 9.1)
    history(p, "irregular", 7)
    p.observe("irregular", 9)
    assert p.status(9.5) == dict(warming=1, overdue=1, pending=1, irregular=1, eligible=1)


async def test_borrowing_rechecks_profit_after_and_during_borrow():
    s = ResourceScheduler(
        2, SchedulingConfig(policy="shared_weight", adapter_batch_window_s=0), flex_index=0
    )
    s._switch_cost = {True: 0.03, False: 0.04}
    s._phase_state = False
    s._rates = [0.00066, 0.00021]
    s._seconds_per_token = 0.00066
    s._borrowing = True
    seen = []

    async def sample(index):
        seen.append(index)

    await s.sample("a", None, 320, True, sample)
    assert seen == [1]
    s._active[0] = 1
    s._work[0] = 320
    await s.sample("a", None, 320, True, sample)
    assert seen == [1, 1]
    s._active[0] = s._work[0] = 0
    await s.close()


async def test_exclusive_trace_separates_drain_queue_phase_service(monkeypatch):
    import tuft.scheduling as module

    now = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    s = ResourceScheduler(2, SchedulingConfig(scheduling_trace_capacity=2), flex_index=0)
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
    assert snap["publication_estimate_s"] == 5
    assert len(snap["decision_trace"]) == 2
    assert "private" not in str(snap)
    await s.close()


async def test_compound_mutation_drains_on_disconnect_and_shutdown():
    s = ResourceScheduler(1, SchedulingConfig(policy="balanced"))
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
