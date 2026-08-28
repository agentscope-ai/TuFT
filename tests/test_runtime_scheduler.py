from __future__ import annotations

import asyncio

from tuft.schedulers.rl_loop_scheduler import TenantLoopProfile
from tuft.schedulers.runtime_scheduler import RLRuntimeScheduler


class _FakeClock:
    def __init__(self) -> None:
        self.now_s = 0.0

    def now(self) -> float:
        return self.now_s

    def advance(self, delta: float) -> None:
        self.now_s += delta


class _FakeRouter:
    def __init__(self) -> None:
        self.active_backend = "fixed"
        self.flex_calls = 0
        self.fixed_calls = 0

    async def switch_to_flex(self) -> None:
        self.flex_calls += 1
        self.active_backend = "flex"

    async def switch_to_fixed(self) -> None:
        self.fixed_calls += 1
        self.active_backend = "fixed"


def _make(clock: _FakeClock, router: _FakeRouter, **overrides) -> RLRuntimeScheduler:
    params = dict(
        router=router,
        clock=clock.now,
        sleep_fn=_make_sleep(clock),
        flex_duty_cycle_enabled=True,
        flex_sampling_min_window_s=1.0,
        flex_switch_backlog_threshold=2,
        flex_duty_cycle_tick_s=1.0,
        flex_duty_cycle_min_dwell_s=0.0,
    )
    params.update(overrides)
    return RLRuntimeScheduler(**params)


def _make_sleep(clock: _FakeClock):
    async def _sleep(delta: float) -> None:
        clock.advance(delta)
        await asyncio.sleep(0)

    return _sleep


async def _tick(n: int = 1) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


def test_duty_cycle_flips_to_sampling_when_idle_with_backlog() -> None:
    async def main() -> None:
        clock = _FakeClock()
        router = _FakeRouter()
        sched = _make(clock, router)
        # One completed training step so the duty cycle is allowed to engage.
        sched.observe_training_ready("t0")
        sched.observe_training_forward_done("t0")
        await sched.observe_training_finish("t0")
        # Sampling backlog but no pending training.
        sched.observe_sampling_submit("t1")
        sched.observe_sampling_submit("t2")

        sched.start_duty_cycle()
        await _tick(4)
        assert router.active_backend == "flex"
        assert sched.duty_switches_to_sampling >= 1
        sched.stop_duty_cycle()

    asyncio.run(main())


def test_duty_cycle_does_not_flip_without_backlog() -> None:
    async def main() -> None:
        clock = _FakeClock()
        router = _FakeRouter()
        sched = _make(clock, router)
        sched.start_duty_cycle()
        await _tick(4)
        assert router.active_backend == "fixed"
        assert sched.duty_switches_to_sampling == 0
        sched.stop_duty_cycle()

    asyncio.run(main())


def test_duty_cycle_flips_back_to_training_when_pending() -> None:
    async def main() -> None:
        clock = _FakeClock()
        router = _FakeRouter()
        sched = _make(clock, router)
        # Start in sampling mode with a window already elapsed.
        router.active_backend = "flex"
        sched._duty_window_start_s = clock.now()
        clock.advance(5.0)
        # A training batch becomes ready.
        sched.observe_training_ready("t1")

        sched.start_duty_cycle()
        await _tick(4)
        assert router.active_backend == "fixed"
        assert sched.duty_switches_to_fixed >= 1
        sched.stop_duty_cycle()

    asyncio.run(main())


def test_duty_cycle_exits_sampling_once_the_window_is_amortized() -> None:
    """A ready burst ends the window, but not before it pays for its own flips."""

    async def main() -> None:
        clock = _FakeClock()
        router = _FakeRouter()
        sched = _make(clock, router, flex_sampling_min_window_s=1.0)
        router.active_backend = "flex"
        sched._duty_window_start_s = clock.now()
        clock.advance(2.0)  # past the 1s floor
        sched.observe_training_ready("t1")

        sched.start_duty_cycle()
        await _tick(4)
        assert router.active_backend == "fixed"
        sched.stop_duty_cycle()

    asyncio.run(main())


def test_duty_cycle_holds_the_sampling_window_until_it_is_amortized() -> None:
    """Without a floor the window collapses to one tick and both flips are waste.

    Training queues behind the closed gate, so pending_training goes positive on
    the very next tick; flipping back then would buy no sampling at all.
    """

    async def main() -> None:
        clock = _FakeClock()
        router = _FakeRouter()
        sched = _make(clock, router, flex_sampling_min_window_s=10.0)
        router.active_backend = "flex"
        sched._duty_window_start_s = clock.now()
        clock.advance(1.0)  # well below the 10s floor
        sched.observe_training_ready("t1")

        sched.start_duty_cycle()
        await _tick(4)
        assert router.active_backend == "flex"
        assert sched.duty_min_window_hold_s > 0

        # Once the floor is cleared the pending burst does end the window.
        clock.advance(10.0)
        await _tick(4)
        assert router.active_backend == "fixed"
        sched.stop_duty_cycle()

    asyncio.run(main())


def test_ensure_training_mode_forces_fixed_before_forward() -> None:
    async def main() -> None:
        clock = _FakeClock()
        router = _FakeRouter()
        sched = _make(clock, router)
        router.active_backend = "flex"
        await sched.ensure_training_mode()
        assert router.active_backend == "fixed"
        assert sched.duty_switches_to_fixed >= 1

    asyncio.run(main())


def test_observation_counters_track_pending_and_backlog() -> None:
    clock = _FakeClock()
    router = _FakeRouter()
    sched = _make(clock, router)
    sched.observe_sampling_submit("t1")
    sched.observe_sampling_submit("t2")
    assert sched._sampling_backlog == 2
    sched.observe_sampling_finish("t1")
    assert sched._sampling_backlog == 1
    sched.observe_training_ready("t1")
    assert sched._pending_training == 1
    # Pending is per-forward: it drops when the forward completes, not when the
    # optimizer step lands (a step issues several forwards).
    sched.observe_training_forward_done("t1")
    assert sched._pending_training == 0


def test_window_state_alternates_training_then_sampling() -> None:
    clock = _FakeClock()
    router = _FakeRouter()
    sched = _make(
        clock,
        router,
        flex_window_mode=True,
        flex_window_period_s=10.0,
        flex_training_window_s=4.0,
    )
    # Epoch starts at clock 0: [0,4) training, [4,10) sampling.
    assert sched._window_state(0.0)[0] is True
    assert sched._window_state(3.9)[0] is True
    assert sched._window_state(4.1)[0] is False
    assert sched._window_state(9.9)[0] is False
    assert sched._window_state(10.1)[0] is True
    # Horizon shrinks toward the next flip.
    assert abs(sched.time_to_next_flip(4.5) - 5.5) < 1e-6


def test_window_loop_flips_to_sampling_only_after_block_drains() -> None:
    async def main() -> None:
        clock = _FakeClock()
        router = _FakeRouter()
        sched = _make(
            clock,
            router,
            flex_window_mode=True,
            flex_window_period_s=10.0,
            flex_training_window_s=2.0,
            flex_duty_cycle_tick_s=1.0,
        )
        sched.observe_training_ready("t0")
        sched.observe_training_forward_done("t0")
        await sched.observe_training_finish("t0")
        # A training burst is still pending: entering the sampling window must
        # overrun (keep the block whole) instead of flipping.
        sched.observe_training_ready("t1")
        sched.start_duty_cycle()
        await _tick(6)
        assert router.active_backend == "fixed"
        assert sched.window_overrun_s > 0
        # Once the block drains (and the quiet gap elapses), sampling is taken.
        sched.observe_training_forward_done("t1")
        await sched.observe_training_finish("t1")
        await _tick(40)
        assert router.active_backend == "flex"
        sched.stop_duty_cycle()

    asyncio.run(main())


def test_sync_tenant_gets_nonzero_delay_budget() -> None:
    clock = _FakeClock()
    router = _FakeRouter()
    sched = _make(clock, router, align_slack_frac=0.5)

    # Two observations establish a period; the tenant never reports a version gap
    # (sync), yet W_i must be a positive latency budget so alignment is possible.
    async def main() -> None:
        for ready in (10.0, 30.0):
            clock.now_s = ready
            sched.observe_training_ready("t1")
            sched.observe_training_start("t1")
            clock.now_s = ready + 1.0
            await sched.observe_training_finish("t1")

    asyncio.run(main())
    profiles = sched.profiler.profiles()
    assert profiles, "profiler should have a profile after two observations"
    assert profiles[0].staleness_window_s > 0.0


class _BrokenRouter(_FakeRouter):
    """Router whose switch_to_fixed always fails (mirrors the adapter-replay crash)."""

    async def switch_to_fixed(self) -> None:
        self.fixed_calls += 1
        raise RuntimeError("dictionary changed size during iteration")


def test_failed_duty_switch_to_fixed_reopens_the_training_gate() -> None:
    """A broken flip must not wedge the gate closed for every later forward.

    Regression: switch_to_fixed raised half-way, so ``_training_open`` stayed
    cleared, every forward timed out on the hold, re-entered the failing flip and
    the cluster starved (flex utilization collapsed 66% -> 25%).
    """

    async def main() -> None:
        clock = _FakeClock()
        router = _BrokenRouter()
        sched = _make(clock, router, flex_window_mode=True, flex_max_training_hold_s=1.0)
        router.active_backend = "flex"
        sched._training_open.clear()

        # Does not raise, and leaves the gate open so training can proceed.
        await sched.acquire_training_slot()
        assert sched.early_flips == 1
        assert sched.duty_switch_failures == 1
        assert sched._training_open.is_set()

        # A second forward now passes straight through instead of timing out again.
        await asyncio.wait_for(sched.acquire_training_slot(), timeout=1.0)
        assert sched.early_flips == 1

    asyncio.run(main())


def test_failed_duty_switch_to_sampling_leaves_training_open() -> None:
    async def main() -> None:
        clock = _FakeClock()

        class _Router(_FakeRouter):
            async def switch_to_flex(self) -> None:
                raise RuntimeError("boom")

        router = _Router()
        sched = _make(clock, router)
        await sched._duty_switch_to_sampling(clock.now())
        assert sched.duty_switch_failures == 1
        assert sched.duty_switches_to_sampling == 0
        assert sched._training_open.is_set()

    asyncio.run(main())


def test_duty_switch_durations_are_reported() -> None:
    async def main() -> None:
        clock = _FakeClock()
        router = _FakeRouter()
        sched = _make(clock, router)

        async def _slow_flex() -> None:
            clock.advance(0.25)
            router.active_backend = "flex"

        router.switch_to_flex = _slow_flex  # type: ignore[method-assign]
        await sched._duty_switch_to_sampling(clock.now())
        snap = sched.snapshot()
        assert snap["duty_switch_to_sampling_s"] == 0.25
        assert snap["duty_switch_failures"] == 0

    asyncio.run(main())


class _StubProfiler:
    """Profiler stand-in returning fixed profiles to maybe_replan."""

    smoothing_alpha = 0.5

    def __init__(self, profiles) -> None:
        self._profiles = tuple(profiles)

    def profiles(self):
        return self._profiles


def _packing_profiles(last_window_s: float):
    """Three tenants on one lane; the third's required delay is the tunable part.

    a/b pack contiguously (a at 0, b delayed 9s behind a's 10s burst); c would
    need an 18s delay, so a small window for c makes the plan infeasible while
    leaving a feasible a/b prefix.
    """
    return [
        TenantLoopProfile(
            tenant_id="a",
            period_s=40.0,
            training_burst_s=10.0,
            ready_time_s=0.0,
            staleness_window_s=100.0,
        ),
        TenantLoopProfile(
            tenant_id="b",
            period_s=40.0,
            training_burst_s=10.0,
            ready_time_s=1.0,
            staleness_window_s=100.0,
        ),
        TenantLoopProfile(
            tenant_id="c",
            period_s=40.0,
            training_burst_s=10.0,
            ready_time_s=2.0,
            staleness_window_s=last_window_s,
        ),
    ]


def test_infeasible_plan_drops_every_hold_by_default() -> None:
    clock = _FakeClock()
    sched = _make(clock, _FakeRouter())
    sched.profiler = _StubProfiler(_packing_profiles(0.5))

    plan = sched.maybe_replan(clock.now())

    assert not plan.feasible
    assert sched.infeasible_replans == 1
    # The a/b prefix was computed and thrown away: this is why the fast loop
    # measured replans 26 / infeasible 25 / hold 0s at 24 tenants.
    assert len(plan.assignments) == 2
    assert sched.holds_armed == 0
    assert sched._hold_until == {}


def test_partial_plan_arms_the_feasible_prefix_of_an_infeasible_plan() -> None:
    clock = _FakeClock()
    sched = _make(clock, _FakeRouter(), fast_loop_partial_plan=True)
    sched.profiler = _StubProfiler(_packing_profiles(0.5))

    plan = sched.maybe_replan(clock.now())

    assert not plan.feasible
    # "a" runs at its natural time (no delay), "b" is held 9s behind it; "c" and
    # everything after it are left at their natural time.
    assert sched.holds_armed == 1
    assert sched.holds_armed_from_prefix == 1
    assert sched._hold_until == {"b": 9.0}
    assert "c" not in sched._hold_until

    asyncio.run(sched.acquire_sampling_permission("b"))
    assert sched.delay_hold_total_s == 9.0
    assert clock.now() == 9.0


def test_disabling_fast_loop_holds_leaves_sampling_untouched() -> None:
    clock = _FakeClock()
    sched = _make(clock, _FakeRouter(), fast_loop_holds_enabled=False)
    # A wide window for "c" makes the whole plan feasible, so holds would be
    # armed if the master switch were on.
    sched.profiler = _StubProfiler(_packing_profiles(100.0))

    plan = sched.maybe_replan(clock.now())

    assert plan.feasible
    assert sched.holds_armed == 0
    assert sched._hold_until == {}
    asyncio.run(sched.acquire_sampling_permission("b"))
    assert sched.delay_hold_total_s == 0.0


def test_lane_utilization_never_exceeds_one_lane() -> None:
    """Two tenants queueing on one lane must not report >100% occupancy.

    Utilization used to sum each tenant's step span, so a tenant that spent the
    step waiting behind another had its wait counted as training. With enough
    tenants the single lane reported 265% busy and the slow loop converted a
    sampling group on that fiction.
    """
    clock = _FakeClock()
    router = _FakeRouter()
    sched = _make(clock, router, flex_duty_cycle_enabled=False)

    async def scenario() -> None:
        # Tenant A holds the lane for 4s while tenant B waits for all of it.
        await sched.acquire_training_lane("A")
        waiter = asyncio.ensure_future(sched.acquire_training_lane("B"))
        await _tick(2)
        clock.advance(4.0)
        sched.release_training_lane("A")
        await waiter
        clock.advance(2.0)
        sched.release_training_lane("B")

    asyncio.run(scenario())

    clock.advance(4.0)  # 10s of wall time, 6s of lane time
    snapshot = sched.snapshot()
    assert snapshot["train_lane_busy_s"] == 6.0
    assert snapshot["train_lane_utilization"] <= 1.0


def test_cancelled_wait_for_the_training_window_does_not_leak_pending() -> None:
    """A forward abandoned while queueing must not leave pending_training high.

    pending_training gates the window loop's flip to sampling, so one leaked
    count per abandoned request permanently pins the flex GPU in training mode.
    """
    clock = _FakeClock()
    router = _FakeRouter()
    sched = _make(clock, router, flex_window_mode=True, flex_max_training_hold_s=100.0)
    sched._training_open.clear()  # flex is in a sampling window

    async def scenario() -> None:
        # Mirror the request path: count the arrival, then wait for the window.
        sched.observe_training_ready("A")
        task = asyncio.ensure_future(_guarded_slot(sched, "A"))
        await _tick(2)
        assert sched.snapshot()["pending_training"] == 1
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())
    assert sched.snapshot()["pending_training"] == 0


def test_state_passes_only_real_scheduler_kwargs() -> None:
    """Guard the config -> scheduler wiring: a renamed field must not reach GPU time.

    ``ServerState`` builds the scheduler with one keyword per eval-config field,
    so a name drift between the two only shows up as a TypeError once a server
    boots on 4 GPUs. Checking the call site statically catches it in CPU tests.
    """
    import ast
    import dataclasses
    from pathlib import Path

    from tuft.config import EvaluationConfig

    state_src = Path(__file__).resolve().parents[1] / "src" / "tuft" / "state.py"
    tree = ast.parse(state_src.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RLRuntimeScheduler"
    ]
    assert len(calls) == 1, "expected exactly one RLRuntimeScheduler construction"

    sched_fields = {f.name for f in dataclasses.fields(RLRuntimeScheduler)}
    eval_fields = set(EvaluationConfig.model_fields)
    for kw in calls[0].keywords:
        assert kw.arg in sched_fields, f"scheduler has no field {kw.arg!r}"
        # The value is always ``ev.<field>``; that field must exist too.
        if isinstance(kw.value, ast.Attribute) and isinstance(kw.value.value, ast.Name):
            if kw.value.value.id == "ev":
                assert kw.value.attr in eval_fields, (
                    f"EvaluationConfig has no field {kw.value.attr!r}"
                )


async def _guarded_slot(sched: RLRuntimeScheduler, tenant_id: str) -> None:
    """The request path's pairing of the arrival count with its release."""
    try:
        await sched.acquire_training_slot()
    finally:
        sched.observe_training_forward_done(tenant_id)

