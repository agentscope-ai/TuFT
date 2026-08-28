"""Server-side RL-loop runtime scheduler (optimal deployment mode).

Combines the scheduler cores in ``rl_loop_scheduler`` into one online
component that the TuFT request path feeds with observations:

- Loop discovery: ``TenantLoopProfiler`` infers period, burst duration, next
  ready time, and staleness window from request timing signals only.
- Fast loop: ``ContiguousTrainingScheduler`` packs training bursts back-to-back
  on one lane and emits one-shot sampling delays ``phi_i``.
- Slow loop: ``FixedFlexCompositionController`` decides Fixed<->Flex
  composition with hysteresis. Two conversion regimes are supported:
  ``identity_tag`` (a fixed group is just a relabeled flex-sampling group,
  near-zero cost) and ``quantized`` (fixed group loads an independently
  quantized base; conversion pays a measured reload cost).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from .rl_loop_scheduler import (
    CompositionAction,
    CompositionState,
    ContiguousTrainingScheduler,
    FastLoopPlan,
    FixedFlexCompositionController,
    LoopObservation,
    SlowLoopConfig,
    TenantLoopProfiler,
    apply_composition_action,
)


logger = logging.getLogger(__name__)


@dataclass
class _TenantRuntime:
    sampling_submitted_s: float | None = None
    training_ready_s: float | None = None
    training_started_s: float | None = None
    observed_gap: float = 0.0
    train_steps: int = 0
    total_training_s: float = 0.0
    # Training-lane seconds this tenant has held for the step in progress.
    step_lane_s: float = 0.0


def _inverse_action(action: CompositionAction) -> CompositionAction:
    """The action that undoes ``action`` in the composition bookkeeping."""
    if action == CompositionAction.FLEX_TO_FIXED:
        return CompositionAction.FIXED_TO_FLEX
    return CompositionAction.FLEX_TO_FIXED


@dataclass
class RLRuntimeScheduler:
    """Online discovery + fast loop + slow loop driver.

    Args:
        replan_every_s: fast-loop replan period.
        slow_loop_enabled: whether composition decisions are executed.
        conversion_mode: ``identity_tag`` or ``quantized``.
        fixed_to_flex_cost_s / flex_to_fixed_cost_s: measured conversion costs
            injected into the time line for the ``quantized`` regime.
        router: optional sampling runtime router used to execute real
            Fixed<->Flex switches (quantized regime only).
    """

    replan_every_s: float = 30.0
    slow_loop_enabled: bool = False
    conversion_mode: str = "identity_tag"
    fixed_to_flex_cost_s: float = 3.8
    flex_to_fixed_cost_s: float = 0.95
    router: Any = None
    # Groups the deployment starts with. ``flex_groups`` counts every
    # training-capable group, which includes GPUs left idle by the initial
    # composition: converting one of those to a fixed sampling group is exactly
    # what the slow loop does under low training pressure, and leaving them out
    # (flex_groups=1) made FLEX_TO_FIXED permanently illegal.
    initial_fixed_groups: int = 1
    initial_flex_groups: int = 1
    clock: Callable[[], float] = time.monotonic
    sleep_fn: Callable[[float], Any] = asyncio.sleep

    # Flex duty-cycle (optimal): flip the flex GPU into sampling while the
    # training lane is idle and sampling has backlog, then back to drain.
    flex_duty_cycle_enabled: bool = False
    flex_sampling_min_window_s: float = 3.0
    flex_switch_backlog_threshold: int = 4
    flex_duty_cycle_tick_s: float = 0.5
    flex_duty_cycle_min_dwell_s: float = 2.0
    # Upper bound on one sampling window, scaled up to *_max_scale when the
    # sampling queue sits far above flex_switch_backlog_threshold. 0 disables the
    # cap and restores the unbounded behaviour.
    flex_duty_window_cap_s: float = 10.0
    flex_duty_window_cap_max_scale: float = 6.0
    # Fast-loop training packing (consolidate bursts into contiguous blocks).
    pack_training: bool = False
    # Phase-locked periodic windows.
    flex_window_mode: bool = False
    flex_window_period_s: float = 60.0
    flex_training_window_s: float = 20.0
    flex_max_training_hold_s: float = 5.0
    align_slack_frac: float = 0.5
    # See AppConfig.rl_fast_loop_holds_enabled / rl_fast_loop_partial_plan.
    fast_loop_holds_enabled: bool = True
    fast_loop_partial_plan: bool = False
    # Quiet time with an empty lane before the block is considered finished.
    flex_block_quiet_s: float = 1.0
    # A gap must be at least this multiple of the measured round-trip switch cost
    # before a flip is spent on it.
    gap_gain_factor: float = 2.0
    # Sampling pipeline used to publish the flex group's L3 horizon.
    sampling_pipeline: Any = None

    profiler: TenantLoopProfiler = field(default_factory=TenantLoopProfiler)
    fast_loop: ContiguousTrainingScheduler = field(default_factory=ContiguousTrainingScheduler)
    slow_loop: FixedFlexCompositionController = field(
        default_factory=lambda: FixedFlexCompositionController(SlowLoopConfig())
    )

    _tenants: dict[str, _TenantRuntime] = field(default_factory=dict, init=False)
    _hold_until: dict[str, float] = field(default_factory=dict, init=False)
    _last_replan_s: float = field(default=0.0, init=False)
    _last_plan: Optional[FastLoopPlan] = field(default=None, init=False)
    _composition: CompositionState = field(
        default_factory=lambda: CompositionState(fixed_groups=1, flex_groups=1), init=False
    )
    _events: list[dict[str, Any]] = field(default_factory=list, init=False)
    # Duty-cycle bookkeeping.
    _pending_training: int = field(default=0, init=False)
    _sampling_backlog: int = field(default=0, init=False)
    _duty_task: Any = field(default=None, init=False)
    # In-flight background re-compositions (see _spawn_conversion).
    _conversion_tasks: list[Any] = field(default_factory=list, init=False)
    # Re-compositions whose replica boot failed and whose accounting was undone.
    failed_conversion_count: int = field(default=0, init=False)
    _duty_stop: bool = field(default=False, init=False)
    _any_training: bool = field(default=False, init=False)
    _training_open: Any = field(default=None, init=False)
    _window_epoch_s: float = field(default=0.0, init=False)
    _last_training_done_s: float = field(default=0.0, init=False)
    windows_training: int = field(default=0, init=False)
    windows_sampling: int = field(default=0, init=False)
    window_overrun_s: float = field(default=0.0, init=False)
    early_flips: int = field(default=0, init=False)
    align_fallbacks: int = field(default=0, init=False)
    aligned_tenants: int = field(default=0, init=False)
    _duty_window_start_s: float = field(default=0.0, init=False)
    _duty_last_switch_s: float = field(default=-1e9, init=False)
    _duty_current_min_window_s: float = field(default=0.0, init=False)
    _duty_recent_switches: Any = field(default=None, init=False)
    duty_time_in_sampling_s: float = field(default=0.0, init=False)
    duty_switches_to_sampling: int = field(default=0, init=False)
    duty_switches_to_fixed: int = field(default=0, init=False)
    # Measured cost of the duty-cycle flips themselves (excludes the cold warmup
    # transform done at startup) so the residual can be attributed to switching.
    duty_switch_to_sampling_s: float = field(default=0.0, init=False)
    # Smoothed length of the training-lane gaps we see, and how many gaps we
    # declined because the flip round trip would not amortize.
    observed_gap_s: float = field(default=0.0, init=False)
    gaps_skipped_not_worth_it: int = field(default=0, init=False)
    duty_switch_to_fixed_s: float = field(default=0.0, init=False)
    duty_switch_failures: int = field(default=0, init=False)
    # Time training waited because the sampling window had not yet amortized its
    # own switch cost (the price paid for gap filling).
    duty_min_window_hold_s: float = field(default=0.0, init=False)
    # Windows ended by the upper bound rather than by training becoming ready.
    duty_window_cap_exits: int = field(default=0, init=False)
    # Windows closed because the sampling queue emptied before the time floor.
    duty_drained_exits: int = field(default=0, init=False)
    # Windows closed on training readiness before the old time floor would have
    # allowed it. Counts how often the floor used to hold the training lane back.
    duty_early_exits: int = field(default=0, init=False)
    # duty_time_in_sampling_s as of the last gap observation (see _observe_gap).
    _gap_sampling_baseline_s: float = field(default=0.0, init=False)
    training_idle_s: float = field(default=0.0, init=False)
    # Single logical training lane: forwards of different tenants never overlap
    # (paper: "Training in TuFT is always a single logical task").
    _training_lane: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    # Wall time the single training lane was actually held. Summing per-tenant
    # step spans instead counted lane-contention waits once per waiting tenant,
    # so "utilization" of a single lane could exceed 1.0 (measured 2.65) and the
    # slow loop converted groups on a pressure signal that was never real.
    _lane_busy_s: float = field(default=0.0, init=False)
    _lane_acquired_s: Optional[float] = field(default=None, init=False)
    _lane_holder: Optional[str] = field(default=None, init=False)
    _start_s: float = field(default=0.0, init=False)
    delay_hold_total_s: float = field(default=0.0, init=False)
    replans: int = field(default=0, init=False)
    infeasible_replans: int = field(default=0, init=False)
    conversion_count: int = field(default=0, init=False)
    conversion_time_s: float = field(default=0.0, init=False)
    # How many one-shot sampling delays were actually armed, and how many of them
    # came from the feasible prefix of an infeasible plan. Without these the fast
    # loop looks active (replans climb) while applying nothing.
    holds_armed: int = field(default=0, init=False)
    holds_armed_from_prefix: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._start_s = self.clock()
        self._composition = CompositionState(
            fixed_groups=max(0, self.initial_fixed_groups),
            flex_groups=max(1, self.initial_flex_groups),
        )
        self._duty_current_min_window_s = self.flex_sampling_min_window_s
        self._duty_recent_switches = deque()
        self._training_open = asyncio.Event()
        self._training_open.set()
        self._window_epoch_s = self._start_s

    # ------------------------------------------------------------------
    # Observation hooks (fed by SamplingController / TrainingController)
    # ------------------------------------------------------------------
    def observe_sampling_submit(self, tenant_id: str) -> None:
        self._tenant(tenant_id).sampling_submitted_s = self.clock()
        self._sampling_backlog += 1

    def observe_sampling_finish(self, tenant_id: str) -> None:
        self._sampling_backlog = max(0, self._sampling_backlog - 1)

    def observe_training_ready(self, tenant_id: str) -> None:
        now = self.clock()
        self._tenant(tenant_id).training_ready_s = now
        if self._pending_training == 0:
            self._observe_gap(now)
        self._pending_training += 1

    def observe_training_start(self, tenant_id: str) -> None:
        self._tenant(tenant_id).training_started_s = self.clock()

    def observe_training_forward_done(self, tenant_id: str) -> None:
        """One forward finished: the lane has one less piece of pending work.

        Pending must be decremented per forward (not per optimizer step): a step
        issues several forwards, so decrementing only on optim_step made the
        counter drift up forever and the scheduler never saw the block drain.
        """
        self._pending_training = max(0, self._pending_training - 1)
        self._last_training_done_s = self.clock()

    def observe_consumed_staleness(self, tenant_id: str, version_gap: int) -> None:
        """Infer the staleness window from the version gap of consumed rollouts."""
        runtime = self._tenant(tenant_id)
        alpha = self.profiler.smoothing_alpha
        runtime.observed_gap = alpha * max(0, version_gap) + (1 - alpha) * runtime.observed_gap

    async def observe_training_finish(self, tenant_id: str) -> None:
        """Record one full loop observation for discovery and maybe replan."""
        now = self.clock()
        runtime = self._tenant(tenant_id)
        burst_s = runtime.step_lane_s
        if burst_s <= 0.0:
            started = runtime.training_started_s or now
            burst_s = max(0.0, now - started)
        runtime.step_lane_s = 0.0
        runtime.total_training_s += burst_s
        runtime.train_steps += 1
        self._any_training = True

        profile = self.profiler._estimates.get(tenant_id)
        period = profile.period_s if profile and profile.period_s else 30.0
        # Staleness window in seconds: sync tenants (gap ~0) tolerate no delay.
        # W_i is the budget for a one-shot delay applied BEFORE sampling. Such a
        # shift moves the tenant's whole loop, so its rollouts are still drawn
        # from the current weights and staleness does not grow; the real bound is
        # a latency budget. Async tenants additionally tolerate their observed
        # version gap. Using 0 for sync tenants (as before) made every alignment
        # infeasible and left training scattered.
        window_s = max(runtime.observed_gap * period, self.align_slack_frac * period)
        self.profiler.record(
            LoopObservation(
                tenant_id=tenant_id,
                sampling_submitted_s=runtime.sampling_submitted_s or now,
                training_ready_s=runtime.training_ready_s or started,
                training_started_s=now - burst_s,
                training_finished_s=now,
                staleness_window_s=window_s,
            )
        )
        self.maybe_replan(now)
        await self._maybe_slow_loop(now)

    async def acquire_training_lane(self, tenant_id: Optional[str] = None) -> None:
        await self._training_lane.acquire()
        self._lane_acquired_s = self.clock()
        self._lane_holder = tenant_id

    def release_training_lane(self, tenant_id: Optional[str] = None) -> None:
        if not self._training_lane.locked():
            return
        if self._lane_acquired_s is not None:
            busy = max(0.0, self.clock() - self._lane_acquired_s)
            self._lane_busy_s += busy
            holder = self._lane_holder or tenant_id
            if holder is not None:
                self._tenant(holder).step_lane_s += busy
        self._lane_acquired_s = None
        self._lane_holder = None
        self._training_lane.release()

    # ------------------------------------------------------------------
    # Fast loop: one-shot sampling delay gate
    # ------------------------------------------------------------------
    def maybe_replan(self, now: Optional[float] = None) -> Optional[FastLoopPlan]:
        now = now if now is not None else self.clock()
        if self.replans > 0 and now - self._last_replan_s < self.replan_every_s:
            return self._last_plan

        profiles = self.profiler.profiles()
        if not profiles:
            return None
        plan = self.fast_loop.plan_fast_loop(
            profiles,
            pack_training=self.pack_training,
            align_period_s=self.flex_window_period_s if self.flex_window_mode else None,
            align_phase_s=self._window_epoch_s,
            align_window_s=self.flex_training_window_s if self.flex_window_mode else None,
        )
        self.align_fallbacks = plan.align_fallbacks
        self.aligned_tenants = plan.aligned
        self._last_plan = plan
        self._last_replan_s = now
        self.replans += 1
        from_prefix = False
        if not plan.feasible:
            self.infeasible_replans += 1
            logger.info(
                "fast loop infeasible for tenant %s (required delay %.2fs > window %.2fs); "
                "%s %d-tenant feasible prefix",
                plan.infeasible_tenant_id,
                plan.required_delay_s,
                plan.staleness_window_s,
                "applying" if self.fast_loop_partial_plan else "dropping",
                len(plan.assignments),
            )
            if not self.fast_loop_partial_plan:
                return plan
            from_prefix = True
        if not self.fast_loop_holds_enabled:
            return plan
        for assignment in plan.assignments:
            if assignment.sampling_delay_s > 0:
                self._hold_until[assignment.tenant_id] = now + assignment.sampling_delay_s
                self.holds_armed += 1
                if from_prefix:
                    self.holds_armed_from_prefix += 1
        return plan

    async def acquire_sampling_permission(self, tenant_id: str) -> None:
        """Apply the one-shot sampling delay assigned by the fast loop."""
        hold_until = self._hold_until.pop(tenant_id, None)
        if hold_until is None:
            return
        delay = hold_until - self.clock()
        if delay > 0:
            self.delay_hold_total_s += delay
            await self.sleep_fn(delay)

    # ------------------------------------------------------------------
    # Flex duty-cycle: use freed training windows for sampling
    # ------------------------------------------------------------------
    def attach_router(self, router: Any) -> None:
        self.router = router

    async def acquire_training_slot(self) -> None:
        """Wait until the flex training window is open.

        While the flex GPU is in a sampling window the gate is closed and
        arriving training requests queue here (pending_training rises) instead
        of running, so the next training window drains them back-to-back
        (compacted). Disabled mode leaves the gate open (no blocking).
        """
        if self._training_open is None or not self.flex_duty_cycle_enabled:
            return
        if not self.flex_window_mode:
            await self._training_open.wait()
            return
        # Window mode: wait for the next training window, but never longer than
        # flex_max_training_hold_s (a misaligned tenant then flips it early).
        try:
            await asyncio.wait_for(
                self._training_open.wait(), timeout=self.flex_max_training_hold_s
            )
        except (asyncio.TimeoutError, TimeoutError):
            self.early_flips += 1
            # _duty_switch_to_fixed never raises, so a failed flip degrades to
            # "training runs anyway" instead of failing this tenant's forward.
            await self._duty_switch_to_fixed(self.clock())

    async def ensure_training_mode(self) -> None:
        """Guarantee the flex backend is in training mode before a forward."""
        if self.router is None:
            return
        if getattr(self.router, "active_backend", "fixed") == "flex":
            await self.router.switch_to_fixed()
            self.duty_switches_to_fixed += 1
            self._duty_last_switch_s = self.clock()

    def start_duty_cycle(self) -> None:
        if not self.flex_duty_cycle_enabled or self.router is None:
            return
        if self._duty_task is not None:
            return
        self._duty_stop = False
        self._duty_last_switch_s = self.clock()
        self._duty_task = asyncio.ensure_future(self._duty_loop())
        logger.info("flex duty-cycle controller started")

    def stop_duty_cycle(self) -> None:
        self._duty_stop = True
        if self._duty_task is not None:
            self._duty_task.cancel()
            self._duty_task = None

    def _window_state(self, now: float) -> tuple[bool, float]:
        """Return (in_training_window, next_boundary_s) for the periodic schedule."""
        period = max(self.flex_window_period_s, 1e-6)
        train_len = min(max(self.flex_training_window_s, 0.0), period)
        pos = (now - self._window_epoch_s) % period
        if pos < train_len:
            return True, now + (train_len - pos)
        return False, now + (period - pos)

    def time_to_next_flip(self, now: float | None = None) -> float:
        now = self.clock() if now is None else now
        return self._window_state(now)[1] - now

    async def _window_loop(self) -> None:
        """Phase-locked alternation: one training block, then one sampling window.

        Training requests are aligned into the training window by the fast loop's
        one-shot sampling delay, so the block is contiguous. If the block has not
        drained when the window ends we overrun rather than cut it (keeping the
        block whole); the flip to sampling happens once the lane is free.
        """
        while not getattr(self, "_duty_stop", False):
            await self.sleep_fn(self.flex_duty_cycle_tick_s)
            if self.router is None:
                continue
            now = self.clock()
            in_training, boundary = self._window_state(now)
            active = getattr(self.router, "active_backend", "fixed")
            if active == "flex":
                self.duty_time_in_sampling_s += self.flex_duty_cycle_tick_s
            # Publish the L3 horizon so the pipeline stops admitting runs that
            # cannot finish before the scheduled flip (paper: flex horizon).
            if self.sampling_pipeline is not None:
                setter = getattr(self.sampling_pipeline, "set_flex_horizon", None)
                if callable(setter):
                    setter(boundary if active == "flex" else None)
            if in_training:
                if active != "fixed":
                    await self._duty_switch_to_fixed(now)
                    self.windows_training += 1
                if self._training_open is not None and not self._training_open.is_set():
                    self._training_open.set()
            else:
                if active == "flex":
                    continue
                if self._pending_training > 0 or self._training_lane.locked():
                    # Block still draining: overrun instead of cutting it.
                    self.window_overrun_s += self.flex_duty_cycle_tick_s
                    continue
                if now - self._last_training_done_s < self.flex_block_quiet_s:
                    # A forward just finished; wait out the quiet gap so we do not
                    # flip in the middle of a tenant's step.
                    continue
                if not self._any_training:
                    continue
                await self._duty_switch_to_sampling(now)
                self.windows_sampling += 1

    async def _duty_loop(self) -> None:
        if self.flex_window_mode:
            await self._window_loop()
            return
        while not getattr(self, "_duty_stop", False):
            await self.sleep_fn(self.flex_duty_cycle_tick_s)
            if self.router is None:
                continue
            now = self.clock()
            active = getattr(self.router, "active_backend", "fixed")
            if active == "flex":
                self.duty_time_in_sampling_s += self.flex_duty_cycle_tick_s
                # Leave the moment training is ready. There is no exit floor:
                # thrash protection belongs at the entry gate, which already
                # requires the observed gap to amortise a round trip
                # (_gap_worth_filling), and holding training back cannot repair a
                # misprediction - it compounds it. Two measured runs show the cost
                # of the floor: at sixteen tenants 338.5s of 470s sampling time
                # was floor-holding at 0.213 req/s, and at four tenants - where
                # training is 82% idle and the mechanism should win easily - 55.0s
                # of 68.0s was floor-holding while training waited, for 38 served
                # requests and a 45.3% wall regression.
                #
                # The exchange rate is what settles it: staying one more second
                # buys the flex group ~0.6 requests, about 7% of aggregate
                # sampling capacity, while costing a full second of the training
                # lane that wall clock is measured by.
                held_s = now - self._duty_window_start_s
                cap_s = self._duty_window_cap_s()
                if self._pending_training > 0:
                    if held_s < self._duty_current_min_window_s:
                        self.duty_early_exits += 1
                    await self._duty_switch_to_fixed(now)
                elif held_s >= cap_s:
                    # Hard upper bound. Exit-on-ready alone was not enough: the
                    # gate that keeps training out while we sample also keeps
                    # _pending_training at 0, so a window could run unbounded. A
                    # measured run held 1040.5s across 8 fills (130.1s each)
                    # against a 3.5s observed gap and lost 32.8% of throughput.
                    self.duty_window_cap_exits += 1
                    await self._duty_switch_to_fixed(now)

            else:
                if self._pending_training == 0:
                    self.training_idle_s += self.flex_duty_cycle_tick_s
                # No _any_training gate: the first sampling wave (before any
                # training exists) is exactly when the flex GPU would otherwise
                # sit idle for minutes. Exit-on-ready handles the race with the
                # first training request.
                if (
                    self._pending_training == 0
                    and not self._training_lane.locked()
                    # Real gap, not an intra-step bubble.
                    and (now - self._last_training_done_s) >= self.flex_block_quiet_s
                    and self._sampling_backlog >= self.flex_switch_backlog_threshold
                    and (now - self._duty_last_switch_s) >= self.flex_duty_cycle_min_dwell_s
                    and self._gap_worth_filling()
                ):
                    await self._duty_switch_to_sampling(now)

    def _duty_window_cap_s(self) -> float:
        """Upper bound on one sampling window.

        Scaled by how far the sampling queue is above the flip threshold: when
        supply is badly short a longer window is genuinely better (replaying the
        measured trace at a quarter of the arrival rate, an unbounded window beat a
        10s cap), but at the measured rate a bounded window nearly halved wall
        time. Always at least one round trip, or the window cannot pay for itself.
        """
        base = max(self.flex_duty_window_cap_s, 0.0)
        if base <= 0.0:
            return float("inf")
        threshold = max(self.flex_switch_backlog_threshold, 1)
        pressure = self._sampling_backlog / threshold
        scale = min(max(pressure, 1.0), self.flex_duty_window_cap_max_scale)
        return max(base * scale, self._mean_switch_round_trip_s() or base)

    def _mean_switch_round_trip_s(self) -> float:
        """Measured cost of one sampling round trip (to sampling and back)."""
        to_s = (
            self.duty_switch_to_sampling_s / self.duty_switches_to_sampling
            if self.duty_switches_to_sampling
            else 0.0
        )
        to_t = (
            self.duty_switch_to_fixed_s / self.duty_switches_to_fixed
            if self.duty_switches_to_fixed
            else 0.0
        )
        return to_s + to_t

    def _gap_worth_filling(self) -> bool:
        """Only spend a flip when the expected gap amortizes the round trip.

        Without this the duty cycle flips into every quiet moment and can pay two
        transforms for a gap shorter than the switches themselves, which makes the
        run slower instead of denser. Costs are unknown until the first flip, so
        the first gap is always taken (that is how they get measured).
        """
        round_trip = self._mean_switch_round_trip_s()
        if round_trip <= 0.0 or self.observed_gap_s <= 0.0:
            return True
        if self.observed_gap_s >= self.gap_gain_factor * round_trip:
            return True
        self.gaps_skipped_not_worth_it += 1
        return False

    def _observe_gap(self, now: float) -> None:
        """Smooth the length of the training-lane gap that just ended.

        Time spent holding the GPU for sampling is subtracted: it is blockage this
        controller caused, not spare capacity it found. Including it made the
        estimate self-reinforcing - a measured run drifted from a 3.5s gap to
        23.5s purely because its own 130s windows stretched the interval, and the
        gain gate then cleared on every single gap (gaps_skipped=0) because the
        inflated estimate always exceeded gain_factor * round_trip.
        """
        if self._last_training_done_s <= 0.0:
            return
        gap = now - self._last_training_done_s
        self_inflicted = max(
            self.duty_time_in_sampling_s - self._gap_sampling_baseline_s, 0.0
        )
        self._gap_sampling_baseline_s = self.duty_time_in_sampling_s
        gap -= self_inflicted
        if gap <= 0.0:
            return
        self.observed_gap_s = (
            gap if self.observed_gap_s == 0.0 else 0.5 * gap + 0.5 * self.observed_gap_s
        )

    async def _duty_switch_to_sampling(self, now: float) -> None:
        # Hold arriving training so it queues and the next window compacts.
        if self._training_open is not None:
            self._training_open.clear()
        t0 = self.clock()
        try:
            await self.router.switch_to_flex()
        except Exception:  # pylint: disable=broad-except
            # The flex GPU stays in training mode: reopen the gate, otherwise
            # every training forward would block on a window that never starts.
            self.duty_switch_failures += 1
            if self._training_open is not None:
                self._training_open.set()
            logger.exception("duty cycle: flex -> sampling failed; staying in training")
            return
        duration = self.clock() - t0
        self.duty_switches_to_sampling += 1
        self.duty_switch_to_sampling_s += duration
        self._duty_last_switch_s = now
        self._duty_window_start_s = now
        self._record_duty_switch(now, duration)
        # Publish the sampling window so the router only admits short requests
        # that finish before the flex GPU must return to training.
        setter = getattr(self.router, "set_flex_horizon", None)
        if setter is not None:
            window = max(self.observed_gap_s, self.flex_sampling_min_window_s)
            setter(now + window)
        logger.info("duty cycle: flex -> sampling (backlog=%d)", self._sampling_backlog)

    async def _duty_switch_to_fixed(self, now: float) -> None:
        setter = getattr(self.router, "set_flex_horizon", None)
        if setter is not None:
            setter(None)
        """Return the flex GPU to training. Never raises.

        The gate is reopened in ``finally``: a switch that failed half-way used to
        leave ``_training_open`` cleared forever, so every subsequent forward hit
        the hold timeout, re-entered here, failed again, and the whole cluster
        starved (observed as a 66% -> 25% collapse in flex utilization).
        """
        t0 = self.clock()
        ok = False
        try:
            await self.router.switch_to_fixed()
            ok = True
        except Exception:  # pylint: disable=broad-except
            self.duty_switch_failures += 1
            logger.exception("duty cycle: flex -> training failed")
        finally:
            if self._training_open is not None:
                self._training_open.set()
        duration = self.clock() - t0
        if not ok:
            return
        self.duty_switches_to_fixed += 1
        self.duty_switch_to_fixed_s += duration
        self._duty_last_switch_s = now
        self._record_duty_switch(now, duration)
        logger.info("duty cycle: flex -> training (pending=%d)", self._pending_training)

    def _record_duty_switch(self, now: float, duration: float) -> None:
        self._duty_recent_switches.append((now, duration))
        cutoff = now - 60.0
        while self._duty_recent_switches and self._duty_recent_switches[0][0] < cutoff:
            self._duty_recent_switches.popleft()
        total = sum(d for _, d in self._duty_recent_switches)
        elapsed = max(now - self._duty_recent_switches[0][0], 1e-9)
        if total / elapsed > 0.10:
            self._duty_current_min_window_s *= 1.5

    # ------------------------------------------------------------------
    # Slow loop: Fixed<->Flex composition
    # ------------------------------------------------------------------
    async def _maybe_slow_loop(self, now: float) -> None:
        if not self.slow_loop_enabled:
            return
        elapsed = max(now - self._start_s, 1e-9)
        pressure = self.slow_loop.observe_pressure(self._lane_busy_s, elapsed)
        expected_gain = pressure * elapsed
        decision = self.slow_loop.decide(
            self._composition, now_s=now, expected_gain_s=expected_gain
        )
        if decision.action == CompositionAction.HOLD:
            return

        cost = (
            self.fixed_to_flex_cost_s
            if decision.action == CompositionAction.FIXED_TO_FLEX
            else self.flex_to_fixed_cost_s
        )
        # The conversion mode picks the cost model, not whether the deployment is
        # actually re-composed. identity_tag used to skip _execute_conversion
        # entirely, so the slow loop only moved bookkeeping: a measured run
        # reported conversion_count=4 and composition 3+5 -> 7+1 while the router
        # still served from 3 replicas, leaving most of the re-composition gain
        # on the table.
        if self.conversion_mode == "identity_tag":
            # Relabeling a flex-sampling group as fixed is near-zero cost: the
            # replica is added/removed but no format conversion or reload runs.
            #
            # The re-composition runs in the background because _maybe_slow_loop
            # is awaited from observe_training_finish, i.e. inside a tenant's
            # training step. Growing the pool boots a vLLM engine, so doing it
            # inline stalled the training lane: the measured 1 flex + 3 fixed
            # growth arm took 1149s against 727s for the static baseline on the
            # same 48-step workload even though it ended with twice the sampling
            # capacity. The decision (and its cost model) is applied immediately;
            # only the replica boot is deferred.
            cost = 0.0
            self._spawn_conversion(decision.action)
        else:
            await self._execute_conversion(decision.action)
            await self.sleep_fn(cost)

        self._composition = apply_composition_action(self._composition, decision.action, now_s=now)
        self.conversion_count += 1
        self.conversion_time_s += cost
        self._events.append(
            {
                "kind": "slow_loop_conversion",
                "t_s": now - self._start_s,
                "action": decision.action.value,
                "mode": self.conversion_mode,
                "cost_s": cost,
                "pressure": decision.smoothed_pressure,
            }
        )
        logger.info(
            "slow loop conversion %s (%s, cost=%.2fs, pressure=%.3f)",
            decision.action.value,
            self.conversion_mode,
            cost,
            decision.smoothed_pressure,
        )

    def _spawn_conversion(self, action: CompositionAction) -> None:
        """Run a re-composition off the training path, rolling back if it fails.

        The bookkeeping is applied as soon as the decision is taken so the
        controller's cost model stays synchronous, but the replica boot can fail
        (a growth run reported ``fixed_groups=7`` while only 4 replicas existed,
        because three ``add_one_fixed_replica`` calls raised out of
        ``_create_engine`` and the accounting had already advanced). Every later
        decision was then made against a pool that did not exist, so a failure
        has to undo the accounting.
        """

        async def _run() -> None:
            try:
                await self._execute_conversion(action)
            except Exception:
                logger.exception("slow loop conversion %s failed", action.value)
                self._composition = apply_composition_action(
                    self._composition, _inverse_action(action), now_s=self.clock()
                )
                self.conversion_count = max(0, self.conversion_count - 1)
                self.failed_conversion_count += 1

        task = asyncio.ensure_future(_run())
        self._conversion_tasks.append(task)
        self._conversion_tasks = [t for t in self._conversion_tasks if not t.done()]

    async def _execute_conversion(self, action: CompositionAction) -> None:
        """Execute the real runtime switch when a router is attached.

        identity_tag mode: shrink/grow the fixed sampling replica pool.
        The removed replica's GPU becomes available for training (identity_tag
        cost is near-zero because the 'fixed' group is just a regular sampling
        replica — no quantization or format change). Adding it back creates a
        fresh sampling replica on a free GPU.
        """
        if self.router is None:
            return
        if action == CompositionAction.FIXED_TO_FLEX:
            # Remove one fixed sampling replica -> that GPU is now free for training
            remove_fn = getattr(self.router, "remove_one_fixed_replica", None)
            if remove_fn is not None:
                await remove_fn()
            else:
                switch = getattr(self.router, "switch_to_flex", None)
                if switch is not None:
                    await switch()
        else:
            # Add one fixed sampling replica back
            add_fn = getattr(self.router, "add_one_fixed_replica", None)
            if add_fn is not None:
                await add_fn()
            else:
                switch = getattr(self.router, "switch_to_fixed", None)
                if switch is not None:
                    await switch()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def _tenant(self, tenant_id: str) -> _TenantRuntime:
        if tenant_id not in self._tenants:
            self._tenants[tenant_id] = _TenantRuntime()
        return self._tenants[tenant_id]

    def snapshot(self) -> dict[str, Any]:
        plan_dump = None
        if self._last_plan is not None:
            plan_dump = {
                "feasible": self._last_plan.feasible,
                "infeasible_tenant_id": self._last_plan.infeasible_tenant_id,
                "assignments": [
                    {
                        "tenant_id": a.tenant_id,
                        "sampling_delay_s": a.sampling_delay_s,
                        "train_start_s": a.train_start_s,
                    }
                    for a in self._last_plan.assignments
                ],
            }
        return {
            "replans": self.replans,
            "infeasible_replans": self.infeasible_replans,
            "delay_hold_total_s": self.delay_hold_total_s,
            "fast_loop_holds_enabled": self.fast_loop_holds_enabled,
            "fast_loop_partial_plan": self.fast_loop_partial_plan,
            "holds_armed": self.holds_armed,
            "holds_armed_from_prefix": self.holds_armed_from_prefix,
            "conversion_count": self.conversion_count,
            "failed_conversion_count": self.failed_conversion_count,
            "conversion_time_s": self.conversion_time_s,
            "conversion_mode": self.conversion_mode,
            "composition": {
                "fixed_groups": self._composition.fixed_groups,
                "flex_groups": self._composition.flex_groups,
                "active_fixed_replicas": (
                    self.router.get_fixed_replica_count()
                    if self.router and hasattr(self.router, "get_fixed_replica_count")
                    else None
                ),
            },
            "last_plan": plan_dump,
            "events": list(self._events),
            "duty_cycle_enabled": self.flex_duty_cycle_enabled,
            "window_mode": self.flex_window_mode,
            "window_period_s": self.flex_window_period_s,
            "training_window_s": self.flex_training_window_s,
            "windows_training": self.windows_training,
            "windows_sampling": self.windows_sampling,
            "window_overrun_s": self.window_overrun_s,
            "early_flips": self.early_flips,
            "align_fallbacks": self.align_fallbacks,
            "aligned_tenants": self.aligned_tenants,
            "duty_cycle_time_in_sampling_s": self.duty_time_in_sampling_s,
            "duty_cycle_switches_to_sampling": self.duty_switches_to_sampling,
            "duty_cycle_switches_to_fixed": self.duty_switches_to_fixed,
            "duty_switch_to_sampling_s": self.duty_switch_to_sampling_s,
            "observed_gap_s": self.observed_gap_s,
            "mean_switch_round_trip_s": self._mean_switch_round_trip_s(),
            "gaps_skipped_not_worth_it": self.gaps_skipped_not_worth_it,
            "duty_window_cap_exits": self.duty_window_cap_exits,
            "duty_drained_exits": self.duty_drained_exits,
            "duty_early_exits": self.duty_early_exits,
            "duty_switch_to_fixed_s": self.duty_switch_to_fixed_s,
            "duty_switch_failures": self.duty_switch_failures,
            "duty_min_window_hold_s": self.duty_min_window_hold_s,
            "training_idle_s": self.training_idle_s,
            "pending_training": self._pending_training,
            "sampling_backlog": self._sampling_backlog,
            "train_lane_busy_s": self._lane_busy_s,
            "train_lane_utilization": (
                self._lane_busy_s / max(self.clock() - self._start_s, 1e-9)
            ),
        }
