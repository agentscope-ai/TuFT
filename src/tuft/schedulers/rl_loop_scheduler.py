from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Literal


class FastLoopStatus(str, Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class TenantLoopProfile:
    """Profile inferred for one tenant RL loop.

    ``ready_time_s`` is the natural time at which the tenant's next training
    burst becomes ready if no sampling delay is inserted. ``staleness_window_s``
    is the largest one-shot sampling delay the scheduler may add before the
    tenant would consume rollouts that are too stale.
    """

    tenant_id: str
    period_s: float
    training_burst_s: float
    ready_time_s: float
    staleness_window_s: float

    def __post_init__(self) -> None:
        if self.period_s <= 0:
            raise ValueError("period_s must be positive")
        if self.training_burst_s < 0:
            raise ValueError("training_burst_s must be non-negative")
        if self.staleness_window_s < 0:
            raise ValueError("staleness_window_s must be non-negative")


@dataclass(frozen=True)
class TrainingAssignment:
    tenant_id: str
    sampling_delay_s: float
    train_start_s: float
    train_end_s: float


@dataclass(frozen=True)
class FastLoopPlan:
    status: FastLoopStatus
    assignments: tuple[TrainingAssignment, ...]
    infeasible_tenant_id: str | None = None
    required_delay_s: float | None = None
    staleness_window_s: float | None = None
    # Tenants whose alignment delay exceeded W_i and therefore ran at their
    # natural (earliest legal) time instead of the aligned window start.
    align_fallbacks: int = 0
    aligned: int = 0

    @property
    def feasible(self) -> bool:
        return self.status == FastLoopStatus.FEASIBLE


class ContiguousTrainingScheduler:
    """Release-ordered greedy packing for TuFT contiguous training.

    This implements Algorithm 1 from the design text: order tenants by natural
    training ready time, tuck every burst against the previous one on a single
    logical training lane, and reject the plan if the one-shot sampling delay
    exceeds the tenant's inferred staleness window.
    """

    def plan_fast_loop(
        self,
        profiles: Iterable[TenantLoopProfile],
        *,
        pack_training: bool = False,
        pack_horizon_s: float = 2.0,
        align_period_s: float | None = None,
        align_phase_s: float = 0.0,
        align_window_s: float | None = None,
    ) -> FastLoopPlan:
        ordered = sorted(profiles, key=lambda profile: (profile.ready_time_s, profile.tenant_id))
        lane_free_s = -math.inf
        assignments: list[TrainingAssignment] = []

        align_fallbacks = 0
        aligned = 0
        for idx, profile in enumerate(ordered):
            ready = profile.ready_time_s
            # Phase alignment: shift this tenant's loop so its training becomes
            # ready inside a scheduled training window. The shift is applied
            # before sampling, so rollouts stay fresh; the bound W_i is the
            # tenant's delay budget.
            if align_period_s and align_period_s > 0:
                offset = (ready - align_phase_s) % align_period_s
                in_window = align_window_s is None or offset < align_window_s
                if not in_window:
                    target = ready + (align_period_s - offset)
                    if target - ready <= profile.staleness_window_s:
                        ready = target
                        aligned += 1
                    else:
                        align_fallbacks += 1
            # Optional packing: when the lane would idle before this burst and the
            # next burst is close, hold this (early-ready) burst until the next
            # ready time so the two run back-to-back, merging scattered idle into
            # one larger flex-sampling window. Bounded by the staleness window.
            if pack_training and ready > lane_free_s and idx + 1 < len(ordered):
                nxt = ordered[idx + 1]
                gap = nxt.ready_time_s - ready
                if 0 < gap <= pack_horizon_s and gap <= profile.staleness_window_s:
                    ready = nxt.ready_time_s
            train_start_s = max(ready, lane_free_s)
            sampling_delay_s = train_start_s - profile.ready_time_s
            if sampling_delay_s > profile.staleness_window_s:
                return FastLoopPlan(
                    status=FastLoopStatus.INFEASIBLE,
                    assignments=tuple(assignments),
                    infeasible_tenant_id=profile.tenant_id,
                    required_delay_s=sampling_delay_s,
                    staleness_window_s=profile.staleness_window_s,
                    align_fallbacks=align_fallbacks,
                    aligned=aligned,
                )
            train_end_s = train_start_s + profile.training_burst_s
            assignments.append(
                TrainingAssignment(
                    tenant_id=profile.tenant_id,
                    sampling_delay_s=sampling_delay_s,
                    train_start_s=train_start_s,
                    train_end_s=train_end_s,
                )
            )
            lane_free_s = train_end_s

        return FastLoopPlan(
            status=FastLoopStatus.FEASIBLE,
            assignments=tuple(assignments),
            align_fallbacks=align_fallbacks,
            aligned=aligned,
        )


@dataclass(frozen=True)
class LoopObservation:
    """Raw per-iteration signals used by loop discovery."""

    tenant_id: str
    sampling_submitted_s: float
    training_ready_s: float
    training_started_s: float
    training_finished_s: float
    staleness_window_s: float

    def __post_init__(self) -> None:
        if self.training_finished_s < self.training_started_s:
            raise ValueError("training_finished_s must be >= training_started_s")
        if self.staleness_window_s < 0:
            raise ValueError("staleness_window_s must be non-negative")


@dataclass
class _TenantLoopEstimate:
    tenant_id: str
    period_s: float | None = None
    training_burst_s: float | None = None
    ready_time_s: float | None = None
    staleness_window_s: float | None = None
    last_ready_s: float | None = None


@dataclass
class TenantLoopProfiler:
    """Online loop discovery for the fast-loop scheduler.

    The profiler intentionally uses only scheduler-visible API timing signals.
    It keeps exponentially smoothed estimates so drift can be absorbed without
    analytically predicting how composition changes affect burst duration.
    """

    smoothing_alpha: float = 0.3
    min_observations_for_period: int = 2
    _estimates: dict[str, _TenantLoopEstimate] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if self.min_observations_for_period < 1:
            raise ValueError("min_observations_for_period must be positive")

    def record(self, observation: LoopObservation) -> None:
        estimate = self._estimates.setdefault(
            observation.tenant_id,
            _TenantLoopEstimate(tenant_id=observation.tenant_id),
        )
        count = self._counts.get(observation.tenant_id, 0)
        if estimate.last_ready_s is not None:
            period_s = observation.training_ready_s - estimate.last_ready_s
            if period_s > 0:
                estimate.period_s = self._smooth(estimate.period_s, period_s)
        training_burst_s = observation.training_finished_s - observation.training_started_s
        estimate.training_burst_s = self._smooth(estimate.training_burst_s, training_burst_s)
        estimate.staleness_window_s = self._smooth(
            estimate.staleness_window_s,
            observation.staleness_window_s,
        )
        estimate.last_ready_s = observation.training_ready_s
        if estimate.period_s is not None:
            estimate.ready_time_s = observation.training_ready_s + estimate.period_s
        else:
            estimate.ready_time_s = observation.training_ready_s
        self._counts[observation.tenant_id] = count + 1

    def profiles(self) -> tuple[TenantLoopProfile, ...]:
        profiles: list[TenantLoopProfile] = []
        for tenant_id, estimate in sorted(self._estimates.items()):
            if self._counts.get(tenant_id, 0) < self.min_observations_for_period:
                continue
            if (
                estimate.period_s is None
                or estimate.training_burst_s is None
                or estimate.ready_time_s is None
                or estimate.staleness_window_s is None
            ):
                continue
            profiles.append(
                TenantLoopProfile(
                    tenant_id=tenant_id,
                    period_s=estimate.period_s,
                    training_burst_s=estimate.training_burst_s,
                    ready_time_s=estimate.ready_time_s,
                    staleness_window_s=estimate.staleness_window_s,
                )
            )
        return tuple(profiles)

    def _smooth(self, old: float | None, new: float) -> float:
        if old is None:
            return new
        alpha = self.smoothing_alpha
        return alpha * new + (1.0 - alpha) * old


@dataclass(frozen=True)
class SamplingRequest:
    """One sampling request before L1/L2/L3 sampling-side scheduling."""

    request_id: str
    tenant_id: str
    adapter_id: str
    adapter_version: int
    arrival_seq: int
    estimated_duration_s: float
    prompt_tokens: int = 0
    max_response_tokens: int = 0
    base_equivalent: bool | None = None

    def __post_init__(self) -> None:
        if self.adapter_version < 0:
            raise ValueError("adapter_version must be non-negative")
        if self.estimated_duration_s < 0:
            raise ValueError("estimated_duration_s must be non-negative")
        if self.arrival_seq < 0:
            raise ValueError("arrival_seq must be non-negative")

    @property
    def adapter_key(self) -> str:
        if self.base_equivalent is not None:
            is_base = self.base_equivalent
        else:
            is_base = self.adapter_version == 0
        return "A0" if is_base else self.adapter_id


@dataclass(frozen=True)
class AdapterRun:
    adapter_key: str
    requests: tuple[SamplingRequest, ...]
    estimated_duration_s: float
    first_arrival_seq: int

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.requests)


class SamplingGroupKind(str, Enum):
    FIXED = "fixed"
    FLEX = "flex"


@dataclass(frozen=True)
class SamplingGroupState:
    group_id: str
    kind: SamplingGroupKind
    available_at_s: float = 0.0
    flex_flip_at_s: float | None = None

    @property
    def horizon_s(self) -> float:
        if self.kind == SamplingGroupKind.FIXED:
            return math.inf
        if self.flex_flip_at_s is None:
            return self.available_at_s
        return self.flex_flip_at_s


@dataclass(frozen=True)
class RoutedRun:
    run: AdapterRun
    group_id: str
    start_s: float
    finish_s: float
    pinned: bool


@dataclass(frozen=True)
class SamplingSchedule:
    runs: tuple[AdapterRun, ...]
    routed_runs: tuple[RoutedRun, ...]
    rejected_runs: tuple[AdapterRun, ...] = ()


@dataclass
class SamplingSideScheduler:
    """Sampling-side L1/L2/L3 scheduler.

    L1 merges base-equivalent adapter-version-0 requests into pseudo-adapter A0.
    L2 reorders requests into stable same-adapter runs. L3 routes whole runs to
    sampling groups, preferring the adapter's pinned group when it can finish
    before the group's horizon. Flex groups therefore quiesce naturally as their
    scheduled flip approaches.
    """

    adapter_pins: dict[str, str] = field(default_factory=dict)

    def build_runs(self, requests: Iterable[SamplingRequest]) -> tuple[AdapterRun, ...]:
        buckets: dict[str, list[SamplingRequest]] = defaultdict(list)
        for request in sorted(requests, key=lambda item: item.arrival_seq):
            buckets[request.adapter_key].append(request)

        runs = [
            AdapterRun(
                adapter_key=adapter_key,
                requests=tuple(bucket),
                estimated_duration_s=sum(request.estimated_duration_s for request in bucket),
                first_arrival_seq=bucket[0].arrival_seq,
            )
            for adapter_key, bucket in buckets.items()
        ]
        return tuple(sorted(runs, key=lambda run: run.first_arrival_seq))

    def route_runs(
        self,
        runs: Iterable[AdapterRun],
        groups: Iterable[SamplingGroupState],
        *,
        now_s: float = 0.0,
    ) -> SamplingSchedule:
        group_available = {group.group_id: max(group.available_at_s, now_s) for group in groups}
        group_by_id = {group.group_id: group for group in groups}
        routed: list[RoutedRun] = []
        rejected: list[AdapterRun] = []
        run_tuple = tuple(runs)

        for run in run_tuple:
            choice = self._choose_group(run, group_by_id, group_available)
            if choice is None:
                rejected.append(run)
                continue
            group_id, pinned = choice
            start_s = group_available[group_id]
            finish_s = start_s + run.estimated_duration_s
            group_available[group_id] = finish_s
            self.adapter_pins[run.adapter_key] = group_id
            routed.append(
                RoutedRun(
                    run=run,
                    group_id=group_id,
                    start_s=start_s,
                    finish_s=finish_s,
                    pinned=pinned,
                )
            )

        return SamplingSchedule(
            runs=run_tuple,
            routed_runs=tuple(routed),
            rejected_runs=tuple(rejected),
        )

    def schedule(
        self,
        requests: Iterable[SamplingRequest],
        groups: Iterable[SamplingGroupState],
        *,
        now_s: float = 0.0,
    ) -> SamplingSchedule:
        runs = self.build_runs(requests)
        return self.route_runs(runs, groups, now_s=now_s)

    def _choose_group(
        self,
        run: AdapterRun,
        group_by_id: dict[str, SamplingGroupState],
        group_available: dict[str, float],
    ) -> tuple[str, bool] | None:
        pinned_group_id = self.adapter_pins.get(run.adapter_key)
        if pinned_group_id is not None and pinned_group_id in group_by_id:
            if self._can_fit(run, group_by_id[pinned_group_id], group_available[pinned_group_id]):
                return pinned_group_id, True

        candidates = [
            group
            for group in group_by_id.values()
            if self._can_fit(run, group, group_available[group.group_id])
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda group: (
                group.kind != SamplingGroupKind.FLEX,
                group_available[group.group_id],
                group.group_id,
            )
        )
        return candidates[0].group_id, False

    def _can_fit(self, run: AdapterRun, group: SamplingGroupState, start_s: float) -> bool:
        finish_s = start_s + run.estimated_duration_s
        return finish_s <= group.horizon_s


class CompositionAction(str, Enum):
    HOLD = "hold"
    FIXED_TO_FLEX = "fixed_to_flex"
    FLEX_TO_FIXED = "flex_to_fixed"


@dataclass(frozen=True)
class SlowLoopConfig:
    high_pressure_threshold: float = 0.75
    low_pressure_threshold: float = 0.35
    smoothing_alpha: float = 0.2
    min_dwell_s: float = 60.0
    fixed_to_flex_cost_s: float = 3.8
    flex_to_fixed_cost_s: float = 0.95
    min_expected_gain_s: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.low_pressure_threshold < self.high_pressure_threshold:
            raise ValueError("thresholds must satisfy 0 < low < high")
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if self.min_dwell_s < 0:
            raise ValueError("min_dwell_s must be non-negative")


@dataclass(frozen=True)
class CompositionState:
    fixed_groups: int
    flex_groups: int
    last_conversion_s: float = -math.inf

    def __post_init__(self) -> None:
        if self.fixed_groups < 0 or self.flex_groups < 0:
            raise ValueError("group counts must be non-negative")
        if self.flex_groups < 1:
            raise ValueError("at least one flex group must remain training-capable")


@dataclass(frozen=True)
class SlowLoopDecision:
    action: CompositionAction
    smoothed_pressure: float
    expected_gain_s: float = 0.0
    reason: str = ""


@dataclass
class FixedFlexCompositionController:
    """Slow-loop hysteresis for Fixed<->Flex composition."""

    config: SlowLoopConfig = field(default_factory=SlowLoopConfig)
    smoothed_pressure: float = 0.0

    def observe_pressure(self, training_demand_s: float, period_s: float) -> float:
        if period_s <= 0:
            raise ValueError("period_s must be positive")
        instant = max(0.0, training_demand_s / period_s)
        alpha = self.config.smoothing_alpha
        self.smoothed_pressure = alpha * instant + (1.0 - alpha) * self.smoothed_pressure
        return self.smoothed_pressure

    def decide(
        self,
        state: CompositionState,
        *,
        now_s: float,
        expected_gain_s: float,
    ) -> SlowLoopDecision:
        if now_s - state.last_conversion_s < self.config.min_dwell_s:
            return SlowLoopDecision(
                action=CompositionAction.HOLD,
                smoothed_pressure=self.smoothed_pressure,
                expected_gain_s=expected_gain_s,
                reason="minimum dwell time has not elapsed",
            )

        required_gain = self._required_gain()
        if expected_gain_s < required_gain:
            return SlowLoopDecision(
                action=CompositionAction.HOLD,
                smoothed_pressure=self.smoothed_pressure,
                expected_gain_s=expected_gain_s,
                reason="expected gain does not amortize conversion cost",
            )

        if (
            self.smoothed_pressure >= self.config.high_pressure_threshold
            and state.fixed_groups > 0
        ):
            return SlowLoopDecision(
                action=CompositionAction.FIXED_TO_FLEX,
                smoothed_pressure=self.smoothed_pressure,
                expected_gain_s=expected_gain_s,
                reason="sustained high training pressure",
            )
        if (
            self.smoothed_pressure <= self.config.low_pressure_threshold
            and state.flex_groups > 1
        ):
            return SlowLoopDecision(
                action=CompositionAction.FLEX_TO_FIXED,
                smoothed_pressure=self.smoothed_pressure,
                expected_gain_s=expected_gain_s,
                reason="sustained low training pressure",
            )
        return SlowLoopDecision(
            action=CompositionAction.HOLD,
            smoothed_pressure=self.smoothed_pressure,
            expected_gain_s=expected_gain_s,
            reason="pressure is inside hysteresis band or no legal group is available",
        )

    def _required_gain(self) -> float:
        return max(
            self.config.fixed_to_flex_cost_s,
            self.config.flex_to_fixed_cost_s,
            self.config.min_expected_gain_s,
        )


def apply_composition_action(
    state: CompositionState,
    action: CompositionAction,
    *,
    now_s: float,
) -> CompositionState:
    if action == CompositionAction.FIXED_TO_FLEX:
        if state.fixed_groups <= 0:
            raise ValueError("no fixed group can be converted to flex")
        return CompositionState(
            fixed_groups=state.fixed_groups - 1,
            flex_groups=state.flex_groups + 1,
            last_conversion_s=now_s,
        )
    if action == CompositionAction.FLEX_TO_FIXED:
        if state.flex_groups <= 1:
            raise ValueError("at least one flex group must remain training-capable")
        return CompositionState(
            fixed_groups=state.fixed_groups + 1,
            flex_groups=state.flex_groups - 1,
            last_conversion_s=now_s,
        )
    return state


RoutingBackendName = Literal["fixed", "flex"]
