"""Bounded loop telemetry and a pure, release-ordered sampling-delay planner.

Times are monotonic seconds. A loop observation closes only on a successful
optimizer operation; execution time excludes queueing, phase changes and I/O.
No GPU placement or parameter storage is managed here.
"""

from __future__ import annotations

import statistics
from collections import OrderedDict, deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoopProfile:
    run: str
    ready: float
    duration: float
    budget: float


@dataclass(frozen=True)
class TrainingSlot:
    run: str
    start: float
    end: float
    delay: float


def plan_training(profiles: list[LoopProfile], lookahead_s: float) -> list[TrainingSlot] | None:
    """Pack nearby bursts; reject any plan whose *total* shift exceeds a budget.

    Moving an early burst to end at the next release consolidates an idle gap
    without delaying the later burst. Budgets include any lane contention.
    Return None when even the natural release order violates a budget.
    """
    ordered = sorted(profiles, key=lambda p: (p.ready, p.run))
    shifts = {p.run: 0.0 for p in ordered}

    def check() -> list[TrainingSlot] | None:
        end = float("-inf")
        slots = []
        for p in ordered:
            start = max(p.ready + shifts[p.run], end)
            if start - p.ready > p.budget + 1e-9:
                return None
            end = start + p.duration
            slots.append(TrainingSlot(p.run, start, end, max(0.0, start - p.ready)))
        return slots

    result = check()
    if result is None:
        return None
    # Work backwards so moving a later burst cannot reopen a gap in a cluster
    # that was already packed. Every candidate still checks the entire lane.
    for i in range(len(ordered) - 2, -1, -1):
        p = ordered[i]
        gap = result[i + 1].start - result[i].end
        if 0 < gap <= lookahead_s:
            old = shifts[p.run]
            shifts[p.run] = result[i].delay + gap
            candidate = check()
            if candidate is None:
                shifts[p.run] = old
            else:
                result = candidate
    return result


@dataclass
class LoopState:
    durations: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    # One delay adjustment for each interval between completed-update arrivals.
    delays: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    ready: float | None = None
    busy_s: float = 0.0
    sampling_started: float | None = None
    hold_until: float = 0.0
    planned_delay: float = 0.0
    generation: int = 0
    observations: int = 0


class LoopProfiler:
    """Recent completed updates, bounded by active-run count and history size."""

    def __init__(self, max_tenants: int = 1024) -> None:
        self.max_tenants = max_tenants
        self.arrivals: OrderedDict[str, deque[float]] = OrderedDict()
        self.states: dict[str, LoopState] = {}

    def _state(self, run: str) -> LoopState:
        self.arrivals.setdefault(run, deque(maxlen=9))
        self.arrivals.move_to_end(run)
        state = self.states.setdefault(run, LoopState())
        while len(self.arrivals) > self.max_tenants:
            old, _ = self.arrivals.popitem(last=False)
            del self.states[old]
        return state

    def forget(self, run: str) -> None:
        self.arrivals.pop(run, None)
        self.states.pop(run, None)

    def observe(self, run: str, now: float, delay: float = 0.0) -> None:
        """Record one completed update's original readiness (also useful in tests)."""
        state = self._state(run)
        history = self.arrivals[run]
        if history and now > history[-1]:
            state.delays.append(delay)
        if not history or now > history[-1]:
            history.append(now)

    def training_ready(self, run: str, now: float) -> None:
        state = self._state(run)
        if state.ready is None:
            if now < state.hold_until:
                # This update did not depend on the held sampling wave. Do not
                # learn a fictitious scheduling delay from that failed prediction.
                state.planned_delay = 0.0
                state.hold_until = 0.0
            state.ready = now

    def executed(self, run: str, seconds: float, *, commit: bool) -> None:
        state = self._state(run)
        state.busy_s += seconds
        if not commit:
            return
        if state.ready is not None:
            self.observe(run, state.ready, state.planned_delay)
            state.durations.append(state.busy_s)
            state.observations += 1
        state.generation += 1
        state.ready = None
        state.busy_s = 0.0
        state.sampling_started = None
        state.hold_until = 0.0
        state.planned_delay = 0.0

    def period(self, run: str) -> float | None:
        history = list(self.arrivals[run])
        if len(history) < 4:
            return None
        delays = list(self.states[run].delays)
        intervals = [b - a - d for a, b, d in zip(history, history[1:], delays, strict=False)]
        if not intervals:
            return None
        period = statistics.median(intervals)
        if period <= 0 or max(abs(p - period) for p in intervals) > 0.2 * period:
            return None
        return period

    def profiles(self, now: float, max_delay: float, fraction: float) -> list[LoopProfile]:
        profiles = []
        for run, history in self.arrivals.items():
            period = self.period(run)
            state = self.states[run]
            if period is None or not state.durations or state.ready is not None:
                continue
            ready = history[-1] + period + state.planned_delay
            if ready <= now:
                continue
            budget = min(max_delay, fraction * period) if state.sampling_started is None else 0.0
            profiles.append(LoopProfile(run, ready, max(state.durations), budget))
        return profiles

    def horizon(self, now: float) -> float | None:
        predictions = []
        for run, history in self.arrivals.items():
            period = self.period(run)
            if period is not None:
                # Never rely on a speculative shift to admit GPU work. Applied
                # holds are advisory and a client can submit an update early.
                predicted = history[-1] + period
                if predicted > now:
                    predictions.append(predicted)
        return min(predictions) if predictions else None

    def sampling_gate(
        self, run: str, now: float, *, max_delay: float, fraction: float, lookahead: float
    ) -> tuple[float, bool]:
        """One shared deadline for the entire wave, never extended by arrivals.

        Recompute on the first sample after each commit. Already-started waves
        have zero shift budget; infeasibility simply disables alignment.
        """
        state = self._state(run)
        infeasible = False
        if state.sampling_started is None:
            plan = plan_training(self.profiles(now, max_delay, fraction), lookahead)
            infeasible = plan is None
            delay = next((s.delay for s in plan or [] if s.run == run), 0.0)
            state.sampling_started = now
            state.hold_until = now + delay
            state.planned_delay = delay
        return state.hold_until, infeasible
