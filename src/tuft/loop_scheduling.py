"""Bounded training-loop telemetry for conservative borrowing.

Times are monotonic seconds. A loop observation closes only on a successful
optimizer operation; execution time excludes queueing, phase changes and I/O.
No GPU placement or parameter storage is managed here.
"""

from __future__ import annotations

import statistics
from collections import OrderedDict, deque
from dataclasses import dataclass, field


@dataclass
class LoopState:
    durations: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    ready: float | None = None
    busy_s: float = 0.0
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

    def observe(self, run: str, now: float) -> None:
        """Record one completed update's original readiness (also useful in tests)."""
        self._state(run)
        history = self.arrivals[run]
        if not history or now > history[-1]:
            history.append(now)

    def training_ready(self, run: str, now: float) -> None:
        state = self._state(run)
        if state.ready is None:
            state.ready = now

    def executed(self, run: str, seconds: float, *, commit: bool) -> None:
        state = self._state(run)
        state.busy_s += seconds
        if not commit:
            return
        if state.ready is not None:
            self.observe(run, state.ready)
            state.durations.append(state.busy_s)
            state.observations += 1
        state.ready = None
        state.busy_s = 0.0

    def period(self, run: str) -> float | None:
        history = list(self.arrivals[run])
        if len(history) < 4:
            return None
        intervals = [b - a for a, b in zip(history, history[1:], strict=False)]
        if not intervals:
            return None
        period = statistics.median(intervals)
        if period <= 0 or max(abs(p - period) for p in intervals) > 0.2 * period:
            return None
        return period

    def status(self, now: float) -> dict[str, int]:
        """Explain eligibility without exposing run identifiers or changing history."""
        counts = dict(warming=0, irregular=0, pending=0, overdue=0, eligible=0)
        for run, history in self.arrivals.items():
            state = self.states[run]
            period = self.period(run)
            if state.ready is not None:
                reason = "pending"
            elif len(history) < 4 or not state.durations:
                reason = "warming"
            elif period is None:
                reason = "irregular"
            elif history[-1] + period <= now:
                reason = "overdue"
            else:
                reason = "eligible"
            counts[reason] += 1
        return counts

    def horizon(self, now: float) -> float | None:
        predictions = []
        for run, history in self.arrivals.items():
            period = self.period(run)
            if period is not None:
                # This is admission guidance; an early update still drains
                # all admitted sampling before touching the training GPU.
                predicted = history[-1] + period
                if predicted > now:
                    predictions.append(predicted)
        return min(predictions) if predictions else None
