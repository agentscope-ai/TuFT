from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, TypeVar


class SwitchMode(str, Enum):
    TRAINING = "training"
    SAMPLING = "sampling"


class SwitchPolicy(str, Enum):
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class SwitchSchedulerConfig:
    """Configuration for switch-based training/sampling schedulers.

    The scheduler implements the common baseline-3/4 semantics:
    1. drain the training queue;
    2. switch to sampling;
    3. keep sampling for at least ``sampling_min_window_s``;
    4. after the minimum window, keep sampling while no training is pending;
    5. once training is pending, switch back and repeat.

    ``policy=ADAPTIVE`` grows the minimum sampling window when recent switching
    overhead exceeds ``max_switch_overhead_fraction`` of elapsed scheduler time.
    """

    sampling_min_window_s: float = 3.0
    policy: SwitchPolicy = SwitchPolicy.FIXED
    max_switch_overhead_fraction: float = 0.10
    adaptive_growth_factor: float = 1.5
    adaptive_window_s: float = 60.0
    max_sampling_min_window_s: float | None = None
    idle_sleep_s: float = 0.001

    def __post_init__(self) -> None:
        if self.sampling_min_window_s < 0:
            raise ValueError("sampling_min_window_s must be non-negative")
        if not (0.0 < self.max_switch_overhead_fraction < 1.0):
            raise ValueError("max_switch_overhead_fraction must be in (0, 1)")
        if self.adaptive_growth_factor <= 1.0:
            raise ValueError("adaptive_growth_factor must be > 1")
        if self.adaptive_window_s <= 0:
            raise ValueError("adaptive_window_s must be positive")
        if self.max_sampling_min_window_s is not None:
            if self.max_sampling_min_window_s < self.sampling_min_window_s:
                raise ValueError("max_sampling_min_window_s must be >= sampling_min_window_s")
        if self.idle_sleep_s < 0:
            raise ValueError("idle_sleep_s must be non-negative")


@dataclass
class SwitchStats:
    training_jobs_completed: int = 0
    sampling_jobs_completed: int = 0
    switches_to_training: int = 0
    switches_to_sampling: int = 0
    total_switch_time_s: float = 0.0
    sampling_windows_completed: int = 0
    sampling_window_s: float = 0.0
    adaptive_adjustments: int = 0


@dataclass(frozen=True)
class SwitchSchedulerEvent:
    kind: str
    timestamp_s: float
    mode: SwitchMode
    detail: dict[str, Any] = field(default_factory=dict)


T = TypeVar("T")
Job = Callable[[], Awaitable[Any]] | Callable[[], Any]
SwitchCallback = Callable[[], Awaitable[Any]] | Callable[[], Any]
Clock = Callable[[], float]


async def _maybe_await(func: Job | SwitchCallback) -> Any:
    result = func()
    if asyncio.iscoroutine(result) or isinstance(result, Coroutine):
        return await result
    return result


class SwitchScheduler:
    """Queue-drain switch scheduler shared by baseline 3 and baseline 4.

    This class is intentionally backend-agnostic: users enqueue coroutine
    factories for training/sampling work and provide callbacks for the two mode
    switches. It can be embedded above TorchTP-only or FSDP/vLLM backends.
    """

    def __init__(
        self,
        config: SwitchSchedulerConfig,
        *,
        switch_to_training: SwitchCallback | None = None,
        switch_to_sampling: SwitchCallback | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self.mode = SwitchMode.TRAINING
        self.current_sampling_min_window_s = config.sampling_min_window_s
        self._switch_to_training = switch_to_training or (lambda: None)
        self._switch_to_sampling = switch_to_sampling or (lambda: None)
        self._clock = clock or time.monotonic
        self._training_queue: Deque[Job] = deque()
        self._sampling_queue: Deque[Job] = deque()
        self._stop_requested = False
        self.stats = SwitchStats(sampling_window_s=self.current_sampling_min_window_s)
        self.events: list[SwitchSchedulerEvent] = []
        self._recent_switches: Deque[tuple[float, float]] = deque()

    @property
    def training_queue_length(self) -> int:
        return len(self._training_queue)

    @property
    def sampling_queue_length(self) -> int:
        return len(self._sampling_queue)

    def enqueue_training(self, job: Job) -> None:
        self._training_queue.append(job)
        self._record("enqueue_training", queue_len=len(self._training_queue))

    def enqueue_sampling(self, job: Job) -> None:
        self._sampling_queue.append(job)
        self._record("enqueue_sampling", queue_len=len(self._sampling_queue))

    def request_stop(self) -> None:
        self._stop_requested = True

    async def run_until_idle(self, *, max_cycles: int | None = None) -> SwitchStats:
        """Run until both queues are empty or ``request_stop`` is called.

        This is the deterministic/evaluation-friendly runner used by unit tests
        and evaluation drivers. Server integration can instead call the smaller
        step methods later if it wants a long-lived background scheduler.
        """
        cycles = 0
        while not self._stop_requested:
            if max_cycles is not None and cycles >= max_cycles:
                break
            if not self._training_queue and not self._sampling_queue:
                break
            await self._drain_training_queue()
            await self._switch(SwitchMode.SAMPLING)
            await self._serve_sampling_window()
            if self._training_queue:
                await self._switch(SwitchMode.TRAINING)
            cycles += 1
        return self.stats

    async def _drain_training_queue(self) -> None:
        if self.mode != SwitchMode.TRAINING:
            await self._switch(SwitchMode.TRAINING)
        self._record("training_drain_start", queue_len=len(self._training_queue))
        while self._training_queue:
            job = self._training_queue.popleft()
            await _maybe_await(job)
            self.stats.training_jobs_completed += 1
            self._record("training_job_done", remaining=len(self._training_queue))
        self._record("training_drain_done")

    async def _serve_sampling_window(self) -> None:
        if self.mode != SwitchMode.SAMPLING:
            await self._switch(SwitchMode.SAMPLING)
        window_start = self._clock()
        self._record("sampling_window_start", min_window_s=self.current_sampling_min_window_s)
        while True:
            if self._sampling_queue:
                job = self._sampling_queue.popleft()
                await _maybe_await(job)
                self.stats.sampling_jobs_completed += 1
                self._record("sampling_job_done", remaining=len(self._sampling_queue))
                continue

            elapsed = self._clock() - window_start
            # Offline/evaluation runner: when both queues are empty there is no
            # useful work to keep the sampling window open for. A long-lived
            # server integration can wait for new requests outside this method.
            if not self._training_queue and not self._sampling_queue:
                break
            if elapsed < self.current_sampling_min_window_s:
                if self.config.idle_sleep_s:
                    await asyncio.sleep(self.config.idle_sleep_s)
                else:
                    await asyncio.sleep(0)
                continue

            # After the minimum window, stay in sampling while no training is
            # pending. If both queues are empty this runner is idle and can stop;
            # in a server integration, a background scheduler would wait for new
            # requests instead.
            if self._training_queue:
                break
            if not self._sampling_queue:
                break
        self.stats.sampling_windows_completed += 1
        self._record("sampling_window_done", elapsed_s=self._clock() - window_start)

    async def _switch(self, target: SwitchMode) -> None:
        if self.mode == target:
            return
        start = self._clock()
        if target == SwitchMode.TRAINING:
            await _maybe_await(self._switch_to_training)
            self.stats.switches_to_training += 1
        else:
            await _maybe_await(self._switch_to_sampling)
            self.stats.switches_to_sampling += 1
        duration = self._clock() - start
        self.mode = target
        self.stats.total_switch_time_s += duration
        self._recent_switches.append((self._clock(), duration))
        self._record("switch", target=target.value, duration_s=duration)
        self._maybe_adjust_sampling_window()

    def _maybe_adjust_sampling_window(self) -> None:
        if self.config.policy != SwitchPolicy.ADAPTIVE:
            return
        now = self._clock()
        cutoff = now - self.config.adaptive_window_s
        while self._recent_switches and self._recent_switches[0][0] < cutoff:
            self._recent_switches.popleft()
        if not self._recent_switches:
            return
        total_switch = sum(duration for _, duration in self._recent_switches)
        oldest = self._recent_switches[0][0]
        elapsed = max(now - oldest, 1e-9)
        fraction = total_switch / elapsed
        if fraction <= self.config.max_switch_overhead_fraction:
            return
        new_window = self.current_sampling_min_window_s * self.config.adaptive_growth_factor
        if self.config.max_sampling_min_window_s is not None:
            new_window = min(new_window, self.config.max_sampling_min_window_s)
        if new_window <= self.current_sampling_min_window_s:
            return
        old = self.current_sampling_min_window_s
        self.current_sampling_min_window_s = new_window
        self.stats.sampling_window_s = new_window
        self.stats.adaptive_adjustments += 1
        self._record(
            "adaptive_window_increase",
            old_window_s=old,
            new_window_s=new_window,
            switch_fraction=fraction,
        )

    def _record(self, kind: str, **detail: Any) -> None:
        self.events.append(
            SwitchSchedulerEvent(
                kind=kind,
                timestamp_s=self._clock(),
                mode=self.mode,
                detail=detail,
            )
        )
