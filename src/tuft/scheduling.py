"""Bounded scheduling over existing TuFT engines; no changes to model numerics.

One event loop owns all bookkeeping. Reservations happen before dispatch, and
remain held until backend completion even if the requesting client disconnects.
The optional flex replica shares a GPU with training: admission closes as soon
as training is queued, and training starts only after admitted samples drain.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from .config import SchedulingConfig
from .exceptions import ResourceExhaustedException


T = TypeVar("T")


class SchedulingOverloadedError(ResourceExhaustedException):
    """The bounded admission queue is full or its wait budget expired."""


@dataclass
class _Request:
    tenant: str
    adapter: str | None
    work: int
    flex_eligible: bool
    operation: Callable[[int], Awaitable[Any]]
    future: asyncio.Future
    submitted: float
    admitted: asyncio.Event
    aligned: bool = False


class LoopProfiler:
    """Conservative median predictor; irregular or stale loops provide no horizon."""

    def __init__(self, max_tenants: int = 1024) -> None:
        self.max_tenants = max_tenants
        self.arrivals: OrderedDict[str, deque[float]] = OrderedDict()

    def observe(self, tenant: str, now: float) -> None:
        history = self.arrivals.setdefault(tenant, deque(maxlen=9))
        if not history or now > history[-1]:
            history.append(now)
        self.arrivals.move_to_end(tenant)
        while len(self.arrivals) > self.max_tenants:
            self.arrivals.popitem(last=False)

    def horizon(self, now: float) -> float | None:
        predictions = []
        for history in self.arrivals.values():
            if len(history) < 4:
                continue
            intervals = [b - a for a, b in zip(history, list(history)[1:], strict=False)]
            period = statistics.median(intervals)
            if period <= 0 or max(abs(p - period) for p in intervals) > 0.2 * period:
                continue
            # Never extrapolate indefinitely across disappeared/paused tenants.
            predicted = history[-1] + period
            if predicted > now:
                predictions.append(predicted)
        return min(predictions) if predictions else None


class ResourceScheduler:
    def __init__(
        self, replicas: int, config: SchedulingConfig, *, flex_index: int | None = None
    ) -> None:
        if replicas < 1 or (flex_index is not None and not 0 <= flex_index < replicas):
            raise ValueError("Invalid replica layout")
        if flex_index is not None and replicas < 2:
            raise ValueError("A flex replica requires persistent sampling capacity")
        self.config = config
        self.flex_index = flex_index
        self._work = [0] * replicas
        self._active = [0] * replicas
        self._adapter: list[str | None] = [None] * replicas
        self._queues: dict[str, deque[_Request]] = {}
        self._tenants: deque[str] = deque()
        self._queued = 0
        self._tasks: set[asyncio.Task] = set()
        self._training_lock = asyncio.Lock()
        self._training_pending = 0
        self._training_active = False
        self._flex_drained = asyncio.Event()
        self._flex_drained.set()
        self._quiet_until = 0.0
        self._wake_handle: asyncio.TimerHandle | None = None
        self._closed = False
        self._cursor = 0
        self._stats: dict[str, float] = defaultdict(float)
        self._started = time.monotonic()
        self.profiler = LoopProfiler()
        self._seconds_per_token: float | None = None
        self._flex_busy_since = 0.0
        self._flex_busy_work = 0
        self._flex_busy_failed = False

    def snapshot(self) -> dict[str, Any]:
        """Bounded aggregate metrics, without tenant identifiers."""
        return {
            **self._stats,
            "elapsed_s": time.monotonic() - self._started,
            "queued": self._queued,
            "active": list(self._active),
            "reserved_tokens": list(self._work),
            "training_pending": self._training_pending,
            "training_active": self._training_active,
            "profiled_tenants": len(self.profiler.arrivals),
        }

    def _track(self, coro: Awaitable) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        # A disconnected client must not leave an unobserved backend exception.
        task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        return task

    async def sample(
        self,
        tenant: str,
        adapter: str | None,
        work: int,
        flex_eligible: bool,
        operation: Callable[[int], Awaitable[T]],
    ) -> T:
        if self._closed:
            raise RuntimeError("Scheduler is closed")
        if self._queued >= self.config.max_queued_requests:
            self._stats["rejected"] += 1
            raise SchedulingOverloadedError("Sampling admission queue is full")
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        request = _Request(
            tenant,
            adapter,
            max(1, work),
            flex_eligible,
            operation,
            future,
            time.monotonic(),
            asyncio.Event(),
        )
        if tenant not in self._queues:
            self._queues[tenant] = deque()
            self._tenants.append(tenant)
        self._queues[tenant].append(request)
        self._queued += 1
        self._stats["peak_queued"] = max(self._stats["peak_queued"], self._queued)
        self._pump()
        waiting_for_admission = True
        try:
            # Only queueing has a deadline. Never cancel running GPU work on a
            # queue timeout, nor release its reservation on caller cancellation.
            await asyncio.wait_for(request.admitted.wait(), self.config.queue_timeout_s)
            waiting_for_admission = False
            return await asyncio.shield(future)
        except TimeoutError as exc:
            if not waiting_for_admission:
                raise
            self._stats["queue_timeouts"] += 1
            future.cancel()
            raise SchedulingOverloadedError("Sampling admission wait expired") from exc
        except asyncio.CancelledError:
            future.cancel()
            raise
        finally:
            if future.cancelled():
                self._remove_queued(request)
                self._pump()

    def _remove_queued(self, request: _Request) -> None:
        queue = self._queues.get(request.tenant)
        if queue is not None and request in queue:
            queue.remove(request)
            self._queued -= 1
            if not queue:
                del self._queues[request.tenant]
                self._tenants.remove(request.tenant)

    def _choose(self, request: _Request, now: float, horizon: float | None) -> int | None:
        if (
            self.flex_index is not None
            and request.flex_eligible
            and (self._training_pending or self._training_active or now < self._quiet_until)
            and now < request.submitted + self.config.sampling_alignment_s
        ):
            # Without this bounded delay, the next rollout wave is irrevocably
            # sent to fixed replicas during the post-update quiet interval, and
            # the training GPU remains idle once that interval ends.
            if not request.aligned:
                request.aligned = True
                self._stats["aligned_requests"] += 1
            self._schedule_wake(request.submitted + self.config.sampling_alignment_s - now)
            return None
        candidates = []
        for index, active in enumerate(self._active):
            limit = self.config.max_inflight_per_replica
            if index == self.flex_index:
                limit = self.config.flex_max_inflight
                if (
                    not request.flex_eligible
                    or self._training_pending
                    or self._training_active
                    or now < self._quiet_until
                ):
                    continue
                if horizon is not None and self._seconds_per_token is not None:
                    # Include already reserved work and a safety factor. This
                    # prediction is admission guidance, never a preemption promise;
                    # unexpected training still drains actual admitted requests.
                    finish = now + 1.5 * self._seconds_per_token * (
                        self._work[index] + request.work
                    )
                    if finish >= horizon:
                        self._stats["window_rejections"] += 1
                        continue
            if active < limit:
                candidates.append(index)
        if not candidates:
            return None
        # Rotate ties; prefer locality only within a bounded extra-work budget.
        best = min(candidates, key=lambda i: (self._work[i], (i - self._cursor) % len(self._work)))
        local = [
            i
            for i in candidates
            if self._adapter[i] == request.adapter
            and self._work[i] <= self._work[best] + request.work * self.config.locality_slack
        ]
        if local:
            best = min(local, key=lambda i: (self._work[i], (i - self._cursor) % len(self._work)))
        self._cursor = (best + 1) % len(self._work)
        return best

    def _wake(self) -> None:
        self._wake_handle = None
        self._pump()

    def _schedule_wake(self, delay: float) -> None:
        loop = asyncio.get_running_loop()
        when = loop.time() + max(0, delay)
        if self._wake_handle is not None:
            if self._wake_handle.when() <= when:
                return
            self._wake_handle.cancel()
        self._wake_handle = loop.call_at(when, self._wake)

    def _pump(self) -> None:
        if self._closed or not self._tenants:
            return
        now = time.monotonic()
        horizon = self.profiler.horizon(now) if self.flex_index is not None else None
        # Round-robin tenant heads: one tenant cannot monopolize pending slots.
        stalled = 0
        while self._tenants and stalled < len(self._tenants):
            # Do not scan every queued tenant on every submission to a full pool.
            if not any(
                active
                < (
                    self.config.flex_max_inflight
                    if i == self.flex_index
                    else self.config.max_inflight_per_replica
                )
                and (
                    i != self.flex_index
                    or (
                        not self._training_pending
                        and not self._training_active
                        and now >= self._quiet_until
                    )
                )
                for i, active in enumerate(self._active)
            ):
                break
            tenant = self._tenants[0]
            request = self._queues[tenant][0]
            index = self._choose(request, now, horizon)
            self._tenants.rotate(-1)
            if index is None:
                stalled += 1
                continue
            stalled = 0
            self._remove_queued(request)
            if index == self.flex_index:
                if not self._active[index]:
                    self._flex_busy_since = time.monotonic()
                    self._flex_busy_work = 0
                    self._flex_busy_failed = False
                self._flex_busy_work += request.work
            self._active[index] += 1
            self._work[index] += request.work
            self._adapter[index] = request.adapter
            if index == self.flex_index:
                self._flex_drained.clear()
            request.admitted.set()
            self._stats["queue_wait_s"] += time.monotonic() - request.submitted
            self._stats["admitted"] += 1
            self._track(self._execute(request, index))

    async def _execute(self, request: _Request, index: int) -> None:
        started = time.monotonic()
        try:
            result = await request.operation(index)
            if not request.future.done():
                request.future.set_result(result)
            self._stats["completed"] += 1
            self._stats[f"replica_{index}_completed"] += 1
        except BaseException as exc:
            self._stats["failed"] += 1
            if index == self.flex_index:
                self._flex_busy_failed = True
            if not request.future.done():
                if isinstance(exc, asyncio.CancelledError):
                    request.future.cancel()
                else:
                    request.future.set_exception(exc)
        finally:
            elapsed = time.monotonic() - started
            self._stats[f"replica_{index}_request_s"] += elapsed
            self._active[index] -= 1
            self._work[index] -= request.work
            if index == self.flex_index and not self._active[index]:
                # Concurrent requests share a decode batch. Summing individual
                # latencies counts the same GPU interval repeatedly and makes
                # a learned window reject useful work as concurrency increases.
                if not self._flex_busy_failed:
                    rate = (time.monotonic() - self._flex_busy_since) / self._flex_busy_work
                    self._seconds_per_token = (
                        rate
                        if self._seconds_per_token is None
                        else 0.2 * rate + 0.8 * self._seconds_per_token
                    )
                self._flex_drained.set()
            self._pump()

    async def training(
        self, operation: Callable[[], Awaitable[T]], *, tenant: str | None = None
    ) -> T:
        if self._closed:
            raise RuntimeError("Scheduler is closed")
        if tenant is not None:
            self.profiler.observe(tenant, time.monotonic())
        # Count readiness synchronously so another sample cannot sneak in first.
        self._training_pending += 1
        task = self._track(self._train(operation))
        # Once admitted, a mutation is irrevocable. Finish its caller's sequence
        # guard too: propagating cancellation here would permit a retry of an
        # optimizer update that the GPU has already committed.
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.done():
                    return task.result()

    async def _train(self, operation: Callable[[], Awaitable[T]]) -> T:
        waiting = time.monotonic()
        try:
            async with self._training_lock:
                await self._flex_drained.wait()
                self._training_active = True
                started = time.monotonic()
                self._stats["exclusive_wait_s"] += started - waiting
                try:
                    result = await operation()
                    self._stats["exclusive_completed"] += 1
                    return result
                finally:
                    self._stats["exclusive_busy_s"] += time.monotonic() - started
                    self._training_active = False
                    self._quiet_until = time.monotonic() + self.config.training_quiet_s
        finally:
            self._training_pending -= 1
            if not self._closed:
                self._schedule_wake(self._quiet_until - time.monotonic())
                self._pump()

    async def close(self) -> None:
        """Reject queued work and drain admitted work before releasing engines."""
        self._closed = True
        if self._wake_handle is not None:
            self._wake_handle.cancel()
        for queue in self._queues.values():
            for request in queue:
                if not request.future.done():
                    request.future.set_exception(RuntimeError("Scheduler is closed"))
                request.admitted.set()
        self._queues.clear()
        self._tenants.clear()
        self._queued = 0
        if self._tasks:
            draining = asyncio.gather(*list(self._tasks), return_exceptions=True)
            while not draining.done():
                try:
                    await asyncio.shield(draining)
                except asyncio.CancelledError:
                    pass
