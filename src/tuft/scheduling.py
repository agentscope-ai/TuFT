"""Bounded scheduling over existing TuFT engines; no changes to model numerics.

One event loop owns all bookkeeping. Reservations happen before dispatch, and
remain held until backend completion even if the requesting client disconnects.
The optional flex replica shares a GPU with training: admission closes as soon
as training is queued, and training starts only after admitted samples drain.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from .borrowing import BorrowingAdmission
from .config import SamplingConfig
from .exceptions import ResourceExhaustedException
from .mutation_queue import MutationQueue
from .sampling_routing import RoundRobinRouter


T = TypeVar("T")

_DISPATCH_LIMIT = 32
_TRAINING_QUIET_S = 0.05
_TRAINING_PRIORITY_BURST = 2


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
    prompt_tokens: int = 0
    output_tokens: int = 1
    sequences: int = 1
    service_started: float | None = None
    peak_sequences: int = 1
    trace_id: int = 0
    last_borrow_rejection: str | None = None


class ResourceScheduler:
    def __init__(
        self, replicas: int, config: SamplingConfig, *, flex_index: int | None = None
    ) -> None:
        if replicas < 1 or (flex_index is not None and not 0 <= flex_index < replicas):
            raise ValueError("Invalid replica layout")
        if flex_index is not None and replicas < 2:
            raise ValueError("A flex replica requires persistent sampling capacity")
        self.config = config
        self.router = RoundRobinRouter(replicas)
        self.borrowing = BorrowingAdmission(config.borrowing)
        self.flex_index = flex_index
        self._work = [0] * replicas
        self._active = [0] * replicas
        self._queues: dict[str, deque[_Request]] = {}
        self._tenants: deque[str] = deque()
        self._queued = 0
        self._tasks: set[asyncio.Task] = set()
        self._training_queue = MutationQueue(
            _TRAINING_PRIORITY_BURST if flex_index is not None else 0
        )
        self._training_pending = 0
        self._training_active = False
        self._flex_drained = asyncio.Event()
        self._flex_drained.set()
        self._quiet_until = 0.0
        self._wake_handle: asyncio.TimerHandle | None = None
        self._closed = False
        self._stats: dict[str, float] = defaultdict(float)
        self._started = time.monotonic()
        self._phase_transition: Callable[[bool], Awaitable[None]] | None = None
        self._phase_lock = asyncio.Lock()
        self._phase_state: bool | None = None
        self._flex_failed = False
        self._trace: deque[dict[str, Any]] = deque(maxlen=config.scheduling_trace_capacity)
        self._rejection_trace: deque[dict[str, Any]] = deque(
            maxlen=config.scheduling_trace_capacity
        )
        self._next_request_id = 0
        self._running: list[dict[int, _Request]] = [{} for _ in range(replicas)]

    def _record(self, event: str, **fields: Any) -> None:
        if self._trace.maxlen:
            self._trace.append(
                {"event": event, "elapsed_s": time.monotonic() - self._started, **fields}
            )

    def set_phase_transition(self, callback: Callable[[bool], Awaitable[None]]) -> None:
        """Install before serving; True releases sampler memory for training."""
        if self.flex_index is None or self._tasks:
            raise RuntimeError("Phase transitions require an idle flex scheduler")
        self._phase_transition = callback
        self._phase_state = None

    async def _enter_phase(self, training: bool) -> None:
        if self._flex_failed:
            raise RuntimeError("Shared GPU is unavailable; restart the backend")
        if self._phase_transition is None:
            return
        # A sampling batch can admit dozens of requests together. Coalesce the
        # wakeup in the driver, avoiding one Ray RPC for every sample.
        async with self._phase_lock:
            if self._flex_failed:
                raise RuntimeError("Shared GPU is unavailable; restart the backend")
            if self._phase_state is training:
                return
            self._phase_state = None
            started = time.monotonic()
            try:
                await self._phase_transition(training)
            except BaseException:
                self.disable_flex()
                raise
            elapsed = time.monotonic() - started
            self._stats["switch_to_training_s" if training else "switch_to_sampling_s"] += elapsed
            self._phase_state = training

    def disable_flex(self) -> None:
        """Stop shared storage use; dedicated sampling of existing snapshots survives."""
        self._flex_failed = True
        self._phase_state = None

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
            "flex_failed": self._flex_failed,
            "decision_trace": list(self._trace),
            "borrow_rejection_trace": list(self._rejection_trace),
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
        *,
        prompt_tokens: int = 0,
        output_tokens: int | None = None,
        num_samples: int = 1,
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
        self._next_request_id += 1
        request.trace_id = self._next_request_id
        request.prompt_tokens = max(0, prompt_tokens)
        request.output_tokens = max(1, output_tokens if output_tokens is not None else work)
        request.sequences = max(1, num_samples)
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
            if request.admitted.is_set():
                # Admission may win in the same event-loop turn as the timer.
                # The reservation now owns GPU work; report its actual result.
                return await asyncio.shield(future)
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

    def _reject_borrow(self, request: _Request, reason: str, now: float) -> None:
        """Count evaluated rejections; retain bounded reason transitions per request."""
        self._stats[f"borrow_rejected_{reason}"] += 1
        if not self._rejection_trace.maxlen or request.last_borrow_rejection == reason:
            return
        request.last_borrow_rejection = reason
        self._rejection_trace.append(
            dict(
                event="borrow_reject",
                elapsed_s=now - self._started,
                request_id=request.trace_id,
                reason=reason,
                queued=self._queued,
                active=list(self._active),
                training_pending=self._training_pending,
                training_active=self._training_active,
                prompt_tokens=request.prompt_tokens,
                output_tokens=request.output_tokens,
                sequences=request.sequences,
            )
        )

    def _dedicated_full(self) -> bool:
        return all(
            active >= self.config.max_inflight_per_replica
            for i, active in enumerate(self._active)
            if i != self.flex_index
        )

    def _borrow_rejection(self, *, eligible: bool, now: float) -> str | None:
        assert self.flex_index is not None
        return self.borrowing.unavailable_reason(
            failed=self._flex_failed,
            eligible=eligible,
            training=bool(self._training_pending or self._training_active),
            quiet=now < self._quiet_until,
            active=self._active[self.flex_index],
            dedicated_full=self._dedicated_full(),
        )

    def _choose(self, request: _Request, now: float) -> int | None:
        dedicated = [
            index
            for index, active in enumerate(self._active)
            if index != self.flex_index and active < self.config.max_inflight_per_replica
        ]
        if dedicated:
            return self.router.choose(dedicated)
        if self.flex_index is None:
            return None
        reason = self._borrow_rejection(eligible=request.flex_eligible, now=now)
        if reason is not None:
            self._reject_borrow(request, reason, now)
            return None
        return self.router.choose([self.flex_index])

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
        # Another completion can replace the grace-period wake with an earlier
        # pump. Re-arm it while demand waits so overflow cannot remain stranded.
        if self.flex_index is not None and not self._flex_failed and now < self._quiet_until:
            self._schedule_wake(self._quiet_until - now)
        dispatch: list[tuple[_Request, int]] = []
        # Round-robin tenant heads: one tenant cannot monopolize pending slots.
        stalled = 0
        while self._tenants and stalled < len(self._tenants) and len(dispatch) < _DISPATCH_LIMIT:
            # Do not scan every queued tenant on every submission to a full pool.
            if not any(
                active
                < (
                    self.config.borrowing.max_inflight
                    if i == self.flex_index
                    else self.config.max_inflight_per_replica
                )
                and (i != self.flex_index or self._borrow_rejection(eligible=True, now=now) is None)
                for i, active in enumerate(self._active)
            ):
                break
            tenant = self._tenants[0]
            request = self._queues[tenant][0]
            index = self._choose(request, now)
            self._tenants.rotate(-1)
            if index is None:
                stalled += 1
                continue
            stalled = 0
            self._remove_queued(request)
            self._active[index] += 1
            self._work[index] += request.work
            self._running[index][id(request)] = request
            sequences = sum(r.sequences for r in self._running[index].values())
            for running in self._running[index].values():
                running.peak_sequences = max(running.peak_sequences, sequences)
            if index == self.flex_index:
                self._flex_drained.clear()
            request.admitted.set()
            self._stats["queue_wait_s"] += time.monotonic() - request.submitted
            self._stats["admitted"] += 1
            self._record(
                "admit",
                request_id=request.trace_id,
                replica=index,
                work=request.work,
                reserved_work=self._work[index],
                sequences=request.sequences,
                prompt_tokens=request.prompt_tokens,
                output_tokens=request.output_tokens,
            )
            dispatch.append((request, index))
        for request, index in dispatch:
            self._track(self._execute(request, index))
        if self._tenants and len(dispatch) == _DISPATCH_LIMIT:
            self._schedule_wake(0)

    async def _execute(self, request: _Request, index: int) -> None:
        started = time.monotonic()
        try:
            if index == self.flex_index:
                await self._enter_phase(False)
            request.service_started = time.monotonic()
            result = await request.operation(index)
            service_s = time.monotonic() - request.service_started
            self._record(
                "complete",
                request_id=request.trace_id,
                replica=index,
                service_s=service_s,
                peak_sequences=request.peak_sequences,
            )
            if not request.future.done():
                request.future.set_result(result)
            self._stats["completed"] += 1
            self._stats[f"replica_{index}_completed"] += 1
        except BaseException as exc:
            self._stats["failed"] += 1
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
            del self._running[index][id(request)]
            if index == self.flex_index and not self._active[index]:
                self._flex_drained.set()
            if not self._closed:
                self._schedule_wake(0)

    async def transaction(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Drain a compound mutation on disconnect/shutdown without holding the GPU gate."""
        if self._closed:
            raise RuntimeError("Scheduler is closed")
        task = self._track(operation())
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.done():
                    return task.result()

    async def training(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        tenant: str | None = None,
        reclaim_sampler_memory: bool = False,
        kind: str = "backward",
    ) -> T:
        if self._closed:
            raise RuntimeError("Scheduler is closed")
        # Count readiness synchronously so another sample cannot sneak in first.
        self._training_pending += 1
        self._pump()  # Keep dedicated sampling moving while borrowing is closed.
        task = self._track(self._train(operation, reclaim_sampler_memory, tenant, kind))
        # Once admitted, a mutation is irrevocable. Finish its caller's sequence
        # guard too: propagating cancellation here would permit a retry of an
        # optimizer update that the GPU has already committed.
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.done():
                    return task.result()

    async def _train(
        self,
        operation: Callable[[], Awaitable[T]],
        reclaim_sampler_memory: bool,
        tenant: str | None,
        kind: str,
    ) -> T:
        waiting = time.monotonic()
        try:
            async with self._training_queue.reserve(
                tenant, kind in {"optimizer", "checkpoint"}
            ) as overtook:
                self._stats["training_priority_overtakes"] += int(overtook)
                acquired = time.monotonic()
                had_flex = not self._flex_drained.is_set()
                await self._flex_drained.wait()
                drained = time.monotonic()
                category = (
                    "training"
                    if tenant is not None and kind in {"backward", "optimizer"}
                    else "maintenance"
                )
                queue_s, drain_s = acquired - waiting, drained - acquired
                self._stats[category + "_queue_wait_s"] += queue_s
                self._stats[category + "_drain_wait_s"] += drain_s
                if had_flex:
                    self._stats[category + "_drain_events"] += 1
                self._training_active = True
                started = time.monotonic()
                self._stats["exclusive_wait_s"] += started - waiting
                try:
                    await self._enter_phase(reclaim_sampler_memory)
                    executing = time.monotonic()
                    self._stats[category + "_phase_s"] += executing - started
                    result = await operation()
                    busy = time.monotonic() - executing
                    self._record(
                        "exclusive",
                        kind=kind,
                        category=category,
                        queue_s=queue_s,
                        drain_s=drain_s,
                        phase_s=executing - started,
                        service_s=busy,
                    )
                    self._stats[
                        "training_execution_s"
                        if tenant is not None and kind in {"backward", "optimizer"}
                        else "maintenance_execution_s"
                    ] += busy
                    self._stats["exclusive_completed"] += 1
                    return result
                finally:
                    self._stats["exclusive_busy_s"] += time.monotonic() - started
                    self._training_active = False
                    if reclaim_sampler_memory:
                        self._quiet_until = time.monotonic() + _TRAINING_QUIET_S
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
