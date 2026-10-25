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

from .config import SchedulingConfig
from .exceptions import ResourceExhaustedException
from .loop_scheduling import LoopProfiler


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
    alignment_until: float = 0.0


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
        self._flex_service_started = False
        self._phase_transition: Callable[[bool], Awaitable[None]] | None = None
        self._phase_lock = asyncio.Lock()
        self._phase_state: bool | None = None
        self._flex_failed = False
        self._switch_cost: dict[bool, float] = {}
        self._last_training_end = 0.0
        self._rates: list[float | None] = [None] * replicas
        self._busy_since = [0.0] * replicas
        self._busy_work = [0] * replicas
        self._busy_failed = [False] * replicas
        self._probe_active = False
        self._borrowing = False
        self._last_dispatched: dict[int, str | None] = {}
        self._trace: deque[dict[str, Any]] = deque(maxlen=config.scheduling_trace_capacity)
        self._drain_cost = 0.0
        self._maintenance_cost: dict[str, float] = {}

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
            previous = self._switch_cost.get(training, elapsed)
            self._switch_cost[training] = 0.2 * elapsed + 0.8 * previous
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
            "profiled_tenants": len(self.profiler.arrivals),
            "loop_observations": sum(s.observations for s in self.profiler.states.values()),
            "sampling_seconds_per_token": list(self._rates),
            "switch_to_training_estimate_s": self._switch_cost.get(True),
            "switch_to_sampling_estimate_s": self._switch_cost.get(False),
            "training_drain_estimate_s": self._drain_cost,
            "publication_estimate_s": sum(self._maintenance_cost.values()),
            "loop_profile_status": self.profiler.status(time.monotonic()),
            "decision_trace": list(self._trace),
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
        request.alignment_until = request.submitted + self.config.sampling_alignment_s
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

    def _borrow_worthwhile(self, request: _Request, now: float, horizon: float | None) -> bool:
        """Admission gate; draining and training priority never depend on this model."""
        if not self.config.cost_aware_borrowing or self.config.policy != "shared_weight":
            return True
        index = self.flex_index
        assert index is not None
        if self._probe_active:
            return False  # At most one bounded cold-start request, not a whole wave.
        if now - self._last_training_end < self.config.borrow_min_dwell_s:
            self._stats["borrow_dwell_rejections"] += 1
            self._schedule_wake(self.config.borrow_min_dwell_s - (now - self._last_training_end))
            return False
        fixed = [i for i in range(len(self._work)) if i != index]
        rates = [self._rates[i] for i in fixed]
        round_trip = self._switch_cost.get(True, 0.0) + (
            self._switch_cost.get(False, 0.0) if self._phase_state is not False else 0.0
        )
        if (
            self._seconds_per_token is None
            or len(self._switch_cost) < 2
            or any(r is None for r in rates)
        ):
            # Learn from a bounded probe only when persistent capacity is full.
            # Never guess a token rate or a switch latency for a cold model.
            backlog = sum(r.work for q in self._queues.values() for r in q if r.flex_eligible)
            probe = (
                request.work <= (self.config.borrow_probe_tokens or self.config.flex_max_tokens)
                and backlog >= self.config.borrow_min_backlog_tokens
                and all(self._active[i] >= self.config.max_inflight_per_replica for i in fixed)
                and (
                    horizon is None
                    or (
                        len(self._switch_cost) == 2
                        and all(r is not None for r in rates)
                        and now
                        + self.config.borrow_safety_factor
                        * (round_trip + request.work * max(r or 0.0 for r in rates))
                        < horizon
                    )
                )
            )
            if not probe:
                self._stats["borrow_cold_rejections"] += 1
            return probe
        fixed_finish = min((self._work[i] + request.work) * (self._rates[i] or 0.0) for i in fixed)
        flex_finish = self.config.borrow_safety_factor * (
            round_trip
            + (self._work[index] + request.work) * self._seconds_per_token
            + (self._drain_cost if horizon is None else 0.0)
        )
        if fixed_finish <= flex_finish or (horizon is not None and now + flex_finish >= horizon):
            self._stats["borrow_cost_rejections"] += 1
            return False
        self._stats["borrow_profitable_candidates"] += 1
        return True

    def _choose(self, request: _Request, now: float, horizon: float | None) -> int | None:
        if (
            self.flex_index is not None
            and request.flex_eligible
            and (self._training_pending or self._training_active or now < self._quiet_until)
            and now < request.alignment_until
            and all(
                active >= self.config.max_inflight_per_replica
                for i, active in enumerate(self._active)
                if i != self.flex_index
            )
        ):
            # Only wait for Flex when persistent capacity is already full.
            # Alignment must never strand an available dedicated sampler.
            if not request.aligned:
                request.aligned = True
                self._stats["aligned_requests"] += 1
            self._schedule_wake(request.alignment_until - now)
            return None
        candidates = []
        for index, active in enumerate(self._active):
            limit = self.config.max_inflight_per_replica
            if index == self.flex_index:
                limit = self.config.flex_max_inflight
                if (
                    self._flex_failed
                    or not request.flex_eligible
                    or self._training_pending
                    or self._training_active
                    or now < self._quiet_until
                ):
                    continue
                if horizon is not None and self._seconds_per_token is not None:
                    # Include already reserved work and a safety factor. This
                    # prediction is admission guidance, never a preemption promise;
                    # unexpected training still drains actual admitted requests.
                    finish = now + self.config.borrow_safety_factor * (
                        self._seconds_per_token * (self._work[index] + request.work)
                        + self._switch_cost.get(True, 0)
                        + (self._switch_cost.get(False, 0) if self._phase_state is not False else 0)
                    )
                    if finish >= horizon:
                        self._stats["window_rejections"] += 1
                        continue
                if not self._borrow_worthwhile(request, now, horizon):
                    continue
            if active < limit:
                candidates.append(index)
        if not candidates:
            return None
        # Rotate ties; prefer locality only within a bounded extra-work budget.
        calibrated = (
            self.config.policy == "shared_weight"
            and self.config.cost_aware_borrowing
            and all(self._rates[i] is not None for i in candidates)
        )
        scores = {
            i: (self._work[i] + request.work) * (self._rates[i] or 0.0)
            if calibrated
            else self._work[i]
            for i in candidates
        }
        best = min(candidates, key=lambda i: (scores[i], (i - self._cursor) % len(self._work)))
        local = [
            i
            for i in candidates
            if self._adapter[i] == request.adapter
            and scores[i]
            <= scores[best]
            + (request.work * (self._rates[best] or 0.0) if calibrated else request.work)
            * self.config.locality_slack
        ]
        if local:
            best = min(local, key=lambda i: (scores[i], (i - self._cursor) % len(self._work)))
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
        if self._queued < self.config.adapter_batch_max_requests:
            # Anchor to the oldest queued arrival: new arrivals cannot prolong
            # the window.
            oldest = min(queue[0].submitted for queue in self._queues.values())
            remaining = oldest + self.config.adapter_batch_window_s - now
            if remaining > 0:
                self._schedule_wake(remaining)
                return
        horizon = self.profiler.horizon(now) if self.flex_index is not None else None
        dispatch: list[tuple[_Request, int]] = []
        # Round-robin tenant heads: one tenant cannot monopolize pending slots.
        stalled = 0
        while (
            self._tenants
            and stalled < len(self._tenants)
            and len(dispatch) < self.config.adapter_batch_max_requests
        ):
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
                        not self._flex_failed
                        and not self._training_pending
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
            if not self._active[index]:
                self._busy_since[index] = time.monotonic()
                self._busy_work[index] = 0
                self._busy_failed[index] = False
            self._busy_work[index] += request.work
            if index == self.flex_index:
                self._borrowing = True
                if not self._active[index]:
                    self._flex_busy_since = time.monotonic()
                    self._flex_busy_work = 0
                    self._flex_busy_failed = False
                    self._flex_service_started = False
                    self._probe_active = (
                        self.config.cost_aware_borrowing
                        and self.config.policy == "shared_weight"
                        and (
                            self._seconds_per_token is None
                            or len(self._switch_cost) < 2
                            or any(r is None for i, r in enumerate(self._rates) if i != index)
                        )
                    )
                    if self._probe_active:
                        self._stats["borrow_probes"] += 1
                self._flex_busy_work += request.work
            self._active[index] += 1
            self._work[index] += request.work
            self._adapter[index] = request.adapter
            if index == self.flex_index:
                self._flex_drained.clear()
            request.admitted.set()
            self._stats["queue_wait_s"] += time.monotonic() - request.submitted
            self._stats["admitted"] += 1
            self._record(
                "admit",
                replica=index,
                work=request.work,
                reserved_work=self._work[index],
                horizon_s=None if horizon is None else horizon - now,
            )
            dispatch.append((request, index))
        # Select and reserve in fair tenant order first. Only reorder the
        # selected set, so adapter locality cannot buy extra admission slots.
        # Each group retains FIFO order and every group starts concurrently;
        # never wait for a long generation to finish before starting another.
        groups: dict[tuple[int, str | None], list[_Request]] = {}
        for request, index in dispatch:
            groups.setdefault((index, request.adapter), []).append(request)
        if dispatch:
            self._stats["adapter_dispatch_batches"] += 1
            self._stats["adapter_dispatch_groups"] += len(groups)
            self._stats["adapter_grouped_requests"] += len(dispatch) - len(groups)
        for (index, adapter), requests in groups.items():
            if index in self._last_dispatched and self._last_dispatched[index] != adapter:
                self._stats["adapter_dispatch_switches"] += 1
            self._last_dispatched[index] = adapter
            self._adapter[index] = adapter
            for request in requests:
                self._track(self._execute(request, index))
        if self._tenants and len(dispatch) == self.config.adapter_batch_max_requests:
            self._schedule_wake(0)

    async def _execute(self, request: _Request, index: int) -> None:
        started = time.monotonic()
        try:
            if index == self.flex_index:
                await self._enter_phase(False)
                if not self._flex_service_started:
                    # Count concurrent engine service once, excluding wake time.
                    self._flex_service_started = True
                    self._flex_busy_since = time.monotonic()
                    self._busy_since[index] = self._flex_busy_since
            result = await request.operation(index)
            if not request.future.done():
                request.future.set_result(result)
            self._stats["completed"] += 1
            self._stats[f"replica_{index}_completed"] += 1
        except BaseException as exc:
            self._stats["failed"] += 1
            self._busy_failed[index] = True
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
            if not self._active[index] and not self._busy_failed[index]:
                rate = (time.monotonic() - self._busy_since[index]) / self._busy_work[index]
                previous = self._rates[index]
                self._rates[index] = rate if previous is None else 0.2 * rate + 0.8 * previous
            if index == self.flex_index and not self._active[index]:
                self._probe_active = False
                self._borrowing = False
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
            # Collect slots released by the same completion wave before making
            # another fair admission set; otherwise each callback dispatches a
            # singleton and defeats grouping under sustained load.
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
        if tenant is not None and kind == "backward":
            self.profiler.training_ready(tenant, time.monotonic())
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
            async with self._training_lock:
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
                    self._drain_cost = 0.2 * drain_s + 0.8 * self._drain_cost
                    self._stats[category + "_drain_events"] += 1
                self._training_active = True
                self._borrowing = False
                started = time.monotonic()
                self._stats["exclusive_wait_s"] += started - waiting
                try:
                    await self._enter_phase(reclaim_sampler_memory)
                    executing = time.monotonic()
                    self._stats[category + "_phase_s"] += executing - started
                    try:
                        result = await operation()
                    except BaseException:
                        if tenant is not None:
                            self.profiler.forget(tenant)
                        raise
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
                    if kind in {"checkpoint", "publication"}:
                        self._maintenance_cost[kind] = (
                            0.2 * busy + 0.8 * self._maintenance_cost.get(kind, busy)
                        )
                    self._stats[
                        "training_execution_s"
                        if tenant is not None and kind in {"backward", "optimizer"}
                        else "maintenance_execution_s"
                    ] += busy
                    if tenant is not None:
                        if kind == "reset":
                            self.profiler.forget(tenant)
                        elif kind in {"backward", "optimizer"}:
                            self.profiler.executed(tenant, busy, commit=kind == "optimizer")
                    self._stats["exclusive_completed"] += 1
                    return result
                finally:
                    self._stats["exclusive_busy_s"] += time.monotonic() - started
                    self._training_active = False
                    if reclaim_sampler_memory:
                        self._last_training_end = time.monotonic()
                        self._quiet_until = time.monotonic() + self.config.training_quiet_s
        except BaseException:
            if tenant is not None:
                self.profiler.forget(tenant)
            raise
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
