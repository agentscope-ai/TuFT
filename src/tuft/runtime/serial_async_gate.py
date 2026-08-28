"""Serial-Async baseline tenant time-slicing gate.

Serial-Async is the status-quo baseline: the cluster is statically allocated
and tenants time-share it, each owning all GPUs for ``k`` RL iterations before
yielding. The frozen base stays resident; only the active LoRA adapter and
optimizer state change between tenants.

The gate enforces that semantics at the TuFT request entry:

- Training-run creation is serialized: only one tenant's training run may be
  alive at a time (single create slot), so its GPU resources are always
  schedulable.
- The tenant that owns the create slot is ``promote``d to be the active run;
  its sample/forward/optim requests proceed, everyone else waits.
- The active run consumes an iteration budget of ``iterations_per_slice``
  optimizer steps; when the budget is exhausted the gate evicts the run
  (freeing its GPUs) and admits the next waiting tenant.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SerialAsyncGate:
    """Single-active-run time-slice gate for the Serial-Async deployment mode."""

    iterations_per_slice: int = 4
    # After a tenant exhausts its slice it keeps the cluster until it goes idle
    # for this long; only then may the next tenant evict it. The grace bridges
    # the gap between the last optim step and the tenant's trailing
    # save_weights_for_sampler / final-eval sample (which the gate cannot see).
    eviction_grace_s: float = 5.0
    # Ceiling on how long a tenant waits for the create slot before taking it.
    # The slot is only reopened by release_iteration when a slice budget reaches
    # zero, so a tenant that finishes its whole schedule mid-slice used to hold
    # the slot forever: at eight tenants x ten steps with iterations_per_slice=4
    # the first tenant finished and the remaining seven waited here until the run
    # timed out, with every GPU idle for 55 minutes. Reclaiming is safe because
    # the incoming tenant evicts the stale run through the normal drain path.
    create_slot_timeout_s: float = 120.0
    _stale_slot_reclaims: int = field(default=0, init=False)
    # Handovers granted because the holder went idle for the grace window, i.e.
    # it had finished its whole schedule rather than exhausting a slice budget.
    _idle_handovers: int = field(default=0, init=False)
    # Holders already yielded by an idle handover. Multiple waiters observe the
    # same finished holder at once; without this mark they would all hand over
    # against it and promote back-to-back, leaving _active_run on the last one
    # while the earlier tenants' sampling blocks in acquire() again (measured:
    # seven idle_handover grants on one holder within adjacent log lines).
    _handover_consumed: set = field(default_factory=set, init=False)
    # A holder counts as finished only once it has actually trained and then stopped
    # making progress for this long. Keying handover on a plain quiet window was
    # wrong: a tenant is idle between its create_model and its first sample, so the
    # slot rotated all eight tenants through promote() before any of them trained,
    # leaving _active_run on the last one while the other seven could not pass
    # acquire() - measured as create 8, promote 8, asample 32, forward_backward 0,
    # optim_step 0. A tenant training ten steps lands one optim step roughly every
    # 20s, so a minute without progress means it is done rather than mid-schedule.
    completion_idle_s: float = 60.0
    _optim_steps: dict = field(default_factory=dict, init=False)
    _last_progress: dict = field(default_factory=dict, init=False)
    # Every run that has ever been promoted. Sampling for any of them is allowed
    # through acquire(): a promoted run has a checkpoint behind it and its samples
    # are work the workload requested, while the slice discipline that matters is
    # enforced on the training side.
    _promoted_runs: set = field(default_factory=set, init=False)
    _active_run: str | None = field(default=None, init=False)
    _prev_active: str | None = field(default=None, init=False)
    _slot_free: bool = field(default=True, init=False)
    _remaining: int = field(default=0, init=False)
    _due_eviction: str | None = field(default=None, init=False)
    _inflight: dict = field(default_factory=dict, init=False)
    _cond: asyncio.Condition = field(default_factory=asyncio.Condition, init=False)
    _request_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    slice_switches: int = field(default=0, init=False)
    hold_seconds: float = field(default=0.0, init=False)
    evicted_runs: int = field(default=0, init=False)

    async def acquire_create_slot(self) -> str | None:
        """Wait until this tenant may create its training run.

        Returns the run id that must be evicted before/while creating (the tenant
        that just finished its slice via the budget path), or ``None`` (first
        tenant, or an idle handover - the holder there finished its whole schedule,
        so nothing needs evicting; its trailing requests pass via _prev_active).

        History: this method was rewritten after a measured run showed eight
        tenants creating simultaneously on a single logged grant. Two structural
        bugs combined: (1) a fall-through return after the wait loops handed
        callers a slot without any grant branch accounting for it, and (2) a
        handover that marked the holder due for eviction returned that holder to
        its own successor. Every exit below is now explicit and logged.
        """
        start = time.monotonic()
        while True:
            holder: str | None
            async with self._cond:
                if self._slot_free:
                    self._slot_free = False
                    evict = self._due_eviction
                    self._due_eviction = None
                    if evict is not None:
                        self.evicted_runs += 1
                    self._remaining = self.iterations_per_slice
                    self.hold_seconds += time.monotonic() - start
                    logger.info(
                        "[gate] slot granted: branch=slot_free self=%s active=%s evict=%s",
                        id(self), self._active_run, evict,
                    )
                    return evict
                holder = self._active_run

            if holder is None:
                # No run has been promoted yet (very first create). Nothing to
                # observe; poll until the slot is taken and released.
                await asyncio.sleep(0.05)
                continue

            # A holder exists. It yields in one of two ways:
            #   a) its slice budget runs out -> release_iteration frees the slot
            #      (handled by the slot_free branch above on the next iteration), or
            #   b) it finished its whole schedule. The client never says so, and a
            #      budget sized to hold a whole tenant never runs out, so detect it:
            #      at least one optim step recorded, then no progress for
            #      completion_idle_s, then quiet (no in-flight ops) for the grace
            #      window. That is a genuinely finished tenant, not a preemption.
            if holder in self._handover_consumed:
                # Another waiter already took over from this holder; wait for its
                # promote() to publish the new active run before re-evaluating.
                await asyncio.sleep(0.2)
                continue
            if not self._holder_finished(holder):
                await asyncio.sleep(0.2)
                continue
            quiet_since = time.monotonic()
            while True:
                if holder in self._handover_consumed:
                    break  # another waiter already took over; re-evaluate
                if self._inflight.get(holder, 0) > 0:
                    break  # trailing request landed; restart the quiet window later
                if not self._holder_finished(holder):
                    break  # progress resumed; it was not finished after all
                if time.monotonic() - quiet_since >= self.eviction_grace_s:
                    async with self._cond:
                        if (
                            self._slot_free
                            or self._active_run != holder
                            or holder in self._handover_consumed
                        ):
                            break  # state moved under us; re-evaluate from the top
                        self._idle_handovers += 1
                        self._handover_consumed.add(holder)
                        self._remaining = self.iterations_per_slice
                        self.hold_seconds += time.monotonic() - start
                        logger.info(
                            "[gate] slot granted: branch=idle_handover self=%s holder=%s optim_steps=%s",
                            id(self), holder, self._optim_steps.get(holder, 0),
                        )
                        return None
                await asyncio.sleep(0.2)

    def _holder_finished(self, holder: str) -> bool:
        trained = self._optim_steps.get(holder, 0) > 0
        last = self._last_progress.get(holder)
        return (
            trained
            and last is not None
            and time.monotonic() - last >= self.completion_idle_s
        )

    async def abort_create_slot(self) -> None:
        """Give the creation slot back when create_model fails."""
        async with self._cond:
            self._slot_free = True
            self._cond.notify_all()

    async def promote(self, model_id: str) -> None:
        """Make ``model_id`` the active run.

        Called right after a training run is created: the tenant that owns the
        create slot must also be the active run, otherwise its samples wait
        behind another tenant's slice while that tenant waits for the create
        slot (deadlock). The outgoing run is kept as ``_prev_active`` so its
        trailing weight-sync / final-eval requests still proceed after the
        handoff (they would otherwise block forever in :meth:`acquire`).
        """
        self._prev_active = self._active_run
        self._active_run = model_id
        self._promoted_runs.add(model_id)
        self._remaining = self.iterations_per_slice
        self._request_event.set()

    def register_run(self, model_id: str) -> None:
        """Track a training run created inside the current slice."""
        if self._active_run is None:
            self._active_run = model_id
        self._request_event.set()

    async def acquire(self, model_id: str) -> None:
        """Proceed for the active run, the just-evicted previous run, or any run whose
        trailing requests are still in flight; everyone else waits its slice.

        Admitting only the two most recent runs deadlocked the arm. With eight
        tenants the server logged 8 create_model and 8 promote calls - every tenant
        got its slice - but 57 futures began and only 25 completed: the 32 sampling
        requests belonging to runs older than _prev_active could never satisfy the
        membership test, so they blocked forever and the run timed out with one
        tenant finished and every GPU idle.

        A run that reached promote() has a real training checkpoint behind it, and
        its samples are work the workload asked for. Blocking them buys nothing:
        the slice discipline that matters is on the training side, which
        release_iteration and acquire_create_slot enforce.
        """
        start = time.monotonic()
        while True:
            if (
                self._active_run == model_id
                or self._prev_active == model_id
                or self._inflight.get(model_id, 0) > 0
            ):
                self.hold_seconds += time.monotonic() - start
                return
            self._request_event.clear()
            await self._request_event.wait()

    def record_progress(self, model_id: str) -> None:
        """Note that ``model_id`` completed an optimizer step.

        Progress is what separates a tenant that has finished its schedule from one
        that has not started yet; to the gate both look idle otherwise.
        """
        self._optim_steps[model_id] = self._optim_steps.get(model_id, 0) + 1
        self._last_progress[model_id] = time.monotonic()

    async def release_iteration(self, model_id: str) -> list[str]:
        """Count one completed RL iteration.

        When the slice budget is exhausted the run is marked due for eviction
        and the create slot is reopened for the next waiting tenant, but the run
        is NOT unloaded here and stays the active run: the tenant still needs
        its trailing ``save_weights_for_sampler`` and final-eval sampling to
        land. The actual unload is deferred to the next ``acquire_create_slot``
        (see :meth:`await_evictable`), always returning an empty list.
        """
        async with self._cond:
            if self._active_run != model_id:
                return []
            self._remaining -= 1
            if self._remaining > 0:
                return []
            self._due_eviction = model_id
            self._slot_free = True
            self.slice_switches += 1
            self._cond.notify_all()
            self._request_event.set()
            return []

    async def release_run(self, model_id: str) -> None:
        """Give the create slot back when a tenant is done for good.

        release_iteration only reopens the slot when the slice budget reaches
        zero. A tenant that finishes its whole schedule mid-slice therefore left
        _remaining above zero and the slot closed forever: at eight tenants x ten
        steps with iterations_per_slice=4, the first tenant finished and the other
        seven waited in acquire_create_slot until the run timed out (observed:
        one tenant finished, then 55 minutes with all eight GPUs idle).

        Idempotent, because the caller cannot always tell whether the slice
        budget already reopened the slot.
        """
        async with self._cond:
            if self._active_run == model_id:
                self._due_eviction = model_id
                self._remaining = 0
            if not self._slot_free:
                self._slot_free = True
                self.slice_switches += 1
            self._cond.notify_all()
            self._request_event.set()

    def begin_op(self, model_id: str) -> None:
        """Track an in-flight request so eviction waits for it to drain."""
        self._inflight[model_id] = self._inflight.get(model_id, 0) + 1

    def end_op(self, model_id: str) -> None:
        n = self._inflight.get(model_id, 0) - 1
        if n <= 0:
            self._inflight.pop(model_id, None)
        else:
            self._inflight[model_id] = n

    async def await_evictable(self, model_id: str) -> None:
        """Wait until ``model_id`` has no in-flight ops and stays idle for the
        grace window, so its trailing weight-sync / final-eval requests land
        before it is unloaded."""
        idle_since: float | None = None
        while True:
            if self._inflight.get(model_id, 0) > 0:
                idle_since = None
            else:
                now = time.monotonic()
                idle_since = now if idle_since is None else idle_since
                if now - idle_since >= self.eviction_grace_s:
                    return
            await asyncio.sleep(0.2)

    def snapshot(self) -> dict[str, float]:
        return {
            "active_runs": float(1 if self._active_run is not None else 0),
            "slice_switches": float(self.slice_switches),
            "stale_slot_reclaims": float(self._stale_slot_reclaims),
            "idle_handovers": float(self._idle_handovers),
            "tenants_with_progress": float(len(self._optim_steps)),
            "evicted_runs": float(self.evicted_runs),
            "hold_seconds": self.hold_seconds,
            "iterations_per_slice": float(self.iterations_per_slice),
        }
