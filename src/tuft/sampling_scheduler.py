"""Sampling request scheduler.

The scheduler sits between :class:`SamplingController` and a
:class:`BaseSamplingBackend` (e.g. ``VLLMSamplingBackend``).  It does
**not** route across backends -- each scheduler instance wraps exactly
one backend.  Its only job is to *reorder* requests before they reach
the engine so that requests sharing the same LoRA adapter are submitted
contiguously, which improves vLLM's LoRA scheduling throughput.

Two coalescing principles are applied (in order):

1. ``is_initial`` adapters collapse to the base-model bucket.
   PEFT initialises ``lora_B`` to zero, so for any freshly created
   adapter the LoRA contribution ``(B @ A) @ x`` is exactly zero and
   the inference output is identical to the base model.  All such
   requests are scheduled as if they had ``lora_id=None``.

2. Inside a short coalescing window the buffered batch is **stably
   sorted by effective lora_id**.  asyncio's task queue is FIFO and
   :meth:`VLLMSamplingBackend.sample` enters its lock in submission
   order, so this sort directly controls the order in which
   ``engine._generate_internal.remote()`` calls reach the vLLM actor's
   mailbox.  Same-adapter requests therefore appear contiguously in
   vLLM's prefill queue, minimising the number of distinct adapters
   live in any concurrent batch.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tinker import types

from .backends.base_backend import BaseSamplingBackend


logger = logging.getLogger(__name__)


@dataclass
class _PendingItem:
    prompt: types.ModelInput
    num_samples: int
    sampling_params: types.SamplingParams
    include_prompt_logprobs: bool
    topk_prompt_logprobs: int
    # None means "route to base model" (no LoRA, or is_initial adapter).
    effective_lora_id: Optional[str]
    future: asyncio.Future = field(repr=False)


class SamplingRequestScheduler(BaseSamplingBackend):
    """Drop-in :class:`BaseSamplingBackend` wrapper that reorders requests.

    Args:
        backend: Underlying sampling backend that actually talks to vLLM.
        coalesce_window_s: After the first queued item arrives, wait at
            most this many seconds for additional items before dispatching
            the batch.  Smaller values reduce latency, larger values give
            the sort more requests to group.  Default 5 ms.
        max_batch_size: Hard cap on items per dispatch round.  The
            dispatcher fires as soon as this is reached, even if the
            window has not elapsed.
        serialize_groups: If True, the dispatcher awaits one adapter's
            group to complete before kicking off the next group.  This
            gives the strongest "minimum distinct adapters per batch"
            guarantee at the cost of tail latency.  Default False.
    """

    def __init__(
        self,
        backend: BaseSamplingBackend,
        *,
        coalesce_window_s: float = 0.005,
        max_batch_size: int = 32,
        serialize_groups: bool = False,
    ) -> None:
        super().__init__(backend.config)
        self._backend = backend
        self._coalesce_window_s = max(0.0, coalesce_window_s)
        self._max_batch_size = max(1, max_batch_size)
        self._serialize_groups = serialize_groups
        self._queue: asyncio.Queue[_PendingItem] = asyncio.Queue()
        self._initial_adapters: set[str] = set()
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._stopping = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def async_init(self) -> None:
        """Initialise the underlying backend and start the dispatcher."""
        await self._backend.async_init()
        if self._dispatcher_task is None:
            self._dispatcher_task = asyncio.create_task(
                self._dispatcher_loop(), name="sampling_request_scheduler"
            )

    async def stop(self) -> None:
        """Stop the dispatcher and reject any still-pending requests."""
        self._stopping = True
        task = self._dispatcher_task
        self._dispatcher_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def get_openai_api_url(self) -> Optional[str]:
        return self._backend.get_openai_api_url()

    # ------------------------------------------------------------------
    # Adapter management (delegated, plus is_initial bookkeeping)
    # ------------------------------------------------------------------
    async def add_adapter(
        self,
        lora_id: str,
        adapter_path: Path,
        is_initial: Optional[bool] = None,
    ) -> None:
        """Register an adapter, deciding whether it is freshly initialised.

        The decision combines two signals:

        * ``is_initial`` argument -- the *authoritative* tag passed in by
          the caller (typically derived from
          :class:`CheckpointMetadata.is_initial`, which is set by the
          training controller based on ``optim_step_count == 0``).
        * On-disk inspection of ``lora_B`` weights at ``adapter_path`` --
          a *behavioural* check that catches cases the metadata cannot
          (e.g. a checkpoint whose gradients were so small that lora_B
          remained exactly zero across one or more optim_steps).

        Combination rules:

        * ``is_initial=True``  -- trusted; skip disk inspection
          (zero I/O cost path).
        * ``is_initial=False`` -- still inspect weights, because the
          caller may have under-counted (the goal is to find *more*
          coalescing opportunities, never *fewer*).  An auto-detected
          all-zero ``lora_B`` upgrades the flag to ``True``.
        * ``is_initial=None``  -- unknown; rely entirely on disk
          inspection.

        Mis-classifying a fresh adapter as "trained" only loses an
        optimisation; the opposite would silently change inference
        semantics, hence inspection conservatively returns ``False`` on
        any read error.
        """
        if is_initial is True:
            effective_is_initial = True
        else:
            detected = await asyncio.to_thread(self._detect_initial_adapter, adapter_path)
            effective_is_initial = bool(is_initial) or detected
            logger.debug(
                "is_initial decision for adapter %s at %s: caller=%r, detected=%s, effective=%s",
                lora_id,
                adapter_path,
                is_initial,
                detected,
                effective_is_initial,
            )
        await self._backend.add_adapter(lora_id, adapter_path, is_initial=effective_is_initial)
        if effective_is_initial:
            self._initial_adapters.add(lora_id)
        else:
            self._initial_adapters.discard(lora_id)

    async def remove_adapter(self, lora_id: str) -> None:
        self._initial_adapters.discard(lora_id)
        await self._backend.remove_adapter(lora_id)

    def mark_initial(self, lora_id: str, is_initial: bool = True) -> None:
        """Manually override the ``is_initial`` flag for an already-added adapter.

        Auto-detection runs at ``add_adapter`` time; use this only to
        force-flip the flag after the fact (e.g. for tests).
        """
        if is_initial:
            self._initial_adapters.add(lora_id)
        else:
            self._initial_adapters.discard(lora_id)

    def is_initial(self, lora_id: str) -> bool:
        return lora_id in self._initial_adapters

    @staticmethod
    def _detect_initial_adapter(adapter_path: Path) -> bool:
        """Return True iff the adapter on disk has all-zero ``lora_B`` weights.

        Reads ``adapter_model.safetensors`` (preferred) or
        ``adapter_model.bin`` and scans every key containing
        ``.lora_B.``.  Any non-zero value, missing file, or read error
        causes the function to conservatively return False -- mis-classifying
        a fresh adapter as "trained" only loses an optimisation; the
        opposite would silently change inference semantics.
        """
        safetensors_file = adapter_path / "adapter_model.safetensors"
        bin_file = adapter_path / "adapter_model.bin"
        try:
            if safetensors_file.exists():
                # Lazy import: safetensors is only needed in real backends,
                # not in CPU-only / dummy test paths.
                from safetensors import safe_open

                with safe_open(str(safetensors_file), framework="pt") as f:
                    lora_b_keys = [k for k in f.keys() if ".lora_B." in k]
                    if not lora_b_keys:
                        return False
                    for k in lora_b_keys:
                        tensor = f.get_tensor(k)
                        # bool() on a 0-d tensor of value 0 is False.
                        if bool(tensor.any()):
                            return False
                    return True
            if bin_file.exists():
                import torch

                state = torch.load(str(bin_file), map_location="cpu", weights_only=True)
                lora_b_keys = [k for k in state.keys() if ".lora_B." in k]
                if not lora_b_keys:
                    return False
                for k in lora_b_keys:
                    if bool(state[k].any()):
                        return False
                return True
        except Exception:
            logger.exception(
                "is_initial auto-detection failed for adapter at %s; "
                "falling back to is_initial=False",
                adapter_path,
            )
            return False
        return False

    # ------------------------------------------------------------------
    # Sampling entry point (intercepted)
    # ------------------------------------------------------------------
    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        if self._dispatcher_task is None or self._stopping:
            # Scheduler not running; pass straight through.
            return await self._backend.sample(
                prompt=prompt,
                num_samples=num_samples,
                sampling_params=sampling_params,
                include_prompt_logprobs=include_prompt_logprobs,
                topk_prompt_logprobs=topk_prompt_logprobs,
                lora_id=lora_id,
            )

        loop = asyncio.get_running_loop()
        item = _PendingItem(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
            effective_lora_id=self._effective_lora_id(lora_id),
            future=loop.create_future(),
        )
        await self._queue.put(item)
        return await item.future

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _effective_lora_id(self, lora_id: Optional[str]) -> Optional[str]:
        """Return the scheduling key.

        ``is_initial`` adapters (and a missing lora_id) all map to
        ``None`` so they share the base-model bucket.
        """
        if lora_id is None:
            return None
        if lora_id in self._initial_adapters:
            return None
        return lora_id

    async def _dispatcher_loop(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while not self._stopping:
                first = await self._queue.get()
                batch: list[_PendingItem] = [first]
                deadline = loop.time() + self._coalesce_window_s
                while len(batch) < self._max_batch_size:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        nxt = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                    batch.append(nxt)

                # Stable sort: requests with the same effective lora_id keep
                # their arrival order; different adapters become contiguous.
                # The empty string is used as the sort key for the base
                # bucket (None) so it always sorts first.
                batch.sort(key=lambda it: it.effective_lora_id or "")

                if logger.isEnabledFor(logging.DEBUG):
                    distinct = {it.effective_lora_id for it in batch}
                    logger.debug(
                        "scheduler dispatching batch_size=%d distinct_adapters=%d",
                        len(batch),
                        len(distinct),
                    )

                if self._serialize_groups:
                    await self._dispatch_serial(batch)
                else:
                    self._dispatch_concurrent(batch)
        except asyncio.CancelledError:
            self._drain_pending(asyncio.CancelledError("scheduler stopped"))
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("sampling scheduler dispatcher crashed")
            self._drain_pending(exc)
            raise

    def _dispatch_concurrent(self, batch: list[_PendingItem]) -> None:
        """Fire all batch items concurrently, preserving sort order.

        ``asyncio.create_task`` schedules tasks FIFO and each task runs
        synchronously up to its first ``await`` -- which inside
        :meth:`VLLMSamplingBackend.sample` is the FIFO ``self._lock``.
        Hence the order of ``engine._generate_internal.remote()`` calls
        seen by the vLLM actor matches the sort order of ``batch``.
        """
        for item in batch:
            asyncio.create_task(self._dispatch_one(item))

    async def _dispatch_serial(self, batch: list[_PendingItem]) -> None:
        """Group by adapter and serialise between groups.

        Forces "one adapter at a time" submission for the strongest
        batching guarantee at the cost of additional tail latency.
        """
        groups: dict[Optional[str], list[_PendingItem]] = {}
        order: list[Optional[str]] = []
        for it in batch:
            if it.effective_lora_id not in groups:
                groups[it.effective_lora_id] = []
                order.append(it.effective_lora_id)
            groups[it.effective_lora_id].append(it)
        for key in order:
            tasks = [asyncio.create_task(self._dispatch_one(it)) for it in groups[key]]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _dispatch_one(self, item: _PendingItem) -> None:
        try:
            result = await self._backend.sample(
                prompt=item.prompt,
                num_samples=item.num_samples,
                sampling_params=item.sampling_params,
                include_prompt_logprobs=item.include_prompt_logprobs,
                topk_prompt_logprobs=item.topk_prompt_logprobs,
                lora_id=item.effective_lora_id,
            )
            if not item.future.done():
                item.future.set_result(result)
        except Exception as exc:
            if not item.future.done():
                item.future.set_exception(exc)

    def _drain_pending(self, exc: BaseException) -> None:
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not item.future.done():
                item.future.set_exception(exc)
