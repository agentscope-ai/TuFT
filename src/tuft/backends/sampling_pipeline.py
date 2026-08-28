"""Sampling-side pipeline (L1/L2/L3) wrapper for the optimal deployment mode.

The pipeline applies the paper's three sampling-side stages in front of a
concrete sampling backend:

- L1 initial-adapter merging: adapters at version 0 are exactly the frozen
  base, so their requests are merged into the pseudo-adapter ``A0`` and served
  through the base path (no LoRA kernels).
- L2 request reordering: requests sharing an adapter key are kept contiguous
  (same-adapter runs) so downstream batching stays adapter-coherent.
- L3 routing with adapter groups: a flex group has a bounded horizon (its next
  scheduled flip); a request whose conservative duration estimate does not fit
  inside the horizon is held until after the flip instead of being admitted.
  The flex group therefore quiesces by itself as the flip approaches.

When disabled the wrapper is a pure pass-through.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Optional

from tinker import types

from tuft.config import ModelConfig

from .base_backend import BaseSamplingBackend


class SamplingPipelineBackend(BaseSamplingBackend):
    """Wraps another sampling backend with the L1/L2/L3 admission logic."""

    def __init__(
        self,
        config: ModelConfig,
        inner: BaseSamplingBackend,
        *,
        estimated_tokens_per_s: float = 2000.0,
        l2_coalesce_max_hold_s: float = 0.0,
    ) -> None:
        super().__init__(config)
        self.inner = inner
        self._estimated_tokens_per_s = max(1.0, estimated_tokens_per_s)
        self._adapter_versions: dict[str, int] = {}
        # Sampling requests arrive keyed by sampling_session_id, while training
        # reports versions keyed by training run id. Without this alias the two
        # key spaces never meet and every lookup falls back to "version 0",
        # which silently dropped the LoRA from every sampling request.
        self._alias_to_run: dict[str, str] = {}
        self._arrival_seq = 0
        self._flex_flip_at_s: float | None = None
        self._last_adapter_key: str | None = None
        # L2 state: the adapter run currently being dispatched, and how many
        # requests of each adapter key are still executing downstream.
        self._l2_max_hold_s = max(0.0, l2_coalesce_max_hold_s)
        self._run_key: str | None = None
        self._inflight_by_key: dict[str, int] = {}
        self.stats = {
            "a0_merged_requests": 0,
            "runs_started": 0,
            "requests_routed": 0,
            "flex_horizon_holds": 0,
            "flex_horizon_hold_s": 0.0,
            "unknown_version_requests": 0,
            "l2_dispatch_runs": 0,
            "l2_coalesce_holds": 0,
            "l2_coalesce_hold_s": 0.0,
        }

    # ------------------------------------------------------------------
    # Control-plane inputs (fed by TrainingController / RLRuntimeScheduler)
    # ------------------------------------------------------------------
    def notify_adapter_version(self, lora_id: str, version: int) -> None:
        self._adapter_versions[lora_id] = version

    def notify_adapter_alias(self, sampling_key: str, training_run_id: str) -> None:
        """Bind a sampling-session key to the training run that produced it.

        Sampling requests carry ``sampling_session_id``; versions are reported
        per training run. L1 must resolve one to the other before deciding that
        an adapter is still at version 0.
        """
        if sampling_key and training_run_id:
            self._alias_to_run[sampling_key] = training_run_id

    def _known_version(self, key: str) -> Optional[int]:
        """Version of an adapter, or None when we have never been told.

        Versions arrive under the training run id (``notify_adapter_version``)
        while ``add_adapter`` seeds version 0 under the sampling-session key, so
        both key spaces have to be consulted: resolving the alias alone made
        every lookup miss and L1 never merged a single version-0 request.
        """
        run_id = self._alias_to_run.get(key)
        if run_id is not None and run_id in self._adapter_versions:
            return self._adapter_versions[run_id]
        return self._adapter_versions.get(key)

    def set_flex_horizon(self, flip_at_s: Optional[float]) -> None:
        """Set the absolute time of the next flex-group flip (None = unbounded)."""
        self._flex_flip_at_s = flip_at_s

    # ------------------------------------------------------------------
    # L1/L2/L3 admission
    # ------------------------------------------------------------------
    def _effective_lora_id(self, lora_id: Optional[str]) -> Optional[str]:
        """L1: version-0 adapters are the identity map -> merge into A0."""
        if lora_id is None:
            return None
        version = self._known_version(lora_id)
        if version is None:
            # Unknown adapter: never assume "untrained". Dropping the LoRA here
            # would silently serve the base model for a trained adapter.
            self.stats["unknown_version_requests"] += 1
            self._track_run(lora_id)
            return lora_id
        if version == 0:
            self.stats["a0_merged_requests"] += 1
            self._track_run("A0")
            return None
        self._track_run(lora_id)
        return lora_id

    def _track_run(self, adapter_key: str) -> None:
        """L2 proxy: count same-adapter contiguous runs."""
        if adapter_key != self._last_adapter_key:
            self.stats["runs_started"] += 1
            self._last_adapter_key = adapter_key

    async def _coalesce_run(self, run_key: str) -> None:
        """L2: keep same-adapter requests contiguous.

        A request that would start a new adapter run waits for the current run
        to drain, bounded by ``l2_coalesce_max_hold_s`` so a late arrival can
        never be starved. Requests of the run already in flight pass straight
        through, which is what makes the runs contiguous downstream.
        """
        if self._l2_max_hold_s <= 0.0:
            if run_key != self._run_key:
                self._run_key = run_key
                self.stats["l2_dispatch_runs"] += 1
            return
        deadline = time.monotonic() + self._l2_max_hold_s
        while (
            self._run_key is not None
            and self._run_key != run_key
            and self._inflight_by_key.get(self._run_key, 0) > 0
        ):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            hold = min(remaining, 0.01)
            self.stats["l2_coalesce_holds"] += 1
            self.stats["l2_coalesce_hold_s"] += hold
            await asyncio.sleep(hold)
        if run_key != self._run_key:
            self._run_key = run_key
            self.stats["l2_dispatch_runs"] += 1

    def _enter_run(self, run_key: str) -> None:
        self._inflight_by_key[run_key] = self._inflight_by_key.get(run_key, 0) + 1

    def _leave_run(self, run_key: str) -> None:
        remaining = self._inflight_by_key.get(run_key, 0) - 1
        if remaining > 0:
            self._inflight_by_key[run_key] = remaining
        else:
            self._inflight_by_key.pop(run_key, None)

    async def _sample_in_run(self, run_key: str, **kwargs: Any) -> types.SampleResponse:
        """Execute one request while it counts towards its adapter run."""
        self._enter_run(run_key)
        try:
            return await self.inner.sample(**kwargs)
        finally:
            self._leave_run(run_key)

    async def _admit_into_horizon(self, num_samples: int, max_tokens: int) -> None:
        """L3: hold requests that would not finish before the flex flip."""
        if self._flex_flip_at_s is None:
            return
        estimated_s = (num_samples * max_tokens) / self._estimated_tokens_per_s
        while True:
            now = time.monotonic()
            # The flip point has passed: the group is flipping/flipped, admit
            # the request instead of racing the mode switch.
            if now >= self._flex_flip_at_s:
                return
            if now + estimated_s <= self._flex_flip_at_s:
                return
            hold = min(self._flex_flip_at_s - now, 1.0)
            self.stats["flex_horizon_holds"] += 1
            self.stats["flex_horizon_hold_s"] += hold
            await asyncio.sleep(hold)
            if self._flex_flip_at_s is None:
                return

    # ------------------------------------------------------------------
    # BaseSamplingBackend interface (delegating)
    # ------------------------------------------------------------------
    async def async_init(self) -> None:
        await self.inner.async_init()

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        effective_lora_id = self._effective_lora_id(lora_id)
        max_tokens = getattr(sampling_params, "max_tokens", None) or 64
        await self._admit_into_horizon(num_samples, max_tokens)
        self.stats["requests_routed"] += 1
        run_key = effective_lora_id or "A0"
        await self._coalesce_run(run_key)
        return await self._sample_in_run(
            run_key,
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
            lora_id=effective_lora_id,
        )

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        self._adapter_versions.setdefault(lora_id, 0)
        await self.inner.add_adapter(lora_id, adapter_path)

    async def remove_adapter(self, lora_id: str) -> None:
        self._adapter_versions.pop(lora_id, None)
        await self.inner.remove_adapter(lora_id)

    def get_openai_api_url(self) -> Any:
        return self.inner.get_openai_api_url()

    def snapshot(self) -> dict[str, Any]:
        return dict(self.stats)
