from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol

from tinker import types

from tuft.config import ModelConfig

from .base_backend import BaseSamplingBackend


class _SamplingLike(Protocol):
    async def async_init(self) -> None: ...

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse: ...

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None: ...

    async def remove_adapter(self, lora_id: str) -> None: ...

    def get_openai_api_url(self) -> Optional[str]: ...


FixedBackendFactory = Callable[[ModelConfig], _SamplingLike]


logger = logging.getLogger(__name__)


class SamplingRuntimeRouter(BaseSamplingBackend):
    """Route sampling between a Flex runtime and an independent fixed runtime.

    The router deliberately keeps the two runtimes separate:
    - Flex sampling can stay zero-copy and unquantized.
    - Fixed sampling can independently load a quantized vLLM base model.

    The initial implementation is intentionally simple: switching is a routing
    decision plus lazy fixed-runtime initialization and adapter path replay.
    More advanced algorithms can be layered on top without changing backend
    interfaces.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        flex_backend: _SamplingLike | None = None,
        fixed_backend: _SamplingLike | None = None,
        fixed_backend_factory: FixedBackendFactory | None = None,
        initial_backend: Literal["flex", "fixed"] = "flex",
        sleep_inactive_backends: bool = True,
        sleep_level: int = 1,
        mode: Literal["exclusive", "fixed_plus_flex"] = "exclusive",
        flex_drain_timeout_s: float = 10.0,
        flex_max_sample_tokens: int = 0,
        flex_target_inflight: int = 0,
    ) -> None:
        super().__init__(config)
        self.flex_backend = flex_backend
        self.fixed_backend = fixed_backend
        self._fixed_backend_factory = fixed_backend_factory
        self._fixed_initialized = fixed_backend is not None
        self._adapter_paths: dict[str, Path] = {}
        self._lock = asyncio.Lock()
        # Whether the router should shrink the inactive backend's GPU footprint
        # on every switch: sleep(level=sleep_level) the fixed vLLM engine (weights
        # offloaded to CPU, KV cache discarded) and transform_to_training() the
        # Flex backend (releases its sampling-mode KV cache/base alias, keeping
        # only the small CUDA-graph/training runtime resident). This avoids the
        # coexistence memory pressure of keeping both runtimes fully resident.
        self.sleep_inactive_backends = sleep_inactive_backends
        self.sleep_level = sleep_level
        # L3: only requests with max_tokens <= this may land on the flex group
        # while it opportunistically samples (0 = no limit).
        self.flex_max_sample_tokens = flex_max_sample_tokens
        # How many requests the flex group may hold at once while it is in the
        # sampling pool. A round-robin 1/N share left it at 0.213-1.068 req/s
        # with 66% of the window idle: the fixed replicas are already saturated,
        # so the marginal request is worth more on flex until flex itself is
        # busy. 0 restores the old 1/N behaviour.
        self.flex_target_inflight = flex_target_inflight
        self.flex_long_request_skips = 0
        self._fixed_asleep = False
        # Routing mode:
        # - "exclusive": a single active backend (original behavior).
        # - "fixed_plus_flex": the fixed backend ALWAYS serves sampling, and the
        #   flex backend JOINS the sampling pool whenever it is in sampling mode
        #   (fast-loop train<->sample switch). Sampling is round-robin load
        #   balanced across whichever backends are active. This matches the
        #   optimal deployment: fixed sampling stays resident; flex training
        #   contributes extra sampling capacity during training idle windows.
        self.mode = mode
        self._flex_in_sampling = False
        self._rr_counter = 0
        # Data-parallel fixed-sampling replicas (fixed_plus_flex mode). Populated
        # lazily by _ensure_fixed_backends when fixed_sampling_data_parallel_size>1.
        self._fixed_backends: list[_SamplingLike] = []
        self._fixed_backends_initialized = False
        # Adapters already pushed into the flex backend. Replay is incremental:
        # _replay_adapters_to_flex runs on every sample() while flex serves, so
        # re-adding every known adapter there costs one engine call per adapter
        # per request.
        self._flex_replayed: set[str] = set()
        # In-flight sampling requests on the flex backend. Flipping back to
        # training tears down the sampling runtime, so those requests have to
        # drain first or they fail against a released engine.
        self._flex_inflight = 0
        # While draining before a flip, flex must stop receiving NEW sampling:
        # otherwise arrivals keep the in-flight count above zero, the drain times
        # out, and the transform tears the runtime down under active requests
        # (which hangs the next training forward).
        self._flex_draining = False
        self._flex_idle: asyncio.Event = asyncio.Event()
        self._flex_idle.set()
        # Horizon-aware (L3) admission: while flex is sampling, only admit a
        # request whose estimated duration fits before the flex GPU must return
        # to training. This is the paper's "pick short requests that finish
        # inside the predicted window" rule; it prevents flex from hoarding a
        # backlog that would stall the flip back to training.
        self._flex_horizon: float | None = None
        self._flex_tokens_per_s = 2000.0
        self.flex_horizon_skips = 0
        self.flex_drain_timeout_s = flex_drain_timeout_s
        self.flex_drain_timeouts = 0
        self.flex_drain_wait_s = 0.0
        if initial_backend == "flex" and flex_backend is None and fixed_backend is not None:
            initial_backend = "fixed"
        self.active_backend: Literal["flex", "fixed"] = initial_backend
        # Adapter-group pinning map: lora_id -> index into _active_sampling_backends
        self._adapter_pin_map: dict[str, int] = {}
        # Where sampling actually landed. Without this the claim "the flex GPU's
        # idle time was spent sampling" has no direct evidence: GPU utilisation
        # cannot separate sampling from the adapter loads a replica also does,
        # and a run where flex joined the pool and received nothing looks the
        # same as one where it carried its share.
        self.flex_sampling_requests = 0
        self.fixed_sampling_requests = 0
        self.per_replica_requests: dict[int, int] = {}
        # Requests admitted onto flex even though the estimate overran the window,
        # because the GPU had nothing else in flight, and how many completed
        # requests the decode-rate estimate has been calibrated from.
        self.flex_horizon_admits_idle = 0
        self.flex_service_samples = 0
        # Requests sent to flex by the window-filling rule rather than by its
        # round-robin share.
        self.flex_fill_admits = 0
        # Times the window had no room left for another request. High values mean
        # windows are too short for the workload's request length, not that the
        # fill rule is idle.
        self.flex_window_fit_zero = 0
        # Requests routed to flex since the current window opened; the first is
        # the cold start and is excluded from rate calibration.
        self._flex_window_index = 0
        # Round-robin cursor for handing out adapter pins, kept separate from
        # _rr_counter so per-request routing cannot skew pin placement.
        self._pin_assign_counter = 0

    async def async_init(self) -> None:
        if self.mode == "fixed_plus_flex":
            # Fixed sampling is always resident in this mode; bring it up eagerly.
            await self._ensure_fixed_backends()
            return
        backend = self._active_runtime()
        if backend is not None:
            await backend.async_init()
            if self.active_backend == "fixed":
                self._fixed_initialized = True

    def notify_flex_sampling(self, in_sampling: bool) -> None:
        """Track whether the flex backend is currently in sampling mode.

        Called by whoever drives the flex fast-loop train<->sample switch. When
        True (and mode=="fixed_plus_flex"), the flex backend joins the fixed
        backend in serving sampling requests. Adapter replay into the flex
        backend is done lazily in ``_replay_adapters_to_flex`` on first join.
        """
        self._flex_in_sampling = in_sampling

    async def _replay_adapters_to_flex(self) -> None:
        """Push adapters the flex backend has not seen yet.

        Iterates a snapshot: ``add_adapter`` can insert while we await, which
        raised "dictionary changed size during iteration" and aborted the switch.
        Only unseen ids are pushed so the steady-state cost is zero.
        """
        if self.flex_backend is None:
            return
        pending = [
            (lora_id, path)
            for lora_id, path in list(self._adapter_paths.items())
            if lora_id not in self._flex_replayed
        ]
        for lora_id, adapter_path in pending:
            try:
                await self.flex_backend.add_adapter(lora_id, adapter_path)
            except Exception:  # already present / benign
                pass
            self._flex_replayed.add(lora_id)

    async def _active_sampling_backends(self) -> list[_SamplingLike]:
        """Backends currently serving sampling in fixed_plus_flex mode."""
        backends: list[_SamplingLike] = []
        # Fixed sampling is always active (all DP replicas).
        backends.extend(await self._ensure_fixed_backends())
        # Flex joins only while it is in sampling mode. Read the flex backend's
        # live mode directly (more reliable than a manually-toggled flag).
        if self.flex_backend is not None and not self._flex_draining:
            flex_mode = getattr(self.flex_backend, "mode", None)
            in_sampling = self._flex_in_sampling
            if flex_mode is not None:
                in_sampling = str(getattr(self.flex_backend, "_mode", "")) == "sampling"
            if in_sampling:
                await self._replay_adapters_to_flex()
                backends.append(self.flex_backend)
        return backends

    def _pick_sampling_backend(
        self,
        lora_id: str | None,
        backends: list[_SamplingLike],
        sampling_params: object = None,
    ) -> _SamplingLike:
        """Choose the replica for one sampling request.

        Adapter pinning keeps a fixed replica's adapter cache warm, but it must
        not starve the flex group. Pinning deliberately never pins to flex
        because flex can flip back to training, and ``sample`` used to consult
        the pin first: since essentially every request carries a ``lora_id``,
        the pin always hit, the round-robin below never ran, and the flex group
        joined the sampling pool without receiving a single request. The flex
        GPU's idle time was switched into sampling mode and then left unused.

        So when flex is in the pool it first takes the share plain round-robin
        would have given it (one slot in ``len(backends)``); everything else
        follows its pin. The horizon and max-token guards in ``sample`` still
        get the last word on whether a request may actually land on flex.
        """
        if self.flex_backend is not None and self.flex_backend in backends:
            # Fill the window rather than trickle into it. While flex is in the
            # pool the fixed replicas are already at their practical ceiling
            # (measured 1.234 req/s each, 65% busy), so a request queued behind
            # them is worth less than the same request on an idle flex GPU
            # (measured 1.068 req/s, only 13% slower). Bounded so the flip back
            # to training does not wait on a deep queue.
            if self.flex_target_inflight > 0:
                capacity = self._window_fit_capacity(sampling_params)
                if capacity <= 0:
                    self.flex_window_fit_zero += 1
                elif self._flex_inflight < capacity:
                    self.flex_fill_admits += 1
                    return self.flex_backend
            else:
                self._rr_counter += 1
                if self._rr_counter % len(backends) == 0:
                    return self.flex_backend
        pinned = self._adapter_pinned_backend(lora_id, backends)
        if pinned is not None:
            return pinned
        idx = self._rr_counter % len(backends)
        self._rr_counter += 1
        return backends[idx]

    def _adapter_pinned_backend(
        self, lora_id: str | None, backends: list[_SamplingLike]
    ) -> _SamplingLike | None:
        """L3 adapter-group pinning: return the pinned replica for this adapter.

        Each adapter (by lora_id) is pinned to a fixed replica so that the
        adapter cache on that replica stays warm. The pin is stable as long as
        the replica set doesn't change. Returns None if no pin exists yet or
        the pinned replica is not in the current active set.
        """
        if not lora_id or not backends:
            return None
        pin_idx = self._adapter_pin_map.get(lora_id)
        if pin_idx is not None and pin_idx < len(backends):
            candidate = backends[pin_idx]
            # Don't pin to flex backend (it may flip away)
            if candidate is not self.flex_backend:
                return candidate
        # Fall through to re-pin. A grown pool used to keep every old pin because
        # the index stayed in range, so replicas added by the slow loop received
        # no pinned traffic at all: a measured growth run from 3 to 7 replicas
        # left the four new GPUs at 0.5-24.6% busy while the original three
        # dropped from 65% to 53%.
        # Assign a new pin. Round-robin over the fixed replicas rather than
        # hashing the id: hash(lora_id) % len(fixed_only) is uniform only in
        # expectation, and with the handful of adapters a run actually has it is
        # visibly lumpy - eight tenants over three replicas landed 5/5/20, so one
        # replica carried four times its share while the pool looked balanced.
        # Pinning by arrival order gives every replica the same count.
        fixed_only = [b for b in backends if b is not self.flex_backend]
        if not fixed_only:
            return None
        pin_idx_new = self._pin_assign_counter % len(fixed_only)
        self._pin_assign_counter += 1
        # Store the index into the full backends list for fast lookup
        full_idx = backends.index(fixed_only[pin_idx_new])
        self._adapter_pin_map[lora_id] = full_idx
        return fixed_only[pin_idx_new]

    def set_flex_horizon(self, horizon: float | None) -> None:
        """Absolute monotonic time until which flex may admit new sampling."""
        # A rising edge (None -> value) opens a new sampling window. Reset the
        # per-window request counter so the calibration can skip exactly the
        # cold-start request of this window and no more.
        if horizon is not None and self._flex_horizon is None:
            self._flex_window_index = 0
        self._flex_horizon = horizon

    def _active_runtime(self) -> _SamplingLike | None:
        if self.active_backend == "flex":
            return self.flex_backend
        return self.fixed_backend

    async def _ensure_fixed_backend(self) -> _SamplingLike:
        if self.fixed_backend is None:
            if self._fixed_backend_factory is None:
                from .sampling_backend import FixedSamplingBackend

                self.fixed_backend = FixedSamplingBackend(self.config)
            else:
                self.fixed_backend = self._fixed_backend_factory(self.config)
        if not self._fixed_initialized:
            await self.fixed_backend.async_init()
            self._fixed_initialized = True
        return self.fixed_backend

    async def _ensure_fixed_backends(self) -> list[_SamplingLike]:
        """Return the data-parallel fixed-sampling replicas, creating them lazily.

        Uses ``fixed_sampling_data_parallel_size`` independent FixedSamplingBackend
        instances (one per GPU) so the fixed sampling group can span multiple GPUs.
        """
        if self._fixed_backends_initialized:
            return self._fixed_backends
        dp = int(getattr(self.config, "fixed_sampling_data_parallel_size", 1) or 1)
        if dp <= 1:
            self._fixed_backends = [await self._ensure_fixed_backend()]
        else:
            from .sampling_backend import FixedSamplingBackend

            backends = []
            for i in range(dp):
                if self._fixed_backend_factory is not None:
                    inst = self._fixed_backend_factory(self.config)
                else:
                    # instance_index gives each DP replica a unique Ray actor name
                    # (VLLMSamplingBackend appends _dp{i} for index>0).
                    inst = FixedSamplingBackend(self.config, instance_index=i)
                backends.append(inst)
            await asyncio.gather(*[b.async_init() for b in backends])
            self._fixed_backends = backends
        self._fixed_backends_initialized = True
        return self._fixed_backends

    async def _drain_flex_inflight(self) -> None:
        """Wait for flex sampling to quiesce before its runtime is torn down.

        Bounded: a stuck request must not block the training lane forever, so on
        timeout we proceed and record it rather than deadlocking the duty cycle.
        """
        self._flex_draining = True
        try:
            if self._flex_inflight <= 0:
                return
            t0 = time.monotonic()
            try:
                await asyncio.wait_for(
                    self._flex_idle.wait(), timeout=self.flex_drain_timeout_s
                )
            except (asyncio.TimeoutError, TimeoutError):
                self.flex_drain_timeouts += 1
                logger.warning(
                    "flex sampling did not drain in %.1fs (%d in flight); switching anyway",
                    self.flex_drain_timeout_s,
                    self._flex_inflight,
                )
            self.flex_drain_wait_s += time.monotonic() - t0
        finally:
            self._flex_draining = False

    async def switch_to_fixed(self) -> None:
        """Switch future sampling calls to fixed sampling.

        Existing adapters are replayed into the fixed backend by path. This is
        sufficient for the current PEFT adapter workflow and avoids coupling the
        router to FlexBackend internals.

        When ``sleep_inactive_backends`` is enabled, the Flex backend is shrunk
        to its minimal (training-mode) footprint before the fixed backend is
        woken up, so the two runtimes never both hold large KV cache/base model
        memory at the same time.
        """
        async with self._lock:
            # Always return the flex backend to its training footprint when
            # switching to fixed. With ``sleep_inactive_backends`` the flex
            # runtime is additionally shrunk; without it (optimal duty-cycle) we
            # still must leave flex in training mode so TrainingController
            # forwards work after a duty-cycle sampling window.
            if self.flex_backend is not None:
                await self._drain_flex_inflight()
                transform_to_training = getattr(self.flex_backend, "transform_to_training", None)
                if transform_to_training is not None:
                    await transform_to_training()
            # Reuse the resident fixed DP replicas; creating a separate singular
            # fixed backend here would collide with the DP replica's Ray actor
            # name created at init.
            fixed_backends = await self._ensure_fixed_backends()
            # In fixed_plus_flex the fixed replicas are resident and add_adapter
            # already pushed every adapter to all of them, so replaying here only
            # added len(adapters) x len(replicas) engine round-trips to the switch
            # latency. Exclusive mode still needs the replay because the fixed
            # backend is created lazily on the first switch.
            if self.mode != "fixed_plus_flex":
                for backend in fixed_backends:
                    for lora_id, adapter_path in list(self._adapter_paths.items()):
                        try:
                            await backend.add_adapter(lora_id, adapter_path)
                        except Exception:  # already present / benign
                            pass
            if self._fixed_asleep:
                for backend in fixed_backends:
                    wake_up = getattr(backend, "wake_up", None)
                    if wake_up is not None:
                        await wake_up()
                self._fixed_asleep = False
            self.active_backend = "fixed"

    async def switch_to_flex(self) -> None:
        if self.flex_backend is None:
            raise ValueError("Cannot switch to flex: no flex backend is attached.")
        async with self._lock:
            if self.sleep_inactive_backends and self.fixed_backend is not None:
                sleep = getattr(self.fixed_backend, "sleep", None)
                if sleep is not None and not self._fixed_asleep:
                    await sleep(level=self.sleep_level)
                    self._fixed_asleep = True
            transform_to_sampling = getattr(self.flex_backend, "transform_to_sampling", None)
            if transform_to_sampling is not None:
                await transform_to_sampling()
            self.active_backend = "flex"

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        if self.mode == "fixed_plus_flex":
            backends = await self._active_sampling_backends()
            if not backends:
                raise ValueError("No active sampling runtime is attached.")
            # L3 adapter-group pinning, with a round-robin share reserved for the
            # flex group so joining the pool actually earns it work.
            backend = self._pick_sampling_backend(lora_id, backends, sampling_params)
            # L3 horizon admission: a request that cannot finish before the flex
            # GPU returns to training would strand in the drain; send it to fixed.
            if backend is self.flex_backend and self._flex_horizon is not None:
                est = self._estimate_flex_service_s(sampling_params)
                if time.monotonic() + est > self._flex_horizon:
                    if self._flex_inflight == 0:
                        # Nothing else to do with this GPU. A measured 8-GPU duty
                        # arm held flex in sampling for 247.5s across 12 windows
                        # and served 6 requests, bouncing 204: the mechanism paid
                        # the switch cost and the training time and got nothing
                        # back. Bouncing leaves the GPU idle for the rest of the
                        # window, a certain loss; admitting risks delaying the
                        # flip by at most flex_drain_timeout_s, and the measured
                        # runs report flex_drain_timeouts=0.
                        self.flex_horizon_admits_idle += 1
                    else:
                        self.flex_horizon_skips += 1
                        fixed_only = [
                            b for b in backends if b is not self.flex_backend
                        ]
                        if fixed_only:
                            backend = fixed_only[self._rr_counter % len(fixed_only)]
            # L3 short-request admission: a long decode on the flex backend would
            # block the flip back to training, so route it to a fixed replica.
            if (
                backend is self.flex_backend
                and self.flex_max_sample_tokens > 0
                and int(getattr(sampling_params, "max_tokens", 0) or 0)
                > self.flex_max_sample_tokens
            ):
                self.flex_long_request_skips += 1
                fixed_only = [b for b in backends if b is not self.flex_backend]
                if fixed_only:
                    backend = fixed_only[self._rr_counter % len(fixed_only)]
            on_flex = backend is self.flex_backend
            started_at = time.monotonic()
            # Steady state unless this is the window's cold-start request. The
            # earlier rule (inflight>0) excluded every request when flex serves
            # one at a time, so in a sampling-scarce run nothing calibrated and
            # the estimate stayed at its 2000 tok/s default. Concurrency also
            # counts as warm.
            if on_flex:
                self._flex_window_index += 1
                flex_steady = self._flex_inflight > 0 or self._flex_window_index > 1
                self.flex_sampling_requests += 1
                self._flex_inflight += 1
                self._flex_idle.clear()
            else:
                flex_steady = False
                self.fixed_sampling_requests += 1
                try:
                    idx = self._fixed_backends.index(backend)
                except ValueError:
                    idx = -1
                self.per_replica_requests[idx] = self.per_replica_requests.get(idx, 0) + 1
            try:
                return await backend.sample(
                    prompt=prompt,
                    num_samples=num_samples,
                    sampling_params=sampling_params,
                    include_prompt_logprobs=include_prompt_logprobs,
                    topk_prompt_logprobs=topk_prompt_logprobs,
                    lora_id=lora_id,
                )
            finally:
                if on_flex:
                    self._observe_flex_service(
                        sampling_params, time.monotonic() - started_at,
                        steady_state=flex_steady,
                    )
                    self._flex_inflight -= 1
                    if self._flex_inflight <= 0:
                        self._flex_idle.set()
        backend = self._active_runtime()
        if backend is None:
            if self.active_backend != "fixed":
                raise ValueError("No active sampling runtime is attached.")
            backend = await self._ensure_fixed_backend()
        return await backend.sample(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
            lora_id=lora_id,
        )

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        self._adapter_paths[lora_id] = adapter_path
        if self.mode == "fixed_plus_flex":
            # Fixed backend always needs the adapter; flex gets it when it joins.
            for fixed in await self._ensure_fixed_backends():
                await fixed.add_adapter(lora_id, adapter_path)
            if self._flex_in_sampling and self.flex_backend is not None:
                try:
                    await self.flex_backend.add_adapter(lora_id, adapter_path)
                except Exception:
                    pass
                self._flex_replayed.add(lora_id)
            return
        backend = self._active_runtime()
        if backend is not None:
            await backend.add_adapter(lora_id, adapter_path)

    async def remove_adapter(self, lora_id: str) -> None:
        self._adapter_paths.pop(lora_id, None)
        self._flex_replayed.discard(lora_id)
        if self.flex_backend is not None:
            await self.flex_backend.remove_adapter(lora_id)
        if self.fixed_backend is not None:
            await self.fixed_backend.remove_adapter(lora_id)

    def get_openai_api_url(self) -> Optional[str]:
        backend = self._active_runtime()
        if backend is None:
            return None
        return backend.get_openai_api_url()

    def _window_fit_capacity(self, sampling_params: object) -> int:
        """How many requests may sit on flex without outliving its window.

        A flat cap deadlocks the run: with eight in flight, whatever has not
        finished when the window closes is torn down with the sampling runtime and
        the client retries retrieve_future forever - a measured t16 arm logged
        1,209,308 timeouts and never wrote its metrics. A flat 1/N share has the
        opposite failure: the flex GPU sat idle for 81% of a window it had just
        paid two flips for.

        So the bound is the window itself. Requests are decoded concurrently, so
        the batch finishes in roughly the time of its longest member rather than
        the sum; what must fit is one request's service time, and the count is
        limited by how much of the window is left relative to that.
        """
        ceiling = max(self.flex_target_inflight, 0)
        if ceiling <= 0:
            return 0
        if self._flex_horizon is None:
            # No window published: fall back to the configured ceiling, which is
            # what exclusive mode and the pre-duty-cycle paths expect.
            return ceiling
        remaining = self._flex_horizon - time.monotonic()
        per_request = self._estimate_flex_service_s(sampling_params)
        if remaining <= 0.0:
            return 0
        if per_request <= 0.0:
            return ceiling
        # A window shorter than one request admits nothing.
        if remaining < per_request:
            return 0
        # Above that, the ceiling applies rather than remaining/per_request. The
        # engine decodes the batch concurrently, so N requests finish in roughly
        # the time of one, not N times it - dividing assumed serial service and
        # contradicted the reasoning above. It also self-locked: per-request
        # latency rises with concurrency, so the calibrated rate fell to
        # 150.4 tok/s, one request became 3.4s, the window was 3.4s, and capacity
        # sat at zero for all 169 attempts in a measured t4 arm.
        return ceiling

    def _estimate_flex_service_s(self, sampling_params: object) -> float:
        """How long one request is expected to occupy the flex GPU.

        Decode is batched across the samples of a prompt, so the cost tracks
        ``max_tokens``, not ``num_samples * max_tokens`` - the old estimate was up
        to num_samples times too large and rejected almost every request.
        """
        max_tokens = int(getattr(sampling_params, "max_tokens", 0) or 0)
        if max_tokens <= 0:
            return 0.0
        rate = self._flex_tokens_per_s if self._flex_tokens_per_s > 0 else 2000.0
        return max_tokens / rate

    def _observe_flex_service(
        self, sampling_params: object, elapsed_s: float, *, steady_state: bool = True
    ) -> None:
        """Fold a completed flex request into the decode-rate estimate.

        The admission test used to divide by a hard-coded 2000 tokens/s that no
        measurement supported, so it could be wrong by any factor and the error
        would only surface as an unexplained refusal to use the flex GPU.

        Only steady-state requests (flex already decoding on arrival) are folded
        in. A request that woke the runtime carries wake and first-token latency
        that is not decode time; including it measured 155.6 tok/s against a
        ~630 tok/s fixed replica and starved the window-fit admission.
        """
        if not steady_state:
            return
        max_tokens = int(getattr(sampling_params, "max_tokens", 0) or 0)
        if max_tokens <= 0 or elapsed_s <= 0.0:
            return
        observed = max_tokens / elapsed_s
        self.flex_service_samples += 1
        alpha = 0.3
        self._flex_tokens_per_s = (
            observed
            if self.flex_service_samples == 1
            else (1.0 - alpha) * self._flex_tokens_per_s + alpha * observed
        )

    def routing_snapshot(self) -> dict[str, Any]:
        """Where sampling landed, for the ablation tables.

        ``per_replica_requests`` is what exposes an unbalanced pool: a slow-loop
        grow that adds replicas nobody routes to shows up here as zeros long
        before it shows up in wall clock.
        """
        total = self.flex_sampling_requests + self.fixed_sampling_requests
        return {
            "flex_sampling_requests": self.flex_sampling_requests,
            "fixed_sampling_requests": self.fixed_sampling_requests,
            "flex_sampling_share": (
                self.flex_sampling_requests / total if total else 0.0
            ),
            "active_fixed_replicas": len(self._fixed_backends),
            "per_replica_requests": {
                str(k): v for k, v in sorted(self.per_replica_requests.items())
            },
            "replicas_never_routed": sum(
                1
                for i in range(len(self._fixed_backends))
                if self.per_replica_requests.get(i, 0) == 0
            ),
            "flex_horizon_skips": self.flex_horizon_skips,
            "flex_horizon_admits_idle": self.flex_horizon_admits_idle,
            "flex_fill_admits": self.flex_fill_admits,
            "flex_target_inflight": self.flex_target_inflight,
            "flex_window_fit_zero": self.flex_window_fit_zero,
            "flex_tokens_per_s_est": round(self._flex_tokens_per_s, 1),
            "flex_service_samples": self.flex_service_samples,
            "flex_long_request_skips": self.flex_long_request_skips,
            "flex_drain_timeouts": self.flex_drain_timeouts,
        }

    # ------------------------------------------------------------------
    # Slow-loop composition: grow/shrink the fixed sampling pool
    # ------------------------------------------------------------------
    def get_fixed_replica_count(self) -> int:
        """Current number of active fixed sampling replicas."""
        return len(self._fixed_backends) if self._fixed_backends_initialized else 0

    async def remove_one_fixed_replica(self) -> bool:
        """Remove one fixed sampling replica (slow loop: FIXED_TO_FLEX).

        Returns True if a replica was successfully removed, False if none
        remain (at least 1 must stay). The removed replica's GPU becomes
        available for training.
        """
        if not self._fixed_backends_initialized or len(self._fixed_backends) <= 1:
            return False
        removed = self._fixed_backends.pop()
        # Invalidate adapter pins that pointed at indices beyond new length
        to_remove = [
            k for k, idx in self._adapter_pin_map.items()
            if idx >= len(self._fixed_backends)
        ]
        for k in to_remove:
            del self._adapter_pin_map[k]
        return True

    async def add_one_fixed_replica(self) -> bool:
        """Add one fixed sampling replica back (slow loop: FLEX_TO_FIXED).

        Returns True if successful. In identity_tag mode this creates a new
        FixedSamplingBackend instance (near-zero cost since it reuses the
        resident model via Ray actor scheduling).
        """
        if not self._fixed_backends_initialized:
            return False
        from .sampling_backend import FixedSamplingBackend
        new_idx = len(self._fixed_backends)
        if self._fixed_backend_factory is not None:
            inst = self._fixed_backend_factory(self.config)
        else:
            inst = FixedSamplingBackend(self.config, instance_index=new_idx)
        await inst.async_init()
        # Replay existing adapters into the new replica
        for lora_id, adapter_path in list(self._adapter_paths.items()):
            try:
                await inst.add_adapter(lora_id, adapter_path)
            except Exception:
                pass
        self._fixed_backends.append(inst)
        # Drop the pins so they are recomputed over the larger pool. Pins are
        # only valid "as long as the replica set doesn't change", and every old
        # pin index stays in range after a grow, so without this the early
        # return in _adapter_pinned_backend sends all adapter traffic back to
        # the original replicas and the new one only ever sees adapter loads.
        # remove_one_fixed_replica already invalidates out-of-range pins; the
        # grow path was missing the mirror of that.
        self._adapter_pin_map.clear()
        self._pin_assign_counter = 0
        return True
