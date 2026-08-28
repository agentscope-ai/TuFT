"""In-memory state containers backing the FastAPI endpoints."""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TypeVar

from pydantic import BaseModel, Field
from tinker import types

from .auth import AuthenticationDB, User
from .backends.sampling_pipeline import SamplingPipelineBackend
from .checkpoints import CheckpointRecord
from .config import AppConfig
from .corrector import build_corrector
from .exceptions import SessionNotFoundException, UserMismatchException
from .futures import FutureStore
from .persistence import get_redis_store, is_persistence_enabled, load_record, save_record
from .runtime.serial_async_gate import SerialAsyncGate
from .runtime.unified_engine_gate import UnifiedEngineGate
from .sampling_controller import SamplingController
from .schedulers.rl_loop_scheduler import FixedFlexCompositionController, SlowLoopConfig
from .schedulers.runtime_scheduler import RLRuntimeScheduler
from .training_controller import TrainingController, TrainingRunRecord


logger = logging.getLogger(__name__)


T = TypeVar("T")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionRecord(BaseModel):
    """Session record with persistence support.

    Sessions are permanent records (no TTL) as they represent user sessions
    that may need to be accessed at any time.
    """

    session_id: str
    tags: list[str]
    user_metadata: dict[str, str] | None = None
    user_id: str
    sdk_version: str
    created_at: datetime = Field(default_factory=_now)
    last_heartbeat: datetime = Field(default_factory=_now)


class SessionManager:
    """Maintains session metadata and heartbeats so other controllers can enforce ownership."""

    REDIS_KEY_PREFIX = "session"

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionRecord] = {}
        self._restore_from_redis()

    def _build_key(self, session_id: str) -> str:
        return get_redis_store().build_key(self.REDIS_KEY_PREFIX, session_id)

    def _restore_from_redis(self) -> None:
        if not is_persistence_enabled():
            return
        store = get_redis_store()
        pattern = store.build_key(self.REDIS_KEY_PREFIX, "*")
        for key in store.keys(pattern):
            record = load_record(key, SessionRecord)
            if record is not None:
                self._sessions[record.session_id] = record

    def _save_session(self, session_id: str) -> None:
        """Save session to Redis (no TTL - permanent record)."""
        if not is_persistence_enabled():
            return
        record = self._sessions.get(session_id)
        if record is not None:
            save_record(self._build_key(session_id), record)

    def _delete_session(self, session_id: str) -> None:
        if not is_persistence_enabled():
            return
        get_redis_store().delete(self._build_key(session_id))

    def create_session(self, request: types.CreateSessionRequest, user: User) -> SessionRecord:
        """Create a new session for the given user and request."""
        session_id = str(uuid.uuid4())
        record = SessionRecord(
            session_id=session_id,
            tags=request.tags,
            user_id=user.user_id,
            user_metadata=request.user_metadata,
            sdk_version=request.sdk_version,
        )
        self._sessions[session_id] = record
        self._save_session(session_id)
        return record

    def require(self, session_id: str) -> SessionRecord:
        record = self._sessions.get(session_id)
        if record is None:
            raise SessionNotFoundException(session_id)
        return record

    def heartbeat(self, session_id: str, user_id: str) -> None:
        record = self.require(session_id)
        if record.user_id != user_id:
            raise UserMismatchException()
        record.last_heartbeat = _now()
        self._save_session(session_id)

    def list_sessions(self, user_id: str) -> list[str]:
        return [k for k, v in self._sessions.items() if v.user_id == user_id]


class ServerState:
    """Application-wide container that wires controllers together
    and exposes a simple façade to FastAPI.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.config.ensure_directories()
        self.config.check_validity()
        self.sessions = SessionManager()
        self._flex_backends = self._create_flex_backends()
        self._sampling_routers: dict = {}
        # Ray placeholder actors that reserve a GPU for the in-process flex
        # runtime (optimal deployment); kept alive for the server lifetime.
        self._gpu_placeholders: list = []
        self.training = TrainingController(
            self.config, shared_backends=self._flex_backends or None
        )
        # For the optimal deployment, the sampling side is a fixed_plus_flex router:
        # fixed sampling (DP) stays resident and the flex backend joins when it
        # switches to sampling mode. Training keeps using the flex backend directly.
        sampling_shared = self._maybe_wrap_flex_with_router(self._flex_backends)
        self.sampling = SamplingController(
            self.config, shared_backends=sampling_shared or None
        )
        self.auth_db = AuthenticationDB(self.config.authorized_users)
        self.future_store = FutureStore()
        self._init_evaluation_components()

    def _create_flex_backends(self) -> dict:
        """Create shared FlexBackend instances for models with training_backend='flex'."""
        flex_backends = {}
        for model_config in self.config.supported_models:
            if getattr(model_config, "training_backend", "hf") == "flex":
                from .backends.flex.torchtp import FusedTorchTPVLLMFlexBackend

                flex_backends[model_config.model_name] = FusedTorchTPVLLMFlexBackend(
                    model_config
                )
        return flex_backends

    def _maybe_wrap_flex_with_router(self, flex_backends: dict) -> dict:
        """Wrap flex sampling backends in a fixed_plus_flex router for optimal mode.

        Training still uses the flex backend directly; the sampling side becomes a
        router that keeps fixed sampling (DP) resident and lets the flex backend
        join when it switches to sampling mode.
        """
        if not flex_backends:
            return flex_backends
        wrapped = dict(flex_backends)
        for model_config in self.config.supported_models:
            ev = getattr(model_config, "evaluation", None)
            if ev is None or getattr(ev, "deployment_mode", None) != "optimal":
                continue
            name = model_config.model_name
            flex_backend = flex_backends.get(name)
            if flex_backend is None:
                continue
            from .backends.sampling_router import SamplingRuntimeRouter

            _ev = getattr(model_config, "evaluation", None)
            router = SamplingRuntimeRouter(
                model_config,
                flex_backend=flex_backend,
                mode="fixed_plus_flex",
                initial_backend="fixed",
                sleep_inactive_backends=False,
                flex_max_sample_tokens=(
                    getattr(_ev, "flex_max_sample_tokens", 0) if _ev else 0
                ),
                flex_target_inflight=(
                    getattr(_ev, "flex_target_inflight", 0) if _ev else 0
                ),
            )
            wrapped[name] = router
            self._sampling_routers[name] = router
            # The flex backend lives in this (non-Ray) server process, so Ray
            # cannot see its GPU. Reserve one device with a placeholder actor
            # and pin flex to it; Ray then schedules the fixed sampling
            # replicas onto the remaining GPUs instead of colliding with flex.
            try:
                import ray

                from .runtime.gpu_placeholder import GpuPlaceholder

                placeholder = GpuPlaceholder.options(num_gpus=1).remote()
                index = ray.get(placeholder.gpu_index.remote(), timeout=120)
                self._gpu_placeholders.append(placeholder)
                pin = getattr(flex_backend, "pin_device", None)
                if callable(pin):
                    pin(index)
                logger.info(
                    "Reserved GPU%d for the flex backend via placeholder actor.", index
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("GPU placeholder reservation failed: %s", exc)
        return wrapped

    def _init_evaluation_components(self) -> None:
        """Wire evaluation-mode components from per-model EvaluationConfig.

        Components live entirely server-side; the Tinker client never sees
        which deployment mode or scheduler is active.
        """
        self.eval_gate: Optional[SerialAsyncGate] = None
        self.unified_gate: Optional[UnifiedEngineGate] = None
        self.rl_scheduler: Optional[RLRuntimeScheduler] = None
        self.corrector = None
        self.sampling_pipeline: Optional[SamplingPipelineBackend] = None
        # Every wrapped entry point. Versions and stats must reach all of them:
        # keeping only one wrapper meant trained versions were invisible to the
        # wrapper that actually served requests.
        self._sampling_pipelines: list[SamplingPipelineBackend] = []
        # Cumulative time requests spent blocked on evaluation gates before
        # executing ("queue wait" segment of the per-iteration breakdown).
        self._gate_wait_training_s = 0.0
        self._gate_wait_sampling_s = 0.0

        for model_config in self.config.supported_models:
            ev = getattr(model_config, "evaluation", None)
            if ev is None:
                continue
            if ev.deployment_mode == "serial_async" and self.eval_gate is None:
                self.eval_gate = SerialAsyncGate(
                    iterations_per_slice=ev.serial_async_iterations_per_slice,
                )
            if ev.deployment_mode == "unified_engine" and self.unified_gate is None:
                # DP>1 unified engines own one gate per replica inside the DP
                # wrappers; a server-global gate would serialize all replicas.
                if getattr(model_config, "data_parallel_size", 1) <= 1:
                    self.unified_gate = UnifiedEngineGate(
                        sampling_min_window_s=ev.unified_engine_sampling_min_window_s,
                    )
            if ev.rl_scheduler_enabled and self.rl_scheduler is None:
                fixed_groups = max(
                    1, int(getattr(model_config, "fixed_sampling_data_parallel_size", 1) or 1)
                )
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                total_gpus = len([p for p in visible.split(",") if p.strip()]) or (
                    fixed_groups + 1
                )
                self.rl_scheduler = RLRuntimeScheduler(
                    replan_every_s=ev.rl_scheduler_replan_every_s,
                    slow_loop_enabled=ev.slow_loop_enabled,
                    conversion_mode=ev.slow_loop_conversion_mode,
                    fixed_to_flex_cost_s=ev.slow_loop_fixed_to_flex_cost_s,
                    flex_to_fixed_cost_s=ev.slow_loop_flex_to_fixed_cost_s,
                    initial_fixed_groups=fixed_groups,
                    # Every GPU the fixed group does not own is training-capable
                    # (the flex backend plus any GPU the initial composition left
                    # idle), so the slow loop can convert spare capacity into
                    # sampling groups instead of being stuck at flex_groups=1.
                    initial_flex_groups=max(1, total_gpus - fixed_groups),
                    slow_loop=FixedFlexCompositionController(
                        SlowLoopConfig(
                            high_pressure_threshold=ev.slow_loop_high_pressure_threshold,
                            low_pressure_threshold=ev.slow_loop_low_pressure_threshold,
                            smoothing_alpha=ev.slow_loop_smoothing_alpha,
                            min_dwell_s=ev.slow_loop_min_dwell_s,
                            fixed_to_flex_cost_s=ev.slow_loop_fixed_to_flex_cost_s,
                            flex_to_fixed_cost_s=ev.slow_loop_flex_to_fixed_cost_s,
                        )
                    ),
                    flex_duty_cycle_enabled=ev.flex_duty_cycle_enabled,
                    flex_sampling_min_window_s=ev.flex_sampling_min_window_s,
                    flex_switch_backlog_threshold=ev.flex_switch_backlog_threshold,
                    flex_duty_cycle_tick_s=ev.flex_duty_cycle_tick_s,
                    flex_duty_cycle_min_dwell_s=ev.flex_duty_cycle_min_dwell_s,
                    flex_duty_window_cap_s=ev.flex_duty_window_cap_s,
                    flex_duty_window_cap_max_scale=ev.flex_duty_window_cap_max_scale,
                    pack_training=ev.rl_pack_training,
                    flex_window_mode=ev.flex_window_mode,
                    flex_window_period_s=ev.flex_window_period_s,
                    flex_training_window_s=ev.flex_training_window_s,
                    flex_max_training_hold_s=ev.flex_max_training_hold_s,
                    align_slack_frac=ev.rl_align_slack_frac,
                    flex_block_quiet_s=ev.flex_block_quiet_s,
                    gap_gain_factor=ev.flex_gap_gain_factor,
                    fast_loop_holds_enabled=ev.rl_fast_loop_holds_enabled,
                    fast_loop_partial_plan=ev.rl_fast_loop_partial_plan,
                )
                # Bind the fixed_plus_flex sampling router so the duty-cycle
                # controller can flip the flex backend between train/sample.
                router = self._sampling_routers.get(model_config.model_name)
                if router is not None:
                    self.rl_scheduler.attach_router(router)
            if ev.mock_corrector_enabled and self.corrector is None:
                self.corrector = build_corrector(
                    enabled=True,
                    latency_ms=ev.mock_corrector_latency_ms,
                    bias=ev.mock_corrector_bias,
                )
            if ev.sampling_pipeline_enabled and self.sampling_pipeline is None:
                # Wrap every sampling entry point (base backends and shared
                # flex backends) with the L1/L2/L3 admission pipeline. Shared
                # flex backends must NOT be replaced in-place: the same dict is
                # held by TrainingController for forward/optim_step, so the
                # sampling controller gets its own wrapped dict instead.
                wrapped: Optional[SamplingPipelineBackend] = None
                for key, backend in list(self.sampling._base_backends.items()):
                    if isinstance(backend, SamplingPipelineBackend):
                        wrapped = wrapped or backend
                        continue
                    wrapper = SamplingPipelineBackend(
                        model_config,
                        backend,
                        l2_coalesce_max_hold_s=getattr(
                            ev, "l2_coalesce_max_hold_s", 0.0
                        ),
                    )
                    self.sampling._base_backends[key] = wrapper
                    self._sampling_pipelines.append(wrapper)
                    wrapped = wrapped or wrapper
                if self.sampling._shared_backends:
                    wrapped_shared: dict = {}
                    for key, backend in self.sampling._shared_backends.items():
                        if isinstance(backend, SamplingPipelineBackend):
                            wrapped_shared[key] = backend
                            wrapped = wrapped or backend
                        else:
                            wrapper = SamplingPipelineBackend(
                        model_config,
                        backend,
                        l2_coalesce_max_hold_s=getattr(
                            ev, "l2_coalesce_max_hold_s", 0.0
                        ),
                    )
                            wrapped_shared[key] = wrapper
                            self._sampling_pipelines.append(wrapper)
                            wrapped = wrapped or wrapper
                    self.sampling._shared_backends = wrapped_shared
                self.sampling_pipeline = wrapped
                # The window scheduler publishes the flex L3 horizon through it.
                if self.rl_scheduler is not None:
                    self.rl_scheduler.sampling_pipeline = wrapped
                logger.info("Sampling pipeline (L1/L2/L3) enabled")

    def _sampling_routing_stats(self) -> dict[str, Any]:
        """Where sampling requests actually landed, per model.

        Reported separately from ``sampling_pipeline`` because it answers a
        different question: the pipeline counters say which routing rules fired,
        this says which GPU served the request. A flex group that joins the
        sampling pool and receives nothing is invisible in the pipeline counters
        and in wall clock, but obvious here.
        """
        stats: dict[str, Any] = {}
        for name, router in (self._sampling_routers or {}).items():
            snap = getattr(router, "routing_snapshot", None)
            if snap is None:
                continue
            stats[name] = snap()
        return stats

    def _sampling_pipeline_stats(self) -> dict[str, Any]:
        """L1/L2/L3 counters summed over every wrapped sampling entry point."""
        pipelines = self._sampling_pipelines or (
            [self.sampling_pipeline] if self.sampling_pipeline is not None else []
        )
        total: dict[str, Any] = {}
        for pipeline in pipelines:
            for key, value in pipeline.snapshot().items():
                if isinstance(value, (int, float)):
                    total[key] = total.get(key, 0) + value
                else:
                    total.setdefault(key, value)
        total["pipeline_wrappers"] = len(pipelines)
        return total

    def _tenant_of_sample(self, request: types.SampleRequest) -> str | None:
        """Best-effort tenant identity for a sampling request.

        A sampling session is seeded from a training-run checkpoint, so the
        owning tenant is the ``training_run_id`` behind that session. Resolving
        to the training_run_id (rather than the sampling_session_id) is what
        lets a sample match the run SerialAsyncGate promotes at create_model
        time. Returns ``None`` for tenant-less (base-model) samples, which the
        evaluation gates must not block.
        """
        session_id = getattr(request, "sampling_session_id", None)
        if session_id:
            record = self.sampling.sampling_sessions.get(session_id)
            if record is not None:
                return record.training_run_id
        model_path = getattr(request, "model_path", None)
        if model_path:
            try:
                parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(str(model_path))
                if parsed.training_run_id:
                    return parsed.training_run_id
            except Exception:  # pylint: disable=broad-except
                pass
        return None

    def evaluation_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"server_pid": os.getpid()}
        if self.eval_gate is not None:
            snapshot["serial_async_gate"] = self.eval_gate.snapshot()
        if self.unified_gate is not None:
            snapshot["unified_engine_gate"] = self.unified_gate.snapshot()
        else:
            # DP unified: aggregate the per-replica gates owned by the wrappers.
            agg: dict[str, float] = {}
            for backend in self.training.training_backends.values():
                snap_fn = getattr(backend, "unified_gate_snapshot", None)
                if callable(snap_fn):
                    for key, value in snap_fn().items():
                        agg[key] = agg.get(key, 0.0) + float(value)
            if agg:
                snapshot["unified_engine_gate"] = agg
        if self.rl_scheduler is not None:
            snapshot["rl_scheduler"] = self.rl_scheduler.snapshot()
        if self.corrector is not None:
            snapshot["corrector"] = self.corrector.snapshot()
        if self.sampling_pipeline is not None:
            snapshot["sampling_pipeline"] = self._sampling_pipeline_stats()
        routing = self._sampling_routing_stats()
        if routing:
            snapshot["sampling_routing"] = routing
        breakdown: dict[str, float] = {
            "gate_wait_training_s": self._gate_wait_training_s,
            "gate_wait_sampling_s": self._gate_wait_sampling_s,
        }
        for backend in self._flex_backends.values():
            switch_stats = getattr(backend, "switch_stats", None)
            if callable(switch_stats):
                for key, value in switch_stats().items():
                    breakdown[key] = breakdown.get(key, 0.0) + float(value)
        snapshot["iteration_breakdown"] = breakdown
        return snapshot

    async def async_init(self) -> None:
        """Put any async initialization logic here"""
        for backend in self._flex_backends.values():
            await backend.async_init()
        await self.sampling.async_init()
        await self._restore_from_checkpoints()
        if self.rl_scheduler is not None:
            self.rl_scheduler.start_duty_cycle()

    async def _restore_from_checkpoints(self) -> None:
        """Restore server state from checkpoints after Redis restore.

        This method handles checkpoint-based recovery:
        1. For each training run restored from Redis, create adapter and load latest checkpoint
        2. Mark ALL futures created after checkpoint's future_id as failed
        3. For training runs without checkpoints, mark all futures as failed
        4. Mark all pending sample futures as failed
        """
        self.future_store.mark_pending_sample_futures_failed()

        # Restore training runs (adapter + checkpoint)
        for model_id, record in self.training.training_runs.items():
            if record.backend is None or record.corrupted:
                continue
            latest_ckpt = await self.training.restore_from_checkpoint(model_id)

            if latest_ckpt is None:
                self.future_store.mark_futures_failed_after_checkpoint(
                    model_id=model_id,
                    checkpoint_future_id=None,
                    error_message=f"No checkpoint found for model {model_id}. Please retry.",
                )
            else:
                self.future_store.mark_futures_failed_after_checkpoint(
                    model_id=model_id,
                    checkpoint_future_id=latest_ckpt.future_id,
                    error_message=(
                        f"Server restored from checkpoint {latest_ckpt.checkpoint_id}. "
                        "Operations after this checkpoint need to be retried."
                    ),
                )

    def create_session(self, request: types.CreateSessionRequest, user: User) -> SessionRecord:
        return self.sessions.create_session(request, user)

    def heartbeat(self, session_id: str, user_id: str) -> None:
        self.sessions.heartbeat(session_id, user_id)

    async def create_model(
        self,
        session_id: str,
        base_model: str,
        lora_config: types.LoraConfig,
        model_owner: str,
        user_metadata: dict[str, str] | None,
    ) -> TrainingRunRecord:
        self.sessions.require(session_id)
        if self.eval_gate is not None:
            # Serial-Async (strict sequential): one tenant owns the cluster for its
            # whole slice, then yields to the next. Creation is serialized so turns
            # are ordered. The outgoing tenant is NOT unloaded (base + worker are
            # shared; finished tenants stay resident and only the active LoRA
            # adapter changes), but we still wait for it to drain before promoting
            # the next run — otherwise the next promote flips the active run while
            # the outgoing tenant's trailing weight-sync / final-eval sample is
            # still in flight and the gate would block it forever.
            prev_id = await self.eval_gate.acquire_create_slot()
            if prev_id is not None:
                await self.eval_gate.await_evictable(prev_id)
            try:
                record = await self.training.create_model(
                    session_id=session_id,
                    base_model=base_model,
                    lora_config=lora_config,
                    model_owner=model_owner,
                    user_metadata=user_metadata,
                )
            except Exception:
                await self.eval_gate.abort_create_slot()
                raise
            logger.info("[DBG] create_model created %s", record.training_run_id)
            self.eval_gate.register_run(record.training_run_id)
            await self.eval_gate.promote(record.training_run_id)
            logger.info("[DBG] create_model promoted + returning %s", record.training_run_id)
            return record
        return await self.training.create_model(
            session_id=session_id,
            base_model=base_model,
            lora_config=lora_config,
            model_owner=model_owner,
            user_metadata=user_metadata,
        )

    def build_supported_models(self) -> list[types.SupportedModel]:
        return self.training.build_supported_models()

    def get_user(self, api_key: str) -> User | None:
        return self.auth_db.authenticate(api_key)

    async def run_forward(
        self,
        model_id: str,
        user_id: str,
        data: list[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        seq_id: int | None,
        *,
        backward: bool,
    ) -> types.ForwardBackwardOutput:
        if self.eval_gate is not None:
            await self.eval_gate.acquire(model_id)
            self.eval_gate.begin_op(model_id)
        if self.unified_gate is not None:
            _wait_t0 = time.perf_counter()
            await self.unified_gate.acquire_training()
            self._gate_wait_training_s += time.perf_counter() - _wait_t0
        if self.rl_scheduler is not None:
            self.rl_scheduler.observe_training_ready(model_id)
        lane_held = False
        try:
            if self.rl_scheduler is not None:
                # Windowed duty-cycle: if the flex GPU is in a sampling window,
                # wait here (training queues up) until the scheduler opens the
                # next compacted training window, instead of force-flipping per
                # request.
                await self.rl_scheduler.acquire_training_slot()
                _wait_t0 = time.perf_counter()
                await self.rl_scheduler.acquire_training_lane(model_id)
                lane_held = True
                self._gate_wait_training_s += time.perf_counter() - _wait_t0
            result = await self.training.run_forward(
                model_id=model_id,
                user_id=user_id,
                data=data,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
                seq_id=seq_id,
                backward=backward,
            )
        finally:
            if self.rl_scheduler is not None:
                self.rl_scheduler.observe_training_forward_done(model_id)
                # Only the holder may release: cancelled while queueing, the lane
                # is locked by somebody else and releasing it would hand two
                # tenants the single training lane at once.
                if lane_held:
                    self.rl_scheduler.release_training_lane(model_id)
            if self.eval_gate is not None:
                self.eval_gate.end_op(model_id)
        if self.unified_gate is not None:
            await self.unified_gate.release_training()
        if self.rl_scheduler is not None:
            self.rl_scheduler.observe_training_start(model_id)
        return result

    async def run_optim_step(
        self, model_id: str, user_id: str, params: types.AdamParams, seq_id: int | None
    ) -> types.OptimStepResponse:
        if self.eval_gate is not None:
            await self.eval_gate.acquire(model_id)
            self.eval_gate.begin_op(model_id)
        if self.unified_gate is not None:
            _wait_t0 = time.perf_counter()
            await self.unified_gate.acquire_training()
            self._gate_wait_training_s += time.perf_counter() - _wait_t0
        try:
            result = await self.training.run_optim_step(
                model_id=model_id, user_id=user_id, params=params, seq_id=seq_id
            )
        finally:
            if self.unified_gate is not None:
                await self.unified_gate.release_training()
            if self.eval_gate is not None:
                self.eval_gate.end_op(model_id)
        # One optimizer step == one RL iteration for slicing. Eviction is deferred
        # to the next create (SerialAsyncGate) so this tenant's trailing weight-sync
        # and final-eval sampling still land before it is unloaded.
        if self.eval_gate is not None:
            # Tell the gate this tenant made progress. Without it the gate cannot
            # distinguish a tenant that has finished its schedule from one that has
            # not started, and its create-slot handover fires in the quiet gap
            # before a tenant's first sample.
            record = getattr(self.eval_gate, "record_progress", None)
            if record is not None:
                record(model_id)
            await self.eval_gate.release_iteration(model_id)
        if self.rl_scheduler is not None:
            await self.rl_scheduler.observe_training_finish(model_id)
            version = self.rl_scheduler._tenant(model_id).train_steps
            for pipeline in self._sampling_pipelines or (
                [self.sampling_pipeline] if self.sampling_pipeline is not None else []
            ):
                pipeline.notify_adapter_version(model_id, version)
        return result

    async def _evict_training_run(self, model_id: str, user_id: str) -> None:
        """Unload an evicted Serial-Async tenant's adapter/optimizer state.

        The FSDP worker and frozen base model stay resident (Serial-Async only
        swaps the active LoRA adapter / optimizer between tenants), so we do NOT
        shut the backend down here — tearing it down would force an expensive,
        hang-prone re-init for the next tenant.
        """
        try:
            await self.training.unload_model(model_id, user_id)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Eviction unload failed for %s: %s", model_id, e)

    async def create_sampling_session(
        self,
        session_id: str,
        base_model: str | None,
        model_path: str | None,
        user_id: str,
        *,
        session_seq_id: int,
    ) -> str:
        self.sessions.require(session_id)
        return await self.sampling.create_sampling_session(
            session_id=session_id,
            user_id=user_id,
            base_model=base_model,
            model_path=model_path,
            session_seq_id=session_seq_id,
        )

    async def run_sample(self, request: types.SampleRequest, user_id: str) -> types.SampleResponse:
        tenant_id = self._tenant_of_sample(request)
        gated = self.eval_gate is not None and tenant_id is not None
        _sampling_submitted = False
        if gated:
            await self.eval_gate.acquire(tenant_id)
            self.eval_gate.begin_op(tenant_id)
        try:
            if self.unified_gate is not None:
                _wait_t0 = time.perf_counter()
                await self.unified_gate.acquire_sampling()
                self._gate_wait_sampling_s += time.perf_counter() - _wait_t0
            if self.rl_scheduler is not None and tenant_id is not None:
                _wait_t0 = time.perf_counter()
                await self.rl_scheduler.acquire_sampling_permission(tenant_id)
                self._gate_wait_sampling_s += time.perf_counter() - _wait_t0
                self.rl_scheduler.observe_sampling_submit(tenant_id)
                _sampling_submitted = True
            response = await self.sampling.run_sample(request, user_id=user_id)
            if self.corrector is not None:
                response = await self.corrector.correct(response)
            return response
        finally:
            if self.rl_scheduler is not None and tenant_id is not None and _sampling_submitted:
                self.rl_scheduler.observe_sampling_finish(tenant_id)
            if gated:
                self.eval_gate.end_op(tenant_id)

    async def save_checkpoint(
        self,
        model_id: str,
        user_id: str,
        name: str | None,
        checkpoint_type: types.CheckpointType,
        seq_id: int | None = None,
    ) -> CheckpointRecord:
        current_future_id = self.future_store.get_current_future_id()
        if self.eval_gate is not None:
            self.eval_gate.begin_op(model_id)
        try:
            return await self.training.save_checkpoint(
                model_id=model_id,
                user_id=user_id,
                name=name,
                checkpoint_type=checkpoint_type,
                future_id=current_future_id,
                seq_id=seq_id,
            )
        finally:
            if self.eval_gate is not None:
                self.eval_gate.end_op(model_id)

    async def load_checkpoint(
        self, model_id: str, user_id: str, path: str, optimizer: bool, seq_id: int | None = None
    ) -> None:
        return await self.training.load_checkpoint(
            model_id=model_id,
            user_id=user_id,
            path=path,
            optimizer=optimizer,
            seq_id=seq_id,
        )

    def delete_checkpoint(self, model_id: str, user_id: str, checkpoint_id: str) -> None:
        self.training.delete_checkpoint(model_id, user_id, checkpoint_id)

    def list_checkpoints(self, model_id: str, user_id: str) -> list[types.Checkpoint]:
        return self.training.list_checkpoints(model_id, user_id)

    def list_user_checkpoints(self, user_id: str) -> list[types.Checkpoint]:
        return self.training.list_user_checkpoints(user_id)

    def set_checkpoint_visibility(
        self,
        model_id: str,
        user_id: str,
        checkpoint_id: str,
        *,
        public: bool,
    ) -> None:
        self.training.set_visibility(
            model_id=model_id,
            user_id=user_id,
            checkpoint_id=checkpoint_id,
            public=public,
        )

    def get_weights_info(self, tinker_path: str, user_id: str) -> types.WeightsInfoResponse:
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        return self.training.get_weights_info(parsed.training_run_id, user_id)

    def build_archive_url(
        self,
        model_id: str,
        user_id: str,
        checkpoint_id: str,
    ) -> types.CheckpointArchiveUrlResponse:
        return self.training.build_archive_url(model_id, user_id, checkpoint_id)

    def list_training_runs(
        self, *, user_id: str, limit: int | None = None, offset: int = 0
    ) -> types.TrainingRunsResponse:
        return self.training.list_training_runs(user_id=user_id, limit=limit, offset=offset)

    def get_training_run_view(self, model_id: str, user_id: str) -> types.TrainingRun:
        return self.training.get_training_run_view(model_id, user_id)

    def get_training_run_record(self, model_id: str, user_id: str):
        """Get the training run record directly (not the view)."""
        return self.training.get_run_record(model_id, user_id)

    def get_model_info(self, model_id: str, user_id: str) -> types.GetInfoResponse:
        return self.training.get_model_info(model_id, user_id=user_id)

    async def unload_model(self, model_id: str, user_id: str) -> None:
        await self.training.unload_model(model_id, user_id=user_id)
        await self.sampling.evict_model(model_id, user_id=user_id)
        # Hand the Serial-Async create slot back. Serial-Async runs one tenant to
        # completion before the next starts, but release_iteration only reopens the
        # slot when the slice budget hits zero - and a budget large enough to hold a
        # whole tenant is never exhausted. Without this the first tenant finished
        # its ten steps and the remaining seven waited in acquire_create_slot
        # forever: 1 create_model, 1 promote, 0 evictions, all eight GPUs idle.
        if self.eval_gate is not None:
            release = getattr(self.eval_gate, "release_run", None)
            if release is not None:
                await release(model_id)

    def get_session_overview(self, session_id: str, user_id: str) -> types.GetSessionResponse:
        record = self.sessions.require(session_id)
        if record.user_id != user_id:
            raise UserMismatchException()
        training_run_ids = [
            run_id
            for run_id, run in self.training.training_runs.items()
            if run.session_id == session_id
        ]
        sampler_ids = [
            sid
            for sid, record in self.sampling.sampling_sessions.items()
            if record.session_id == session_id
        ]
        return types.GetSessionResponse(training_run_ids=training_run_ids, sampler_ids=sampler_ids)

    def list_sessions(
        self, user_id: str, *, limit: int | None = None, offset: int = 0
    ) -> types.ListSessionsResponse:
        sessions = self.sessions.list_sessions(user_id=user_id)
        total = len(sessions)
        start = min(offset, total)
        if limit is None:
            subset = sessions[start:]
        else:
            subset = sessions[start : min(start + limit, total)]
        return types.ListSessionsResponse(sessions=subset)

    def get_sampler_info(self, sampler_id: str, user_id: str) -> types.GetSamplerResponse:
        return self.sampling.get_sampler_info(
            sampler_id=sampler_id,
            user_id=user_id,
            default_base_model=self.config.supported_models[0].model_name,
        )
