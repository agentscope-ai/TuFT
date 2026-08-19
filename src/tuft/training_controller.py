"""Training controller for managing training runs and routing requests."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Dict, List, Optional, TypeVar

from opentelemetry.trace import StatusCode
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from tinker import types

from .backends import BaseTrainingBackend
from .checkpoints import CheckpointMetadata, CheckpointRecord, compute_tree_size
from .config import (
    DEFAULT_LORA_ALPHA_RATIO,
    FSDP_QV_TARGET_MODULES,
    AppConfig,
    ModelConfig,
    compute_lora_alpha,
)
from .exceptions import (
    CheckpointAccessDeniedException,
    CheckpointIncompatibleException,
    CheckpointMetadataReadException,
    CheckpointNotFoundException,
    InvalidRequestException,
    SequenceConflictException,
    UnknownModelException,
    UserMismatchException,
)
from .persistence import (
    delete_record,
    get_redis_store,
    is_persistence_enabled,
    load_record,
    save_record,
    save_records_atomic,
)
from .telemetry.metrics import get_metrics
from .telemetry.tracing import get_tracer


_get_tracer = lambda: get_tracer("tuft.training_controller")  # noqa: E731


logger = logging.getLogger(__name__)

T = TypeVar("T")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TrainingRunRecord(BaseModel):
    """Training run record with persistence support.

    Runtime-only fields (backend, _execution_lock) are excluded from serialization.
    Checkpoints are stored separately with their own keys.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    training_run_id: str
    base_model: str
    lora_rank: int
    # LoRA target-module selection. None means the record predates these fields
    # and cannot safely be resumed. Its checkpoints remain available as sources
    # because their adapter_config.json records the actual module geometry.
    train_attn: bool | None = None
    train_mlp: bool | None = None
    train_unembed: bool | None = None
    # Concrete module geometry resolved when the adapter was created. None is
    # accepted only while reading older persisted records, which become read-only.
    target_modules: list[str] | None = None
    session_id: str
    model_owner: str
    user_metadata: dict[str, str] | None = None
    created_at: datetime = Field(default_factory=_now)
    last_request_time: datetime = Field(default_factory=_now)
    # Checkpoints are stored separately, excluded from serialization
    checkpoints: Dict[str, CheckpointRecord] = Field(default_factory=dict, exclude=True)
    sampler_checkpoints: Dict[str, CheckpointRecord] = Field(default_factory=dict, exclude=True)
    next_training_checkpoint: int = 1
    next_sampler_checkpoint: int = 1
    corrupted: bool = False
    next_seq_id: int = 1
    # Runtime-only fields, excluded from serialization
    backend: BaseTrainingBackend | None = Field(default=None, exclude=True)
    # Private attribute for execution lock (not a model field)
    _execution_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    @property
    def has_legacy_lora_state(self) -> bool:
        """Whether this run lacks the complete, effective LoRA geometry."""
        return self.target_modules is None or any(
            flag is None for flag in (self.train_attn, self.train_mlp, self.train_unembed)
        )

    def lora_config(self) -> types.LoraConfig:
        """Rebuild the LoRA config this run's adapter was created with.

        Legacy records are invalidated on restore, so every resumable run has
        complete geometry here.
        """
        assert self.train_attn is not None
        assert self.train_mlp is not None
        assert self.train_unembed is not None
        return types.LoraConfig(
            rank=self.lora_rank,
            train_attn=self.train_attn,
            train_mlp=self.train_mlp,
            train_unembed=self.train_unembed,
        )

    def to_training_run(self) -> types.TrainingRun:
        training_checkpoint = self._latest_checkpoint(self.checkpoints)
        sampler_checkpoint = self._latest_checkpoint(self.sampler_checkpoints)
        return types.TrainingRun(
            training_run_id=self.training_run_id,
            base_model=self.base_model,
            model_owner=self.model_owner,
            is_lora=True,
            corrupted=self.corrupted,
            lora_rank=self.lora_rank,
            last_request_time=self.last_request_time,
            last_checkpoint=training_checkpoint,
            last_sampler_checkpoint=sampler_checkpoint,
            user_metadata=self.user_metadata,
        )

    def _latest_checkpoint(self, items: Dict[str, CheckpointRecord]) -> types.Checkpoint | None:
        if not items:
            return None
        latest = max(items.values(), key=lambda record: record.created_at)
        return latest.tinker_checkpoint


class TrainingController:
    """Tracks training runs, enforces request ordering.

    Routes work into ModelBackend instances.
    """

    REDIS_KEY_PREFIX = "training_run"

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.training_backends = self._create_backends(config.supported_models)
        # TODO: add a mechanism to manage training_runs
        self.training_runs: Dict[str, TrainingRunRecord] = {}
        self._restore_from_redis()

    def _create_backends(self, model_configs: List[ModelConfig]) -> Dict[str, BaseTrainingBackend]:
        backends: Dict[str, BaseTrainingBackend] = {}
        # FSDP port allocation: 29500, 29501, ... by order of FSDP models in supported_models
        fsdp_model_names = [
            c.model_name for c in model_configs if getattr(c, "training_backend", "hf") == "fsdp"
        ]
        for config in model_configs:
            fsdp_index: Optional[int] = None
            if config.model_name in fsdp_model_names:
                fsdp_index = fsdp_model_names.index(config.model_name)
            backends[config.model_name] = BaseTrainingBackend.create_backend(
                config,
                fsdp_index=fsdp_index,
                worker_venv_path=self.config.worker_venv_path,
            )
            # Log the LoRA scaling in effect: the "hf" backend used
            # lora_alpha = rank before lora_alpha_ratio existed, so operators
            # upgrading need to see which scaling their runs now get.
            logger.info(
                "Training backend for %s: %s, lora_alpha = rank * %d",
                config.model_name,
                getattr(config, "training_backend", "hf"),
                config.lora_alpha_ratio,
            )
        return backends

    async def shutdown(self) -> None:
        """Shut down all training backends and release GPU/Ray resources."""
        for backend in self.training_backends.values():
            try:
                await backend.shutdown()
            except Exception:
                logger.exception("Failed to shut down training backend for %s", backend.base_model)

    def _effective_lora_alpha(self, training_run: TrainingRunRecord) -> int:
        """LoRA alpha the run's adapters are built with, from the model config.

        Single source of truth for both writing checkpoint metadata and validating
        a load, so the two can never disagree.
        """
        model_config = self._model_config_for(training_run.base_model)
        ratio = model_config.lora_alpha_ratio if model_config else DEFAULT_LORA_ALPHA_RATIO
        return compute_lora_alpha(training_run.lora_rank, ratio)

    def _build_key(self, model_id: str) -> str:
        return get_redis_store().build_key(self.REDIS_KEY_PREFIX, model_id)

    def _build_checkpoint_key(self, model_id: str, checkpoint_id: str) -> str:
        return get_redis_store().build_key(self.REDIS_KEY_PREFIX, model_id, "ckpt", checkpoint_id)

    def _build_sampler_checkpoint_key(self, model_id: str, checkpoint_id: str) -> str:
        return get_redis_store().build_key(
            self.REDIS_KEY_PREFIX, model_id, "sampler_ckpt", checkpoint_id
        )

    def _restore_from_redis(self) -> None:
        """Restore training runs from Redis on startup."""
        if not is_persistence_enabled():
            return
        store = get_redis_store()
        # Match only top-level training runs (3 parts: namespace::prefix::model_id)
        for key in store.keys(store.build_key(self.REDIS_KEY_PREFIX, "*")):
            parts = key.split("::")
            if len(parts) != 3:
                continue
            record = load_record(key, TrainingRunRecord)
            if record is None:
                continue
            model_id = record.training_run_id
            # Restore checkpoints (stored separately, not subject to TTL)
            self._restore_checkpoints(model_id, record)
            self._restore_sampler_checkpoints(model_id, record)
            # Restore backend reference
            if record.base_model in self.training_backends:
                record.backend = self.training_backends[record.base_model]
            else:
                record.corrupted = True
            if record.has_legacy_lora_state:
                record.corrupted = True
                logger.warning(
                    "Training run %s does not record its complete effective LoRA geometry "
                    "and is therefore read-only; checkpoints with a valid "
                    "adapter_config.json remain available to load into a new run.",
                    model_id,
                )
            self.training_runs[model_id] = record

    def _restore_checkpoints(self, model_id: str, record: TrainingRunRecord) -> None:
        store = get_redis_store()
        pattern = self._build_checkpoint_key(model_id, "*")
        record.checkpoints = {}
        for key in store.keys(pattern):
            ckpt = load_record(key, CheckpointRecord)
            if ckpt is not None:
                record.checkpoints[ckpt.checkpoint_id] = ckpt

    def _restore_sampler_checkpoints(self, model_id: str, record: TrainingRunRecord) -> None:
        store = get_redis_store()
        pattern = self._build_sampler_checkpoint_key(model_id, "*")
        record.sampler_checkpoints = {}
        for key in store.keys(pattern):
            ckpt = load_record(key, CheckpointRecord)
            if ckpt is not None:
                record.sampler_checkpoints[ckpt.checkpoint_id] = ckpt

    def _save_training_run(self, model_id: str) -> None:
        """Save training run to Redis (no TTL - permanent record)."""
        if not is_persistence_enabled():
            return
        record = self.training_runs.get(model_id)
        if record is not None:
            save_record(self._build_key(model_id), record)

    def _save_checkpoint(self, model_id: str, checkpoint_id: str) -> None:
        """Save checkpoint to Redis (no TTL - permanent record)."""
        if not is_persistence_enabled():
            return
        record = self.training_runs.get(model_id)
        if record is not None:
            ckpt = record.checkpoints.get(checkpoint_id)
            if ckpt is not None:
                save_record(self._build_checkpoint_key(model_id, checkpoint_id), ckpt)

    def _save_sampler_checkpoint(self, model_id: str, checkpoint_id: str) -> None:
        """Save sampler checkpoint to Redis (no TTL - permanent record)."""
        if not is_persistence_enabled():
            return
        record = self.training_runs.get(model_id)
        if record is not None:
            ckpt = record.sampler_checkpoints.get(checkpoint_id)
            if ckpt is not None:
                save_record(self._build_sampler_checkpoint_key(model_id, checkpoint_id), ckpt)

    def _save_training_run_with_checkpoint(
        self, model_id: str, checkpoint_id: str, checkpoint_type: types.CheckpointType
    ) -> None:
        """Save training run and checkpoint atomically using Redis transaction.

        This ensures consistency if the server crashes between saves.
        No TTL is used for these records as they are permanent.
        """
        if not is_persistence_enabled():
            return
        record = self.training_runs.get(model_id)
        if record is None:
            return

        if checkpoint_type == "training":
            ckpt = record.checkpoints.get(checkpoint_id)
            ckpt_key = self._build_checkpoint_key(model_id, checkpoint_id)
        else:
            ckpt = record.sampler_checkpoints.get(checkpoint_id)
            ckpt_key = self._build_sampler_checkpoint_key(model_id, checkpoint_id)

        if ckpt is None:
            # Defensive fallback: checkpoint should exist at this point since
            # _save_training_run_with_checkpoint is called after adding the checkpoint
            # to the target_map. This branch handles unexpected edge cases (e.g., code
            # refactoring that changes call order) to ensure the training run is still
            # persisted even if the checkpoint lookup fails.
            logger.warning(
                "Checkpoint %s not found for model %s during persistence, "
                "saving training run without checkpoint",
                checkpoint_id,
                model_id,
            )
            save_record(self._build_key(model_id), record)
            return

        # Save both atomically (no TTL for permanent records)
        save_records_atomic(
            [
                (self._build_key(model_id), record),
                (ckpt_key, ckpt),
            ]
        )

    def _delete_training_run(self, model_id: str) -> None:
        if not is_persistence_enabled():
            return
        store = get_redis_store()
        store.delete(self._build_key(model_id))
        store.delete_pattern(self._build_checkpoint_key(model_id, "*"))
        store.delete_pattern(self._build_sampler_checkpoint_key(model_id, "*"))

    def _delete_checkpoint_record(self, model_id: str, checkpoint_id: str) -> None:
        if not is_persistence_enabled():
            return
        delete_record(self._build_checkpoint_key(model_id, checkpoint_id))

    def _delete_sampler_checkpoint_record(self, model_id: str, checkpoint_id: str) -> None:
        if not is_persistence_enabled():
            return
        delete_record(self._build_sampler_checkpoint_key(model_id, checkpoint_id))

    async def _with_sequence_guard(
        self,
        record: TrainingRunRecord,
        seq_id: int | None,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        async with record._execution_lock:
            if seq_id is not None:
                expected = record.next_seq_id
                if seq_id != expected:
                    raise SequenceConflictException(expected=expected, got=seq_id)

            result = await operation()

            if seq_id is not None:
                record.next_seq_id += 1
            # Save the updated next_seq_id to Redis
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_training_run, record.training_run_id)
            return result

    async def create_model(
        self,
        session_id: str,
        base_model: str,
        lora_config: types.LoraConfig,
        model_owner: str,
        user_metadata: dict[str, str] | None,
    ) -> TrainingRunRecord:
        model_id = str(uuid.uuid4())
        with _get_tracer().start_as_current_span("training_controller.create_model") as span:
            span.set_attribute("tuft.training_run_id", model_id)
            span.set_attribute("tuft.session_id", session_id)
            span.set_attribute("tuft.base_model", base_model)
            span.set_attribute("tuft.lora_rank", lora_config.rank)
            try:
                logger.info("Creating model %s", model_id)

                if base_model not in self.training_backends:
                    raise UnknownModelException(model_name=base_model)
                backend = self.training_backends[base_model]
                target_modules = self._effective_target_modules(base_model, lora_config)
                record = TrainingRunRecord(
                    training_run_id=model_id,
                    base_model=base_model,
                    lora_rank=lora_config.rank,
                    train_attn=lora_config.train_attn,
                    train_mlp=lora_config.train_mlp,
                    train_unembed=lora_config.train_unembed,
                    target_modules=target_modules,
                    session_id=session_id,
                    model_owner=model_owner,
                    user_metadata=user_metadata,
                    backend=backend,
                )
                await backend.create_adapter(model_id, lora_config)
                self.training_runs[model_id] = record
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._save_training_run, model_id)

                # Update metrics
                get_metrics().training_models_active.add(1, {"base_model": base_model})
                return record
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR)
                raise

    def get_run_record(
        self,
        model_id: str,
        user_id: str,
        enforce_user_match: bool = True,
    ) -> TrainingRunRecord:
        record = self.training_runs.get(model_id)
        if record is None:
            raise UnknownModelException(model_name=model_id)
        if enforce_user_match and record.model_owner != user_id:
            raise UserMismatchException()
        return record

    @staticmethod
    def _require_resumable_run(record: TrainingRunRecord) -> None:
        if record.has_legacy_lora_state:
            raise InvalidRequestException(
                f"Training run {record.training_run_id} does not record its complete effective "
                "LoRA target geometry and therefore cannot be resumed. Create a new training "
                "run with the same LoRA configuration and load a checkpoint that contains "
                "adapter/adapter_config.json."
            )

    def build_supported_models(self) -> list[types.SupportedModel]:
        return [
            types.SupportedModel(model_name=model.model_name)
            for model in self.config.supported_models
        ]

    def update_activity(self, model_id: str, user_id: str) -> None:
        record = self.get_run_record(model_id, user_id)
        record.last_request_time = datetime.now(timezone.utc)
        self._save_training_run(model_id)

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
        record = self.get_run_record(model_id, user_id)
        self._require_resumable_run(record)
        self.update_activity(model_id, user_id)

        span_name = (
            "training_controller.run_forward_backward"
            if backward
            else "training_controller.run_forward"
        )
        with _get_tracer().start_as_current_span(span_name) as span:
            span.set_attribute("tuft.training_run_id", model_id)
            span.set_attribute("tuft.session_id", record.session_id)
            span.set_attribute("tuft.backward", backward)
            span.set_attribute("tuft.data_count", len(data))
            span.set_attribute("tuft.loss_fn", loss_fn)

            logger.info("Forward/backward begin for %s", model_id)
            start_time = time.perf_counter()

            # Count total input tokens for metrics
            total_tokens = sum(len(datum.model_input.to_ints()) for datum in data)

            async def _operation() -> types.ForwardBackwardOutput:
                if record.backend is None:
                    raise UnknownModelException(model_name=model_id)
                result = await record.backend.forward(
                    data,
                    lora_id=model_id,
                    loss_fn=loss_fn,
                    loss_fn_config=loss_fn_config,
                    backward=backward,
                )
                return result

            result = await self._with_sequence_guard(record, seq_id, _operation)

            # Record tokens per second metric
            duration = time.perf_counter() - start_time
            if total_tokens > 0 and duration > 0:
                tokens_per_second = total_tokens / duration
                get_metrics().training_tokens_per_second.record(
                    tokens_per_second, {"base_model": record.base_model}
                )

            return result

    async def run_optim_step(
        self, model_id: str, user_id: str, params: types.AdamParams, seq_id: int | None
    ) -> types.OptimStepResponse:
        record = self.get_run_record(model_id, user_id)
        self._require_resumable_run(record)
        self.update_activity(model_id, user_id)

        with _get_tracer().start_as_current_span("training_controller.run_optim_step") as span:
            span.set_attribute("tuft.training_run_id", model_id)
            span.set_attribute("tuft.session_id", record.session_id)
            span.set_attribute("tuft.learning_rate", params.learning_rate)

            logger.info("Optimizer step begin for %s", model_id)

            async def _operation() -> types.OptimStepResponse:
                if record.backend is None:
                    raise UnknownModelException(model_name=model_id)
                result = await record.backend.optim_step(adam_params=params, lora_id=model_id)
                logger.info("Optimizer step completed for %s", model_id)
                return result

            return await self._with_sequence_guard(record, seq_id, _operation)

    async def unload_model(self, model_id: str, user_id: str) -> None:
        # TODO: Ensure that all created training runs can be unloaded to reduce
        # GPU memory usage.
        if model_id not in self.training_runs:
            raise UnknownModelException(model_name=model_id)
        record = self.training_runs[model_id]
        if record.model_owner != user_id:
            raise UserMismatchException()
        base_model = record.base_model
        if record.backend is not None:
            await record.backend.remove_adapter(model_id)
        del self.training_runs[model_id]
        self._delete_training_run(model_id)

        # Update metrics
        get_metrics().training_models_active.add(-1, {"base_model": base_model})

    def list_training_runs(
        self, *, user_id: str, limit: int | None = None, offset: int = 0
    ) -> types.TrainingRunsResponse:
        runs = [
            record.to_training_run()
            for record in self.training_runs.values()
            if record.model_owner == user_id
        ]
        runs.sort(key=lambda run: run.last_request_time, reverse=True)
        total = len(runs)
        start = min(offset, total)
        end = total if limit is None else min(start + limit, total)
        paged = runs[start:end]
        cursor = types.Cursor(offset=offset, limit=limit or total, total_count=total)
        return types.TrainingRunsResponse(training_runs=paged, cursor=cursor)

    def get_training_run_view(self, model_id: str, user_id: str) -> types.TrainingRun:
        record = self.get_run_record(model_id=model_id, user_id=user_id)
        return record.to_training_run()

    def get_model_info(self, model_id: str, user_id: str) -> types.GetInfoResponse:
        record = self.get_run_record(model_id=model_id, user_id=user_id)
        model_data = types.ModelData(
            arch="toy-transformer",
            model_name=record.base_model,
            tokenizer_id=record.base_model,
        )
        return types.GetInfoResponse(
            model_data=model_data,
            model_id=model_id,
            is_lora=True,
            lora_rank=record.lora_rank,
            model_name=record.base_model,
        )

    async def save_checkpoint(
        self,
        model_id: str,
        user_id: str,
        name: str | None,
        checkpoint_type: types.CheckpointType,
        future_id: int = 0,
        seq_id: int | None = None,
    ) -> CheckpointRecord:
        """Save a checkpoint for the given training run."""
        training_run = self.get_run_record(model_id=model_id, user_id=user_id)
        self._require_resumable_run(training_run)

        with _get_tracer().start_as_current_span("training_controller.save_checkpoint") as span:
            span.set_attribute("tuft.training_run_id", model_id)
            span.set_attribute("tuft.session_id", training_run.session_id)
            span.set_attribute("tuft.checkpoint_type", checkpoint_type)

            async def _operation() -> CheckpointRecord:
                counter_attr = (
                    "next_training_checkpoint"
                    if checkpoint_type == "training"
                    else "next_sampler_checkpoint"
                )
                counter = getattr(training_run, counter_attr)
                checkpoint_name = name or f"checkpoint-{counter:04d}"
                checkpoint_id = f"{model_id}/{checkpoint_name}"
                logger.info("Checkpoint save begin: %s", checkpoint_id)

                setattr(training_run, counter_attr, counter + 1)
                assert self.config.checkpoint_dir is not None
                checkpoint = CheckpointRecord.from_training_run(
                    training_run_id=training_run.training_run_id,
                    checkpoint_name=checkpoint_name,
                    owner_name=training_run.model_owner,
                    checkpoint_type=checkpoint_type,
                    checkpoint_root_dir=self.config.checkpoint_dir,
                    exist_ok=True,
                )
                checkpoint.future_id = future_id
                checkpoint.seq_id = seq_id
                target_map = (
                    training_run.checkpoints
                    if checkpoint_type == "training"
                    else training_run.sampler_checkpoints
                )
                if training_run.backend is not None:
                    await training_run.backend.save_state(
                        lora_id=training_run.training_run_id,
                        checkpoint_record=checkpoint,
                        optimizer=(checkpoint_type == "training"),
                    )

                lora_alpha = self._effective_lora_alpha(training_run)

                # Write metadata once so metadata.json exists
                checkpoint.save_metadata(
                    base_model=training_run.base_model,
                    session_id=training_run.session_id,
                    lora_rank=training_run.lora_rank,
                    lora_alpha=lora_alpha,
                    train_attn=training_run.train_attn,
                    train_mlp=training_run.train_mlp,
                    train_unembed=training_run.train_unembed,
                    target_modules=training_run.target_modules,
                )

                # Compute total size including metadata.json
                checkpoint.size_bytes = compute_tree_size(checkpoint.path)

                # Persist the correct size into metadata.json
                checkpoint.save_metadata(
                    base_model=training_run.base_model,
                    session_id=training_run.session_id,
                    lora_rank=training_run.lora_rank,
                    lora_alpha=lora_alpha,
                    train_attn=training_run.train_attn,
                    train_mlp=training_run.train_mlp,
                    train_unembed=training_run.train_unembed,
                    target_modules=training_run.target_modules,
                )
                # save the checkpoint record in the training run
                target_map[checkpoint_name] = checkpoint

                # Save training run and checkpoint atomically to prevent inconsistency
                # if server crashes between saves
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._save_training_run_with_checkpoint,
                    model_id,
                    checkpoint_name,
                    checkpoint_type,
                )

                # Update metrics
                metrics = get_metrics()
                metrics.training_checkpoints_saved.add(
                    1, {"model_id": model_id, "checkpoint_type": checkpoint_type}
                )
                logger.info("Checkpoint saved: %s", checkpoint_id)
                metrics.training_checkpoint_size.record(
                    checkpoint.size_bytes,
                    {"model_id": model_id, "checkpoint_type": checkpoint_type},
                )

                return checkpoint

            return await self._with_sequence_guard(training_run, seq_id, _operation)

    async def load_checkpoint(
        self,
        model_id: str,
        user_id: str,
        path: str,
        optimizer: bool,
        seq_id: int | None = None,
    ) -> None:
        """Load a checkpoint."""
        try:
            assert self.config.checkpoint_dir is not None
            parsed_checkpoint = CheckpointRecord.from_tinker_path(
                path,
                self.config.checkpoint_dir,
            )
        except FileNotFoundError as exc:
            raise CheckpointNotFoundException(checkpoint_id=model_id) from exc
        source_model_id = parsed_checkpoint.training_run_id or model_id
        source_training_run = self.get_run_record(
            source_model_id, user_id, enforce_user_match=False
        )

        collection = (
            source_training_run.checkpoints
            if parsed_checkpoint.checkpoint_type == "training"
            else source_training_run.sampler_checkpoints
        )

        checkpoint = collection.get(parsed_checkpoint.checkpoint_id)
        if checkpoint is None:
            raise CheckpointNotFoundException(checkpoint_id=parsed_checkpoint.checkpoint_id)
        try:
            metadata = checkpoint.metadata
        except FileNotFoundError as exc:
            raise CheckpointMetadataReadException(
                checkpoint_id=parsed_checkpoint.checkpoint_id
            ) from exc
        if not (metadata.public or metadata.owner_name == user_id):
            raise CheckpointAccessDeniedException(checkpoint_id=parsed_checkpoint.checkpoint_id)

        destination_training_run = self.get_run_record(model_id, user_id)
        self._require_resumable_run(destination_training_run)
        if destination_training_run.backend is None:
            raise UnknownModelException(model_name=model_id)

        checkpoint_id = parsed_checkpoint.checkpoint_id
        logger.info("Checkpoint load begin: %s", checkpoint_id)

        async def _operation() -> None:
            assert destination_training_run.backend is not None
            self._check_adapter_compatible(
                checkpoint_id=checkpoint_id,
                checkpoint=checkpoint,
                metadata=metadata,
                destination=destination_training_run,
            )
            await destination_training_run.backend.load_state(
                lora_id=destination_training_run.training_run_id,
                checkpoint_record=checkpoint,
                optimizer=optimizer,
            )
            logger.info("Checkpoint loaded: %s", checkpoint_id)

        await self._with_sequence_guard(destination_training_run, seq_id, _operation)

    def _check_adapter_compatible(
        self,
        checkpoint_id: str,
        checkpoint: CheckpointRecord,
        metadata: CheckpointMetadata,
        destination: TrainingRunRecord,
    ) -> None:
        """Reject a checkpoint whose adapter geometry differs from the destination run.

        Validated against the checkpoint itself rather than the source run record,
        since the checkpoint is what the weights on disk were written from. Loading
        a mismatched adapter is silent rather than loud: peft skips a checkpoint's
        adapter_config.json entirely when the destination adapter already exists,
        and only collects the unmatched keys.

        Target-module geometry is compared as concrete module sets, not raw flags.
        The PEFT adapter config is ground truth, with checkpoint metadata as the
        fallback when it records a concrete module list. Checkpoints that only
        record modifier flags are rejected rather than guessed from server
        configuration that may have changed.
        """
        if metadata.base_model != destination.base_model:
            raise InvalidRequestException(
                f"Cannot load checkpoint {checkpoint_id} from base model "
                f"{metadata.base_model} into a training run on base model "
                f"{destination.base_model}."
            )
        if metadata.lora_rank is not None and metadata.lora_rank != destination.lora_rank:
            raise InvalidRequestException(
                f"Cannot load checkpoint {checkpoint_id} with LoRA rank {metadata.lora_rank} "
                f"into a training run with LoRA rank {destination.lora_rank}."
            )
        # Alpha is baked into every FSDP slot from lora_alpha_ratio and checkpoint
        # loading only copies weights, so validate it independently of geometry.
        checkpoint.validate_lora_alpha(self._effective_lora_alpha(destination))

        saved_modules = checkpoint.saved_target_modules
        if saved_modules is None:
            raise InvalidRequestException(
                f"Cannot load checkpoint {checkpoint_id}: neither metadata nor "
                "adapter_config.json provides a supported explicit target_modules list. "
                "PEFT regex-string targets and checkpoints without an explicit target-module "
                "list cannot be validated."
            )
        if destination.target_modules is None:
            raise InvalidRequestException(
                f"Cannot load checkpoint {checkpoint_id}: destination training run "
                f"{destination.training_run_id} does not record effective LoRA target "
                "modules and is not resumable. Create a new training run that records its "
                "effective target modules."
            )
        source_modules = set(saved_modules)
        destination_modules = set(destination.target_modules)
        if source_modules != destination_modules:
            from .backends.lora_modules import gated_deltanet_mismatch_hint

            hint = gated_deltanet_mismatch_hint(source_modules, destination_modules)
            raise InvalidRequestException(
                f"Cannot load checkpoint {checkpoint_id} targeting LoRA modules "
                f"{sorted(source_modules)} into a training run targeting "
                f"{sorted(destination_modules)}." + (f" {hint}" if hint else "")
            )

    def _model_config_for(self, base_model: str) -> ModelConfig | None:
        return next(
            (model for model in self.config.supported_models if model.model_name == base_model),
            None,
        )

    def _effective_target_modules(
        self, base_model: str, lora_config: types.LoraConfig
    ) -> list[str]:
        """Resolve the concrete geometry persisted for a newly created run."""

        model_config = self._model_config_for(base_model)
        if model_config is None:
            raise UnknownModelException(model_name=base_model)
        if model_config.training_backend == "fsdp" and model_config.fsdp_qv_only:
            # fsdp_qv_only is a documented override of all client modifiers.
            return list(FSDP_QV_TARGET_MODULES)
        if lora_config.train_unembed:
            from .backends.lora_modules import unembed_target_modules

            if unembed_target_modules(str(model_config.model_path)) == []:
                raise InvalidRequestException(
                    f"train_unembed=True has no effect on base model {base_model}: TuFT "
                    "does not support LoRA on the Qwen embedding or unembedding weights, "
                    "and vLLM could not serve such an adapter. Tinker's hosted service "
                    "adapts embed_tokens here, so results would differ from Tinker. Set "
                    "train_unembed=False to create the run without it."
                )
        if model_config.training_backend == "fsdp" and model_config.fsdp_target_modules is not None:
            # Persist the explicit slot geometry. FSDPTrainingBackend separately
            # rejects resolvable client modifiers that request a different set.
            return list(model_config.fsdp_target_modules)
        from .backends.lora_modules import get_target_modules

        try:
            return get_target_modules(
                str(model_config.model_path),
                lora_config,
                qwen_gated_deltanet_full_lora=model_config.qwen_gated_deltanet_full_lora,
            )
        except ValueError as exc:
            raise InvalidRequestException(
                f"Cannot resolve effective LoRA target modules for base model {base_model}: "
                f"{exc}. Use a supported model series or configure explicit FSDP targets."
            ) from exc

    def delete_checkpoint(self, model_id: str, user_id: str, checkpoint_id: str) -> None:
        training_run = self.get_run_record(model_id, user_id)
        removed = training_run.checkpoints.pop(checkpoint_id, None)
        is_sampler = False
        if removed is None:
            removed = training_run.sampler_checkpoints.pop(checkpoint_id, None)
            is_sampler = True
        if removed is None:
            raise CheckpointNotFoundException(checkpoint_id=checkpoint_id)
        removed.delete()

        self._save_training_run(model_id)
        if is_sampler:
            self._delete_sampler_checkpoint_record(model_id, checkpoint_id)
        else:
            self._delete_checkpoint_record(model_id, checkpoint_id)

    def list_checkpoints(self, model_id: str, user_id: str) -> list[types.Checkpoint]:
        training_run = self.get_run_record(model_id, user_id)
        checkpoints = [item.tinker_checkpoint for item in training_run.checkpoints.values()]
        checkpoints += [
            item.tinker_checkpoint for item in training_run.sampler_checkpoints.values()
        ]
        checkpoints.sort(key=lambda ckpt: ckpt.time)
        return checkpoints

    def list_user_checkpoints(
        self,
        user_id: str,
    ) -> list[types.Checkpoint]:
        checkpoints: list[types.Checkpoint] = []
        training_runs = [run for run in self.training_runs.values() if run.model_owner == user_id]
        for run in training_runs:
            checkpoints.extend([item.tinker_checkpoint for item in run.checkpoints.values()])
        checkpoints.sort(key=lambda item: item.time, reverse=True)
        return checkpoints

    def set_visibility(
        self, model_id: str, checkpoint_id: str, user_id: str, *, public: bool
    ) -> None:
        training_run = self.get_run_record(model_id=model_id, user_id=user_id)
        target = training_run.checkpoints.get(checkpoint_id)
        is_sampler = False
        if target is None:
            target = training_run.sampler_checkpoints.get(checkpoint_id)
            is_sampler = True
        if target is None:
            raise CheckpointNotFoundException(checkpoint_id=checkpoint_id)
        target.set_visibility(public)

        if is_sampler:
            self._save_sampler_checkpoint(model_id, checkpoint_id)
        else:
            self._save_checkpoint(model_id, checkpoint_id)

    def build_archive_url(
        self,
        model_id: str,
        user_id: str,
        checkpoint_id: str,
    ) -> types.CheckpointArchiveUrlResponse:
        training_run = self.get_run_record(model_id, user_id)
        checkpoint = training_run.checkpoints.get(
            checkpoint_id
        ) or training_run.sampler_checkpoints.get(checkpoint_id)
        if checkpoint is None:
            raise CheckpointNotFoundException(checkpoint_id=checkpoint_id)
        expires = datetime.now(timezone.utc) + timedelta(minutes=15)
        return types.CheckpointArchiveUrlResponse(url=checkpoint.path.as_uri(), expires=expires)

    def get_weights_info(self, model_id: str, user_id: str) -> types.WeightsInfoResponse:
        training_run = self.get_run_record(model_id, user_id)
        return types.WeightsInfoResponse(
            base_model=training_run.base_model,
            is_lora=True,
            lora_rank=training_run.lora_rank,
        )

    def get_latest_checkpoint(self, model_id: str) -> CheckpointRecord | None:
        record = self.training_runs.get(model_id)
        if record is None:
            return None
        all_checkpoints = list(record.checkpoints.values()) + list(
            record.sampler_checkpoints.values()
        )
        if not all_checkpoints:
            return None
        return max(all_checkpoints, key=lambda c: c.created_at)

    async def restore_from_checkpoint(self, model_id: str) -> CheckpointRecord | None:
        record = self.training_runs.get(model_id)
        if record is None or record.backend is None:
            return None
        if record.has_legacy_lora_state:
            record.corrupted = True
            return None
        latest_ckpt = self.get_latest_checkpoint(model_id)
        if latest_ckpt is None:
            # The run owns no checkpoint to restore from - it has not saved one
            # yet, or it was seeded by load_weights from another run's
            # checkpoint. Recreate the adapter anyway so the run stays usable:
            # without it the backend has no adapter under this id and every
            # later request fails with "Adapter not found" for good.
            try:
                await record.backend.create_adapter(model_id, record.lora_config())
            except Exception:  # pylint: disable=broad-except
                logger.exception("Failed to create adapter for model %s during restore", model_id)
            return None
        # A lora_alpha_ratio change between restarts would resume the run at a
        # different update scale. Mark the run corrupted instead of retrying the
        # load, which cannot succeed while the configs disagree.
        try:
            latest_ckpt.validate_lora_alpha(self._effective_lora_alpha(record))
        except CheckpointIncompatibleException as exc:
            record.corrupted = True
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_training_run, model_id)
            logger.error("Cannot restore %s: %s", model_id, exc.detail)
            return latest_ckpt
        # load_state calls load_adapter which creates the adapter from the
        # checkpoint on disk.  Calling create_adapter first causes PEFT's
        # load_adapter to fail with "adapter already exists" on HFTrainingBackend.
        # Only call create_adapter as a fallback if load_state fails.
        try:
            await record.backend.load_state(
                lora_id=model_id,
                checkpoint_record=latest_ckpt,
                optimizer=(latest_ckpt.checkpoint_type == "training"),
            )
        except Exception as exc:  # pylint: disable=broad-except
            # load_state failed – try create_adapter + load_state as fallback
            logger.warning(
                "load_state failed for %s (%s), trying create_adapter fallback", model_id, exc
            )
            try:
                await record.backend.create_adapter(model_id, record.lora_config())
            except Exception:
                logger.exception("Failed to create adapter for model %s during restore", model_id)
            try:
                await record.backend.load_state(
                    lora_id=model_id,
                    checkpoint_record=latest_ckpt,
                    optimizer=(latest_ckpt.checkpoint_type == "training"),
                )
            except Exception as exc:
                # If loading still fails, mark as corrupted but still return
                # the checkpoint so that futures AFTER the checkpoint are
                # marked as failed (not ALL futures).
                record.corrupted = True
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._save_training_run, model_id)
                logger.warning(
                    "Checkpoint load failed for %s, marking the run corrupted: %s; "
                    "returning checkpoint with future_id=%d for future cleanup",
                    model_id,
                    exc,
                    latest_ckpt.future_id,
                )
                return latest_ckpt

        return latest_ckpt
