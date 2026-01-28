"""Training controller for managing training runs and routing requests."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Dict, List, TypeVar

from opentelemetry.trace import StatusCode
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from tinker import types

from .backends import BaseTrainingBackend
from .checkpoints import CheckpointRecord
from .config import AppConfig, ModelConfig
from .exceptions import (
    CheckpointAccessDeniedException,
    CheckpointMetadataReadException,
    CheckpointNotFoundException,
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
        for config in model_configs:
            backends[config.model_name] = BaseTrainingBackend.create_backend(config)
        return backends

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
                self._reserve_seq_id(record, seq_id)
                # Save the updated next_seq_id to Redis
                self._save_training_run(record.training_run_id)
            return await operation()

    def _reserve_seq_id(self, record: TrainingRunRecord, seq_id: int) -> None:
        expected = record.next_seq_id
        if seq_id != expected:
            raise SequenceConflictException(expected=expected, got=seq_id)
        record.next_seq_id += 1

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
                record = TrainingRunRecord(
                    training_run_id=model_id,
                    base_model=base_model,
                    lora_rank=lora_config.rank,
                    session_id=session_id,
                    model_owner=model_owner,
                    user_metadata=user_metadata,
                    backend=backend,
                )
                await backend.create_adapter(model_id, lora_config)
                self.training_runs[model_id] = record
                self._save_training_run(model_id)

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
                logger.info("Forward/backward completed for %s", model_id)
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
                checkpoint.size_bytes = checkpoint.path.stat().st_size
                checkpoint.save_metadata(
                    base_model=training_run.base_model,
                    session_id=training_run.session_id,
                    lora_rank=training_run.lora_rank,
                )
                # save the checkpoint record in the training run
                target_map[checkpoint_name] = checkpoint

                # Save training run and checkpoint atomically to prevent inconsistency
                # if server crashes between saves
                self._save_training_run_with_checkpoint(model_id, checkpoint_name, checkpoint_type)

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
            parsed_checkpoint = CheckpointRecord.from_tinker_path(path, self.config.checkpoint_dir)
        except FileNotFoundError as exc:
            raise CheckpointNotFoundException(checkpoint_id=model_id) from exc
        source_model_id = parsed_checkpoint.training_run_id or model_id
        training_run = self.get_run_record(source_model_id, user_id, enforce_user_match=False)

        collection = (
            training_run.checkpoints
            if parsed_checkpoint.checkpoint_type == "training"
            else training_run.sampler_checkpoints
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
        if metadata.public or (metadata.owner_name == user_id):
            if training_run.backend is None:
                raise UnknownModelException(model_name=model_id)

            checkpoint_id = parsed_checkpoint.checkpoint_id
            logger.info("Checkpoint load begin: %s", checkpoint_id)

            async def _operation() -> None:
                assert training_run.backend is not None
                await training_run.backend.load_state(
                    lora_id=training_run.training_run_id,
                    checkpoint_record=checkpoint,
                    optimizer=optimizer,
                )
                logger.info("Checkpoint loaded: %s", checkpoint_id)

            await self._with_sequence_guard(training_run, seq_id, _operation)
        else:
            raise CheckpointAccessDeniedException(checkpoint_id=parsed_checkpoint.checkpoint_id)

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
        latest_ckpt = self.get_latest_checkpoint(model_id)
        if latest_ckpt is None:
            return None
        record = self.training_runs.get(model_id)
        if record is None or record.backend is None:
            return None
        try:
            await record.backend.create_adapter(model_id, types.LoraConfig(rank=record.lora_rank))
        except Exception:
            logger.exception("Failed to create adapter for model %s during restore", model_id)
        try:
            await record.backend.load_state(
                lora_id=model_id,
                checkpoint_record=latest_ckpt,
                optimizer=(latest_ckpt.checkpoint_type == "training"),
            )
        except Exception:  # pylint: disable=broad-except
            # If loading fails, mark as corrupted
            record.corrupted = True
            self._save_training_run(model_id)
            return None

        return latest_ckpt
