"""Training controller for managing training runs and routing requests."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Dict, List, TypeVar

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

T = TypeVar("T")


@dataclass
class TrainingRunRecord:
    training_run_id: str
    base_model: str
    lora_rank: int
    session_id: str
    backend: BaseTrainingBackend
    model_owner: str
    user_metadata: dict[str, str] | None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_request_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checkpoints: Dict[str, "CheckpointRecord"] = field(default_factory=dict)
    sampler_checkpoints: Dict[str, "CheckpointRecord"] = field(default_factory=dict)
    next_training_checkpoint: int = 1
    next_sampler_checkpoint: int = 1
    corrupted: bool = False
    next_seq_id: int = 1
    _execution_lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._execution_lock = asyncio.Lock()

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

    def _latest_checkpoint(self, items: Dict[str, "CheckpointRecord"]) -> types.Checkpoint | None:
        if not items:
            return None
        latest = max(items.values(), key=lambda record: record.created_at)
        return latest.tinker_checkpoint


class TrainingController:
    """Tracks training runs, enforces request ordering.

    Routes work into ModelBackend instances.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.training_backends = self._create_backends(config.supported_models)
        # TODO: add a mechanism to manage training_runs
        self.training_runs: Dict[str, TrainingRunRecord] = {}

    def _create_backends(self, model_configs: List[ModelConfig]) -> Dict[str, BaseTrainingBackend]:
        backends: Dict[str, BaseTrainingBackend] = {}
        for config in model_configs:
            backends[config.model_name] = BaseTrainingBackend.create_backend(config)
        return backends

    async def _with_sequence_guard(
        self,
        record: TrainingRunRecord,
        seq_id: int | None,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        async with record._execution_lock:
            if seq_id is not None:
                self._reserve_seq_id(record, seq_id)
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
        if base_model not in self.training_backends:
            raise UnknownModelException(model_name=base_model)
        backend = self.training_backends[base_model]
        record = TrainingRunRecord(
            training_run_id=model_id,
            base_model=base_model,
            lora_rank=lora_config.rank,
            session_id=session_id,
            model_owner=model_owner,
            backend=backend,
            user_metadata=user_metadata,
        )
        await backend.create_adapter(model_id, lora_config)
        self.training_runs[model_id] = record
        return record

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

        async def _operation() -> types.ForwardBackwardOutput:
            return await record.backend.forward(
                data,
                lora_id=model_id,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
                backward=backward,
            )

        return await self._with_sequence_guard(record, seq_id, _operation)

    async def run_optim_step(
        self, model_id: str, user_id: str, params: types.AdamParams, seq_id: int | None
    ) -> types.OptimStepResponse:
        record = self.get_run_record(model_id, user_id)
        self.update_activity(model_id, user_id)

        async def _operation() -> types.OptimStepResponse:
            return await record.backend.optim_step(adam_params=params, lora_id=model_id)

        return await self._with_sequence_guard(record, seq_id, _operation)

    async def unload_model(self, model_id: str, user_id: str) -> None:
        # TODO: Ensure that all created training runs can be unloaded to reduce
        # GPU memory usage.
        if model_id not in self.training_runs:
            raise UnknownModelException(model_name=model_id)
        if self.training_runs[model_id].model_owner != user_id:
            raise UserMismatchException()
        await self.training_runs[model_id].backend.remove_adapter(model_id)
        del self.training_runs[model_id]

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
    ) -> CheckpointRecord:
        """Save a checkpoint for the given training run."""
        training_run = self.get_run_record(model_id=model_id, user_id=user_id)
        counter_attr = (
            "next_training_checkpoint"
            if checkpoint_type == "training"
            else "next_sampler_checkpoint"
        )
        counter = getattr(training_run, counter_attr)
        checkpoint_name = name or f"checkpoint-{counter:04d}"
        setattr(training_run, counter_attr, counter + 1)
        checkpoint = CheckpointRecord.from_training_run(
            training_run_id=training_run.training_run_id,
            checkpoint_name=checkpoint_name,
            owner_name=training_run.model_owner,
            checkpoint_type=checkpoint_type,
            checkpoint_root_dir=self.config.checkpoint_dir,
            exist_ok=True,
        )
        target_map = (
            training_run.checkpoints
            if checkpoint_type == "training"
            else training_run.sampler_checkpoints
        )
        await training_run.backend.save_state(
            lora_id=training_run.training_run_id,
            checkpoint_record=checkpoint,
            # only "training" need to save optimizer
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
        return checkpoint

    async def load_checkpoint(
        self,
        model_id: str,
        user_id: str,
        path: str,
        optimizer: bool,
    ) -> None:
        """Load a checkpoint."""
        training_run = self.get_run_record(model_id, user_id, enforce_user_match=False)
        try:
            parsed_checkpoint = CheckpointRecord.from_tinker_path(path, self.config.checkpoint_dir)
        except FileNotFoundError as exc:
            raise CheckpointNotFoundException(checkpoint_id=model_id) from exc

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
            await training_run.backend.load_state(
                lora_id=training_run.training_run_id,
                checkpoint_record=checkpoint,
                optimizer=optimizer,
            )
        else:
            raise CheckpointAccessDeniedException(checkpoint_id=parsed_checkpoint.checkpoint_id)

    def delete_checkpoint(self, model_id: str, user_id: str, checkpoint_id: str) -> None:
        training_run = self.get_run_record(model_id, user_id)
        removed = training_run.checkpoints.pop(checkpoint_id, None)
        if removed is None:
            removed = training_run.sampler_checkpoints.pop(checkpoint_id, None)
        if removed is None:
            raise CheckpointNotFoundException(checkpoint_id=checkpoint_id)
        removed.delete()

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
        target = training_run.checkpoints.get(
            checkpoint_id
        ) or training_run.sampler_checkpoints.get(checkpoint_id)
        if target is None:
            raise CheckpointNotFoundException(checkpoint_id=checkpoint_id)
        target.set_visibility(public)

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
