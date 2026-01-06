"""In-memory state containers backing the FastAPI endpoints."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, TypeVar

from fastapi import HTTPException, status

from tinker import types

from .backends import BaseSamplingBackend, BaseTrainingBackend
from .config import AppConfig, ModelConfig
from .futures import FutureStore

T = TypeVar("T")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _build_tinker_path(
    training_run_id: str, checkpoint_type: types.CheckpointType, checkpoint_id: str
) -> str:
    folder = "weights" if checkpoint_type == "training" else "sampler_weights"
    return f"tinker://{training_run_id}/{folder}/{checkpoint_id}"


@dataclass
class SessionRecord:
    session_id: str
    tags: list[str]
    user_metadata: dict[str, str] | None
    sdk_version: str
    created_at: datetime = field(default_factory=_now)
    last_heartbeat: datetime = field(default_factory=_now)


@dataclass
class CheckpointRecord:
    checkpoint_id: str
    checkpoint_type: types.CheckpointType
    path: Path
    created_at: datetime
    size_bytes: int
    public: bool = False

    def to_api(self, training_run_id: str) -> types.Checkpoint:
        return types.Checkpoint(
            checkpoint_id=self.checkpoint_id,
            checkpoint_type=self.checkpoint_type,
            time=self.created_at,
            tinker_path=_build_tinker_path(
                training_run_id, self.checkpoint_type, self.checkpoint_id
            ),
            size_bytes=self.size_bytes,
            public=self.public,
        )


@dataclass
class TrainingRunRecord:
    training_run_id: str
    base_model: str
    lora_config: types.LoraConfig
    session_id: str
    backend: BaseTrainingBackend
    user_metadata: dict[str, str] | None
    created_at: datetime = field(default_factory=_now)
    last_request_time: datetime = field(default_factory=_now)
    checkpoints: Dict[str, CheckpointRecord] = field(default_factory=dict)
    sampler_checkpoints: Dict[str, CheckpointRecord] = field(default_factory=dict)
    next_training_checkpoint: int = 1
    next_sampler_checkpoint: int = 1
    corrupted: bool = False
    next_seq_id: int = 1
    _execution_lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._execution_lock = asyncio.Lock()

    def to_training_run(self, owner: str) -> types.TrainingRun:
        training_checkpoint = self._latest_checkpoint(self.checkpoints)
        sampler_checkpoint = self._latest_checkpoint(self.sampler_checkpoints)
        return types.TrainingRun(
            training_run_id=self.training_run_id,
            base_model=self.base_model,
            model_owner=owner,
            is_lora=True,
            corrupted=self.corrupted,
            lora_config=self.lora_config,
            last_request_time=self.last_request_time,
            last_checkpoint=training_checkpoint,
            last_sampler_checkpoint=sampler_checkpoint,
            user_metadata=self.user_metadata,
        )

    def _latest_checkpoint(self, items: Dict[str, CheckpointRecord]) -> types.Checkpoint | None:
        if not items:
            return None
        latest = max(items.values(), key=lambda record: record.created_at)
        return latest.to_api(self.training_run_id)


@dataclass
class SamplingSessionRecord:
    sampling_session_id: str
    session_id: str
    model_id: str | None
    base_model: str | None
    model_path: str | None
    session_seq_id: int
    last_seq_id: int = -1
    history: list["SamplingHistoryEntry"] = field(default_factory=list)


@dataclass
class SamplingHistoryEntry:
    seq_id: int
    prompt_token_count: int
    prompt_hash: str
    created_at: datetime = field(default_factory=_now)


class SessionManager:
    """Maintains session metadata and heartbeats so other controllers can enforce ownership."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionRecord] = {}

    def create_session(self, request: types.CreateSessionRequest) -> SessionRecord:
        session_id = str(uuid.uuid4())
        record = SessionRecord(
            session_id=session_id,
            tags=request.tags,
            user_metadata=request.user_metadata,
            sdk_version=request.sdk_version,
        )
        self._sessions[session_id] = record
        return record

    def require(self, session_id: str) -> SessionRecord:
        record = self._sessions.get(session_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown session")
        return record

    def heartbeat(self, session_id: str) -> None:
        record = self.require(session_id)
        record.last_heartbeat = _now()

    def list_sessions(self) -> list[str]:
        return sorted(self._sessions.keys())


class TrainingController:
    """Tracks training runs, enforces request ordering.

    Routes work into ModelBackend instances.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.training_backends = self._create_backends(config.supported_models)
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
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="sequence_conflict",
            )
        record.next_seq_id += 1

    def create_model(
        self,
        session_id: str,
        base_model: str,
        lora_config: types.LoraConfig,
        user_metadata: dict[str, str] | None,
    ) -> TrainingRunRecord:
        model_id = str(uuid.uuid4())
        if base_model not in self.training_backends:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Unknown base model {base_model}",
            )
        backend = self.training_backends[base_model]
        record = TrainingRunRecord(
            training_run_id=model_id,
            base_model=base_model,
            lora_config=lora_config,
            session_id=session_id,
            backend=backend,
            user_metadata=user_metadata,
        )
        self.training_runs[model_id] = record
        return record

    def get_run_record(self, model_id: str) -> TrainingRunRecord:
        record = self.training_runs.get(model_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown model")
        return record

    def build_supported_models(self) -> list[types.SupportedModel]:
        return [
            types.SupportedModel(model_name=model.model_name)
            for model in self.config.supported_models
        ]

    def update_activity(self, model_id: str) -> None:
        record = self.get_run_record(model_id)
        record.last_request_time = _now()

    async def run_forward(
        self,
        model_id: str,
        data: list[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        seq_id: int | None,
        *,
        backward: bool,
    ) -> types.ForwardBackwardOutput:
        record = self.get_run_record(model_id)
        self.update_activity(model_id)

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
        self, model_id: str, params: types.AdamParams, seq_id: int | None
    ) -> types.OptimStepResponse:
        record = self.get_run_record(model_id)
        self.update_activity(model_id)

        async def _operation() -> types.OptimStepResponse:
            return await record.backend.optim_step(adam_params=params, lora_id=model_id)

        return await self._with_sequence_guard(record, seq_id, _operation)

    def unload_model(self, model_id: str) -> None:
        if model_id not in self.training_runs:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown model")
        del self.training_runs[model_id]

    def list_training_runs(
        self, *, limit: int | None = None, offset: int = 0
    ) -> types.TrainingRunsResponse:
        runs = [
            record.to_training_run(self.config.model_owner)
            for record in self.training_runs.values()
        ]
        runs.sort(key=lambda run: run.last_request_time, reverse=True)
        total = len(runs)
        start = min(offset, total)
        end = total if limit is None else min(start + limit, total)
        paged = runs[start:end]
        cursor = types.Cursor(offset=offset, limit=limit or total, total_count=total)
        return types.TrainingRunsResponse(training_runs=paged, cursor=cursor)

    def get_training_run_view(self, model_id: str) -> types.TrainingRun:
        record = self.get_run_record(model_id)
        return record.to_training_run(self.config.model_owner)

    def get_model_info(self, model_id: str) -> types.GetInfoResponse:
        record = self.get_run_record(model_id)
        model_data = types.ModelData(
            arch="toy-transformer",
            model_name=record.base_model,
            tokenizer_id=record.base_model,
        )
        return types.GetInfoResponse(
            model_data=model_data,
            model_id=model_id,
            is_lora=True,
            lora_rank=record.lora_config.rank,
            model_name=record.base_model,
        )


class CheckpointStore:
    """Bridges in-memory checkpoint metadata with the JSON blobs persisted on disk."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _write_metadata(
        self,
        training_run: TrainingRunRecord,
        checkpoint: CheckpointRecord,
        model_state: object | None = None,
    ) -> None:
        existing_state: object | None = None
        if checkpoint.path.exists():
            try:
                existing = json.loads(checkpoint.path.read_text())
                existing_state = existing.get("model_state")
            except json.JSONDecodeError:
                existing_state = None

        payload = {
            "model_id": training_run.training_run_id,
            "checkpoint_type": checkpoint.checkpoint_type,
            "name": checkpoint.checkpoint_id,
            "created_at": checkpoint.created_at.isoformat(),
            "session_id": training_run.session_id,
            "base_model": training_run.base_model,
            "lora_rank": training_run.lora_config.rank,
            "size_bytes": checkpoint.size_bytes,
            "public": checkpoint.public,
            "tinker_path": checkpoint.to_api(training_run.training_run_id).tinker_path,
            "model_state": model_state if model_state is not None else existing_state,
        }
        checkpoint.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        checkpoint.size_bytes = checkpoint.path.stat().st_size
        payload["size_bytes"] = checkpoint.size_bytes
        checkpoint.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save_checkpoint(
        self,
        training_run: TrainingRunRecord,
        name: str | None,
        checkpoint_type: types.CheckpointType,
    ) -> CheckpointRecord:
        counter_attr = (
            "next_training_checkpoint"
            if checkpoint_type == "training"
            else "next_sampler_checkpoint"
        )
        counter = getattr(training_run, counter_attr)
        checkpoint_name = name or f"checkpoint-{counter:04d}"
        setattr(training_run, counter_attr, counter + 1)
        folder = "weights" if checkpoint_type == "training" else "sampler_weights"
        path = self.config.checkpoint_dir / training_run.training_run_id / folder / checkpoint_name
        path.parent.mkdir(parents=True, exist_ok=True)
        created = _now()
        checkpoint = CheckpointRecord(
            checkpoint_id=checkpoint_name,
            checkpoint_type=checkpoint_type,
            path=path,
            created_at=created,
            size_bytes=0,
        )
        target_map = (
            training_run.checkpoints
            if checkpoint_type == "training"
            else training_run.sampler_checkpoints
        )
        target_map[checkpoint_name] = checkpoint
        return checkpoint

    async def load_checkpoint(
        self, training_run: TrainingRunRecord, path: str, optimizer: bool
    ) -> None:
        _ = optimizer
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(path)
        if parsed.training_run_id != training_run.training_run_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Checkpoint belongs to a different model",
            )
        collection = (
            training_run.checkpoints
            if parsed.checkpoint_type == "training"
            else training_run.sampler_checkpoints
        )
        checkpoint_key = parsed.checkpoint_id.split("/", 1)[-1]
        checkpoint = collection.get(checkpoint_key)
        if checkpoint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Checkpoint not found"
            )
        try:
            document = json.loads(checkpoint.path.read_text())
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupt file
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Corrupted checkpoint metadata",
            ) from exc
        model_state = document.get("model_state")
        if model_state is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Checkpoint missing model state",
            )
        await training_run.backend.load_state(model_state)

    def delete_checkpoint(self, training_run: TrainingRunRecord, checkpoint_id: str) -> None:
        removed = training_run.checkpoints.pop(checkpoint_id, None)
        if removed is None:
            removed = training_run.sampler_checkpoints.pop(checkpoint_id, None)
        if removed is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Checkpoint not found"
            )
        with contextlib.suppress(FileNotFoundError):
            removed.path.unlink()

    def list_checkpoints(self, training_run: TrainingRunRecord) -> list[types.Checkpoint]:
        checkpoints = [
            item.to_api(training_run.training_run_id) for item in training_run.checkpoints.values()
        ]
        checkpoints += [
            item.to_api(training_run.training_run_id)
            for item in training_run.sampler_checkpoints.values()
        ]
        checkpoints.sort(key=lambda ckpt: ckpt.time)
        return checkpoints

    def list_user_checkpoints(
        self, training_runs: Dict[str, TrainingRunRecord]
    ) -> list[types.Checkpoint]:
        checkpoints: list[types.Checkpoint] = []
        for record in training_runs.values():
            checkpoints.extend(self.list_checkpoints(record))
        checkpoints.sort(key=lambda item: item.time, reverse=True)
        return checkpoints

    def set_visibility(
        self, training_run: TrainingRunRecord, checkpoint_id: str, *, public: bool
    ) -> None:
        target = training_run.checkpoints.get(
            checkpoint_id
        ) or training_run.sampler_checkpoints.get(checkpoint_id)
        if target is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Checkpoint not found"
            )
        target.public = public
        self._write_metadata(training_run, target)

    def build_archive_url(
        self, training_run: TrainingRunRecord, checkpoint_id: str
    ) -> types.CheckpointArchiveUrlResponse:
        checkpoint = training_run.checkpoints.get(
            checkpoint_id
        ) or training_run.sampler_checkpoints.get(checkpoint_id)
        if checkpoint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Checkpoint not found"
            )
        expires = _now() + timedelta(minutes=15)
        return types.CheckpointArchiveUrlResponse(url=checkpoint.path.as_uri(), expires=expires)

    def get_weights_info(self, training_run: TrainingRunRecord) -> types.WeightsInfoResponse:
        return types.WeightsInfoResponse(
            base_model=training_run.base_model,
            is_lora=True,
            lora_rank=training_run.lora_config.rank,
        )


class SamplingController:
    """Manages sampling sessions and connects them to the correct training or base-model backend."""

    def __init__(self, config: AppConfig, training_controller: TrainingController) -> None:
        self.config = config
        self._training = training_controller
        self.sampling_sessions: Dict[str, SamplingSessionRecord] = {}
        self._base_backends: Dict[str, BaseSamplingBackend] = self._create_backends(
            config.supported_models
        )

    async def async_init(self) -> None:
        """Perform any async initialization here."""
        init_tasks = [backend.async_init() for backend in self._base_backends.values()]
        await asyncio.gather(*init_tasks)

    def _create_backends(self, model_configs: List[ModelConfig]) -> Dict[str, BaseSamplingBackend]:
        backends: Dict[str, BaseSamplingBackend] = {}
        for config in model_configs:
            backends[config.model_name] = BaseSamplingBackend.create_backend(config)
        return backends

    def create_sampling_session(
        self,
        *,
        session_id: str,
        base_model: str | None,
        model_path: str | None,
        session_seq_id: int,
    ) -> str:
        backend_ref: str | None = None
        base_model_ref: str | None = None
        if base_model:
            base_model_ref = base_model
            backend_ref = base_model
            if base_model not in self._base_backends:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Unknown base model %s".format(),
                )
        elif model_path:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Sampling from model_path is not supported yet",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Missing model reference",
            )
        sampling_session_id = str(uuid.uuid4())
        self.sampling_sessions[sampling_session_id] = SamplingSessionRecord(
            sampling_session_id=sampling_session_id,
            session_id=session_id,
            model_id=backend_ref,
            base_model=base_model_ref,
            model_path=model_path,
            session_seq_id=session_seq_id,
        )
        return sampling_session_id

    def _hash_prompt(self, prompt: types.ModelInput) -> str:
        tokens = ",".join(str(token) for token in prompt.to_ints())
        return hashlib.sha1(tokens.encode("utf-8")).hexdigest()[:16]

    def _record_sequence(
        self, record: SamplingSessionRecord, seq_id: int, prompt: types.ModelInput
    ) -> None:
        if seq_id <= record.last_seq_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="sequence_conflict",
            )
        record.last_seq_id = seq_id
        entry = SamplingHistoryEntry(
            seq_id=seq_id,
            prompt_token_count=len(prompt.to_ints()),
            prompt_hash=self._hash_prompt(prompt),
        )
        record.history.append(entry)

    def _resolve_backend_from_record(self, record: SamplingSessionRecord) -> BaseSamplingBackend:
        # TODO: support model_id mapping to a lora request structure
        if record.base_model:
            return self._base_backends[record.base_model]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Broken sampling session"
        )

    def resolve_backend(self, request: types.SampleRequest) -> BaseSamplingBackend:
        if request.sampling_session_id:
            record = self.sampling_sessions.get(request.sampling_session_id)
            if record is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Unknown sampling session",
                )
            if request.seq_id is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Missing seq_id for sampling session",
                )
            self._record_sequence(record, request.seq_id, request.prompt)
            return self._resolve_backend_from_record(record)
        if request.model_path:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Sampling from trained models is not supported yet",
            )
        if request.base_model:
            backend = self._base_backends.get(request.base_model)
            if backend is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Unknown base model %s".format(),
                )
            return backend
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model specified for sampling",
        )

    async def run_sample(self, request: types.SampleRequest) -> types.SampleResponse:
        backend = self.resolve_backend(request)
        prompt = request.prompt
        sampling_params = request.sampling_params
        num_samples = request.num_samples
        include_prompt_logprobs = bool(request.prompt_logprobs)
        topk_prompt_logprobs = request.topk_prompt_logprobs or 0
        return await backend.sample(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
        )

    def evict_model(self, model_id: str) -> None:
        for sampling_id, record in list(self.sampling_sessions.items()):
            if record.model_id == model_id:
                del self.sampling_sessions[sampling_id]

    def get_sampler_info(
        self, sampler_id: str, default_base_model: str
    ) -> types.GetSamplerResponse:
        record = self.sampling_sessions.get(sampler_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown sampler")
        base = record.base_model
        if base is None and record.model_id:
            base = self._training.get_run_record(record.model_id).base_model
        return types.GetSamplerResponse(
            sampler_id=sampler_id,
            base_model=base or default_base_model,
            model_path=record.model_path,
        )


class ServerState:
    """Application-wide container that wires controllers together
    and exposes a simple façade to FastAPI.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.config.ensure_directories()
        self.config.check_validity()
        self.sessions = SessionManager()
        self.training = TrainingController(self.config)
        self.checkpoints = CheckpointStore(self.config)
        self.sampling = SamplingController(self.config, self.training)
        self.future_store = FutureStore()

    async def async_init(self) -> None:
        """Put any async initialization logic here"""
        await self.sampling.async_init()

    def create_session(self, request: types.CreateSessionRequest) -> SessionRecord:
        return self.sessions.create_session(request)

    def heartbeat(self, session_id: str) -> None:
        self.sessions.heartbeat(session_id)

    def create_model(
        self,
        session_id: str,
        base_model: str,
        lora_config: types.LoraConfig,
        user_metadata: dict[str, str] | None,
    ) -> TrainingRunRecord:
        self.sessions.require(session_id)
        return self.training.create_model(session_id, base_model, lora_config, user_metadata)

    def get_training_run(self, model_id: str) -> TrainingRunRecord:
        return self.training.get_run_record(model_id)

    def build_supported_models(self) -> list[types.SupportedModel]:
        return self.training.build_supported_models()

    async def run_forward(
        self,
        model_id: str,
        data: list[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        seq_id: int | None,
        *,
        backward: bool,
    ) -> types.ForwardBackwardOutput:
        return await self.training.run_forward(
            model_id, data, loss_fn, loss_fn_config, seq_id, backward=backward
        )

    async def run_optim_step(
        self, model_id: str, params: types.AdamParams, seq_id: int | None
    ) -> types.OptimStepResponse:
        return await self.training.run_optim_step(model_id, params, seq_id)

    def create_sampling_session(
        self,
        session_id: str,
        base_model: str | None,
        model_path: str | None,
        *,
        session_seq_id: int,
    ) -> str:
        self.sessions.require(session_id)
        return self.sampling.create_sampling_session(
            session_id=session_id,
            base_model=base_model,
            model_path=model_path,
            session_seq_id=session_seq_id,
        )

    async def run_sample(self, request: types.SampleRequest) -> types.SampleResponse:
        return await self.sampling.run_sample(request)

    def save_checkpoint(
        self,
        model_id: str,
        name: str | None,
        checkpoint_type: types.CheckpointType,
    ) -> CheckpointRecord:
        training_run = self.training.get_run_record(model_id)
        return self.checkpoints.save_checkpoint(training_run, name, checkpoint_type)

    async def load_checkpoint(self, model_id: str, path: str, optimizer: bool) -> None:
        training_run = self.training.get_run_record(model_id)
        await self.checkpoints.load_checkpoint(training_run, path, optimizer)

    def delete_checkpoint(self, model_id: str, checkpoint_id: str) -> None:
        training_run = self.training.get_run_record(model_id)
        self.checkpoints.delete_checkpoint(training_run, checkpoint_id)

    def list_checkpoints(self, model_id: str) -> list[types.Checkpoint]:
        training_run = self.training.get_run_record(model_id)
        return self.checkpoints.list_checkpoints(training_run)

    def list_user_checkpoints(self) -> list[types.Checkpoint]:
        return self.checkpoints.list_user_checkpoints(self.training.training_runs)

    def set_checkpoint_visibility(self, model_id: str, checkpoint_id: str, *, public: bool) -> None:
        training_run = self.training.get_run_record(model_id)
        self.checkpoints.set_visibility(training_run, checkpoint_id, public=public)

    def get_weights_info(self, tinker_path: str) -> types.WeightsInfoResponse:
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        training_run = self.training.get_run_record(parsed.training_run_id)
        return self.checkpoints.get_weights_info(training_run)

    def build_archive_url(
        self, model_id: str, checkpoint_id: str
    ) -> types.CheckpointArchiveUrlResponse:
        training_run = self.training.get_run_record(model_id)
        return self.checkpoints.build_archive_url(training_run, checkpoint_id)

    def list_training_runs(
        self, *, limit: int | None = None, offset: int = 0
    ) -> types.TrainingRunsResponse:
        return self.training.list_training_runs(limit=limit, offset=offset)

    def get_training_run_view(self, model_id: str) -> types.TrainingRun:
        return self.training.get_training_run_view(model_id)

    def get_model_info(self, model_id: str) -> types.GetInfoResponse:
        return self.training.get_model_info(model_id)

    def unload_model(self, model_id: str) -> None:
        self.training.unload_model(model_id)
        self.sampling.evict_model(model_id)

    def get_session_overview(self, session_id: str) -> types.GetSessionResponse:
        self.sessions.require(session_id)
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
        self, *, limit: int | None = None, offset: int = 0
    ) -> types.ListSessionsResponse:
        sessions = self.sessions.list_sessions()
        total = len(sessions)
        start = min(offset, total)
        if limit is None:
            subset = sessions[start:]
        else:
            subset = sessions[start : min(start + limit, total)]
        return types.ListSessionsResponse(sessions=subset)

    def get_sampler_info(self, sampler_id: str) -> types.GetSamplerResponse:
        return self.sampling.get_sampler_info(
            sampler_id, self.config.supported_models[0].model_name
        )
