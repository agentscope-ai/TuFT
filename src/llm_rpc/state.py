"""In-memory state containers backing the FastAPI endpoints."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, TypeVar

from tinker import types

from .checkpoints import CheckpointRecord
from .config import AppConfig
from .exceptions import SessionNotFoundException
from .futures import FutureStore
from .sampling_controller import SamplingController
from .training_controller import TrainingController, TrainingRunRecord

T = TypeVar("T")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SessionRecord:
    session_id: str
    tags: list[str]
    user_metadata: dict[str, str] | None
    sdk_version: str
    created_at: datetime = field(default_factory=_now)
    last_heartbeat: datetime = field(default_factory=_now)


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
            raise SessionNotFoundException(session_id)
        return record

    def heartbeat(self, session_id: str) -> None:
        record = self.require(session_id)
        record.last_heartbeat = _now()

    def list_sessions(self) -> list[str]:
        return sorted(self._sessions.keys())


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
        self.sampling = SamplingController(self.config)
        self.future_store = FutureStore()

    async def async_init(self) -> None:
        """Put any async initialization logic here"""
        await self.sampling.async_init()

    def create_session(self, request: types.CreateSessionRequest) -> SessionRecord:
        return self.sessions.create_session(request)

    def heartbeat(self, session_id: str) -> None:
        self.sessions.heartbeat(session_id)

    async def create_model(
        self,
        session_id: str,
        base_model: str,
        lora_config: types.LoraConfig,
        user_metadata: dict[str, str] | None,
    ) -> TrainingRunRecord:
        self.sessions.require(session_id)
        return await self.training.create_model(session_id, base_model, lora_config, user_metadata)

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

    async def create_sampling_session(
        self,
        session_id: str,
        base_model: str | None,
        model_path: str | None,
        *,
        session_seq_id: int,
    ) -> str:
        self.sessions.require(session_id)
        return await self.sampling.create_sampling_session(
            session_id=session_id,
            base_model=base_model,
            model_path=model_path,
            session_seq_id=session_seq_id,
        )

    async def run_sample(self, request: types.SampleRequest) -> types.SampleResponse:
        return await self.sampling.run_sample(request)

    async def save_checkpoint(
        self,
        model_id: str,
        name: str | None,
        checkpoint_type: types.CheckpointType,
    ) -> CheckpointRecord:
        return await self.training.save_checkpoint(
            model_id=model_id,
            name=name,
            checkpoint_type=checkpoint_type,
        )

    async def load_checkpoint(self, model_id: str, path: str, optimizer: bool) -> None:
        return await self.training.load_checkpoint(
            model_id=model_id,
            path=path,
            optimizer=optimizer,
        )

    def delete_checkpoint(self, model_id: str, checkpoint_id: str) -> None:
        training_run = self.training.get_run_record(model_id)
        self.training.delete_checkpoint(training_run, checkpoint_id)

    def list_checkpoints(self, model_id: str) -> list[types.Checkpoint]:
        training_run = self.training.get_run_record(model_id)
        return self.training.list_checkpoints(training_run)

    def list_user_checkpoints(self) -> list[types.Checkpoint]:
        return self.training.list_user_checkpoints(self.training.training_runs)

    def set_checkpoint_visibility(self, model_id: str, checkpoint_id: str, *, public: bool) -> None:
        training_run = self.training.get_run_record(model_id)
        self.training.set_visibility(training_run, checkpoint_id, public=public)

    def get_weights_info(self, tinker_path: str) -> types.WeightsInfoResponse:
        parsed = types.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
        training_run = self.training.get_run_record(parsed.training_run_id)
        return self.training.get_weights_info(training_run)

    def build_archive_url(
        self, model_id: str, checkpoint_id: str
    ) -> types.CheckpointArchiveUrlResponse:
        training_run = self.training.get_run_record(model_id)
        return self.training.build_archive_url(training_run, checkpoint_id)

    def list_training_runs(
        self, *, limit: int | None = None, offset: int = 0
    ) -> types.TrainingRunsResponse:
        return self.training.list_training_runs(limit=limit, offset=offset)

    def get_training_run_view(self, model_id: str) -> types.TrainingRun:
        return self.training.get_training_run_view(model_id)

    def get_model_info(self, model_id: str) -> types.GetInfoResponse:
        return self.training.get_model_info(model_id)

    async def unload_model(self, model_id: str) -> None:
        await self.training.unload_model(model_id)
        await self.sampling.evict_model(model_id)

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
