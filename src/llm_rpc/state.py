"""In-memory state containers backing the FastAPI endpoints."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Dict, TypeVar

from tinker import types

from .auth import AuthenticationDB, User
from .backends import BaseSamplingBackend, BaseTrainingBackend
from .checkpoints import CheckpointRecord
from .config import AppConfig
from .exceptions import SessionNotFoundException, UserMismatchException
from .futures import FutureStore
from .persistence import PersistedMarker, persistable, redis_persistent
from .persistence.serializers import ModelSerializer
from .sampling_controller import SamplingController
from .training_controller import TrainingController, TrainingRunRecord

T = TypeVar("T")

# TODO: remove this after we have a better way to mark/detect non-serializable types
ModelSerializer.add_skip_types(BaseTrainingBackend, BaseSamplingBackend)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@persistable
@dataclass
class SessionRecord:
    session_id: str
    tags: list[str]
    user_metadata: dict[str, str] | None
    user_id: str
    sdk_version: str
    created_at: datetime = field(default_factory=_now)
    last_heartbeat: datetime = field(default_factory=_now)


@redis_persistent()
class SessionManager:
    """Maintains session metadata and heartbeats so other controllers can enforce ownership."""

    # Persisted field - auto-synced to Redis
    _sessions: Annotated[Dict[str, SessionRecord], PersistedMarker()]

    def __init__(self) -> None:
        # _sessions is auto-initialized by @redis_persistent decorator
        pass

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

    def list_sessions(self, user_id: str) -> list[str]:
        """List all session IDs belonging to the given user."""
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
        self.training = TrainingController(self.config)
        self.sampling = SamplingController(self.config)
        self.auth_db = AuthenticationDB(self.config.authorized_users)
        self.future_store = FutureStore()

    async def async_init(self) -> None:
        """Put any async initialization logic here"""
        await self.sampling.async_init()
        await self._restore_from_checkpoints()

    async def _restore_from_checkpoints(self) -> None:
        """Restore server state from checkpoints after Redis restore.

        This method handles checkpoint-based recovery:
        1. For each model_id with pending futures, find the latest checkpoint
        2. Load the checkpoint to restore model state
        3. Mark all futures created after the checkpoint as failed
        4. For models without checkpoints, mark all pending futures as failed
        """
        if not self.future_store.needs_checkpoint_recovery:
            return

        pending_by_model = self.future_store.get_pending_futures_by_model()

        for model_id, _pending_futures in pending_by_model.items():
            if model_id is None:
                # Futures without model_id (e.g., sample operations)
                # Mark all as failed since we can't restore them
                self.future_store.mark_futures_failed_after_checkpoint(
                    model_id=None,
                    checkpoint_time=None,
                    error_message="Server restarted. Please retry.",
                )
                continue

            # Try to restore from checkpoint
            latest_ckpt = await self.training.restore_from_checkpoint(model_id)

            if latest_ckpt is None:
                # No checkpoint available - mark all pending as failed
                self.future_store.mark_futures_failed_after_checkpoint(
                    model_id=model_id,
                    checkpoint_time=None,
                    error_message=f"No checkpoint found for model {model_id}. Please retry.",
                )
            else:
                # Mark futures after the checkpoint as failed
                self.future_store.mark_futures_failed_after_checkpoint(
                    model_id=model_id,
                    checkpoint_time=latest_ckpt.created_at,
                    error_message=(
                        f"Server restored from checkpoint {latest_ckpt.checkpoint_id}. "
                        "Operations after this checkpoint need to be retried."
                    ),
                )

        self.future_store.clear_recovery_flag()

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
        return await self.training.run_forward(
            model_id=model_id,
            user_id=user_id,
            data=data,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            seq_id=seq_id,
            backward=backward,
        )

    async def run_optim_step(
        self, model_id: str, user_id: str, params: types.AdamParams, seq_id: int | None
    ) -> types.OptimStepResponse:
        return await self.training.run_optim_step(
            model_id=model_id, user_id=user_id, params=params, seq_id=seq_id
        )

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
        return await self.sampling.run_sample(request, user_id=user_id)

    async def save_checkpoint(
        self,
        model_id: str,
        user_id: str,
        name: str | None,
        checkpoint_type: types.CheckpointType,
    ) -> CheckpointRecord:
        return await self.training.save_checkpoint(
            model_id=model_id,
            user_id=user_id,
            name=name,
            checkpoint_type=checkpoint_type,
        )

    async def load_checkpoint(
        self, model_id: str, user_id: str, path: str, optimizer: bool
    ) -> None:
        return await self.training.load_checkpoint(
            model_id=model_id,
            user_id=user_id,
            path=path,
            optimizer=optimizer,
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

    def get_model_info(self, model_id: str, user_id: str) -> types.GetInfoResponse:
        return self.training.get_model_info(model_id, user_id=user_id)

    async def unload_model(self, model_id: str, user_id: str) -> None:
        await self.training.unload_model(model_id, user_id=user_id)
        await self.sampling.evict_model(model_id, user_id=user_id)

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
