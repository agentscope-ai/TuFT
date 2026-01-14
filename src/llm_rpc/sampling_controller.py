from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Dict, List, Tuple

from tinker import types

from .backends import BaseSamplingBackend
from .checkpoints import CheckpointRecord
from .config import AppConfig, ModelConfig
from .exceptions import (
    CheckpointMetadataReadException,
    CheckpointNotFoundException,
    MissingSequenceIDException,
    SequenceConflictException,
    SessionNotFoundException,
    UnknownModelException,
)
from .persistence import PersistedMarker, persistable, redis_persistent, unwrap_proxy


@persistable
@dataclass
class SamplingSessionRecord:
    sampling_session_id: str
    session_id: str
    model_id: str
    base_model: str
    model_path: str | None
    session_seq_id: int
    last_seq_id: int = -1
    history: list["SamplingHistoryEntry"] = field(default_factory=list)


@persistable
@dataclass
class SamplingHistoryEntry:
    seq_id: int
    prompt_token_count: int
    prompt_hash: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@redis_persistent(
    restore_callback="_rebuild_sampling_state",
)
class SamplingController:
    """Manages sampling sessions and connects them to the correct training or base-model backend."""

    # Persisted field - auto-synced to Redis
    sampling_sessions: Annotated[Dict[str, SamplingSessionRecord], PersistedMarker()]

    def __init__(
        self,
        config: AppConfig,
    ) -> None:
        self.config = config
        # sampling_sessions is auto-initialized by @redis_persistent decorator
        self._base_backends: Dict[str, BaseSamplingBackend] = self._create_backends(
            config.supported_models
        )

    def _rebuild_sampling_state(self) -> None:
        """Restore callback: validate sampling sessions after Redis restore.

        Called only when data is restored from Redis (server restart scenario).
        Note: Adapter re-loading is done asynchronously in async_init.
        """
        invalid_sessions = []
        for session_id, record in list(self.sampling_sessions.items()):
            actual_record = unwrap_proxy(record)
            # Validate base_model is still supported
            if actual_record.base_model and actual_record.base_model not in self._base_backends:
                invalid_sessions.append(session_id)

        # Remove invalid sessions
        for session_id in invalid_sessions:
            del self.sampling_sessions[session_id]
            # TODO: add warning log here (should not delete session if it is not invalid)

    async def async_init(self) -> None:  # Note: run after _rebuild_sampling_state
        """Perform any async initialization here, including adapter reloading."""
        init_tasks = [backend.async_init() for backend in self._base_backends.values()]
        await asyncio.gather(*init_tasks)

        # Re-add adapters in trinity vllm engine
        for session_id, record in list(self.sampling_sessions.items()):
            actual_record = unwrap_proxy(record)
            if actual_record.model_path and actual_record.base_model:
                adapter_path = Path(actual_record.model_path)
                if adapter_path.exists() and actual_record.base_model in self._base_backends:
                    try:
                        backend = self._base_backends[actual_record.base_model]
                        await backend.add_adapter(
                            lora_id=actual_record.sampling_session_id,
                            adapter_path=adapter_path,
                        )
                    except Exception:
                        del self.sampling_sessions[session_id]
                        # TODO: add warning log here (should not delete if not invalid)
                else:
                    # Delete session if adapter doesn't exist OR base_model not in backends
                    del self.sampling_sessions[session_id]
                    # TODO: add warning log here (should not delete session if it is not invalid)

    def _create_backends(self, model_configs: List[ModelConfig]) -> Dict[str, BaseSamplingBackend]:
        backends: Dict[str, BaseSamplingBackend] = {}
        for config in model_configs:
            backends[config.model_name] = BaseSamplingBackend.create_backend(config)
        return backends

    async def create_sampling_session(
        self,
        *,
        session_id: str,
        base_model: str | None,
        model_path: str | None,
        session_seq_id: int,
    ) -> str:
        base_model_ref: str | None = None
        adapter_path: Path | None = None
        sampling_session_id = str(uuid.uuid4())

        if model_path:
            # model_path should have higher priority than base_model
            try:
                parsed_checkpoint = CheckpointRecord.from_tinker_path(
                    model_path, self.config.checkpoint_dir
                )
            except FileNotFoundError as exc:
                raise CheckpointNotFoundException(
                    checkpoint_id=model_path,
                ) from exc

            if not parsed_checkpoint.path.exists():
                raise CheckpointNotFoundException(
                    checkpoint_id=parsed_checkpoint.checkpoint_id,
                )
            try:
                base_model_ref = parsed_checkpoint.metadata.get("base_model")
            except FileNotFoundError as exc:
                raise CheckpointMetadataReadException(
                    checkpoint_id=parsed_checkpoint.checkpoint_id,
                ) from exc
            adapter_path = parsed_checkpoint.adapter_path
            if base_model_ref not in self._base_backends:
                raise UnknownModelException(model_name=base_model_ref)
            sampling_backend = self._base_backends[base_model_ref]
            await sampling_backend.add_adapter(
                lora_id=sampling_session_id, adapter_path=parsed_checkpoint.adapter_path
            )
            # TODO: remove adapter when session is deleted
        elif base_model:
            base_model_ref = base_model
            if base_model_ref not in self._base_backends:
                raise UnknownModelException(model_name=base_model_ref)
        else:
            raise UnknownModelException(model_name="None")
        self.sampling_sessions[sampling_session_id] = SamplingSessionRecord(
            sampling_session_id=sampling_session_id,
            session_id=session_id,
            model_id=sampling_session_id,
            base_model=base_model_ref,
            model_path=str(adapter_path) if adapter_path else None,
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
            raise SequenceConflictException(expected=record.last_seq_id + 1, got=seq_id)
        record.last_seq_id = seq_id
        entry = SamplingHistoryEntry(
            seq_id=seq_id,
            prompt_token_count=len(prompt.to_ints()),
            prompt_hash=self._hash_prompt(prompt),
        )
        record.history.append(entry)

    def _resolve_backend(
        self, request: types.SampleRequest
    ) -> Tuple[BaseSamplingBackend, str | None]:
        """Resolve the appropriate backend for the sampling request.

        Args:
            request: The sampling request.

        Returns:
            A tuple of the resolved backend and the LoRA ID if applicable.
        """
        if request.sampling_session_id:
            record = self.sampling_sessions.get(request.sampling_session_id)
            if record is None:
                raise SessionNotFoundException(session_id=request.sampling_session_id)
            if request.seq_id is None:
                raise MissingSequenceIDException()
            self._record_sequence(record, request.seq_id, request.prompt)
            if record.base_model not in self._base_backends:
                raise UnknownModelException(model_name=record.base_model)
            if record.model_path is None:
                lora_id = None
            else:
                lora_id = record.sampling_session_id
            return self._base_backends[record.base_model], lora_id
        raise SessionNotFoundException(session_id="None")

    async def run_sample(self, request: types.SampleRequest) -> types.SampleResponse:
        backend, lora_id = self._resolve_backend(request)
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
            lora_id=lora_id,
        )

    async def evict_model(self, model_id: str) -> None:
        for sampling_id, record in list(self.sampling_sessions.items()):
            if record.model_id == model_id:
                del self.sampling_sessions[sampling_id]

    def get_sampler_info(
        self, sampler_id: str, default_base_model: str
    ) -> types.GetSamplerResponse:
        record = self.sampling_sessions.get(sampler_id)
        if record is None:
            raise SessionNotFoundException(session_id=sampler_id)
        base = record.base_model
        return types.GetSamplerResponse(
            sampler_id=sampler_id,
            base_model=base or default_base_model,
            model_path=record.model_path,
        )
