from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional

from tinker import types

from tuft.backends.base_backend import BaseSamplingBackend, BaseTrainingBackend
from tuft.checkpoints import CheckpointRecord
from tuft.config import ModelConfig


class FlexBackendMode(StrEnum):
    TRAINING = "training"
    SAMPLING = "sampling"


class TransformDirection(StrEnum):
    TRAINING_TO_SAMPLING = "training_to_sampling"
    SAMPLING_TO_TRAINING = "sampling_to_training"


@dataclass(frozen=True)
class TransformResult:
    direction: TransformDirection
    source_mode: FlexBackendMode
    target_mode: FlexBackendMode
    base_model_transformed: bool
    zero_copy: bool
    supported: bool
    message: str
    metrics: dict[str, float] | None = None
    source_released: bool = False


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _metrics_from(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(key): float(metric) for key, metric in value.items() if _is_number(metric)}
    metrics = getattr(value, "metrics", None)
    if isinstance(metrics, dict):
        return {str(key): float(metric) for key, metric in metrics.items() if _is_number(metric)}
    return {}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _zero_copy_from(value: Any) -> bool:
    zero_copy = getattr(value, "zero_copy", None)
    if isinstance(zero_copy, bool):
        return zero_copy
    if isinstance(value, dict):
        raw = value.get("zero_copy")
        if isinstance(raw, bool):
            return raw
        if _is_number(raw):
            return bool(raw)
    return False


class FlexBackend(BaseTrainingBackend, BaseSamplingBackend):  # pyright: ignore[reportIncompatibleMethodOverride]
    """Backend base for converting one base model between training and sampling.

    A transform changes base-model ownership from the source mode to the target
    mode. Concrete implementations must not keep source-side training/sampling
    runtime state after a successful transform; only target-side base model state
    and unavoidable base storage ownership needed by the target mode may remain.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._mode = FlexBackendMode.TRAINING
        self._base_model_transformed_to_sampling = False
        self._last_transform_metrics: dict[str, float] = {}
        self._vllm_base_alias_ready = False
        self._transform_lock = asyncio.Lock()

    @property
    def mode(self) -> FlexBackendMode:
        return self._mode

    @property
    def base_model_transformed_to_sampling(self) -> bool:
        return self._base_model_transformed_to_sampling

    @property
    def last_transform_metrics(self) -> dict[str, float]:
        return dict(self._last_transform_metrics)

    async def async_init(self) -> None:
        pass

    async def transform(
        self,
        direction: TransformDirection | str,
        *,
        force: bool = False,
    ) -> TransformResult:
        direction = TransformDirection(direction)
        if direction == TransformDirection.TRAINING_TO_SAMPLING:
            return await self.transform_to_sampling(force=force)
        if direction == TransformDirection.SAMPLING_TO_TRAINING:
            return await self.transform_to_training(force=force)
        raise ValueError(f"Unsupported transform direction: {direction}")

    async def transform_to_sampling(self, *, force: bool = False) -> TransformResult:
        async with self._transform_lock:
            source_mode = self._mode
            if self._base_model_transformed_to_sampling and not force:
                self._mode = FlexBackendMode.SAMPLING
                return TransformResult(
                    direction=TransformDirection.TRAINING_TO_SAMPLING,
                    source_mode=source_mode,
                    target_mode=FlexBackendMode.SAMPLING,
                    base_model_transformed=False,
                    zero_copy=self._vllm_base_alias_ready,
                    supported=True,
                    message="Backend is already in sampling mode.",
                    metrics=dict(self._last_transform_metrics),
                    source_released=False,
                )

            started = time.perf_counter()
            transform_result = await _maybe_await(self._transform_to_sampling_impl(force=force))
            metrics = _metrics_from(transform_result)
            metrics["transform_ms"] = (time.perf_counter() - started) * 1000
            supported = not (metrics.get("base_transform_supported") == 0.0)
            if not supported:
                return TransformResult(
                    direction=TransformDirection.TRAINING_TO_SAMPLING,
                    source_mode=source_mode,
                    target_mode=source_mode,
                    base_model_transformed=False,
                    zero_copy=False,
                    supported=False,
                    message="Backend does not support training-to-sampling transform.",
                    metrics=dict(metrics),
                    source_released=False,
                )

            zero_copy = _zero_copy_from(transform_result)
            self._mode = FlexBackendMode.SAMPLING
            self._base_model_transformed_to_sampling = True
            self._vllm_base_alias_ready = zero_copy
            self._last_transform_metrics = metrics
            source_released = bool(metrics.get("source_released", 0.0))
            return TransformResult(
                direction=TransformDirection.TRAINING_TO_SAMPLING,
                source_mode=source_mode,
                target_mode=FlexBackendMode.SAMPLING,
                base_model_transformed=True,
                zero_copy=zero_copy,
                supported=True,
                message="Backend transformed to sampling mode.",
                metrics=dict(metrics),
                source_released=source_released,
            )

    async def transform_to_training(self, *, force: bool = False) -> TransformResult:
        async with self._transform_lock:
            source_mode = self._mode
            if self._mode == FlexBackendMode.TRAINING and not force:
                return TransformResult(
                    direction=TransformDirection.SAMPLING_TO_TRAINING,
                    source_mode=source_mode,
                    target_mode=FlexBackendMode.TRAINING,
                    base_model_transformed=False,
                    zero_copy=self._vllm_base_alias_ready,
                    supported=True,
                    message="Backend is already in training mode.",
                    metrics=dict(self._last_transform_metrics),
                    source_released=False,
                )

            started = time.perf_counter()
            transform_result = await _maybe_await(self._transform_to_training_impl(force=force))
            metrics = _metrics_from(transform_result)
            metrics["transform_ms"] = (time.perf_counter() - started) * 1000
            supported = not (metrics.get("base_transform_supported") == 0.0)
            if not supported:
                return TransformResult(
                    direction=TransformDirection.SAMPLING_TO_TRAINING,
                    source_mode=source_mode,
                    target_mode=source_mode,
                    base_model_transformed=False,
                    zero_copy=self._vllm_base_alias_ready,
                    supported=False,
                    message="Backend does not support sampling-to-training transform.",
                    metrics=dict(metrics),
                    source_released=False,
                )

            self._mode = FlexBackendMode.TRAINING
            self._last_transform_metrics = metrics
            zero_copy = _zero_copy_from(transform_result) or self._vllm_base_alias_ready
            source_released = bool(metrics.get("source_released", 0.0))
            return TransformResult(
                direction=TransformDirection.SAMPLING_TO_TRAINING,
                source_mode=source_mode,
                target_mode=FlexBackendMode.TRAINING,
                base_model_transformed=True,
                zero_copy=zero_copy,
                supported=True,
                message="Backend transformed to training mode.",
                metrics=dict(metrics),
                source_released=source_released,
            )

    async def _transform_to_sampling_impl(self, *, force: bool = False) -> dict[str, float]:
        return {"base_transform_supported": 0.0}

    async def _transform_to_training_impl(self, *, force: bool = False) -> dict[str, float]:
        return {}

    async def forward(
        self,
        data: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool = False,
    ) -> types.ForwardBackwardOutput:
        raise NotImplementedError(
            "FlexBackend training forward is implemented by concrete backends."
        )

    async def create_adapter(self, lora_id: str, lora_config: types.LoraConfig) -> None:
        raise NotImplementedError("FlexBackend create_adapter is implemented by concrete backends.")

    async def remove_adapter(self, lora_id: str) -> None:
        raise NotImplementedError("FlexBackend remove_adapter is implemented by concrete backends.")

    async def optim_step(
        self,
        adam_params: types.AdamParams,
        lora_id: str,
    ) -> types.OptimStepResponse:
        raise NotImplementedError("FlexBackend optim_step is implemented by concrete backends.")

    async def save_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer: bool,
    ) -> None:
        raise NotImplementedError("FlexBackend save_state is implemented by concrete backends.")

    async def load_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer: bool,
    ) -> None:
        raise NotImplementedError("FlexBackend load_state is implemented by concrete backends.")

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        raise NotImplementedError("FlexBackend sampling is implemented by concrete backends.")

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        raise NotImplementedError("FlexBackend add_adapter is implemented by concrete backends.")

    def get_openai_api_url(self) -> Optional[str]:
        return None
