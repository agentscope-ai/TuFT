from __future__ import annotations

from pathlib import Path

import pytest

from tuft.backends.flex import FlexBackend, FlexBackendMode, TransformDirection
from tuft.config import ModelConfig


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model_name="fake-model",
        model_path=Path("/tmp/fake-model"),
        max_model_len=128,
    )


class TransformOnlyFlexBackend(FlexBackend):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.to_sampling_calls = 0
        self.to_training_calls = 0

    async def _transform_to_sampling_impl(self, *, force: bool = False) -> dict[str, float]:
        self.to_sampling_calls += 1
        return {
            "zero_copy": 1.0,
            "to_sampling_force": float(force),
            "to_sampling_calls": float(self.to_sampling_calls),
        }

    async def _transform_to_training_impl(self, *, force: bool = False) -> dict[str, float]:
        self.to_training_calls += 1
        return {
            "to_training_force": float(force),
            "to_training_calls": float(self.to_training_calls),
        }


@pytest.mark.asyncio
async def test_transform_to_sampling_switches_mode(model_config: ModelConfig) -> None:
    backend = TransformOnlyFlexBackend(model_config)

    result = await backend.transform_to_sampling()

    assert result.supported
    assert result.base_model_transformed
    assert result.zero_copy
    assert result.direction == TransformDirection.TRAINING_TO_SAMPLING
    assert result.source_mode == FlexBackendMode.TRAINING
    assert result.target_mode == FlexBackendMode.SAMPLING
    assert backend.mode == FlexBackendMode.SAMPLING
    assert backend.base_model_transformed_to_sampling
    assert backend.to_sampling_calls == 1


@pytest.mark.asyncio
async def test_transform_to_sampling_skips_repeated_switch(model_config: ModelConfig) -> None:
    backend = TransformOnlyFlexBackend(model_config)

    first = await backend.transform_to_sampling()
    second = await backend.transform_to_sampling()
    forced = await backend.transform_to_sampling(force=True)

    assert first.base_model_transformed
    assert not second.base_model_transformed
    assert second.supported
    assert forced.base_model_transformed
    assert backend.to_sampling_calls == 2
    assert forced.metrics is not None
    assert forced.metrics["to_sampling_force"] == 1.0


@pytest.mark.asyncio
async def test_transform_to_training_switches_mode(model_config: ModelConfig) -> None:
    backend = TransformOnlyFlexBackend(model_config)

    await backend.transform_to_sampling()
    result = await backend.transform_to_training(force=True)

    assert result.supported
    assert result.base_model_transformed
    assert result.direction == TransformDirection.SAMPLING_TO_TRAINING
    assert result.source_mode == FlexBackendMode.SAMPLING
    assert result.target_mode == FlexBackendMode.TRAINING
    assert backend.mode == FlexBackendMode.TRAINING
    assert backend.to_training_calls == 1
    assert result.metrics is not None
    assert result.metrics["to_training_force"] == 1.0


@pytest.mark.asyncio
async def test_default_transform_reports_unsupported(model_config: ModelConfig) -> None:
    backend = FlexBackend(model_config)

    result = await backend.transform(TransformDirection.TRAINING_TO_SAMPLING)

    assert not result.supported
    assert not result.base_model_transformed
    assert result.source_mode == FlexBackendMode.TRAINING
    assert result.target_mode == FlexBackendMode.TRAINING
    assert backend.mode == FlexBackendMode.TRAINING


@pytest.mark.asyncio
async def test_transform_to_training_is_idempotent(model_config: ModelConfig) -> None:
    backend = TransformOnlyFlexBackend(model_config)

    result = await backend.transform_to_training()

    assert result.supported
    assert not result.base_model_transformed
    assert result.source_mode == FlexBackendMode.TRAINING
    assert result.target_mode == FlexBackendMode.TRAINING
    assert backend.to_training_calls == 0
