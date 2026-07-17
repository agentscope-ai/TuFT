from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tuft.backends.flex import FlexBackendMode
from tuft.backends.flex.torchtp import FusedTorchTPVLLMFlexBackend
from tuft.backends.flex.torchtp_zero_copy import (
    call_collective_rpc,
    make_cuda_ipc_descriptor_dict,
    summarize_injection_results,
)
from tuft.config import ModelConfig


class FakeTensor:
    is_cuda = True
    dtype = "fake.bfloat16"

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.contiguous_called = False

    def contiguous(self) -> "FakeTensor":
        self.contiguous_called = True
        return self


class FakeTrainingModel:
    def __init__(self) -> None:
        self.eval_called = 0
        self.train_called = 0
        self.released = False
        self.from_base_storage = False

    def eval(self) -> None:
        self.eval_called += 1

    def train(self) -> None:
        self.train_called += 1


class FakeVLLMEngine:
    def __init__(self) -> None:
        self.collective_rpc_calls: list[tuple[Any, tuple[Any, ...]]] = []
        self.released = False

    def collective_rpc(self, fn: Any, args: tuple[Any, ...]) -> list[dict[str, Any]]:
        self.collective_rpc_calls.append((fn, args))
        return [
            {
                "rank": 0,
                "injected": 2,
                "verified": 2,
                "mismatched": 0,
                "skipped": 1,
                "max_diff": 0.0,
            }
        ]


class FakeRemoteMethod:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def remote(self, *args: Any) -> list[dict[str, Any]]:
        self.calls.append(args)
        return [{"rank": 0, "injected": 1, "verified": 1}]


class FakeActorEngine:
    def __init__(self) -> None:
        self.collective_rpc = FakeRemoteMethod()


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model_name="fake-model",
        model_path=Path("/tmp/fake-model"),
        max_model_len=128,
        tensor_parallel_size=1,
    )


def fake_descriptor_factory(tensor: FakeTensor) -> tuple[str, tuple[int, ...]]:
    return ("fake-ipc", tensor.shape)


def fake_state_dict_builder(model: FakeTrainingModel) -> dict[str, FakeTensor]:
    return {
        "model.layers.0.self_attn.qkv_proj.weight": FakeTensor((4, 4)),
        "model.layers.0.mlp.gate_up_proj.weight": FakeTensor((8, 4)),
    }


def fake_training_runtime_releaser(model: FakeTrainingModel) -> None:
    model.released = True


def fake_sampling_runtime_releaser(engine: FakeVLLMEngine) -> None:
    engine.released = True


def fake_training_runtime_factory(
    config: ModelConfig,
    base_storage: list[Any],
    descriptors: list[dict[str, dict[str, Any]]] | None,
) -> FakeTrainingModel:
    assert config.model_name == "fake-model"
    assert base_storage
    assert descriptors
    model = FakeTrainingModel()
    model.from_base_storage = True
    return model


def test_make_cuda_ipc_descriptor_dict_uses_descriptor_factory() -> None:
    tensors = {"weight": FakeTensor((2, 3))}

    descriptors, keepalive = make_cuda_ipc_descriptor_dict(
        tensors,
        rank=0,
        world_size=1,
        descriptor_factory=fake_descriptor_factory,
        require_cuda=True,
    )

    assert descriptors["weight"]["shape"] == (2, 3)
    assert descriptors["weight"]["ipc"] == ("fake-ipc", (2, 3))
    assert keepalive == [tensors["weight"]]
    assert tensors["weight"].contiguous_called


@pytest.mark.asyncio
async def test_fused_torchtp_vllm_transform_injects_via_collective_rpc(
    model_config: ModelConfig,
) -> None:
    training_model = FakeTrainingModel()
    engine = FakeVLLMEngine()
    backend = FusedTorchTPVLLMFlexBackend(
        model_config,
        training_model=training_model,
        vllm_engine=engine,
        state_dict_builder=fake_state_dict_builder,
        descriptor_factory=fake_descriptor_factory,
        training_runtime_releaser=fake_training_runtime_releaser,
        require_cuda_ipc=True,
    )

    result = await backend.transform_to_sampling()

    assert result.supported
    assert result.zero_copy
    assert result.base_model_transformed
    assert backend.mode == FlexBackendMode.SAMPLING
    assert training_model.eval_called == 1
    assert len(engine.collective_rpc_calls) == 1
    _, args = engine.collective_rpc_calls[0]
    all_rank_descriptors, verify = args
    assert verify is False
    assert len(all_rank_descriptors) == 1
    assert set(all_rank_descriptors[0]) == {
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
    }
    assert result.metrics is not None
    assert result.metrics["ipc_injected:sum"] == 2.0
    assert result.metrics["ipc_descriptors:sum"] == 2.0
    assert result.source_released
    assert backend.training_model is None
    assert training_model.released


@pytest.mark.asyncio
async def test_fused_torchtp_vllm_reverse_transform_requires_training_builder(
    model_config: ModelConfig,
) -> None:
    training_model = FakeTrainingModel()
    backend = FusedTorchTPVLLMFlexBackend(
        model_config,
        training_model=training_model,
        vllm_engine=FakeVLLMEngine(),
        state_dict_builder=fake_state_dict_builder,
        descriptor_factory=fake_descriptor_factory,
        require_cuda_ipc=True,
    )

    await backend.transform_to_sampling()
    result = await backend.transform_to_training()

    assert not result.supported
    assert not result.base_model_transformed
    assert not result.source_released
    assert backend.mode == FlexBackendMode.SAMPLING
    assert training_model.train_called == 0
    assert backend.training_model is None


@pytest.mark.asyncio
async def test_fused_torchtp_vllm_reverse_transform_builds_training_and_releases_sampling(
    model_config: ModelConfig,
) -> None:
    training_model = FakeTrainingModel()
    engine = FakeVLLMEngine()
    backend = FusedTorchTPVLLMFlexBackend(
        model_config,
        training_model=training_model,
        vllm_engine=engine,
        state_dict_builder=fake_state_dict_builder,
        descriptor_factory=fake_descriptor_factory,
        training_runtime_factory=fake_training_runtime_factory,
        sampling_runtime_releaser=fake_sampling_runtime_releaser,
        require_cuda_ipc=True,
    )

    await backend.transform_to_sampling()
    result = await backend.transform_to_training()

    assert result.supported
    assert result.base_model_transformed
    assert result.zero_copy
    assert result.source_released
    assert backend.mode == FlexBackendMode.TRAINING
    assert engine.released
    assert backend.vllm_engine is None
    assert isinstance(backend.training_model, FakeTrainingModel)
    assert backend.training_model.from_base_storage


def test_summarize_injection_results_preserves_zero_copy_metrics() -> None:
    metrics = summarize_injection_results(
        [
            {"injected": 2, "verified": 2, "mismatched": 0, "skipped": 1, "max_diff": 0.0},
            {"injected": 3, "verified": 3, "mismatched": 0, "skipped": 2, "max_diff": 0.5},
        ]
    )

    assert metrics["zero_copy"] == 1.0
    assert metrics["base_transform_supported"] == 1.0
    assert metrics["ipc_injected:sum"] == 5.0
    assert metrics["ipc_skipped:sum"] == 3.0
    assert metrics["ipc_max_diff:max"] == 0.5


@pytest.mark.asyncio
async def test_call_collective_rpc_actor_passes_args_as_third_positional() -> None:
    engine = FakeActorEngine()
    payload = ([{"weight": {"ipc": ("fake",)}}], False)

    result = await call_collective_rpc(engine, lambda worker: worker, payload)

    assert result == [{"rank": 0, "injected": 1, "verified": 1}]
    assert len(engine.collective_rpc.calls) == 1
    method, timeout, args, kwargs = engine.collective_rpc.calls[0]
    assert callable(method)
    assert timeout is None
    assert args is payload
    assert kwargs is None
