from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tinker import types

from tuft.backends.flex import FlexBackendMode
from tuft.backends.flex.torchtp import FusedTorchTPVLLMFlexBackend
from tuft.backends.flex.torchtp_training import FusedQKVLoRA, RowwiseLoRALinear
from tuft.backends.flex.torchtp_zero_copy import (
    call_collective_rpc,
    make_cuda_ipc_descriptor_dict,
    summarize_injection_results,
)
from tuft.checkpoints import CheckpointRecord
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


class FakeTrainableModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.adapter_biases = torch.nn.ParameterDict()
        self.active_adapter: str | None = None
        self.config = SimpleNamespace(vocab_size=8)
        self.train_called = 0

    def train(self, mode: bool = True):
        self.train_called += 1
        return super().train(mode)

    def forward(self, input_ids, attention_mask=None, return_dict=True, **_kwargs):
        batch, seq_len = input_ids.shape
        vocab = self.config.vocab_size
        vocab_offsets = torch.arange(vocab, dtype=torch.float32, device=input_ids.device)
        adapter_bias = torch.tensor(0.0, device=input_ids.device)
        if self.active_adapter is not None:
            adapter_bias = self.adapter_biases[self.active_adapter].sum()
        logits = (self.base_bias + adapter_bias) * vocab_offsets.view(1, 1, vocab)
        logits = logits.expand(batch, seq_len, vocab).contiguous()
        return SimpleNamespace(logits=logits)


@pytest.fixture
def model_config() -> ModelConfig:
    # Pin the conservative post-init-alias / destroy-recreate path so these
    # focused unit tests (whose fake engines only implement the injection
    # collective_rpc) stay isolated from the optimal sleep/wake + pre-capture
    # default config. Tests that target those features opt in via model_copy.
    return ModelConfig(
        model_name="fake-model",
        model_path=Path("/tmp/fake-model"),
        max_model_len=128,
        tensor_parallel_size=1,
        sampling_enforce_eager=True,
        sampling_pre_capture_alias=False,
        sampling_disable_custom_all_reduce=False,
        sampling_enable_sleep_mode=False,
        sampling_keep_runtime_on_training=False,
        sampling_flex_kv_cache_only_sleep=False,
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


def test_fused_torchtp_vllm_compilation_config_disables_graph_for_no_eager(
    model_config: ModelConfig,
) -> None:
    backend = FusedTorchTPVLLMFlexBackend(model_config)
    assert backend._vllm_compilation_config() is None

    no_eager_config = model_config.model_copy(update={"sampling_enforce_eager": False})
    no_eager_backend = FusedTorchTPVLLMFlexBackend(no_eager_config)
    assert no_eager_backend._vllm_compilation_config() == {"cudagraph_mode": 0}

    pre_capture_config = model_config.model_copy(
        update={
            "sampling_enforce_eager": False,
            "sampling_pre_capture_alias": True,
        }
    )
    pre_capture_backend = FusedTorchTPVLLMFlexBackend(pre_capture_config)
    assert pre_capture_backend._vllm_compilation_config() is None

    graph_disabled_config = model_config.model_copy(update={"sampling_disable_cudagraph": True})
    graph_disabled_backend = FusedTorchTPVLLMFlexBackend(graph_disabled_config)
    assert graph_disabled_backend._vllm_compilation_config() == {"cudagraph_mode": 0}


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


@pytest.mark.asyncio
async def test_fused_torchtp_vllm_second_transform_reuses_ipc_descriptors(
    model_config: ModelConfig,
) -> None:
    """Steady-state switch skips state_dict/descriptor rebuild via descriptor reuse."""
    factory_calls: list[Any] = []

    def counting_descriptor_factory(tensor: FakeTensor) -> tuple[str, tuple[int, ...]]:
        factory_calls.append(tensor)
        return ("fake-ipc", tensor.shape)

    builder_calls: list[Any] = []

    def counting_state_dict_builder(model: FakeTrainingModel) -> dict[str, FakeTensor]:
        builder_calls.append(model)
        return fake_state_dict_builder(model)

    engines: list[FakeVLLMEngine] = []

    def engine_factory(config: ModelConfig) -> FakeVLLMEngine:
        engine = FakeVLLMEngine()
        engines.append(engine)
        return engine

    training_model = FakeTrainingModel()
    backend = FusedTorchTPVLLMFlexBackend(
        model_config,
        training_model=training_model,
        vllm_engine_factory=engine_factory,
        state_dict_builder=counting_state_dict_builder,
        descriptor_factory=counting_descriptor_factory,
        training_runtime_factory=fake_training_runtime_factory,
        sampling_runtime_releaser=fake_sampling_runtime_releaser,
        require_cuda_ipc=True,
    )

    first = await backend.transform_to_sampling()
    assert first.supported
    assert first.metrics is not None
    assert first.metrics["descriptor_reused:sum"] == 0.0
    first_factory_calls = len(factory_calls)
    assert first_factory_calls == 2
    assert len(builder_calls) == 1

    await backend.transform_to_training()
    second = await backend.transform_to_sampling()

    assert second.supported
    assert second.metrics is not None
    # Second switch reuses cached descriptors: no new factory/builder calls.
    assert second.metrics["descriptor_reused:sum"] == 1.0
    assert len(factory_calls) == first_factory_calls
    assert len(builder_calls) == 1
    # New engine still received the cached descriptors via inject.
    assert len(engines) == 2
    _, args = engines[1].collective_rpc_calls[0]
    all_rank_descriptors, _verify = args
    assert set(all_rank_descriptors[0]) == {
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
    }

    # force=True must bypass reuse and rebuild descriptors.
    await backend.transform_to_training()
    forced = await backend.transform_to_sampling(force=True)
    assert forced.metrics is not None
    assert forced.metrics["descriptor_reused:sum"] == 0.0
    assert len(factory_calls) == first_factory_calls * 2


@pytest.mark.asyncio
async def test_fused_torchtp_vllm_can_sleep_and_wake_sampling_runtime(
    model_config: ModelConfig,
) -> None:
    config = model_config.model_copy(
        update={
            "sampling_keep_runtime_on_training": True,
            "sampling_flex_kv_cache_only_sleep": True,
            "sampling_sleep_wake_tags": ["weights", "kv_cache"],
        }
    )
    training_model = FakeTrainingModel()
    engine = FakeVLLMEngine()
    backend = FusedTorchTPVLLMFlexBackend(
        config,
        training_model=training_model,
        vllm_engine=engine,
        state_dict_builder=fake_state_dict_builder,
        descriptor_factory=fake_descriptor_factory,
        training_runtime_factory=fake_training_runtime_factory,
        sampling_runtime_releaser=fake_sampling_runtime_releaser,
        require_cuda_ipc=True,
    )
    calls: list[str] = []

    async def fake_sleep() -> dict[str, float]:
        calls.append("sleep")
        backend._sampling_runtime_asleep = True
        return {"sampling_runtime_slept:sum": 1.0, "sampling_sleep_ms:max": 12.0}

    async def fake_wake() -> dict[str, float]:
        calls.append("wake")
        backend._sampling_runtime_asleep = False
        return {"sampling_runtime_woke:sum": 1.0, "sampling_wake_ms:max": 7.0}

    backend._sleep_sampling_runtime = fake_sleep  # type: ignore[method-assign]
    backend._wake_sampling_runtime = fake_wake  # type: ignore[method-assign]

    await backend.transform_to_sampling()
    to_training = await backend.transform_to_training()
    assert calls == ["sleep"]
    assert backend.mode == FlexBackendMode.TRAINING
    assert backend.vllm_engine is engine
    assert not engine.released
    assert to_training.metrics is not None
    assert to_training.metrics["sampling_runtime_slept:sum"] == 1.0

    to_sampling = await backend.transform_to_sampling()
    assert calls == ["sleep", "wake"]
    assert backend.mode == FlexBackendMode.SAMPLING
    assert to_sampling.metrics is not None
    assert to_sampling.metrics["sampling_runtime_woke:sum"] == 1.0


@pytest.mark.asyncio
async def test_fused_torchtp_vllm_rebuilds_slept_runtime_for_lora_snapshots(
    model_config: ModelConfig,
) -> None:
    config = model_config.model_copy(
        update={
            "sampling_keep_runtime_on_training": True,
            "sampling_enable_sleep_mode": True,
            "sampling_pre_capture_alias": False,
            "sampling_rebuild_runtime_for_lora": True,
        }
    )
    old_engine = FakeVLLMEngine()
    new_engine = FakeVLLMEngine()
    wake_calls: list[str] = []

    backend = FusedTorchTPVLLMFlexBackend(
        config,
        training_model=FakeTrainingModel(),
        vllm_engine=old_engine,
        state_dict_builder=fake_state_dict_builder,
        vllm_engine_factory=lambda _config: new_engine,
        descriptor_factory=fake_descriptor_factory,
        training_runtime_factory=fake_training_runtime_factory,
        sampling_runtime_releaser=fake_sampling_runtime_releaser,
        require_cuda_ipc=True,
    )
    backend._sampling_runtime_asleep = True

    async def fake_wake() -> dict[str, float]:
        wake_calls.append("wake")
        backend._sampling_runtime_asleep = False
        return {"sampling_runtime_woke:sum": 1.0}

    def fake_snapshot() -> None:
        backend._sampling_adapter_snapshots = {
            "adapter": {"rank": 1, "alpha": 1, "tensors": {}, "step": 0}
        }

    backend._wake_sampling_runtime = fake_wake  # type: ignore[method-assign]
    backend._snapshot_adapters_for_sampling = fake_snapshot  # type: ignore[method-assign]
    backend._register_all_sampling_adapters = lambda: 0  # type: ignore[method-assign]

    result = await backend.transform_to_sampling()

    assert result.supported
    assert wake_calls == []
    assert old_engine.released
    assert backend.vllm_engine is new_engine
    assert not backend._sampling_runtime_asleep


@pytest.mark.asyncio
async def test_fused_torchtp_vllm_rebuilds_slept_runtime_for_lora_snapshots(
    model_config: ModelConfig,
) -> None:
    config = model_config.model_copy(
        update={
            "sampling_keep_runtime_on_training": True,
            "sampling_enable_sleep_mode": True,
            "sampling_pre_capture_alias": False,
            "sampling_rebuild_runtime_for_lora": True,
        }
    )
    old_engine = FakeVLLMEngine()
    new_engine = FakeVLLMEngine()
    wake_calls: list[str] = []

    backend = FusedTorchTPVLLMFlexBackend(
        config,
        training_model=FakeTrainingModel(),
        vllm_engine=old_engine,
        state_dict_builder=fake_state_dict_builder,
        vllm_engine_factory=lambda _config: new_engine,
        descriptor_factory=fake_descriptor_factory,
        training_runtime_factory=fake_training_runtime_factory,
        sampling_runtime_releaser=fake_sampling_runtime_releaser,
        require_cuda_ipc=True,
    )
    backend._sampling_runtime_asleep = True

    async def fake_wake() -> dict[str, float]:
        wake_calls.append("wake")
        backend._sampling_runtime_asleep = False
        return {"sampling_runtime_woke:sum": 1.0}

    def fake_snapshot() -> None:
        backend._sampling_adapter_snapshots = {
            "adapter": {"rank": 1, "alpha": 1, "tensors": {}, "step": 0}
        }

    backend._wake_sampling_runtime = fake_wake  # type: ignore[method-assign]
    backend._snapshot_adapters_for_sampling = fake_snapshot  # type: ignore[method-assign]
    backend._register_all_sampling_adapters = lambda: 0  # type: ignore[method-assign]

    result = await backend.transform_to_sampling()

    assert result.supported
    assert wake_calls == []
    assert old_engine.released
    assert backend.vllm_engine is new_engine
    assert not backend._sampling_runtime_asleep


def test_fused_torchtp_lora_wrappers_register_multiple_ranks() -> None:
    import tuft.backends.flex.torchtp_training as torchtp_training

    class MiniAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_size = 2
            self.kv_size = 2
            self.qkv_proj = torch.nn.Linear(4, 6, bias=False)
            self.o_proj = torch.nn.Linear(6, 4, bias=False)

    class MiniLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = MiniAttention()

    class MiniInner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([MiniLayer()])

    class MiniModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = MiniInner()

    model = MiniModel()
    attention = model.model.layers[0].self_attn

    params_a = torchtp_training.apply_fused_torchtp_lora(
        model,
        lora_rank=4,
        lora_alpha=4,
        lora_id="rank4",
    )
    params_b = torchtp_training.apply_fused_torchtp_lora(
        model,
        lora_rank=2,
        lora_alpha=2,
        lora_id="rank2",
    )

    assert isinstance(attention.qkv_proj, FusedQKVLoRA)
    assert isinstance(attention.o_proj, RowwiseLoRALinear)
    assert len(params_a) == 8
    assert len(params_b) == 8
    assert attention.qkv_proj.adapter_parameters("rank4")[0].shape == (4, 4)
    assert attention.qkv_proj.adapter_parameters("rank2")[0].shape == (2, 4)
    torchtp_training.set_fused_torchtp_lora_adapter(model, "rank4")
    assert attention.qkv_proj.active_adapter == "rank4"
    torchtp_training.set_fused_torchtp_lora_adapter(model, "rank2")
    assert attention.qkv_proj.active_adapter == "rank2"


def test_peft_export_matches_adapter_params_and_keeps_base_frozen(tmp_path: Path) -> None:
    """fused_adapter_peft_state_dict must export PEFT-format tensors equal to the
    in-memory adapter parameters, and must never modify base weights."""
    import json

    import tuft.backends.flex.torchtp_training as torchtp_training

    class MiniAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_size = 2
            self.kv_size = 2
            self.qkv_proj = torch.nn.Linear(4, 6, bias=False)
            self.o_proj = torch.nn.Linear(6, 4, bias=False)

    class MiniLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = MiniAttention()
            self.mlp = torch.nn.Module()
            self.mlp.gate_size = 3
            self.mlp.up_size = 3
            self.mlp.gate_up_proj = torch.nn.Linear(4, 6, bias=False)
            self.mlp.down_proj = torch.nn.Linear(6, 4, bias=False)

    class MiniInner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([MiniLayer()])

    class MiniModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = MiniInner()
            self.lm_head = torch.nn.Linear(4, 8, bias=False)

    torch.manual_seed(7)
    model = MiniModel()
    attention = model.model.layers[0].self_attn
    torchtp_training.apply_fused_torchtp_lora(model, lora_rank=2, lora_alpha=2, lora_id="a")

    qkv_adapter = attention.qkv_proj.adapters[attention.qkv_proj._adapter_key_by_id["a"]]
    o_adapter = attention.o_proj.adapters[attention.o_proj._adapter_key_by_id["a"]]
    with torch.no_grad():
        qkv_adapter.q_B.normal_(0, 0.1)
        o_adapter.lora_B.normal_(0, 0.1)

    original_qkv_w = attention.qkv_proj.base.weight.detach().clone()
    original_o_w = attention.o_proj.base.weight.detach().clone()

    tensors = torchtp_training.fused_adapter_peft_state_dict(model, "a")
    attn_prefix = "base_model.model.model.layers.0.self_attn"
    mlp_prefix = "base_model.model.model.layers.0.mlp"
    assert set(tensors) == {
        f"{attn_prefix}.{proj}.lora_{ab}.weight"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
        for ab in ("A", "B")
    } | {
        f"{mlp_prefix}.{proj}.lora_{ab}.weight"
        for proj in ("gate_proj", "up_proj", "down_proj")
        for ab in ("A", "B")
    }
    assert torch.equal(tensors[f"{attn_prefix}.q_proj.lora_A.weight"], qkv_adapter.q_A.detach().cpu())
    assert torch.equal(tensors[f"{attn_prefix}.q_proj.lora_B.weight"], qkv_adapter.q_B.detach().cpu())
    assert torch.equal(tensors[f"{attn_prefix}.o_proj.lora_B.weight"], o_adapter.lora_B.detach().cpu())

    # Base weights must remain untouched by the export.
    assert torch.equal(attention.qkv_proj.base.weight, original_qkv_w)
    assert torch.equal(attention.o_proj.base.weight, original_o_w)

    # PEFT dir round-trip: files exist and config is valid JSON with right rank.
    torchtp_training.write_peft_adapter_dir(
        tmp_path / "adapter",
        tensors,
        rank=2,
        alpha=2,
        base_model_name_or_path="test-model",
    )
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()
    config = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert config["r"] == 2
    assert config["peft_type"] == "LORA"
    assert config["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


@pytest.mark.asyncio
async def test_fused_torchtp_training_api_runs_forward_backward_and_optim_step(
    model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import tuft.backends.flex.torchtp_training as torchtp_training

    trainable_model = FakeTrainableModel()

    def fake_apply_lora(
        model,
        *,
        lora_rank,
        lora_alpha,
        tp_group=None,
        lora_id="default",
        **_kwargs,
    ):
        assert model is trainable_model
        parameter = torch.nn.Parameter(torch.zeros(lora_rank))
        model.adapter_biases[lora_id] = parameter
        model.active_adapter = lora_id
        return [parameter]

    def fake_set_adapter(model, lora_id):
        model.active_adapter = lora_id

    def fake_remove_adapter(model, lora_id):
        model.adapter_biases.pop(lora_id, None)
        if model.active_adapter == lora_id:
            model.active_adapter = None

    monkeypatch.setattr(torchtp_training, "apply_fused_torchtp_lora", fake_apply_lora)
    monkeypatch.setattr(torchtp_training, "set_fused_torchtp_lora_adapter", fake_set_adapter)
    monkeypatch.setattr(
        torchtp_training,
        "remove_fused_torchtp_lora_adapter",
        fake_remove_adapter,
    )
    backend = FusedTorchTPVLLMFlexBackend(model_config, training_model=trainable_model)

    await backend.create_adapter("adapter", types.LoraConfig(rank=4, seed=123))
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=[1, 2, 3], dtype="int64", shape=[3]),
            "weights": types.TensorData(data=[1.0, 1.0, 1.0], dtype="float32", shape=[3]),
        },
    )
    output = await backend.forward(
        data=[datum],
        lora_id="adapter",
        loss_fn="cross_entropy",
        loss_fn_config=None,
        backward=True,
    )
    optim = await backend.optim_step(types.AdamParams(learning_rate=1e-3), lora_id="adapter")

    assert output.metrics["loss:sum"] > 0
    assert output.loss_fn_outputs[0]["logprobs"].shape == [3]
    assert optim.metrics["step:max"] == 1.0
    assert backend._training_step == 1

    checkpoint = CheckpointRecord(
        checkpoint_id="ckpt",
        owner_name="tester",
        checkpoint_type="training",
        training_run_id="run",
        path=tmp_path / "ckpt",
    )
    saved_bias = trainable_model.adapter_biases["adapter"].detach().clone()
    await backend.save_state("adapter", checkpoint, optimizer=True)
    trainable_model.adapter_biases["adapter"].data.add_(10.0)
    await backend.load_state("adapter", checkpoint, optimizer=True)
    assert torch.allclose(trainable_model.adapter_biases["adapter"].detach(), saved_bias)

    await backend.create_adapter("adapter_rank2", types.LoraConfig(rank=2, seed=456))
    assert trainable_model.adapter_biases["adapter"].shape == (4,)
    assert trainable_model.adapter_biases["adapter_rank2"].shape == (2,)
    await backend.forward(
        data=[datum],
        lora_id="adapter_rank2",
        loss_fn="cross_entropy",
        loss_fn_config=None,
        backward=True,
    )
    assert trainable_model.active_adapter == "adapter_rank2"
    await backend.forward(
        data=[datum],
        lora_id="adapter",
        loss_fn="cross_entropy",
        loss_fn_config=None,
        backward=False,
    )
    assert trainable_model.active_adapter == "adapter"


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


def test_register_all_sampling_adapters_skips_unchanged(model_config, monkeypatch) -> None:
    """Incremental registration: an adapter whose training step is unchanged
    since the last flip must not be re-exported or re-registered."""
    from tuft.backends.flex import torchtp as torchtp_mod

    backend = torchtp_mod.FusedTorchTPVLLMFlexBackend(model_config)
    backend.vllm_engine = FakeVLLMEngine()
    monkeypatch.setattr(backend, "_rank", lambda: 0)

    exports: list[str] = []
    registers: list[str] = []

    def fake_write(adapter_dir, tensors, **kwargs):
        exports.append(str(adapter_dir))

    from tuft.backends.flex import torchtp_training as tt_mod

    monkeypatch.setattr(tt_mod, "write_peft_adapter_dir", fake_write, raising=False)
    monkeypatch.setattr(
        backend,
        "_register_lora_with_engine",
        lambda lora_id, path: registers.append(lora_id) or True,
    )

    backend._sampling_adapter_snapshots = {
        "a": {"rank": 4, "alpha": 8, "tensors": {"x": 1}, "step": 1},
        "b": {"rank": 4, "alpha": 8, "tensors": {"x": 1}, "step": 1},
    }

    # First flip: both adapters are new -> both exported + registered.
    n1 = backend._register_all_sampling_adapters()
    assert n1 == 2
    assert sorted(registers) == ["a", "b"]

    # Second flip, no weight change (same steps): both skipped.
    exports.clear()
    registers.clear()
    n2 = backend._register_all_sampling_adapters()
    assert n2 == 0
    assert exports == []
    assert registers == []

    # 'a' advances a training step -> only 'a' is re-exported and re-registered.
    backend._sampling_adapter_snapshots["a"]["step"] = 2
    exports.clear()
    registers.clear()
    n3 = backend._register_all_sampling_adapters()
    assert registers == ["a"]
    assert n3 == 1


def test_adapter_export_runs_off_the_flip_path(model_config, monkeypatch) -> None:
    """The PEFT write must happen right after optim_step, not while a flip blocks.

    A flip that reuses a background export must not write the adapter again; that
    inline write (once per resident adapter) was the dominant flip cost.
    """
    import asyncio

    from tuft.backends.flex import torchtp as torchtp_mod
    from tuft.backends.flex import torchtp_training as tt_mod

    backend = torchtp_mod.FusedTorchTPVLLMFlexBackend(model_config)
    backend.vllm_engine = FakeVLLMEngine()
    monkeypatch.setattr(backend, "_rank", lambda: 0)
    monkeypatch.setattr(backend, "_effective_lora_options", lambda cfg: {"rank": 16, "alpha": 32})
    backend.training_model = object()
    backend._adapter_configs = {"a": object()}

    writes: list[str] = []
    registers: list[str] = []
    monkeypatch.setattr(
        tt_mod, "fused_adapter_peft_state_dict", lambda model, lora_id: {"w": 1}, raising=False
    )
    monkeypatch.setattr(
        tt_mod,
        "write_peft_adapter_dir",
        lambda adapter_dir, tensors, **kw: writes.append(str(adapter_dir)),
        raising=False,
    )
    monkeypatch.setattr(
        backend, "_register_lora_with_engine", lambda lora_id, path: registers.append(lora_id) or True
    )

    async def main() -> None:
        backend._adapter_update_step["a"] = 1
        backend.schedule_adapter_export("a")
        # The snapshot is taken synchronously; the write is still in flight.
        assert backend._sampling_adapter_snapshots["a"]["step"] == 1
        await backend.drain_adapter_exports()

    asyncio.run(main())
    assert writes == [str(backend._sampling_adapter_dirs["a"])], writes
    assert backend._sampling_adapter_exported_step["a"] == 1

    # A flip now only registers: no second write for the same adapter version.
    assert backend._register_all_sampling_adapters() == 1
    assert len(writes) == 1, writes
    assert registers == ["a"]

    # A new optimizer step invalidates it: the next flip must write again.
    backend._adapter_update_step["a"] = 2
    backend._sampling_adapter_snapshots["a"]["step"] = 2
    backend._register_all_sampling_adapters()
    assert len(writes) == 2, writes


def test_offload_named_tensors_batches_one_transfer_per_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter offload must stage per dtype, not copy tensor by tensor.

    A per-tensor ``.cpu()`` blocks on its own device sync. One adapter holds
    hundreds of LoRA tensors and a multi-tenant run keeps tens of adapters
    resident, which made the offload -- not the zero-copy transform -- the
    dominant term in the training->sampling flip (save_state 3.5-7.8s against
    162-244ms for the transform), so the duty cycle could not amortize a flip
    against a training gap. Counting the staging concatenations keeps that
    regression from creeping back in.
    """
    cat_calls = []
    real_cat = torch.cat

    def counting_cat(tensors, *args, **kwargs):  # type: ignore[no-untyped-def]
        tensors = list(tensors)
        cat_calls.append(len(tensors))
        return real_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", counting_cat)

    named = [
        ("a.weight", torch.randn(4, 8, dtype=torch.bfloat16)),
        ("b.weight", torch.randn(16, dtype=torch.bfloat16)),
        ("c.weight", torch.randn(3, 3, dtype=torch.float32)),
        ("d.weight", None),
    ]
    out = FusedTorchTPVLLMFlexBackend._offload_named_tensors_to_cpu(named)

    # bfloat16 group of 2 and float32 group of 1: one staging copy per dtype.
    assert sorted(cat_calls) == [1, 2]

    assert out["d.weight"] is None, "a missing gradient must stay missing"
    for name, tensor in named:
        if tensor is None:
            continue
        saved = out[name]
        assert saved.device.type == "cpu"
        assert saved.shape == tensor.shape
        assert saved.dtype == tensor.dtype
        assert torch.equal(saved, tensor)
        # Slices of the staging buffer are cloned off it, so neither the pinned
        # buffer nor the source adapter storage is kept alive by the snapshot.
        assert saved.data_ptr() != tensor.data_ptr()
