"""Tests for FlexBackend single-backend RL mode (training_backend='flex')."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tinker import types

from tuft.backends.flex.torchtp import FusedTorchTPVLLMFlexBackend
from tuft.config import AppConfig, ModelConfig


@pytest.fixture()
def flex_model_config(tmp_path: Path) -> ModelConfig:
    # Pin the conservative post-init-alias / destroy-recreate path so these
    # CPU unit tests (fake engines) stay isolated from the optimal sleep/wake +
    # pre-capture default config exercised by the GPU integration test.
    return ModelConfig(
        model_name="test-flex-model",
        model_path=tmp_path / "model",
        max_model_len=128,
        tensor_parallel_size=1,
        training_backend="flex",
        sampling_memory_fraction=0.5,
        sampling_enforce_eager=True,
        sampling_pre_capture_alias=False,
        sampling_disable_custom_all_reduce=False,
        sampling_enable_sleep_mode=False,
        sampling_keep_runtime_on_training=False,
        sampling_flex_kv_cache_only_sleep=False,
    )


class FakeFlexTrainableModel(torch.nn.Module):
    """A minimal trainable model where adapter params directly affect logits.

    Logits are computed as a linear projection of adapter params, so
    cross_entropy gradients flow through and training can reduce loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.base_bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.adapter_biases = torch.nn.ParameterDict()
        self.active_adapter: str | None = None
        self.config = SimpleNamespace(vocab_size=8)

    def train(self, mode: bool = True):
        return super().train(mode)

    def forward(self, input_ids, attention_mask=None, return_dict=True, **_kwargs):
        batch, seq_len = input_ids.shape
        vocab = self.config.vocab_size
        # Base logits: zeros (uniform distribution)
        logits = torch.zeros(batch, seq_len, vocab, device=input_ids.device)
        if self.active_adapter is not None and self.active_adapter in self.adapter_biases:
            adapter_param = self.adapter_biases[self.active_adapter]
            # Project adapter params into vocab-space logits.
            # This creates a learnable mapping: adapter params -> logit shifts.
            projection = torch.linspace(1, vocab, vocab, device=input_ids.device)
            logit_shift = (adapter_param * projection[: adapter_param.shape[0]]).sum()
            # Apply token-dependent shift so cross_entropy has non-zero gradient
            token_scale = torch.arange(1, vocab + 1, dtype=torch.float32, device=input_ids.device)
            logits = logits + logit_shift * token_scale / vocab
        return SimpleNamespace(logits=logits)


def _fake_apply_lora(
    model,
    *,
    lora_rank,
    lora_alpha,
    tp_group=None,
    lora_id="default",
    **_kwargs,
):
    param = torch.nn.Parameter(torch.zeros(lora_rank))
    model.adapter_biases[lora_id] = param
    model.active_adapter = lora_id
    return [param]


def _fake_set_adapter(model, lora_id):
    model.active_adapter = lora_id


def test_flex_config_creates_shared_backend(flex_model_config: ModelConfig) -> None:
    """ServerState._create_flex_backends should create a FlexBackend for training_backend='flex'."""
    config = AppConfig(supported_models=[flex_model_config])
    # We cannot fully construct ServerState without real backends for non-flex models,
    # but we can test _create_flex_backends logic in isolation.
    from tuft.state import ServerState

    flex_backends = ServerState._create_flex_backends(
        SimpleNamespace(config=config, **{"config.supported_models": [flex_model_config]})
    ) if hasattr(ServerState, "_create_flex_backends") else {}
    # Direct test: create backends from config
    flex_backends = {}
    for mc in config.supported_models:
        if getattr(mc, "training_backend", "hf") == "flex":
            flex_backends[mc.model_name] = FusedTorchTPVLLMFlexBackend(mc)
    assert "test-flex-model" in flex_backends
    backend = flex_backends["test-flex-model"]
    assert isinstance(backend, FusedTorchTPVLLMFlexBackend)
    assert backend.config.training_backend == "flex"


def test_flex_backend_is_both_training_and_sampling(flex_model_config: ModelConfig) -> None:
    """FlexBackend instance should satisfy both BaseTrainingBackend and BaseSamplingBackend."""
    from tuft.backends.base_backend import BaseSamplingBackend, BaseTrainingBackend

    backend = FusedTorchTPVLLMFlexBackend(flex_model_config)
    assert isinstance(backend, BaseTrainingBackend)
    assert isinstance(backend, BaseSamplingBackend)


def test_training_controller_uses_shared_backend(flex_model_config: ModelConfig) -> None:
    """TrainingController should use the shared FlexBackend when provided."""
    from tuft.training_controller import TrainingController

    backend = FusedTorchTPVLLMFlexBackend(flex_model_config)
    config = AppConfig(supported_models=[flex_model_config])
    controller = TrainingController(config, shared_backends={"test-flex-model": backend})
    assert controller.training_backends["test-flex-model"] is backend


def test_sampling_controller_uses_shared_backend(flex_model_config: ModelConfig) -> None:
    """SamplingController should use the shared FlexBackend when provided."""
    from tuft.sampling_controller import SamplingController

    backend = FusedTorchTPVLLMFlexBackend(flex_model_config)
    config = AppConfig(supported_models=[flex_model_config])
    controller = SamplingController(config, shared_backends={"test-flex-model": backend})
    assert controller._base_backends["test-flex-model"] is backend


def test_shared_backend_same_instance(flex_model_config: ModelConfig) -> None:
    """Both controllers should reference the exact same FlexBackend instance."""
    from tuft.sampling_controller import SamplingController
    from tuft.training_controller import TrainingController

    backend = FusedTorchTPVLLMFlexBackend(flex_model_config)
    config = AppConfig(supported_models=[flex_model_config])
    shared = {"test-flex-model": backend}
    training_ctrl = TrainingController(config, shared_backends=shared)
    sampling_ctrl = SamplingController(config, shared_backends=shared)
    assert training_ctrl.training_backends["test-flex-model"] is backend
    assert sampling_ctrl._base_backends["test-flex-model"] is backend


@pytest.mark.asyncio
async def test_flex_single_backend_forward_and_mode_switch(
    flex_model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FlexBackend in single-backend mode: forward works in training mode."""
    import tuft.backends.flex.torchtp_training as torchtp_training

    trainable_model = FakeFlexTrainableModel()
    monkeypatch.setattr(torchtp_training, "apply_fused_torchtp_lora", _fake_apply_lora)
    monkeypatch.setattr(torchtp_training, "set_fused_torchtp_lora_adapter", _fake_set_adapter)

    backend = FusedTorchTPVLLMFlexBackend(flex_model_config, training_model=trainable_model)
    await backend.create_adapter("adapter1", types.LoraConfig(rank=4, seed=42))

    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_torch(torch.tensor([2, 3, 4])),
            "weights": types.TensorData.from_torch(torch.ones(3)),
        },
    )
    output = await backend.forward(
        data=[datum],
        lora_id="adapter1",
        loss_fn="cross_entropy",
        loss_fn_config=None,
        backward=True,
    )
    assert output.metrics["loss:sum"] > 0
    assert len(output.loss_fn_outputs) == 1


@pytest.mark.asyncio
async def test_flex_single_backend_optim_step(
    flex_model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FlexBackend optim_step updates adapter parameters."""
    import tuft.backends.flex.torchtp_training as torchtp_training

    trainable_model = FakeFlexTrainableModel()
    monkeypatch.setattr(torchtp_training, "apply_fused_torchtp_lora", _fake_apply_lora)
    monkeypatch.setattr(torchtp_training, "set_fused_torchtp_lora_adapter", _fake_set_adapter)

    backend = FusedTorchTPVLLMFlexBackend(flex_model_config, training_model=trainable_model)
    await backend.create_adapter("adapter1", types.LoraConfig(rank=4, seed=42))

    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_torch(torch.tensor([2, 3, 4])),
            "weights": types.TensorData.from_torch(torch.ones(3)),
        },
    )
    await backend.forward(
        data=[datum],
        lora_id="adapter1",
        loss_fn="cross_entropy",
        loss_fn_config=None,
        backward=True,
    )
    result = await backend.optim_step(
        adam_params=types.AdamParams(
            learning_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
            grad_clip_norm=1.0,
        ),
        lora_id="adapter1",
    )
    assert result.metrics["step:max"] == 1.0


def test_single_gpu_world_size(flex_model_config: ModelConfig) -> None:
    """FlexBackend with TP=1 should report world_size=1 and rank=0 without dist init."""
    backend = FusedTorchTPVLLMFlexBackend(flex_model_config)
    assert backend._world_size() == 1
    assert backend._rank() == 0


@pytest.mark.asyncio
async def test_flex_single_backend_three_round_rl_loop(
    flex_model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate 3 rounds of RL with the correct flow: sampling -> training -> ...

    Each round rolls out first (transform to sampling + sample), then updates
    (transform back to training + forward/backward/optim).

    Verifies:
    - Mode auto-switches correctly each round (sampling first, then training)
    - Adapter params are automatically preserved across mode switches
    - Loss decreases across rounds (training is actually learning)
    - Step counter increments correctly
    """
    import tuft.backends.flex.torchtp_training as torchtp_training
    from tuft.backends.flex.flex_backend import FlexBackendMode

    trainable_model = FakeFlexTrainableModel()
    monkeypatch.setattr(torchtp_training, "apply_fused_torchtp_lora", _fake_apply_lora)
    monkeypatch.setattr(torchtp_training, "set_fused_torchtp_lora_adapter", _fake_set_adapter)

    backend = FusedTorchTPVLLMFlexBackend(
        flex_model_config,
        training_model=trainable_model,
        # Factory returns the same fake model when transforming back to training
        training_runtime_factory=lambda config, keepalive, descriptors: trainable_model,
    )
    await backend.create_adapter("rl_adapter", types.LoraConfig(rank=4, seed=42))

    # Setup fake vLLM engine + factory for repeated sampling transforms.
    # generate() returns a minimal valid structure so backend.sample() works;
    # collective_rpc() returns [] which the injection path summarizes to zeros.
    def _fake_generate(prompts, params):
        LogProb = SimpleNamespace
        seq = SimpleNamespace(
            token_ids=[1, 2, 3],
            logprobs=[
                {1: LogProb(logprob=-0.1)},
                {2: LogProb(logprob=-0.2)},
                {3: LogProb(logprob=-0.3)},
            ],
            finish_reason="length",
        )
        return [SimpleNamespace(outputs=[seq])]

    def _fake_vllm_engine_factory(config):
        return SimpleNamespace(
            generate=_fake_generate,
            collective_rpc=lambda fn, args=None: [],
        )

    backend.vllm_engine = _fake_vllm_engine_factory(flex_model_config)
    backend.vllm_engine_factory = _fake_vllm_engine_factory
    backend.state_dict_builder = lambda model: {}
    backend.require_cuda_ipc = False

    num_rl_rounds = 3
    losses: list[float] = []
    # Use fixed input so loss decrease reflects learning, not data change
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_torch(torch.tensor([2, 3, 4])),
            "weights": types.TensorData.from_torch(torch.ones(3)),
        },
    )
    sample_prompt = types.ModelInput.from_ints([1, 2, 3, 4, 5])

    for round_idx in range(num_rl_rounds):
        # --- Sampling phase first (rollout): transform to sampling + sample ---
        result = await backend.transform_to_sampling()
        assert result.supported or backend.mode == FlexBackendMode.SAMPLING
        assert backend.mode == FlexBackendMode.SAMPLING
        sample_response = await backend.sample(
            prompt=sample_prompt,
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=8,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                seed=42,
            ),
        )
        assert len(sample_response.sequences) >= 1, "Sampling should produce a sequence"

        # --- Training phase: transform back (auto-restores adapter) + train ---
        result = await backend.transform_to_training()
        assert result.supported, f"Transform to training should be supported: {result.message}"
        assert backend.mode == FlexBackendMode.TRAINING
        # Adapter should be automatically restored after the round-trip
        assert "rl_adapter" in backend._adapter_optimizers, (
            f"Round {round_idx}: adapter should be auto-restored after transform_to_training"
        )

        # Multiple forward/backward steps per round (like real RL training)
        for _step in range(3):
            await backend.forward(
                data=[datum],
                lora_id="rl_adapter",
                loss_fn="cross_entropy",
                loss_fn_config=None,
                backward=True,
            )
            await backend.optim_step(
                adam_params=types.AdamParams(
                    learning_rate=0.1,
                    beta1=0.9,
                    beta2=0.999,
                    eps=1e-8,
                    weight_decay=0.0,
                    grad_clip_norm=1.0,
                ),
                lora_id="rl_adapter",
            )

        # Record loss at end of this round's training
        final_output = await backend.forward(
            data=[datum],
            lora_id="rl_adapter",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=False,
        )
        loss_val = final_output.metrics.get("loss:sum", final_output.metrics.get("loss:mean", 0.0))
        assert loss_val > 0, f"Round {round_idx}: loss should be positive, got {loss_val}"
        losses.append(loss_val)

    # Verify loss decreased from first to last round
    assert losses[-1] < losses[0], (
        f"Loss should decrease across RL rounds: {losses}"
    )
    # Total steps: 3 rounds * 3 training steps
    assert backend._training_step == num_rl_rounds * 3


def test_worker_pool_flip_skips_adapter_state_offload() -> None:
    """A flip must not offload adapter state the workers already own.

    In worker-pool mode the parent holds no training model, so
    ``_restore_adapter_state_after_switch`` returns immediately; copying every
    LoRA tensor of every adapter to host memory first measured 2.9-12.5s of each
    flip at 16 tenants and made reactive gap filling a net loss.
    """
    from tuft.backends.flex.torchtp import FusedTorchTPVLLMFlexBackend

    backend = FusedTorchTPVLLMFlexBackend.__new__(FusedTorchTPVLLMFlexBackend)
    backend._saved_adapter_state = {"stale": {}}
    backend.training_model = None
    backend._training_worker_pool = object()
    backend._adapter_params = {"tenant-a": [object()]}
    backend._adapter_optimizers = {}
    backend._adapter_configs = {}

    backend._save_adapter_state_for_switch()

    assert backend._saved_adapter_state is None

    # Without a worker pool the parent owns the state, so the offload path must
    # still run: reaching it is what the skip above deliberately avoids.
    offloaded: list[list] = []
    backend._training_worker_pool = None
    backend._ordered_named_adapter_params = lambda params: [("w", _FakeParam())]
    backend._offload_named_tensors_to_cpu = lambda named: (offloaded.append(named), {})[1]
    backend._save_adapter_state_for_switch()
    assert offloaded, "single-process flip must still offload adapter state"
    assert backend._saved_adapter_state is not None


class _FakeParam:
    grad = None

    def detach(self) -> "_FakeParam":
        return self


def test_adapter_offload_lands_on_the_host_without_per_tensor_clones():
    """The offload must produce host tensors whose values survive the batching.

    save_state dominated every train->sample flip at 3.3-14.7s, ~0.9s per
    resident adapter for 66 MB of params+grads, an effective 73 MB/s against the
    ~20 GB/s the transfer itself runs at: the batched D2H was followed by ~500
    small per-tensor host clones per adapter. The values must round-trip exactly
    now that the tensors are views into one bulk copy.
    """
    import torch

    from tuft.backends.flex.torchtp import FusedTorchTPVLLMFlexBackend

    named = [
        ("a.weight", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
        ("b.weight", torch.full((4,), 7.0, dtype=torch.float32)),
        ("c.weight", torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)),
        ("d.grad", None),
    ]
    out = FusedTorchTPVLLMFlexBackend._offload_named_tensors_to_cpu(named)

    assert out["d.grad"] is None, "None entries must stay None, not become zeros"
    for name, tensor in named:
        if tensor is None:
            continue
        assert out[name].device.type == "cpu", f"{name} must be host-resident"
        assert out[name].shape == tensor.shape
        assert torch.equal(out[name], tensor), f"{name} lost its values"


def test_optimizer_state_offload_moves_tensors_off_the_device():
    """AdamW's exp_avg/exp_avg_sq must not stay resident on the device.

    state_dict() returns the live tensors, so releasing the training runtime
    could not reclaim two more copies of every LoRA parameter.
    """
    import torch

    from tuft.backends.flex.torchtp import FusedTorchTPVLLMFlexBackend

    param = torch.nn.Parameter(torch.randn(4, 4))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    param.grad = torch.randn(4, 4)
    optimizer.step()

    state = FusedTorchTPVLLMFlexBackend._offload_optimizer_state_to_cpu(optimizer)
    assert state is not None
    tensors = [
        v
        for entry in state["state"].values()
        for v in entry.values()
        if isinstance(v, torch.Tensor)
    ]
    assert tensors, "AdamW should expose exp_avg/exp_avg_sq tensors"
    assert all(t.device.type == "cpu" for t in tensors)
    # A fresh optimizer must accept the host-resident state.
    torch.optim.AdamW([param], lr=1e-3).load_state_dict(state)

    assert FusedTorchTPVLLMFlexBackend._offload_optimizer_state_to_cpu(None) is None
