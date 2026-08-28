from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tinker import types

from tuft.backends.base_backend import BaseSamplingBackend, BaseTrainingBackend
from tuft.backends.torchtp_backend import TorchTPSamplingBackend, TorchTPTrainingBackend
from tuft.config import ModelConfig


class FakeKVModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.config = SimpleNamespace(vocab_size=8, use_cache=True)
        self.calls: list[dict[str, object]] = []

    def forward(self, input_ids, past_key_values=None, use_cache=False, return_dict=True, **_kwargs):
        self.calls.append(
            {
                "shape": tuple(input_ids.shape),
                "has_past": past_key_values is not None,
                "use_cache": use_cache,
            }
        )
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, self.config.vocab_size), -100.0)
        logits[..., 3] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=("cached", len(self.calls)))


def test_torchtp_backend_factory_registration(monkeypatch) -> None:
    monkeypatch.delenv("TUFT_CPU_TEST", raising=False)
    config = ModelConfig(
        model_name="test",
        model_path=Path("/tmp/model"),
        max_model_len=128,
        training_backend="torchtp",
    )

    training = BaseTrainingBackend.create_backend(config)
    sampling = BaseSamplingBackend.create_backend(config)

    assert isinstance(training, TorchTPTrainingBackend)
    assert isinstance(sampling, TorchTPSamplingBackend)


def test_explicit_torchtp_sampling_backend_registration(monkeypatch) -> None:
    monkeypatch.delenv("TUFT_CPU_TEST", raising=False)
    config = ModelConfig(
        model_name="test",
        model_path=Path("/tmp/model"),
        max_model_len=128,
        sampling_backend="torchtp",
    )

    sampling = BaseSamplingBackend.create_backend(config)

    assert isinstance(sampling, TorchTPSamplingBackend)


async def _sample_with_fake_model() -> tuple[types.SampleResponse, FakeKVModel]:
    model = FakeKVModel()
    backend = TorchTPSamplingBackend(
        ModelConfig(model_name="test", model_path=Path("/tmp/model"), max_model_len=128),
        model=model,
    )
    response = await backend.sample(
        prompt=types.ModelInput.from_ints([1, 2]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=3, temperature=0.0),
    )
    return response, model


@pytest.mark.asyncio
async def test_torchtp_sampling_uses_kv_cache() -> None:
    response, model = await _sample_with_fake_model()

    assert len(response.sequences) == 1
    assert list(response.sequences[0].tokens) == [3, 3, 3]
    assert len(response.sequences[0].logprobs) == 3
    assert model.calls[0] == {"shape": (1, 2), "has_past": False, "use_cache": True}
    assert model.calls[1] == {"shape": (1, 1), "has_past": True, "use_cache": True}
    assert model.calls[2] == {"shape": (1, 1), "has_past": True, "use_cache": True}
