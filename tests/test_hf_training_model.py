import asyncio
import logging
from types import SimpleNamespace

import pytest
from tinker import types


@pytest.mark.asyncio
async def test_forward_backward_empties_cuda_cache_once_after_all_micro_batches(
    monkeypatch,
):
    import torch

    from tuft.backends.hf_training_model import HFTrainingModel

    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.lm_head = torch.nn.Linear(8, 16, bias=False)

        def forward(self, input_ids, **_kwargs):
            return SimpleNamespace(logits=self.lm_head(self.embed(input_ids)))

    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=tokens),
            loss_fn_inputs={
                "target_tokens": types.TensorData(
                    data=[token + 1 for token in tokens], dtype="int64"
                ),
                "weights": types.TensorData(data=[1.0] * len(tokens), dtype="float32"),
            },
        )
        for tokens in ([1, 2, 3], [4, 5])
    ]

    model = HFTrainingModel.__new__(HFTrainingModel)
    model.config = SimpleNamespace(micro_batch_size=1)  # type: ignore[assignment]
    model.model = TinyCausalLM()  # type: ignore[assignment]
    model._lock = asyncio.Lock()
    model.logger = logging.getLogger(__name__)
    monkeypatch.setattr(model, "_activate_adapter", lambda _lora_id: None)

    empty_cache_calls = 0

    def record_empty_cache():
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", record_empty_cache)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)

    output = await model.forward(
        data=data,
        lora_id="test",
        loss_fn="cross_entropy",
        loss_fn_config=None,
        backward=True,
    )

    assert len(output.loss_fn_outputs) == 2
    assert empty_cache_calls == 1
