import asyncio
import logging
from types import SimpleNamespace

import pytest
from tinker import types

from tuft.checkpoints import CheckpointRecord


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


def _checkpoint_record(tmp_path, training_run_id: str) -> CheckpointRecord:
    return CheckpointRecord.from_training_run(
        training_run_id=training_run_id,
        checkpoint_name="ckpt",
        owner_name="tester",
        checkpoint_type="training",
        checkpoint_root_dir=tmp_path,
    )


@pytest.mark.parametrize("legacy_layout", [False, True])
def test_optimizer_state_resolves_across_runs(tmp_path, legacy_layout) -> None:
    """The optimizer file must be findable by a run other than the one that saved it.

    Naming it after the saving run made optimizer=True a silent no-op whenever a
    checkpoint was restored into a different training run.
    """
    from tuft.backends.hf_training_model import (
        OPTIMIZER_STATE_FILENAME,
        _resolve_optimizer_state_path,
    )

    record = _checkpoint_record(tmp_path, "source-run")
    record.optimizer_path.mkdir(parents=True, exist_ok=True)
    file_name = "source-run.pt" if legacy_layout else OPTIMIZER_STATE_FILENAME
    (record.optimizer_path / file_name).write_bytes(b"adam-state")

    # Resolution depends only on the checkpoint, never on the run loading it.
    resolved = _resolve_optimizer_state_path(record)

    assert resolved is not None
    assert resolved.name == file_name


def test_optimizer_state_absent_resolves_to_none(tmp_path) -> None:
    from tuft.backends.hf_training_model import _resolve_optimizer_state_path

    record = _checkpoint_record(tmp_path, "source-run")
    record.optimizer_path.mkdir(parents=True, exist_ok=True)

    assert _resolve_optimizer_state_path(record) is None


@pytest.mark.asyncio
async def test_load_state_restores_optimizer_into_a_different_run(tmp_path) -> None:
    import torch

    from tuft.backends.hf_training_model import OPTIMIZER_STATE_FILENAME, HFTrainingModel

    class FakePeftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))
            self.loaded_adapters: list[str] = []

        def load_adapter(self, model_id, adapter_name):
            self.loaded_adapters.append(adapter_name)

        def set_adapter(self, adapter_name):
            pass

    model = HFTrainingModel.__new__(HFTrainingModel)
    model.model = FakePeftModel()  # type: ignore[assignment]
    model.adapter_optimizer = {}
    model._lock = asyncio.Lock()
    model.logger = logging.getLogger(__name__)

    # Source run saves optimizer state with a non-zero step count.
    record = _checkpoint_record(tmp_path, "source-run")
    record.optimizer_path.mkdir(parents=True, exist_ok=True)
    saved = torch.optim.AdamW(list(model.model.parameters()))
    model.model.weight.grad = torch.ones(4)
    saved.step()
    torch.save(saved.state_dict(), record.optimizer_path / OPTIMIZER_STATE_FILENAME)

    await model.load_state(lora_id="destination-run", checkpoint_record=record, optimizer=True)

    restored = model.adapter_optimizer["destination-run"]
    assert model.model.loaded_adapters == ["destination-run"]
    # A fresh AdamW has an empty state dict; the restored one carries the step.
    assert restored.state_dict()["state"], "optimizer state was silently dropped"
    assert restored.state_dict()["state"][0]["step"] == saved.state_dict()["state"][0]["step"]
