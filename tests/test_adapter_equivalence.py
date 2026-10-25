import json

import pytest
import torch
from safetensors.torch import save_file

from tuft.backends.adapter_equivalence import is_zero_lora_snapshot


@pytest.fixture
def snapshot(tmp_path):
    config = {"peft_type": "LORA", "r": 2, "lora_alpha": 4, "bias": "none"}
    tensors = {"layer.lora_A.weight": torch.ones(2, 4), "layer.lora_B.weight": torch.zeros(4, 2)}

    def write(config_updates=None, tensor_updates=None):
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({**config, **(config_updates or {})})
        )
        save_file({**tensors, **(tensor_updates or {})}, tmp_path / "adapter_model.safetensors")
        return tmp_path

    return write


def test_only_verified_zero_snapshot_can_use_base(snapshot):
    assert is_zero_lora_snapshot(snapshot())
    assert not is_zero_lora_snapshot(
        snapshot(tensor_updates={"layer.lora_B.weight": torch.ones(4, 2)})
    )


@pytest.mark.parametrize(
    "config",
    [
        {"use_dora": True},
        {"bias": "all"},
        {"modules_to_save": ["lm_head"]},
        {"target_parameters": ["experts"]},
        {"lora_alpha": float("nan")},
        {"alpha_pattern": {"layer": float("inf")}},
        {"rank_pattern": {"layer": 0}},
        {"layer_replication": [[0, 2]]},
    ],
)
def test_extra_state_or_unknown_geometry_is_not_merged(snapshot, config):
    assert not is_zero_lora_snapshot(snapshot(config_updates=config))


@pytest.mark.parametrize(
    "tensors",
    [
        {"layer.lora_A.weight": torch.full((2, 4), float("inf"))},
        {"layer.lora_B.weight": torch.zeros(4, 3)},
        {"layer.bias": torch.zeros(4)},
        {"other.lora_B.weight": torch.zeros(4, 2)},
    ],
)
def test_nonfinite_extra_or_unpaired_weights_are_not_merged(snapshot, tensors):
    assert not is_zero_lora_snapshot(snapshot(tensor_updates=tensors))


def test_missing_and_malformed_files_are_not_merged(tmp_path):
    assert not is_zero_lora_snapshot(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{broken}")
    assert not is_zero_lora_snapshot(tmp_path)
