import pytest
import torch

from tuft.loss_fn import get_loss_fn


def test_get_loss_fn():
    loss_fn_names = [
        "cross_entropy",
        "importance_sampling",
        "ppo",
        "cispo",
        "dro",
    ]
    for name in loss_fn_names:
        loss_fn = get_loss_fn(name)
        assert callable(loss_fn), f"Loss function for {name} should be callable."

    invalid_name = "invalid_loss_fn"
    with pytest.raises(ValueError) as exc_info:
        get_loss_fn(invalid_name)
    assert str(exc_info.value) == f"Loss function {invalid_name} not found."


def test_cross_entropy_loss():
    loss_fn = get_loss_fn("cross_entropy")
    loss_fn_inputs = {
        "target_logprobs": torch.tensor([-0.2, -0.5, -0.3]),
        "weights": torch.tensor([1.0, 1.0, 1.0]),
    }
    loss_fn_config = {}
    loss, metrics = loss_fn(loss_fn_inputs, loss_fn_config)
    expected_loss = 1.0  # -( -0.2 -0.5 -0.3 ) = 1.0
    assert torch.isclose(loss, torch.tensor(expected_loss)), "Cross-entropy loss mismatch."
    assert metrics["loss:sum"] == expected_loss, "Cross-entropy metric mismatch."


def test_importance_sampling_loss():
    loss_fn = get_loss_fn("importance_sampling")
    loss_fn_inputs = {
        "target_logprobs": torch.tensor([-0.1, -0.4, -0.5]),
        "logprobs": torch.tensor([-0.2, -0.5, -0.25]),
        "advantages": torch.tensor([1.0, -1.0, 0.5]),
    }
    loss_fn_config = {}
    loss, metrics = loss_fn(loss_fn_inputs, loss_fn_config)
    expected_loss = -0.3894003927707672
    assert torch.isclose(
        loss, torch.tensor(expected_loss), rtol=0.001
    ), "Importance sampling loss mismatch."
    assert torch.isclose(
        torch.tensor([metrics["loss:sum"]]), torch.tensor(expected_loss)
    ), "Importance sampling metric mismatch."


def test_ppo_loss():
    loss_fn = get_loss_fn("ppo")
    loss_fn_inputs = {
        "target_logprobs": torch.tensor([-0.2, -0.5, -0.4]),
        "logprobs": torch.tensor([-0.1, -0.4, -0.2]),
        "advantages": torch.tensor([1.0, -1.0, 0.5]),
    }
    loss_fn_config = {
        "clip_low_threshold": 0.8,
        "clip_high_threshold": 1.2,
    }
    loss, metrics = loss_fn(loss_fn_inputs, loss_fn_config)
    expected_loss = -0.4093653
    assert torch.isclose(loss, torch.tensor(expected_loss), rtol=0.001), "PPO loss mismatch."
    assert torch.isclose(
        torch.tensor([metrics["loss:sum"]]), torch.tensor(expected_loss), rtol=0.001
    ), "PPO metric mismatch."


def test_cispo_loss():
    loss_fn = get_loss_fn("cispo")
    loss_fn_inputs = {
        "target_logprobs": torch.tensor([-0.3, -0.6, -0.2]),
        "logprobs": torch.tensor([-0.2, -0.5, -0.1]),
        "advantages": torch.tensor([1.0, -1.0, 0.5]),
    }
    loss_fn_config = {
        "clip_low_threshold": 0.85,
        "clip_high_threshold": 1.15,
    }
    loss, metrics = loss_fn(loss_fn_inputs, loss_fn_config)
    expected_loss = -0.1810
    assert torch.isclose(loss, torch.tensor(expected_loss), rtol=0.001), "CISPO loss mismatch."
    assert torch.isclose(
        torch.tensor([metrics["loss:sum"]]), torch.tensor(expected_loss), rtol=0.001
    ), "CISPO metric mismatch."


def test_dro_loss():
    loss_fn = get_loss_fn("dro")
    loss_fn_inputs = {
        "target_logprobs": torch.tensor([-0.4, -0.3, -0.5]),
        "logprobs": torch.tensor([-0.2, -0.1, -0.4]),
        "advantages": torch.tensor([1.0, -1.0, 0.5]),
    }
    loss_fn_config = {
        "beta": 0.05,
    }
    loss, metrics = loss_fn(loss_fn_inputs, loss_fn_config)
    expected_loss = 0.3522
    assert torch.isclose(loss, torch.tensor(expected_loss), rtol=0.001), "DRO loss mismatch."
    assert torch.isclose(
        torch.tensor([metrics["loss:sum"]]), torch.tensor(expected_loss), rtol=0.001
    ), "DRO metric mismatch."
