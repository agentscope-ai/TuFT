import pytest

from tuft.exceptions import (
    LossFunctionInputShapeMismatchException,
    LossFunctionMissingInputException,
    LossFunctionNotFoundException,
    LossFunctionUnknownMetricReductionException,
)


@pytest.mark.gpu
def test_get_loss_fn():
    from tuft.loss_fn import get_loss_fn

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
    with pytest.raises(LossFunctionNotFoundException) as exc_info:
        get_loss_fn(invalid_name)
    assert str(exc_info.value) == f"Loss function {invalid_name} not found."


@pytest.mark.gpu
def test_cross_entropy_loss():
    import torch

    from tuft.loss_fn import get_loss_fn

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

    # shape mismatch
    loss_fn_inputs_shape = {
        "target_logprobs": torch.tensor([-0.2, -0.5]),
        "weights": torch.tensor([1.0, 1.0, 1.0]),
    }
    with pytest.raises(LossFunctionInputShapeMismatchException):
        loss_fn(loss_fn_inputs_shape, loss_fn_config)

    # missing input
    loss_fn_inputs_missing = {
        "weights": torch.tensor([1.0, 1.0, 1.0]),
    }
    with pytest.raises(LossFunctionMissingInputException):
        loss_fn(loss_fn_inputs_missing, loss_fn_config)


@pytest.mark.gpu
def test_importance_sampling_loss():
    import torch

    from tuft.loss_fn import get_loss_fn

    loss_fn = get_loss_fn("importance_sampling")
    loss_fn_inputs = {
        "target_logprobs": torch.tensor([-0.1, -0.4, -0.5]),
        "logprobs": torch.tensor([-0.2, -0.5, -0.25]),
        "advantages": torch.tensor([1.0, -1.0, 0.5]),
    }
    loss_fn_config = {}
    loss, metrics = loss_fn(loss_fn_inputs, loss_fn_config)
    expected_loss = -0.3894003927707672
    assert torch.isclose(loss, torch.tensor(expected_loss), rtol=0.001), (
        "Importance sampling loss mismatch."
    )
    assert torch.isclose(torch.tensor([metrics["loss:sum"]]), torch.tensor(expected_loss)), (
        "Importance sampling metric mismatch."
    )

    # shape mismatch
    loss_fn_inputs_shape = {
        "target_logprobs": torch.tensor([-0.1, -0.4]),
        "logprobs": torch.tensor([-0.2, -0.5, -0.25]),
        "advantages": torch.tensor([1.0, -1.0, 0.5]),
    }
    with pytest.raises(LossFunctionInputShapeMismatchException):
        loss_fn(loss_fn_inputs_shape, loss_fn_config)

    # missing input
    loss_fn_inputs_missing = {
        "target_logprobs": torch.tensor([-0.1, -0.4, -0.5]),
        "logprobs": torch.tensor([-0.2, -0.5, -0.25]),
    }
    with pytest.raises(LossFunctionMissingInputException):
        loss_fn(loss_fn_inputs_missing, loss_fn_config)


@pytest.mark.gpu
def test_ppo_loss():
    import torch

    from tuft.loss_fn import get_loss_fn

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

    # shape mismatch
    loss_fn_inputs_shape = {
        "target_logprobs": torch.tensor([-0.2, -0.5]),
        "logprobs": torch.tensor([-0.1, -0.4, -0.2]),
        "advantages": torch.tensor([1.0, -1.0, 0.5]),
    }
    with pytest.raises(LossFunctionInputShapeMismatchException):
        loss_fn(loss_fn_inputs_shape, loss_fn_config)

    # missing input
    loss_fn_inputs_missing = {
        "target_logprobs": torch.tensor([-0.2, -0.5, -0.4]),
        "logprobs": torch.tensor([-0.1, -0.4, -0.2]),
    }
    with pytest.raises(LossFunctionMissingInputException):
        loss_fn(loss_fn_inputs_missing, loss_fn_config)


@pytest.mark.gpu
def test_cispo_loss():
    import torch

    from tuft.loss_fn import get_loss_fn

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

    # shape mismatch
    loss_fn_inputs_shape = {
        "target_logprobs": torch.tensor([-0.3, -0.6]),
        "logprobs": torch.tensor([-0.2, -0.5, -0.1]),
        "advantages": torch.tensor([1.0, -1.0, 0.5]),
    }
    with pytest.raises(LossFunctionInputShapeMismatchException):
        loss_fn(loss_fn_inputs_shape, loss_fn_config)

    # missing input
    loss_fn_inputs_missing = {
        "target_logprobs": torch.tensor([-0.3, -0.6, -0.2]),
        "logprobs": torch.tensor([-0.2, -0.5, -0.1]),
    }
    with pytest.raises(LossFunctionMissingInputException):
        loss_fn(loss_fn_inputs_missing, loss_fn_config)


@pytest.mark.gpu
def test_dro_loss():
    import torch

    from tuft.loss_fn import get_loss_fn

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

    # shape mismatch
    loss_fn_inputs_shape = {
        "target_logprobs": torch.tensor([-0.4, -0.3]),
        "logprobs": torch.tensor([-0.2, -0.1, -0.4]),
        "advantages": torch.tensor([1.0, -1.0, 0.5]),
    }
    with pytest.raises(LossFunctionInputShapeMismatchException):
        loss_fn(loss_fn_inputs_shape, loss_fn_config)

    # missing input
    loss_fn_inputs_missing = {
        "target_logprobs": torch.tensor([-0.4, -0.3, -0.5]),
        "logprobs": torch.tensor([-0.2, -0.1, -0.4]),
    }
    with pytest.raises(LossFunctionMissingInputException):
        loss_fn(loss_fn_inputs_missing, loss_fn_config)


def test_loss_fn_metrics_reduction():
    import torch

    from tuft.loss_fn import metrics_reduction

    metric_list = [
        {"loss:mean": 0.5, "accuracy:sum": 2, "time:min": 0.8, "time:max": 0.9},
        {"loss:mean": 0.3, "accuracy:sum": 3, "time:min": 0.7, "time:max": 0.95},
        {"loss:mean": 0.4, "accuracy:sum": 5, "time:min": 0.75, "time:max": 0.85},
    ]
    weights = [2.0, 3.0, 5.0]
    reduced_metrics = metrics_reduction(metric_list, weights)

    expected_loss_mean = (0.5 * 2 + 0.3 * 3 + 0.4 * 5) / sum(weights)
    expected_accuracy_sum = 2 + 3 + 5
    expected_time_min = min(0.8, 0.7, 0.75)
    expected_time_max = max(0.9, 0.95, 0.85)

    assert torch.isclose(
        torch.tensor(reduced_metrics["loss:mean"]), torch.tensor(expected_loss_mean)
    ), "Reduced loss:mean mismatch."
    assert reduced_metrics["accuracy:sum"] == expected_accuracy_sum, "Reduced accuracy:sum mismatch."
    assert reduced_metrics["time:min"] == expected_time_min, "Reduced time:min mismatch."
    assert reduced_metrics["time:max"] == expected_time_max, "Reduced time:max mismatch."

    metric_list_empty = []
    weights_empty = []
    reduced_metrics_empty = metrics_reduction(metric_list_empty, weights_empty)
    assert reduced_metrics_empty == {}, "Reduced metrics for empty input should be empty."

    metric_unknown = [
        {"loss:unknown": 0.5},
    ]
    weights_unknown = [1.0]
    with pytest.raises(LossFunctionUnknownMetricReductionException):
        metrics_reduction(metric_unknown, weights_unknown)
