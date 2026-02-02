from typing import Callable, Dict, Tuple

from tinker.lib.chunked_fwdbwd_helpers import REDUCE_MAP
from torch import Tensor
from typing_extensions import TypeAlias

from ..exceptions import (
    LossFunctionInputShapeMismatchException,
    LossFunctionMissingInputException,
    LossFunctionNotFoundException,
    LossFunctionUnknownMetricReductionException,
)


LossFnType: TypeAlias = Callable[
    [Dict[str, Tensor], Dict[str, float]], Tuple[Tensor, Dict[str, float]]
]

LOSS_FN = {
    "cispo": "tuft.loss_fn.cispo.cispo_loss",
    "cross_entropy": "tuft.loss_fn.cross_entropy.cross_entropy_loss",
    "dro": "tuft.loss_fn.dro.dro_loss",
    "importance_sampling": "tuft.loss_fn.importance_sampling.importance_sampling_loss",
    "ppo": "tuft.loss_fn.ppo.ppo_loss",
}


def get_loss_fn(loss_fn_name: str) -> LossFnType:
    """Retrieve the loss function by name."""
    if loss_fn_name not in LOSS_FN:
        raise LossFunctionNotFoundException(loss_fn_name)

    module_path, func_name = LOSS_FN[loss_fn_name].rsplit(".", 1)
    module = __import__(module_path, fromlist=[func_name])
    return getattr(module, func_name)


def _check_loss_fn_inputs(
    loss_fn_inputs: Dict[str, Tensor], required_keys: Tuple[str, ...], check_shapes: bool = False
) -> None:
    """Check if all required keys are present in loss_fn_inputs and optionally
    check if their shapes match."""
    for key in required_keys:
        if key not in loss_fn_inputs:
            raise LossFunctionMissingInputException(key)

    if check_shapes:
        shapes = [loss_fn_inputs[key].shape for key in required_keys]
        if not all(shape == shapes[0] for shape in shapes):
            raise LossFunctionInputShapeMismatchException(shapes)


def metrics_reduction(
    metric_list: list[dict[str, float]],
    weights: list[float],
) -> dict[str, float]:
    """Aggregate metrics from multiple batches.

    Modified from tinker.lib.chunked_fwdbwd_helpers._metrics_reduction
    """
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    result = {}
    for key in keys:
        _, reduction = key.split(":")
        if reduction not in REDUCE_MAP:
            raise LossFunctionUnknownMetricReductionException(reduction)
        if not all(key in m for m in metric_list):
            continue
        reduce_fn = REDUCE_MAP[reduction]
        values = [m[key] for m in metric_list]

        if reduction in ["mean", "slack"]:
            result[key] = reduce_fn(values, weights)
        elif reduction in ["unique"]:
            result[key] = values[0]
            result.update({f"{key}_{i + 1}": v for i, v in enumerate(values[1:])})
        else:
            result[key] = reduce_fn(values)
    return result
