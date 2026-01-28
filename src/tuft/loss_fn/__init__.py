from typing import Callable, Dict, Tuple

from torch import Tensor
from typing_extensions import TypeAlias

from ..exceptions import (
    LossFunctionInputShapeMismatchException,
    LossFunctionMissingInputException,
    LossFunctionNotFoundException,
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
