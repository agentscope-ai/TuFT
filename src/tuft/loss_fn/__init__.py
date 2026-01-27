from typing import Callable, Dict, Tuple

from torch import Tensor
from typing_extensions import TypeAlias

LossFnType: TypeAlias = Callable[
    [Dict[str, Tensor], Dict[str, float]], Tuple[Tensor, Dict[str, float]]
]

LOSS_FN = {
    "cross_entropy": "tuft.loss_fn.cross_entropy.cross_entropy_loss",
    "ppo": "tuft.loss_fn.ppo.ppo_loss",
}


def get_loss_fn(loss_fn_name: str) -> LossFnType:
    """Retrieve the loss function by name."""
    if loss_fn_name not in LOSS_FN:
        raise ValueError(f"Loss function {loss_fn_name} not found.")

    module_path, func_name = LOSS_FN[loss_fn_name].rsplit(".", 1)
    module = __import__(module_path, fromlist=[func_name])
    return getattr(module, func_name)
