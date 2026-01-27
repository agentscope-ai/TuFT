from typing import Callable, Dict, Tuple

from torch import Tensor
from typing_extensions import TypeAlias


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
        raise ValueError(f"Loss function {loss_fn_name} not found.")

    module_path, func_name = LOSS_FN[loss_fn_name].rsplit(".", 1)
    module = __import__(module_path, fromlist=[func_name])
    return getattr(module, func_name)
