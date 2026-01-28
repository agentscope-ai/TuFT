from typing import Dict, Tuple

import torch

from . import _check_loss_fn_inputs


def cross_entropy_loss(
    loss_fn_inputs: Dict[str, torch.Tensor], loss_fn_config: Dict[str, float]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Computes the Cross Entropy loss.

    Args:
        loss_fn_inputs: A dictionary of tensors required for the loss function.
            Expected keys: "target_logprobs", "weights".
        loss_fn_config: A dictionary of configuration parameters for the loss function.
            (No expected keys for this loss function.)
    Returns:
        A tuple containing the computed loss and a dictionary of metrics.
    """
    _check_loss_fn_inputs(loss_fn_inputs, ("target_logprobs", "weights"), check_shapes=True)
    target_logprobs = loss_fn_inputs["target_logprobs"]
    weights = loss_fn_inputs["weights"]

    loss = -(target_logprobs * weights).sum()
    return loss, {"loss:sum": loss.item()}
