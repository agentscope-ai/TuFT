from typing import Dict, Tuple

import torch

from . import _check_loss_fn_inputs


def importance_sampling_loss(
    loss_fn_inputs: Dict[str, torch.Tensor], loss_fn_config: Dict[str, float]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Computes the importance sampling loss.

    Args:
        loss_fn_inputs: A dictionary of tensors required for the loss function.
            Expected keys: "target_logprobs", "logprobs", "advantages".
        loss_fn_config: This parameter is unused.

    Returns:
        A tuple containing the computed loss and a dictionary of metrics.
    """
    _check_loss_fn_inputs(
        loss_fn_inputs, ("target_logprobs", "logprobs", "advantages"), check_shapes=True
    )
    target_logprobs = loss_fn_inputs["target_logprobs"]
    sampling_logprobs = loss_fn_inputs["logprobs"]
    advantages = loss_fn_inputs["advantages"]

    # Compute probability ratio
    prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
    # Compute importance-weighted loss
    loss = -(prob_ratio * advantages).sum()

    return loss, {"loss:sum": loss.item()}
