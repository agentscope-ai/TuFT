from typing import Dict, Tuple

import torch


def dro_loss(
    loss_fn_inputs: Dict[str, torch.Tensor], loss_fn_config: Dict[str, float]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Computes the Distributionally Robust Optimization (DRO) loss.

    Args:
        loss_fn_inputs: A dictionary of tensors required for the loss function.
            Expected keys: "target_logprobs", "logprobs", "advantages".
        loss_fn_config: A dictionary of configuration parameters for the loss function.
            Expected keys: "beta".

    Returns:
        A tuple containing the computed loss and a dictionary of metrics.
    """
    target_logprobs = loss_fn_inputs["target_logprobs"]
    sampling_logprobs = loss_fn_inputs["logprobs"]
    advantages = loss_fn_inputs["advantages"]
    beta = loss_fn_config.get("beta", 1.0)

    # Compute quadratic penalty term
    quadratic_term = (target_logprobs - sampling_logprobs) ** 2
    # Compute DRO objective
    dro_objective = target_logprobs * advantages - 0.5 * beta * quadratic_term
    # DRO loss is negative of objective
    loss = -dro_objective.sum()

    return loss, {"loss:sum": loss.item()}
