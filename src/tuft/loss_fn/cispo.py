from typing import Dict, Tuple

import torch


def cispo_loss(
    loss_fn_inputs: Dict[str, torch.Tensor], loss_fn_config: Dict[str, float]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Computes the Clipped Importance Sampling Policy Optimization (CISPO) loss.

    Args:
        loss_fn_inputs: A dictionary of tensors required for the loss function.
            Expected keys: "target_logprobs", "logprobs", "advantages".
        loss_fn_config: A dictionary of configuration parameters for the loss function.
            Expected keys: "clip_low_threshold", "clip_high_threshold".

    Returns:
        A tuple containing the computed loss and a dictionary of metrics.
    """
    target_logprobs = loss_fn_inputs["target_logprobs"]
    sampling_logprobs = loss_fn_inputs["logprobs"]
    advantages = loss_fn_inputs["advantages"]
    clip_low_threshold = loss_fn_config.get("clip_low_threshold", 0.9)
    clip_high_threshold = loss_fn_config.get("clip_high_threshold", 1.1)

    # Compute probability ratio
    prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
    # Apply clipping
    clipped_ratio = torch.clamp(prob_ratio, clip_low_threshold, clip_high_threshold)
    # Compute CISPO objective (detach the clipped ratio)
    cispo_objective = clipped_ratio.detach() * target_logprobs * advantages
    # CISPO loss is negative of objective
    loss = -cispo_objective.sum()

    return loss, {"loss:sum": loss.item()}
