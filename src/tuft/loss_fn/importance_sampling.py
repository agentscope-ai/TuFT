from typing import Dict, Tuple

import torch


def importance_sampling_loss(
    loss_fn_inputs: Dict[str, torch.Tensor], loss_fn_config: Dict[str, float]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_logprobs = loss_fn_inputs["target_logprobs"]
    sampling_logprobs = loss_fn_inputs["logprobs"]
    advantages = loss_fn_inputs["advantages"]

    prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
    loss = -(prob_ratio * advantages).sum()

    return loss, {"loss:sum": loss.item()}
