from typing import Dict, Tuple

import torch


def cross_entropy_loss(
    loss_fn_inputs: Dict[str, torch.Tensor], loss_fn_config: Dict[str, float]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_logprobs = loss_fn_inputs["target_logprobs"]
    weights = loss_fn_inputs["weights"]

    loss = -(target_logprobs * weights).sum()
    return loss, {"loss:sum": loss.item()}
