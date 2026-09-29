"""Response-normalized, dual-clipped PPO with an optional K2 KL penalty."""

import math

import torch

from . import _check_loss_fn_inputs


def trinity_ppo_loss(
    loss_fn_inputs: dict[str, torch.Tensor], loss_fn_config: dict[str, float]
) -> tuple[torch.Tensor, dict[str, float]]:
    """Match Trinity's sequence-mean PPO objective across accumulated requests.

    Inputs are [datum, token] tensors: target_logprobs (model output), logprobs
    (old policy), advantages, and either a binary response mask or weights whose
    positive entries identify response tokens. Prompt, padding and discarded
    responses must be zero-masked. ref_logprobs is required when kl_coef > 0.

    Each datum contributes its masked token mean, divided by num_total_datums.
    The caller must supply the SAME positive full optimizer-batch datum count
    to every request, even when the SDK or backend further splits that batch.
    All-masked datums contribute zero but remain in that global denominator.
    Diagnostics are sums so they compose across requests, micro-batches and
    ranks; obtain token means by dividing by trinity/response_tokens:sum.
    """
    _check_loss_fn_inputs(
        loss_fn_inputs, ("target_logprobs", "logprobs", "advantages"), check_shapes=True
    )
    target = loss_fn_inputs["target_logprobs"]
    if target.ndim != 2:
        raise ValueError("trinity_ppo expects [datum, token] tensors")

    config = loss_fn_config
    clip_range = config.get("clip_range", 0.2)
    clip_ratio_c = config.get("clip_ratio_c", 3.0)
    kl_coef = config.get("kl_coef", 0.001)
    total = config.get("num_total_datums", 0.0)
    if not math.isfinite(total) or total < max(target.shape[0], 1) or int(total) != total:
        raise ValueError("num_total_datums must be the positive full-batch datum count")
    if not math.isfinite(clip_range) or not 0 <= clip_range < 1:
        raise ValueError("clip_range must be finite and in [0, 1)")
    if not math.isfinite(clip_ratio_c) or clip_ratio_c <= 1:
        raise ValueError("clip_ratio_c must be finite and greater than 1")
    if not math.isfinite(kl_coef) or kl_coef < 0:
        raise ValueError("kl_coef must be finite and non-negative")

    mask_key = "mask" if "mask" in loss_fn_inputs else "weights"
    _check_loss_fn_inputs(loss_fn_inputs, ("target_logprobs", mask_key), check_shapes=True)
    mask = loss_fn_inputs[mask_key]
    if mask_key == "mask" and not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask must contain only zero or one")
    valid = mask > 0
    dtype = torch.float64 if target.dtype == torch.float64 else torch.float32
    # Sanitize masked positions before exp/square: multiplying inf or nan by
    # zero afterwards would still poison otherwise valid batches and gradients.
    current = target.to(dtype).masked_fill(~valid, 0)
    old = loss_fn_inputs["logprobs"].detach().to(dtype).masked_fill(~valid, 0)
    advantages = loss_fn_inputs["advantages"].detach().to(dtype).masked_fill(~valid, 0)
    ratio = (current - old).clamp(-20, 20).exp()
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(1 - clip_range, 1 + clip_range)
    objective = torch.maximum(unclipped, clipped)
    objective = torch.where(
        advantages < 0, torch.minimum(objective, -advantages * clip_ratio_c), objective
    )

    kl = None
    if kl_coef > 0 or "ref_logprobs" in loss_fn_inputs:
        _check_loss_fn_inputs(
            loss_fn_inputs, ("target_logprobs", "ref_logprobs"), check_shapes=True
        )
        reference = loss_fn_inputs["ref_logprobs"].detach().to(dtype).masked_fill(~valid, 0)
        kl = 0.5 * (current - reference).square()
        objective = objective + kl_coef * kl

    per_datum = objective.masked_fill(~valid, 0).sum(-1) / valid.sum(-1).clamp_min(1)
    loss = per_datum.sum() / total
    with torch.no_grad():
        valid_ratio = ratio[valid]
        metrics = {
            "loss:sum": loss.item(),
            "trinity/response_tokens:sum": float(valid.sum().item()),
            "trinity/ratio_sum:sum": valid_ratio.sum().item(),
            "trinity/ratio_squared_sum:sum": valid_ratio.square().sum().item(),
            "trinity/clipped_tokens:sum": float(
                ((valid_ratio < 1 - clip_range) | (valid_ratio > 1 + clip_range)).sum().item()
            ),
        }
        if kl is not None:
            metrics["trinity/kl_sum:sum"] = kl[valid].sum().item()
    return loss, metrics
