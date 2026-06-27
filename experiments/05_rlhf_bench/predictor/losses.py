"""Loss + evaluation metrics for the framework-mismatch predictor.

Loss = L_token + λ_seq · L_seq + λ_bias · L_bias
  L_token : token-level L1 between predicted Δ̂_t and true Δ_t
  L_seq   : sequence-level L1 between Σ_t Δ̂_t and Σ_t Δ_t,
            normalised by sequence length so long sequences don't dominate
  L_bias  : |mean batch residual|, explicit penalty on systematic bias
            — directly attacks the most damaging mismatch component for RL

Eval metrics report both BASELINE (no correction, Δ̂ = 0) and CORRECTED
side by side, so the predictor's value is directly visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch


# ──────────────────────────────────────────────────────────────────────
def predictor_loss(
    pred: torch.Tensor,  # [B, T]
    target: torch.Tensor,  # [B, T]
    mask: torch.Tensor,  # [B, T]  bool
    lambda_seq: float = 1.0,
    lambda_bias: float = 5.0,
):
    m = mask.float()
    n = m.sum().clamp_min(1.0)

    diff = (pred - target) * m
    abs_diff = diff.abs()

    L_token = abs_diff.sum() / n

    seq_sum_err = diff.sum(dim=1).abs()  # [B]
    seq_len = m.sum(dim=1).clamp_min(1.0)  # [B]
    L_seq = (seq_sum_err / seq_len).mean()

    L_bias = diff.sum().abs() / n

    total = L_token + lambda_seq * L_seq + lambda_bias * L_bias
    return total, {
        "L_token": L_token.detach().item(),
        "L_seq": L_seq.detach().item(),
        "L_bias": L_bias.detach().item(),
        "L_total": total.detach().item(),
    }


def scalar_predictor_loss(
    pred: torch.Tensor,  # [B, T]  but scalar per sequence (constant across T)
    target: torch.Tensor,  # [B, T]
    mask: torch.Tensor,  # [B, T]
    lambda_bias: float = 5.0,
):
    """Loss for a per-sequence scalar predictor.

    pred is assumed constant over T for each sequence.  We collapse it to
    [B] and regress against the per-sequence mean of target.
    """
    m = mask.float()
    seq_len = m.sum(dim=1).clamp_min(1.0)  # [B]
    target_mean = (target * m).sum(dim=1) / seq_len  # [B]

    # pred is constant per sequence, so any position gives the scalar
    pred_scalar = pred[:, 0]  # [B]

    diff = pred_scalar - target_mean  # [B]
    L_scalar = diff.abs().mean()
    L_bias = diff.sum().abs() / max(diff.numel(), 1)

    total = L_scalar + lambda_bias * L_bias
    return total, {
        "L_token": L_scalar.detach().item(),  # for logging compatibility
        "L_seq": L_scalar.detach().item(),
        "L_bias": L_bias.detach().item(),
        "L_total": total.detach().item(),
    }


# ──────────────────────────────────────────────────────────────────────
@dataclass
class EvalSummary:
    n_tokens: int
    n_seqs: int

    # token-level: BEFORE correction (pred = 0)
    baseline_token_mae: float
    baseline_token_bias: float
    # token-level: AFTER correction
    corrected_token_mae: float
    corrected_token_bias: float

    # sequence-level cumulative diff = log IS weight
    baseline_mean_log_is: float  # mean over sequences
    corrected_mean_log_is: float
    baseline_median_abs_log_is: float
    corrected_median_abs_log_is: float

    # PPO-clip behaviour: fraction of seqs with |IS-1| > threshold
    baseline_clip01: float
    corrected_clip01: float
    baseline_clip02: float
    corrected_clip02: float

    # explained variance: how much of the Δ variance the predictor captures
    delta_r2: float

    def pretty(self) -> str:
        lines = [
            f"n_tokens={self.n_tokens:,}  n_seqs={self.n_seqs:,}",
            "",
            "                       baseline   corrected   improvement",
            f"  token  MAE         {self.baseline_token_mae:9.5f}   {self.corrected_token_mae:9.5f}   "
            f"{self._impr(self.baseline_token_mae, self.corrected_token_mae)}",
            f"  token  bias        {self.baseline_token_bias:9.5f}   {self.corrected_token_bias:9.5f}   "
            f"{self._impr(self.baseline_token_bias, self.corrected_token_bias)}",
            f"  seq    mean_log_is {self.baseline_mean_log_is:9.4f}   {self.corrected_mean_log_is:9.4f}   "
            f"{self._impr(abs(self.baseline_mean_log_is), abs(self.corrected_mean_log_is))}",
            f"  seq    med|log_is| {self.baseline_median_abs_log_is:9.4f}   {self.corrected_median_abs_log_is:9.4f}   "
            f"{self._impr(self.baseline_median_abs_log_is, self.corrected_median_abs_log_is)}",
            f"  seq    clip01      {self.baseline_clip01:9.4f}   {self.corrected_clip01:9.4f}   "
            f"{self._impr(self.baseline_clip01, self.corrected_clip01)}",
            f"  seq    clip02      {self.baseline_clip02:9.4f}   {self.corrected_clip02:9.4f}   "
            f"{self._impr(self.baseline_clip02, self.corrected_clip02)}",
            f"  delta  R²          (baseline 0.0)  {self.delta_r2:9.4f}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _impr(b: float, c: float) -> str:
        if b == 0:
            return "    n/a"
        return f"{(b - c) / b * 100:+6.1f}%"


@torch.no_grad()
def evaluate(model, loader, device) -> EvalSummary:
    model.eval()

    sum_abs_baseline = 0.0
    sum_abs_corrected = 0.0
    sum_signed_baseline = 0.0
    sum_signed_corrected = 0.0
    sum_n = 0

    # variance bookkeeping for R²
    sum_target = 0.0
    sum_target_sq = 0.0
    sum_resid_sq = 0.0

    baseline_log_is: List[float] = []
    corrected_log_is: List[float] = []

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        target = batch["sampling_lps"] - batch["training_lps"]  # true Δ
        pred = model(
            batch["token_ids"],
            batch["sampling_lps"],
            batch["mask"],
            batch["n_prompt_tokens"],
            batch["temperature"],
            batch["lora_rank"],
        )

        m = batch["mask"].float()
        n = m.sum().item()

        baseline_resid = target * m  # = target − 0
        corrected_resid = (target - pred) * m  # = target − pred

        sum_abs_baseline += baseline_resid.abs().sum().item()
        sum_abs_corrected += corrected_resid.abs().sum().item()
        sum_signed_baseline += baseline_resid.sum().item()
        sum_signed_corrected += corrected_resid.sum().item()
        sum_n += int(n)

        # R² on target (Δ): 1 - SS_res / SS_tot
        sum_target += baseline_resid.sum().item()
        sum_target_sq += (baseline_resid**2).sum().item()
        sum_resid_sq += (corrected_resid**2).sum().item()

        # per-sequence cumulative diff (= log importance weight)
        baseline_log_is.extend(baseline_resid.sum(dim=1).cpu().tolist())
        # corrected sampling_lp = sampling_lp - pred; corrected log IS = (corr - training) summed
        # = (target - pred).sum
        corrected_log_is.extend(corrected_resid.sum(dim=1).cpu().tolist())

    n = max(sum_n, 1)
    n_seq = max(len(baseline_log_is), 1)

    log_clip01 = (math.log(1.1), math.log(0.9))
    log_clip02 = (math.log(1.2), math.log(0.8))

    def clip_rate(xs, hi_lo):
        hi, lo = hi_lo
        return sum(1 for x in xs if x > hi or x < lo) / max(len(xs), 1)

    def median_abs(xs):
        if not xs:
            return 0.0
        ys = sorted(abs(x) for x in xs)
        return ys[len(ys) // 2]

    # R² on Δ (token level)
    mean_t = sum_target / n
    ss_tot = sum_target_sq - n * mean_t * mean_t
    ss_res = sum_resid_sq
    delta_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return EvalSummary(
        n_tokens=n,
        n_seqs=n_seq,
        baseline_token_mae=sum_abs_baseline / n,
        baseline_token_bias=abs(sum_signed_baseline) / n,
        corrected_token_mae=sum_abs_corrected / n,
        corrected_token_bias=abs(sum_signed_corrected) / n,
        baseline_mean_log_is=sum(baseline_log_is) / n_seq,
        corrected_mean_log_is=sum(corrected_log_is) / n_seq,
        baseline_median_abs_log_is=median_abs(baseline_log_is),
        corrected_median_abs_log_is=median_abs(corrected_log_is),
        baseline_clip01=clip_rate(baseline_log_is, log_clip01),
        corrected_clip01=clip_rate(corrected_log_is, log_clip01),
        baseline_clip02=clip_rate(baseline_log_is, log_clip02),
        corrected_clip02=clip_rate(corrected_log_is, log_clip02),
        delta_r2=float(delta_r2),
    )
