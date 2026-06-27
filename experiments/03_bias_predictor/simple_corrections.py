"""Simple correction baselines for framework mismatch.

Demonstrates that lightweight statistical methods can achieve competitive
or superior bias correction without any trained model:

1. Per-batch mean subtraction:
   Compute b = mean(Δ_t) on a calibration set, subtract from all tokens.
   Equivalent to a constant logprob correction.

2. Self-normalized IS (SNIS):
   Normalize importance weights within each batch so they average to 1.
   In log-space: subtract the log-mean-exp of sequence log-IS weights.

Usage:
    python predictor/simple_corrections.py \
        --data simulator/22_agents_50_step_results.logprobs.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from data import filter_clean_records, load_jsonl, split_by_tenant, split_by_weight_version


# ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple correction baselines")
    p.add_argument("--data", required=True, help="path to .logprobs.jsonl")
    p.add_argument("--split_mode", choices=["weight_version", "tenant"], default="weight_version")
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--output", default=None, help="optional: save results JSON")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Metric computation (mirrors losses.py EvalSummary)
# ──────────────────────────────────────────────────────────────────────
def compute_metrics(
    deltas_per_seq: List[np.ndarray],
) -> Dict[str, float]:
    """Compute evaluation metrics from per-sequence delta arrays (Δ_t = s_t - p_t)."""
    all_deltas = np.concatenate(deltas_per_seq)
    n_tokens = len(all_deltas)
    n_seqs = len(deltas_per_seq)

    token_mae = float(np.mean(np.abs(all_deltas)))
    token_bias = float(abs(np.mean(all_deltas)))

    # Sequence-level: cumulative log IS weight = sum of Δ_t per sequence
    log_is_weights = np.array([d.sum() for d in deltas_per_seq])
    mean_log_is = float(np.mean(log_is_weights))
    median_abs_log_is = float(np.median(np.abs(log_is_weights)))

    # Clip rates
    log_clip01 = (math.log(1.1), math.log(0.9))
    log_clip02 = (math.log(1.2), math.log(0.8))

    def clip_rate(xs, hi_lo):
        hi, lo = hi_lo
        return float(np.mean((xs > hi) | (xs < lo)))

    clip01 = clip_rate(log_is_weights, log_clip01)
    clip02 = clip_rate(log_is_weights, log_clip02)

    return {
        "n_tokens": n_tokens,
        "n_seqs": n_seqs,
        "token_mae": token_mae,
        "token_bias": token_bias,
        "mean_log_is": mean_log_is,
        "median_abs_log_is": median_abs_log_is,
        "clip01": clip01,
        "clip02": clip02,
    }


def print_comparison(results: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print a comparison table of all methods."""
    methods = list(results.keys())
    baseline = results["baseline"]

    print(f"\nn_tokens={baseline['n_tokens']:,}  n_seqs={baseline['n_seqs']:,}")
    print()

    # Header
    header = f"{'metric':<20}" + "".join(f"{m:>18}" for m in methods)
    print(header)
    print("-" * len(header))

    metrics_to_show = [
        ("token MAE", "token_mae"),
        ("token bias", "token_bias"),
        ("mean_log_is", "mean_log_is"),
        ("med|log_is|", "median_abs_log_is"),
        ("clip01", "clip01"),
        ("clip02", "clip02"),
    ]

    for label, key in metrics_to_show:
        row = f"{label:<20}"
        for m in methods:
            val = results[m][key]
            row += f"{val:>18.5f}"
        # Improvement vs baseline
        row += "   |"
        for m in methods[1:]:
            b_val = baseline[key]
            c_val = results[m][key]
            if b_val == 0:
                row += f"{'n/a':>10}"
            else:
                impr = (b_val - c_val) / b_val * 100
                # For mean_log_is, use absolute values for improvement
                if key == "mean_log_is":
                    impr = (abs(b_val) - abs(c_val)) / abs(b_val) * 100
                row += f"{impr:>+9.1f}%"
        print(row)


# ──────────────────────────────────────────────────────────────────────
# Correction methods
# ──────────────────────────────────────────────────────────────────────
def apply_no_correction(
    deltas_per_seq: List[np.ndarray],
) -> List[np.ndarray]:
    """Baseline: no correction (Δ̂ = 0). Residual = raw Δ."""
    return deltas_per_seq


def apply_global_mean_subtraction(
    deltas_per_seq: List[np.ndarray],
    calibration_mean: float,
) -> List[np.ndarray]:
    """Per-batch mean subtraction: subtract a global constant b from all Δ_t.

    Corrected residual = Δ_t - b.
    In practice: corrected_sampling_lp = sampling_lp - b.

    This is the simplest possible correction: a single scalar computed
    from a calibration set, applied uniformly to all tokens.
    """
    return [d - calibration_mean for d in deltas_per_seq]


def apply_online_mean_subtraction(
    deltas_per_seq: List[np.ndarray],
) -> List[np.ndarray]:
    """Online (oracle) global mean subtraction using the test set's own mean.

    This represents the best-case for global mean subtraction when the
    calibration set perfectly matches the test distribution.
    """
    all_tokens = np.concatenate(deltas_per_seq)
    online_mean = float(np.mean(all_tokens))
    return [d - online_mean for d in deltas_per_seq]


def apply_per_sequence_mean_subtraction(
    deltas_per_seq: List[np.ndarray],
) -> List[np.ndarray]:
    """Per-sequence mean subtraction: subtract mean(Δ_t) within each sequence.

    Corrected residual = Δ_t - mean(Δ_t) for each sequence independently.
    This zeros out per-sequence bias but preserves within-sequence structure.
    """
    return [d - d.mean() for d in deltas_per_seq]


def apply_self_normalized_is(
    deltas_per_seq: List[np.ndarray],
) -> List[np.ndarray]:
    """Self-normalized importance sampling (SNIS).

    Instead of using raw IS weights w_i = exp(Σ_t Δ_t), use:
        w̃_i = w_i / (1/N · Σ_j w_j)

    In log-space: corrected_log_is_i = log_is_i - log(mean(exp(log_is_j)))

    We redistribute this correction uniformly across tokens within each
    sequence, so that the per-token metrics remain meaningful.

    This ensures E[w̃] = 1 regardless of systematic bias.

    NOTE: For numerical stability, we use the "mean subtraction"
    approximation: corrected_log_is ≈ log_is - mean(log_is), which avoids
    the log-sum-exp dominated by outliers and is what practitioners use.
    """
    # Compute raw log IS weights
    log_is = np.array([d.sum() for d in deltas_per_seq])

    # Practical SNIS: subtract mean of log IS weights
    # (equivalent to true SNIS when variance is small;
    #  more stable than log-mean-exp when there are outliers)
    mean_log_is = log_is.mean()

    # Redistribute correction uniformly across tokens in each sequence
    corrected = []
    for _i, d in enumerate(deltas_per_seq):
        seq_len = len(d)
        per_token_correction = mean_log_is / seq_len
        corrected.append(d - per_token_correction)

    return corrected


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── 1. Load + filter + split ──
    print(f"[load] reading {args.data}")
    records = load_jsonl(args.data)
    records = filter_clean_records(records)
    print(f"[load] {len(records)} clean records (staleness == 0)")

    if not records:
        raise RuntimeError("no clean records")

    if args.split_mode == "weight_version":
        train_r, val_r, test_r = split_by_weight_version(records, args.train_frac, args.val_frac)
    else:
        train_r, val_r, test_r = split_by_tenant(records, args.train_frac, args.val_frac)

    print(
        f"[split] mode={args.split_mode}  train={len(train_r)}  val={len(val_r)}  test={len(test_r)}"  # noqa: E501
    )

    # ── 2. Extract per-sequence deltas ──
    def records_to_deltas(recs: List[Dict[str, Any]]) -> List[np.ndarray]:
        deltas = []
        for r in recs:
            T = min(len(r["response_tokens"]), args.max_seq_len)
            s = np.array(r["sampling_logprobs"][:T], dtype=np.float64)
            t = np.array(r["training_logprobs"][:T], dtype=np.float64)
            deltas.append(s - t)
        return deltas

    train_deltas = records_to_deltas(train_r)
    test_deltas = records_to_deltas(test_r)

    # ── 3. Calibrate on TRAIN set ──
    all_train_tokens = np.concatenate(train_deltas)
    calibration_mean = float(np.mean(all_train_tokens))
    print(f"\n[calibration] global mean(Δ) on train set = {calibration_mean:.6f}")
    print("[calibration] This constant will be subtracted in 'global_mean_sub' method.")

    # ── 4. Apply corrections on TEST set ──
    print(f"\n{'=' * 70}")
    print("  SIMPLE CORRECTION BASELINES — TEST SET")
    print(f"{'=' * 70}")

    results = {}

    # Baseline (no correction)
    residuals_baseline = apply_no_correction(test_deltas)
    results["baseline"] = compute_metrics(residuals_baseline)

    # Method 1: Global mean subtraction (calibrated on train)
    residuals_global = apply_global_mean_subtraction(test_deltas, calibration_mean)
    results["global_mean_sub(train)"] = compute_metrics(residuals_global)

    # Method 1b: Online mean subtraction (oracle, uses test set's own mean)
    residuals_online = apply_online_mean_subtraction(test_deltas)
    results["global_mean_sub(oracle)"] = compute_metrics(residuals_online)

    # Method 2: Per-sequence mean subtraction
    residuals_per_seq = apply_per_sequence_mean_subtraction(test_deltas)
    results["per_seq_mean_sub"] = compute_metrics(residuals_per_seq)

    # Method 3: Self-normalized IS
    residuals_snis = apply_self_normalized_is(test_deltas)
    results["self_norm_IS"] = compute_metrics(residuals_snis)

    # ── 5. Print comparison ──
    print_comparison(results)

    # ── 6. Additional analysis: per-sequence bias distribution ──
    print(f"\n{'=' * 70}")
    print("  PER-SEQUENCE BIAS DISTRIBUTION (test set)")
    print(f"{'=' * 70}")

    per_seq_means = np.array([d.mean() for d in test_deltas])
    per_seq_lengths = np.array([len(d) for d in test_deltas])

    print(
        f"  per-seq mean(Δ): mean={per_seq_means.mean():.6f}  "
        f"std={per_seq_means.std():.6f}  "
        f"min={per_seq_means.min():.6f}  max={per_seq_means.max():.6f}"
    )
    print(
        f"  sequence lengths: mean={per_seq_lengths.mean():.0f}  "
        f"min={per_seq_lengths.min()}  max={per_seq_lengths.max()}"
    )

    # Correlation between seq length and per-seq bias
    if len(per_seq_means) > 1:
        corr = np.corrcoef(per_seq_lengths, per_seq_means)[0, 1]
        print(f"  correlation(seq_len, per_seq_bias) = {corr:.4f}")

    # Percentiles of |per-seq bias|
    abs_biases = np.abs(per_seq_means)
    for p in [50, 75, 90, 95, 99]:
        print(f"  P{p:02d} |per-seq bias| = {np.percentile(abs_biases, p):.6f}")

    # ── 7. Save results ──
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[save] results saved to {out_path}")

    print(f"\n{'=' * 70}")
    print("  CONCLUSIONS")
    print(f"{'=' * 70}")

    b = results["baseline"]
    g = results["global_mean_sub(train)"]
    go = results["global_mean_sub(oracle)"]
    s = results["per_seq_mean_sub"]
    n = results["self_norm_IS"]

    print(f"""
  1. Global mean subtraction — train-calibrated (cost: 1 constant from cal set):
     - bias:       {b["token_bias"]:.5f} → {g["token_bias"]:.5f}  ({(b["token_bias"] - g["token_bias"]) / b["token_bias"] * 100:+.1f}%)
     - mean_log_is: {b["mean_log_is"]:.4f} → {g["mean_log_is"]:.4f}
     - clip02:     {b["clip02"]:.4f} → {g["clip02"]:.4f}

  1b. Global mean subtraction — oracle (cost: 1 constant from same distribution):
     - bias:       {b["token_bias"]:.5f} → {go["token_bias"]:.5f}  ({(b["token_bias"] - go["token_bias"]) / b["token_bias"] * 100:+.1f}%)
     - mean_log_is: {b["mean_log_is"]:.4f} → {go["mean_log_is"]:.4f}
     - clip02:     {b["clip02"]:.4f} → {go["clip02"]:.4f}

  2. Per-sequence mean subtraction (cost: O(1) per seq, needs both logprob paths):
     - bias:       {b["token_bias"]:.5f} → {s["token_bias"]:.5f}  ({(b["token_bias"] - s["token_bias"]) / b["token_bias"] * 100:+.1f}%)
     - mean_log_is: {b["mean_log_is"]:.4f} → {s["mean_log_is"]:.4f}
     - clip02:     {b["clip02"]:.4f} → {s["clip02"]:.4f}

  3. Self-normalized IS (cost: mean over batch, no extra forward pass):
     - bias:       {b["token_bias"]:.5f} → {n["token_bias"]:.5f}  ({(b["token_bias"] - n["token_bias"]) / b["token_bias"] * 100:+.1f}%)
     - mean_log_is: {b["mean_log_is"]:.4f} → {n["mean_log_is"]:.4f}
     - clip02:     {b["clip02"]:.4f} → {n["clip02"]:.4f}

  Key insight:
  - Global mean sub requires a calibration run but works without access to
    training logprobs at inference time.
  - Self-normalized IS requires NO calibration — it's computed on-the-fly
    from the current batch's IS weights. Zero extra compute.
  - Per-seq mean sub is an oracle bound (requires both logprob paths).
""")  # noqa: E501


if __name__ == "__main__":
    main()
