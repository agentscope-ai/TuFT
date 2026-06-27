"""Per-sequence bias predictor.

Goal: predict mean(Δ_t) for each sequence using only features available at
sampling time (no training logprobs needed). Then subtract this predicted
bias from all tokens in the sequence to reduce clip rate.

Oracle bound: per-seq mean subtraction → clip02 = 0 (requires training logprobs).
This predictor approximates the oracle using observable features only.

Usage:
    python predictor/seq_bias_predictor.py \
        --data simulator/22_agents_50_step_results.logprobs.jsonl

Features used (all available at sampling/inference time):
    - temperature
    - lora_rank (normalized)
    - n_prompt_tokens
    - response_length
    - mean(sampling_logprobs)
    - std(sampling_logprobs)
    - min(sampling_logprobs)
    - median(sampling_logprobs)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from data import filter_clean_records, load_jsonl, split_by_tenant, split_by_weight_version
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "temperature",
    "lora_rank",
    "n_prompt_tokens",
    "response_length",
    "mean_sampling_lp",
    "std_sampling_lp",
    "min_sampling_lp",
    "median_sampling_lp",
    "max_sampling_lp",
    "p10_sampling_lp",
    "p90_sampling_lp",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-sequence bias predictor")
    p.add_argument("--data", required=True, help="path to .logprobs.jsonl")
    p.add_argument("--split_mode", choices=["weight_version", "tenant"], default="weight_version")
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument(
        "--model", choices=["linear", "ridge", "gbdt"], default="gbdt", help="regression model type"
    )
    p.add_argument("--output", default=None, help="save results JSON")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────
def extract_features_and_targets(
    records: List[Dict[str, Any]],
    max_seq_len: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Extract per-sequence features and bias targets.

    Returns:
        X: [N, n_features] feature matrix
        y: [N] per-sequence mean(Δ_t) = target bias
        deltas: list of per-token Δ arrays (for evaluation)
    """
    features_list = []
    targets = []
    deltas = []

    for r in records:
        T = min(len(r["response_tokens"]), max_seq_len)
        s_lps = np.array(r["sampling_logprobs"][:T], dtype=np.float64)
        t_lps = np.array(r["training_logprobs"][:T], dtype=np.float64)
        delta = s_lps - t_lps

        # Target: per-sequence mean bias
        seq_bias = delta.mean()

        # Features (all observable at sampling time)
        feat = [
            float(r["temperature"]),
            float(r["lora_rank"]) / 64.0,  # normalized
            float(r["n_prompt_tokens"]),
            float(T),  # response length
            float(s_lps.mean()),
            float(s_lps.std()),
            float(s_lps.min()),
            float(np.median(s_lps)),
            float(s_lps.max()),
            float(np.percentile(s_lps, 10)),
            float(np.percentile(s_lps, 90)),
        ]

        features_list.append(feat)
        targets.append(seq_bias)
        deltas.append(delta)

    return np.array(features_list), np.array(targets), deltas


# ──────────────────────────────────────────────────────────────────────
# Evaluation metrics (same as simple_corrections.py)
# ──────────────────────────────────────────────────────────────────────
def compute_clip_metrics(deltas_per_seq: List[np.ndarray]) -> Dict[str, float]:
    """Compute seq-level clip metrics from per-sequence residual arrays."""
    all_tokens = np.concatenate(deltas_per_seq)
    n_tokens = len(all_tokens)
    n_seqs = len(deltas_per_seq)

    token_mae = float(np.mean(np.abs(all_tokens)))
    token_bias = float(abs(np.mean(all_tokens)))

    log_is = np.array([d.sum() for d in deltas_per_seq])
    mean_log_is = float(np.mean(log_is))
    median_abs_log_is = float(np.median(np.abs(log_is)))

    log_clip01 = (math.log(1.1), math.log(0.9))
    log_clip02 = (math.log(1.2), math.log(0.8))

    def clip_rate(xs, hi_lo):
        hi, lo = hi_lo
        return float(np.mean((xs > hi) | (xs < lo)))

    return {
        "n_tokens": n_tokens,
        "n_seqs": n_seqs,
        "token_mae": token_mae,
        "token_bias": token_bias,
        "mean_log_is": mean_log_is,
        "median_abs_log_is": median_abs_log_is,
        "clip01": clip_rate(log_is, log_clip01),
        "clip02": clip_rate(log_is, log_clip02),
    }


def print_results(results: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print comparison."""
    methods = list(results.keys())
    baseline = results["baseline"]

    print(f"\nn_tokens={baseline['n_tokens']:,}  n_seqs={baseline['n_seqs']:,}")
    print()
    header = f"{'metric':<20}" + "".join(f"{m:>20}" for m in methods)
    print(header)
    print("-" * len(header))

    for label, key in [
        ("token MAE", "token_mae"),
        ("token bias", "token_bias"),
        ("mean_log_is", "mean_log_is"),
        ("med|log_is|", "median_abs_log_is"),
        ("clip01", "clip01"),
        ("clip02", "clip02"),
    ]:
        row = f"{label:<20}"
        for m in methods:
            row += f"{results[m][key]:>20.5f}"
        # improvement vs baseline
        row += "  |"
        for m in methods[1:]:
            bv = baseline[key]
            cv = results[m][key]
            if key == "mean_log_is":
                bv, cv = abs(bv), abs(cv)
            if bv == 0:
                row += f"{'n/a':>9}"
            else:
                row += f"{(bv - cv) / bv * 100:>+8.1f}%"
        print(row)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── 1. Load data ──
    print(f"[load] reading {args.data}")
    records = load_jsonl(args.data)
    records = filter_clean_records(records)
    print(f"[load] {len(records)} clean records (staleness == 0)")

    if args.split_mode == "weight_version":
        train_r, val_r, test_r = split_by_weight_version(records, args.train_frac, args.val_frac)
    else:
        train_r, val_r, test_r = split_by_tenant(records, args.train_frac, args.val_frac)
    print(
        f"[split] mode={args.split_mode}  train={len(train_r)}  val={len(val_r)}  test={len(test_r)}"
    )

    # ── 2. Extract features ──
    print("\n[features] extracting per-sequence features...")
    X_train, y_train, deltas_train = extract_features_and_targets(train_r, args.max_seq_len)
    X_test, y_test, deltas_test = extract_features_and_targets(test_r, args.max_seq_len)

    print(f"  train: {X_train.shape[0]} seqs, {len(FEATURE_NAMES)} features")
    print(f"  test:  {X_test.shape[0]} seqs")
    print(f"  target stats (train): mean={y_train.mean():.6f} std={y_train.std():.6f}")
    print(f"  target stats (test):  mean={y_test.mean():.6f} std={y_test.std():.6f}")

    # ── 3. Feature correlation analysis ──
    print(f"\n{'=' * 60}")
    print("  FEATURE CORRELATION WITH per-seq bias")
    print(f"{'=' * 60}")

    for i, name in enumerate(FEATURE_NAMES):
        corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
        print(f"  {name:<25} r = {corr:+.4f}")

    # ── 4. Train model ──
    print(f"\n{'=' * 60}")
    print(f"  TRAINING: {args.model} regressor")
    print(f"{'=' * 60}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if args.model == "linear":
        model = LinearRegression()
    elif args.model == "ridge":
        model = Ridge(alpha=1.0)
    else:  # gbdt
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    model.fit(X_train_scaled, y_train)

    # ── 5. Evaluate regression quality ──
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    print(
        f"\n  [train] R² = {r2_score(y_train, y_pred_train):.4f}  "
        f"MAE = {mean_absolute_error(y_train, y_pred_train):.6f}"
    )
    print(
        f"  [test]  R² = {r2_score(y_test, y_pred_test):.4f}  "
        f"MAE = {mean_absolute_error(y_test, y_pred_test):.6f}"
    )

    # Feature importance (for GBDT)
    if args.model == "gbdt":
        print("\n  Feature importance (GBDT):")
        importances = model.feature_importances_
        for idx in np.argsort(importances)[::-1]:
            print(f"    {FEATURE_NAMES[idx]:<25} {importances[idx]:.4f}")

    # ── 6. Apply correction and compute clip metrics ──
    print(f"\n{'=' * 60}")
    print("  CORRECTION RESULTS (test set)")
    print(f"{'=' * 60}")

    results = {}

    # Baseline (no correction)
    results["baseline"] = compute_clip_metrics(deltas_test)

    # Oracle: per-seq mean subtraction
    oracle_deltas = [d - d.mean() for d in deltas_test]
    results["oracle(per-seq)"] = compute_clip_metrics(oracle_deltas)

    # Global mean subtraction (from train set)
    global_mean = y_train.mean()
    global_deltas = [d - global_mean for d in deltas_test]
    results["global_mean"] = compute_clip_metrics(global_deltas)

    # Per-sequence predictor correction
    pred_deltas = []
    for i, d in enumerate(deltas_test):
        pred_deltas.append(d - y_pred_test[i])
    results[f"predictor({args.model})"] = compute_clip_metrics(pred_deltas)

    print_results(results)

    # ── 7. Summary ──
    b = results["baseline"]
    o = results["oracle(per-seq)"]
    g = results["global_mean"]
    p = results[f"predictor({args.model})"]

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"""
  clip02 comparison:
    baseline:              {b["clip02"]:.4f}
    global mean sub:       {g["clip02"]:.4f}  ({(b["clip02"] - g["clip02"]) / b["clip02"] * 100:+.1f}%)
    predictor ({args.model}):      {p["clip02"]:.4f}  ({(b["clip02"] - p["clip02"]) / b["clip02"] * 100:+.1f}%)
    oracle (per-seq):      {o["clip02"]:.4f}  ({(b["clip02"] - o["clip02"]) / b["clip02"] * 100:+.1f}%)

  med|log_is| comparison:
    baseline:              {b["median_abs_log_is"]:.4f}
    global mean sub:       {g["median_abs_log_is"]:.4f}  ({(b["median_abs_log_is"] - g["median_abs_log_is"]) / b["median_abs_log_is"] * 100:+.1f}%)
    predictor ({args.model}):      {p["median_abs_log_is"]:.4f}  ({(b["median_abs_log_is"] - p["median_abs_log_is"]) / b["median_abs_log_is"] * 100:+.1f}%)
    oracle (per-seq):      {o["median_abs_log_is"]:.4f}  ({(b["median_abs_log_is"] - o["median_abs_log_is"]) / b["median_abs_log_is"] * 100:+.1f}%)

  Predictor captures {r2_score(y_test, y_pred_test) * 100:.1f}% of per-seq bias variance (R²).
  Remaining gap to oracle = room for better features / model.
""")

    # ── 8. Save ──
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "model": args.model,
            "regression_r2_test": r2_score(y_test, y_pred_test),
            "regression_mae_test": mean_absolute_error(y_test, y_pred_test),
            "results": results,
        }
        with open(out, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"[save] {out}")


if __name__ == "__main__":
    main()
