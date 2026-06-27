"""Length-bin & Clip asymmetry analysis.

Experiment 1: Length-bin analysis
  - Show that bias (log_w_seq = sum of Δ_t) grows linearly with sequence length
  - Compare: raw vs global-mean-subtraction vs ML predictor correction
  - Output: line chart (length bin × mean log_w_seq)

Experiment 2: Clip asymmetry analysis
  - Show that raw clipping is asymmetric (positive >> negative)
  - After correction: should be roughly symmetric
  - Output: grouped bar chart

Experiment 3: Length × Clip cross analysis
  - Per-bin clip rates for raw vs corrected
  - Output: grouped bar chart by length bin

Usage:
    cd evaluation/predictor
    .venv/bin/python length_clip_analysis.py \
        --data ../simulator/logprobs/tp2_fsdp1_id0_noquant.jsonl \
        --checkpoint checkpoints/mlp_22agents_v1/best.pt \
        --output_dir ../simulator/results/length_clip_analysis

    # With tau choices: log(1.2)=0.182, log(1.5)=0.405
    .venv/bin/python length_clip_analysis.py \
        --data ../simulator/logprobs/tp2_fsdp1_id0_noquant.jsonl \
        --checkpoint checkpoints/mlp_22agents_v1/best.pt \
        --tau 0.182 \
        --output_dir ../simulator/results/length_clip_analysis
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── predictor imports ──────────────────────────────────────────────────────────
import torch
from data import (
    LogprobMismatchDataset,
    collate,
    filter_clean_records,
    load_jsonl,
    split_by_weight_version,
)
from model import build_model
from seq_bias_predictor import extract_features_and_targets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Length-bin & Clip asymmetry analysis")
    p.add_argument("--data", required=True, help="path to .logprobs.jsonl (staleness=0 records)")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="path to MLP/transformer best.pt (ML correction). "
        "If omitted, only GBDT seq-bias predictor is used.",
    )
    p.add_argument(
        "--tau",
        type=float,
        default=math.log(1.2),
        help="clip threshold in log-space (default: log(1.2)=0.182)",
    )
    p.add_argument(
        "--bins",
        nargs="+",
        type=int,
        default=[0, 100, 200, 300, 400],
        help="bin edges (open-right last bin). default: 0 100 200 300 400",
    )
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--output_dir", default="results/length_clip_analysis")
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
def load_nn_checkpoint(ckpt_path: str, device: str):
    """Load a trained NN predictor (transformer or MLP)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_saved = ckpt.get("args", {})
    model_type = args_saved.get("model", "transformer")
    model_kwargs: Dict[str, Any] = {"vocab_size": ckpt["vocab_size"]}
    if model_type == "transformer":
        model_kwargs.update(
            {
                "d_model": args_saved.get("d_model", 128),
                "token_emb_dim": args_saved.get("token_emb_dim", 32),
                "n_heads": args_saved.get("n_heads", 4),
                "n_layers": args_saved.get("n_layers", 2),
            }
        )
    else:
        model_kwargs.update(
            {
                "token_emb_dim": args_saved.get("token_emb_dim", 32),
                "hidden": args_saved.get("hidden", 128),
            }
        )
    model = build_model(model_type, **model_kwargs)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(
        f"[model] loaded {model_type} checkpoint, "
        f"{sum(p.numel() for p in model.parameters()):,} params"
    )
    return model, model_type


def get_nn_corrections(
    model,
    model_type: str,
    records: List[Dict[str, Any]],
    max_seq_len: int,
    device: str,
    batch_size: int = 32,
) -> List[float]:
    """Run NN inference → per-sequence predicted bias scalar."""
    ds = LogprobMismatchDataset(records, max_seq_len=max_seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    all_pred_biases = []
    with torch.no_grad():
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            sampling_lps = batch["sampling_lps"].to(device)
            mask = batch["mask"].to(device)
            n_prompt = batch["n_prompt_tokens"].to(device)
            temp = batch["temperature"].to(device)
            lora_r = batch["lora_rank"].to(device)

            # Forward
            pred = model(token_ids, sampling_lps, mask, n_prompt, temp, lora_r)
            # pred: [B, T] per-token predictions
            # per-sequence bias = mean of predicted corrections over valid tokens
            for b in range(pred.shape[0]):
                seq_mask = mask[b]
                seq_pred = pred[b][seq_mask]
                all_pred_biases.append(float(seq_pred.mean().cpu()))

    return all_pred_biases  # list[float], one per record


# ──────────────────────────────────────────────────────────────────────────────
# Build all corrections
# ──────────────────────────────────────────────────────────────────────────────
def build_corrections(
    records: List[Dict[str, Any]],
    max_seq_len: int,
    calibration_mean: float,
    gbdt_pred_biases: Optional[List[float]] = None,
    nn_pred_biases: Optional[List[float]] = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Returns dict of method_name → list of per-token residual arrays.
    Residual = delta - correction  (delta = sampling_lp - training_lp)
    """
    raw_deltas = []
    for r in records:
        T = min(len(r["response_tokens"]), max_seq_len)
        s = np.array(r["sampling_logprobs"][:T], dtype=np.float64)
        t = np.array(r["training_logprobs"][:T], dtype=np.float64)
        raw_deltas.append(s - t)

    corrections = {"raw": list(raw_deltas)}

    # Global mean subtraction (constant b from calibration set)
    corrections["global_mean"] = [d - calibration_mean for d in raw_deltas]

    # GBDT per-sequence predictor
    if gbdt_pred_biases is not None:
        corrections["ml_predictor"] = [d - gbdt_pred_biases[i] for i, d in enumerate(raw_deltas)]

    # NN predictor (if available)
    if nn_pred_biases is not None:
        corrections["nn_predictor"] = [d - nn_pred_biases[i] for i, d in enumerate(raw_deltas)]

    # Oracle: per-sequence mean subtraction (requires training logprobs — upper bound)
    corrections["oracle (per-seq)"] = [d - d.mean() for d in raw_deltas]

    return corrections


# ──────────────────────────────────────────────────────────────────────────────
# Length-bin utilities
# ──────────────────────────────────────────────────────────────────────────────
def make_bins(bin_edges: List[int]) -> List[Tuple[int, int]]:
    """Convert edge list to (lo, hi) pairs with open last bin → (last, inf)."""
    bins = [(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
    bins.append((bin_edges[-1], math.inf))
    return bins


def bin_label(lo: int, hi) -> str:
    if hi == math.inf:
        return f"[{lo},+∞)"
    return f"[{lo},{int(hi)})"


def assign_bins(lengths: np.ndarray, bins: List[Tuple]) -> np.ndarray:
    """Return integer bin index for each sequence length."""
    idx = np.full(len(lengths), len(bins) - 1, dtype=int)
    for i, (lo, hi) in enumerate(bins):
        if hi == math.inf:
            mask = lengths >= lo
        else:
            mask = (lengths >= lo) & (lengths < hi)
        idx[mask] = i
    return idx


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 1: Length-bin mean log_w_seq
# ──────────────────────────────────────────────────────────────────────────────
def experiment1_length_bin(
    records: List[Dict[str, Any]],
    corrections: Dict[str, List[np.ndarray]],
    bins: List[Tuple],
    output_dir: Path,
) -> None:
    lengths = np.array([len(d) for d in corrections["raw"]])
    bin_ids = assign_bins(lengths, bins)
    labels = [bin_label(lo, hi) for lo, hi in bins]

    # Compute log_w_seq = sum(delta) per sequence, then mean per bin
    method_names = list(corrections.keys())
    # display names
    display = {
        "raw": "Raw (no correction)",
        "global_mean": "Global mean subtraction",
        "ml_predictor": "ML predictor (GBDT)",
        "nn_predictor": "ML predictor (NN)",
        "oracle (per-seq)": "Oracle (per-seq mean sub)",
    }
    colors = {
        "raw": "#d62728",
        "global_mean": "#ff7f0e",
        "ml_predictor": "#1f77b4",
        "nn_predictor": "#2ca02c",
        "oracle (per-seq)": "#9467bd",
    }

    # Collect bin stats
    bin_stats: Dict[str, Dict] = {m: {} for m in method_names}
    for m in method_names:
        deltas = corrections[m]
        log_ws = np.array([d.sum() for d in deltas])
        for bi, (lo, hi) in enumerate(bins):
            mask = bin_ids == bi
            n = int(mask.sum())
            if n == 0:
                bin_stats[m][bi] = {"mean": np.nan, "n": 0}
            else:
                bin_stats[m][bi] = {"mean": float(log_ws[mask].mean()), "n": n}

    # Print table
    print(f"\n{'=' * 70}")
    print("  EXPERIMENT 1: Length-bin mean log(w_seq)")
    print(f"{'=' * 70}")
    header = (
        f"{'Bin':<14}"
        + "".join(f"{'N':>8}")
        + "".join(f"{display.get(m, m):>22}" for m in method_names)
    )
    print(header)
    print("-" * len(header))
    for bi, lbl in enumerate(labels):
        n = bin_stats[method_names[0]][bi]["n"]
        row = f"{lbl:<14}{n:>8}" + "".join(
            f"{bin_stats[m][bi]['mean']:>22.4f}" for m in method_names
        )
        print(row)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(bins))
    for m in method_names:
        means = [bin_stats[m][bi]["mean"] for bi in range(len(bins))]
        ns = [bin_stats[m][bi]["n"] for bi in range(len(bins))]
        label = display.get(m, m)
        color = colors.get(m, None)
        ax.plot(x, means, "o-", lw=2, ms=7, label=label, color=color)

    # annotate sample counts on x-axis
    tick_labels = [
        f"{lbl}\n(n={bin_stats[method_names[0]][bi]['n']})" for bi, lbl in enumerate(labels)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.axhline(0, ls="--", color="gray", lw=1, alpha=0.7)
    ax.set_xlabel("Sequence Length Bin (response tokens)", fontsize=12)
    ax.set_ylabel("Mean log(w_seq)  [= mean Σ_t Δ_t per sequence]", fontsize=11)
    ax.set_title(
        "Experiment 1: Length-Dependent Bias in Sequence IS Weights\n"
        "(Raw shows linear growth with length; corrections should flatten)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = output_dir / "exp1_length_bin_mean_log_w.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[exp1] saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 2: Clip asymmetry
# ──────────────────────────────────────────────────────────────────────────────
def experiment2_clip_asymmetry(
    corrections: Dict[str, List[np.ndarray]],
    tau: float,
    output_dir: Path,
) -> None:
    display = {
        "raw": "Raw",
        "global_mean": "Global mean sub",
        "ml_predictor": "ML predictor (GBDT)",
        "nn_predictor": "ML predictor (NN)",
        "oracle (per-seq)": "Oracle (per-seq)",
    }
    method_names = list(corrections.keys())

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT 2: Clip Asymmetry  (τ = {tau:.4f} = log({math.exp(tau):.3f}))")
    print(f"{'=' * 70}")
    print(
        f"{'Method':<26} {'pos_clip':>10} {'neg_clip':>10} {'total_clip':>12} {'ratio pos/neg':>15}"
    )
    print("-" * 73)

    stats = {}
    for m in method_names:
        deltas = corrections[m]
        log_ws = np.array([d.sum() for d in deltas])
        n = len(log_ws)
        pos = int((log_ws > tau).sum())
        neg = int((log_ws < -tau).sum())
        total = pos + neg
        ratio = pos / max(neg, 1)
        stats[m] = {
            "pos": pos,
            "neg": neg,
            "total": total,
            "pos_rate": pos / n,
            "neg_rate": neg / n,
            "total_rate": total / n,
            "ratio": ratio,
        }
        lbl = display.get(m, m)
        print(f"{lbl:<26} {pos / n:>10.4f} {neg / n:>10.4f} {total / n:>12.4f} {ratio:>15.2f}")

    # Grouped bar chart
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute counts
    ax = axes[0]
    x = np.arange(len(method_names))
    w = 0.35
    bars_pos = ax.bar(
        x - w / 2,
        [stats[m]["pos_rate"] for m in method_names],
        w,
        label=f"Positive clip (log_w > +{tau:.3f})",
        color="#d62728",
        alpha=0.85,
    )
    bars_neg = ax.bar(
        x + w / 2,
        [stats[m]["neg_rate"] for m in method_names],
        w,
        label=f"Negative clip (log_w < -{tau:.3f})",
        color="#1f77b4",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([display.get(m, m) for m in method_names], fontsize=9, rotation=10)
    ax.set_ylabel("Clip rate (fraction of sequences)", fontsize=11)
    ax.set_title(f"Clip Asymmetry: pos vs neg clip rates\n(τ = {tau:.3f})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: pos/neg ratio
    ax2 = axes[1]
    ratios = [stats[m]["ratio"] for m in method_names]
    bar_colors = ["#d62728" if r > 1.5 else "#2ca02c" for r in ratios]
    ax2.bar(x, ratios, color=bar_colors, alpha=0.85)
    ax2.axhline(1.0, ls="--", color="gray", lw=1.5, label="Perfect symmetry (ratio=1)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([display.get(m, m) for m in method_names], fontsize=9, rotation=10)
    ax2.set_ylabel("Ratio: positive clips / negative clips", fontsize=11)
    ax2.set_title("Clip Symmetry Ratio (closer to 1.0 = fairer)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Experiment 2: Correction Changes Clipping Fairness, Not Just Rate",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    out = output_dir / "exp2_clip_asymmetry.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[exp2] saved → {out}")

    # 2×2 table print
    print(f"\n  2×2 table (absolute clip counts, n={len(corrections['raw'])} seqs):")
    print(f"  {'Method':<26} {'pos_count':>12} {'neg_count':>12}")
    for m in method_names:
        lbl = display.get(m, m)
        print(f"  {lbl:<26} {stats[m]['pos']:>12}  {stats[m]['neg']:>12}")


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 3: Length × Clip cross-analysis
# ──────────────────────────────────────────────────────────────────────────────
def experiment3_length_clip(
    corrections: Dict[str, List[np.ndarray]],
    bins: List[Tuple],
    tau: float,
    output_dir: Path,
) -> None:
    lengths = np.array([len(d) for d in corrections["raw"]])
    bin_ids = assign_bins(lengths, bins)
    labels = [bin_label(lo, hi) for lo, hi in bins]

    display = {
        "raw": "Raw",
        "global_mean": "Global mean sub",
        "ml_predictor": "ML predictor (GBDT)",
        "nn_predictor": "ML predictor (NN)",
        "oracle (per-seq)": "Oracle (per-seq)",
    }
    method_names = list(corrections.keys())
    colors_map = {
        "raw": "#d62728",
        "global_mean": "#ff7f0e",
        "ml_predictor": "#1f77b4",
        "nn_predictor": "#2ca02c",
        "oracle (per-seq)": "#9467bd",
    }

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT 3: Length × Clip Rate  (τ = {tau:.4f})")
    print(f"{'=' * 70}")
    header = f"{'Bin':<14}" + "".join(f"{display.get(m, m):>20}" for m in method_names)
    print(header)
    print("-" * len(header))

    bin_clip: Dict[str, List[float]] = {m: [] for m in method_names}
    for m in method_names:
        deltas = corrections[m]
        log_ws = np.array([d.sum() for d in deltas])
        for bi in range(len(bins)):
            mask = bin_ids == bi
            n = int(mask.sum())
            if n == 0:
                bin_clip[m].append(np.nan)
            else:
                rate = float(((log_ws[mask] > tau) | (log_ws[mask] < -tau)).mean())
                bin_clip[m].append(rate)

    for bi, lbl in enumerate(labels):
        row = f"{lbl:<14}" + "".join(
            f"{bin_clip[m][bi]:>20.4f}" if not np.isnan(bin_clip[m][bi]) else f"{'n/a':>20}"
            for m in method_names
        )
        print(row)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(bins))
    for m in method_names:
        ax.plot(
            x, bin_clip[m], "o-", lw=2, ms=7, label=display.get(m, m), color=colors_map.get(m, None)
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Sequence Length Bin", fontsize=12)
    ax.set_ylabel(f"Clip rate (|log_w| > τ={tau:.3f})", fontsize=11)
    ax.set_title(
        "Experiment 3: Length × Clip Rate Cross-Analysis\n"
        "(Raw: long sequences clipped disproportionately; corrected: uniform)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = output_dir / "exp3_length_clip_cross.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[exp3] saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"[load] reading {args.data}")
    records = load_jsonl(args.data)
    records = filter_clean_records(records)
    print(f"[load] {len(records)} clean records (staleness == 0)")
    if not records:
        print("[error] no clean records found")
        sys.exit(1)

    # split: use all records for analysis (test split for fair eval)
    train_r, val_r, test_r = split_by_weight_version(records, args.train_frac, args.val_frac)
    print(f"[split] train={len(train_r)}  val={len(val_r)}  test={len(test_r)}")

    # Calibration mean from train set
    train_deltas = []
    for r in train_r:
        T = min(len(r["response_tokens"]), args.max_seq_len)
        s = np.array(r["sampling_logprobs"][:T], dtype=np.float64)
        t = np.array(r["training_logprobs"][:T], dtype=np.float64)
        train_deltas.extend((s - t).tolist())
    calibration_mean = float(np.mean(train_deltas))
    print(f"[calibration] global mean(Δ) on train = {calibration_mean:.6f}")

    # ── 2. GBDT predictor (trained on train_r) ───────────────────────────────
    print("\n[gbdt] training per-sequence bias predictor on train split...")
    X_train, y_train, _ = extract_features_and_targets(train_r, args.max_seq_len)
    X_test, y_test, _ = extract_features_and_targets(test_r, args.max_seq_len)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    gbdt = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42
    )
    gbdt.fit(X_train_sc, y_train)
    gbdt_pred = gbdt.predict(X_test_sc).tolist()

    from sklearn.metrics import r2_score

    print(f"[gbdt] test R² = {r2_score(y_test, gbdt_pred):.4f}")

    # ── 3. NN predictor (optional) ────────────────────────────────────────────
    nn_pred = None
    if args.checkpoint:
        print(f"\n[nn] loading checkpoint: {args.checkpoint}")
        try:
            nn_model, nn_type = load_nn_checkpoint(args.checkpoint, args.device)
            nn_pred = get_nn_corrections(nn_model, nn_type, test_r, args.max_seq_len, args.device)
            print(f"[nn] got {len(nn_pred)} per-seq predictions")
        except Exception as e:
            print(f"[nn] WARNING: failed to load checkpoint: {e}")
            nn_pred = None

    # ── 4. Build all corrections on TEST set ──────────────────────────────────
    print("\n[corrections] building all method corrections on test set...")
    corrections = build_corrections(
        test_r,
        args.max_seq_len,
        calibration_mean,
        gbdt_pred_biases=gbdt_pred,
        nn_pred_biases=nn_pred,
    )
    print(f"[corrections] methods: {list(corrections.keys())}")

    # ── 5. Bin setup ──────────────────────────────────────────────────────────
    bins = make_bins(args.bins)
    print(f"\n[bins] {[bin_label(lo, hi) for lo, hi in bins]}")
    lengths = np.array([len(d) for d in corrections["raw"]])
    bin_ids = assign_bins(lengths, bins)
    for bi, (lo, hi) in enumerate(bins):
        n = int((bin_ids == bi).sum())
        print(
            f"  {bin_label(lo, hi):>12}: {n} sequences  "
            f"(mean_len={lengths[bin_ids == bi].mean():.0f}"
            if n > 0
            else "",
            end="",
        )
        if n > 0:
            print(")")
        else:
            print()

    # ── 6. Run experiments ────────────────────────────────────────────────────
    experiment1_length_bin(test_r, corrections, bins, output_dir)
    experiment2_clip_asymmetry(corrections, args.tau, output_dir)
    experiment3_length_clip(corrections, bins, args.tau, output_dir)

    # ── 7. Combined figure ────────────────────────────────────────────────────
    print(f"\n[done] all outputs saved to: {output_dir}")
    print("  exp1_length_bin_mean_log_w.png")
    print("  exp2_clip_asymmetry.png")
    print("  exp3_length_clip_cross.png")

    # Print key summary
    log_ws_raw = np.array([d.sum() for d in corrections["raw"]])
    lbl_raw = f"raw mean_log_w={log_ws_raw.mean():.4f}, std={log_ws_raw.std():.4f}"
    print(f"\n[summary] {lbl_raw}")
    tau = args.tau
    pos_raw = float((log_ws_raw > tau).mean())
    neg_raw = float((log_ws_raw < -tau).mean())
    print(
        f"[summary] raw clip: pos={pos_raw:.4f}  neg={neg_raw:.4f}  ratio={pos_raw / max(neg_raw, 1e-9):.2f}x"
    )
    if "ml_predictor" in corrections:
        log_ws_ml = np.array([d.sum() for d in corrections["ml_predictor"]])
        pos_ml = float((log_ws_ml > tau).mean())
        neg_ml = float((log_ws_ml < -tau).mean())
        print(
            f"[summary] ml   clip: pos={pos_ml:.4f}  neg={neg_ml:.4f}  ratio={pos_ml / max(neg_ml, 1e-9):.2f}x"
        )


if __name__ == "__main__":
    main()
