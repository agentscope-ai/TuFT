"""Plot predictor training curves from training_log.jsonl.

Reads:
  <run_dir>/training_log.jsonl   one JSON record per row, with "epoch",
                                 optional "train" (loss components) and
                                 "val" (full EvalSummary fields)
  <run_dir>/test_summary.json    final test set EvalSummary

Produces a single PNG with 8 panels:
  row 1:  train losses · token MAE · token bias · Δ R²
  row 2:  seq mean_log_is · seq med|log_is| · clip01 · clip02

Usage:
  python predictor/plot.py --run_dir predictor/checkpoints/transformer_22agents_v1
  python predictor/plot.py --run_dir <dir>  --compare predictor/checkpoints/mlp_22agents_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────
def load_run(run_dir: Path) -> Dict[str, Any]:
    """Return:
    epochs : list[int]
    train  : dict of loss component series (L_total, L_token, L_seq, L_bias)
    val    : dict of EvalSummary field series (one per recorded epoch)
    test   : dict (final test summary) or None
    best   : (epoch, corrected_token_mae) for the best checkpoint
    """
    log_path = run_dir / "training_log.jsonl"
    test_path = run_dir / "test_summary.json"

    epochs: List[int] = []
    train_series: Dict[str, List[float]] = {}
    val_series: Dict[str, List[float]] = {}

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            epochs.append(rec["epoch"])
            if "train" in rec:
                for k, v in rec["train"].items():
                    train_series.setdefault(k, []).append(v)
            else:
                # epoch=0 has no train losses
                for k in ("L_total", "L_token", "L_seq", "L_bias"):
                    train_series.setdefault(k, []).append(np.nan)
            if "val" in rec:
                for k, v in rec["val"].items():
                    val_series.setdefault(k, []).append(v)
            else:
                # any epoch missing val -> NaN so x-axis stays aligned
                pass

    # best ckpt = min corrected_token_mae (matches train.py logic)
    cm = val_series.get("corrected_token_mae", [])
    if cm:
        best_idx = int(np.argmin(cm))
        best = (epochs[best_idx], cm[best_idx], best_idx)
    else:
        best = None

    test = None
    if test_path.exists():
        test = json.loads(test_path.read_text())

    return {
        "name": run_dir.name,
        "epochs": np.array(epochs),
        "train": {k: np.array(v) for k, v in train_series.items()},
        "val": {k: np.array(v) for k, v in val_series.items()},
        "test": test,
        "best": best,
    }


# ──────────────────────────────────────────────────────────────────────
# Panel painters
# ──────────────────────────────────────────────────────────────────────
def _vline_best(ax, run, color="black"):
    if run["best"] is not None:
        e, _, _ = run["best"]
        ax.axvline(e, ls=":", color=color, lw=1, alpha=0.6, label=f"{run['name']} best (ep {e})")


def _hline_baseline(ax, value, label):
    ax.axhline(value, ls="--", color="gray", lw=1, alpha=0.7, label=label)


def _plot_train_losses(ax, run):
    eps = run["epochs"]
    tr = run["train"]
    for key, color in [
        ("L_total", "#2c3e50"),
        ("L_token", "#1f77b4"),
        ("L_seq", "#2ca02c"),
        ("L_bias", "#d62728"),
    ]:
        if key in tr:
            ax.plot(eps, tr[key], "-o", ms=3, lw=1.4, color=color, label=key)
    ax.set_title("Train losses")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def _plot_paired(
    ax,
    runs,
    baseline_key,
    corrected_key,
    title,
    ylabel=None,
    log_y=False,
    baseline_label="baseline (no correction)",
):
    """Plot baseline as a horizontal line + corrected as a curve, possibly for
    multiple runs."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    baseline_drawn = False
    for i, run in enumerate(runs):
        v = run["val"]
        if corrected_key not in v:
            continue
        if not baseline_drawn and baseline_key in v and len(v[baseline_key]) > 0:
            # baseline doesn't change with epoch (it's the same val set), so
            # take the median across recorded epochs to be robust to the
            # epoch=0 special case.
            base_val = float(np.nanmedian(v[baseline_key]))
            _hline_baseline(ax, base_val, f"{baseline_label} ({base_val:.4f})")
            baseline_drawn = True
        c = colors[i % len(colors)]
        ax.plot(
            run["epochs"],
            v[corrected_key],
            "-o",
            ms=3,
            lw=1.4,
            color=c,
            label=f"corrected · {run['name']}",
        )
        _vline_best(ax, run, color=c)

    ax.set_title(title)
    ax.set_xlabel("epoch")
    if ylabel:
        ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("symlog", linthresh=0.01)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def _plot_r2(ax, runs):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ax.axhline(0.0, ls="--", color="gray", lw=1, alpha=0.7, label="baseline R² = 0")
    for thr, lbl in [(0.4, "usable (>0.4)"), (0.7, "strong (>0.7)")]:
        ax.axhline(thr, ls=":", color="green", lw=0.8, alpha=0.5)
        ax.text(
            0.02,
            thr + 0.01,
            lbl,
            transform=ax.get_yaxis_transform(),
            fontsize=7,
            color="green",
            alpha=0.7,
        )
    for i, run in enumerate(runs):
        v = run["val"]
        if "delta_r2" not in v:
            continue
        c = colors[i % len(colors)]
        ax.plot(run["epochs"], v["delta_r2"], "-o", ms=3, lw=1.4, color=c, label=f"{run['name']}")
        _vline_best(ax, run, color=c)
    ax.set_title("Δ R² (predictor explains how much variance)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("R²")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def _plot_clip(ax, runs, key="clip01", label_thr="ε=0.1"):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    base_key = f"baseline_{key}"
    corr_key = f"corrected_{key}"
    baseline_drawn = False
    for i, run in enumerate(runs):
        v = run["val"]
        if corr_key not in v:
            continue
        if not baseline_drawn and base_key in v:
            base_val = float(np.nanmedian(v[base_key]))
            _hline_baseline(ax, base_val, f"baseline ({base_val:.3f})")
            baseline_drawn = True
        c = colors[i % len(colors)]
        ax.plot(
            run["epochs"],
            v[corr_key],
            "-o",
            ms=3,
            lw=1.4,
            color=c,
            label=f"corrected · {run['name']}",
        )
        _vline_best(ax, run, color=c)
    ax.set_title(f"Sequence-level clip rate ({label_thr})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("fraction of sequences clipped")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")


# ──────────────────────────────────────────────────────────────────────
def make_summary_text(runs) -> str:
    lines = []
    for run in runs:
        lines.append(f"=== {run['name']} ===")
        if run["best"] is not None:
            e, mae, _ = run["best"]
            lines.append(f"  best ckpt @ epoch {e}  corrected_token_mae={mae:.5f}")
        if run["test"] is not None:
            t = run["test"]
            lines.append(
                f"  TEST  MAE {t['baseline_token_mae']:.4f}->{t['corrected_token_mae']:.4f}  "
                f"bias {t['baseline_token_bias']:.4f}->{t['corrected_token_bias']:.4f}  "
                f"R²={t['delta_r2']:.4f}  clip01 {t['baseline_clip01']:.3f}->{t['corrected_clip01']:.3f}"  # noqa: E501
            )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="checkpoint dir containing training_log.jsonl")
    ap.add_argument(
        "--compare", nargs="*", default=[], help="extra run_dirs to overlay for comparison"
    )
    ap.add_argument(
        "--output", default=None, help="output PNG path (default: <run_dir>/training_curves.png)"
    )
    args = ap.parse_args()

    runs = [load_run(Path(args.run_dir))]
    for d in args.compare:
        runs.append(load_run(Path(d)))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    _plot_train_losses(axes[0, 0], runs[0])  # only show first run's losses
    _plot_paired(
        axes[0, 1], runs, "baseline_token_mae", "corrected_token_mae", "Token MAE", ylabel="MAE"
    )
    _plot_paired(
        axes[0, 2],
        runs,
        "baseline_token_bias",
        "corrected_token_bias",
        "Token bias |E[Δ̂-Δ]|",
        ylabel="|bias|",
    )
    _plot_r2(axes[0, 3], runs)

    _plot_paired(
        axes[1, 0],
        runs,
        "baseline_mean_log_is",
        "corrected_mean_log_is",
        "Sequence mean log(IS weight)",
        ylabel="mean Σ_t (Δ̂-Δ)",
        log_y=False,
    )
    _plot_paired(
        axes[1, 1],
        runs,
        "baseline_median_abs_log_is",
        "corrected_median_abs_log_is",
        "Sequence median |log(IS weight)|",
        ylabel="median |Σ_t (Δ̂-Δ)|",
    )
    _plot_clip(axes[1, 2], runs, key="clip01", label_thr="ε=0.1")
    _plot_clip(axes[1, 3], runs, key="clip02", label_thr="ε=0.2")

    summary = make_summary_text(runs)
    fig.suptitle(
        f"Predictor training curves — {runs[0]['name']}"
        + (" + comparisons" if len(runs) > 1 else "")
        + "\n"
        + summary,
        fontsize=10,
        ha="center",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out = args.output or (Path(args.run_dir) / "training_curves.png")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[plot] saved -> {out}")


if __name__ == "__main__":
    main()
