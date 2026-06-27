"""
Plot RLHF comparison results.

Usage:
    python evaluation/plot_rlhf_compare.py --input evaluation/results/compare
    python evaluation/plot_rlhf_compare.py --input evaluation/results/compare --output figs/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


GROUP_COLORS = {
    "baseline": "#e74c3c",  # red
    "global": "#3498db",  # blue
    "predictor": "#2ecc71",  # green
}
GROUP_LABELS = {
    "baseline": "Baseline (no correction)",
    "global": "Global mean subtraction",
    "predictor": "MLP predictor correction",
}
GROUP_LS = {
    "baseline": "-",
    "global": "--",
    "predictor": "-.",
}


def smooth(xs: List[float], window: int = 5) -> List[float]:
    """Simple moving average."""
    if len(xs) <= window:
        return xs
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window // 2)
        hi = min(len(xs), i + window // 2 + 1)
        vals = [v for v in xs[lo:hi] if not math.isnan(v)]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out


def load_group_log(log_path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def extract_series(records: List[Dict], key: str, path: Optional[str] = None) -> List[float]:
    out = []
    for r in records:
        try:
            v = r[path][key] if path else r[key]
            out.append(float(v) if v is not None else float("nan"))
        except Exception:
            out.append(float("nan"))
    return out


def plot_comparison(
    all_data: Dict[str, List[Dict]],
    output_path: Path,
    smooth_window: int = 5,
):
    groups = [g for g in ["baseline", "global", "predictor"] if g in all_data]
    if not groups:
        print("[plot] No data to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "RLHF Mismatch Correction: Training Dynamics Comparison\n"
        "(Lower clip02 / std_cum = better sample utilization)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    panel_specs = [
        # (row, col, title, y_label, key, path, lower_better)
        (0, 0, "Mean Reward (per batch)", "Mean Reward", "mean_reward", None, False),
        (
            0,
            1,
            "Raw clip02 (before correction)\n[fraction of seqs with |IS-1|>0.2]",
            "clip02 rate",
            "clip02",
            "raw",
            True,
        ),
        (
            0,
            2,
            "Corrected clip02 (after correction)\n[effective IS ratio quality]",
            "clip02 rate",
            "clip02",
            "corrected",
            True,
        ),
        (
            1,
            0,
            "Raw std_cum_diff\n[per-seq bias heterogeneity, σ of Σ Δ_t]",
            "std(Σ Δ_t)",
            "std_cum_diff",
            "raw",
            True,
        ),
        (
            1,
            1,
            "Corrected std_cum_diff\n[residual after correction]",
            "std(Σ Δ_t)",
            "std_cum_diff",
            "corrected",
            True,
        ),
        (1, 2, "Raw mean_abs_diff\n[token-level MAE]", "MAE (token)", "mean_abs_diff", "raw", True),
    ]

    for row, col, title, ylabel, key, path, lower_better in panel_specs:
        ax = axes[row][col]

        for g in groups:
            records = all_data[g]
            steps = [r["step"] for r in records]
            vals = extract_series(records, key, path)
            vals_sm = smooth(vals, smooth_window)

            color = GROUP_COLORS.get(g, "gray")
            ls = GROUP_LS.get(g, "-")
            label = GROUP_LABELS.get(g, g)

            # raw data (transparent)
            ax.plot(steps, vals, color=color, alpha=0.2, linewidth=0.8)
            # smoothed
            ax.plot(
                steps,
                vals_sm,
                color=color,
                linestyle=ls,
                linewidth=2,
                label=label,
                marker="",
                zorder=3,
            )

        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel("Training Step", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

        # 标注方向
        direction = "↓ better" if lower_better else "↑ better"
        ax.text(
            0.98,
            0.02,
            direction,
            transform=ax.transAxes,
            fontsize=7,
            ha="right",
            va="bottom",
            color="gray",
            style="italic",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved to {output_path}")
    plt.close()


def plot_summary_bars(
    all_data: Dict[str, List[Dict]],
    output_path: Path,
    last_n: int = 10,
):
    """汇总柱状图：最后N步平均值对比。"""
    groups = [g for g in ["baseline", "global", "predictor"] if g in all_data]
    if not groups:
        return

    metrics = [
        ("mean_reward", None, "Mean Reward\n(last 10 steps)", False),
        ("clip02", "raw", "clip02 (raw)\n[lower=better]", True),
        ("clip02", "corrected", "clip02 (corrected)\n[lower=better]", True),
        ("std_cum_diff", "raw", "std_cum_diff (raw)\n[lower=better]", True),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))
    fig.suptitle(f"Summary: Last {last_n} Steps Average", fontsize=12, fontweight="bold")

    for ax, (key, path, title, lower_better) in zip(axes, metrics, strict=False):
        vals_per_group = {}
        for g in groups:
            records = all_data[g][-last_n:]
            series = extract_series(records, key, path)
            clean = [v for v in series if not math.isnan(v)]
            vals_per_group[g] = float(np.mean(clean)) if clean else 0.0

        colors = [GROUP_COLORS.get(g, "gray") for g in groups]
        x = range(len(groups))
        bars = ax.bar(x, [vals_per_group[g] for g in groups], color=colors, alpha=0.8)

        # 标注数值
        for bar, g in zip(bars, groups, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{vals_per_group[g]:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 标注提升百分比（相对 baseline）
        if "baseline" in vals_per_group:
            b_val = vals_per_group["baseline"]
            for g in groups:
                if g == "baseline":
                    continue
                c_val = vals_per_group[g]
                if b_val != 0:
                    impr = (b_val - c_val) / abs(b_val) * 100
                    if not lower_better:
                        impr = -impr  # for reward, higher is better → flip sign convention
                    idx = groups.index(g)
                    ax.text(
                        idx,
                        vals_per_group[g] * 0.5,
                        f"{impr:+.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                        fontweight="bold",
                    )

        ax.set_title(title, fontsize=9)
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [GROUP_LABELS.get(g, g) for g in groups], fontsize=7, rotation=15, ha="right"
        )
        ax.grid(True, alpha=0.3, axis="y")
        direction = "↓ better" if lower_better else "↑ better"
        ax.set_ylabel(direction, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Summary bars saved to {output_path}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", required=True, help="directory with *_log.jsonl files or all_results.json"
    )
    p.add_argument("--output", default=None, help="output directory (default: same as input)")
    p.add_argument("--smooth", type=int, default=5, help="smoothing window")
    p.add_argument("--last_n", type=int, default=10, help="last N steps for summary")
    args = p.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir

    # ── 加载数据 ──────────────────────────────────────────────────────────
    all_data: Dict[str, List[Dict]] = {}

    # 优先从 all_results.json 加载
    all_results_path = input_dir / "all_results.json"
    if all_results_path.exists():
        with open(all_results_path) as f:
            all_data = json.load(f)
        print(f"[load] Loaded from {all_results_path}")
    else:
        # 从各组的 log.jsonl 文件加载
        for group in ["baseline", "global", "predictor"]:
            log_path = input_dir / f"{group}_log.jsonl"
            if log_path.exists():
                all_data[group] = load_group_log(log_path)
                print(f"[load] {group}: {len(all_data[group])} steps from {log_path}")

    if not all_data:
        print(f"[error] No data found in {input_dir}")
        return

    # ── 打印数据摘要 ──────────────────────────────────────────────────────
    for g, records in all_data.items():
        steps = [r["step"] for r in records]
        print(f"  {g}: {len(records)} records, steps [{min(steps)}, {max(steps)}]")

    # ── 绘图 ──────────────────────────────────────────────────────────────
    plot_comparison(
        all_data,
        output_dir / "rlhf_comparison.png",
        smooth_window=args.smooth,
    )
    plot_summary_bars(
        all_data,
        output_dir / "rlhf_summary_bars.png",
        last_n=args.last_n,
    )
    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
