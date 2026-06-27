"""
Plot RLHF Benchmark Results
============================
绘制 rlhf_bench.py 的输出结果：
  - 训练 reward 曲线（per step）
  - Test accuracy 曲线（per eval step）
  - Bias (mean_diff) 校正效果曲线
  - std_cum_diff 曲线

使用方式:
  python evaluation/plot_bench.py --input evaluation/results/bench --output evaluation/results/bench
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUP_META = {
    "reinforce_baseline": {"label": "REINFORCE (baseline)", "color": "#E74C3C", "ls": "-"},
    "reinforce_predictor": {"label": "REINFORCE + predictor", "color": "#E74C3C", "ls": "--"},
    "reinforce_global_mean": {"label": "REINFORCE + global_mean", "color": "#E74C3C", "ls": ":"},
    "grpo_baseline": {"label": "GRPO (baseline)", "color": "#2980B9", "ls": "-"},
    "grpo_predictor": {"label": "GRPO + predictor", "color": "#2980B9", "ls": "--"},
    "grpo_global_mean": {"label": "GRPO + global_mean", "color": "#2980B9", "ls": ":"},
}


def smooth(vals: List[float], w: int) -> List[float]:
    if w <= 1:
        return vals
    out = []
    for i in range(len(vals)):
        s = max(0, i - w + 1)
        chunk = [v for v in vals[s : i + 1] if not math.isnan(v)]
        out.append(float(np.mean(chunk)) if chunk else float("nan"))
    return out


def load_results(input_dir: Path) -> Dict[str, List[Dict]]:
    all_json = input_dir / "all_results.json"
    if all_json.exists():
        with open(all_json) as f:
            return json.load(f)
    # fallback: load from individual jsonl files
    results = {}
    for p in sorted(input_dir.glob("*_log.jsonl")):
        group = p.stem.replace("_log", "")
        records = []
        with open(p) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        if records:
            results[group] = records
    return results


def extract(records: List[Dict], *keys) -> List[float]:
    vals = []
    for r in records:
        try:
            v = r
            for k in keys:
                v = v[k]
            vals.append(float(v) if v is not None else float("nan"))
        except Exception:
            vals.append(float("nan"))
    return vals


def plot_bench(input_dir: Path, output_dir: Path, smooth_w: int = 5) -> None:
    results = load_results(input_dir)
    if not results:
        print("No results found.")
        return

    print(f"[plot] Loaded groups: {list(results.keys())}")

    # ── Figure 1: Training curves (2×3) ──────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "RLHF Bias Correction Benchmark\n"
        "Solid=baseline, Dashed=predictor | Red=REINFORCE, Blue=GRPO",
        fontsize=13,
    )

    panels = [
        ("Mean Reward\n(higher=better)", "mean_reward", None, True),
        ("Raw Bias (mean_diff)\n[lower abs = better]", "mean_diff", "raw", False),
        ("Corrected Bias (mean_diff)\n[lower abs = better]", "mean_diff", "corrected", False),
        ("Raw std_cum_diff\n[lower=better]", "std_cum_diff", "raw", False),
        ("Corrected std_cum_diff\n[lower=better]", "std_cum_diff", "corrected", False),
        ("Raw clip02\n[lower=better]", "clip02", "raw", False),
    ]

    for ax, (title, key, path, higher_better) in zip(axes.flat, panels, strict=False):
        for group, records in results.items():
            meta = GROUP_META.get(group, {"label": group, "color": "gray", "ls": "-"})
            steps = [r["step"] for r in records]
            if path:
                vals = extract(records, path, key)
            else:
                vals = extract(records, key)
            svals = smooth(vals, smooth_w)
            ax.plot(
                steps,
                svals,
                color=meta["color"],
                linestyle=meta["ls"],
                linewidth=1.8,
                label=meta["label"],
            )
            # raw scatter (faint)
            valid = [(s, v) for s, v in zip(steps, vals, strict=False) if not math.isnan(v)]
            if valid:
                xs, ys = zip(*valid, strict=False)
                ax.scatter(xs, ys, color=meta["color"], alpha=0.2, s=8)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training Step")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if not higher_better:
            ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = output_dir / "bench_training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved training curves to {out_path}")

    # ── Figure 2: Test Accuracy curve ─────────────────────────────────────────
    has_test = any(any(r.get("test_acc") is not None for r in recs) for recs in results.values())
    if has_test:
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        ax2.set_title(
            "Test Accuracy on GSM8K Held-out Set\n(Greedy decoding, exact match)", fontsize=12
        )
        for group, records in results.items():
            meta = GROUP_META.get(group, {"label": group, "color": "gray", "ls": "-"})
            eval_steps = [r["step"] for r in records if r.get("test_acc") is not None]
            eval_accs = [r["test_acc"] for r in records if r.get("test_acc") is not None]
            if eval_steps:
                ax2.plot(
                    eval_steps,
                    eval_accs,
                    color=meta["color"],
                    linestyle=meta["ls"],
                    linewidth=2.2,
                    marker="o",
                    markersize=5,
                    label=meta["label"],
                )
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Test Accuracy")
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path2 = output_dir / "bench_test_acc.png"
        plt.savefig(out_path2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot] Saved test accuracy to {out_path2}")

    # ── Figure 2b: Response length curve ─────────────────────────────────────
    has_len = any(
        any(r.get("response_length_stats", {}).get("mean") is not None for r in recs)
        for recs in results.values()
    )
    if has_len:
        fig2b, ax2b = plt.subplots(figsize=(9, 5))
        ax2b.set_title("Mean Response Length per Step", fontsize=12)
        for group, records in results.items():
            meta = GROUP_META.get(group, {"label": group, "color": "gray", "ls": "-"})
            steps = [r["step"] for r in records]
            lens = [r.get("response_length_stats", {}).get("mean", float("nan")) for r in records]
            valid = [(s, v) for s, v in zip(steps, lens, strict=False) if not math.isnan(v)]
            if valid:
                xs, ys = zip(*valid, strict=False)
                ax2b.plot(
                    xs,
                    ys,
                    color=meta["color"],
                    linestyle=meta["ls"],
                    linewidth=1.8,
                    marker="o",
                    markersize=3,
                    label=meta["label"],
                )
        ax2b.set_xlabel("Training Step")
        ax2b.set_ylabel("Mean Response Tokens")
        ax2b.legend(fontsize=9)
        ax2b.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path2b = output_dir / "bench_response_length.png"
        plt.savefig(out_path2b, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot] Saved response length to {out_path2b}")

    # ── Figure 3: Summary bar chart ────────────────────────────────────────────
    last_n = 10
    summary_metrics = [
        ("Mean Reward\n(last 10 steps)", "mean_reward", None, True),
        ("Test Accuracy\n(last eval)", "test_acc", None, True),
        ("Corrected Bias\n|mean_diff| (last 10)", "mean_diff", "corrected", False),
        ("Corrected std_cum\n(last 10)", "std_cum_diff", "corrected", False),
    ]

    fig3, axes3 = plt.subplots(1, 4, figsize=(18, 5))
    fig3.suptitle("Summary: Last 10 Steps Average", fontsize=13, fontweight="bold")

    for ax, (title, key, path, higher_better) in zip(axes3, summary_metrics, strict=False):
        groups_here = list(results.keys())
        vals_here = []
        for g in groups_here:
            recs = results[g][-last_n:]
            if path:
                vs = [
                    r[path][key]
                    for r in recs
                    if path in r and key in r[path] and r[path][key] is not None
                ]
            else:
                vs = [r[key] for r in recs if key in r and r[key] is not None]
            vs = [v for v in vs if not math.isnan(float(v))]
            vals_here.append(float(np.mean(vs)) if vs else float("nan"))

        colors_here = [GROUP_META.get(g, {}).get("color", "gray") for g in groups_here]
        labels_here = [GROUP_META.get(g, {}).get("label", g) for g in groups_here]

        bars = ax.bar(range(len(groups_here)), vals_here, color=colors_here, alpha=0.85)

        # 标注数值和相对变化
        for _i, (bar, v) in enumerate(zip(bars, vals_here, strict=False)):
            if math.isnan(v):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 标注相对 baseline 的改善
        baseline_vals = {}
        for i, g in enumerate(groups_here):
            algo = g.rsplit("_", 1)[0]
            corr = g.rsplit("_", 1)[1]
            if corr == "baseline":
                baseline_vals[algo] = vals_here[i]

        for i, g in enumerate(groups_here):
            algo = g.rsplit("_", 1)[0]
            corr = g.rsplit("_", 1)[1]
            if corr == "predictor" and algo in baseline_vals:
                bv = baseline_vals[algo]
                cv = vals_here[i]
                if not math.isnan(bv) and not math.isnan(cv) and bv != 0:
                    pct = (cv - bv) / abs(bv) * 100
                    sign = "+" if pct > 0 else ""
                    ax.text(
                        bars[i].get_x() + bars[i].get_width() / 2,
                        bars[i].get_height() / 2,
                        f"{sign}{pct:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="white",
                    )

        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(groups_here)))
        ax.set_xticklabels(
            [label.replace(" + ", "\n+\n").replace(" (", "\n(") for label in labels_here],
            fontsize=7,  # noqa: E501
        )
        ax.grid(True, axis="y", alpha=0.3)
        arrow = "↑" if higher_better else "↓"
        ax.set_ylabel(f"{arrow} better")

    plt.tight_layout()
    out_path3 = output_dir / "bench_summary_bars.png"
    plt.savefig(out_path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved summary bars to {out_path3}")

    print(f"\nAll plots saved to {output_dir}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="evaluation/results/bench")
    p.add_argument("--output", default="evaluation/results/bench")
    p.add_argument("--smooth", type=int, default=5)
    args = p.parse_args()
    plot_bench(Path(args.input), Path(args.output), args.smooth)


if __name__ == "__main__":
    main()
