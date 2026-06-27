"""
Plot cross-seed aggregated results for MATH GRPO benchmark.

Reads per-seed jsonl logs from evaluation/results/bench_math_seed{42,43,44}/
and produces:
  - Cross-seed test accuracy trajectories (mean ± std band)
  - Cross-seed bias (corrected mean_diff) trajectories
  - Cross-seed std_cum_diff trajectories
  - Final test accuracy bar chart with per-seed scatter
  - Mean response length trajectories
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
    "grpo_baseline": {"label": "GRPO (baseline)", "color": "#2980B9", "ls": "-"},
    "grpo_predictor": {"label": "GRPO + predictor", "color": "#27AE60", "ls": "--"},
    "grpo_global_mean": {"label": "GRPO + global_mean", "color": "#E67E22", "ls": ":"},
}
GROUP_ORDER = ["grpo_baseline", "grpo_global_mean", "grpo_predictor"]


def load_seed(seed_dir: Path, group: str) -> List[Dict]:
    p = seed_dir / f"{group}_log.jsonl"
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f]


def extract_traj(records: List[Dict], *path) -> np.ndarray:
    out = []
    for r in records:
        v = r
        try:
            for k in path:
                v = v[k]
            out.append(float(v))
        except Exception:
            out.append(float("nan"))
    return np.array(out, dtype=float)


def steps_of(records: List[Dict]) -> np.ndarray:
    return np.array([r["step"] for r in records], dtype=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="evaluation/results")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--output", default="evaluation/results/bench_math_multiseed")
    args = p.parse_args()

    base = Path(args.base)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Load all data: dict[group] -> list of np.ndarray (one per seed)
    data: Dict[str, List[Dict]] = {}
    steps = None
    for g in GROUP_ORDER:
        data[g] = []
        for s in args.seeds:
            recs = load_seed(base / f"bench_math_seed{s}", g)
            if recs:
                data[g].append(recs)
                if steps is None:
                    steps = steps_of(recs)

    if steps is None:
        print("No data found.")
        return

    n_steps = len(steps)
    seeds_str = ",".join(str(s) for s in args.seeds)

    # ── Fig 1: Test accuracy trajectories (mean ± std band) ───────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"MATH GRPO Test Accuracy Across Seeds ({seeds_str})\nMean ± std", fontsize=13)
    for g in GROUP_ORDER:
        trajs = []
        for recs in data[g]:
            trajs.append(np.array([r.get("test_acc", float("nan")) for r in recs], dtype=float))
        if not trajs:
            continue
        M = np.full((len(trajs), n_steps), np.nan)
        for i, t in enumerate(trajs):
            L = min(len(t), n_steps)
            M[i, :L] = t[:L]
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        meta = GROUP_META[g]
        ax.plot(
            steps,
            mean,
            color=meta["color"],
            linestyle=meta["ls"],
            linewidth=2.2,
            label=meta["label"],
        )
        ax.fill_between(steps, mean - std, mean + std, color=meta["color"], alpha=0.15)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "multiseed_test_acc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out / 'multiseed_test_acc.png'}")

    # ── Fig 1b: Mean reward trajectories (mean ± std band) ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"MATH GRPO Mean Reward Across Seeds ({seeds_str})\nMean ± std", fontsize=13)
    for g in GROUP_ORDER:
        trajs = []
        for recs in data[g]:
            trajs.append(np.array([r.get("mean_reward", float("nan")) for r in recs], dtype=float))
        if not trajs:
            continue
        M = np.full((len(trajs), n_steps), np.nan)
        for i, t in enumerate(trajs):
            L = min(len(t), n_steps)
            M[i, :L] = t[:L]
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        meta = GROUP_META[g]
        ax.plot(
            steps,
            mean,
            color=meta["color"],
            linestyle=meta["ls"],
            linewidth=2.2,
            label=meta["label"],
        )
        ax.fill_between(steps, mean - std, mean + std, color=meta["color"], alpha=0.15)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "multiseed_reward.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out / 'multiseed_reward.png'}")

    # ── Fig 2: Corrected bias (mean_diff) trajectories ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"MATH GRPO Corrected Bias (mean_diff) Across Seeds ({seeds_str})", fontsize=13)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
    for g in GROUP_ORDER:
        trajs = [extract_traj(recs, "corrected", "mean_diff") for recs in data[g]]
        M = np.full((len(trajs), n_steps), np.nan)
        for i, t in enumerate(trajs):
            L = min(len(t), n_steps)
            M[i, :L] = t[:L]
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        meta = GROUP_META[g]
        ax.plot(
            steps,
            mean,
            color=meta["color"],
            linestyle=meta["ls"],
            linewidth=2.0,
            label=meta["label"],
        )
        ax.fill_between(steps, mean - std, mean + std, color=meta["color"], alpha=0.15)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Corrected mean_diff")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "multiseed_corr_bias.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out / 'multiseed_corr_bias.png'}")

    # ── Fig 3: Corrected std_cum_diff trajectories ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"MATH GRPO Corrected std_cum_diff Across Seeds ({seeds_str})", fontsize=13)
    for g in GROUP_ORDER:
        trajs = [extract_traj(recs, "corrected", "std_cum_diff") for recs in data[g]]
        M = np.full((len(trajs), n_steps), np.nan)
        for i, t in enumerate(trajs):
            L = min(len(t), n_steps)
            M[i, :L] = t[:L]
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        meta = GROUP_META[g]
        ax.plot(
            steps,
            mean,
            color=meta["color"],
            linestyle=meta["ls"],
            linewidth=2.0,
            label=meta["label"],
        )
        ax.fill_between(steps, mean - std, mean + std, color=meta["color"], alpha=0.15)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Corrected std_cum_diff")
    ax.set_ylim(0, 15)  # clip explosions so typical behavior is visible
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "multiseed_std_cum.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out / 'multiseed_std_cum.png'}")

    # ── Fig 4: Mean response length trajectories ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"MATH GRPO Mean Response Length Across Seeds ({seeds_str})", fontsize=13)
    for g in GROUP_ORDER:
        trajs = []
        for recs in data[g]:
            trajs.append(
                np.array(
                    [r.get("response_length_stats", {}).get("mean", float("nan")) for r in recs],
                    dtype=float,
                )
            )
        M = np.full((len(trajs), n_steps), np.nan)
        for i, t in enumerate(trajs):
            L = min(len(t), n_steps)
            M[i, :L] = t[:L]
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        meta = GROUP_META[g]
        ax.plot(
            steps,
            mean,
            color=meta["color"],
            linestyle=meta["ls"],
            linewidth=2.0,
            label=meta["label"],
        )
        ax.fill_between(steps, mean - std, mean + std, color=meta["color"], alpha=0.15)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Response Tokens")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "multiseed_response_length.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out / 'multiseed_response_length.png'}")

    # ── Fig 5: Final test accuracy bar chart with per-seed scatter ─────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(f"MATH GRPO Final Test Accuracy\nMean ± std over seeds ({seeds_str})", fontsize=13)
    means, stds, all_vals = [], [], []
    for g in GROUP_ORDER:
        vals = []
        for recs in data[g]:
            if recs:
                vals.append(recs[-1].get("test_acc", float("nan")))
        vals = [v for v in vals if not (v is None or math.isnan(v))]
        means.append(np.mean(vals) if vals else 0)
        stds.append(np.std(vals) if vals else 0)
        all_vals.append(vals)

    x = np.arange(len(GROUP_ORDER))
    colors = [GROUP_META[g]["color"] for g in GROUP_ORDER]
    ax.bar(
        x, means, yerr=stds, capsize=8, color=colors, alpha=0.8, edgecolor="black", linewidth=0.6
    )
    # scatter individual seeds
    for i, vals in enumerate(all_vals):
        jitter = np.random.default_rng(i).uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color="black", zorder=5, s=40, alpha=0.7)
    # annotate
    for i, (m, s, vals) in enumerate(zip(means, stds, all_vals, strict=False)):
        ax.text(
            i,
            m + s + 0.02,
            f"{m:.3f}±{s:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        for _v in vals:
            pass
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_META[g]["label"] for g in GROUP_ORDER], fontsize=10)
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    # relative improvement annotation
    bm = means[0]
    for i in range(1, len(GROUP_ORDER)):
        delta = means[i] - bm
        pct = delta / bm * 100 if bm else 0
        sign = "+" if delta >= 0 else ""
        ax.annotate(
            f"{sign}{delta:+.3f} ({sign}{pct:.1f}%)",
            xy=(i, means[i]),
            xytext=(i, means[i] - 0.12),
            ha="center",
            fontsize=9,
            color="green" if delta > 0 else "red",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="green" if delta > 0 else "red", lw=1.2),
        )

    plt.tight_layout()
    plt.savefig(out / "multiseed_final_acc_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out / 'multiseed_final_acc_bars.png'}")

    # ── Fig 5b: Side-by-side final test_acc vs mean_reward summary ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # left: final test_acc
    ax = axes[0]
    ax.set_title(f"Final Test Accuracy\nMean ± std over seeds ({seeds_str})", fontsize=12)
    acc_means, acc_stds = [], []
    for g in GROUP_ORDER:
        vals = []
        for recs in data[g]:
            if recs:
                vals.append(recs[-1].get("test_acc", float("nan")))
        vals = [v for v in vals if not math.isnan(v)]
        acc_means.append(np.mean(vals) if vals else 0)
        acc_stds.append(np.std(vals) if vals else 0)
    ax.bar(
        x,
        acc_means,
        yerr=acc_stds,
        capsize=8,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.6,
    )
    for i, (m, s) in enumerate(zip(acc_means, acc_stds, strict=False)):
        ax.text(
            i, m + s + 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_META[g]["label"] for g in GROUP_ORDER], fontsize=9)
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    # right: final mean_reward (last 5 steps avg to smooth noise)
    ax = axes[1]
    ax.set_title(
        f"Final Mean Reward\nAvg of last 5 steps, mean ± std over seeds ({seeds_str})", fontsize=12
    )
    rw_means, rw_stds = [], []
    for g in GROUP_ORDER:
        per_seed = []
        for recs in data[g]:
            if recs:
                vals = [r.get("mean_reward", float("nan")) for r in recs[-5:]]
                vals = [v for v in vals if not math.isnan(v)]
                if vals:
                    per_seed.append(np.mean(vals))
        rw_means.append(np.mean(per_seed) if per_seed else 0)
        rw_stds.append(np.std(per_seed) if per_seed else 0)
    ax.bar(
        x,
        rw_means,
        yerr=rw_stds,
        capsize=8,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.6,
    )
    for i, (m, s) in enumerate(zip(rw_means, rw_stds, strict=False)):
        ax.text(
            i, m + s + 0.03, f"{m:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_META[g]["label"] for g in GROUP_ORDER], fontsize=9)
    ax.set_ylabel("Mean Reward")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "multiseed_acc_reward_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out / 'multiseed_acc_reward_summary.png'}")

    # ── Fig 6: robust bias stats (median over all steps) + explosion count ────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Panel 1: median |mean_diff| over all steps
    ax = axes[0]
    ax.set_title("Median |corrected mean_diff|\n(over all steps, robust to spikes)", fontsize=12)
    m_vals, s_vals = [], []
    for g in GROUP_ORDER:
        per_seed = []
        for recs in data[g]:
            vs = np.abs(extract_traj(recs, "corrected", "mean_diff"))
            per_seed.append(float(np.nanmedian(vs)))
        m_vals.append(np.mean(per_seed) if per_seed else 0)
        s_vals.append(np.std(per_seed) if per_seed else 0)
    ax.bar(
        x, m_vals, yerr=s_vals, capsize=8, color=colors, alpha=0.8, edgecolor="black", linewidth=0.6
    )
    for i, (m, s) in enumerate(zip(m_vals, s_vals, strict=False)):
        ax.text(
            i, m + s + 0.0001, f"{m:.5f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_META[g]["label"] for g in GROUP_ORDER], fontsize=8)
    ax.set_ylabel("|mean_diff|")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: median std_cum_diff over all steps
    ax = axes[1]
    ax.set_title("Median corrected std_cum_diff\n(over all steps, robust to spikes)", fontsize=12)
    m_vals, s_vals = [], []
    for g in GROUP_ORDER:
        per_seed = []
        for recs in data[g]:
            vs = extract_traj(recs, "corrected", "std_cum_diff")
            per_seed.append(float(np.nanmedian(vs)))
        m_vals.append(np.mean(per_seed) if per_seed else 0)
        s_vals.append(np.std(per_seed) if per_seed else 0)
    ax.bar(
        x, m_vals, yerr=s_vals, capsize=8, color=colors, alpha=0.8, edgecolor="black", linewidth=0.6
    )
    for i, (m, s) in enumerate(zip(m_vals, s_vals, strict=False)):
        ax.text(
            i, m + s + 0.05, f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_META[g]["label"] for g in GROUP_ORDER], fontsize=8)
    ax.set_ylabel("std_cum_diff")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: explosion count (std_cum > 10) per 60 steps
    ax = axes[2]
    ax.set_title("Explosion frequency\n(steps with std_cum_diff > 10, per 60 steps)", fontsize=12)
    m_vals, s_vals = [], []
    for g in GROUP_ORDER:
        per_seed = []
        for recs in data[g]:
            vs = extract_traj(recs, "corrected", "std_cum_diff")
            per_seed.append(float(np.sum(vs > 10)))
        m_vals.append(np.mean(per_seed) if per_seed else 0)
        s_vals.append(np.std(per_seed) if per_seed else 0)
    ax.bar(
        x, m_vals, yerr=s_vals, capsize=8, color=colors, alpha=0.8, edgecolor="black", linewidth=0.6
    )
    for i, (m, s) in enumerate(zip(m_vals, s_vals, strict=False)):
        ax.text(
            i, m + s + 0.1, f"{m:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_META[g]["label"] for g in GROUP_ORDER], fontsize=8)
    ax.set_ylabel("count / 60 steps")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out / "multiseed_bias_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out / 'multiseed_bias_bars.png'}")

    # ── Print summary table (median over all steps, robust to explosions) ──────
    print("\n" + "=" * 78)
    print("CROSS-SEED SUMMARY  (test_acc=final step; bias=median over all steps)")
    print("=" * 78)
    print(
        f"{'group':22s} {'test_acc':>10s} {'±std':>8s} {'med|bias|':>10s} {'med_stdcum':>11s} {'n_explode':>10s}"  # noqa: E501
    )
    for i, g in enumerate(GROUP_ORDER):
        vals = all_vals[i]
        m = np.mean(vals)
        s = np.std(vals)
        # corrected bias median over all steps
        per_seed = []
        for recs in data[g]:
            vs = np.abs(extract_traj(recs, "corrected", "mean_diff"))
            per_seed.append(float(np.nanmedian(vs)))
        cb = np.mean(per_seed) if per_seed else float("nan")
        per_seed2 = []
        for recs in data[g]:
            vs = extract_traj(recs, "corrected", "std_cum_diff")
            per_seed2.append(float(np.nanmedian(vs)))
        sc = np.mean(per_seed2) if per_seed2 else float("nan")
        per_seed3 = []
        for recs in data[g]:
            vs = extract_traj(recs, "corrected", "std_cum_diff")
            per_seed3.append(float(np.sum(vs > 10)))
        ne = np.mean(per_seed3) if per_seed3 else float("nan")
        print(f"{g:22s} {m:>10.3f} {s:>8.3f} {cb:>10.5f} {sc:>11.3f} {ne:>10.1f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
