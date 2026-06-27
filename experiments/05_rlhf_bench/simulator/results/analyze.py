"""
Comprehensive analysis script for simulator results.

Usage:
    python analyze.py <target_dir>

    <target_dir>  directory containing a ``res_logs/`` subfolder. Every
                  ``*.json`` file inside ``res_logs/`` is treated as one
                  backend / server configuration, and the script compares
                  performance across all of them.

                  For backward compatibility, if ``res_logs/`` does not
                  exist, the script falls back to discovering
                  ``results*.json`` directly under ``<target_dir>``.

                  Each JSON file's stem is used as its display label
                  (e.g. ``orgin.json`` -> ``orgin``).

Outputs:
    - Console text report
    - PNG figures saved under ``<target_dir>/res_figures/``

Figures:
    plot_reward_<backend>.png          -- reward curve per task, A vs shared
    plot_staleness_<backend>.png       -- per-step staleness distribution
    plot_runtime_<backend>.png         -- sampling / training / sync stacked bars
    plot_mode_aggregate.png            -- A vs shared aggregate metrics
    plot_cross_run_comparison.png      -- final acc + mean staleness across backends
    plot_scheduling_latency_improvement.png
                                       -- per-tenant sample latency across backends
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TASKS = ["gsm8k", "countdown", "math", "mbpp", "humaneval", "ifeval"]
MODES = ["A", "shared"]


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze simulator result JSON files.")
    parser.add_argument(
        "target_dir",
        nargs="?",
        default=None,
        help="Directory containing results*.json files. "
        "Defaults to the directory containing this script.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────────────────────────────────
def load_runs(target_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Discover all ``*.json`` files inside ``<target_dir>/res_logs/`` (sorted)
    and load them. Each file is treated as one backend / server configuration
    and the file stem is used as its label.

    Falls back to ``<target_dir>/results*.json`` when ``res_logs/`` is missing,
    to remain compatible with older result layouts.

    Returns {label: json_dict}.
    """
    res_logs = target_dir / "res_logs"
    if res_logs.is_dir():
        candidates = sorted(res_logs.glob("*.json"))
        source = res_logs
    else:
        candidates = sorted(target_dir.glob("results*.json"))
        source = target_dir

    if not candidates:
        print(f"[ERROR] No JSON result files found in {source}")
        sys.exit(1)

    runs: dict[str, dict[str, Any]] = {}
    for fp in candidates:
        with open(fp, "r") as f:
            runs[fp.stem] = json.load(f)
        print(f"loaded {fp}  (backend label = {fp.stem})")
    return runs


def split_id(tenant_id: str) -> tuple[str, str]:
    task, mode = tenant_id.rsplit("-", 1)
    return task, mode


# ──────────────────────────────────────────────────────────────────────
# Per-run text report
# ──────────────────────────────────────────────────────────────────────
def text_report(label: str, data: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print(f" REPORT: {label}")
    print("=" * 88)
    print(f"backend           : {data.get('backend')}")
    print(f"base_model        : {data.get('base_model')}")
    print(
        f"total wall-clock  : {data.get('total_wall_clock_seconds'):.2f} s "
        f"({data.get('total_wall_clock_seconds') / 60:.2f} min)"
    )
    per = data["per_tenant"]
    print(f"# tenants         : {len(per)}")

    print("\n── Accuracy / Reward ─────────────────────────────────────────────────────────")
    print(
        f"{'tenant':22s} {'final_acc':>10s} {'init_R':>8s} {'final_R':>8s} {'Δ_R':>8s} "
        f"{'mean_R':>8s}"
    )
    for tid, m in per.items():
        rc = m["reward_curve"]
        init_r = rc[0][1] if rc else 0.0
        final_r = rc[-1][1] if rc else 0.0
        mean_r = float(np.mean([r for _, r in rc])) if rc else 0.0
        print(
            f"{tid:22s} {m['final_accuracy']:>10.4f} {init_r:>8.4f} "
            f"{final_r:>8.4f} {final_r - init_r:>+8.4f} {mean_r:>8.4f}"
        )

    print("\n── Staleness ──────────────────────────────────────────────────────────────────")
    print(f"{'tenant':22s} {'mean':>8s} {'gap=0':>6s} {'gap=1':>6s} {'gap=2':>6s} {'gap≥3':>6s}")
    for tid, m in per.items():
        agg: dict[int, int] = defaultdict(int)
        for step in m["staleness_per_step"]:
            for k, v in step["distribution"].items():
                agg[int(k)] += v
        total = sum(agg.values()) or 1
        g0 = agg.get(0, 0) / total * 100
        g1 = agg.get(1, 0) / total * 100
        g2 = agg.get(2, 0) / total * 100
        g3 = sum(v for k, v in agg.items() if k >= 3) / total * 100
        print(
            f"{tid:22s} {m['mean_staleness']:>8.3f} {g0:>5.1f}% {g1:>5.1f}% {g2:>5.1f}% {g3:>5.1f}%"
        )

    print("\n── Runtime breakdown (seconds) ────────────────────────────────────────────────")
    print(
        f"{'tenant':22s} {'sample':>8s} {'train':>8s} {'sync':>8s} {'samp_lat_ms':>12s} "
        f"{'tot_samp':>9s} {'used':>5s}"
    )
    for tid, m in per.items():
        used = m["train_steps_completed"] * 64  # buffer_size assumed 64
        print(
            f"{tid:22s} {m['total_sampling_seconds']:>8.1f} "
            f"{m['total_training_seconds']:>8.1f} "
            f"{m['total_sync_weights_seconds']:>8.1f} "
            f"{m['mean_sample_latency_ms']:>12.1f} "
            f"{m['total_samples']:>9d} {used:>5d}"
        )

    print("\n── Per-Mode aggregate (avg across tasks) ──────────────────────────────────────")
    by_mode: dict[str, list[dict[str, Any]]] = {"A": [], "shared": []}
    for tid, m in per.items():
        _, mode = split_id(tid)
        if mode in by_mode:
            by_mode[mode].append(m)
    for mode, ms in by_mode.items():
        if not ms:
            continue
        print(
            f"  mode={mode:6s}  "
            f"acc={np.mean([m['final_accuracy'] for m in ms]):.4f}  "
            f"stale={np.mean([m['mean_staleness'] for m in ms]):.3f}  "
            f"tot_samp={np.mean([m['total_samples'] for m in ms]):.0f}  "
            f"samp_lat={np.mean([m['mean_sample_latency_ms'] for m in ms]):.0f}ms  "
            f"sample_t={np.mean([m['total_sampling_seconds'] for m in ms]):.1f}s  "
            f"train_t={np.mean([m['total_training_seconds'] for m in ms]):.1f}s  "
            f"sync_t={np.mean([m['total_sync_weights_seconds'] for m in ms]):.1f}s"
        )


def cross_run_text_summary(runs: dict[str, dict[str, Any]]) -> None:
    """Print per-tenant latency comparison across all backends.

    The first backend (alphabetically) is used as the baseline, and every
    other backend reports its absolute / relative delta against it.
    """
    if len(runs) < 2:
        return
    labels = list(runs.keys())
    base_label = labels[0]
    other_labels = labels[1:]
    base_per = runs[base_label]["per_tenant"]

    # tenants present in every backend
    common = [t for t in base_per if all(t in runs[lbl]["per_tenant"] for lbl in other_labels)]
    if not common:
        return

    print("\n" + "=" * 88)
    print(f" CROSS-BACKEND: baseline = {base_label}  (mean_sample_latency_ms)")
    print(" backends      : " + ", ".join(labels))
    print("=" * 88)

    # Header
    header = f"{'tenant':22s} {base_label:>14s}"
    for lbl in other_labels:
        header += f" {lbl:>14s} {'Δ_ms':>8s} {'Δ_%':>8s}"
    print(header)

    # Per-tenant rows
    base_lat: list[float] = []
    other_lat: dict[str, list[float]] = {lbl: [] for lbl in other_labels}
    for tid in common:
        b = base_per[tid]["mean_sample_latency_ms"]
        base_lat.append(b)
        row = f"{tid:22s} {b:>14.1f}"
        for lbl in other_labels:
            o = runs[lbl]["per_tenant"][tid]["mean_sample_latency_ms"]
            other_lat[lbl].append(o)
            pct = (o - b) / b * 100 if b else 0.0
            row += f" {o:>14.1f} {o - b:>+8.1f} {pct:>+7.1f}%"
        print(row)

    # Average row
    print("-" * len(header))
    mb = float(np.mean(base_lat))
    avg_row = f"{'AVERAGE':22s} {mb:>14.1f}"
    for lbl in other_labels:
        mo = float(np.mean(other_lat[lbl]))
        pct = (mo - mb) / mb * 100 if mb else 0.0
        avg_row += f" {mo:>14.1f} {mo - mb:>+8.1f} {pct:>+7.1f}%"
    print(avg_row)


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────
def plot_reward_curves(runs: dict[str, dict[str, Any]], fig_dir: Path) -> None:
    for label, data in runs.items():
        per = data["per_tenant"]
        tasks_found = [t for t in TASKS if any(f"{t}-{m}" in per for m in MODES)]
        n = len(tasks_found)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).flatten()
        for ax, task in zip(axes, tasks_found):
            for mode, style in [("A", "o-"), ("shared", "s--")]:
                tid = f"{task}-{mode}"
                if tid not in per:
                    continue
                rc = per[tid]["reward_curve"]
                ax.plot(
                    [s for s, _ in rc],
                    [r for _, r in rc],
                    style,
                    label=f"{mode}  (acc={per[tid]['final_accuracy']:.3f})",
                )
            ax.set_title(task)
            ax.set_xlabel("train step")
            ax.set_ylabel("mean reward")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle(f"Reward curves — {label}", fontsize=13)
        fig.tight_layout()
        out = fig_dir / f"plot_reward_{label}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"saved {out}")


def plot_staleness_per_step(runs: dict[str, dict[str, Any]], fig_dir: Path) -> None:
    for label, data in runs.items():
        per = data["per_tenant"]
        n = len(per)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = np.array(axes).flatten()
        for ax, (tid, m) in zip(axes, per.items()):
            steps_data = m["staleness_per_step"]
            steps = [s["step"] for s in steps_data]
            all_gaps = sorted({int(k) for s in steps_data for k in s["distribution"]})
            bottoms = np.zeros(len(steps))
            cmap = plt.cm.viridis(np.linspace(0.2, 0.9, max(len(all_gaps), 1)))
            for gap, color in zip(all_gaps, cmap):
                vals = np.array([s["distribution"].get(str(gap), 0) for s in steps_data])
                ax.bar(steps, vals, bottom=bottoms, label=f"gap={gap}", color=color)
                bottoms += vals
            ax.set_title(f"{tid} (mean={m['mean_staleness']:.2f})", fontsize=10)
            ax.set_xlabel("train step")
            ax.set_ylabel("# samples")
            ax.legend(fontsize=7, loc="upper left")
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle(f"Staleness distribution per step — {label}", fontsize=13)
        fig.tight_layout()
        out = fig_dir / f"plot_staleness_{label}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"saved {out}")


def plot_runtime_breakdown(runs: dict[str, dict[str, Any]], fig_dir: Path) -> None:
    for label, data in runs.items():
        per = data["per_tenant"]
        ids = list(per.keys())
        sampling = [per[t]["total_sampling_seconds"] for t in ids]
        training = [per[t]["total_training_seconds"] for t in ids]
        syncing = [per[t]["total_sync_weights_seconds"] for t in ids]
        x = np.arange(len(ids))
        fig, ax = plt.subplots(figsize=(max(10, len(ids) * 1.1), 5.5))
        ax.bar(x, sampling, label="sampling", color="#4c72b0")
        ax.bar(x, training, bottom=sampling, label="training", color="#dd8452")
        ax.bar(
            x,
            syncing,
            bottom=np.array(sampling) + np.array(training),
            label="sync_weights",
            color="#55a868",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=35, ha="right")
        ax.set_ylabel("seconds")
        ax.set_title(f"Runtime breakdown — {label}")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out = fig_dir / f"plot_runtime_{label}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"saved {out}")


def plot_mode_aggregate(runs: dict[str, dict[str, Any]], fig_dir: Path) -> None:
    metrics = [
        ("final_accuracy", "Final accuracy"),
        ("mean_staleness", "Mean staleness"),
        ("mean_sample_latency_ms", "Sample latency (ms)"),
        ("total_samples", "Total samples"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    for ax, (key, title) in zip(axes, metrics):
        labels = list(runs.keys())
        x = np.arange(len(labels))
        width = 0.35
        a_vals, s_vals = [], []
        for label in labels:
            per = runs[label]["per_tenant"]
            a_vals.append(float(np.mean([per[f"{t}-A"][key] for t in TASKS if f"{t}-A" in per])))
            s_vals.append(
                float(np.mean([per[f"{t}-shared"][key] for t in TASKS if f"{t}-shared" in per]))
            )
        ax.bar(x - width / 2, a_vals, width, label="-A (Per-User)")
        ax.bar(x + width / 2, s_vals, width, label="-shared")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = fig_dir / "plot_mode_aggregate.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


def plot_cross_run_comparison(runs: dict[str, dict[str, Any]], fig_dir: Path) -> None:
    if len(runs) < 2:
        return
    labels = list(runs.keys())
    # Use tenants common to ALL backends, preserving baseline ordering
    base_per = runs[labels[0]]["per_tenant"]
    common = [t for t in base_per if all(t in runs[lbl]["per_tenant"] for lbl in labels[1:])]
    if not common:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    width = 0.8 / len(labels)
    x = np.arange(len(common))
    for i, label in enumerate(labels):
        per = runs[label]["per_tenant"]
        axes[0].bar(x + i * width, [per[t]["final_accuracy"] for t in common], width, label=label)
        axes[1].bar(x + i * width, [per[t]["mean_staleness"] for t in common], width, label=label)
    for ax, ylabel, title in zip(
        axes,
        ["final accuracy", "mean staleness"],
        ["Final accuracy across runs", "Mean staleness across runs"],
    ):
        ax.set_xticks(x + width * (len(labels) - 1) / 2)
        ax.set_xticklabels(common, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = fig_dir / "plot_cross_run_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


def plot_scheduling_latency_improvement(runs: dict[str, dict[str, Any]], fig_dir: Path) -> None:
    """Per-tenant + aggregate sample-latency comparison across N backends.

    The first backend (alphabetically) is used as the baseline; every other
    backend's bar is annotated with its relative delta vs the baseline.
    """
    if len(runs) < 2:
        return
    labels = list(runs.keys())
    base_label = labels[0]
    base_per = runs[base_label]["per_tenant"]
    tenants = [t for t in base_per if all(t in runs[lbl]["per_tenant"] for lbl in labels)]
    if not tenants:
        return

    # Collect per-backend latency vectors
    lat_by_backend: dict[str, list[float]] = {
        lbl: [runs[lbl]["per_tenant"][t]["mean_sample_latency_ms"] for t in tenants]
        for lbl in labels
    }
    mean_by_backend = {lbl: float(np.mean(v)) for lbl, v in lat_by_backend.items()}
    base_lat = lat_by_backend[base_label]
    mean_base = mean_by_backend[base_label]

    # Color palette: baseline red, others picked from tab10
    cmap = plt.get_cmap("tab10")
    colors = {base_label: "#c44e52"}
    for i, lbl in enumerate(labels[1:]):
        colors[lbl] = cmap(i % 10)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1.4]})

    # ── Left: per-tenant grouped bars ─────────────────────────────────
    # Per-tenant baseline = the SLOWEST backend for that tenant.
    ax = axes[0]
    x = np.arange(len(tenants))
    n = len(labels)
    total_w = 0.8
    w = total_w / n
    # Per-tenant max latency (used as the "slowest" baseline for delta)
    per_tenant_max = [max(lat_by_backend[lbl][j] for lbl in labels) for j in range(len(tenants))]
    for i, lbl in enumerate(labels):
        offset = (i - (n - 1) / 2) * w
        ax.bar(
            x + offset,
            lat_by_backend[lbl],
            w,
            label=lbl,
            color=colors[lbl],
            edgecolor="black",
            linewidth=0.5,
        )
        # Annotate delta vs the slowest backend for that tenant
        for j, (xi, o) in enumerate(zip(x, lat_by_backend[lbl])):
            slow = per_tenant_max[j]
            if slow <= 0 or o == slow:
                # Slowest bar itself: mark as 0% reference
                ax.annotate(
                    "base",
                    xy=(xi + offset, o),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6,
                    color="#666666",
                )
                continue
            d = (o - slow) / slow * 100  # negative => faster than slowest
            ax.annotate(
                f"{d:+.0f}%",
                xy=(xi + offset, o),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#2a7a2a" if d < 0 else "#a02020",
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(tenants, rotation=35, ha="right")
    ax.set_ylabel("mean_sample_latency_ms", fontsize=11)
    ax.set_title(
        f"Per-tenant sample latency across {n} backends (% vs slowest backend per tenant)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ymax = max(max(v) for v in lat_by_backend.values())
    ax.set_ylim(0, ymax * 1.18)

    # ── Right: aggregate per backend ──────────────────────────────────
    # Aggregate baseline = the SLOWEST backend by mean latency.
    ax = axes[1]
    means = [mean_by_backend[lbl] for lbl in labels]
    slow_label = max(mean_by_backend, key=mean_by_backend.get)
    mean_slow = mean_by_backend[slow_label]
    bars = ax.bar(
        labels,
        means,
        color=[colors[lbl] for lbl in labels],
        edgecolor="black",
        linewidth=0.7,
        width=0.6,
    )
    for bar, lbl, val in zip(bars, labels, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.0f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        if lbl == slow_label:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                "slowest\n(base)",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#444444",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888888", lw=1.0),
            )
            continue
        pct = (val - mean_slow) / mean_slow * 100 if mean_slow else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{pct:+.1f}%",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#2a7a2a" if pct < 0 else "#a02020",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", ec="#2a7a2a" if pct < 0 else "#a02020", lw=1.2
            ),
        )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("mean_sample_latency_ms", fontsize=11)
    ax.set_title(
        f"Average across {len(tenants)} tenants (% vs slowest = {slow_label})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Sample latency comparison across {n} backends "
        f"(baseline = slowest run; aggregate slowest = {slow_label})",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    fig.tight_layout()
    out = fig_dir / "plot_scheduling_latency_improvement.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # Resolve target directory
    if args.target_dir is None:
        target_dir = Path(__file__).resolve().parent
    else:
        target_dir = Path(args.target_dir).resolve()

    if not target_dir.is_dir():
        print(f"[ERROR] Not a directory: {target_dir}")
        sys.exit(1)

    fig_dir = target_dir / "res_figures"
    fig_dir.mkdir(exist_ok=True)

    print(f"target_dir : {target_dir}")
    print(f"figures    : {fig_dir}")

    runs = load_runs(target_dir)

    for label, data in runs.items():
        text_report(label, data)
    cross_run_text_summary(runs)

    print(f"\n── Generating figures into {fig_dir} ─────────────────────────────")
    plot_reward_curves(runs, fig_dir)
    plot_staleness_per_step(runs, fig_dir)
    plot_runtime_breakdown(runs, fig_dir)
    plot_mode_aggregate(runs, fig_dir)
    if len(runs) >= 2:
        plot_cross_run_comparison(runs, fig_dir)
        plot_scheduling_latency_improvement(runs, fig_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
