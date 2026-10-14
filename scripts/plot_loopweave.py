"""Plot every paired repetition from the real-engine throughput measurements.

Usage: uv run --no-project --with matplotlib python scripts/plot_loopweave.py RESULTS.json
"""

import argparse
import csv
import io
import json
import statistics
from pathlib import Path

import matplotlib
from summarize_loopweave import comparison_runs


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/loopweave/performance.png"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if data.get("returncode", 0) or not data["runs"]:
        raise ValueError("Successful performance runs are required")
    runs = comparison_runs(data)
    policies = ["round_robin", "loopweave"]
    rows = {
        p: sorted((r for r in runs if r["policy"] == p), key=lambda r: r["repeat"])
        for p in policies
    }
    for policy in policies[1:]:
        if [r["repeat"] for r in rows[policies[0]]] != [r["repeat"] for r in rows[policy]]:
            raise ValueError("Paired repetitions are required")
        for before, after in zip(rows[policies[0]], rows[policy], strict=True):
            if (before["output_tokens"], before["training_steps"]) != (
                after["output_tokens"],
                after["training_steps"],
            ):
                raise ValueError("Work differs across policies")
    colors = ["#3867A8", "#16836B"]
    positions = list(range(len(policies)))
    labels = ["Static", "LoopWeave"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, title, ylabel, metric in [
        (
            axes[0],
            "Allocated GPU time per training step",
            "GPU-seconds (lower is better)",
            lambda r: r["gpu_seconds"] / r["training_steps"],
        ),
        (
            axes[1],
            "Mean rollout request latency",
            "Seconds (lower is better)",
            lambda r: r["sample_latency_mean_s"],
        ),
        (
            axes[2],
            "Observed peak GPU memory",
            "Total GiB across allocated GPUs",
            lambda r: max(
                sum(g["memory_bytes"] for g in s["gpu"]) / 2**30 for s in r["gpu_samples"]
            ),
        ),
    ]:
        values = [[metric(r) for r in rows[p]] for p in policies]
        means = [statistics.mean(v) for v in values]
        ax.bar(positions, means, color=colors, alpha=0.25, width=0.55, zorder=2)
        for repetition in zip(*values, strict=True):
            ax.plot(positions, repetition, color="#777777", alpha=0.5, linewidth=1, zorder=3)
            ax.scatter(positions, repetition, color=colors, s=35, zorder=4)
        for x, mean in enumerate(means):
            ax.text(
                x,
                max(values[x]) + max(means) * 0.05,
                f"{mean:.2f}",
                ha="center",
                fontsize=11,
                weight="bold",
            )
        ax.set_title(title, loc="left", fontsize=11, pad=26)
        ax.text(
            0,
            1.01,
            f"{100 * (means[-1] / means[0] - 1):+.1f}% vs static",
            transform=ax.transAxes,
            color="#16836B" if means[-1] < means[0] else "#9A4E35",
            fontsize=10,
        )
        ax.set_xticks(positions, labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(map(max, values)) * 1.2)
        ax.set_xlim(-0.65, len(policies) - 0.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2, zorder=0)
    gpu_count = rows[policies[0]][0].get("allocated_gpus", 2)
    gpu_rows = list(csv.reader(io.StringIO(data["environment"]["gpus"])))
    gpu_names = sorted({row[0].strip() for row in gpu_rows[1:] if row})
    fig.suptitle(
        f"TuFT: same {gpu_count} {', '.join(gpu_names)} GPUs, identical work",
        x=0.06,
        ha="left",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.06,
        0.03,
        f"{data['environment']['model_name']} · "
        f"{rows[policies[0]][0]['workload']['tenants']} tenants · "
        f"{rows[policies[0]][0]['workload']['output_tokens_per_sample']}-token rollouts · "
        f"paired runs: {len(rows[policies[0]])} · "
        "bars show means; points show each run\n"
        "Timed workload; server initialization and correctness checks excluded.",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.24, wspace=0.35)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
