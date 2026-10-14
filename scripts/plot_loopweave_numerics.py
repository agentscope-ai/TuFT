"""Plot recorded fixed-history numerical controls, without changing their scores."""

import argparse
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text())["models"]
    fig, ax = plt.subplots(figsize=(10, 5))
    series = [
        ("Normal: repeat", "#3867A8", [r["normal_repeat_error"] for r in rows]),
        ("Normal: cold-cache switch", "#B87820", [r["normal_cold_switch_error"] for r in rows]),
        ("Single split: repeat", "#16836B", [r["single_split_repeat_error"] for r in rows]),
        (
            "Single split: 3 cold-cache switches",
            "#5A438B",
            [max(r["single_split_cold_switch_errors"]) for r in rows],
        ),
    ]
    maximum = max(max(values) for _, _, values in series)
    for index, (label, color, values) in enumerate(series):
        positions = [i + (index - 1.5) * 0.19 for i in range(len(rows))]
        ax.bar(positions, values, width=0.17, color=color, label=label, zorder=2)
        for x, value in zip(positions, values, strict=True):
            ax.text(x, value + max(maximum, 0.001) * 0.025, f"{value:.3f}", ha="center", size=10)
    ax.set_xticks(range(len(rows)), [r["model"] for r in rows])
    ax.set_ylim(0, max(maximum, 0.001) * 1.3)
    ax.set_ylabel("Maximum absolute prompt log-probability change")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    fig.suptitle("Same-engine numerical controls · HF · two B200 GPUs", x=0.1, ha="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.19), ncol=2, frameon=False, fontsize=9)
    fig.text(
        0.1,
        0.035,
        "Separate control runs, identical histories within each comparison. "
        "All single-split differences are exactly zero.\n"
        "Single-split checks also repeat requests on an ordinary sampler. "
        "These are not throughput or batch-invariance measurements.",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.75)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
