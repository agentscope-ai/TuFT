"""
Plot 2x2 multi-panel figure comparing Tinker (baseline) vs TuFT across training rounds.

Panels:
  Top-left:     mean_abs_diff vs round
  Top-right:    max_cum_diff vs round
  Bottom-left:  std_is_weight vs round
  Bottom-right: p_out_clip_02 vs round
"""

import json
import sys

import matplotlib.pyplot as plt


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_comparison.py <tinker_results.json> <tuft_results.json>")
        sys.exit(1)

    tinker_path = sys.argv[1]
    tuft_path = sys.argv[2]

    tinker_data = load_results(tinker_path)
    tuft_data = load_results(tuft_path)

    tinker_rounds = [d["round"] for d in tinker_data]
    tuft_rounds = [d["round"] for d in tuft_data]

    metrics = [
        ("mean_abs_diff", "Mean Absolute Diff", "Mean |Sampling − Training| Logprob Diff"),
        ("max_cum_diff", "Max Cumulative Diff", "Max |Cumulative Diff| across Sequences"),
        ("std_is_weight", "Std Dev of IS Weight", "Std Dev of IS Weight"),
        ("p_out_clip_02", "P(|IS weight − 1| > 0.2)", "Proportion of Sequences Outside Clip ε=0.2"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (key, ylabel, title) in zip(axes.flat, metrics, strict=False):
        tinker_vals = [d[key] for d in tinker_data]
        tuft_vals = [d[key] for d in tuft_data]

        ax.plot(
            tinker_rounds,
            tinker_vals,
            "o-",
            color="#1f77b4",
            linewidth=2,
            markersize=4,
            label="Tinker (baseline)",
        )
        ax.plot(
            tuft_rounds,
            tuft_vals,
            "s-",
            color="#d62728",
            linewidth=2,
            markersize=4,
            label="TuFT (ours)",
        )

        ax.set_xlabel("Training Round")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Training vs Sampling Logprob Mismatch: Tinker vs TuFT",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = "mismatch_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved to {out_path}")


if __name__ == "__main__":
    main()
