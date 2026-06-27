"""Plot Test 1: Adapter Independence — simple bar chart."""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "./results/test1_adapter_independence.json"
    with open(path) as f:
        data = json.load(f)

    sc = data["statistical_comparison"]
    diffs = {
        "Same": sc["diffs_same_adapter"],
        "Different": sc["diffs_diff_adapter"],
        "Mixed": sc["diffs_mixed_adapter"],
    }
    accs = {
        "Same": sc.get("acc_same_adapter", [x * 32 for x in sc["diffs_same_adapter"]]),
        "Different": sc.get("acc_diff_adapter", [x * 32 for x in sc["diffs_diff_adapter"]]),
        "Mixed": sc.get("acc_mixed_adapter", [x * 32 for x in sc["diffs_mixed_adapter"]]),
    }

    labels = list(diffs.keys())
    means_pt = [np.mean(diffs[k]) for k in labels]
    stds_pt = [np.std(diffs[k]) for k in labels]
    means_acc = [np.mean(accs[k]) for k in labels]
    stds_acc = [np.std(accs[k]) for k in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(labels, means_pt, yerr=stds_pt, capsize=5, color=["#4C72B0", "#DD8452", "#55A868"])
    ax1.set_ylabel("Mean |diff| per token")
    ax1.set_title("Per-Token Diff")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))

    ax2.bar(labels, means_acc, yerr=stds_acc, capsize=5, color=["#4C72B0", "#DD8452", "#55A868"])
    ax2.set_ylabel("Accumulated |diff| per seq")
    ax2.set_title("Accumulated Diff")

    fig.suptitle("Test 1: Adapter Independence (vs Solo Reference)", fontweight="bold")
    plt.tight_layout()
    out = path.replace(".json", ".png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
