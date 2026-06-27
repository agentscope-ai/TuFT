"""Plot Test 2: Batch Size & Position Effect — simple line charts."""

import json
import sys

import matplotlib.pyplot as plt


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "./results/test2_batch_position.json"
    with open(path) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # --- Batch size effect ---
    exp_b = data["exp_b_batch_size_effect"]
    bs_keys = sorted(exp_b.keys(), key=lambda x: int(x))
    batch_sizes = [int(k) for k in bs_keys]
    means_pt = [exp_b[k]["mean_diff"] for k in bs_keys]

    ax1.plot(range(len(batch_sizes)), means_pt, "o-", color="#4C72B0", linewidth=2, markersize=7)
    ax1.set_xticks(range(len(batch_sizes)))
    ax1.set_xticklabels(batch_sizes)
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Mean |diff| per token")
    ax1.set_title("Batch Size Effect")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
    ax1.grid(alpha=0.3)

    # --- Position sweep ---
    exp_c = data["exp_c_position_sweep"]
    pos_keys = sorted(exp_c.keys(), key=lambda x: int(x.split("_")[1]))
    positions = [int(k.split("_")[1]) for k in pos_keys]
    means_pos = [exp_c[k]["mean_diff"] for k in pos_keys]

    ax2.plot(positions, means_pos, "s-", color="#55A868", linewidth=2, markersize=7)
    ax2.set_xticks(positions)
    ax2.set_xlabel("Position in Batch")
    ax2.set_ylabel("Mean |diff| per token")
    ax2.set_title("Position Effect (BS=8)")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
    ax2.grid(alpha=0.3)

    fig.suptitle("Test 2: Batch Size & Position Effect", fontweight="bold")
    plt.tight_layout()
    out = path.replace(".json", ".png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
