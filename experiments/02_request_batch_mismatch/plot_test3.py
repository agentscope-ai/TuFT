"""Plot Test 3: Framework Mismatch — per-prompt bar chart."""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "./results/test3_framework_mismatch.json"
    with open(path) as f:
        data = json.load(f)

    per_prompt = data.get("per_prompt", {})
    cross = data["cross_framework"]
    keys = sorted(cross.keys())

    if per_prompt:
        # Select vLLM_BS=1 combinations to show PyTorch BS effect per prompt
        selected = [k for k in keys if k.startswith("vllm_bs1_")]
        if not selected:
            selected = keys[:4]

        n_prompts = len(per_prompt[selected[0]])
        x = np.arange(n_prompts)
        width = 0.8 / len(selected)
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        for i, key in enumerate(selected):
            pts = per_prompt[key]
            vals_pt = [p["mean_abs_diff"] for p in pts]
            vals_acc = [p["accumulated"] for p in pts]
            label = key.replace("vllm_bs", "vLLM=").replace("_vs_pytorch_bs", ",PT=")
            ax1.bar(x + i * width - 0.4 + width / 2, vals_pt, width, label=label, color=colors[i])
            ax2.bar(x + i * width - 0.4 + width / 2, vals_acc, width, label=label, color=colors[i])

        ax1.set_xticks(x)
        ax1.set_xticklabels([f"P{i}" for i in range(n_prompts)])
        ax1.set_xlabel("Prompt")
        ax1.set_ylabel("Mean |diff| per token")
        ax1.set_title("Per-Token Diff")
        ax1.legend(fontsize=7)
        ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))

        ax2.set_xticks(x)
        ax2.set_xticklabels([f"P{i}" for i in range(n_prompts)])
        ax2.set_xlabel("Prompt")
        ax2.set_ylabel("Accumulated |diff|")
        ax2.set_title("Accumulated Diff")
        ax2.legend(fontsize=7)

    else:
        # Fallback: aggregate bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        labels = [k.replace("vllm_bs", "v").replace("_vs_pytorch_bs", "/p") for k in keys]
        vals_pt = [cross[k]["avg_mean_abs_diff"] for k in keys]
        vals_acc = [cross[k]["avg_accumulated"] for k in keys]

        ax1.bar(range(len(keys)), vals_pt, color="#4C72B0")
        ax1.set_xticks(range(len(keys)))
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax1.set_ylabel("Mean |diff| per token")
        ax1.set_title("Per-Token Diff")

        ax2.bar(range(len(keys)), vals_acc, color="#DD8452")
        ax2.set_xticks(range(len(keys)))
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax2.set_ylabel("Accumulated |diff|")
        ax2.set_title("Accumulated Diff")

    fig.suptitle(
        "Test 3: Framework Mismatch (vLLM Generation vs PyTorch Forward)", fontweight="bold"
    )
    plt.tight_layout()
    out = path.replace(".json", ".png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
