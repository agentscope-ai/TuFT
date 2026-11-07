"""Inspect one short shared-weight calibration before a full comparison."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from report_coding_comparison import summarize


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    arm = json.loads(args.result.read_text())["shared_weight"]
    summary = summarize(arm)
    rows = arm["engine_metrics"]
    assert rows and all("vllm:num_requests_running" in r for r in rows[-1]["replicas"])
    peaks = [
        max(row["replicas"][i].get("vllm:num_requests_running", 0) for row in rows)
        for i in range(2)
    ]
    adapters = [
        max(row["replicas"][i].get("running_lora_adapters", 0) for row in rows) for i in range(2)
    ]
    preemptions = [
        rows[-1]["replicas"][i].get("vllm:num_preemptions_total", 0)
        - rows[0]["replicas"][i].get("vllm:num_preemptions_total", 0)
        for i in range(2)
    ]
    summary.update(
        peak_running_sequences=peaks, peak_running_adapters=adapters, preemptions=preemptions
    )
    summary["checks"] = dict(
        dedicated_batching=peaks[1] >= 24 and adapters[1] >= 12,
        bounded_borrowing=0 < peaks[0] <= 4,
        shared_batching=peaks[0] >= 2,
        memory_headroom=max(summary["peak_gpu_memory_gib"]) < 72,
        output_truncation=summary["output_limit_fraction"] < 0.05,
        no_preemptions=max(preemptions) == 0,
        no_queue_timeouts=summary["scheduler"]["queue_timeouts"] == 0,
    )
    summary["ready_for_comparison"] = all(summary["checks"].values())
    folder = args.result.parent
    (folder / "calibration-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i, name in enumerate(("Shared GPU", "Dedicated sampler")):
        axes[0, 0].plot(
            [r["elapsed_s"] for r in rows],
            [r["replicas"][i].get("vllm:num_requests_running", 0) for r in rows],
            label=name,
        )
        axes[0, 1].plot(
            [r["elapsed_s"] for r in arm["gpu"]],
            [r["gpu"][i]["sm_percent"] for r in arm["gpu"]],
            label=name,
            alpha=0.65,
        )
        axes[1, 0].plot(
            [r["elapsed_s"] for r in arm["gpu"]],
            [r["gpu"][i]["memory_bytes"] / 1024**3 for r in arm["gpu"]],
            label=name,
        )
    axes[0, 0].set(title="Actual running vLLM sequences", ylabel="Sequences", xlabel="Seconds")
    axes[0, 1].set(title="GPU utilization", ylabel="NVML utilization (%)", xlabel="Seconds")
    axes[1, 0].set(title="GPU memory", ylabel="GiB", xlabel="Seconds")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.legend()
    passes = [
        np.mean([e["stage_rewards"][t] == 1 for e in arm["episodes"]]) * 100 for t in range(4)
    ]
    bars = axes[1, 1].bar(range(1, 5), passes, color="#147d92")
    axes[1, 1].bar_label(bars, fmt="%.1f")
    axes[1, 1].set(
        title="Responses passing all tests",
        xlabel="Turn",
        ylabel="Percent",
        ylim=(0, 110),
        xticks=range(1, 5),
    )
    fig.suptitle("Coding-agent calibration · 16 adapters · 32 dedicated / 4 borrowed requests")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(folder / f"calibration.{ext}", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
