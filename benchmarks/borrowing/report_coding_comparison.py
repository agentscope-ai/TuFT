"""Summarize a saved coding-agent comparison; no cloud calls or experiment reruns."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def summarize(arm) -> dict[str, Any]:
    requests = arm["requests"]
    expected_updates = arm["tenants"] * arm["rounds"]
    assert len(arm["updates"]) == expected_updates, "Incomplete training updates"
    assert len(arm["episodes"]) == expected_updates * arm["trajectories"], "Incomplete trajectories"
    assert len(requests) == sum(e.get("turns", 4) for e in arm["episodes"]), "Missing responses"
    assert arm["frozen_base_unchanged"], "Frozen base verification failed"
    if arm["policy"] == "shared_weight":
        assert arm["shared"]["verified_alias_bytes"] > 0
        assert arm["shared"]["copied_parameters"] == 0
    latencies = [r["duration_s"] for r in requests]
    outputs = [r["output_tokens"] for r in requests]
    stages = [s for e in arm["episodes"] for s in e["stage_rewards"]]
    samples = arm["gpu"]
    rollout_end = max(r["start_s"] + r["duration_s"] for r in requests)
    rollout_samples = [s for s in samples if s["elapsed_s"] <= rollout_end]
    training_tokens = sum(r["prompt_tokens"] + r["output_tokens"] - 1 for r in requests)
    before, after = arm["scheduler_before"], arm["scheduler_after"]
    scheduler_delta = {
        key: after.get(key, 0) - before.get(key, 0)
        for key in (
            "admitted",
            "completed",
            "queue_wait_s",
            "queue_timeouts",
            "training_queue_wait_s",
            "training_drain_wait_s",
            "training_execution_s",
            "switch_to_training_s",
            "switch_to_sampling_s",
            "borrow_probes",
            "borrow_probe_requests",
            "borrow_cold_rejections",
            "borrow_cost_rejections",
            "replica_0_live_rate_updates",
            "replica_1_live_rate_updates",
            "replica_0_completed",
            "replica_1_completed",
        )
    }
    return dict(
        scheduler=scheduler_delta,
        shared_request_fraction=(
            scheduler_delta["replica_0_completed"] / len(requests)
            if arm["policy"] == "shared_weight"
            else 0.0
        ),
        mean_admission_wait_s=scheduler_delta["queue_wait_s"] / scheduler_delta["admitted"],
        update_p95_s=float(np.percentile([u["duration_s"] for u in arm["updates"]], 95)),
        peak_gpu_memory_gib=[
            max(s["gpu"][i]["memory_bytes"] for s in samples) / 1024**3
            for i in range(len(samples[0]["gpu"]))
        ],
        elapsed_s=arm["elapsed_s"],
        requests=len(requests),
        updates=len(arm["updates"]),
        requests_per_s=len(requests) / arm["elapsed_s"],
        output_tokens=sum(outputs),
        output_tokens_per_s=sum(outputs) / arm["elapsed_s"],
        mean_output_tokens=float(np.mean(outputs)),
        output_limit_fraction=float(np.mean([r["hit_output_limit"] for r in requests])),
        prompt_range=[
            min(r["prompt_tokens"] for r in requests),
            max(r["prompt_tokens"] for r in requests),
        ],
        request_p50_s=float(np.percentile(latencies, 50)),
        request_p95_s=float(np.percentile(latencies, 95)),
        mean_stage_test_fraction=float(np.mean(stages)),
        final_stage_all_tests_pass_fraction=float(
            np.mean([e["reward"] == 1 for e in arm["episodes"]])
        ),
        gpu_utilization_percent=[
            float(np.mean([s["gpu"][i]["sm_percent"] for s in samples]))
            for i in range(len(samples[0]["gpu"]))
        ],
        rollout_end_s=rollout_end,
        rollout_gpu_utilization_percent=[
            float(np.mean([s["gpu"][i]["sm_percent"] for s in rollout_samples]))
            for i in range(len(samples[0]["gpu"]))
        ],
        training_input_tokens=training_tokens,
        training_tokens_per_service_s=training_tokens / scheduler_delta["training_execution_s"],
        frozen_base_unchanged=arm["frozen_base_unchanged"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    arms = json.loads(args.result.read_text())
    summary = {name: summarize(arm) for name, arm in arms.items()}
    output = args.result.parent
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    labels = ["Static", "Shared weight"]
    # Older saved comparisons used the retired balanced policy.
    baseline = "static" if "static" in summary else "balanced"
    values = [summary[k] for k in (baseline, "shared_weight")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, key, title, unit in (
        (axes[0, 0], "elapsed_s", "Workload completion", "Seconds"),
        (axes[0, 2], "output_tokens_per_s", "Generated-token throughput", "Tokens / second"),
        (axes[1, 0], "request_p95_s", "Sampling latency (p95)", "Seconds"),
        (axes[1, 1], "update_p95_s", "Training update latency (p95)", "Seconds"),
        (
            axes[1, 2],
            "training_tokens_per_service_s",
            "Training service throughput",
            "Input tokens / second",
        ),
    ):
        bars = ax.bar(labels, [v[key] for v in values], color=["#8597a7", "#147d92"])
        ax.bar_label(bars, fmt="%.1f")
        ax.set(title=title, ylabel=unit)
        ax.margins(y=0.18)
    x = np.arange(2)
    for i, (label, value) in enumerate(zip(labels, values, strict=True)):
        bars = axes[0, 1].bar(
            x + (i - 0.5) * 0.35,
            value["gpu_utilization_percent"],
            width=0.35,
            label=label,
            color=["#8597a7", "#147d92"][i],
        )
        axes[0, 1].bar_label(bars, fmt="%.1f")
    axes[0, 1].set(
        title="Mean GPU utilization",
        ylabel="NVML utilization (%)",
        xticks=x,
        xticklabels=["GPU 0", "GPU 1"],
        ylim=(0, 110),
    )
    axes[0, 1].legend()
    provenance = json.loads((output / "provenance.json").read_text())
    model = provenance["model"].split("/")[-1]
    fig.suptitle(f"{model} · {provenance['tenants']} LoRA tenants · 2 H100s · coding tasks")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"comparison.{extension}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharey=True)
    for ax, key, label in zip(axes, (baseline, "shared_weight"), labels, strict=True):
        samples = arms[key]["gpu"]
        for i in range(2):
            ax.plot(
                [s["elapsed_s"] for s in samples],
                [s["gpu"][i]["sm_percent"] for s in samples],
                label=f"GPU {i}",
                alpha=0.7,
                linewidth=0.6,
            )
        ax.set(title=label, xlabel="Elapsed seconds", ylabel="NVML utilization (%)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "gpu-timeline.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
