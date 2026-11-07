"""Report the saved before/after scheduler pair; never launches GPU work."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from report_coding_comparison import summarize


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def detail(arm):
    summary = summarize(arm)
    # GPU monitoring continues through final integrity checks; exclude that tail.
    samples = [s for s in arm["gpu"] if s["elapsed_s"] <= arm["elapsed_s"]]
    summary["gpu_utilization_percent"] = [
        float(np.mean([s["gpu"][i]["sm_percent"] for s in samples])) for i in range(2)
    ]
    before, after = arm["scheduler_before"], arm["scheduler_after"]
    trace = [
        event for event in after["decision_trace"] if event["elapsed_s"] >= before["elapsed_s"]
    ]
    operations = [event for event in trace if event["event"] == "exclusive"]
    optimizer = [event for event in operations if event["kind"] == "optimizer"]
    assert len(optimizer) == summary["updates"], "Trace lost optimizer events"
    summary["optimizer"] = dict(
        count=len(optimizer),
        queue_p50_s=float(np.percentile([e["queue_s"] for e in optimizer], 50)),
        queue_p95_s=float(np.percentile([e["queue_s"] for e in optimizer], 95)),
        queue_max_s=max(e["queue_s"] for e in optimizer),
        queue_sum_s=sum(e["queue_s"] for e in optimizer),
        service_sum_s=sum(e["service_s"] for e in optimizer),
    )
    summary["drain_events_s"] = [e["drain_s"] for e in operations if e["drain_s"] > 0.01]
    summary["max_drain_s"] = max(summary["drain_events_s"], default=0)
    for key in ("borrow_duration_rejections", "training_priority_overtakes", "failed"):
        summary["scheduler"][key] = after.get(key, 0) - before.get(key, 0)
    summary["allocated_gpu_s"] = arm["elapsed_s"] * 2
    summary["updates_per_gpu_hour"] = summary["updates"] * 3600 / summary["allocated_gpu_s"]
    summary["arm_wall_s"] = arm.get("arm_wall_s")
    summary["shared"] = arm["shared"]
    first_updates = [u["finished_s"] for u in arm["updates"] if u["step"] == 0]
    summary["first_update_finished_s"] = min(first_updates)
    summary["first_wave_updates_finished_s"] = max(first_updates)
    summary["dedicated_gpu_idle_sample_fraction"] = float(
        np.mean([s["gpu"][1]["sm_percent"] == 0 for s in samples])
    )
    backward = [e for e in operations if e["kind"] == "backward"]
    summary["backward"] = dict(
        first_service_s=backward[0]["service_s"],
        median_service_s=float(np.median([e["service_s"] for e in backward])),
        service_sum_s=sum(e["service_s"] for e in backward),
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    arms = json.loads(args.result.read_text())
    assert (
        arms["before"]["initial_adapter_fingerprints"]
        == arms["after"]["initial_adapter_fingerprints"]
    )
    summaries = {key: detail(arm) for key, arm in arms.items()}
    output = args.result.parent
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    labels = ["Before fixes", "After fixes"]
    values = [summaries[key] for key in ("before", "after")]
    panels = [
        ([v["elapsed_s"] for v in values], "Workload completion", "Seconds"),
        ([v["updates_per_gpu_hour"] for v in values], "Resource efficiency", "Updates / GPU-hour"),
        (
            [v["scheduler"]["training_drain_wait_s"] for v in values],
            "Training waits for sampling to drain",
            "Seconds",
        ),
        ([v["optimizer"]["queue_p95_s"] for v in values], "Optimizer queue wait (p95)", "Seconds"),
        ([v["request_p95_s"] for v in values], "Sampling latency (p95)", "Seconds"),
        (
            [v["scheduler"]["replica_0_completed"] for v in values],
            "Requests served on shared GPU",
            "Requests",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (data, title, unit) in zip(axes.flat, panels, strict=True):
        bars = ax.bar(labels, data, color=["#64748b", "#0f766e"])
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set(title=title, ylabel=unit, ylim=(0, max(data) * 1.25 + 0.01))
    fig.suptitle("Qwen3.5-4B · 16 LoRA tenants · 2 H100s · shared weights in both arms")
    fig.text(
        0.5,
        0.015,
        "One ordered comparison · matching initial adapters · 384 samples and 32 updates per arm",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    for extension in ("png", "svg"):
        fig.savefig(output / f"scheduler-comparison.{extension}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=True)
    for ax, key, label in zip(axes, ("before", "after"), labels, strict=True):
        samples = [s for s in arms[key]["gpu"] if s["elapsed_s"] <= arms[key]["elapsed_s"]]
        for index, name in enumerate(("Shared trainer/sampler", "Dedicated sampler")):
            ax.plot(
                [s["elapsed_s"] for s in samples],
                [s["gpu"][index]["sm_percent"] for s in samples],
                label=name,
                linewidth=0.7,
                alpha=0.7,
            )
        ax.set(title=label, ylabel="NVML utilization (%)", xlabel="Elapsed seconds", ylim=(0, 105))
        ax.legend(loc="upper right")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"gpu-timeline.{extension}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for key, label, color in zip(("before", "after"), labels, ("#64748b", "#0f766e"), strict=True):
        arm = arms[key]
        completions = sorted(r["start_s"] + r["duration_s"] for r in arm["requests"])
        updates = sorted(u["finished_s"] for u in arm["updates"])
        for ax, times in zip(axes, (completions, updates), strict=True):
            ax.step([0, *times], range(len(times) + 1), where="post", label=label, color=color)
            ax.legend(loc="upper left")
    axes[0].set(ylabel="Completed sampling requests", title="Progress through the same workload")
    axes[1].set(ylabel="Completed training updates", xlabel="Elapsed seconds")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"workload-progress.{extension}", dpi=160)
    plt.close(fig)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
