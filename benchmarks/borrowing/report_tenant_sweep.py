"""Summarize the SSH tenant sweep from saved telemetry only."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def summarize(data):
    elapsed = data["elapsed_s"]
    before, after = data["scheduler_before"], data["scheduler_after"]
    origin = before["elapsed_s"]
    samples = [s for s in data["scheduling_samples"] if s["elapsed_s"] <= elapsed]
    durations = np.diff([s["elapsed_s"] for s in samples] + [elapsed])

    def seconds(predicate):
        return float(sum(dt for s, dt in zip(samples, durations, strict=True) if predicate(s)))

    events = [e for e in after["decision_trace"] if e["elapsed_s"] >= origin]
    shared = after.get("replica_0_completed", 0) - before.get("replica_0_completed", 0)
    probes = after.get("borrow_probe_requests", 0) - before.get("borrow_probe_requests", 0)
    first_update = min(u["finished_s"] for u in data["updates"])
    gpu_samples = [s for s in data["gpu"] if s["elapsed_s"] <= elapsed]
    gpu_durations = np.diff([s["elapsed_s"] for s in gpu_samples] + [elapsed])
    gpu_mean = [
        float(
            sum(
                dt * s["gpu"][i]["sm_percent"]
                for s, dt in zip(gpu_samples, gpu_durations, strict=True)
            )
            / elapsed
        )
        for i in range(2)
    ]
    return dict(
        elapsed_s=elapsed,
        requests=len(data["requests"]),
        trainer_gpu_utilization_percent=gpu_mean[0],
        sampler_gpu_utilization_percent=gpu_mean[1],
        no_ready_training_percent=100
        * seconds(lambda s: not s["training_active"] and not s["training_pending"])
        / elapsed,
        training_execution_s=after.get("training_execution_s", 0)
        - before.get("training_execution_s", 0),
        successful_trajectories=sum(e["success"] for e in data["episodes"]),
        trajectories=len(data["episodes"]),
        output_limited_requests=sum(r["hit_output_limit"] for r in data["requests"]),
        updates=len(data["updates"]),
        updates_per_gpu_hour=len(data["updates"]) * 3600 / (elapsed * 2),
        sampling_p95_s=float(np.percentile([r["duration_s"] for r in data["requests"]], 95)),
        training_update_p95_s=float(np.percentile([u["duration_s"] for u in data["updates"]], 95)),
        training_executing_s=seconds(lambda s: s["training_active"]),
        no_ready_training_s=seconds(
            lambda s: not s["training_active"] and not s["training_pending"]
        ),
        no_ready_training_with_inflight_sampling_s=seconds(
            lambda s: not s["training_active"]
            and not s["training_pending"]
            and sum(s["active"]) > 0
        ),
        no_ready_training_with_sampling_backlog_s=seconds(
            lambda s: not s["training_active"] and not s["training_pending"] and s["queued"] > 0
        ),
        ready_training_not_executing_s=seconds(
            lambda s: not s["training_active"] and s["training_pending"] > 0
        ),
        shared_sampling_active_s=seconds(lambda s: s["active"][0] > 0),
        shared_requests=shared,
        cold_probe_requests=probes,
        learned_requests=shared - probes,
        shared_admissions_after_first_update=sum(
            e["event"] == "admit"
            and e.get("replica") == 0
            and e["elapsed_s"] - origin > first_update
            for e in events
        ),
        training_drain_s=after.get("training_drain_wait_s", 0)
        - before.get("training_drain_wait_s", 0),
        output_tokens=sum(r["output_tokens"] for r in data["requests"]),
        per_tenant_updates={
            str(i): sum(u["tenant"] == i for u in data["updates"]) for i in range(data["tenants"])
        },
        frozen_base_unchanged=data["frozen_base_unchanged"],
        copied_base_parameters=data["shared"]["copied_parameters"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summaries = {}
    for tenants in (1, 2, 4, 8):
        for arm in ("off", "on"):
            name = f"tenants-{tenants}-{arm}"
            p = args.directory / f"{name}.json"
            if p.exists():
                summaries[name] = summarize(json.loads(p.read_text()))
    (args.directory / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = [
        ("updates_per_gpu_hour", "Updates / GPU-hour"),
        ("no_ready_training_with_sampling_backlog_s", "No ready training + sampling backlog (s)"),
        ("training_drain_s", "Training wait for borrowed sampling (s)"),
        ("sampling_p95_s", "Sampling p95 (s)"),
        ("training_update_p95_s", "Training-update p95 (s)"),
        ("learned_requests", "Learned borrowed requests"),
    ]
    for ax, (metric, title) in zip(axes.flat, panels, strict=True):
        for arm in ("off", "on"):
            xs = [n for n in (1, 2, 4, 8) if f"tenants-{n}-{arm}" in summaries]
            ax.plot(
                xs, [summaries[f"tenants-{n}-{arm}"][metric] for n in xs], marker="o", label=arm
            )
        ax.set(title=title, xlabel="Tenants", xticks=[1, 2, 4, 8])
        ax.set_ylim(bottom=0, top=max(1.0, ax.get_ylim()[1]))
        ax.legend(title="Borrowing")
    fig.suptitle("Qwen3.5-4B · 2 GPUs · 8 trajectories and 2 updates per tenant")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(args.directory / f"tenant-sweep.{ext}", dpi=160)

    utilization, util_axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(
        util_axes,
        (
            "no_ready_training_percent",
            "trainer_gpu_utilization_percent",
            "sampler_gpu_utilization_percent",
        ),
        ("No ready training work", "Trainer GPU utilization", "Dedicated sampler GPU utilization"),
        strict=True,
    ):
        for arm in ("off", "on"):
            xs = [n for n in (1, 2, 4, 8) if f"tenants-{n}-{arm}" in summaries]
            ax.plot(
                xs, [summaries[f"tenants-{n}-{arm}"][metric] for n in xs], marker="o", label=arm
            )
        ax.set(
            title=title,
            xlabel="Tenants",
            ylabel="Percent of measured time",
            xticks=[1, 2, 4, 8],
            ylim=(0, 100),
        )
        ax.legend(title="Borrowing")
    utilization.suptitle("Time-weighted telemetry · Qwen3.5-4B · 2 GPUs")
    utilization.tight_layout()
    for ext in ("png", "svg"):
        utilization.savefig(args.directory / f"tenant-utilization.{ext}", dpi=160)
