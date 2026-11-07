"""Plot existing static-allocation telemetry; never launches GPU work."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def time_bins(samples, elapsed, width=10):
    edges = np.arange(0, elapsed + width, width)
    times = np.array([s["elapsed_s"] for s in samples])
    ends = np.r_[times[1:], elapsed]
    values = np.array([[g["sm_percent"] for g in s["gpu"]] for s in samples])
    means = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        weights = np.maximum(0, np.minimum(ends, min(hi, elapsed)) - np.maximum(times, lo))
        means.append(np.average(values, axis=0, weights=weights) if weights.sum() else [np.nan] * 2)
    return (edges[:-1] + np.minimum(edges[1:], elapsed)) / 2, np.array(means)


def observation_summary(data):
    lo, hi = data["independent_clients"]["measurement_window_s"]
    samples = data["gpu"]
    times = np.array([s["elapsed_s"] for s in samples])
    ends = np.r_[times[1:], data["elapsed_s"]]
    weights = np.maximum(0, np.minimum(ends, hi) - np.maximum(times, lo))
    gpu = np.array([[g["sm_percent"] for g in s["gpu"]] for s in samples])
    if abs(weights.sum() - (hi - lo)) > 0.1:
        raise ValueError("Incomplete observation-window GPU telemetry")
    utilization = np.average(gpu, axis=0, weights=weights)
    rows = data["scheduling_samples"]
    times = np.array([r["elapsed_s"] for r in rows])
    ends = np.r_[times[1:], data["elapsed_s"]]
    durations = np.maximum(0, np.minimum(ends, hi) - np.maximum(times, lo))
    requests = [r for r in data["requests"] if lo <= r["start_s"] < hi]
    updates = [u for u in data["updates"] if lo <= u["finished_s"] - u["duration_s"] < hi]
    completions = [u for u in data["updates"] if lo <= u["finished_s"] < hi]

    def percentile(items, q):
        return float(np.percentile([r["duration_s"] for r in items], q)) if items else None

    return dict(
        window_s=[lo, hi],
        trainer_gpu_utilization_percent=float(utilization[0]),
        sampler_gpu_utilization_percent=float(utilization[1]),
        no_ready_training_percent=100
        * sum(
            dt
            for r, dt in zip(rows, durations, strict=True)
            if not r["training_active"] and not r["training_pending"]
        )
        / (hi - lo),
        average_sampling_queue=float(
            sum(dt * r["queued"] for r, dt in zip(rows, durations, strict=True)) / (hi - lo)
        ),
        submitted_sampling_requests=len(requests),
        submitted_training_updates=len(updates),
        completed_training_updates=len(completions),
        sampling_p50_s=percentile(requests, 50),
        sampling_p95_s=percentile(requests, 95),
        training_p50_s=percentile(updates, 50),
        training_p95_s=percentile(updates, 95),
        per_tenant_completed_updates={
            str(i): sum(u["tenant"] == i for u in completions) for i in range(data["tenants"])
        },
    )


def main(directory, output):
    tenant_counts = (1, 2, 4, 8)
    datasets = [
        json.loads((directory / f"tenants-{n}-off.json").read_text()) for n in tenant_counts
    ]
    time_limit = max(d["elapsed_s"] for d in datasets)
    sampling_limit = 1.1 * max(r["duration_s"] for d in datasets for r in d["requests"])
    training_limit = 1.1 * max(u["duration_s"] for d in datasets for u in d["updates"])
    fig, axes = plt.subplots(4, 3, figsize=(17, 13), squeeze=False, sharey="col", sharex=True)
    # Match panel height to its y span, so each tenant has the same physical height.
    phase, phase_axes = plt.subplots(
        4,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [n + 0.2 for n in tenant_counts]},
    )
    summary = []
    independent = False
    for row, (tenants, d) in enumerate(zip(tenant_counts, datasets, strict=True)):
        elapsed = d["elapsed_s"]
        independent = d.get("independent_clients")
        x, y = time_bins(d["gpu"], elapsed)
        ax = axes[row, 0]
        ax.plot(x, y[:, 0], label="Trainer GPU", color="tab:blue")
        ax.plot(x, y[:, 1], label="Dedicated sampler GPU", color="tab:orange")
        ax.set(ylabel=f"{tenants} tenant(s)\nGPU utilization (%)", ylim=(0, 100))
        if row == 0:
            ax.legend(fontsize=8)
            ax.set_title("GPU utilization · time-weighted 10 s bins")
        ax = axes[row, 1]
        requests = d["requests"]
        for tenant in range(tenants):
            rows = [r for r in requests if r["tenant"] == tenant]
            ax.scatter(
                [r["start_s"] for r in rows],
                [r["duration_s"] for r in rows],
                s=6,
                alpha=0.4,
                color=f"C{tenant}",
            )
        centers, med, p95 = [], [], []
        for lo in np.arange(0, elapsed, 30):
            values = [r["duration_s"] for r in requests if lo <= r["start_s"] < lo + 30]
            centers.append(lo + 15)
            med.append(np.median(values) if len(values) >= 10 else np.nan)
            p95.append(np.percentile(values, 95) if len(values) >= 10 else np.nan)
        ax.plot(centers, med, color="black", marker=".", label="30 s bin median")
        ax.plot(
            centers, p95, color="crimson", marker=".", linestyle="--", label="30 s bin p95 (n ≥ 10)"
        )
        ax.set(ylabel="Sampling latency (s)", ylim=(0, sampling_limit))
        if row == 0:
            ax.set_title("Sampling latency vs request submission time")
            ax.legend(fontsize=8)
        ax = axes[row, 2]
        for tenant in range(tenants):
            updates = [u for u in d["updates"] if u["tenant"] == tenant]
            ax.scatter(
                [u["finished_s"] - u["duration_s"] for u in updates],
                [u["duration_s"] for u in updates],
                label=f"Tenant {tenant}",
                color=f"C{tenant}",
                s=32,
            )
        ax.set(
            ylabel="Training update latency (s)",
            ylim=(0, training_limit),
        )
        ax.legend(fontsize=7, ncol=min(4, tenants), loc="lower left")
        if row == 0:
            ax.set_title("Training latency vs update submission time")
        for ax in axes[row]:
            ax.set(xlim=(0, time_limit), xlabel="Seconds since workload start")
            ax.grid(alpha=0.2)
            if independent:
                ax.axvspan(*independent["measurement_window_s"], color="grey", alpha=0.08)
        ax = phase_axes[row]
        for tenant in range(tenants):
            rs = [r for r in requests if r["tenant"] == tenant]
            us = [u for u in d["updates"] if u["tenant"] == tenant]
            ax.broken_barh(
                [(r["start_s"], r["duration_s"]) for r in rs],
                (tenant - 0.35, 0.7),
                facecolors="tab:orange",
                alpha=0.35,
            )
            ax.broken_barh(
                [(u["finished_s"] - u["duration_s"], u["duration_s"]) for u in us],
                (tenant - 0.35, 0.7),
                facecolors="tab:blue",
                alpha=0.85,
            )
        for publication in d["publications"]:
            if publication.get("start_s") is not None:
                tenant = int(publication["tenant"].split("-")[-1])
                ax.broken_barh(
                    [(publication["start_s"], publication["duration_s"])],
                    (tenant - 0.35, 0.7),
                    facecolors="tab:purple",
                    alpha=0.7,
                )
        if independent:
            ax.axvspan(*independent["measurement_window_s"], color="grey", alpha=0.08)
        ax.set(
            xlim=(0, time_limit),
            ylim=(-0.6, tenants - 0.4),
            yticks=range(tenants),
            ylabel=f"{tenants} tenant(s)\nTenant ID",
            xlabel="Seconds since workload start" if row == len(tenant_counts) - 1 else "",
        )
        ax.grid(axis="x", alpha=0.2)
        summary.append(
            dict(
                observation_window=observation_summary(d) if independent else None,
                tenants=tenants,
                elapsed_s=elapsed,
                sampling_p50_s=float(np.median([r["duration_s"] for r in requests])),
                sampling_p95_s=float(np.percentile([r["duration_s"] for r in requests], 95)),
                training_p50_s=float(np.median([u["duration_s"] for u in d["updates"]])),
                training_p95_s=float(np.percentile([u["duration_s"] for u in d["updates"]], 95)),
            )
        )
    timing = (
        "Independent seeded arrivals and client delays; grey: observation window"
        if independent
        else "Existing five-second stagger; clients are not initially uncorrelated"
    )
    fig.suptitle(
        "Static allocation · Qwen3.5-4B · 2 GPUs · " + timing + "\n"
        "Latency includes queueing; colors identify tenants. Empty bins have no percentile line.",
        fontsize=12,
    )
    phase.suptitle(
        "Static allocation · client phases over time\n"
        "Orange: outstanding sampling. Blue: submitted training, including queueing. "
        "Purple: publication (when timestamps are available).",
        fontsize=12,
    )
    for f, name in [(fig, "static-timelines"), (phase, "static-client-phases")]:
        f.tight_layout(rect=(0, 0, 1, 0.95))
        for ext in ("png", "svg"):
            f.savefig(output / f"{name}.{ext}", dpi=160)
    (output / "static-latency-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    main(args.directory, args.output)
