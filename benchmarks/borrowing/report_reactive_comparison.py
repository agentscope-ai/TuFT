"""Compare saved current static/borrowing arms; no GPU work."""

import argparse
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_static_timelines import observation_summary, time_bins


def main(directory: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    data = [json.loads((directory / f"tenants-8-{arm}.json").read_text()) for arm in ("off", "on")]
    labels = ["Static allocation", "GPU borrowing"]
    summary = []
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True, sharey="col")
    phase, paxes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)
    limit = max(d["elapsed_s"] for d in data)
    for row, (label, d) in enumerate(zip(labels, data, strict=True)):
        s = observation_summary(d)
        lo, hi = s["window_s"]
        before, after = d["scheduler_before"], d["scheduler_after"]
        origin = before["elapsed_s"]
        trace = [
            dict(e, t=e["elapsed_s"] - origin)
            for e in after["decision_trace"]
            if e["elapsed_s"] >= origin
        ]
        admissions = [e for e in trace if e["event"] == "admit"]
        borrowed = [e for e in admissions if e["replica"] == 0]
        drains = [
            e["drain_s"] for e in trace if e["event"] == "exclusive" and e["category"] == "training"
        ]
        completed = [r for r in d["requests"] if lo <= r["start_s"] + r["duration_s"] < hi]
        s.update(
            label=label,
            total_requests=len(d["requests"]),
            total_updates=len(d["updates"]),
            borrowed_admissions=len(borrowed),
            window_borrowed_admissions=sum(lo <= e["t"] < hi for e in borrowed),
            borrowed_completions=after.get("replica_0_completed", 0)
            - before.get("replica_0_completed", 0),
            training_drain_max_s=max(drains, default=0),
            training_drain_total_s=sum(drains),
            window_completed_requests=len(completed),
            window_output_tokens_per_s=sum(r["output_tokens"] for r in completed) / (hi - lo),
            task_successes=sum(e["success"] for e in d["episodes"]),
            trajectories=len(d["episodes"]),
            output_limited_requests=sum(r["hit_output_limit"] for r in d["requests"]),
            frozen_base_unchanged=d["frozen_base_unchanged"],
            copied_base_parameters=d["shared"]["copied_parameters"],
        )
        assert len(admissions) == len(d["requests"]), "Incomplete admission trace"
        rows = d["scheduling_samples"]
        times = np.array([r["elapsed_s"] for r in rows])
        ends = np.r_[times[1:], d["elapsed_s"]]
        weights = np.maximum(0, np.minimum(ends, hi) - np.maximum(times, lo))
        s["idle_trainer_with_dedicated_overflow_percent"] = (
            100
            * sum(
                w
                for r, w in zip(rows, weights, strict=True)
                if not r["training_active"]
                and not r["training_pending"]
                and r["queued"] > 0
                and r["active"][1] >= 32
            )
            / (hi - lo)
        )
        s["window_borrowed_active_seconds"] = sum(
            w for r, w in zip(rows, weights, strict=True) if r["active"][0] > 0
        )
        s["borrowed_admissions_after_first_update"] = sum(
            e["t"] > min(u["finished_s"] for u in d["updates"]) for e in borrowed
        )
        s["first_borrowed_admission_s"] = min((e["t"] for e in borrowed), default=None)
        s["last_borrowed_admission_s"] = max((e["t"] for e in borrowed), default=None)
        s["borrowed_service_p95_s"] = (
            float(
                np.percentile(
                    [
                        e["service_s"]
                        for e in trace
                        if e["event"] == "complete" and e["replica"] == 0
                    ],
                    95,
                )
            )
            if borrowed
            else None
        )
        s["completed_updates_before_window"] = sum(u["finished_s"] < lo for u in d["updates"])
        s["completed_updates_by_window_end"] = sum(u["finished_s"] < hi for u in d["updates"])
        s["training_operations"] = {}
        for kind in ("backward", "optimizer"):
            events = [e for e in trace if e["event"] == "exclusive" and e["kind"] == kind]
            s["training_operations"][kind] = (
                {
                    "count": len(events),
                    "queue_p95_s": float(np.percentile([e["queue_s"] for e in events], 95)),
                    "drain_max_s": max((e["drain_s"] for e in events), default=0),
                    "service_p50_s": float(np.median([e["service_s"] for e in events])),
                }
                if events
                else {}
            )
        publications = [p for p in d["publications"] if p.get("start_s") is not None]
        publication_events = [
            e for e in trace if e["event"] == "exclusive" and e["kind"] == "publication"
        ]
        s["publication_p95_s"] = (
            float(np.percentile([p["duration_s"] for p in publications], 95))
            if publications
            else None
        )
        s["publication_max_s"] = max((p["duration_s"] for p in publications), default=0)
        s["publication_gate_queue_max_s"] = max(
            (e["queue_s"] for e in publication_events), default=0
        )
        s["publication_gate_service_max_s"] = max(
            (e["service_s"] for e in publication_events), default=0
        )
        s["window_dedicated_empty_seconds"] = float(
            sum(w for r, w in zip(rows, weights, strict=True) if r["active"][1] == 0)
        )
        summary.append(s)
        x, y = time_bins(d["gpu"], d["elapsed_s"])
        axes[row, 0].plot(x, y[:, 0], label="Trainer GPU")
        axes[row, 0].plot(x, y[:, 1], label="Dedicated sampler GPU")
        axes[row, 0].set(ylabel=f"{label}\nGPU utilization (%)", ylim=(0, 100))
        axes[row, 0].legend(fontsize=8)
        for tenant in range(8):
            rs = [r for r in d["requests"] if r["tenant"] == tenant]
            us = [u for u in d["updates"] if u["tenant"] == tenant]
            axes[row, 1].scatter(
                [r["start_s"] for r in rs],
                [r["duration_s"] for r in rs],
                s=6,
                alpha=0.4,
                color=f"C{tenant}",
            )
            axes[row, 2].scatter(
                [u["finished_s"] - u["duration_s"] for u in us],
                [u["duration_s"] for u in us],
                s=25,
                color=f"C{tenant}",
            )
            paxes[row].broken_barh(
                [(r["start_s"], r["duration_s"]) for r in rs],
                (tenant - 0.35, 0.7),
                facecolors="tab:orange",
                alpha=0.35,
            )
            paxes[row].broken_barh(
                [(u["finished_s"] - u["duration_s"], u["duration_s"]) for u in us],
                (tenant - 0.35, 0.7),
                facecolors="tab:blue",
                alpha=0.85,
            )
        for pub in d["publications"]:
            if pub.get("start_s") is not None:
                paxes[row].broken_barh(
                    [(pub["start_s"], pub["duration_s"])],
                    (int(pub["tenant"].split("-")[-1]) - 0.35, 0.7),
                    facecolors="tab:purple",
                    alpha=0.7,
                )
        for ax in [*axes[row], paxes[row]]:
            ax.axvspan(lo, hi, color="grey", alpha=0.08)
            ax.set(xlim=(0, limit), xlabel="Seconds since workload start")
            ax.grid(alpha=0.2)
        paxes[row].set(ylabel=f"{label}\nTenant ID", ylim=(-0.6, 7.6), yticks=range(8))
        axes[row, 1].set(
            ylabel="Sampling latency (s)",
            ylim=(0, 1.1 * max(r["duration_s"] for d in data for r in d["requests"])),
        )
        axes[row, 2].set(
            ylabel="Training update latency (s)",
            ylim=(0, 1.1 * max(u["duration_s"] for d in data for u in d["updates"])),
        )
    for ax, title in zip(
        axes[0],
        [
            "GPU busy fraction · 10 s bins",
            "Sampling latency vs submission time",
            "Training latency vs submission time",
        ],
        strict=True,
    ):
        ax.set_title(title)
    fig.suptitle(
        "Qwen3.5-4B · 2 GPUs · 8 independent tenants\n"
        "Grey: 240–840 s observation window. Latency includes queueing."
    )
    phase.suptitle(
        "Client phases · equal tenant row height\n"
        "Orange: outstanding sampling; blue: training including queueing; "
        "purple: adapter publication"
    )
    for name, f in [("comparison-timelines", fig), ("comparison-client-phases", phase)]:
        f.tight_layout()
        f.savefig(output / f"{name}.png", dpi=160)
        f.savefig(output / f"{name}.svg")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.directory, args.output)
