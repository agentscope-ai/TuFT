"""Analyze saved borrowing-budget telemetry without launching cloud work."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
from report_scheduler_fixes import detail


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ARMS = ("budget_5s", "budget_35s")
LABELS = ("5-second budget", "35-second budget")
COLORS = ("#64748b", "#0f766e")


def summarize(arm):
    summary = detail(arm)
    before, after = arm["scheduler_before"], arm["scheduler_after"]
    origin = before["elapsed_s"]
    trace = [
        dict(e, elapsed_s=e["elapsed_s"] - origin)
        for e in after["decision_trace"]
        if e["elapsed_s"] >= origin
    ]
    admissions = [e for e in trace if e["event"] == "admit"]
    assert len(admissions) == summary["requests"], "Admission trace incomplete"
    shared = [e for e in admissions if e["replica"] == 0]
    summary["max_drain_s"] = max(
        (e["drain_s"] for e in trace if e["event"] == "exclusive"), default=0
    )
    assert len(shared) == summary["scheduler"]["replica_0_completed"]
    probes = summary["scheduler"]["borrow_probe_requests"]
    rejected = [e for e in after["borrow_rejection_trace"] if e["elapsed_s"] >= origin]
    intervals, active, started = [], set(), None
    for event in trace:
        if event.get("replica") != 0 or event["event"] not in ("admit", "complete"):
            continue
        if event["event"] == "admit":
            if not active:
                started = event["elapsed_s"]
            active.add(event["request_id"])
        else:
            active.remove(event["request_id"])
            if not active:
                intervals.append(dict(start_s=started, end_s=event["elapsed_s"]))
    assert not active
    summary["borrowing"] = dict(
        budget_s=(
            arm["model_config"]["sampling"]["borrowing"]["max_duration_s"]
            if "sampling" in arm["model_config"]
            else arm["model_config"]["scheduling"]["borrow_max_duration_s"]
        ),
        probe_requests=probes,
        non_probe_requests=len(shared) - probes,
        total_requests=len(shared),
        first_admission_s=min((e["elapsed_s"] for e in shared), default=None),
        last_admission_s=max((e["elapsed_s"] for e in shared), default=None),
        admissions_after_first_update=sum(
            e["elapsed_s"] > summary["first_update_finished_s"] for e in shared
        ),
        busy_intervals=intervals,
        shared_service_durations_s=[
            e["service_s"] for e in trace if e["event"] == "complete" and e["replica"] == 0
        ],
        rejection_counts={
            k: v - before.get(k, 0) for k, v in after.items() if k.startswith("borrow_rejected_")
        },
        saved_rejection_events=len(rejected),
        saved_rejection_reasons=dict(Counter(e["reason"] for e in rejected)),
        saved_rejection_estimate_sources=dict(Counter(e["estimate_source"] for e in rejected)),
    )
    return summary


def save_figure(fig, output, name):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    arms = json.loads(args.result.read_text())
    assert (
        arms[ARMS[0]]["initial_adapter_fingerprints"]
        == arms[ARMS[1]]["initial_adapter_fingerprints"]
    )
    summary = {key: summarize(arms[key]) for key in ARMS}
    before, after = [summary[key] for key in ARMS]
    verdict = dict(
        borrowing_beyond_cold_probes=after["borrowing"]["non_probe_requests"] > 0,
        borrowing_after_first_update=after["borrowing"]["admissions_after_first_update"] > 0,
        max_drain_within_20s=after["max_drain_s"] <= 20,
        update_p95_within_10_percent=after["update_p95_s"] <= before["update_p95_s"] * 1.10,
        elapsed_change_fraction=after["elapsed_s"] / before["elapsed_s"] - 1,
        output_token_change_fraction=after["output_tokens"] / before["output_tokens"] - 1,
        note="One ordered pair; thresholds evaluate this workload, not a production guarantee.",
    )
    output = args.result.parent
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "evaluation.json").write_text(json.dumps(verdict, indent=2) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = [
        ("elapsed_s", "Completion time", "Seconds"),
        ("request_p95_s", "Sampling latency (p95)", "Seconds"),
        ("update_p95_s", "Training-update latency (p95)", "Seconds"),
        ("max_drain_s", "Longest wait for shared sampling to drain", "Seconds"),
        ("updates_per_gpu_hour", "Resource efficiency", "Updates / GPU-hour"),
    ]
    for ax, (key, title, unit) in zip(axes.flat, panels, strict=False):
        values = [summary[k][key] for k in ARMS]
        bars = ax.bar(LABELS, values, color=COLORS)
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set(title=title, ylabel=unit, ylim=(0, max(values) * 1.25 + 0.01))
    ax = axes.flat[-1]
    probes = [summary[k]["borrowing"]["probe_requests"] for k in ARMS]
    learned = [summary[k]["borrowing"]["non_probe_requests"] for k in ARMS]
    ax.bar(LABELS, probes, label="Cold probes", color="#94a3b8")
    bars = ax.bar(LABELS, learned, bottom=probes, label="Learned borrowing", color="#0f766e")
    ax.bar_label(
        bars, labels=[str(int(p + n)) for p, n in zip(probes, learned, strict=True)], padding=3
    )
    ax.set(title="Shared-sampler requests", ylabel="Requests")
    ax.margins(y=0.25)
    ax.legend()
    fig.suptitle("Qwen3.5-4B · 16 LoRA tenants · 2 H100s · current scheduler in both arms")
    fig.text(
        0.5,
        0.015,
        "One ordered pair · identical starting adapters · 384 requests and 32 updates per arm",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save_figure(fig, output, "budget-comparison")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, key, label in zip(axes, ARMS, LABELS, strict=True):
        rows = [
            json.loads(line[9:])
            for line in (output / f"{key}.log").read_text().splitlines()
            if line.startswith("PROGRESS ")
        ]
        times = [r["elapsed_s"] for r in rows]
        # The first probe finishes before the first progress snapshot. Use the
        # complete event trace so the figure does not hide short borrowing bursts.
        origin = arms[key]["scheduler_before"]["elapsed_s"]
        shared_times, shared_counts, active = [0.0], [0], 0
        for event in arms[key]["scheduler_after"]["decision_trace"]:
            if (
                event["elapsed_s"] >= origin
                and event.get("replica") == 0
                and event["event"] in ("admit", "complete")
            ):
                active += 1 if event["event"] == "admit" else -1
                shared_times.append(event["elapsed_s"] - origin)
                shared_counts.append(active)
        shared_times.append(arms[key]["elapsed_s"])
        shared_counts.append(active)
        ax.step(
            shared_times,
            shared_counts,
            where="post",
            label="Shared active",
            color="#0f766e",
        )
        ax.step(
            times,
            [r["active"][1] for r in rows],
            where="post",
            label="Dedicated active",
            color="#64748b",
        )
        ax.step(times, [r["queued"] for r in rows], where="post", label="Queued", color="#d97706")
        ax.fill_between(
            times,
            0,
            35,
            where=[r["training_active"] for r in rows],
            step="post",
            color="#cbd5e1",
            alpha=0.35,
            label="Training active",
        )
        ax.set(title=label, ylabel="Requests", ylim=(0, 36))
        ax.legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Elapsed seconds (shared: event trace; other series: 15-second snapshots)")
    fig.tight_layout()
    save_figure(fig, output, "borrowing-timeline")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=True)
    for ax, key, label in zip(axes, ARMS, LABELS, strict=True):
        samples = [s for s in arms[key]["gpu"] if s["elapsed_s"] <= arms[key]["elapsed_s"]]
        for index, name in enumerate(("Shared trainer/sampler", "Dedicated sampler")):
            ax.plot(
                [s["elapsed_s"] for s in samples],
                [s["gpu"][index]["sm_percent"] for s in samples],
                label=name,
                alpha=0.7,
                linewidth=0.7,
            )
        ax.set(title=label, ylabel="NVML utilization (%)", xlabel="Elapsed seconds", ylim=(0, 105))
        ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, output, "gpu-timeline")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
