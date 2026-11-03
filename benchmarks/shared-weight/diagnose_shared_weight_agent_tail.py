"""Offline cohort attribution for saved agent telemetry; never launches GPU work."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from report_shared_weight_agents import percentile


def diagnose(data: dict) -> dict:
    assert data["returncode"] == 0 and not data["timed_out"]
    baseline = data["runs"][0]
    assert baseline["policy"] == "balanced"
    threshold = percentile([r["duration_s"] for r in baseline["requests"]], 0.95)
    rows = []
    for run in data["runs"]:
        requests = run["requests"]
        before, after = run["scheduler_before"], run["scheduler_after"]
        origin = before["elapsed_s"]
        trace = [e for e in after["decision_trace"] if e["elapsed_s"] >= origin]
        admissions = [e for e in trace if e["event"] == "admit"]
        assert len(admissions) == len(requests) == 576
        assert (
            len(after["decision_trace"])
            < run["model_config"]["scheduling"]["scheduling_trace_capacity"]
        )
        first = [r for r in requests if r["step"] == 0 and r["turn"] == 0]
        first_round = [r for r in requests if r["step"] == 0]
        later = [r for r in requests if r["step"] > 0]
        # Separate clocks have a constant origin offset. Snapshot bracketing
        # bounds that offset by the non-measured time at the two boundaries.
        clock_bound = after["elapsed_s"] - origin - run["elapsed_s"]
        assert 0 <= clock_bound < 0.01
        first_round_end = max(r["start_s"] + r["duration_s"] for r in first_round)
        wave = [e for e in admissions if e["elapsed_s"] - origin < first_round_end]
        assert len(wave) == len(first_round) == 96
        # Token-work signatures distinguish turn 0 from turns 1/2 in this data.
        # This is a cohort join, never a fabricated individual request join.
        work = Counter(r["prompt_tokens"] + 96 for r in first)
        assert set(work).isdisjoint(r["prompt_tokens"] + 96 for r in first_round if r["turn"] != 0)
        first_admits = [e for e in wave if e["work"] in work]
        assert Counter(e["work"] for e in first_admits) == work
        mean_admit = statistics.mean(e["elapsed_s"] - origin for e in first_admits)
        pre = mean_admit - statistics.mean(r["start_s"] for r in first)
        post = statistics.mean(r["start_s"] + r["duration_s"] for r in first) - mean_admit
        sorted_requests = sorted(requests, key=lambda r: r["duration_s"])
        rank = (len(requests) - 1) * 0.95
        p95 = percentile([r["duration_s"] for r in requests], 0.95)
        exclusive = [e for e in trace if e["event"] == "exclusive"]
        first_exclusive = min(
            e["elapsed_s"] - origin - e["service_s"] - e["phase_s"] - e["drain_s"]
            for e in exclusive
        )
        slow = [r for r in requests if r["duration_s"] > threshold]
        assert max(r["start_s"] + r["duration_s"] for r in slow) < (first_exclusive - clock_bound)
        fixed = 1 if run["policy"] == "shared_weight" else 0
        fixed_admits = [e for e in admissions if e["replica"] == fixed]
        rows.append(
            dict(
                policy=run["policy"],
                sample_p95_s=p95,
                p95_rank_one_based=rank + 1,
                p95_bracketing_requests=[
                    {k: r[k] for k in ("tenant", "step", "trajectory", "turn", "duration_s")}
                    for r in sorted_requests[int(rank) : int(rank) + 2]
                ],
                above_baseline_p95_by_round_turn={
                    f"{step}/{turn}": count
                    for (step, turn), count in sorted(
                        Counter((r["step"], r["turn"]) for r in slow).items()
                    )
                },
                above_own_p95_by_round_turn={
                    f"{step}/{turn}": count
                    for (step, turn), count in sorted(
                        Counter(
                            (r["step"], r["turn"]) for r in requests if r["duration_s"] > p95
                        ).items()
                    )
                },
                first_turn_mean_s=statistics.mean(r["duration_s"] for r in first),
                first_turn_pre_admit_mean_bounds_s=[pre - clock_bound, pre],
                first_turn_post_admit_mean_bounds_s=[post, post + clock_bound],
                first_turn_admissions_by_replica=dict(Counter(e["replica"] for e in first_admits)),
                first_round_admissions_by_replica=dict(Counter(e["replica"] for e in wave)),
                first_round_flex_admission_times_s=[
                    e["elapsed_s"] - origin
                    for e in wave
                    if run["policy"] == "shared_weight" and e["replica"] == 0
                ],
                first_fixed_replenishment_s=fixed_admits[16]["elapsed_s"] - origin,
                initial_fixed_reserved_work=fixed_admits[15]["reserved_work"],
                first_exclusive_gate_s=first_exclusive,
                clock_offset_upper_bound_s=clock_bound,
                initial_sampler_rates=before["sampling_seconds_per_token"],
                cold_probes=after.get("borrow_probes", 0) - before.get("borrow_probes", 0),
                first_round_quantiles={
                    str(q): percentile([r["duration_s"] for r in first_round], q)
                    for q in (0.5, 0.65, 0.70, 0.75, 0.95)
                },
                later_rounds_p95_s=percentile([r["duration_s"] for r in later], 0.95),
                per_round_p95_s=[
                    percentile([r["duration_s"] for r in requests if r["step"] == step], 0.95)
                    for step in range(run["rounds"])
                ],
            )
        )
    return {"baseline_p95_threshold_s": threshold, "runs": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    result = diagnose(data)
    target = args.input.with_name(args.input.stem + "-tail-diagnosis.json")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(target)
    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for run, color, label in zip(
            data["runs"], ["#526b84", "#159782"], ["Static", "Shared-weight borrowing"], strict=True
        ):
            first = sorted(r["duration_s"] for r in run["requests"] if r["step"] == 0)
            axes[0].step(
                first,
                [(i + 1) / len(first) for i in range(len(first))],
                where="post",
                color=color,
                label=label,
            )
            full = sorted(r["duration_s"] for r in run["requests"])
            axes[1].plot(range(1, len(full) + 1), full, color=color, label=label)
        axes[0].axhline(0.70, color="#888888", linestyle=":", label="First-round 70th percentile")
        axes[0].set(
            xlabel="Sampling latency (seconds)",
            ylabel="Fraction of first-round requests",
            title="First-round distributions cross",
            ylim=(0.45, 1.01),
        )
        axes[1].axvline(547.25, color="#888888", linestyle=":", label="Full-run p95 rank")
        axes[1].set(
            xlabel="Request rank, sorted by latency (576 total)",
            ylabel="Sampling latency (seconds)",
            title="A broader startup tail raises pooled p95",
            xlim=(525, 576),
        )
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        for suffix in (".png", ".svg"):
            fig.savefig(target.with_suffix(suffix), dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    main()
