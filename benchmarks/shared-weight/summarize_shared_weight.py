"""Summarize raw paired GPU runs without substituting simulated measurements."""

import argparse
import json
import statistics
from pathlib import Path


def performance_data(data):
    """Unwrap a bounded job without reporting partial/failed pairs as success."""
    if data.get("returncode", 0) or data.get("timed_out", False):
        raise ValueError("Successful performance runs are required")
    return data.get("benchmark", data)


def comparison_runs(data):
    """Select a static/shared pair, requiring evidence of physical weight sharing."""
    runs = []
    for row in data["runs"]:
        if row["policy"] == "round_robin":
            runs.append(row)
        elif row["policy"] == "shared_weight" and row.get("shared_weights"):
            runs.append({**row, "policy": "shared_weight"})
    if {r["policy"] for r in runs} != {"round_robin", "shared_weight"}:
        raise ValueError(
            "Static and physically shared Shared-weight borrowing measurements are required"
        )
    return runs


def summarize(data):
    data = performance_data(data)
    if data.get("returncode", 0) or not data["runs"]:
        raise ValueError("Successful performance runs are required")
    if any(row.get("diagnostic_only") for row in data["runs"]):
        raise ValueError("Numerical diagnostics are not performance measurements")
    groups = {}
    for row in comparison_runs(data):
        groups.setdefault(row["policy"], []).append(row)
    result = {}
    for policy, rows in groups.items():
        result[policy] = {
            "runs": len(rows),
            "elapsed_median_s": statistics.median(r["elapsed_s"] for r in rows),
            "steps_per_s_median": statistics.median(r["steps_per_s"] for r in rows),
            "gpu_seconds_per_step": sum(r["gpu_seconds"] for r in rows)
            / sum(r["training_steps"] for r in rows),
            "sample_latency_mean_s": statistics.mean(r["sample_latency_mean_s"] for r in rows),
            "sample_latency_p95_s": statistics.mean(r["sample_latency_p95_s"] for r in rows),
            "train_latency_mean_s": statistics.mean(r["train_latency_mean_s"] for r in rows),
            "total_output_tokens": sum(r["output_tokens"] for r in rows),
        }
        samples = [sample for row in rows for sample in row.get("gpu_samples", [])]
        if samples:
            total_memory = [
                sum(gpu["memory_bytes"] for gpu in sample["gpu"]) / 2**30 for sample in samples
            ]
            result[policy]["peak_total_memory_gib"] = max(total_memory)
            result[policy]["mean_total_memory_gib"] = statistics.mean(total_memory)
            result[policy]["gpu_sm_percent_mean"] = statistics.mean(
                gpu["sm_percent"] for sample in samples for gpu in sample["gpu"]
            )
            result[policy]["peak_memory_gib_per_gpu"] = [
                max(sample["gpu"][i]["memory_bytes"] for sample in samples) / 2**30
                for i in range(len(samples[0]["gpu"]))
            ]
        if rows[0].get("shared_weights"):
            result[policy]["shared_base_storage_gib"] = statistics.mean(
                r["shared_weights"]["awake"]["shared_storage_bytes"] / 2**30 for r in rows
            )
            result[policy]["reclaimed_sampler_memory_gib"] = statistics.mean(
                r["shared_weights"]["reclaimed_bytes"] / 2**30 for r in rows
            )
            for phase in ("sleep", "wake"):
                total_s = sum(r["shared_weights_measured_counters"][phase + "_s"] for r in rows)
                count = sum(r["shared_weights_measured_counters"][phase + "_count"] for r in rows)
                result[policy][phase + "_mean_ms"] = 1000 * total_s / count if count else None
    comparisons = (("round_robin", "shared_weight", "shared_weight_vs_static"),)
    for before_policy, after_policy, label in comparisons:
        if before_policy not in groups or after_policy not in groups:
            continue
        baseline = {r["repeat"]: r for r in groups[before_policy]}
        if set(baseline) != {r["repeat"] for r in groups[after_policy]}:
            raise ValueError("Paired repetitions are required")
        ratios = []
        for row in groups[after_policy]:
            before = baseline[row["repeat"]]
            if (row["training_steps"], row["output_tokens"]) != (
                before["training_steps"],
                before["output_tokens"],
            ):
                raise ValueError("Work differs across paired runs; do not report a speedup")
            ratios.append(row["steps_per_s"] / before["steps_per_s"])
        result[label] = {
            "values": ratios,
            "median": statistics.median(ratios),
            "min": min(ratios),
            "max": max(ratios),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(summarize(json.loads(args.input.read_text())), indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report)
