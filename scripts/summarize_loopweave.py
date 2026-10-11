"""Summarize raw paired GPU runs without substituting simulated measurements."""

import argparse
import json
import statistics
from pathlib import Path


def summarize(data):
    groups = {}
    for row in data["runs"]:
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
            result[policy]["gpu_sm_percent_mean"] = statistics.mean(
                gpu["sm_percent"] for sample in samples for gpu in sample["gpu"]
            )
            result[policy]["peak_memory_gib_per_gpu"] = [
                max(sample["gpu"][i]["memory_bytes"] for sample in samples) / 2**30
                for i in range(len(samples[0]["gpu"]))
            ]
    if "round_robin" in groups and "loopweave" in groups:
        baseline = {r["repeat"]: r for r in groups["round_robin"]}
        ratios = []
        for row in groups["loopweave"]:
            before = baseline[row["repeat"]]
            if (row["training_steps"], row["output_tokens"]) != (
                before["training_steps"],
                before["output_tokens"],
            ):
                raise ValueError("Work differs across paired runs; do not report a speedup")
            ratios.append(row["steps_per_s"] / before["steps_per_s"])
        result["paired_speedup"] = {
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
