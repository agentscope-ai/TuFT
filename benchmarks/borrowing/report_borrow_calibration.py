"""Compare two saved calibrations of the same model; no GPU work."""

import argparse
import json
from pathlib import Path

import matplotlib
from report_coding_comparison import summarize


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("single_probe", type=Path)
    parser.add_argument("batch_probe", type=Path)
    args = parser.parse_args()
    paths = [args.single_probe, args.batch_probe]
    provenance = [json.loads((p.parent / "provenance.json").read_text()) for p in paths]
    for key in (
        "model",
        "model_revision",
        "tenants",
        "trajectories",
        "updates_per_tenant",
        "max_output_tokens",
        "max_num_seqs",
        "max_loras",
        "kv_cache_gib",
    ):
        assert provenance[0][key] == provenance[1][key], f"Different configuration: {key}"
    results = [summarize(json.loads(p.read_text())["shared_weight"]) for p in paths]
    output = args.batch_probe.parent
    comparison = dict(zip(("single_probe", "batch_probe"), results, strict=True))
    (output / "borrowing-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    panels = [
        (
            [r["scheduler"]["replica_0_completed"] for r in results],
            "Shared sampling requests",
            "Requests",
        ),
        (
            [r["rollout_gpu_utilization_percent"][0] for r in results],
            "Shared GPU during rollout",
            "NVML utilization (%)",
        ),
        (
            [r["output_tokens_per_s"] for r in results],
            "Whole-workload throughput",
            "Output tokens/s",
        ),
        ([r["request_p95_s"] for r in results], "Sampling latency (p95)", "Seconds"),
        (
            [r["update_p95_s"] for r in results],
            "Training update latency (p95)",
            "Seconds",
        ),
        ([r["elapsed_s"] for r in results], "Workload completion", "Seconds"),
    ]
    for ax, (values, title, ylabel) in zip(axes.flat, panels, strict=True):
        bars = ax.bar(["Single probe", "Batch probe"], values, color=["#64748b", "#147d92"])
        ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set(title=title, ylabel=ylabel, ylim=(0, max(values) * 1.2 + 0.01))
    fig.suptitle("Qwen3.5-4B · 16 tenants · one calibration per configuration")
    fig.text(
        0.5,
        0.015,
        "Same task shape and sampler limits; independent adapter initialization. "
        "Sampling drain <1 ms in both runs. Not a static/shared comparison.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    for ext in ("png", "svg"):
        fig.savefig(output / f"borrowing-comparison.{ext}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
