"""Summarize one causal agent pair and save standalone figures; no GPU work."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def summarize(data: dict) -> list[dict]:
    if data.get("returncode") or data.get("timed_out") or len(data["runs"]) != 2:
        raise ValueError("A successful complete pair is required for comparison")
    rows = []
    for run in data["runs"]:
        before, after = run["scheduler_before"], run["scheduler_after"]
        delta = {
            key: value - before.get(key, 0)
            for key, value in after.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        rows.append(
            dict(
                policy=run["policy"],
                elapsed_s=run["elapsed_s"],
                updates=len(run["updates"]),
                updates_per_s=len(run["updates"]) / run["elapsed_s"],
                gpu_s_per_update=2 * run["elapsed_s"] / len(run["updates"]),
                requests=len(run["requests"]),
                episodes=len(run["episodes"]),
                prompt_tokens=sum(r["prompt_tokens"] for r in run["requests"]),
                output_tokens=sum(r["output_tokens"] for r in run["requests"]),
                prompt_range=[
                    min(r["prompt_tokens"] for r in run["requests"]),
                    max(r["prompt_tokens"] for r in run["requests"]),
                ],
                sample_mean_s=statistics.mean(r["duration_s"] for r in run["requests"]),
                sample_p95_s=percentile([r["duration_s"] for r in run["requests"]], 0.95),
                episode_p95_s=percentile([r["duration_s"] for r in run["episodes"]], 0.95),
                update_mean_s=statistics.mean(r["duration_s"] for r in run["updates"]),
                reward_mean=statistics.mean(r["reward"] for r in run["episodes"]),
                valid_tool_fraction=sum(r["valid_tools"] for r in run["episodes"])
                / (2 * len(run["episodes"])),
                publication_mean_s=statistics.mean(r["duration_s"] for r in run["publications"]),
                peak_gpu_gib=max(g["memory_bytes"] for sample in run["gpu"] for g in sample["gpu"])
                / 1024**3,
                tenant_finish_s={
                    str(i): max(u["finished_s"] for u in run["updates"] if u["tenant"] == i)
                    for i in range(run["tenants"])
                },
                scheduler_delta=delta,
                profile_status=after["loop_profile_status"],
            )
        )
    assert rows[0]["updates"] == rows[1]["updates"]
    assert rows[0]["episodes"] == rows[1]["episodes"]
    return rows


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    rows = summarize(data)
    target = args.input.with_suffix("")
    Path(str(target) + "-summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    colors = ["#526b84", "#159782"]
    labels = ["Static (matched admission)", "LoopWeave"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i, (run, row) in enumerate(zip(data["runs"], rows, strict=True)):
        times = sorted(u["finished_s"] for u in run["updates"])
        axes[0, 0].step(
            [0] + times, list(range(len(times) + 1)), where="post", color=colors[i], label=labels[i]
        )
        latencies = sorted(r["duration_s"] for r in run["requests"])
        axes[0, 1].plot(
            latencies,
            [(j + 1) / len(latencies) for j in range(len(latencies))],
            color=colors[i],
            label=labels[i],
        )
        parts = [
            "training_queue_wait_s",
            "training_drain_wait_s",
            "training_phase_s",
            "training_execution_s",
        ]
        bottom = 0
        for key, color in zip(parts, ["#e6b566", "#c76161", "#9175b5", "#648b9c"], strict=True):
            value = row["scheduler_delta"].get(key, 0) / row["updates"]
            axes[1, 0].bar(
                i,
                value,
                bottom=bottom,
                color=color,
                label=key.removeprefix("training_").removesuffix("_s") if i == 0 else None,
            )
            bottom += value
        for gpu in (0, 1):
            axes[1, 1].plot(
                [s["elapsed_s"] for s in run["gpu"]],
                [s["gpu"][gpu]["memory_bytes"] / 1024**3 for s in run["gpu"]],
                color=colors[i],
                linestyle="-" if gpu == 0 else "--",
                alpha=0.8,
                label=f"{labels[i]} GPU {gpu}",
            )
    axes[0, 0].set(xlabel="Measured workload time (s)", ylabel="Completed updates")
    axes[0, 1].set(xlabel="Sampling request latency (s)", ylabel="Fraction completed")
    axes[1, 0].set(
        xticks=[0, 1], xticklabels=labels, ylabel="Seconds per update (sum of its operations)"
    )
    axes[1, 1].set(xlabel="Measured workload time (s)", ylabel="GPU memory (GiB)")
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, axis="y")
    matched = all(rows[0][k] == rows[1][k] for k in ("prompt_tokens", "output_tokens"))
    reference = data["runs"][0]
    fig.suptitle(
        f"Qwen3-4B · {reference['tenants']} tenants · 3-turn tool tasks · "
        f"{reference['rounds']} updates/tenant\nOne ordered pair · "
        + ("identical token counts" if matched else "different generated token counts")
        + " · eager sampling"
    )
    for extension in ("png", "svg"):
        fig.savefig(str(target) + "." + extension, dpi=180)
    # Keep the full-run CDF primary; show the first-wave sensitivity explicitly.
    extra, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    sensitivity = []
    for i, run in enumerate(data["runs"]):
        rounds = range(run["rounds"])
        samples = [
            [r["duration_s"] for r in run["requests"] if r["step"] == step] for step in rounds
        ]
        ax[0].plot(
            list(rounds),
            [statistics.mean(v) for v in samples],
            "o-",
            color=colors[i],
            label=labels[i],
        )
        ax[1].plot(
            list(rounds),
            [percentile(v, 0.95) for v in samples],
            "o-",
            color=colors[i],
            label=labels[i],
        )
        later = [r["duration_s"] for r in run["requests"] if r["step"] > 0]
        first = max(u["finished_s"] for u in run["updates"] if u["step"] == 0)
        sensitivity.append(
            dict(
                policy=run["policy"],
                after_first_update_milestone_s=run["elapsed_s"] - first,
                later_rounds_sample_mean_s=statistics.mean(later) if later else None,
                later_rounds_sample_p95_s=percentile(later, 0.95) if later else None,
            )
        )
    for panel, metric in zip(ax, ("Mean", "p95"), strict=True):
        panel.set(
            xlabel="Update round (0 is the first measured wave)",
            ylabel=f"{metric} sampling latency (s)",
        )
        panel.legend(fontsize=8)
        panel.grid(alpha=0.2)
    extra.suptitle("First-wave sensitivity · full-run metrics remain primary")
    for extension in ("png", "svg"):
        extra.savefig(str(target) + "-latency-rounds." + extension, dpi=180)
    Path(str(target) + "-sensitivity.json").write_text(json.dumps(sensitivity, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
