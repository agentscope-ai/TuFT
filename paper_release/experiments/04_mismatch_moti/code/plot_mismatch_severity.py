"""Plot mismatch severity across three runtime settings on common rounds."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TINKER = "#DBE2EF"
SELF = "#3F72AF"
QUANT = "#112D4E"
TEXT = "#112D4E"
GRID = "#DBE2EF"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = int(q * (len(ys) - 1))
    return float(ys[idx])


def _load_json_summary(path: Path, label: str) -> list[dict[str, Any]]:
    with path.open() as f:
        rows = json.load(f)
    out: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        r["series"] = label
        r["round"] = int(r["round"])
        out.append(r)
    return out


def _aggregate_quant_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {
            "diffs": [],
            "abs_diffs": [],
            "cum_diffs": [],
            "is_weights": [],
        }
    )

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            step = int(row["step"])
            sampling = row["sampling_logprobs"]
            training = row["training_logprobs"]
            n = min(len(sampling), len(training))
            if n <= 0:
                continue
            diffs = [float(sampling[i]) - float(training[i]) for i in range(n)]
            cum_diff = sum(diffs)
            is_weight = math.exp(max(min(cum_diff, 50.0), -50.0))
            bucket = by_step[step]
            bucket["diffs"].extend(diffs)
            bucket["abs_diffs"].extend(abs(x) for x in diffs)
            bucket["cum_diffs"].append(cum_diff)
            bucket["is_weights"].append(is_weight)

    rows: list[dict[str, Any]] = []
    for step, values in sorted(by_step.items()):
        diffs = values["diffs"]
        abs_diffs = values["abs_diffs"]
        cum_diffs = values["cum_diffs"]
        is_weights = values["is_weights"]
        if not diffs:
            continue
        rows.append(
            {
                "series": label,
                "round": step,
                "mean_diff": mean(diffs),
                "mean_abs_diff": mean(abs_diffs),
                "max_abs_diff": max(abs_diffs),
                "std_diff": pstdev(diffs),
                "num_tokens": len(diffs),
                "mean_abs_cum_diff": mean(abs(x) for x in cum_diffs),
                "mean_cum_diff": mean(cum_diffs),
                "max_cum_diff": max(abs(x) for x in cum_diffs),
                "std_cum_diff": pstdev(cum_diffs),
                "mean_is_weight": mean(is_weights),
                "std_is_weight": pstdev(is_weights),
                "min_is_weight": min(is_weights),
                "max_is_weight": max(is_weights),
                "p99_is_weight": _quantile(is_weights, 0.99),
                "p_out_clip_02": sum(1 for w in is_weights if abs(w - 1.0) > 0.2) / len(is_weights),
                "p_out_clip_01": sum(1 for w in is_weights if abs(w - 1.0) > 0.1) / len(is_weights),
            }
        )
    return rows


def load_all() -> dict[str, list[dict[str, Any]]]:
    return {
        "Tinker": _load_json_summary(DATA_DIR / "mismatch_results_tinker.json", "Tinker"),
        "Self-hosted": _load_json_summary(DATA_DIR / "mismatch_results_tuft.json", "Self-hosted"),
        "Self-hosted+Quant": _aggregate_quant_jsonl(
            DATA_DIR / "tp2_fsdp1_id0_quant.jsonl", "Self-hosted+Quant"
        ),
    }


def _common_rounds(series: dict[str, list[dict[str, Any]]]) -> list[int]:
    max_common_round = min(max(r["round"] for r in rows) for rows in series.values())
    round_sets = [set(r["round"] for r in rows if r["round"] <= max_common_round) for rows in series.values()]
    common = set.intersection(*round_sets)
    return sorted(common)


def write_summary_csv(series: dict[str, list[dict[str, Any]]], common_rounds: list[int]) -> None:
    fields = [
        "series",
        "n_common_rounds",
        "common_round_min",
        "common_round_max",
        "mean_mean_abs_diff",
        "median_mean_abs_diff",
        "max_mean_abs_diff",
        "mean_max_abs_diff",
        "median_p99_is_weight",
        "mean_clip01",
    ]
    with (OUT_DIR / "mismatch_severity_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, rows in series.items():
            rows = [r for r in rows if r["round"] in common_rounds]
            writer.writerow(
                {
                    "series": name,
                    "n_common_rounds": len(rows),
                    "common_round_min": min(common_rounds),
                    "common_round_max": max(common_rounds),
                    "mean_mean_abs_diff": mean(r["mean_abs_diff"] for r in rows),
                    "median_mean_abs_diff": median(r["mean_abs_diff"] for r in rows),
                    "max_mean_abs_diff": max(r["mean_abs_diff"] for r in rows),
                    "mean_max_abs_diff": mean(r["max_abs_diff"] for r in rows),
                    "median_p99_is_weight": median(r["p99_is_weight"] for r in rows),
                    "mean_clip01": mean(r["p_out_clip_01"] for r in rows),
                }
            )


def _plot_common_round_metric(
    series: dict[str, list[dict[str, Any]]],
    common_rounds: list[int],
    *,
    metric: str,
    ylabel: str,
    output_stem: str,
    abs_value: bool = False,
) -> None:
    colors = {"Tinker": TINKER, "Self-hosted": SELF, "Self-hosted+Quant": QUANT}
    markers = {"Tinker": "o", "Self-hosted": "s", "Self-hosted+Quant": "D"}

    fig, ax = plt.subplots(figsize=(4.2, 2.35))
    for name, rows in series.items():
        rows = sorted([r for r in rows if r["round"] in common_rounds], key=lambda r: r["round"])
        xs = [r["round"] for r in rows]
        ys = [abs(r[metric]) if abs_value else r[metric] for r in rows]
        ax.plot(
            xs,
            ys,
            marker=markers[name],
            markersize=3.6,
            linewidth=1.6,
            color=colors[name],
            markeredgecolor=TEXT if name == "Tinker" else "white",
            markeredgewidth=0.35,
            label=name,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_xlim(min(common_rounds) - 0.4, max(common_rounds) + 0.4)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.45, alpha=0.6, color=GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)

    fig.savefig(OUT_DIR / f"{output_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{output_stem}.png", dpi=300, bbox_inches="tight")


def main() -> None:
    series = load_all()
    common_rounds = _common_rounds(series)
    write_summary_csv(series, common_rounds)

    _plot_common_round_metric(
        series,
        common_rounds,
        metric="mean_abs_diff",
        ylabel="Mean |Δ logprob|",
        output_stem="mismatch_severity_comparison",
    )
    _plot_common_round_metric(
        series,
        common_rounds,
        metric="mean_cum_diff",
        ylabel="|Mean cumulative Δ logprob|",
        output_stem="mismatch_sequence_cumulative_comparison",
        abs_value=True,
    )
    print(f"Common rounds: {min(common_rounds)}-{max(common_rounds)}")
    print(OUT_DIR / "mismatch_sequence_cumulative_comparison.pdf")


if __name__ == "__main__":
    main()
