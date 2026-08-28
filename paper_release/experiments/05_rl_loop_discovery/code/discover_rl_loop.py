"""Discover periodic RL loops from request-level timestamps.

Input data contains request arrivals from multiple tenants.  Users do not
explicitly declare RL loops, so this script infers each tenant's training loop
from observed training request timestamps and evaluates how quickly the next
training request becomes predictable.

Outputs:
  results/loop_discovery_summary.csv
  results/loop_discovery_summary.json
  results/prediction_convergence.{pdf,png}
  results/tenant_periods.{pdf,png}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


@dataclass
class TenantLoopSummary:
    tenant_id: str
    group: str
    task: str
    request_rate: float
    expected_period_s: float
    buffer_size: int
    n_sampling_requests: int
    n_training_requests: int
    median_loop_period_s: float
    mean_loop_period_s: float
    p25_loop_period_s: float
    p75_loop_period_s: float
    cv_loop_period: float
    iqr_over_median: float
    mae_after_1_loop_s: float
    mae_after_2_loops_s: float
    mae_after_3_loops_s: float
    mae_after_5_loops_s: float
    relative_mae_after_3_loops: float


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    frac = pos - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def _std(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _tenant_group(tenant_id: str) -> str:
    return "Group" if tenant_id.endswith("-shared") else "Personal"


def load_requests(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_tenant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_tenant[row["tenant_id"]].append(row)
    for rows in by_tenant.values():
        rows.sort(key=lambda r: r["monotonic_ns"])
    return dict(by_tenant)


def _training_times_s(rows: list[dict[str, Any]]) -> list[float]:
    times = [r["monotonic_ns"] for r in rows if r.get("request_kind") == "training"]
    if not times:
        return []
    t0 = times[0]
    return [(t - t0) / 1e9 for t in times]


def _prediction_errors(times_s: list[float], warmup_intervals: int) -> list[float]:
    """Predict next training arrival using rolling median of observed intervals.

    ``warmup_intervals`` means the first N inter-arrival intervals are only used
    for loop discovery.  Evaluation starts from the next training request.
    """

    if len(times_s) < warmup_intervals + 2:
        return []
    intervals = [times_s[i] - times_s[i - 1] for i in range(1, len(times_s))]
    errors: list[float] = []
    for next_idx in range(warmup_intervals + 1, len(times_s)):
        observed_intervals = intervals[: next_idx - 1]
        if len(observed_intervals) < warmup_intervals:
            continue
        period_hat = median(observed_intervals)
        pred = times_s[next_idx - 1] + period_hat
        errors.append(abs(pred - times_s[next_idx]))
    return errors


def summarize_tenant(tenant_id: str, rows: list[dict[str, Any]]) -> TenantLoopSummary | None:
    training_times = _training_times_s(rows)
    if len(training_times) < 3:
        return None

    intervals = [training_times[i] - training_times[i - 1] for i in range(1, len(training_times))]
    med_period = median(intervals)
    mean_period = mean(intervals)
    p25 = _quantile(intervals, 0.25)
    p75 = _quantile(intervals, 0.75)
    cv = _std(intervals) / mean_period if mean_period > 0 else 0.0
    iqr_over_med = (p75 - p25) / med_period if med_period > 0 else 0.0

    first = rows[0]
    sampling_count = sum(1 for r in rows if r.get("request_kind") == "sampling")
    training_count = sum(1 for r in rows if r.get("request_kind") == "training")

    def mae_after(warmup: int) -> float:
        errs = _prediction_errors(training_times, warmup)
        return mean(errs) if errs else 0.0

    mae3 = mae_after(3)

    return TenantLoopSummary(
        tenant_id=tenant_id,
        group=_tenant_group(tenant_id),
        task=str(first.get("task", "")),
        request_rate=float(first.get("request_rate", 0.0)),
        expected_period_s=float(first.get("expected_period_s", 0.0)),
        buffer_size=int(first.get("buffer_size", 0)),
        n_sampling_requests=sampling_count,
        n_training_requests=training_count,
        median_loop_period_s=med_period,
        mean_loop_period_s=mean_period,
        p25_loop_period_s=p25,
        p75_loop_period_s=p75,
        cv_loop_period=cv,
        iqr_over_median=iqr_over_med,
        mae_after_1_loop_s=mae_after(1),
        mae_after_2_loops_s=mae_after(2),
        mae_after_3_loops_s=mae3,
        mae_after_5_loops_s=mae_after(5),
        relative_mae_after_3_loops=mae3 / med_period if med_period > 0 else 0.0,
    )


def summarize_all(by_tenant: dict[str, list[dict[str, Any]]]) -> list[TenantLoopSummary]:
    summaries: list[TenantLoopSummary] = []
    for tenant_id, rows in sorted(by_tenant.items()):
        summary = summarize_tenant(tenant_id, rows)
        if summary is not None:
            summaries.append(summary)
    return summaries


def write_outputs(output_dir: Path, summaries: list[TenantLoopSummary]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(TenantLoopSummary.__dataclass_fields__.keys())
    with (output_dir / "loop_discovery_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))
    with (output_dir / "loop_discovery_summary.json").open("w") as f:
        json.dump([asdict(s) for s in summaries], f, indent=2)


def plot_prediction_convergence(output_dir: Path, by_tenant: dict[str, list[dict[str, Any]]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    warmups = [1, 2, 3, 4, 5, 6, 8]
    group_values: dict[str, list[float]] = {"Personal": [], "Group": []}
    per_warmup: dict[str, list[float]] = {"Personal": [], "Group": []}

    fig, ax = plt.subplots(figsize=(3.6, 2.25))
    colors = {"Personal": "#3F72AF", "Group": "#112D4E"}

    for group in ["Personal", "Group"]:
        means: list[float] = []
        for w in warmups:
            rel_errors: list[float] = []
            for tenant_id, rows in by_tenant.items():
                if _tenant_group(tenant_id) != group:
                    continue
                times = _training_times_s(rows)
                if len(times) < w + 2:
                    continue
                intervals = [times[i] - times[i - 1] for i in range(1, len(times))]
                med_period = median(intervals)
                errs = _prediction_errors(times, w)
                if errs and med_period > 0:
                    rel_errors.append(mean(errs) / med_period)
            means.append(mean(rel_errors) if rel_errors else 0.0)
            if w == 3:
                per_warmup[group] = rel_errors
        ax.plot(warmups, means, marker="o", linewidth=1.8, markersize=4.0, color=colors[group], label=group)
        group_values[group] = means

    ax.set_xlabel("Observed training intervals")
    ax.set_ylabel("Relative prediction error")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_convergence.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "prediction_convergence.png", dpi=300, bbox_inches="tight")


def plot_tenant_periods(output_dir: Path, summaries: list[TenantLoopSummary]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows = sorted(summaries, key=lambda s: s.median_loop_period_s)
    labels = [s.tenant_id for s in rows]
    y = list(range(len(rows)))
    med = [s.median_loop_period_s for s in rows]
    xerr_low = [max(0.0, s.median_loop_period_s - s.p25_loop_period_s) for s in rows]
    xerr_high = [max(0.0, s.p75_loop_period_s - s.median_loop_period_s) for s in rows]
    colors = ["#112D4E" if s.group == "Group" else "#3F72AF" for s in rows]

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.barh(y, med, xerr=[xerr_low, xerr_high], color=colors, alpha=0.9, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xlabel("Training-loop period (s), median with IQR")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "tenant_periods.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "tenant_periods.png", dpi=300, bbox_inches="tight")


def print_summary(summaries: list[TenantLoopSummary]) -> None:
    print("tenant,group,n_train,median_period_s,iqr/median,mae3_s,rel_mae3")
    for s in summaries:
        print(
            f"{s.tenant_id},{s.group},{s.n_training_requests},"
            f"{s.median_loop_period_s:.2f},{s.iqr_over_median:.3f},"
            f"{s.mae_after_3_loops_s:.2f},{s.relative_mae_after_3_loops:.3f}"
        )
    for group in ["Personal", "Group"]:
        rows = [s for s in summaries if s.group == group]
        if not rows:
            continue
        print(
            f"\n[{group}] tenants={len(rows)}, "
            f"median_period={median([s.median_loop_period_s for s in rows]):.2f}s, "
            f"mean_iqr/median={mean([s.iqr_over_median for s in rows]):.3f}, "
            f"mean_mae3={mean([s.mae_after_3_loops_s for s in rows]):.2f}s, "
            f"mean_rel_mae3={mean([s.relative_mae_after_3_loops for s in rows]):.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover periodic RL loops from request timestamps")
    parser.add_argument("--input", type=Path, default=Path("data/requests.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_tenant = load_requests(args.input)
    summaries = summarize_all(by_tenant)
    write_outputs(args.output_dir, summaries)
    plot_prediction_convergence(args.output_dir, by_tenant)
    plot_tenant_periods(args.output_dir, summaries)
    print_summary(summaries)
    print(f"\nWrote outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
