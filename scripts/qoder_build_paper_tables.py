#!/usr/bin/env python3
"""Build normalized TuFT campaign tables from completed simulator runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "phase", "steps_comparable", "arm", "complete", "tenants", "steps",
    "samples", "samples_per_step", "wall_s", "samples_per_hour",
    "steps_per_hour", "gpu_util_avg", "l1_a0_merged", "l1_unknown_version",
    "requests_routed", "l2_hold_s", "l2_dispatch_runs", "holds_armed",
    "delay_hold_s", "duty_to_sampling", "duty_to_fixed", "flex_switches",
    "train_lane_util", "conversions", "fixed_groups", "flex_groups",
    "active_fixed_replicas",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def rounded(value: float, digits: int = 1) -> float:
    return round(float(value), digits)


def collect(root: Path) -> list[dict]:
    rows: list[dict] = []
    for sim_path in sorted(root.glob("**/simulator_results.json")):
        run_dir = sim_path.parent
        rel = run_dir.relative_to(root)
        if len(rel.parts) < 2:
            continue
        phase = rel.parts[0]
        arm = "/".join(rel.parts[1:])
        sim = load(sim_path)
        metrics_path = run_dir / "tuft_eval_metrics.json"
        metrics = load(metrics_path) if metrics_path.exists() else {}
        per_tenant = sim.get("per_tenant", {})
        steps = sum(int(v.get("train_steps_completed", v.get("steps", 0)) or 0) for v in per_tenant.values())
        samples = sum(int(v.get("total_samples", v.get("samples", 0)) or 0) for v in per_tenant.values())
        wall_s = float(sim.get("total_wall_clock_seconds", 0) or 0)
        scheduler = metrics.get("rl_scheduler", {})
        pipeline = metrics.get("sampling_pipeline", {})
        breakdown = metrics.get("iteration_breakdown", {})
        gpu = metrics.get("gpu_utilization", {})
        composition = scheduler.get("composition", {})
        routing_all = metrics.get("sampling_routing", {})
        routing = next(iter(routing_all.values()), {}) if isinstance(routing_all, dict) else {}
        rows.append({
            "phase": phase,
            "steps_comparable": False,
            "arm": arm,
            "complete": bool(sim.get("all_tenants_completed", not sim.get("incomplete_tenants"))),
            "tenants": len(per_tenant),
            "steps": steps,
            "samples": samples,
            "samples_per_step": rounded(samples / steps) if steps else 0.0,
            "wall_s": rounded(wall_s),
            "samples_per_hour": rounded(samples / wall_s * 3600) if wall_s else 0.0,
            "steps_per_hour": rounded(steps / wall_s * 3600) if wall_s else 0.0,
            "gpu_util_avg": rounded(gpu.get("gpu_util_avg", 0)),
            "l1_a0_merged": int(pipeline.get("a0_merged_requests", 0) or 0),
            "l1_unknown_version": int(pipeline.get("unknown_version_requests", 0) or 0),
            "requests_routed": int(pipeline.get("requests_routed", 0) or 0),
            "l2_hold_s": rounded(pipeline.get("l2_coalesce_hold_s", 0)),
            "l2_dispatch_runs": int(pipeline.get("l2_dispatch_runs", 0) or 0),
            "holds_armed": int(scheduler.get("holds_armed", 0) or 0),
            "delay_hold_s": rounded(scheduler.get("delay_hold_total_s", 0)),
            "duty_to_sampling": int(scheduler.get("duty_cycle_switches_to_sampling", 0) or 0),
            "duty_to_fixed": int(scheduler.get("duty_cycle_switches_to_fixed", 0) or 0),
            "flex_switches": rounded(breakdown.get("flex_switches", 0)),
            "train_lane_util": rounded(scheduler.get("train_lane_utilization", 0), 3),
            "conversions": int(scheduler.get("conversion_count", 0) or 0),
            "fixed_groups": int(composition.get("fixed_groups", 0) or 0),
            "flex_groups": int(composition.get("flex_groups", 0) or 0),
            "active_fixed_replicas": int(routing.get("active_fixed_replicas", 0) or 0),
        })
    for phase in {row["phase"] for row in rows}:
        phase_rows = [row for row in rows if row["phase"] == phase and row["complete"]]
        comparable = bool(phase_rows) and len({row["steps"] for row in phase_rows}) == 1
        for row in phase_rows:
            row["steps_comparable"] = comparable
    return rows


def render_text(rows: list[dict]) -> str:
    lines: list[str] = []
    for phase in sorted({row["phase"] for row in rows}):
        group = [row for row in rows if row["phase"] == phase]
        lines.extend(["", f"== {phase}", "arm                                      ok   wall_s  steps/h   samples/h smp/step  gpu%   lane"])
        base = next((row["steps_per_hour"] for row in group if row["complete"]), 0)
        for row in group:
            delta = ((row["steps_per_hour"] / base - 1) * 100) if base else 0
            lines.append(
                f"{row['arm']:<40} {str(row['complete']):<5} {row['wall_s']:7.1f} "
                f"{row['steps_per_hour']:8.1f} {row['samples_per_hour']:11.1f} "
                f"{row['samples_per_step']:8.1f} {row['gpu_util_avg']:5.1f} {row['train_lane_util']:6.3f}  vs_base={delta:+.1f}%"
            )
        lines.append("  design points fired:")
        for row in group:
            lines.append(
                f"    {row['arm']:<36} L1={row['l1_a0_merged']:5d} unk={row['l1_unknown_version']:5d} "
                f"L2hold={row['l2_hold_s']:7.1f}s holds={row['holds_armed']:4d} "
                f"duty>smp={row['duty_to_sampling']:4d} conv={row['conversions']:3d} "
                f"replicas={row['active_fixed_replicas']}"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = collect(root)
    (root / "paper_tables.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (root / "paper_tables.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    text = render_text(rows)
    (root / "paper_tables.txt").write_text(text)
    print(text, end="")
    print(f"rows={len(rows)}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
