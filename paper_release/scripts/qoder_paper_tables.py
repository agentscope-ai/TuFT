#!/usr/bin/env python
"""Aggregate the paper27 data tables (throughput main table, scaling matrix, bias).

Reads the completed simulator arms under exp/results/paper17 (skipping parked
``_``-prefixed dirs) and the logprob-dump bias summary, then emits:
  * a scaling matrix  rows=tenant(t1..t32) x cols=deployment, cell=steps/hour
  * a t8 main table   (all deployments at t8)
  * a sequence-logprob-bias table (per framework)
Output: markdown to stdout + JSON to exp/results/paper27/paper_tables.json.

Only COMPLETE arms (all_tenants_completed) are reported as numbers; incomplete
arms show as '-'.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P17 = ROOT / "exp/results/paper17"
BIAS = ROOT / "exp/logprob_dump/bias_summary_4b.json"
OUT = ROOT / "exp/results/paper27/paper_tables.json"

MODE_ORDER = [
    "optimal_8gpu",
    "static_disagg_8gpu",
    "serial_async_8gpu",
    "colocate_2copies_8gpu",
    "unified_engine_8gpu",
]
TENANT_ORDER = ["t1", "t2", "t4", "t8", "t16", "t32"]


def _load(p: Path) -> dict:
    try:
        return json.load(open(p))
    except Exception:
        return {}


def _steps_per_hour(sim: dict) -> float | None:
    per = sim.get("per_tenant", {})
    if not per or not sim.get("all_tenants_completed"):
        return None
    steps = sum(v.get("train_steps_completed", 0) for v in per.values())
    wall = float(sim.get("total_wall_clock_seconds") or 0.0)
    if wall <= 0 or steps <= 0:
        return None
    return steps / wall * 3600


def collect() -> dict:
    """Map (tenant, mode) -> steps/hour for complete arms."""
    table: dict[tuple[str, str], float] = {}
    for sim_path in sorted(P17.rglob("simulator_results.json")):
        if any(part.startswith("_") for part in sim_path.relative_to(P17).parts):
            continue
        rel = sim_path.parent.relative_to(P17)
        parts = rel.parts
        if len(parts) < 2:
            continue
        tenant, mode = parts[0], parts[1]
        # normalize unified campaign variant
        if mode.startswith("unified_engine"):
            mode = "unified_engine_8gpu"
        if tenant not in TENANT_ORDER or mode not in MODE_ORDER:
            continue
        sph = _steps_per_hour(_load(sim_path))
        if sph is not None:
            key = (tenant, mode)
            table[key] = max(table.get(key, 0.0), sph)
    return table


def main() -> None:
    table = collect()

    # scaling matrix
    print("## Scaling matrix (steps/hour, complete arms only)")
    header = "| tenant | " + " | ".join(m.replace("_8gpu", "") for m in MODE_ORDER) + " |"
    print(header)
    print("|" + "---|" * (len(MODE_ORDER) + 1))
    for t in TENANT_ORDER:
        cells = []
        for m in MODE_ORDER:
            v = table.get((t, m))
            cells.append(f"{v:.0f}" if v else "-")
        print(f"| {t} | " + " | ".join(cells) + " |")

    # t8 main table
    print("\n## t8 main table")
    print("| deployment | steps/hour |")
    print("|---|---|")
    for m in MODE_ORDER:
        v = table.get(("t8", m))
        print(f"| {m.replace('_8gpu', '')} | {v:.1f} |" if v else f"| {m.replace('_8gpu','')} | - |")

    # bias table
    print("\n## sequence-logprob bias (mean |per-seq (sampling-training)|)")
    print("| framework | mean|bias| | rows |")
    print("|---|---|---|")
    if BIAS.exists():
        b = json.load(open(BIAS))
        for tag, r in b.items():
            print(f"| {tag} | {r['x_axis_bias']:.5f} | {r['rows_usable']} |")

    payload = {
        "scaling": {f"{t}/{m}": round(v, 1) for (t, m), v in sorted(table.items())},
        "t8": {m: round(table[("t8", m)], 1) for m in MODE_ORDER if ("t8", m) in table},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
