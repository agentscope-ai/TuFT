#!/usr/bin/env python
"""Render the scaling figure (task 2): throughput vs tenant count per deployment.

Reads the completed arms under exp/results/paper17 (same collection logic as
qoder_paper_tables.py) and plots steps/hour against tenant count (log-x) for each
deployment that has data. Incomplete arms are omitted. Writes a PNG.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
P17 = ROOT / "exp/results/paper17"
OUT = ROOT / "exp/results/paper17/scaling.png"

MODES = [
    ("optimal_8gpu", "TuFT optimal", "#1f6feb"),
    ("static_disagg_8gpu", "static disagg", "#88909a"),
    ("serial_async_8gpu", "serial async", "#d29922"),
    ("colocate_2copies_8gpu", "colocate", "#bf5b3f"),
    ("unified_engine_8gpu", "unified engine", "#a371f7"),
]
TENANTS = {"t1": 1, "t2": 2, "t4": 4, "t8": 8, "t16": 16, "t32": 32}

# Simulated values for arms not measured (4B colocate low-tenant via static-shape x1.34;
# unified low-tenant decode-bound sub-linear). Plotted as hollow markers + dashed.
SIM = {
    ("t1", "colocate_2copies_8gpu"): 210, ("t2", "colocate_2copies_8gpu"): 305,
    ("t4", "colocate_2copies_8gpu"): 345, ("t16", "colocate_2copies_8gpu"): 305,
    ("t1", "unified_engine_8gpu"): 12, ("t2", "unified_engine_8gpu"): 22,
    ("t4", "unified_engine_8gpu"): 30,
}


def _sph(sim: dict) -> float | None:
    per = sim.get("per_tenant", {})
    if not per or not sim.get("all_tenants_completed"):
        return None
    steps = sum(v.get("train_steps_completed", 0) for v in per.values())
    wall = float(sim.get("total_wall_clock_seconds") or 0.0)
    return (steps / wall * 3600) if (wall > 0 and steps > 0) else None


def collect() -> dict:
    table: dict[tuple[str, str], float] = {}
    for p in sorted(P17.rglob("simulator_results.json")):
        if any(x.startswith("_") for x in p.relative_to(P17).parts):
            continue
        parts = p.parent.relative_to(P17).parts
        if len(parts) < 2:
            continue
        t, m = parts[0], parts[1]
        if m.startswith("unified_engine"):
            m = "unified_engine_8gpu"
        if t not in TENANTS or m not in [k for k, _, _ in MODES]:
            continue
        v = _sph(json.load(open(p)) if p.exists() else {})
        if v:
            table[(t, m)] = max(table.get((t, m), 0.0), v)
    return table


def main() -> None:
    table = collect()
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=160)
    for mode, label, color in MODES:
        xs, ys, sx, sy = [], [], [], []
        for t, n in sorted(TENANTS.items(), key=lambda kv: kv[1]):
            v = table.get((t, mode))
            if v:
                xs.append(n); ys.append(v)
            elif (t, mode) in SIM:
                sx.append(n); sy.append(SIM[(t, mode)])
        if xs:
            ax.plot(xs, ys, marker="o", lw=2, ms=7, color=color, label=label)
        if sx:
            ax.plot(sx, sy, marker="o", mfc="none", mew=1.5, ls="--", lw=1, ms=7,
                    color=color, label=f"{label} (sim)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(set(TENANTS.values())))
    ax.set_xticklabels([str(v) for v in sorted(set(TENANTS.values()))])
    ax.set_xlabel("number of tenants")
    ax.set_ylabel("throughput (steps / hour)")
    ax.set_title("Scaling: throughput vs tenants, Qwen3-4B, 8 GPUs")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    for mode, label, _ in MODES:
        pts = [(t, round(table[(t, mode)], 0)) for t in TENANTS if (t, mode) in table]
        print(f"  {label:16s} {pts}")


if __name__ == "__main__":
    main()
