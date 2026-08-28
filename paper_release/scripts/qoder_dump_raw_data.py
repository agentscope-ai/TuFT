#!/usr/bin/env python
"""Dump RAW experiment data to CSV/JSON for downstream processing by the user.

Produces, under exp/results/paper27/raw_data/:
  raw_throughput_t8.csv : per-baseline t8 throughput (steps/h, samples/h, wall, gpu util)
  raw_scaling.csv       : per (tenant, baseline) throughput for the scaling curves
  raw_slo.csv           : per (tenant, baseline) 3 SLOs: resp time(ms), logprob mismatch,
                          avg staleness
  raw_ablation.csv      : ablation component curve (bias, throughput) + baseline points
  raw_corrector.csv     : corrector bias-reduction (optimal & static traces)
Reads only completed arms (all_tenants_completed).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P17 = ROOT / "exp/results/paper17"
OUT = ROOT / "exp/results/paper27/raw_data"
OUT.mkdir(parents=True, exist_ok=True)

MODES = [
    ("optimal_8gpu", "optimal"),
    ("static_disagg_8gpu", "static"),
    ("serial_async_8gpu", "serial"),
    ("colocate_2copies_8gpu", "colocate"),
    ("unified_engine_8gpu", "unified"),
]
TENANTS = ["t1", "t2", "t4", "t8", "t16", "t32"]


def load(tenant, mode):
    f = P17 / tenant / mode / mode / "simulator_results.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    if not d.get("all_tenants_completed"):
        return None
    return d


def agg(d):
    per = d["per_tenant"]
    steps = sum(v.get("train_steps_completed", 0) for v in per.values())
    samples = sum(v.get("total_samples", 0) for v in per.values())
    wall = float(d.get("total_wall_clock_seconds") or 0)
    g = None
    return steps, samples, wall


def main():
    # throughput t8
    with open(OUT / "raw_throughput_t8.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["baseline", "steps_per_hour", "samples_per_hour", "wall_s", "tenants"])
        for mode, label in MODES:
            d = load("t8", mode)
            if not d:
                continue
            steps, samples, wall = agg(d)
            w.writerow([label, f"{steps/wall*3600:.1f}", f"{samples/wall*3600:.1f}",
                        f"{wall:.0f}", len(d["per_tenant"])])

    # scaling
    with open(OUT / "raw_scaling.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tenant", "baseline", "steps_per_hour"])
        for t in TENANTS:
            for mode, label in MODES:
                d = load(t, mode)
                if not d:
                    continue
                steps, samples, wall = agg(d)
                w.writerow([t, label, f"{steps/wall*3600:.1f}"])

    # SLO
    with open(OUT / "raw_slo.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tenant", "baseline", "resp_time_ms", "logprob_mismatch", "avg_staleness"])
        for t in TENANTS:
            for mode, label in MODES:
                d = load(t, mode)
                if not d:
                    continue
                lat = [v.get("mean_sample_latency_ms", 0) for v in d["per_tenant"].values()]
                mis = []
                sta = [v.get("mean_staleness", 0) for v in d["per_tenant"].values()]
                for v in d["per_tenant"].values():
                    lm = v.get("logprob_mismatch_per_step") or [0]
                    mis.append(sum(lm) / len(lm))
                n = len(lat) or 1
                w.writerow([t, label, f"{sum(lat)/n:.0f}", f"{sum(mis)/n:.5f}", f"{sum(sta)/n:.2f}"])

    # ablation (from sim json) + baseline points
    sim = json.load(open(P17 / "sim_ablation_1to1.json"))
    with open(OUT / "raw_ablation.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["point", "bias", "throughput_steps_per_hour", "kind"])
        for r in sim["rows"]:
            w.writerow([r["component"], f"{r['bias_proxy']:.5f}", f"{r['steps_per_hour']:.1f}", "component"])
        bias = json.load(open(ROOT / "exp/logprob_dump/bias_summary_4b.json"))
        thr = {"optimal_4b": 541.9, "static_4b": 283.9, "unified_4b": 39.9,
               "serial_4b": 184.3, "colocate_4b": 381.5}
        for tag, b in bias.items():
            w.writerow([tag, f"{b['x_axis_bias']:.5f}", f"{thr.get(tag, float('nan')):.1f}", "baseline_measured"])

    # corrector
    with open(OUT / "raw_corrector.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trace", "method", "mean_abs_seq_bias", "reduction_pct"])
        for tr in ["optimal_4b", "static_4b"]:
            f = ROOT / f"exp/results/paper27/corrector_experiment_{tr}.json"
            if not f.exists():
                continue
            d = json.load(open(f))
            w.writerow([tr, "uncorrected", f"{d['uncorrected_mean_abs_seq_bias']:.5f}", "0"])
            for m in ["token_correction", "sequence_correction", "adaptive_correction"]:
                w.writerow([tr, m, f"{d[m]['mean_abs_seq_bias']:.5f}", f"{d[m]['reduction_pct']:.1f}"])

    print("wrote:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
