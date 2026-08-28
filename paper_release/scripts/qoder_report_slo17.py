#!/usr/bin/env python3
"""Extract the paper's SLO table from a campaign directory.

Three axes, all from data the simulator already records per tenant:

  response    total_training_seconds, total_sampling_seconds and
              total_sync_weights_seconds are reported separately rather than as
              one number, since the split is what shows where an arm spends its
              wall clock.
  freshness   staleness_per_step carries, for every training step, the
              distribution of weight-version lag over the samples that step
              consumed. The mean over all steps is the average version a
              training step trains on; p95 and max come from the same
              distribution. mean_staleness is the simulator's own average and is
              printed alongside as a cross-check.
  latency     mean_sample_latency_ms per tenant, aggregated sample-weighted.

Staleness tolerance is per tenant (0..3 in these workloads), so the fraction of
consumed samples that exceeded their own tenant's limit is reported too - that is
the SLO violation rate, not a global threshold.

Usage:
    python scripts/qoder_report_slo17.py exp/results/paper17/t8
    python scripts/qoder_report_slo17.py exp/results/paper17          # all tags
"""

from __future__ import annotations

import glob
import json
import os
import sys


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def summarize(results_path: str) -> dict | None:
    try:
        with open(results_path) as fh:
            doc = json.load(fh)
    except Exception:
        return None
    per = doc.get("per_tenant") or {}
    if not per:
        return None

    wall = float(doc.get("total_wall_clock_seconds") or 0.0)
    train_s = sampling_s = sync_s = 0.0
    steps = samples = 0
    lat_weighted = 0.0
    lags: list[int] = []
    over = 0
    consumed = 0
    mean_stale_weighted = 0.0

    for tenant in per.values():
        train_s += float(tenant.get("total_training_seconds") or 0.0)
        sampling_s += float(tenant.get("total_sampling_seconds") or 0.0)
        sync_s += float(tenant.get("total_sync_weights_seconds") or 0.0)
        n = int(tenant.get("total_samples") or 0)
        steps += int(tenant.get("train_steps_completed") or 0)
        samples += n
        lat_weighted += float(tenant.get("mean_sample_latency_ms") or 0.0) * n
        mean_stale_weighted += float(tenant.get("mean_staleness") or 0.0) * n

        limit = tenant.get("staleness_limit")
        for entry in tenant.get("staleness_per_step") or []:
            for lag_str, count in (entry.get("distribution") or {}).items():
                lag = int(lag_str)
                count = int(count)
                lags.extend([lag] * count)
                consumed += count
                if limit is not None and lag > int(limit):
                    over += count

    lags.sort()
    return {
        "tenants": len(per),
        "wall_s": wall,
        "steps": steps,
        "samples": samples,
        "steps_per_h": steps / wall * 3600 if wall else 0.0,
        "samples_per_h": samples / wall * 3600 if wall else 0.0,
        # response, split by phase; these are summed over tenants running
        # concurrently, so they exceed wall clock and are meant to be read as a
        # ratio between phases rather than as a timeline.
        "training_s": train_s,
        "sampling_s": sampling_s,
        "sync_weights_s": sync_s,
        "phase_ratio_sampling": sampling_s / (train_s + sampling_s + sync_s)
        if (train_s + sampling_s + sync_s)
        else 0.0,
        "mean_sample_latency_ms": lat_weighted / samples if samples else 0.0,
        # freshness
        "version_lag_mean": sum(lags) / len(lags) if lags else 0.0,
        "version_lag_p95": _pct(lags, 0.95),
        "version_lag_max": lags[-1] if lags else 0,
        "mean_staleness_reported": mean_stale_weighted / samples if samples else 0.0,
        "consumed_samples": consumed,
        "over_own_limit_frac": over / consumed if consumed else 0.0,
        "completed": bool(doc.get("all_tenants_completed")),
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    root = sys.argv[1].rstrip("/")

    found = sorted(glob.glob(f"{root}/*/*/simulator_results.json")) + sorted(
        glob.glob(f"{root}/*/*/*/simulator_results.json")
    )
    seen = set()
    rows = []
    for path in found:
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        s = summarize(path)
        if s is None:
            continue
        parts = path.split(os.sep)
        # .../<root>/<tag>/<arm>/<arm>/simulator_results.json
        arm = parts[-2]
        tag = parts[-4] if len(parts) >= 4 else ""
        s["arm"] = arm
        s["tag"] = tag if tag != os.path.basename(root) else ""
        rows.append(s)

    if not rows:
        print(f"no completed runs under {root}")
        return

    hdr = (
        f"{'tag':>5s} {'arm':22s} {'wall':>8s} {'steps/h':>8s} "
        f"{'train_s':>8s} {'sample_s':>9s} {'sync_s':>7s} {'smp%':>5s} "
        f"{'lat_ms':>8s} {'lag_avg':>8s} {'p95':>4s} {'max':>4s} {'over%':>6s} {'ok':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['tag']:>5s} {r['arm']:22s} {r['wall_s']:8.1f} {r['steps_per_h']:8.1f} "
            f"{r['training_s']:8.1f} {r['sampling_s']:9.1f} {r['sync_weights_s']:7.1f} "
            f"{r['phase_ratio_sampling']*100:5.1f} {r['mean_sample_latency_ms']:8.1f} "
            f"{r['version_lag_mean']:8.2f} {r['version_lag_p95']:4.0f} {r['version_lag_max']:4d} "
            f"{r['over_own_limit_frac']*100:6.1f} {str(r['completed']):>5s}"
        )

    out = os.path.join(root, "slo17.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
