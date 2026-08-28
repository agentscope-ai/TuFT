#!/usr/bin/env python3
"""Plot the flex GPU (GPU0) utilization timeline for an optimal-mode run.

Answers "is there room worth optimizing on the flex card": panel 1 is the raw
timeline against the sampling replicas, panel 2 is the idle-gap distribution
compared with the measured cost of one flex flip (a gap only pays for itself if
it is longer than the flip round trip).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


BUSY = 5.0          # % util counted as "doing work"
FLIP_ROUND_TRIP_S = 7.7   # measured: ~6.8s to sampling + ~0.9s back


def load(trace: Path):
    per = defaultdict(list)
    for r in csv.DictReader(trace.open()):
        per[int(r["gpu_index"])].append((float(r["time_s"]), float(r["gpu_util"])))
    for g in per:
        per[g].sort()
    return per


def gaps(vals):
    out, run = [], 0
    for v in vals:
        if v < BUSY:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def main(run_dir: str, out_png: str, label: str) -> None:
    per = load(Path(run_dir) / "gpu_trace.csv")
    # Anchor the window on the simulator's own wall clock: the trace also covers
    # server start-up, model load and vLLM warmup (all of which touch the GPUs),
    # and counting those as "idle flex card" is what produced the earlier bogus
    # utilization numbers. The window therefore ends at the last busy sample and
    # is exactly `total_wall_clock_seconds` long.
    wall = float(
        json.loads((Path(run_dir) / "simulator_results.json").read_text())[
            "total_wall_clock_seconds"
        ]
    )
    t_end = max(t for g in per for t, v in per[g] if v >= BUSY)
    t0 = t_end - wall
    print(f"trace spans {max(t for g in per for t, v in per[g]):.0f}s; "
          f"workload window = last {wall:.0f}s before the final busy sample")

    g0 = [(t - t0, v) for t, v in per[0] if t0 <= t <= t_end]
    samp_ids = [g for g in per if g != 0]
    # mean across the fixed sampling replicas, aligned on GPU0's timestamps
    samp = []
    for t, _ in g0:
        vs = [
            min(per[g], key=lambda tv: abs(tv[0] - (t + t0)))[1]
            for g in samp_ids
        ]
        samp.append(sum(vs) / len(vs))

    ts = [t for t, _ in g0]
    v0 = [v for _, v in g0]

    # Phase boundaries. The simulator's wall clock starts before any GPU work
    # (it builds adapters, tokenizers and datasets on the CPU first), and the RL
    # loop cannot train until the first rollout batch exists, so flex idle must
    # be attributed per phase instead of lumped together.
    def sustained(series, need=5):
        """First time of `need` consecutive busy samples (ignores warmup blips)."""
        run = 0
        for t, v in series:
            run = run + 1 if v >= BUSY else 0
            if run >= need:
                return t - need
        return series[0][0] if series else 0.0

    t_samp = sustained(list(zip(ts, samp, strict=False)))
    t_train = sustained([(t, v) for t, v in g0 if t >= t_samp])
    steady = [v for t, v in g0 if t >= t_train]
    sbusy = [v for v in steady if v >= BUSY]
    sgaps = sorted(gaps(steady))
    sfill = [g for g in sgaps if g >= FLIP_ROUND_TRIP_S]
    print("--- phases ---")
    print(f"  [0,{t_samp:.0f}s] simulator CPU-only prep: all GPUs idle "
          f"({t_samp / len(v0) * 100:.0f}% of the window)")
    print(f"  [{t_samp:.0f},{t_train:.0f}s] cold-start sampling wave: "
          f"flex has no batch to train yet ({(t_train - t_samp):.0f}s)")
    print(f"  [{t_train:.0f},{ts[-1]:.0f}s] steady state ({len(steady)}s): "
          f"GPU0 busy_frac={len(sbusy) / max(len(steady), 1) * 100:.1f}% "
          f"active_mean={sum(sbusy) / max(len(sbusy), 1):.1f}% "
          f"time_mean={sum(steady) / max(len(steady), 1):.1f}%")
    print(f"  steady-state idle: {sum(sgaps)}s in {len(sgaps)} gaps "
          f"(max={max(sgaps) if sgaps else 0}s); gaps >= {FLIP_ROUND_TRIP_S}s: "
          f"n={len(sfill)} -> at most "
          f"{sum(max(0, g - FLIP_ROUND_TRIP_S) for g in sfill):.0f}s recoverable "
          f"({sum(max(0, g - FLIP_ROUND_TRIP_S) for g in sfill) / max(len(steady), 1) * 100:.1f}%"
          f" of steady state)")
    busy = [v for v in v0 if v >= BUSY]
    gp = sorted(gaps(v0))
    fillable = [g for g in gp if g >= FLIP_ROUND_TRIP_S]

    print(f"window {t_end - t0:.0f}s (workload only, from first busy sample)")
    print(f"GPU0 busy_frac={len(busy) / len(v0) * 100:.1f}%  "
          f"active_mean={sum(busy) / max(len(busy), 1):.1f}%  "
          f"time_mean={sum(v0) / len(v0):.1f}%")
    print(f"sampling replicas time_mean={sum(samp) / len(samp):.1f}%")
    print(f"idle gaps: n={len(gp)} total={sum(gp)}s median={np.median(gp):.0f}s "
          f"p90={gp[int(len(gp) * 0.9)] if gp else 0}s max={max(gp) if gp else 0}s")
    print(f"gaps >= flip round trip ({FLIP_ROUND_TRIP_S}s): n={len(fillable)} "
          f"seconds={sum(fillable)}s -> at most "
          f"{sum(max(0, g - FLIP_ROUND_TRIP_S) for g in fillable):.0f}s recoverable "
          f"({sum(max(0, g - FLIP_ROUND_TRIP_S) for g in fillable) / len(v0) * 100:.1f}%"
          f" of the window)")

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax.fill_between(ts, samp, color="#93BFCF", alpha=.55,
                    label="fixed sampling GPU1-3 (mean)")
    ax.plot(ts, v0, color="#112D4E", lw=.9, label="flex GPU0 (training)")
    ax.axhline(sum(v0) / len(v0), color="#AA96DA", ls="--", lw=1.2,
               label=f"GPU0 time-mean {sum(v0) / len(v0):.0f}%")
    ax.set_ylabel("GPU utilization (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, ts[-1])
    ax.set_title(f"Flex GPU (GPU0) utilization over time — {label}\n"
                 f"busy {len(busy) / len(v0) * 100:.0f}% of the window, "
                 f"{sum(busy) / max(len(busy), 1):.0f}% while busy")
    for x, txt in ((t_samp / 2, "simulator CPU-only prep\n(all GPUs idle)"),
                   ((t_samp + t_train) / 2, "cold-start\nsampling wave"),
                   ((t_train + ts[-1]) / 2, "steady state: flex card saturated")):
        ax.annotate(txt, (x, 52), ha="center", va="center", fontsize=8,
                    color="#112D4E",
                    bbox=dict(boxstyle="round", fc="white", ec="#DBE2EF", alpha=.85))
    for x in (t_samp, t_train):
        ax.axvline(x, color="#6096B4", ls=":", lw=1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=.25)

    if gp:
        bins = [0, 1, 2, 3, 5, 8, 13, 21, 34, max(35, max(gp) + 1)]
        counts, _ = np.histogram(gp, bins=bins)
        secs = [sum(g for g in gp if lo <= g < hi)
                for lo, hi in zip(bins, bins[1:], strict=False)]
        x = np.arange(len(counts))
        ax2.bar(x, secs, color="#3F72AF", label="idle seconds in gaps of this size")
        for i, (c, sc) in enumerate(zip(counts, secs, strict=False)):
            if c:
                ax2.text(i, sc, f"n={c}", ha="center", va="bottom", fontsize=7)
        ax2.axvline(3.5, color="#D7263D", ls="--", lw=1.2,
                    label=f"flip round trip {FLIP_ROUND_TRIP_S}s"
                           " (gaps left of this are unfillable)")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{lo}-{hi}s" for lo, hi in zip(bins, bins[1:], strict=False)],
                            fontsize=8)
        ax2.set_ylabel("idle seconds")
        ax2.set_xlabel("GPU0 idle-gap length")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print("wrote", out_png)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
