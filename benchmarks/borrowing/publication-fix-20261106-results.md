# Checkpoint priority and publication readiness

The fixes removed the long publication stall. They did not establish a throughput
benefit from borrowing on this workload: sampling became faster, but training still
waited for long borrowed requests to finish.

## Changes

HF checkpoints now carry their tenant identity, so the existing mutation queue can
prioritize them over unrelated backward work while preserving per-tenant ordering.
Previously, these checkpoints entered as global barriers. FSDP already supplied
this identity; its frozen-base sharding and group exclusion are unchanged.

Publishing a sampling session now waits only for dedicated replicas. The borrowed
replica loads the immutable adapter snapshot on its first admitted request, under
the same GPU reservation as generation. Concurrent requests load it once. If that
load fails, the request releases the training GPU and retries on dedicated capacity.
Removal drains accepted borrowed work before deleting its registration.

No new configuration, duration estimator, client-readiness signal or partial-rollout
support was added.

## Matched GPU comparison

One static-then-borrowing pair used Qwen3.5-4B and the same two GPUs, cached runtime,
model revision, seeds and workload as the [previous comparison](reactive-comparison-20261106-results.md).
Eight independently arriving tenants each ran eight four-turn coding trajectories,
trained their adapter, published a snapshot, and repeated. Dedicated capacity was
32; borrowing allowed four requests with an 8,192-token request allowance.

The observation window was 240–840 seconds. Latencies include queueing and follow
requests submitted in that window through completion. Training latency covers
backward and optimizer calls; publication is separate. Output throughput credits
requests completed within the window. Utilization is time-weighted NVML GPU busy
fraction, not compute efficiency.

| Metric | Static | Borrowing |
| --- | ---: | ---: |
| Trainer GPU utilization | 79.7% | 81.8% |
| Dedicated GPU utilization | 41.1% | 39.4% |
| Average of both GPUs | 60.4% | 60.6% |
| Completed output tokens/s | 501.5 | 487.2 |
| Completed sampling requests | 539 | 519 |
| Sampling p50 | 51.7 s | 43.3 s |
| Sampling p95 | 69.5 s | 65.1 s |
| Training p50 | 36.1 s | 67.2 s |
| Training p95 | 60.5 s | 101.4 s |
| Updates completed within window | 15 | 17 |
| Cumulative updates by 840 s | 16 | 17 |
| Borrowed admissions within window | 0 | 9 |

Borrowing reduced sampling p95 by 6.3%, but increased training p95 by 67.7% and
reduced output throughput by 2.8%. The two-update window difference includes a
boundary effect: static completed one update before the window, borrowing none.
Across the entire timed workloads, static completed 24 updates / 768 requests in
1,058.9 seconds; borrowing completed 23 updates / 736 requests in 1,053.9 seconds.
These are different amounts of work because new rounds stop at a time boundary.
One ordered pair does not establish a general speedup.

## What improved, and what remains

Across full timed runs, borrowing publication p95 fell from **181.5 to 32.0 seconds**
compared with the previous revision. Its longest checkpoint queue wait was 31.5
seconds and checkpoint execution took at most 0.63 seconds. Dedicated publication
no longer enters the training gate. Both arms had **zero seconds with an empty
dedicated sampler during the observation window**, versus the previous borrowing
run's roughly 154-second gap. Dedicated utilization recovered from 28.3% to 39.4%.

Borrowing served 17 requests in total, nine after the first update. All completed.
The remaining training cost is visible directly: three drains took **16.8, 49.8 and
28.9 seconds** (95.5 seconds total). Backward median execution was nearly identical,
30.40 seconds static versus 30.58 seconds borrowing, but backward queue p95 rose
from 31.7 to 53.3 seconds. Optimizer queue p95 was about 30.7 seconds in both arms.

The next issue is admission of long, non-preemptible generations when a client is
about to train. A token cap bounds request size, not training delay. Client-supplied
rollout/training readiness is a focused candidate for that problem; this experiment
does not test it. Readiness and partial rollout support remain separate follow-ups.

## Validation and artifacts

- CPU suite: **722 passed, 37 skipped**; final focused regressions: **22 passed**.
  Tests cover HF/FSDP checkpoint priority, controller-level publication during
  unrelated training, lazy-load cancellation, rollback, dedicated fallback and
  removal. Existing FSDP sharding tests passed. Type checking had zero errors;
  Ruff and the documentation build passed.
- GPU pair: identical initial adapter fingerprints, unchanged frozen weights,
  zero copied base parameters, every tenant trained, no failed requests and no
  output-limit truncation. Task success was 7/192 versus 9/184 trajectories; this
  is a systems comparison, not evidence of learning quality.
- This GPU run used HF. It does not measure FSDP performance. Including warmup and
  shutdown, the pair used two GPUs for **46.7 minutes**, or **1.56 allocated GPU-hours**.
  Both GPUs and the CPU tool service were released. No repeat run was launched.

[Utilization and latency](publication-fix-20261106-analysis/comparison-timelines.png)
([SVG](publication-fix-20261106-analysis/comparison-timelines.svg));
[client phases](publication-fix-20261106-analysis/comparison-client-phases.png)
([SVG](publication-fix-20261106-analysis/comparison-client-phases.svg));
[metrics](publication-fix-20261106-analysis/summary.json);
[validation](publication-fix-20261106-analysis/validation.json).
Axes are aligned between arms, with equal tenant row heights.

Raw telemetry, source snapshots and environment provenance are retained separately from this repository.

```bash
python benchmarks/borrowing/report_reactive_comparison.py \
  benchmarks/borrowing/runs/publication-fix-20261106 \
  benchmarks/borrowing/publication-fix-20261106-analysis
```
