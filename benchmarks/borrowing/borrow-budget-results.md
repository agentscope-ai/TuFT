# Borrowing-budget comparison — 2026-11-04

**Learned borrowing now occurs, but it does not persist across rollout waves.**
The 35-second budget served eight probes and four learned requests on the shared
GPU; the 5-second control served four probes. Neither arm borrowed after its first
training update. Training did not wait materially for sampling to drain, but this
run did not exercise a borrowed batch overlapping the arrival of training work.

| Measurement | 5-second budget | 35-second budget |
| --- | ---: | ---: |
| Completion time | 576.53 s | 557.47 s |
| Sampling latency p95 | 84.66 s | 90.82 s |
| Training-update latency p95 | 76.73 s | 65.19 s |
| Optimizer queue wait p95 | 48.42 s | 40.64 s |
| Longest sampling drain wait | <1 ms | <1 ms |
| Shared requests: probes / learned | 4 / 0 | 8 / 4 |
| Shared admissions after first update | 0 | 0 |
| Trainer GPU active utilization | 30.47% | 33.47% |
| Dedicated GPU active utilization | 29.60% | 30.17% |
| Updates per allocated GPU-hour, timed workload | 99.91 | 103.32 |
| Generated tokens | 216,394 | 219,521 |

Completion was 3.3% faster, training-update p95 15.0% lower, and sampling p95
7.3% higher in this single ordered pair. These observations do not establish a
general speedup. Outputs differed by 1.45%; sampling and tool feedback make the
two workloads similar, not identical. GPU-active utilization is sampled NVML
activity, not FLOP utilization. Startup and warmup are excluded from the timed
metrics but included in the cost estimate.

## Setup and budget choice

Qwen3.5-4B on two H100s, using the HF backend: one shared trainer/sampler and one
dedicated sampler. Both arms used the current scheduler, 16 nonzero rank-8 LoRA
adapters, 32 dedicated sequences, 16 adapters per batch, and 16 GiB KV capacity.
Shared sampling remained capped at four concurrent requests. Each arm completed
96 four-turn coding trajectories, 384 sampling requests, and 32 training updates.
Prompts ranged from 1,500 to 2,899 tokens; output allowance was 1,536 tokens.

An initial calibration observed about 19-second probes. A proposed 20-second
budget would reject those once the 1.5 safety factor was applied, so the measured
comparison used 35 seconds. This is an admission estimate, not a preemption
deadline or a claim that training can tolerate a 35-second delay. The evaluation
retained a 20-second actual-drain threshold and a 10% training-update p95 regression
threshold. Both passed here, with the overlap limitation above.

Representative training and sampling warmup ran before timing. Warmup restores
seeded adapter parameters and optimizer state; cross-arm initial hashes matched.
LoRA B starts at a small nonzero value to exercise distinct adapters. Frozen-base
checks and shared-storage checks passed, with zero copied base parameters. There
were no scheduler failures, admission timeouts, or output-limit hits. FSDP code and
sharding were unchanged; this small GPU comparison did not exercise FSDP.

## Why borrowing stopped

The first two shared batches completed in about 13 seconds. The learned batch
was predicted at 16.30 seconds but actually took 30.61–34.29 seconds. All shared
sampling finished by 79.64 seconds; the first update completed at 209.57 seconds.

After observing that longer batch, the upper-envelope estimator assigned a later
1,505-token prompt 34.29 seconds of service. With switching cost and the 1.5 safety
factor, estimated occupancy was **51.54 seconds**, exceeding the 35-second budget.
That rejection occurred with no training active or pending. The saved trace has
24 such duration-rejection events for prompts below 1,600 tokens after the first
update. Rejection counters count repeated scheduler evaluations, not unique
requests.

At 330 and 345 seconds, the dedicated sampler had 32 active requests and 16 queued;
the borrowed-GPU sampler was idle and no training was pending. **Sampling demand was
available.** Cross-batch fallback takes the maximum across history, while coarse
prompt/output buckets also mix requests with different realized output lengths.
A long observation can therefore block later short requests, and no shared
completions arrive to refresh that history. The trace does not prove every rejected
request would actually have fit the budget.

The next focused improvement is to separate comparable workload history from
cross-shape fallback and evaluate bounded refresh probes. Start with CPU trace
replay and tests; keep conservative GPU ownership and drain handling. Simply
raising the budget again would not resolve this feedback problem.

Task success remained low: final-stage all-tests-pass was 4/96 versus 0/96.
Every trajectory still completed its four turns and supplied training data.
This is a system-behavior check, not evidence of learning quality. The higher
sampling p95 also needs further attribution before any broad performance claim.

## Artifacts, validation, and spending

- [Summary](borrow-budget-summary.json), [evaluation](borrow-budget-evaluation.json),
  [rejection evidence](borrow-budget-diagnosis.json), and
  [source hashes/configuration](borrow-budget-provenance.json).
- [Comparison](borrow-budget-budget-comparison.png),
  [borrowing and queue timeline](borrow-budget-borrowing-timeline.png), and
  [GPU timeline](borrow-budget-gpu-timeline.png), with SVG versions alongside.
- Full telemetry, logs, launch-source snapshots, CPU check logs, and artifact hashes:
  `runs/borrow-budget-20261104T004506Z/`. This directory is ignored by Git.
- Modal volume `tuft-shared-weight-experiments` retains the raw run directory.
  [Completed app](https://modal.com/apps/ekzhu67/main/ap-W7PueyVKERM6cxOaNvTJ4U).
- The warmup restoration regression and workload tests passed: 13 tests. Earlier
  workload/policy preflight passed 30 tests. Ruff passed; Pyright reported no errors
  and one existing missing-psutil-stubs warning.

Estimated requested compute cost was **$9.79 total**, approximately $10 allowing
for small CPU setup costs: $1.71 stopped calibration, $3.23 unsuccessful initial
pair, and $4.85 valid pair. The unsuccessful pair was stopped by the initial-adapter
integrity guard after warmup changed the weights. Restoring both weights and
optimizer state fixed the harness; that pair is excluded from performance results.
Its telemetry and the calibration are retained under their run IDs in
[cost accounting](borrow-budget-cost.json). This is a rate-based estimate using
[Modal pricing](https://modal.com/pricing), not an invoice; storage and credits are
excluded. All experiment containers were confirmed stopped.

To regenerate figures locally, without GPUs:

```bash
python benchmarks/borrowing/report_borrow_budget.py \
  benchmarks/borrowing/runs/borrow-budget-20261104T004506Z/comparison.json
```

The report writes summary/evaluation JSON and PNG/SVG figures beside the input.
Keep both per-arm `.log` files there for the queue timeline.
