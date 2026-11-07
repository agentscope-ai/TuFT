# Borrowing versus static allocation

Borrowing served real overflow repeatedly, but this configuration did not improve
end-to-end throughput. Shorter sampling latency came with longer training and
publication waits. The next fix should address checkpoint and publication queueing.

## Matched comparison

Both arms used commit `bf5e578`, Qwen3.5-4B, two GPUs, eight independent tenants,
and the user guide's four-turn coding workload. Each tenant ran eight trajectories,
then trained and published its adapter. Dedicated capacity was 32 requests;
borrowing allowed four requests with `max_request_tokens: 8192`. The static arm
kept the same shared-base placement and disabled borrowed admission with a one-token
borrowed-request cap.

The table uses the same 240–840 second window in both arms. Request latency includes
queueing and follows requests submitted in that window through completion.
Training latency includes backward and optimizer calls; publication is separate.
GPU percentages are time-weighted NVML busy readings, not compute efficiency.

| Metric | Static | Borrowing |
| --- | ---: | ---: |
| Trainer GPU utilization | 79.6% | 78.0% |
| Dedicated GPU utilization | 40.6% | 28.3% |
| Average of both GPUs | 60.1% | 53.2% |
| Completed output tokens/s | 496.5 | 353.8 |
| Completed sampling requests | 530 | 353 |
| Sampling p95 | 67.2 s | 61.9 s |
| Training p95 | 84.2 s | 173.3 s |
| Updates completed within window | 16 | 16 |
| Cumulative updates by 840 s | 17 | 16 |
| Borrowed requests admitted within window | 0 | 16 |

Completed output tokens/s credits the output tokens of requests finishing in the
window. Sampling latency cohorts contain 510 and 346 requests; training cohorts
contain 17 and 14 updates. Equal window update counts conceal a boundary effect:
one static update finished before 240 seconds; none did with borrowing.

The guide's earlier eight-tenant static run recorded 76.3% trainer utilization,
39.6% dedicated utilization, 71.7 s sampling p95, 64.1 s training p95, and 16 window
updates. It remains a historical reference. The new pair runs the current code on
the same server and container as each other. The server/container differ from the
guide run, so the matched pair is the performance comparison.

## What the telemetry explains

See the [detailed diagnosis](reactive-comparison-20261106-diagnosis.md) for the
critical timeline and a CPU reproduction of the queue barrier.

Borrowing is no longer limited to initial probes: 41 requests used the trainer GPU
across the full run, and 33 were admitted after the first training update. The
static arm had an idle trainer with dedicated overflow for 15.0% of the window.
There was work available to borrow.

Training waited up to **50.3 seconds** for accepted borrowed requests to finish
(99.5 seconds total across the full run). Median backward execution itself was
almost unchanged: 30.47 s static versus 30.41 s borrowing. Backward queue p95 grew
from 54.5 to 118.9 seconds, and optimizer queue p95 from 30.6 to 60.8 seconds.
These operation statistics cover the full timed runs.

Publication p95 grew from **61.9 to 181.5 seconds**. The borrowed sampler's
publication operation took at most 0.13 seconds; its queue wait reached 90.4
seconds. End-to-end publication also includes checkpoint saving and its queue.
The dedicated sampler had no active requests from approximately 677 to 831 seconds
in the borrowing arm. The phase plot shows clients waiting on training and
publication together during that gap.

Two code paths deserve attention before another performance run:

- `HFTrainingBackend.save_state` does not pass `scheduler_tenant`, so its checkpoint
  enters `MutationQueue` as a global barrier and cannot use tenant-scoped priority.
- `DPSamplingBackend._add_adapter` waits for every replica. The borrowed replica's
  registration enters the training exclusion gate as an unidentified publication,
  so dedicated sampling cannot resume until that registration also completes.

Keep the shared-GPU exclusion gate, but avoid making dedicated rollout readiness
wait on unrelated training. Tenant-scoped checkpoint scheduling and separate
borrowed-replica readiness are the next focused changes. Long non-preemptible
borrowed requests also need a tighter admission bound or explicit client readiness;
this run does not establish a suitable bound.

## Validation and scope

Both arms preserved frozen weights, copied zero base parameters into the borrowed
sampler, started with identical adapter fingerprints, completed every admitted
trajectory, and had no output-limit truncation. Full totals were 736 requests / 23
updates in 1,045.1 s static and 672 requests / 21 updates in 1,193.1 s borrowing.
The finite admission window means these are different completed-work totals.
Task success was 11/184 and 3/168 trajectories; this is a systems test, not a claim
about RL learning quality or identical generated trajectories.

This was one ordered pair, static then borrowing, with the HF backend. It does not
measure FSDP performance. Including startup and warmup, the pair occupied two GPUs
for 48.8 minutes (about 1.63 allocated GPU-hours). Both GPUs and the tool service
were released afterward. No follow-up GPU experiment was launched.

## Saved artifacts

- [Utilization and latency](reactive-comparison-20261106-analysis/comparison-timelines.png)
  ([SVG](reactive-comparison-20261106-analysis/comparison-timelines.svg)).
- [Client phases](reactive-comparison-20261106-analysis/comparison-client-phases.png)
  ([SVG](reactive-comparison-20261106-analysis/comparison-client-phases.svg)).
- [Metrics](reactive-comparison-20261106-analysis/summary.json) and
  [validation/provenance](reactive-comparison-20261106-analysis/validation.json).
- [Report generator](report_reactive_comparison.py).

Raw telemetry, source snapshots and environment provenance are retained separately from this repository.

```bash
python benchmarks/borrowing/report_reactive_comparison.py \
  benchmarks/borrowing/runs/reactive-20261106 \
  benchmarks/borrowing/reactive-comparison-20261106-analysis
```
