# Shared weights (experimental): benchmarks

The [user guide](../../docs/sphinx_doc/source/user-guide/shared-weight.md) explains
how to enable this feature, the benchmark setup, results, and limitations.
It includes latency, allocated GPU time and sampled GPU utilization.

This directory contains the feature's benchmark code and example configurations:

- `benchmark_shared_weight.py`: real-engine correctness and performance workload.
- `benchmark_shared_weight_agents.py`: multi-tenant, three-turn agent workload.
- `tuft_config.shared-weight.h100.yaml`: HF example using two H100 GPUs.
- `tuft_config.shared-weight.fsdp.h100.yaml`: FSDP example using four H100 GPUs.

The example YAML files are starting points, not the eager, long-context settings
used for the saved agent comparison. Exact run settings and software versions
are recorded in each run's `model_config` and `versions` in the raw telemetry.

## Saved data and figures

- [Raw telemetry](agent-telemetry.json) and [summary](agent-results.json).
- [Agent results](agent-results.png) and [GPU utilization](gpu-utilization.png).
- [Startup-tail analysis](startup-tail.json) and [figure](startup-tail.png).

These measurements predate the PR extraction. Policy labels were renamed; raw
measurements are unchanged. The excluded predictive planner applied no holds;
the raw `infeasible_plans` counter comes from that component. Reactive alignment
was zero in this run and remains disabled by default. No performance change from
the extraction or later CPU fixes is claimed.

GPU utilization was calculated later from these saved readings, without a new GPU
run. It is the arithmetic mean of NVML GPU-active percentages sampled roughly
every 0.2 seconds, not SM occupancy or FLOP utilization. The two-GPU average is
36.48% static versus 40.44% shared weights (+3.97 percentage points).

## Latency diagnosis and measurement limits

The reported p95 is sampling-request latency over the entire timed workload,
including the first rollout round. Server initialization and warmup are excluded.
We examined the first round because the full-run p95 rose from 4.072 s to 5.075 s,
even though average latency and workload completion time improved. It is a check
on the cost of borrowing, not the feature's main optimization target.

All 29 requests above the shared-weight arm's pooled p95 were first-turn requests
in the initial round. Later-round p95 was 1.937 s versus 2.813 s static. Warmup
sampling bypassed the scheduler, and the dedicated sampler's initial service was
slower in the shared-weight run. The borrowing sampler made nine serial cold
probes while waiting for the dedicated sampler to drain. Training and publication
drains did not cause these outliers. No predictive holds were applied.

![Startup-tail diagnosis](startup-tail.png)

The saved telemetry does not explain why the dedicated sampler started more
slowly. Calibration under sustained load, correlated scheduler/engine timing, and
representative warmup remain follow-up investigations. Later-round results help
locate the slow requests; they do not replace the full-run result or establish
steady-state latency in a long-running deployment.

Training service also improved, so the wall-time gain cannot all be attributed
to scheduling. This workload verifies multi-tenant system behavior, not learning
convergence or numerical equivalence between training and inference.

## Regenerate reports without GPUs

From the repository root, using an environment with Matplotlib installed:

```bash
mkdir -p /tmp/shared-weight-report
cp benchmarks/shared-weight/agent-telemetry.json /tmp/shared-weight-report/
python benchmarks/shared-weight/report_shared_weight_agents.py /tmp/shared-weight-report/agent-telemetry.json
python benchmarks/shared-weight/diagnose_shared_weight_agent_tail.py /tmp/shared-weight-report/agent-telemetry.json --plot
```

Outputs go to `/tmp/shared-weight-report/` and use the input stem: `agent-telemetry-summary.json`, `agent-telemetry.png`,
`agent-telemetry-gpu-utilization.png`, and `agent-telemetry-tail-diagnosis.json/png`,
plus SVG, per-round latency figures, and `agent-telemetry-sensitivity.json`. The archived files above use shorter names.
For the other workload, `summarize_shared_weight.py` summarizes the output of
`benchmark_shared_weight.py`.

GPU integration tests require explicit `--gpu`; they cover shared storage, adapter
isolation, publication, phase changes, and FSDP shard/gradient equivalence. No new
GPU experiments were run for this documentation update.
