# LoopWeave adaptation: validation record

The implementation is an opt-in scheduler for the existing HF trainer and vLLM
samplers. It preserves tenant adapters, optimizer state, checkpoint publication,
and TuFT's sequence guards. It can borrow the HF training GPU between updates;
training closes admission and drains outstanding shared sampling first.

The paper/prototype investigation and iteration history are in
[the work log](../../docs/loopweave-work-log.md). Architecture, configuration,
compatibility and operating limits are in
[the development guide](../../docs/sphinx_doc/source/development/loopweave.md).

## Completed performance results

Three matched pairs on two H100 80GB GPUs, Qwen3-4B, four tenants and 512-token
rollouts improved throughput by **1.265× median paired speedup** (range
1.257–1.289×). Every run produced 196,608 tokens and completed 12 training steps
plus eight publications. These are steady-state results for this workload, not
an assertion that all deployments improve.

| Metric | Static allocation | Shared GPU | Change |
| --- | ---: | ---: | ---: |
| Allocated GPU-seconds per training step, pooled | 4.392 | 3.457 | **21.3% lower** |
| Mean rollout request latency | 6.828 s | 4.675 s | **31.5% lower** |
| Mean of per-run p95 rollout latencies | 7.449 s | 5.772 s | 22.5% lower |
| Mean training latency, including admission | 1.453 s | 1.659 s | **14.2% higher** |
| Mean NVML GPU busy percentage | 39.4% | 64.4% | +25.0 percentage points |
| Peak resident memory per GPU, NVML order | 11.95 / 73.05 GiB | 48.64 / 72.84 GiB | More shared-GPU memory |

| Pair | Execution order | Static elapsed | Shared elapsed | Throughput ratio |
| --- | --- | ---: | ---: | ---: |
| 0 | Static, shared | 27.196 s | 21.501 s | 1.265× |
| 1 | Shared, static | 25.406 s | 20.205 s | 1.257× |
| 2 | Static, shared | 26.451 s | 20.521 s | 1.289× |

![All paired measurements](performance.png)

[Raw measurements](gpu-Qwen3-4B-512.json) ·
[Machine-readable summary](final-summary.json) ·
[H100 configuration](../../config/tuft_config.loopweave.h100.yaml)

The timed code is commit `26afb88`; its source and harness hashes are in the raw
file. Follow-up fixes preserve legacy persistence signatures and estimate
concurrent service rate by busy interval. The timed three-round runs had zero
window rejections: this comparison demonstrates GPU borrowing and rollout
alignment, not a measured benefit from learned loop prediction. A separate
five-round GPU test validates the final source after those fixes. Runtime:
torch 2.11.0+cu130, vLLM 0.24.0, transformers 5.19.0, Python 3.12.13.

## What the comparison measures

Both policies receive two H100 GPUs. Static allocation uses one HF trainer and
one vLLM sampler. The new policy uses the same trainer and fixed sampler, plus a
second vLLM engine resident on the training GPU. Both policies use the same model,
LoRA rank, training examples, rollout token budget and CUDA-graph setting within
each pair. The baseline is TuFT's existing `round_robin` execution path.

Four independent users each execute three sample/train/publish rounds. Every
round performs forward/backward and an optimizer step. Updated sampling snapshots
are published between rounds: 12 training steps and eight publications per run.
The final throughput workload requests 32 samples per user per round, each with a
32-token prompt and 512 generated tokens: **196,608 output tokens per run**.
Training uses four 64-token sequences per step. This synthetic loop verifies
serving/training functionality and throughput, not RL task quality.

Startup, model loading, CUDA-graph capture, warmup, and explicit correctness
checks are outside the timing interval. Both base and nonzero-LoRA paths are
warmed on every engine. Timing includes sampling, training, checkpoint export and
publication of updated adapters. Policy order alternates across repetitions.
Raw results include full configuration, model revision, dependency versions,
source and harness fingerprints, every 200 ms NVML observation, per-tenant losses,
latencies, and scheduler counters from the measured interval.

GPU-seconds = two allocated GPUs × elapsed seconds. This is resource reservation
cost within the timed loop, not the total bill for the validation job, measured
kernel execution time, or power consumption. NVML busy
percentages and overlapping request-seconds are not substitutes for throughput.

## Earlier iterations (retained, including regressions)

Each row compares matched policies within that experiment. Different rows have
different workloads or execution settings; do not compare their elapsed times as
an implementation-only improvement.

| Experiment | Static seconds | Shared seconds | Throughput ratio | Raw data |
| --- | ---: | ---: | ---: | --- |
| 0.6B, eager, small wave, limits 16/8 | 18.240 | 26.906 | 0.678× | [JSON](gpu-pilot-constrained.json) |
| 0.6B, eager, limits 32/16 | 21.779 | 22.055 | 0.987× | [JSON](gpu-pilot-tuned.json) |
| 4B, eager, updated snapshots, before alignment | 25.010 | 30.051 | 0.832× | [JSON](gpu-4b-before-alignment.json) |
| 4B, eager, bounded alignment | 28.378 | 32.450 | 0.875× | [JSON](gpu-4b-aligned-eager.json) |
| 4B, CUDA graphs, 32 samples/user, 128 output tokens | 11.516 | 11.013 | 1.046× | [JSON](gpu-4b-throughput-pilot.json) |

The first two pilots held the initial sampling snapshot throughout timing. Later
runs publish trained adapters between rounds. The eager pilots used eight CPU
cores; the CUDA-graph throughput runs reserve 16 and set OpenMP/MKL threads to one
per process, identically for both policies. The 128-token graph pilot is a single
pair and is not evidence of a statistically established 4.6% gain.

The pilots exposed insufficient admission capacity, rollouts routed entirely to
the fixed sampler during the post-training quiet interval, and first-use LoRA
compilation inside measurements. The implementation/harness corrections and
regression tests are recorded in the work log. GPU sharing remains workload
sensitive; adding a resident sampler does not guarantee better efficiency.

## Correctness and regression checks

The full local test suite passed **614 tests**, with **32 GPU tests skipped**.
Repository hooks passed lint, formatting, isolated type checking, tests and secret
scanning. The final five-round two-H100 integration test also passed: ten training
steps, eight publications and 10,240 output tokens. Its source fingerprint matches
final implementation commit `5bd0005`. See [GPU test evidence](gpu-integration.xml)
and [runtime metadata](gpu-integration-Qwen3-4B.json).
[Validation metadata](validation.json) also records the earlier 604-test
CPU selection and 38 focused scheduler/adapter/controller tests; these overlap
and must not be summed.

The focused tests cover queue capacity and 429 errors, timeouts, tenant FIFO and
fair admission, token-work routing, failure cleanup, cancellation after dispatch,
exactly-once sequence advancement for cancelled optimizer calls, drain-safe
shutdown, physical placement checks, loop prediction, bounded rollout alignment,
HTTP routing, and conservative zero-adapter verification.

GPU correctness assertions verify that a real optimizer update changes one
user's LoRA weights while another user's weights stay identical, optimizer
checkpoint restoration restores the original adapter, a trained snapshot can be
published and sampled, nonzero snapshots retain their adapter path, and verified
zero snapshots match the normal adapter's greedy tokens/log probabilities.
These checks run before every measured policy. Zero-adapter comparisons apply to
the opt-in policy, where that optimization exists.

## Reproduction

From this checkout, with Modal configured:

```bash
modal run scripts/benchmark_loopweave_modal.py --repeats 3 \
  --model Qwen/Qwen3-4B --output-tokens 512 --validate-tests
python scripts/summarize_loopweave.py \
  benchmarks/loopweave/gpu-Qwen3-4B-512.json \
  --output benchmarks/loopweave/final-summary.json
TMPDIR=/tmp RAY_TMPDIR=/tmp uv run pytest
```

The Modal job is ephemeral and deploys no service. It builds the checkout's
runtime dependencies over the published TuFT image and removes the old incompatible
FlashAttention wheel. The harness records the actual torch/vLLM/transformers
versions; they must be considered when comparing a future rerun.
Use `--repeats 0 --validate-tests` to run only the current five-round GPU
integration test. Recreate the chart with:

```bash
uv run --no-project --with matplotlib python scripts/plot_loopweave.py \
  benchmarks/loopweave/gpu-Qwen3-4B-512.json
```

## Limits

GPU borrowing supports one base model, HF training, TP=1 and at least one persistent
sampler. FSDP retains its existing allocation; it can use the independent balanced
sampling policy. Both base-model copies and the KV pool remain resident on the
shared GPU, so memory use increases. CUDA IPC weight sharing and dynamic FSDP
repartitioning are not implemented. OpenAI HTTP traffic stays on fixed replicas
and is not included in token-work routing estimates. Unexpected training waits
for admitted shared samples to finish; there is no hard wall-clock latency bound.
The server does not change log probabilities or infer acceptable policy staleness.
