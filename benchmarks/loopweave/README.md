# Practical LoopWeave validation

Publication note: see [LOCAL_ARTIFACTS.md](LOCAL_ARTIFACTS.md) for raw result
files retained locally because the commit scanner flags their provenance hashes.

**Policy update, 2026-10-12:** completed-loop telemetry, a budgeted fast-loop
planner and cost-aware borrowing have since been implemented with CPU tests.
All GPU measurements and figures below predate that update and retain their
original source hashes. They are not performance validation of the new policy.
See [the work log](../../docs/loopweave-work-log.md) for the CPU validation record.
The new policy passes **661 CPU tests** (37 skipped), with pinned lint/type
checks passing. [Validation metadata](scheduling-cpu-validation.json) and
[raw JUnit](scheduling-cpu-junit.xml) record the tested source and test results.

LoopWeave shares the trainer's frozen base with a colocated sampler, preserves
per-tenant LoRA and optimizer state, and borrows the training GPU group between
updates. FSDP keeps the frozen base sharded. Admitted sampling drains before
training; vLLM sleep releases its KV pool. Published adapter snapshots remain
explicit and isolated from live optimizer updates.

There are two comparison configurations:

- **Static TuFT:** a training group and a dedicated sampling group, each with its
  own base storage.
- **LoopWeave:** the same groups plus a sampler on the training group that aliases
  existing frozen weights. It does not allocate another resident base copy there.

`policy: loopweave` always enables sharing. There is no separate sharing option
or duplicate-base borrowing mode. Old `share_base_weights: true` configs migrate
without changing behavior; explicit `false` with LoopWeave fails with guidance.
Static `round_robin` remains the default. Unsupported shared layouts fail at startup.

## Configuration and implementation

Use an explicit `sampling_kv_cache_memory_bytes` budget per sampler rank:

- [HF two-GPU example](../../config/tuft_config.loopweave.yaml)
- [HF Qwen3.8-27B / two B200s](../../config/tuft_config.loopweave.shared.qwen_dense.b200.yaml)
- [FSDP Qwen3.6-35B-A3B / four H200s](../../config/tuft_config.loopweave.fsdp.shared.qwen_moe.h200.yaml)

The [development guide](../../docs/sphinx_doc/source/development/loopweave.md)
describes admission, failure handling, storage ownership and supported layouts.
The [work log](../../docs/loopweave-work-log.md) records the iterations.
Released torch 2.11.0, vLLM 0.24.0 and PEFT 0.20.0 are used. Explicit CUDA IPC
reducers and PEFT's expert-parametrization compatibility hook are internal,
version-specific interfaces; no unreleased dependency or custom CUDA binary is needed.

## Static versus LoopWeave measurements

The figures below select static and physically shared rows from the saved
experiments. Raw artifacts are unchanged. These are **existing measurements
replotted after mode consolidation**, not measurements of a new speedup caused
by removing the configuration switch. Their source revisions are recorded in
each raw artifact and differ from the current configuration cleanup.

| Workload | Static GPU-s / step | LoopWeave GPU-s / step | Static peak GiB | LoopWeave peak GiB |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B, HF, two B200s | 10.906 | 7.364 | 127.892 | 123.257 |
| Qwen3.6-35B-A3B, FSDP/TP=2, four H200s | 21.854 | 25.689 | 245.488 | 223.174 |

Each large-model comparison has **one paired repetition**, equal allocated GPUs,
12 training steps, 49,152 generated tokens and 8 publications per policy.
Initialization and correctness probes are excluded; NVML is sampled every 200ms,
so memory is an observed peak rather than a guaranteed maximum. Token budgets
and training data match; LoRA initialization and generated token identities are
not held identical across actor restarts. These are preliminary workload-specific
measurements, not confidence intervals or a task-quality evaluation.

Dense HF uses **32.5% less GPU time** than static. MoE FSDP uses **17.5% more GPU
time**, while reducing mean rollout latency by 40.7% and observed peak memory by
22.31 GiB. Its 40 serialized training/maintenance operations occupy 66.04s of a
77.07s interval; five sleep/wake pairs cost 2.48s. Request waits overlap and must
not be summed as additional wall-clock cost. This is not a fundamental limit:
reducing serialized work and avoiding borrowing during training-heavy workloads
are opportunities requiring targeted validation. The measured short-rollout MoE
workload should retain static scheduling when throughput takes priority.

[Dense figure](qwen38-27b-hf-static-comparison.png) ·
[Dense summary](qwen38-27b-hf-static-summary.json) ·
[Dense raw measurements](qwen-shared-hf-Qwen3.8-27B-repeats-1.json)

[MoE figure](qwen36-35b-fsdp-static-comparison.png) ·
[MoE summary](qwen36-35b-fsdp-static-summary.json) ·
[MoE raw measurements](qwen-shared-fsdp-Qwen3.6-35B-A3B-repeats-1.json)

Earlier three-repeat Qwen3-4B comparisons used 24.2% less pooled GPU time for HF
and 10.5% less for FSDP than static. The complete earlier record, including
retired duplicate-mode experiments, remains in [HISTORICAL.md](HISTORICAL.md).
Current reporting excludes duplicate rows and requires sharing evidence before
labeling a historical row as LoopWeave.

## Correctness and numerical scope

The pre-consolidation [full-checkpoint validation record](qwen-validation.json)
covers Qwen3.8-27B and Qwen3.6-35B-A3B on both HF and FSDP, with zero copied
base parameters, tenant isolation, checkpoint restoration and unchanged frozen
hashes. The [FSDP storage/gradient probes](fsdp-shards-validation.json) compare
exact outputs and gradients with ordinary FSDP, including activation checkpointing.
The [hybrid vLLM probes](qwen-vllm-layout-validation.json) cover GDN and expert LoRA.

[Strict HF numerical controls](qwen-numerical-controls.png) on two B200s produced
exactly zero prompt-score difference on repeats, cache reset and three cold-cache
adapter switches for both full checkpoints. They use shrink `split_k=1` through
vLLM's published `VLLM_TUNED_CONFIG_FOLDER` option; the measured rank-8 B200
configuration is saved in [lora-kernels](lora-kernels/NVIDIA_B200_SHRINK.json).
Normal kernels show repeated-score variation even without switching. The strict
control is not a claim of FSDP or cross-batch determinism, and its throughput cost
is not measured by the normal-kernel figures. Raw scores are preserved.

The mode-consolidation regression results are tracked in
[shared-only-validation.json](shared-only-validation.json). Previous GPU results
are retained under their original source hashes, not relabeled as current runs.

Current validation passes **643 CPU tests** (37 GPU-only skips), HF integration
in both deterministic and normal modes, and FSDP/TP=2 integration. Each GPU
integration completes 10 training steps, 8 publications and 10,240 output tokens,
with zero copied base parameters, unchanged frozen weights, tenant isolation and
checkpoint restoration. Deterministic switching scores are exact. Lint,
formatting, type checks and all six example configurations also pass.

The fresh Qwen3-0.6B HF smoke comparison completes 12 steps and 49,152 output
tokens per policy: 9.654s static versus 9.800s LoopWeave. This one short run
verifies the current two-policy harness; it does **not** demonstrate a speedup.
Observed peak memory is 22.61 GiB static versus 39.73 GiB LoopWeave. The
additional sampler has a 16-GiB KV budget while awake: sharing the frozen base
does not eliminate its KV/graph allocations. This small-model profile therefore
needs a workload-sized cache budget before being treated as an efficiency target.
[Smoke figure](shared-only-smoke-performance.png) ·
[Smoke summary](shared-only-smoke-summary.json) ·
[Raw smoke and HF integration results](loopweave-static-gpu-Qwen3-0.6B-128.json) ·
[FSDP integration](fsdp-loopweave-static-Qwen3-0.6B-repeats-0.json).

## Reproduction

Small functional checks use the existing Modal skill and ephemeral jobs:

```sh
modal run scripts/benchmark_loopweave_modal.py --model Qwen/Qwen3-0.6B \
  --repeats 1 --output-tokens 128 --validate-tests
modal run scripts/verify_fsdp_shared_modal.py --full --repeats 0
TMPDIR=/tmp uv run pytest
```

The performance runner alternates static/LoopWeave order across repetitions.
There is no `--shared-weights` option. Full checkpoint validation uses:

```sh
modal run scripts/verify_qwen_shared_modal.py --model Qwen/Qwen3.8-27B --backend hf
modal run scripts/verify_qwen_shared_modal.py --model Qwen/Qwen3.6-35B-A3B --backend fsdp
```

Use `--revision` with the commit in the corresponding artifact for pinned reruns.
Hybrid runs use text-only BF16 backbones, 4-GiB KV budgets, maximum 64 sequences,
Triton GDN/MoE kernels, and `VLLM_USE_AOT_COMPILE=0` (torch.compile/CUDA graphs
remain enabled). Training uses flash-linear-attention 0.5.2. HF requires TP=1;
FSDP and sampler TP sizes must match and fit on one node. Quantization, dynamic
resharding and other architectures are not covered. Startup still needs temporary
full-model/destination allocations. Producer failure requires backend restart.

Recreate either comparison from its raw JSON with:

```sh
uv run --no-project --with matplotlib python scripts/plot_loopweave.py RESULTS.json --output FIGURE.png
python scripts/summarize_loopweave.py RESULTS.json --output SUMMARY.json
```
