# LoopWeave scheduling in TuFT

The opt-in scheduler uses idle HF training compute for bounded sampling work.
It keeps the existing HF/PEFT trainer, vLLM engines, checkpoint formats, losses,
and Tinker sequence guards. Each tenant retains its own adapter and optimizer;
sampling sessions retain their published snapshot identity.

## Configuration

`config/tuft_config.loopweave.yaml` is a two-GPU example. Under
`supported_models[].scheduling`, select:

| Policy | Resources and behavior |
| --- | --- |
| `round_robin` | Existing behavior; default, no new admission queue. |
| `balanced` | Existing dedicated sampling replicas, bounded fair admission and outstanding-token routing. Works with HF or FSDP training. |
| `loopweave` | One shared HF/vLLM GPU plus persistent sampling GPUs; admission closes and drains before training. |

`config/tuft_config.loopweave.h100.yaml` records the measured four-tenant Qwen3-4B
throughput profile on two H100 80GB GPUs. It enables CUDA graphs and raises the
fixed/shared in-flight limits to 128/64. The conservative defaults bound smaller
drain batches; they are not a universal throughput optimum. Short eager-mode
pilots regressed, so keep this policy opt-in and measure the intended workload.

`loopweave` requires one configured base model, `training_backend: hf`,
`tensor_parallel_size: 1`, `colocate: true`, both capabilities, and
`data_parallel_size >= 2`. The DP count includes the shared sampler. Therefore
DP=2 uses **two total GPUs**, versus three for a disaggregated trainer plus two
sampling replicas. Startup verifies the trainer and shared sampler actually have
the same Ray node/GPU assignment. Use an otherwise idle Ray cluster to allow
fractional actors to pack together.

`sampling_memory_fraction` reserves vLLM's memory budget on the shared GPU.
The trainer and its activations, gradients, optimizer states and adapters must
fit in the remaining memory. Both base-model copies and the sampling KV pool
remain resident. This implementation shares compute time; it does not share base
weight storage or dynamically release KV memory. Dedicated samplers retain the
existing 0.9 memory budget. Validate the intended model and batch size before
raising concurrency or context lengths.

## Admission and ordering

The scheduler limits each replica's in-flight request count and maintains a
bounded pending queue. Queued users take turns, preserving FIFO admission within
each user's queue. The routing cost is `num_samples * (prompt_tokens + max_tokens)`.
Replica affinity is a tie-breaker within `locality_slack` times the incoming
request's cost; it cannot override arbitrarily large load differences. vLLM still
owns token scheduling, continuous batching, and KV-cache management.

Only requests within `flex_max_tokens` can enter the shared sampler, up to
`flex_max_inflight` at once. All others remain on persistent samplers. After four
consistent training arrivals, a bounded median profiler predicts the next update.
Measured shared-sampler service time, reserved work, and a 1.5 safety factor reject
admissions predicted to cross that boundary. Irregular and expired predictions
are discarded. Prediction is a heuristic, not a hard latency guarantee.

Any arriving training operation immediately closes shared sampling admission.
Already admitted requests drain; fixed samplers continue serving. Training
mutations execute one at a time, with a configurable short quiet interval after
completion to allow the following optimizer/checkpoint request to arrive without
reopening sampling. Eligible rollouts can wait up to `sampling_alignment_s` for
this training burst/quiet interval to end. This prevents an entire next-round
wave from being sent to fixed replicas just before shared capacity becomes free.
The budget is a per-request maximum; when it expires, the fixed route is eligible
even if training is still running. Long requests are not deliberately delayed.
An unexpected update can wait for the bounded admitted batch;
the scheduler does not preempt tokens or claim an absolute time bound.

Queue-full and admission-timeout errors use TuFT's resource-exhausted status (429).
A queue timeout applies
only before dispatch, never to the duration of a running GPU operation.
Cancellation removes pending samples. Admitted samples retain their reservation
until actual backend completion. Admitted training mutations finish along with
their sequence bookkeeping even if the caller cancels, avoiding duplicate
optimizer updates on retries. Shutdown rejects pending samples and drains
admitted work before killing either backend.

## OpenAI and compatibility

OpenAI HTTP/streaming requests use persistent replicas only. The shared sampler's
embedded OpenAI server is disabled, so HTTP cannot bypass the shared GPU gate.
HTTP traffic currently retains round-robin routing and is not represented in the
Tinker scheduler's outstanding-token counters. A mixed HTTP/Tinker deployment
therefore has approximate load estimates on persistent replicas.

With `merge_zero_adapters: true` (the default for opt-in policies), snapshots use
the base sampling path only after their safetensors weights are verified: all
LoRA B factors are zero, all A factors are finite, pairs have compatible shapes,
and no bias, DoRA, saved modules, layer replication, per-layer rank/alpha overrides,
fused parameter targets or unknown weight keys
are present. Metadata alone never enables this optimization. Nonzero, unsupported
and unreadable snapshots take the normal adapter path. Public snapshot identity
is retained. Set the option to false to disable this fast path.

There is no automatic log-probability calibration. The scheduler does not rewrite
returned tokens or probabilities. As with existing vLLM batching and replica
routing, bitwise numerical equality across different execution paths is not
promised; the GPU harness explicitly checks zero-adapter equivalence.
Clients retain control of RL policy freshness; the server does not infer an
algorithm's acceptable staleness from elapsed time.

## Relation to the paper and prototype

The source reference is `feat/flex_backend` at
`73232a7b5d8b62bc56e47df98b04fe5794d536dd`, with an earlier sampling-only scheduler
on `feat/sample_requests_scheduling`. The prototype's CUDA IPC path requires a
custom fused Qwen3 tensor-parallel trainer and private vLLM allocator hooks. It is
not storage-compatible with main's HF/FSDP backends. Porting it wholesale would
also replace current LoRA geometry, scaling, and loss behavior.

This practical variant adopts fixed/shared execution, verified initial-adapter
merging, fair bounded admission, adapter affinity, loop prediction, and drain-safe
transitions. It does not claim
the paper's zero-copy switching, sampling-delay optimization, multi-group dynamic
FSDP repartitioning, or learned probability correction. Extending those mechanisms
requires independent correctness and performance evidence.

## Validation and measurements

Run `pytest tests/test_scheduling.py tests/test_scheduling_integration.py tests/test_adapter_equivalence.py` for
deterministic admission, cancellation, shutdown, physical-placement, controller
ordering and isolation tests. The integration tests replace only the engine/Ray
boundary; they use real controllers, factories and sequence guards.

Run `modal run scripts/benchmark_loopweave_modal.py --repeats 3 --model Qwen/Qwen3-4B --validate-tests` for ephemeral
two-H100 real-engine comparisons. The harness aligns the published container with
the checkout's dependencies, uses a pinned model revision, warms base and LoRA kernels,
alternates policy order, and records raw NVML samples and per-run throughput and
latency. Training data and sampling token budgets are identical across policies.
Each tenant publishes its updated snapshot between measured rounds.
The throughput case uses four tenants, 32 samples per tenant per round, three
rounds, 512 generated tokens per sample, fixed/flex limits 128/64, CUDA graphs,
a 544-token shared-request budget,
and a 0.25-second alignment budget. These are explicitly measured settings,
not a claim that the conservative defaults fit every workload.
GPU-seconds are allocated GPU count multiplied by measured elapsed time, not
inferred kernel execution time. The benchmark is functionality/performance
validation, not task-quality evaluation or a reproduction of the paper's claims.

`DPSamplingBackend.scheduler.snapshot()` exposes aggregate queue, reservation,
completion and timing data. Summed request seconds overlap under concurrency and
must not be interpreted as GPU utilization. `exclusive_busy_s` includes training
and adapter/checkpoint maintenance. See `docs/loopweave-work-log.md` and
`benchmarks/loopweave/` for the validation record.
