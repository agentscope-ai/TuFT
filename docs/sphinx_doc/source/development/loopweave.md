# LoopWeave scheduling in TuFT

The opt-in scheduler uses idle training compute for bounded sampling work.
It keeps the existing HF/PEFT and FSDP2 trainers, vLLM engines, checkpoint formats, losses,
and Tinker sequence guards. Each tenant retains its own adapter and optimizer;
sampling sessions retain their published snapshot identity.

## Configuration

`config/tuft_config.loopweave.yaml` is a two-GPU example. Under
`supported_models[].scheduling`, select:

| Policy | Resources and behavior |
| --- | --- |
| `round_robin` | Existing behavior; default, no new admission queue. |
| `balanced` | Existing dedicated sampling replicas, bounded fair admission and outstanding-token routing. Works with HF or FSDP training. |
| `loopweave` | A shared trainer/sampler GPU group plus persistent sampling GPUs; admission closes and drains before training. |

`config/tuft_config.loopweave.h100.yaml` records the measured four-tenant Qwen3-4B
throughput profile on two H100 80GB GPUs. It enables CUDA graphs and raises the
fixed/shared in-flight limits to 128/64. The conservative defaults bound smaller
drain batches; they are not a universal throughput optimum. Short eager-mode
pilots regressed, so keep this policy opt-in and measure the intended workload.

`loopweave` requires `colocate: true` and both capabilities for each base model,
and `data_parallel_size >= 2`. HF requires `tensor_parallel_size: 1`; FSDP requires
`fsdp_num_gpus == tensor_parallel_size`. The DP count includes the shared sampler.
HF with DP=2 uses **two total GPUs**. Two FSDP ranks with TP=2 and DP=2 use **four
total GPUs**, versus six for separate training and two samplers. Startup verifies
physical placement. HF and FSDP trainers own full GPU reservations, and each
borrowing sampler starts on its trainer's exact node and rank-ordered devices.
Distinct base models therefore reserve distinct training groups. The FSDP group
must fit on one node for CUDA IPC sharing.

LoopWeave always shares the frozen base. Set an explicit
`sampling_kv_cache_memory_bytes` budget per sampler rank, sized together with
training activations, optimizer state and graph memory. There is no supported
mode that places a second resident base copy on the training group.
`round_robin` retains static TuFT behavior and remains the default.

## Shared frozen weights

Selecting `scheduling.policy: loopweave` enables physical frozen-base sharing.
Older configs with `share_base_weights: true` are accepted and normalized;
`share_base_weights: false` with LoopWeave is rejected with migration guidance.
The flag is no longer serialized or exposed as a configuration option.
`config/tuft_config.loopweave.h100.yaml` provides the two-H100 configuration.
The implemented mappings cover unquantized Qwen3 and Qwen3.5-based dense/MoE
text backbones, using the pinned torch 2.11.0 / vLLM 0.24.0 runtime. Qwen3.8-27B
uses the dense Qwen3.5 architecture; Qwen3.6-35B-A3B uses its MoE architecture.
Both full checkpoints pass HF and FSDP integration; see
`benchmarks/loopweave/qwen-validation.json` for revisions and numerical limits.
Qwen3.8-Flash-Next uses a different architecture and is outside scope.
Other architectures, dtypes and layouts fail at
startup; there is no silent fallback to duplicate base weights.
Set `sampling_kv_cache_memory_bytes` explicitly (per sampling GPU/TP rank). For the
two-H100 Qwen3-4B validation profile this is `17179869184` (16 GiB). IPC storage
and released loader allocations make automatic process-local memory profiling
unreliable for this mode; an explicit KV budget also makes comparisons reproducible.
It overrides vLLM's fractional cache sizing. Both HF and FSDP trainers reserve
full GPUs; their borrowing samplers request no additional GPU reservation.
Size the KV budget together with model, graph,
training activation and optimizer memory for the intended hardware.

The HF trainer packs frozen q/k/v and gate/up tensors into contiguous slabs once,
before accepting work. Its original HF Parameters become views into those slabs,
preserving HF modules, adapter names, checkpoint formats and optimizer behavior.
The sampler's custom loader installs CUDA IPC aliases during model loading,
before profiling and graph capture. Vocabulary weights can retain a separate
allocation when vLLM padding differs. The worker verifies that shared pointers
remain aliases after LoRA wrapping and engine initialization.

After admitted sampling drains, training operations put the flex sampler into level-1
sleep. The shared base remains resident in trainer-owned storage; vLLM releases
its KV pool and preserves private weight/adapter allocations through its normal
sleep implementation. Sampling and sampler adapter maintenance wake it first.
Repeated phase requests coalesce. A failed transition disables further GPU
operations through that engine. Shutdown releases sampler aliases before ending
the trainer process that owns their storage. Fixed samplers remain available.
The loader currently creates temporary destination allocations before rebinding
them, so startup must fit those allocations too. An actor failure requires a
backend restart; CUDA IPC aliases cannot survive loss of their producer.

Updated LoRA snapshots are still published separately. Frozen base sharing does
not expose live optimizer updates to sampling or relax tenant isolation.

### FSDP sharded base storage

`config/tuft_config.loopweave.fsdp.shared.h100.yaml` configures two FSDP ranks and
two TP=2 sampling replicas on four H100s. Frozen base matrices remain sharded,
and LoRA parameters, gradients and optimizer state retain FSDP semantics. Only
small normalization and MoE router weights are replicated, matching vLLM's execution layout.

Public FSDP shard placement puts attention output and MLP down projections on
column shards, and q/k/v, gate/up and vocabulary matrices on row shards. Local
q/k/v and gate/up shards are packed into contiguous slabs; public
`load_state_dict(assign=True)` refreshes FSDP's storage references before any
optimizer is created. CUDA IPC exports these persistent local shards. Full
layer tensors gathered during training remain temporary and are never exported.
The sampler's TP rank imports the corresponding FSDP rank's shards before graph
capture. Training still gathers and reshards full layers normally.

Hybrid attention needs independently partitioned Q/K/V segments, including its
convolution channels. Public torch parametrizations store these tensors in rank
order and reconstruct logical HF order after each layer's all-gather. Expert
gate/up matrices use the same scheme along their intermediate dimension; expert
down matrices shard that dimension directly. These transformations preserve
LoRA coordinates and checkpoints but add temporary permutations during training.
Recurrent decay scales use FP32, matching vLLM and the HF computation.

`sampling_language_model_only: true` omits unused vision encoders. Shared
samplers always use text-only loading and Triton MoE kernels, which retain the
original BF16 expert layout. `sampling_moe_backend: triton` applies the same
kernel choice to dedicated samplers for controlled comparisons.
`sampling_gdn_prefill_backend: triton` avoids FlashInfer's larger startup build;
the default remains vLLM's automatic selection. The hybrid validation image
uses `sampling_max_num_seqs: 64`: recurrent caches require one state block per
active sequence, and a small explicit KV budget may not fit vLLM's default graph
capture concurrency. Set this limit together with the cache budget. The image
includes flash-linear-attention 0.5.2 for training. vLLM 0.24 rejects batch-invariant
mode for GDN attention; hybrid numerical controls are recorded separately.
Full Qwen3.8-27B and Qwen3.6-35B-A3B HF numerical controls use vLLM's
published `VLLM_TUNED_CONFIG_FOLDER` interface to select LoRA shrink `split_k=1`.
Ordinary/shared repeats, cache reset and three cold-cache sleep/wake transitions
all produced exactly equal prompt log probabilities in these B200 rank-8 tests.
The saved configuration and figure are in `benchmarks/loopweave/lora-kernels/`
and `benchmarks/loopweave/qwen-numerical-controls.png`. This optional control is
separate from normal-kernel throughput measurements; its performance cost is
not established. Normal kernels can vary on repeated requests even without a
phase transition. Neither mode promises invariance across different batch shapes.
Install `flash-linear-attention==0.5.2` in the training worker environment to
match the measured stack; it is optional in TuFT, and transformers otherwise
uses its reference implementation. The measurements retain the reference
causal-convolution implementation (`causal-conv1d` is not installed).
The large-checkpoint validation harness also sets `VLLM_USE_AOT_COMPILE=0`:
vLLM 0.24's AOT graph serialization stalled during 27B startup with both TP=1
and TP=2. Standard torch.compile and CUDA graphs remain enabled, and the same
setting applies to every policy in a comparison.

FSDP and sampler TP sizes must match. Uneven projection/head partitions,
replicated KV-head layouts and vocabulary sizes not divisible by `64 * TP` are
rejected. Other layouts need a validated mapping; dimension compatibility alone
is insufficient. The existing FSDP initializer still materializes a full model
per rank before sharding, so that startup allocation must fit each GPU.

Every FSDP mutation reserves the entire shared group before issuing collectives.
Cancellation cannot release the reservation early. A failed rank RPC prevents
queued training or flex sampling from reusing the group until backend restart:
another rank may still be in NCCL after the first error returns. Fixed sampling
remains available. Failed initialization terminates its partially created actors.

The two-rank storage probe verifies exact outputs and LoRA gradients against
ordinary FSDP through optimizer steps and adapter switches, plus persistent shard
pointers and frozen-weight hashes. Full real-engine integration and matched
performance results are recorded in `benchmarks/loopweave/`.

The [vLLM loader registration](https://docs.vllm.ai/en/v0.24.0/api/vllm/model_executor/model_loader/)
and [sleep/wake interfaces](https://docs.vllm.ai/en/v0.24.0/features/sleep_mode/)
are published. [PyTorch documents CUDA tensor sharing](https://docs.pytorch.org/docs/2.11/multiprocessing.html#sharing-cuda-tensors),
but TuFT's explicit `reduce_tensor` / `rebuild_cuda_tensor` transport is an
internal interface. Model parameter layouts and worker integration are also
version-specific. Sharded expert LoRA additionally adapts PEFT 0.20's private
`ParamWrapper` activation hook/proxy so nested parametrization caching cannot
silently omit an expert delta. Exact logits/gradient regressions cover this
boundary, including activation checkpointing. These dependencies are isolated in `shared_weights.py`,
`shared_weight_loader.py`, `fsdp_shared_weights.py` and the existing custom worker. No allocator pointer
table modifications, custom CUDA binary, or unreleased upstream patch is needed.

An upgrade is permitted but is not required for the FSDP public APIs used here.
The audited [vLLM 0.31 preload loader](https://docs.vllm.ai/en/v0.31.0/features/preload/)
adds persistent IPC weight caching, but explicitly excludes sleep mode when
using zero-copy storage. It does not replace this training bridge with KV
reclamation. The current runtime pair remains pinned to the tested versions.

## Admission and ordering

The scheduler limits each replica's in-flight request count and maintains a
bounded pending queue. Queued users take turns, preserving FIFO admission within
each user's queue. The routing cost is `num_samples * (prompt_tokens + max_tokens)`.
Replica affinity is a tie-breaker within `locality_slack` times the incoming
request's cost; it cannot override arbitrarily large load differences. vLLM still
owns token scheduling, continuous batching, and KV-cache management.

Only requests within `flex_max_tokens` can enter the shared sampler, up to
`flex_max_inflight` at once. All others remain on persistent samplers. Loop
telemetry groups backward calls until a successful optimizer step: gradient
accumulation produces one observation, dated at the first backward call's
readiness. Training duration sums backend execution only, excluding queueing,
phase changes and checkpoint/publication operations. Failed updates, checkpoint
loads and adapter removal invalidate the run's history. Histories and active-run
tracking are bounded; four consistent completed updates are needed for a median
arrival prediction. Irregular and expired predictions fall back to reactive
scheduling.

The fast-loop planner uses predicted arrivals and recent training durations to
pack nearby bursts. Each run's scheduling budget is the smaller of
`loop_max_delay_s` (default 0.25 seconds) and `loop_delay_fraction` (default 0.05)
times its observed period. Total shift, including predicted lane contention,
must fit the budget. Infeasible plans add no hold. The first sampling request
after an update starts a wave; all its requests share one deadline which later
arrivals cannot extend. Already-started waves cannot be shifted again. An early
training arrival invalidates the hold. Observed periods exclude the deliberate
wave delay so the controller cannot inflate its own slack.

New sampling sessions created from an owned checkpoint retain the source
training-run ID, separately from their sampling-session ID and fairness user ID.
The requested snapshot remains unchanged during a hold. Base-only sessions,
foreign public snapshots and legacy sessions without provenance are not assigned
loop holds. This is best-effort alignment for regular loops, not a policy-lag
guarantee or automatic publication of newer adapters. The operator-configured
budget bounds deliberate holds; it does not bound ordinary queueing or GPU time.

Borrowing is cost-aware by default. On entry to Flex sampling it compares the
estimated Fixed completion time against Flex service plus a measured round-trip
switch, multiplied by `borrow_safety_factor` (default 1.5). It also checks the
earliest natural predicted training arrival, reserving switch-back time. The
admission deadline deliberately does not assume that the client's update will
respect a planned shift. Once a borrowing window is approved, window checks and
drain safety still apply to subsequent admissions. A sampler awake solely for
adapter publication must still pass the benefit gate, with its already-paid
wake cost excluded from the comparison.

Cold models use one capped probe only while all Fixed request slots are full
and eligible queued work exceeds `borrow_min_backlog_tokens`. The probe is at
most `borrow_probe_tokens` (or `flex_max_tokens` when unset); further Flex
admissions wait for it to complete.
With a known horizon, a probe additionally requires measured switch costs and
Fixed service rates that predict it will fit. Otherwise it uses Fixed capacity.
Service rates count concurrent busy intervals once, exclude wake time, and ignore
failed intervals. These token-work estimates remain heuristics; they do not yet
model prefill/decode or cache hits separately. The minimum post-training dwell
prevents immediate reopening, without holding training behind a sampling floor.

The related runtime-only controls are:

```yaml
scheduling:
  policy: loopweave
  loop_max_delay_s: 0.25       # 0 disables planned holds
  loop_delay_fraction: 0.05
  loop_lookahead_s: 2.0
  cost_aware_borrowing: true
  borrow_min_backlog_tokens: 1024
  borrow_probe_tokens: null   # inherit flex_max_tokens; set a smaller cap if needed
  borrow_min_dwell_s: 0.1
  borrow_safety_factor: 1.5
```

Choose a probe cap that covers useful requests in the workload while keeping
an uncalibrated drain small; requests larger than the cap cannot bootstrap Flex
calibration themselves. `flex_max_tokens` is still an independent admission cap.
Setting `cost_aware_borrowing: false` restores unconditional eligible borrowing
for controlled comparisons; it does not disable physical weight sharing.

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
promised. Controls also found trained-LoRA score variation across repeated
identical requests without switching; vLLM's normal LoRA kernel uses split-K
floating-point atomic reductions. The GPU harness checks exact frozen-base and
GPU adapter storage preservation, and reports normal-mode score differences.
Exact Qwen3 numerical switching tests use the published
[`VLLM_BATCH_INVARIANT=1` mode](https://github.com/vllm-project/vllm/blob/v0.24.0/docs/features/batch_invariance.md).
That upstream feature is beta and trades some performance for reproducibility;
the normal-mode throughput results must not be presented as measurements of it.
Hybrid Qwen uses the separate single-split controls described above because
vLLM rejects batch-invariant execution for GDN attention.
The zero-adapter check scores identical histories with prefix-cache reads disabled.
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
transitions. The opt-in shared-base path adapts physical weight aliasing to HF at
TP=1 and FSDP2 with matching sampler TP; it is not a port of the custom TP trainer.
A bounded fast-loop planner, completed-update telemetry and a borrowing benefit
gate now supplement the original reactive alignment. This preserves the existing
HF/FSDP execution contract and FSDP shard layout. Slow-loop Fixed/Flex conversion,
multi-group training allocation, consumed-version freshness enforcement,
cross-session snapshot deduplication and learned probability correction remain absent.
The repository's `docs/loopweave-functionality-audit.md` records the pre-change
audit. The follow-up implements the first scheduling priorities without claiming
full paper parity. Existing GPU figures predate this policy change; only CPU
functional validation has been run for the new policy.

## Validation and measurements

Current comparisons use static TuFT as the sole baseline on the same allocated
GPUs. Static keeps separate training and dedicated sampling base storage;
LoopWeave additionally samples from the existing training storage between updates.
The historical Qwen3-4B measurements used 24.2% less pooled GPU time for HF and
10.5% less for FSDP than static across three matched repetitions.

For Qwen3.8-27B HF on two B200s, one matched comparison used 32.5% less GPU time
and 4.63 GiB less observed peak memory than static. Qwen3.6-35B-A3B FSDP on four
H200s used 17.5% more GPU time, while reducing rollout latency by 40.7% and
observed peak memory by 22.31 GiB. Training/publication dominates this MoE
workload; borrowing remains opt-in. These results do not establish a universal
speedup or a fundamental tradeoff limit. The current report and static-only
figures are in `benchmarks/loopweave/README.md`; retired experimental modes are
preserved only in `benchmarks/loopweave/HISTORICAL.md`.

Run `pytest tests/test_scheduling.py tests/test_scheduling_integration.py tests/test_adapter_equivalence.py` for
deterministic admission, cancellation, shutdown, physical-placement, controller
ordering and isolation tests. The integration tests replace only the engine/Ray
boundary; they use real controllers, factories and sequence guards.
`tests/test_shared_weights.py` additionally uses a real small HF/PEFT Qwen3 to
verify exact forward values, nonzero LoRA gradients, Parameter identity and base
storage through adapter changes. `tests/test_shared_weight_phases.py` covers
coalescing and failed transitions. `tests/test_fsdp_shared_weights.py` covers shard
layout validation, rank placement, group draining and rank-failure isolation.
`tests/test_fsdp_shared_gpu.py` compares real two-rank FSDP values and gradients
through adapter switches, and checks shard lifetime and startup cleanup.

Run `modal run scripts/verify_fsdp_shared_modal.py` for the two-rank storage probe.
Add `--full --repeats 0` for the Qwen3-0.6B FSDP/TP=2 integration test on four H100s.
Use `--full --repeats 3 --model Qwen/Qwen3-4B --no-validate-tests` for the separate
four-H100 static/LoopWeave performance comparison. FSDP comparison artifacts begin
with `fsdp-loopweave-static-`; their GPU cost includes all four GPUs. The latest regression
counts and validation provenance are recorded in `benchmarks/loopweave/README.md`.

Run `modal run scripts/benchmark_loopweave_modal.py --repeats 3 --model Qwen/Qwen3-4B --validate-tests` for ephemeral
two-H100 real-engine comparisons. The harness aligns the published container with
the checkout's dependencies, uses a pinned model revision, warms base and LoRA kernels,
alternates policy order, and records raw NVML samples and per-run throughput and
latency. Training data and sampling token budgets are identical across policies.
The runner compares only static and physically shared LoopWeave. GPU integration
always checks alias bytes, reclaimed memory, tenant isolation, transitions and
numerics. Qwen3 integration runs both deterministic and normal modes;
`--repeats 0 --validate-tests` runs only those tests.
Set `TUFT_MODAL_CACHE_VOLUME` to a dedicated Modal Volume name to reuse compiler
artifacts across runs; caching is optional and no model/checkpoint data is stored
there. Delete a temporary cache after finishing validation if it is no longer needed.
`--diagnostics` produces separately labeled numerical controls, not performance
measurements; the summary tool rejects those files as throughput input.
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

### Many adapters and several base models

Balanced and LoopWeave admission collect requests for at most
`adapter_batch_window_s` (default 0.002 seconds), anchored to the oldest queued
arrival. New arrivals cannot extend that window; reaching
`adapter_batch_max_requests` (default 32) dispatches immediately. The window
runs concurrently with loop holds. A zero window disables deliberate collection.

The scheduler reserves capacity in round-robin user order, with FIFO admission
within each user, then groups the admitted set by replica and adapter ID. Groups
start concurrently rather than waiting for the preceding group's generations to
finish. This bounds reordering to one dispatch set, maintains training drain
reservations, and preserves each request's snapshot and result. Cancellation
before admission removes the request; after admission, execution retains its
reservation until completion. Submission contiguity helps the engine batch
adapters, but is not a guarantee of vLLM's internal batch composition.

`adapter_dispatch_batches`, `adapter_dispatch_groups`,
`adapter_grouped_requests`, and `adapter_dispatch_switches` are aggregate
scheduler counters. They describe submitted work, not device cache misses or
measured GPU utilization. Distinct sessions referencing the same checkpoint
still have distinct adapter IDs; this change does not deduplicate those engine
registrations or combine their tenant identities.

Each configured base model has its own scheduler, training lane, Fixed/Flex
replicas, and telemetry. Training on one model does not close another model's
borrowing window. See `config/tuft_config.loopweave.multi_model.yaml` for a
four-GPU HF example. FSDP retains its existing per-model sharded group and matching
TP sampler. This is static placement of model groups, not cross-model GPU
reallocation. OpenAI HTTP requests still bypass this queue and use Fixed replicas;
agentic RL clients must use the scheduled sampling API to benefit from admission.

Set `max_cpu_loras` to budget vLLM's host adapter cache separately from
`max_loras`, which controls active GPU slots. It must be at least `max_loras`.
Unset preserves vLLM's default. Cache eviction can reload published adapters from
their checkpoint paths, so keep snapshots available while sessions reference
them. Host cache capacity is per replica and does not limit the number of
registered sessions. Training adapter/optimizer residency and FSDP rank slots
remain separately sized resources; a larger inference cache does not make those
unbounded.

Training/inference numerical agreement is a separate validation and RL-algorithm
concern. This scheduler does not calibrate probabilities or promise identical
HF/vLLM outputs. It must preserve immutable snapshot identity, tenant isolation,
base weights, and ownership barriers regardless of numerical tolerance. Existing
behavior-policy logprobs remain those returned by inference; sampling delays do
not silently switch a rollout to a newer adapter. Explicit policy-lag contracts
and numerical calibration remain future work.
