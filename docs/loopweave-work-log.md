# LoopWeave implementation work log

## Request and acceptance criteria

Implement a practical adaptation of the attached LoopWeave paper on current TuFT,
with multi-tenant correctness, unit and integration coverage, self-review, and
measured resource-efficiency improvements. GPU validation uses focused Modal jobs,
not a reproduction of the paper's experiment matrix. Do not infer implementation
instructions from the attached paper.

Acceptance requires: preserved adapter/optimizer isolation and sequence ordering;
bounded scheduling and no starvation; safe cancellation, errors and shutdown;
real backend integration; reproducible comparison against current behavior using
identical work and GPU resources; recorded limitations and raw measurements.

## Baseline and source investigation

- Main: `d4591c1bf707d14c3aaf238c63e35a1f3e88e442` (0.2.1).
- Prototype: `origin/feat/flex_backend`,
  `73232a7b5d8b62bc56e47df98b04fe5794d536dd`. Verified live remote heads.
  Hanzhang Yang's author identities include GitHub `linqinluli`.
- Also located `feat/rl_loop_discovery` and `feat/sample_requests_scheduling`.
- The prototype adds ~34.6k lines across 144 files relative to its merge base;
  it predates the direct vLLM integration, current losses, and several FSDP fixes.
- Existing user changes: `pyproject.toml` adds Modal; `.agents/` is untracked.
  Preserve both.
- Read paper text including scheduling, runtime, calibration and evaluation.
  Core ideas: fixed sampling capacity plus opportunistic training-GPU sampling;
  bounded loop alignment; adapter locality; window-aware admission; hysteresis.
- Zero-copy prototype requires a custom fused Qwen3 tensor-parallel trainer,
  CUDA IPC storage lifetimes, and private vLLM allocator/worker hooks. Current HF
  and FSDP storage cannot simply be aliased into that runtime. Its trainer also
  differs from main in LoRA scaling and loss coverage.
- Current DP sampler routes round-robin without measuring outstanding work.
  Current colocation reserves memory fractions but does not coordinate execution.
  The OpenAI path bypasses Tinker sampling, so any shared-GPU execution gate must
  explicitly handle that path as well.

## Working design and validation sequence

1. Add an opt-in scheduler on the existing backend contracts. Bound in-flight
   sampling, account for outstanding token work, preserve adapter identity, and
   expose measurements. Retain an explicit baseline for comparisons.
2. Integrate fixed sampling replicas with a colocated HF/vLLM replica, lending
   idle training compute to sampling. Stop admission and drain before training;
   keep a fixed replica for long/unknown requests and OpenAI streaming traffic.
   Use existing engines and memory budgets; do not claim zero-copy weight sharing.
3. Exercise correctness and cancellation adversarially; measure heterogeneous
   workload scheduling and real GPU end-to-end training/sampling.
4. Review measurements and implementation, correct regressions, repeat focused
   validation, and document measured scope and operational limits.

Calibration is not an automatic score transformation: the paper explicitly says
its corrected scores are not behavior-policy probabilities. Preserve existing
sampling probabilities unless a separately validated API contract supports this.

## Progress

- Environment: existing `.venv` has torch 2.11.0+cu130, transformers 5.19.0,
  Modal 1.6.1; no local CUDA device. Installed pypdf for attachment extraction.
- No implementation or performance claims yet.

## Implementation iterations

- Added opt-in `balanced` and `loopweave` model scheduling policies. The default
  remains the original round-robin path. Shared mode is explicitly HF/TP1 with
  one shared sampler and at least one fixed sampler; FSDP continues unchanged.
- Added bounded per-user FIFO queues with round-robin admission, outstanding-token
  routing and bounded adapter affinity, conservative loop-horizon prediction,
  fixed fallback, and aggregate diagnostics.
- All shared-GPU training mutations and sampler adapter registrations close and
  drain sampling admission. Startup verifies the actual node/GPU placement.
- OpenAI traffic remains on fixed replicas. Disabled the shared replica's HTTP
  server so requests cannot bypass the execution gate.
- Self-review corrections: admitted training cancellation must finish the sequence
  guard (otherwise a retry duplicates an optimizer mutation); shutdown cancellation
  must not release live GPU reservations; backend timeouts must not be mislabeled
  as queue timeouts; token estimates must match the backend's actual default.
- Added controller integration tests with controllable engine boundaries, plus
  deterministic tests for fairness, bounded queueing, cancellation, failure,
  shutdown, loop prediction and invalid layouts. Current focused checks pass.
- Added real-engine Modal harness: two H100 GPUs, pinned Qwen3-0.6B revision,
  multi-tenant sampling/training, raw NVML observations, and adapter/checkpoint
  isolation assertions outside measured intervals.

## Validation issues and resolutions (no performance conclusions yet)

- Restricted local socket access prevents existing mock-HTTP tests; running the
  authorized tests with socket access resolves those errors.
- The published `ghcr.io/agentscope-ai/tuft:latest` currently contains vLLM 0.14
  and an older torch stack. The harness upgrades to this checkout's dependencies
  and removes the old FlashAttention wheel, which has an incompatible torch ABI.
- GPU startup then exposed missing CUDA 13 NVRTC library discovery by cuDNN;
  the harness now explicitly places the pinned wheels' NVIDIA library directories
  on `LD_LIBRARY_PATH`. These are environment fixes shared by both policies.
- Broad CPU regression reached completion; one FSDP factory test and OpenAI test
  setup failed because the host's default temporary path exceeds Ray's 107-byte
  Unix socket limit. Those tests all passed when retried with
  `TMPDIR=/tmp RAY_TMPDIR=/tmp`.
- Earlier prototype branches were also inspected: `feat/rl_loop_discovery`
  currently contains mismatch-probe/FSDP changes, not the later loop scheduler;
  `feat/sample_requests_scheduling` contains a 462-line adapter coalescing wrapper.
  The complete hierarchical runtime and experiments are on `feat/flex_backend`.

## First measured iteration

- Two H100s, Qwen3-0.6B, four tenants, 12 updates and 12,288 sampled tokens:
  static allocation 18.24 s; constrained hybrid 26.91 s (0.678x throughput).
  Both passed real adapter isolation, checkpoint restoration and new-snapshot
  sampling. Raw data: `benchmarks/loopweave/gpu-pilot-constrained.json`.
- This is a regression, not a success: increased aggregate SM utilization did
  not translate into efficiency. The pilot's fixed/flex limits (16/8) restricted
  a 32-request wave and produced extra queueing; p95 increased 5.45 to 9.83 s.
  Rerunning the same workload with fixed/flex limits 32/16.
- Added the paper's verified zero-adapter fast path, with fail-closed checks for
  finite factors, complete compatible A/B pairs and absence of extra state. New
  CPU checks plus scheduling/controller checks pass (32 tests).
- Strengthened subsequent benchmarks to publish trained snapshots between rounds
  and compare the verified-zero path to normal LoRA on the same engine. Added
  pinned Qwen3-4B as a more compute-intensive functionality/performance case.
  Earlier pilot data remains separate; workloads must not be compared across
  the snapshot-publication change.
- Tuned 0.6B pilot: static 21.78 s, hybrid 22.05 s, effectively no speedup.
  Inspection of per-replica counts exposed an admission bug: all 101 requests
  were placed on the fixed sampler during the post-training quiet interval,
  leaving the extra sampler unused. Added a bounded sampling-alignment delay
  that lets the next rollout use newly freed capacity, with a fixed-replica
  fallback when its budget expires. Regression tests cover both outcomes.

## Further performance review

- The first 4B updated-snapshot workload regressed: 25.01 s static versus
  30.05 s hybrid. Bounded alignment correctly activates both replicas, but the
  eager-mode aligned run still regressed (28.38 versus 32.45 s). Zero-adapter
  greedy tokens and log probabilities matched exactly in the latter run.
- GPU logs explicitly reported first-use LoRA Triton compilation on the shared
  sampler. Added nonzero-LoRA warmup on every replica, plus separate measured
  scheduler counters so initialization cannot masquerade as GPU borrowing.
- The small eager-mode waves are largely launch-bound. A separate throughput
  case now uses CUDA graphs for **both** policies, 16 CPU cores, one OpenMP/MKL
  thread per process, and 32 samples per tenant (128 concurrent requests), with
  fixed/flex admission limits 128/64. These are a new workload/environment;
  only policies within each matched pair will be compared.
- Self-review caught overload being surfaced as a generic server error. It now
  inherits TuFT's existing resource-exhausted exception (429), with a regression
  assertion. Round-robin behavior remains the default.

## Final validation preparation

- The CUDA-graph 128-token pilot completed identical 49,152-token work in
  11.516 s static versus 11.013 s shared (1.046x), with mean sampling latency
  2.229 versus 1.678 s. Each replica completed 192 requests during measurement.
  This single modest gain is not treated as a stable performance conclusion.
- The final run alternates policy order across three repetitions using 512-token
  rollouts (196,608 tokens, 12 training steps, eight publications per policy/run).
  It also executes the actual two-GPU pytest integration test. Earlier shorter
  and eager-mode results remain in the report rather than being discarded.
- Full local test hook: 612 passed, 32 GPU tests skipped. Lint, formatting,
  isolated type checking and secret scanning all pass. Fixed the type-check
  hook's Tinker constraint to match pyproject's existing >=0.25,<0.26 range;
  its previous open-ended >=0.18.2 installed an incompatible API. Public model
  revision hashes are explicitly allowlisted for the secret scanner.
- Implementation commit: `26afb88`. Existing user edits to pyproject.toml and
  .agents remain separate. Stale sandbox test processes were stopped after
  successful replacement runs with the required local socket access.
- Final self-review found that adding a default scheduling field to persisted
  AppConfig signatures would block restart of existing servers. Scheduling is
  now excluded as runtime-only metadata, alongside capabilities. A regression
  reconstructs the old signature, allows a scheduling-only change, and still
  rejects LoRA-rank drift. This follow-up changes persistence serialization only;
  the in-progress GPU runs use commit 26afb88 with persistence disabled, so their
  timed execution path is unaffected. Targeted persistence checks pass.
- A second late review correction measures shared-sampler service rate over a
  complete concurrent busy interval. Per-request elapsed time double-counted
  that interval and could overestimate drain time after stable loop histories
  formed. Failed intervals do not update the estimate. Deterministic tests cover
  both cases. The three-round performance runs had no window rejections; the results therefore
  demonstrate borrowing/alignment, not a learned-prediction performance benefit.
  A separate five-round real-GPU integration run checks the final source.

## Completed results

- Three matched 4B / 512-token pairs all improved: static/shared seconds were
  27.196/21.501, 25.406/20.205 and 26.451/20.521. Median paired throughput gain
  is 26.5% (range 25.7–28.9%). Pooled allocated GPU-seconds per training step fell
  4.392 to 3.457 (21.3%); mean sampling latency fell 6.828 to 4.675 s (31.5%).
- Tradeoffs: mean training latency increased 1.453 to 1.659 s (14.2%); the shared
  GPU retained substantially more memory. Earlier eager-mode regressions remain
  in the report. This is a workload-specific steady-state result, not a claim
  about startup cost, universal speedup, task quality, or the paper's full system.
- Each measured shared run completed 192 requests on each replica, with no window
  rejections. Every policy/run passed real adapter-isolation, checkpoint-restore
  and new-snapshot assertions; zero-adapter greedy tokens/log probabilities
  matched exactly in all shared runs.
- Final source commit `5bd0005`: all repository hooks pass, 614 local tests pass
  (32 GPU tests skipped locally), and the separate five-round real-GPU test passes
  on two H100s: 10 updates, eight publications and 10,240 generated tokens.
  Its recorded source hash matches the local final source. The first GPU pytest
  launch found an unrelated installed `scripts` package shadowing the checkout;
  loading the harness by its explicit checkout path resolved it.
- Raw measurements, all earlier pilots, JUnit evidence, summary JSON and chart:
  `benchmarks/loopweave/README.md`. Reproduction commands and the measured H100
  configuration are linked there. No persistent service was deployed.
- Artifact commit secret scanning flagged generated SHA-256 provenance strings.
  Reviewed only the flagged source/harness fingerprint fields and added precise
  false-positive entries to .secrets.baseline; no detector was disabled and no
  credential was added. The final timed source and harness hashes were checked
  against commit 26afb88, and the GPU-test hash matches commit 5bd0005.

## Shared base weights follow-up

The user correctly identified a missing central mechanism: the first adaptation
borrowed GPU compute but retained separate HF/vLLM base weights. The feature
branch already implements CUDA IPC aliases with a custom tensor-parallel trainer.
The paper also retains separate vocabulary allocations where layouts differ and
publishes updated LoRA snapshots separately; those are not omissions.

Implementation in progress: preserve the current HF trainer and its LoRA/loss/
checkpoint semantics. For dense BF16 Qwen3 on TP=1, pack frozen q/k/v and gate/up
weights once and make the original HF Parameters views into that storage. A
custom vLLM loader binds CUDA IPC tensors before profiling and graph capture.
This avoids prototype allocator-pointer-table edits. Documented level-1 sleep
and wake reclaim KV memory at scheduler drain boundaries; any private vocabulary
padding and LoRA buffers remain the sampler's responsibility. The producer stays
alive until sampler shutdown. Opt-in `scheduling.share_base_weights` fails closed
outside torch 2.11.0 / vLLM 0.24.0 and the supported layout.

Compatibility audit: CUDA IPC itself is public, and PyTorch documents sharing
CUDA tensors through multiprocessing. Explicit `reduce_tensor` /
`rebuild_cuda_tensor` calls remain an internal interface needed by this Ray
transport. vLLM's loader extension and sleep/wake interfaces are published, but
model parameter names/layouts and the worker lifecycle remain version-specific.
The public vLLM 0.24 IPC weight-transfer engine loads received tensors into model
weights; that does not establish persistent trainer/sampler storage aliasing.

Validation plan: exact CPU HF outputs and LoRA gradients before/after packing,
storage/shape/dtype failures, scheduler phase ordering and failure handling,
real CUDA alias and memory-release checks, five-round tenant/checkpoint tests,
then matched static / duplicate-base / shared-base measurements. Historical
benchmark artifacts will remain unchanged.

- First GPU test: existing resident-copy mode passed. Shared mode verified
  aliases through several sleep/wake cycles and reclaimed 41.66 GiB (only
  0.17 GiB backed up; the shared base stayed resident), but its greedy output
  differed from the fixed sampler. This is a correctness blocker under diagnosis,
  not an accepted result. Added full base-parameter hashes before/after switching.
- The same run exposed unreliable automatic cache sizing: a nominal 45% budget
  produced a 41.43 GiB KV pool. CUDA IPC and released loading allocations do not
  fit vLLM's process-local accounting assumptions. Added the documented explicit
  `sampling_kv_cache_memory_bytes` option, required for shared-base mode and
  excluded from checkpoint compatibility signatures. Comparisons will give every
  sampling engine the same 16 GiB cache budget.
- CPU suite: 622 passed / 33 skipped before the explicit cache-budget addition.
  Sandboxed threaded async tests stalled; the identical test passed outside the
  sandbox, and the full unrestricted suite completed in 51.85 seconds.
- Follow-up runs verified identical base-parameter hashes at startup and after
  sleep/wake, with 7.49 GiB of unique shared base storage. With the explicit
  cache budget, sleeping reclaimed 16.21 GiB. Long greedy sequences diverged
  after 61–97 tokens; teacher-forcing identical histories exposed a maximum
  log-probability difference of 0.25. Numerical acceptance remains pending a
  duplicate-engine control and within-engine before/after comparisons. No
  throughput result from these diagnostic runs is accepted as final evidence.
- The buffer diagnostic initially failed on a scalar tensor's byte view;
  flattening before viewing fixes this, with a scalar/BF16 regression test.
- Current local regression: `TMPDIR=/tmp .venv/bin/pytest` passed 625 tests,
  with 33 GPU-dependent skips (89.64 s). The inherited longer temporary path
  exceeds Ray's UNIX socket pathname limit. Pyright reports 0 errors and four
  existing missing-source warnings; its hook observed concurrent edits, so the
  checker result and hook bookkeeping are reported separately.
- Automatic approval review rejected adding three existing placeholder/checksum
  findings to the secret-scanner baseline as an unauthorized suppression. The
  baseline remains unchanged; these findings are separate from implementation
  correctness and must not be silently suppressed.
- Isolated compiled controls resolved the sharing-specific concern. The shared
  engine's entire 159-position fixed-input score vector was bitwise identical
  to an earlier ordinary engine. Later ordinary engines differed from each
  other by up to 0.2578, and from the earlier ordinary engine by up to 0.3226.
  Each engine was bitwise stable before/after warmup or sleep/wake, and all
  non-LoRA model-buffer hashes matched. Computation graphs and Inductor cache
  keys also matched. This establishes baseline execution variation; it does
  not identify the precise kernel-level cause. Raw controls are retained under
  `benchmarks/loopweave/diagnostic-*.json`.
- An eager-mode shared-vs-ordinary control produced exactly equal fixed-input
  scores before and after sleep/wake, including the generated continuation
  when teacher-forced. Long greedy sequences still diverged with different
  cache histories, demonstrating why that is a poor storage-equivalence test.
- Acceptance now requires exact full frozen-parameter hashes and exact
  same-engine fixed-input scores across base and trained-adapter sleep/wake
  transitions. Cross-engine errors are reported rather than silently hidden
  or used with an arbitrarily widened tolerance. The zero-adapter probe now
  teacher-forces identical inputs and disables prefix-cache reads, retaining
  its existing 0.005 tolerance.
- The trained-adapter exact-score assertion also needed a proper control:
  repeated identical requests *without switching* differed by 0.1249666, while
  the sleep/wake comparison differed by 0.1244316. Every actual GPU LoRA slot
  was byte-identical across the transition. vLLM 0.24's normal LoRA shrink
  kernel uses split-K floating-point atomic reductions; its published
  `VLLM_BATCH_INVARIANT=1` mode selects deterministic kernels (including
  split_k=1). Validation now separates exact numerical checks in that mode
  from default-mode functional/performance checks, which preserve and report
  both repeat and switching score differences. This avoids pretending that
  the backend is deterministic or hiding a widened tolerance.
- Added opt-in Modal compiler caching via `TUFT_MODAL_CACHE_VOLUME`, including
  cache commits on failed validation, to reduce repeated compilation. No model
  or result data is stored in that cache. Final results remain local artifacts.
- Both final five-round shared-weight GPU integration modes passed on two H100s:
  normal execution and batch-invariant execution. Each completed ten optimizer
  steps, eight publications, 10,240 generated tokens and 12 sleep/wake cycles.
  Both verified 7.492 GiB of unique physical base aliases and 16.207 GiB released
  during training, frozen base hashes, GPU LoRA byte preservation, isolation and
  checkpoint restoration. Deterministic repeat/switch score errors were exactly
  zero. Normal repeat/switch errors were 0.136406 / 0.124953 and are retained.
  Evidence: `shared-weights-gpu-integration-Qwen3-4B.json`, with both JUnit reports,
  and `shared-numerical-summary.json` under `benchmarks/loopweave/`.
- Final performance: three matched triples, rotating execution order, identical
  16 GiB KV budgets per sampler, Qwen3-4B, four tenants, two H100s, normal graph
  execution. Every run completed 196,608 tokens, 12 optimizer steps and eight
  publications. Shared versus static achieved 1.294x median paired throughput
  (range 1.294–1.370x), 24.2% lower pooled GPU-seconds per step and 34.6% lower
  mean rollout latency. Training latency including admission rose 22.7%.
- Against duplicate-base borrowing, sharing reduced observed peak total memory
  from 64.35 to 55.34 GiB (9.01 GiB / 14.0%). Its median paired throughput ratio
  was 0.982x (range 0.980–1.043x), while pooled GPU time differed by only 0.17%.
  This establishes reduced memory with comparable throughput, not a speedup
  from physical sharing alone. The static policy retains the lowest peak memory.
  Measured transitions averaged 67.2 ms sleep / 114.0 ms wake (nine each).
  No prediction-window rejections occurred; learned prediction is not the
  demonstrated source of improvement in this short workload.
- Raw performance, summary and figure: `shared-weights-gpu-Qwen3-4B-512.json`,
  `shared-weights-summary.json`, `shared-weights-performance.png`. Timed source
  and benchmark fingerprints match the final checkout. The GPU integration
  source fingerprint also matches; its harness predates only the three-policy
  execution-order rotation. Initialization/temporary loader allocations and
  correctness probes are excluded from timing; memory is sampled every 200 ms.
- Final local regression: 625 passed / 33 skipped in 55.22 seconds. Added a real
  PEFT transition assertion covering a second tenant with another rank and an
  additional target module: base pointers survive and the original adapter's
  outputs restore exactly. Pinned lint/format checks pass. Type checking covers
  new untracked modules explicitly, as well as tracked changes. The scanner
  baseline remains unchanged; recorded SHA-256 digests also produce findings.
- Remaining operating limits are explicit: opt-in dense BF16 Qwen3 TP=1,
  torch 2.11.0 / vLLM 0.24.0, temporary duplicate allocation during startup,
  producer-lifetime coupling, separately published snapshots and no live-weight
  sampling during optimizer mutation. Further KV-budget/phase-policy tuning may
  improve the measured tradeoff; these results do not claim its theoretical limit.
- Final pinned Ruff, formatting and Pyright hooks passed on every changed Python
  file, explicitly including new modules; `git diff --check` passed. Read-only
  scanning of changed files found 14 high-entropy SHA-256 digest entries in raw
  evidence, all reviewed as source/harness or model-buffer checksums. No scanner
  control or baseline was changed. `shared-validation.json` records checks and
  GPU run links. All runs finished, and the task-created Modal compiler cache
  `tuft-shared-validation-20261011` was deleted; local evidence is retained.

## FSDP extension and version audit (2026-10-11)

The user authorized newer torch/vLLM versions and requested both trainer backends.
The design preserves frozen-base sharding in FSDP. Replicating the entire frozen
base via `ignored_params` would change its memory tradeoff and is not the chosen
implementation. Only small normalization weights are replicated to match vLLM.

- Audited released vLLM 0.31.0 (Oct 5), which pins torch 2.13.0 and adds the
  published `vllm preload` / `ipc_cache` path. Its zero-copy mode explicitly
  excludes sleep mode, and its daemon supplies vLLM-format weights, not FSDP
  training ownership. An upgrade alone does not replace the training bridge.
  Sources: <https://docs.vllm.ai/en/v0.31.0/features/preload/> and
  <https://github.com/vllm-project/vllm/blob/v0.31.0/requirements/cuda.txt>.
- torch 2.11 already supplies public `shard_placement_fn`, `ignored_params`,
  `DTensor.from_local`, and `load_state_dict(assign=True)` support. FSDP's
  registered state-load hooks refresh its all-gather input storage; writing
  `.to_local().data` alone would leave stale internal references. No private
  FSDP parameter-state mutation is needed. Source:
  <https://docs.pytorch.org/docs/2.11/distributed.fsdp.fully_shard.html>.
- Arrange frozen q/k/v, gate/up, embedding and output-head shards by rows;
  attention output and MLP down-projection shards by columns. Pack only the
  local fused groups, refresh FSDP references before optimizers exist, and export
  persistent local shards to matching vLLM TP ranks. Full training layers remain
  temporary all-gather buffers and are never exported.
- FSDP owns the entire reserved GPU group. The colocated engine is created after
  rank placement, on the same node with precisely the rank-ordered physical
  devices. The shared scheduler drains the entire sampler group before any
  training collective; adapter/checkpoint mutations use the same reservation.
  Uneven heads, replicated KV-head layouts, misaligned vocabulary partitions,
  mismatched FSDP/TP sizes, and cross-node sharing fail closed.
- Added a two-rank GPU storage/numerical probe, local layout/placement tests and
  FSDP cancellation/drain regression coverage. Initial probe infrastructure
  needed a remote import-path fix; the tiny model's eight-dimensional attention
  heads rejected cuDNN SDPA, so its storage-equivalence check uses eager attention.
  Full-model integration retains the production attention configuration.
- Two-rank storage probe passed on two H100s (24.07 s): exact forward values and
  LoRA gradients versus ordinary FSDP across three optimizer steps and alternating
  adapters; persistent frozen shards retain their pointers and hashes. It checks
  real DTensor base and LoRA parameters, including column-sharded projections.
  Evidence: `benchmarks/loopweave/fsdp-shards-validation.json`.
- Self-review added a failure guard: an early rank failure may leave peers inside
  NCCL, so shared sampling and already-queued updates cannot reuse that group.
  Rank RPC failures mark it unusable until restart; the phase callback and the
  operation check after acquiring the reservation both enforce this. A local
  regression proves only the first failed optimizer dispatch occurs, flex
  sampling fails closed, and fixed sampling remains available. Failed startup
  also terminates every actor created by that initialization attempt.
- Full Qwen3-0.6B FSDP/TP=2 integration passed on four H100s (318.38 s including
  initialization): ten optimizer steps, eight publications, 10,240 tokens and
  12 group sleep/wake cycles. Both ranks verified physical aliases, totaling
  1.110 GiB of unique base storage; training reclaimed 32.180 GiB. Deterministic
  repeat/switch score errors were zero. Both normal and deterministic HF
  regressions also passed on two H100s after the multi-rank loader changes.
- Startup memory review found that shard-placement dictionary keys kept the
  original full Parameters alive until after `empty_cache`. The dictionary is
  now cleared immediately after FSDP wrapping so cleanup can release those
  allocations. The incomplete first 4B performance job was stopped before any
  timed result; the final comparison was restarted with this fix and finalized
  harness annotations. A GPU weak-reference assertion checks that full
  parameter objects are gone when cleanup runs.
- The final GPU storage probe passed with that weak-reference assertion (23.02 s).
  The final local full suite before this allocation-lifetime-only cleanup passed
  635 tests / 34 skipped in 54.87 s; focused cleanup validation uses the real
  two-rank CUDA/FSDP path. Pinned lint/format and isolated type checks cover all
  changed and newly added Python modules.

- Final local regression after startup cleanup: **635 passed, 34 skipped** in
  **51.75 s**. Pinned Ruff 0.14.14 lint/format and Pyright 1.1.408 passed all
  changed/new Python files. Both shared configurations validate. The user's
  `pyproject.toml` Modal addition and `.agents/` files are preserved.
- Completed all nine Qwen3-4B runs on four H100s, with static, duplicate-base and
  shared-base policies each occupying every order position once. Every run
  generated 196,608 tokens with 12 optimizer steps and eight publications;
  all four tenants have finite losses and equal token counts. Correctness probes
  pass before timing. Final source/harness fingerprints match the raw artifact.
- FSDP shared versus static: **1.091× median paired throughput** (1.086–1.177×),
  **10.5% less pooled GPU time per step**, and **30.3% lower rollout latency**.
  Training latency including admission is 33.2% higher. Shared versus duplicate:
  observed peak **109.01 versus 122.19 GiB** (13.18 GiB / 10.8% less), with
  **4.6% more pooled GPU time** and a **0.946× median paired throughput ratio**
  (0.922–1.003×). Static peak remains lowest at 73.16 GiB. This is an explicit
  memory/throughput tradeoff, not a claim of universal dominance or optimality.
- Every shared run verifies both ranks, **7.493 GiB** of unique aliases, zero
  copied base parameters, **32.332 GiB** of sampler memory released for training,
  unchanged base hashes and GPU adapter bytes. Three timed sleep/wake cycles per
  run average 56.7 / 117.6 ms. No prediction-window rejections occur, so the data
  does not isolate a benefit from loop prediction. Normal-mode score variation
  remains recorded; deterministic integration checks are separate.
- Saved raw measurements, summary and chart under `benchmarks/loopweave/fsdp-shared-*`;
  `dual-backend-validation.json` records test provenance and remaining limits.
  GPU app `ap-l7zGoJ4H7Gs8GTBnwdk7So` completed and its container list is empty.
  No persistent cache was created. Read-only secret scanning finds recorded
  evidence digests; no production Python findings or baseline changes.
- Self-review covered physical rank mapping, persistent versus gathered FSDP
  storage, optimizer identity, adapter/checkpoint ordering, cancellation, early
  collective failure, startup allocation lifetime, both backend regressions,
  balanced benchmark ordering, equal work and exact source provenance. The
  implementation remains opt-in, dense BF16 Qwen3, and version guarded. Further
  workload-specific KV sizing, startup allocation reduction and scheduling
  tuning remain opportunities, not measured claims in this validation.

### Qwen hybrid dense and MoE extension (in progress)

- Confirmed scope: Qwen3.8-27B dense and Qwen3.5/3.6 MoE, starting with
  Qwen3.6-35B-A3B; Qwen3.8-Flash-Next is excluded. Text-only flows preserve
  frozen-base FSDP sharding. Existing Qwen3 benchmark numbers above predate
  this extension and are not evidence for hybrid-model performance.
- Added hybrid Gated DeltaNet projection/conv layouts and expert-matrix
  placements. Public torch parametrizations reconstruct logical HF ordering
  after FSDP all-gather while keeping persistent storage in TP shard order.
- Exact CPU logits/LoRA-gradient tests pass for dense and MoE with both layouts.
  Nested PEFT parameter wrappers required an isolated private compatibility
  hook to avoid caching a base-only expert weight before adding its LoRA delta.
  GPU correctness, recurrent-cache transitions, full checkpoint validation,
  performance measurements and saved figures remain pending.
- Expanded two-H100 FSDP probe passed all three model families in 45.13s,
  comparing three optimizer steps and alternating two adapters against ordinary
  FSDP with exact logits/gradients and unchanged persistent base storage.
  A follow-up also enables production activation checkpointing and FLA kernels.
- Full CPU suite: 639 passed, 38 skipped, 53.61s. The first real hybrid serving
  run exposed vLLM 0.24's explicit rejection of `VLLM_BATCH_INVARIANT=1` for
  GDN_ATTN. Hybrid validation now uses ordinary mode and retains exact storage
  checks; Qwen3's separate deterministic evidence remains valid for Qwen3 only.
- The validation image now pins flash-linear-attention 0.5.2. Text-only
  sampling is configurable for all replicas, keeping unused vision encoders
  out of the hybrid comparisons. Large checkpoints await the serving probes.
- Real vLLM dense and MoE probes passed (2 tests, 281.14s, H100). They verify
  all base aliases, nonzero GDN-only/dense and expert-only/MoE adapter effects,
  and exact token/log-prob preservation over sleep/wake. Saved JUnit evidence
  in `qwen-vllm-layout-validation.json`.
- Optimized FSDP validation passes with production activation checkpointing
  and 128-wide GDN heads (3 tests, 153.84s, two H100s). The 8-wide CPU fixture
  stalls in FLA autotuning on CUDA; changing CUDA loading/prefetch did not fix
  that artificial shape. Production FSDP settings remain unchanged, and the
  CUDA fixture now uses actual Qwen head widths. CPU layout tests retain small
  dimensions and also pass with activation checkpointing (4 tests, 10.78s).
- Qwen3.5-0.8B real-checkpoint HF integration has exact initial frozen-weight
  hashes across independent and shared samplers. Full-checkpoint validation is
  now running for Qwen3.8-27B and Qwen3.6-35B-A3B with both backends. Planned
  performance contrasts use 27B/HF and 35B-A3B/FSDP, equal 4-GiB KV budgets,
  Triton GDN prefill and MoE kernels across all policies, and 128 output tokens.
- Full CPU suite now passes 640 tests (38 GPU skips) in 53.24s. Pinned Ruff
  lint/format pass; the pinned type checker caught overly narrow inferred
  kwargs types in two model fixtures, now annotated explicitly.
- Qwen3.8-27B HF startup correctly rejected a recurrent-cache capacity mismatch:
  128 requested sequences versus 83 available Mamba blocks with a 4-GiB KV
  budget. Added `sampling_max_num_seqs` and set controlled comparisons to 64
  for every policy; the failed configuration is retained as a diagnostic.
- Full Qwen3.5-0.8B HF integration passes tenant/checkpoint/base-alias checks,
  zero-LoRA equivalence (exact), and frozen/base-byte invariants. Compiled trained
  scores show a 0.0537 maximum sleep/wake log-prob difference versus 0.0000286
  on the immediate repeat. Cache-reset and ordinary-sampler controls are pending;
  byte preservation alone does not close this numerical validation question.

- Full MoE HF startup verifies exact base hashes, but the harness incorrectly
  required the optional zero-adapter optimization for expert snapshots. The
  production detector intentionally rejects parameter-target LoRA; corrected
  the harness to compare explicit base versus explicit initialized adapter
  without changing that conservative detector. Retrying full validation.
- Profiling both HF/TP=1 and FSDP/TP=2 Qwen3.8-27B startup found
  repeated Python pickle stacks in vLLM AOT graph serialization, rather than
  GPU computation. The bounded hybrid harness now sets the documented
  `VLLM_USE_AOT_COMPILE=0`, retaining torch.compile and CUDA graphs. This
  setting is recorded in result provenance and applied equally to each policy.
- Qwen3.5-0.8B cache-reset control passed full HF integration in 514.96s.
  Repeat error 0.00002658; cache reset error 0.00000000;
  sleep/wake error 0.00002956. The earlier 0.0537 difference
  did not recur. Added three transitions and a 0.001 cold-cache log-probability
  error bound for hybrid acceptance; rerunning this stronger check. It is a
  validation threshold, not a change to returned probabilities.
- Stronger Qwen3.5-0.8B HF validation passes in 443.14s: three cold-cache
  transitions have maximum log-prob errors 0.00003516,
  0.00013443, and 0.00003516,
  all below the predeclared 0.001 bound. Full tenant/checkpoint/publication
  checks pass with unchanged base storage. Earlier failed numerical evidence
  remains separately recorded; this is bounded BF16 validation, not bitwise
  equality for hybrid compiled inference.
- Full Qwen3.6-35B-A3B HF integration passes (1135.53s): 10 training
  steps, 8 publications, 10,240 generated tokens, tenant/checkpoint checks,
  64.56 GiB unique shared base storage and zero copied base parameters.
  Ordinary same-engine trained repeats vary by 0.36598 log-prob, versus
  0.04957 after sleep/wake and 0.31474 after cache reset. MoE routing makes
  the dense 0.001 bound inappropriate without an independent control. A
  separate full MoE run now measures three ordinary-sampler repeats and
  three switches; MoE acceptance compares switching against the measured
  ordinary/shared-control spread plus 0.001 and reports all raw scores.
- Full Qwen3.8-27B FSDP integration passes (1329.55s), preserving
  50.10 GiB unique shared base shards with zero copied base parameters and
  unchanged frozen hashes through 10 training steps / 8 publications.
- First Qwen3.8-27B HF matched triple (two B200s, 49,152 output tokens and
  12 steps each): static / duplicate / shared elapsed 65.436 / 48.658 /
  44.185 seconds; observed peak 127.892 / 178.411 / 123.257 GiB. Saved
  `qwen38-27b-hf-performance.png` and summary. One paired comparison is
  preliminary timing evidence; the physical storage reduction is separately
  verified. Full HF integration and stricter numerical controls continue.
- Large dense models also show ordinary repeat variation (HF 0.0882,
  FSDP 0.1686 max log-prob), so a small-model tolerance cannot be generalized.
  Added an independent strict control using the published vLLM LoRA tuning
  folder with shrink split_k=1. It avoids atomic split-K reduction without
  enabling unsupported batch-invariant GDN; strict dense/MoE checkpoint
  runs require repeat and three switch errors below 0.001. Performance runs
  retain normal kernels and are recorded separately.
- Superseded the extra normal-kernel MoE control (ap-WUcPqNnbro16RZd4K6SkLu)
  with strict single-split dense/MoE controls, which also include independent
  ordinary-sampler repeats. Stopped that redundant job before accepting any
  result; full normal-kernel MoE integration evidence remains saved.
- Full Qwen3.8-27B HF integration also passes (524.94s). Its completed
  raw artifact now contains all three performance policies and the separate
  10-step integration result; regenerated the figure and summary from that
  completed artifact. Both backends now have full dense-checkpoint evidence.
- Strict full-checkpoint HF controls passed on two B200s: Qwen3.8-27B in
  897.50s and Qwen3.6-35B-A3B in 1215.42s. Both have exactly zero score
  difference on ordinary/shared repeats, cache reset and all three cold-cache
  sleep/wake transitions; GPU adapter bytes remain unchanged. Each also passes
  the full 10-step/8-publication tenant/checkpoint integration. Saved raw results,
  `qwen-numerical-summary.json`, `qwen-numerical-controls.png`, and the actual
  B200 shrink tuning file. These controls do not measure the tuning's throughput
  cost or promise batch-shape invariance. Added explicit revision pinning to the
  full-checkpoint Modal wrapper for reproducing the recorded model revisions.
- Full Qwen3.6-35B-A3B FSDP performance triple finished: static / duplicate /
  shared elapsed 65.563 / 75.937 / 77.067s; observed peak 245.488 / 335.079 /
  223.174 GiB. All policies complete 12 steps, 49,152 output tokens and
  8 publications. Sharing saves 111.90 GiB versus duplicate but costs 1.5%
  GPU time, and is 17.5% slower than static. Shared rollout latency falls
  40.7% versus static; training request latency increases. Five phase pairs
  cost 2.48s; 40 serialized operations occupy 66.04s of the 77.07s run.
  Report this tradeoff explicitly, without claiming a universal throughput
  win or a fundamental limit. Separate full MoE FSDP integration continues.
- Full Qwen3.6-35B-A3B FSDP integration passes in 573.31s: 10 steps,
  8 publications, 10,240 generated tokens, tenant isolation, checkpoint
  restoration, unchanged frozen hashes, 64.60 GiB shared base shards and
  zero copied base parameters. Normal trained repeat/switch/cache-reset
  score differences remain explicitly recorded; the strict HF controls
  are separate evidence, not a claim of deterministic FSDP inference.
- Saved the final MoE FSDP summary and figure, plus `qwen-validation.json`
  with all seven full-checkpoint integrations, source/artifact digests,
  revisions, numerical controls and limitations. Final product source matches
  every new full-checkpoint run. Local suite: 640 pass / 38 GPU skips;
  separate GPU probes: 3 FSDP and 2 hybrid vLLM tests pass. Pinned Ruff,
  formatting and pyright pass; read-only secret scan of 29 changed Python
  files finds nothing; whitespace checks pass. Closed stale idle validation
  apps; the final ephemeral MoE app completed normally. No deployment or
  persistent experiment cache was created by the hybrid validation runs.

### Shared-only LoopWeave consolidation

- User clarified that duplicate-base borrowing has no role in the practical
  system. `policy: loopweave` now requires physical frozen-base sharing;
  the storage requirement is derived from the policy, not a separate switch.
  Old matching `share_base_weights` values normalize away; an explicit false
  value with LoopWeave is rejected with static-mode migration guidance.
- Removed optional-sharing startup branches and the duplicate GPU-test variant.
  Controller tests now require alias configuration before sampler initialization.
  All six example configurations use an explicit KV budget and automatic sharing.
- Performance runners now execute only static and LoopWeave, with alternating
  order. Removed the `--shared-weights` toggle. New artifact names prevent
  overwriting the historical triples. Reporting excludes historical duplicate
  rows and requires physical-sharing evidence before normalizing old policy names.
- Moved the earlier report to `benchmarks/loopweave/HISTORICAL.md`; current
  README, summaries and figures compare only static versus LoopWeave. Existing
  raw measurements retain their hashes and are explicitly pre-consolidation
  evidence. No new large-model speedup is claimed by this configuration cleanup.
- Full CPU suite: 643 passed / 37 GPU skips (113.53s). Pinned lint, formatting,
  pyright and read-only secret scan of 31 Python files pass. All six example
  configs validate. Small HF and FSDP GPU regressions are running.
- The first HF launch failed before GPU execution because pytest modified an
  uploaded `.pyc` file. Excluded generated bytecode caches from Modal source
  mounts and retried. No product change or test suppression was needed.
- Current shared-only FSDP GPU integration passes in 463.39s on four H100s:
  10 steps, 8 publications, 10,240 output tokens, zero copied parameters,
  unchanged frozen weights, exact deterministic repeats and three switches.
  Its source digest matches the current product. The small HF two-policy
  benchmark also completes equal work (12 steps / 49,152 output tokens),
  with 9.654s static and 9.800s LoopWeave. This short smoke workload does
  not demonstrate a speedup; separate HF integration checks continue.
- Both current HF integrations pass: deterministic in 284.29s and normal in
  148.15s. Each completes 10 steps / 8 publications / 10,240 tokens with
  zero copied base parameters and unchanged frozen weights. Deterministic
  repeated/cold-switch scores are exact; normal variations remain recorded.
  Saved the complete two-policy smoke artifact, summary and figure, and
  finalized `shared-only-validation.json`. All current GPU runs match the
  product-source digest; previous full-Qwen measurements retain their original
  source hashes. The small smoke run is functional/performance verification,
  not a new throughput-improvement claim.
- The small HF smoke also records 22.61 versus 39.73 GiB observed peak memory:
  its additional awake sampler has a 16-GiB KV budget. Zero copied base
  parameters do not imply zero additional sampler memory. Preserved this cost
  in the figure/report; workload-sized KV budgets remain a practical tuning
  requirement. All three Modal apps from this consolidation attempt are stopped
  with zero tasks, including the initial upload failure.

## Paper and code functionality audit

- Read the attached paper and the pinned prototype at
  `73232a7b5d8b62bc56e47df98b04fe5794d536dd` against current product source
  `7ad8072722ecb27bf499a1f3a6793221e3bc95f99d7f179d4c168e61496010d6`.
  Saved the paper, extracted text, and relevant prototype source beneath the
  existing ignored `downloads/loopweave/` directory. None is tracked or staged.
- Saved the detailed coverage matrix, source references, and acceptance criteria
  in [loopweave-functionality-audit.md](loopweave-functionality-audit.md).
- Corrected the previous characterization: physical sharing and safe borrowing
  are implemented, but the paper's fast-loop burst-coalescing planner and
  slow-loop allocation controller are absent, not merely unverified or untuned.
  Other gaps include full-loop telemetry, delay/freshness budgets, planned
  deadlines, a borrowing benefit gate, and same-snapshot request grouping.
- Found prototype limitations too: mock-only service calibration, unwired
  consumed-staleness feedback, and incomplete resource conversion/trainer growth.
  Porting the prototype wholesale would not establish complete paper parity.
- Inspected existing raw measurements: timed smoke/dense/MoE comparisons have no
  window rejections; separate integrations exercise that heuristic. More rounds
  could exercise prediction, but cannot activate controllers that do not exist.
- Updated the development documentation to state these limitations explicitly.
  No product code changes, tests, simulations, or experiments in this audit.

## Practical scheduling follow-up on 2026-10-12

Implemented the first three scheduling priorities from the audit. No GPU jobs
or experiments were launched; existing GPU figures describe the earlier policy.

- Added bounded completed-update telemetry in `src/tuft/loop_scheduling.py`.
  Multiple backward calls accumulate into one observation at successful
  optimizer completion. Readiness is the first backward submission; execution
  duration excludes phase switching, queueing and maintenance. Load/reset,
  removal and failed updates invalidate the run's history. Natural-period
  estimates subtract the deliberate wave delay and reject irregular/stale data.
- Persisted sampling-session provenance separately from fairness/user and
  adapter identity. Owned checkpoint sessions link to the source training run;
  foreign public snapshots, base-only sessions and legacy records without
  provenance remain unlinked. Holding a wave never replaces its adapter.
- Added a pure, bounded-look-ahead planner that packs nearby bursts backwards
  from later arrivals and checks total shift against each run's budget. One
  deadline applies to the entire wave after an optimizer step. Already-started
  waves have no further shift budget; infeasible plans fall back without a hold.
  Unexpected early updates invalidate obsolete holds. Existing per-user FIFO,
  cancellation, overload and drain semantics remain intact.
- Added cost-aware borrowing using observed Fixed/Flex service rates, measured
  switching costs, a switch-back deadline reserve and a post-training dwell.
  Cold calibration permits one capped request only under ready backlog and full
  Fixed slots. The probe cap inherits `flex_max_tokens` unless explicitly set.
  Service measurements count overlapping GPU requests once and exclude wake
  time and failed intervals. Admission conservatively uses the natural arrival
  horizon instead of relying on a client obeying the planned shift.
- Self-review caught and corrected obsolete holds on early training, possible
  double-counting of wake time, calibration deadlock with an already-known
  horizon, and publication waking the sampler without passing the borrowing
  gate. Delayed work is excluded from the ready-backlog trigger. Regression
  tests cover these cases; no cost model or safety assertion was disabled.
- HF and FSDP backend wrappers supply the same backward/optimizer/reset events.
  FSDP placement, frozen-base shard layout and tensor storage code are unchanged
  by this follow-up. A new CPU integration exercises both ranks' existing
  dispatch/aggregation path; this is not a new real-GPU correctness claim.

Validation of final product source
`63e6c4d27fe69685a487eec17c934fa6a5e1ca27a82bf8a249318b87802c335d`:

- Full CPU suite: **661 passed, 37 skipped**, zero failures/errors, 55.34 seconds
  reported by pytest. Includes 18 additional tests beyond the previous 643-test
  baseline, covering planning, telemetry, controller provenance, same-user run
  isolation, public snapshots, two-rank FSDP dispatch and borrowing decisions.
- Pinned Ruff 0.14.14 lint/format and Pyright 1.1.408 pass. Read-only secret scan
  of the ten touched Python files reports zero findings. No baseline or
  suppression edits. The initial restricted test attempt was interrupted; the
  tests completed with the local socket access needed by the CPU fixtures.
- Saved [validation metadata](../benchmarks/loopweave/scheduling-cpu-validation.json)
  with product/test hashes and [raw JUnit](../benchmarks/loopweave/scheduling-cpu-junit.xml).
  Documentation and report identify all older GPU figures as pre-policy-change
  evidence. No throughput or GPU-efficiency improvement is claimed for this
  unmeasured policy version.

The practical default remains one sharded Flex group plus persistent sampling.
Multi-group allocation, consumed-version freshness contracts, adapter-run queue
grouping and unified OpenAI accounting remain separate future work. This change
does not attempt to reproduce every paper claim or add probability calibration.

### PR #118 adaptation review and branch publication (2026-10-14)

- Requested branch name: `loopweave`; upstream repository is the existing
  `origin` remote, `agentscope-ai/TuFT`. Preserve the local Modal dependency
  edit and `.agents/` separately from the implementation checkpoint. The paper
  and extracted prototype remain ignored under `downloads/loopweave/`.
- Reviewed PR #118, `feat/sample_scheduling`, at
  `3bd7c0b3784ab666a3755a20cf353d54c2843702`:
  https://github.com/agentscope-ai/TuFT/pull/118 . Its scope is one-backend
  sampling request coalescing, with adapter sorting, optional serial dispatch,
  initial-adapter folding, and per-adapter metrics. It does not implement
  shared training/sampling storage, FSDP ownership transitions, loop telemetry,
  fast-loop planning, or cost-aware borrowing. It adds no test files.
- Reusable portion: short-window, arrival-ordered grouping by immutable adapter
  snapshot. Integrate this into the existing bounded ResourceScheduler rather
  than wrapping it in an independent queue. Keep user fairness, run telemetry,
  snapshot identity, work reservations, cancellation, and training drain
  semantics separate. Serial completion of adapter groups is not a suitable
  default because it can sacrifice concurrent generation throughput.
- Current code already folds verified zero-LoRA snapshots into base inference
  under `merge_zero_adapters`. Keep its conservative safetensors/config/factor
  checks. PR #118 trusts `optim_step_count == 0` even though loaded weights can
  already be trained; its weight fallback checks only B matrices and misses
  other modified state such as bias/modules_to_save. Do not import that path.
- CPU-only probes against the exact added scheduler source confirmed:
  (1) stopping while collecting a batch leaves the already-dequeued request
  unresolved; (2) a cancelled queued request still reaches the backend;
  (3) inherited `shutdown()` does not stop its dispatcher or underlying backend.
  Its default concurrent dispatcher also has an unbounded queue and no bound
  on dispatched tasks; max_batch_size caps only each submission round.
- The current product-source SHA-256 still matches the saved 661-pass CPU
  validation. No GPU experiments were launched for this review. PR #118 was
  inspected but not merged or adapted in this checkpoint; the next useful
  change would be bounded snapshot grouping with fairness/lifecycle tests.

### Many-adapter batching and multiple base models (2026-10-17)

- Numerical training/inference agreement remains a separate workstream. This
  change preserves snapshot selection, returned inference probabilities, and
  storage/ownership invariants; it does not add probability calibration or a
  policy-lag contract.
- Added a bounded adapter collection window (2 ms by default, at most 32
  admissions per dispatch set). Tenant round-robin/FIFO admission and token
  reservations happen before grouping by replica and adapter. Groups execute
  concurrently, and new arrivals cannot extend the oldest request's window.
  Completions coalesce for one event-loop turn so a wave of freed slots can
  form a useful batch under sustained load. Added aggregate grouping/switch
  counters without adapter-label cardinality.
- Self-review found that configuration still rejected multiple colocated bases,
  and fractional HF actors could be packed onto the wrong training device when
  more than one base was configured. HF LoopWeave trainers now reserve full
  GPUs. Borrowing samplers are created after trainer placement is known, using
  the same strict node/device binding already used by FSDP. The FSDP shard layout,
  TP mapping, collectives, and optimizer semantics are unchanged. Legacy
  fractional colocation is still rejected in a multi-model configuration.
- Exposed `max_cpu_loras` independently of active GPU `max_loras`, validated at
  configuration load. Added a two-base/four-GPU deployment example; it is a
  sizing example, not a new measured GPU configuration.
- Added CPU tests for fair grouping, fixed batching deadlines, cancellation and
  shutdown during collection, training priority, configuration validation, full
  trainer reservations, strict sampler node/device binding, and two-model
  controller/checkpoint/sampling integration with 16 adapters and 64 requests.
- The deterministic eight-request dispatch test reduces adapter transitions
  from seven (FIFO order) to one while returning every original request's
  result. This is queue behavior, not evidence of GPU speedup. The two-model
  integration test holds training on the first base while the second continues
  borrowing and checks that all published adapter identities remain isolated.
- Test iterations corrected a fake-clock fixture that did not advance the new
  collection timer and a Ray fixture that used an invalid node ID. CPU tests
  involving threaded persistence run outside the restricted socket sandbox.
- No GPU experiments were launched. This iteration still needs GPU validation,
  especially for the revised HF placement. Cross-model elastic reallocation,
  HTTP streaming admission, cross-session snapshot deduplication, and training
  adapter/optimizer offload remain outside this change.
- Final CPU regression: 677 passed, 37 skipped, no failures/errors (54.66 s
  reported by pytest). Pinned Ruff 0.14.14 lint/format and Pyright 1.1.408 pass;
  read-only secret scanning of changed code/config/docs finds no secrets.
  Reports: `benchmarks/loopweave/many-adapter-cpu-validation.json` and
  `benchmarks/loopweave/many-adapter-cpu-junit.xml`. The multi-model test uses an
  ingress barrier after sequence persistence so batching assertions are
  independent of filesystem/thread timing.

### Bounded GPU validation (2026-10-17)

- User authorized small GPU checks while avoiding repeated experiments. Added a
  launcher with no retries, unique result filenames, a 19-minute subprocess
  limit and 20-minute function limit. It preserves partial results and reporting
  rejects failed or timed-out jobs. Extended the benchmark CLI to support enough
  rounds for loop telemetry and a host cache for many published snapshots.
- Stopped the first HF/FSDP starts before measured work: inspection showed the
  old harness admitted the entire request wave to Fixed, preventing the new
  cold borrowing probe. Corrected the performance admission limit to match the
  engine's 64 active sequences; small correctness fixtures scale it to half
  their wave. New CPU preflights demonstrate ready backlog and exactly one
  bounded probe. Targeted CPU suite: 20 passed.
- Ran one corrected HF pair and one distinct two-rank FSDP integration; no
  successful run was repeated. Both real shared-weight paths passed alias,
  unchanged-base, adapter update/isolation, publication, checkpoint restore,
  switching and final-drain checks. The FSDP test passed in 393.91 s; ten updates,
  eight publications, and 10,240 output tokens. Per-rank frozen storage remains
  596,115,456 bytes, with no copied base parameters.
- HF used the same two H200s (Modal upgraded requested H100s at the same price),
  Qwen3-0.6B, eight tenants, eight rounds, 64 updates and 131,072 output tokens.
  Static: 26.076 s. LoopWeave: 29.987 s, **15.0% more allocated GPU time**.
  Sampling p95 improved 29.5%, but mean training latency rose 18.0%; aggregate
  peak memory increased from 11.140 to 16.276 GiB. This is a measured regression,
  not achievement of the resource-efficiency objective.
- Flex completed 293 of 1,024 measured HF requests; 64 loop observations were
  recorded. The planner applied no holds and rejected 15 infeasible plans.
  Sleep/wake consumed 0.902 s, insufficient by itself to explain the 3.911 s
  regression. Existing telemetry does not separate training queueing from Flex
  drain time. Next policy work should add that distinction and account for
  effects on whole-loop progress before another paid comparison.
- Estimated requested-resource compute: **$3.347**, including $0.559 for the
  canceled starts. Used container-start to app-stop windows, not only measured
  generation time. All four ephemeral apps stopped. CPU/RAM actual usage and
  credits can change the invoice. No additional large-model or numerical-control
  sweeps were run; multi-base GPU placement remains a separate coverage gap.
- Saved raw results, PNG/SVG figures, summaries, JUnit reports and a cost ledger.
  See `benchmarks/loopweave/budget-validation-20261017.md`. The HF raw artifact
  remains local because the existing secret scanner flags its public provenance
  hashes; the scanner baseline is unchanged. The FSDP raw artifact and summaries
  pass scanning and are included with the branch.
