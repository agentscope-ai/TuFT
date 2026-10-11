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
