# LoopWeave paper and code functionality audit

**Historical audit of the 2026-10-11 source snapshot.** The 2026-10-12 follow-up
implements completed-update telemetry, a bounded fast-loop planner and
cost-aware borrowing. See [the work log](loopweave-work-log.md) and the current
[development guide](sphinx_doc/source/development/loopweave.md). The findings
below remain the record of what was missing before that change, not a claim
about the updated checkout. The follow-up has CPU validation only.

The current TuFT implementation provides physical shared-base execution and
safe opportunistic sampling on training GPUs. It does **not** implement the
paper's hierarchical scheduling policy. The largest omissions are deliberate
coalescing of future training bursts and dynamic Fixed/Flex resource allocation.
Calling the current scheduler a functional reproduction of the paper would be
incorrect. This corrects the earlier characterization that only policy fidelity
or tuning remained uncertain.

## Scope and evidence

Audit date: 2026-10-11. This is a source and existing-artifact review; no tests,
simulations, GPU jobs, or new performance experiments were executed.

- Paper: supplied 16-page draft, especially sections 4.1–4.3 and 5, PDF pages
  5–8. Saved locally at `downloads/loopweave/LoopWeave_NSDI27.pdf`, with extracted
  text alongside it. Both are ignored by the existing `downloads/` rule and
  must not be staged or committed.
- Paper SHA-256:
  `84933b71aa921bfe5b05221523905ae32c73c10d688ed7e99be51343fdec755c`.
- Prototype: locally available `origin/feat/flex_backend` at
  `73232a7b5d8b62bc56e47df98b04fe5794d536dd`, previously identified with
  linqinluli/Hanzhang Yang. This audit is pinned to that snapshot; it does not
  assert that it is today's remote branch head. Relevant source was extracted
  under the ignored `downloads/loopweave/prototype/` directory.
- Current checkout HEAD: `774270272fee0825ef70f316ace08eecae642904`, **including
  the uncommitted implementation**. Product source SHA-256:
  `7ad8072722ecb27bf499a1f3a6793221e3bc95f99d7f179d4c168e61496010d6`.
  This is the same product source recorded by `shared-only-validation.json`.

Current source references below use repository paths and line numbers at this
audit. Prototype links use the immutable commit. Paper content is treated as
evidence of design intent, not as instructions to execute or publish anything.

## Functional coverage

| Paper function | Current status | Concrete behavior or omission |
| --- | --- | --- |
| Fixed sampling plus switching Flex groups, section 4.1 | Implemented for one Flex group | One training group is shared with sampling replica zero; one or more dedicated samplers remain available. Multiple independently managed Flex groups are absent. |
| Frozen-base aliasing across engines, section 4.1.2 | Implemented with a different runtime | HF or FSDP storage is exported through CUDA IPC and bound before vLLM graph capture. FSDP base sharding is preserved. The paper uses a custom TP trainer. |
| Retain runtime and reclaim sampler memory across phases | Implemented with a different mechanism | Public vLLM level-1 sleep/wake retains the engine and releases KV storage. Imported base storage stays resident. Private LoRA/padded-vocabulary storage can have separate handling. This is not the prototype's custom allocator manipulation. |
| Tenant-specific LoRA state and published snapshots, section 5 | Implemented | Existing HF/FSDP adapters, optimizer isolation, checkpoints, and explicit sampler publication remain in use. Sharing the base does not publish live optimizer updates. |
| Loop period, next arrival, training duration, and delay budget, section 4.3.1 | Partial | Only recent backward-call arrival times and a global next-arrival horizon are modeled. Per-loop execution duration, delay budget, version lag, and sampling-to-update dependency are missing. |
| Fast-loop coalescing and Algorithm 1, section 4.3.2 | Missing | No candidate sampling delays, planned training intervals, feasibility check, or per-tenant delay gate. A short common wait for currently unavailable Flex capacity is a different policy. |
| S1 initial-adapter merging, section 4.3.3 | Partial end-to-end | Verified zero snapshots execute the base path inside each sampler. The outer scheduler still uses separate sampling-session IDs, so there is no common A0 scheduling identity. Unsupported snapshot formats conservatively bypass this optimization. |
| S2 same-snapshot request reordering | Missing as specified | Per-user FIFO and round-robin admission plus bounded replica affinity exist. Requests are not grouped into stable same-snapshot runs. |
| S3 window-aware routing | Partial | A token cap and learned aggregate service-time heuristic can reject work near a predicted arrival. There is no planned switching deadline, per-group schedule, or explicit long/multi-turn classification. |
| Slow-loop Fixed/Flex allocation, section 4.3.4 | Missing | Group roles and trainer size are fixed at startup. No utilization controller, hysteresis, dwell timer, conversion transaction, or training DP expansion/contraction. |
| Independent quantized Fixed path | Missing in LoopWeave mode | The common model configuration rejects quantization for the whole LoopWeave pool, including Fixed replicas. Quantization elsewhere in TuFT is a separate capability. |
| Learned sampling-side Calibrator, section 4.2 | Intentionally absent | Returned scores are not corrected. Numerical controls and deterministic kernel settings are not a learned trainer-reference calibration model. |
| Freshness-aware scheduling | Missing server feedback | Snapshot identity is preserved, but consumed rollout versions and per-tenant delay/freshness budgets do not enter scheduling. Client algorithms still own their policy-lag controls. |
| Unified sampling admission across APIs | Partial | Tinker requests use the scheduler. OpenAI/streaming requests stay on Fixed replicas and bypass its queue/work counters. |

Execution evidence is in the existing
[validation manifest](../benchmarks/loopweave/shared-only-validation.json) and
[experiment report](../benchmarks/loopweave/README.md). Those checks support the
implemented execution paths; they do not establish functionality for missing
controllers.

## What the current scheduling policy actually does

The complete decision path is small enough to describe directly:

1. Compute request work as `num_samples * (prompt_tokens + max_tokens)`.
   Only requests below `flex_max_tokens` are eligible for Flex.
2. Queue by user and visit queue heads round-robin, preserving user FIFO.
3. If Flex is training, has pending exclusive work, or is in the post-operation
   quiet interval, hold an eligible request for at most `sampling_alignment_s`.
   During that hold it cannot use an otherwise available Fixed replica either.
4. Among eligible replicas with free request slots, choose the least reserved
   token work, with rotating ties and a bounded preference for the last adapter.
5. If an arrival horizon and service-rate estimate both exist, exclude Flex when
   `now + 1.5 * seconds_per_token * (reserved_work + request_work) >= horizon`.
6. Any training or adapter-maintenance operation immediately closes Flex
   admission, drains admitted work, takes a single exclusive lock, enters the
   required phase, and executes. A common quiet interval follows completion.

References: `src/tuft/scheduling.py:43` (`LoopProfiler`), `:213` (`_choose`),
`:280` (`_pump`), `:370` (`training`), `:394` (`_train`);
`src/tuft/backends/sampling_backend.py:544` (`sample`);
`src/tuft/config.py:63` (`SchedulingConfig`).

This is reactive GPU borrowing with a predictive admission heuristic. It cannot
choose to postpone tenant A's sampling now so that A's **future** update arrives
next to tenant B's update. For example, given predicted arrivals at 10 and 12
seconds, one-second training bursts, and a two-second budget for A, the paper's
planner can delay A's predecessor sampling and place the two bursts together.
The current scheduler has no representation for that choice, even with perfect
arrival histories.

## Scheduling gaps and their consequences

### 1. Complete loop observations and identity are missing

The paper estimates `(period, next arrival, training duration, delay budget)` for
each tenant loop and refreshes the plan when workload or resource allocation
changes. Our profiler stores up to nine arrival timestamps, needs four arrivals,
and uses a median interval only when every interval lies within 20 percent of
that median. It discards elapsed predictions and returns the earliest remaining
prediction across all observed runs. It has no per-loop training duration or
explicit prediction uncertainty.

The observation hook fires on **each backward call**, not each completed RL
iteration: `training_backend.py:91` and `fsdp_training_backend.py:70` feed
`scheduling.py:380`. Several gradient-accumulation calls can therefore create
short intra-update intervals mixed with long rollout intervals. This can disable
the predictor or give it the wrong event to predict. Optimizer completion is not
used to close a loop observation.

Sampling fairness is keyed by authenticated user
(`sampling_controller.py:399`); profiling is keyed by training LoRA/run ID;
adapter affinity is keyed by sampling-session ID. These keys serve different
purposes. The current global horizon does not require them to match, but a
per-loop delay gate cannot work without a stable mapping from a sampling
snapshot/session to its training run and version. Multiple runs belonging to
one user must remain distinguishable.

**Required:** explicit loop/run identity separate from fairness identity;
sampling and training-ready timestamps; per-operation lane time accumulated
through optimizer completion; publication/consumption version metadata; drift,
idle-tenant removal, and configuration-change handling. Maintenance and scheduler
wait time must not become training demand or natural loop-period observations.

### 2. The fast-loop planner and delay gate are absent

The paper chooses nonnegative sampling delays, orders predicted arrivals, and
checks `start_i = max(arrival_i + delay_i, previous_finish)` with total delay no
larger than the tenant's budget. It replans or changes allocation if infeasible.
The prototype has a real planner and one-shot gate:
[ContiguousTrainingScheduler](https://github.com/agentscope-ai/TuFT/blob/73232a7b5d8b62bc56e47df98b04fe5794d536dd/src/tuft/schedulers/rl_loop_scheduler.py#L65)
and
[runtime planning](https://github.com/agentscope-ai/TuFT/blob/73232a7b5d8b62bc56e47df98b04fe5794d536dd/src/tuft/schedulers/runtime_scheduler.py#L312).

None of the current `ResourceScheduler` fields represents a plan, a tenant's
delay budget, or a training interval. `sampling_alignment_s` is a common
per-request wait for existing training to finish. `training_quiet_s` suppresses
reopening after an operation. Neither implements Algorithm 1 or the paper's
causal shaping of future update arrivals.

**Required:** bounded look-ahead planning, explicit feasibility and fallback,
one-shot holds bound to the appropriate rollout wave/version, expiration of old
plans, and replanning on actual arrivals/completions. A hold must cover the
sampling work that causally precedes the update; delaying one arbitrary request
in a concurrent rollout wave is insufficient.

### 3. Delay and freshness budgets are not modeled

There is no per-tenant `W_i`, permitted policy-version lag, consumed-version
feedback, or freshness-related admission action in the current scheduler.
The global admission timeout bounds queue waiting, not extra scheduling delay
or end-to-end completion. Per-user queue rotation is useful fairness, but is not
a tenant latency or freshness guarantee.

The paper distinguishes delay budgets from permitted version lag. Section 6.1
also locates lag admission controls on the tenant side. We should not claim that
the paper requires an entirely server-owned hard staleness guarantee, nor infer
an algorithm's acceptable lag from timing alone.

**Required:** explicit latency-delay budget and optional client-provided
freshness contract, observed version feedback, and accounting that distinguishes
scheduler-induced holds from backend queuing. Preserve immutable snapshot
semantics; holding a request must not silently upgrade its adapter version.

### 4. Admission windows are not coupled to a training schedule

The current horizon is an unshifted arrival prediction, not a scheduled training
start or switching point. The single service-rate estimate is measured over
whole Flex busy intervals using estimated token work. It cannot distinguish
prefill from decode, concurrency levels, adapter rank, model path, or prefix
cache state. Unknown/irregular histories disable the horizon check entirely;
only the token/request caps remain. A short request in a multi-turn session is
not explicitly identified as multi-turn or unpredictable.

Sleep/wake durations are recorded by `vllm_engine.py:245` but never fed back to
the scheduler. There is no explicit reserve for switching back before training,
nor a prediction-error margin calibrated by workload. The 1.5 multiplier is a
fixed heuristic. An unexpected update safely drains admitted requests, but can
wait for their entire completion; there is no wall-time bound or preemption.

**Required:** planned per-group deadlines, request/batch completion estimates,
separate switch-back reserve, uncertainty-aware Fixed fallback, and explicit
overrun measurement. Preserve the current drain/cancellation safety.

### 5. Borrowing has no measured benefit gate

Once an eligible request selects idle Flex, the current code wakes it regardless
of whether the useful gap amortizes a round-trip switch or whether Fixed capacity
already suffices. Quiet time and request limits are static configuration, not
adaptive switch-cost hysteresis.

The prototype includes additional practical logic beyond the paper's compact
description:
[`_gap_worth_filling`](https://github.com/agentscope-ai/TuFT/blob/73232a7b5d8b62bc56e47df98b04fe5794d536dd/src/tuft/schedulers/runtime_scheduler.py#L572)
compares observed gaps with measured round-trip costs; its duty loop also checks
backlog and dwell time. These mechanisms are absent here. They still do not prove
optimal decisions, and we should not copy prototype timing constants blindly.

**Required:** measure actual phase costs and useful idle gaps, subtract
self-induced delay, and compare predicted useful sampling work against switching
cost and training delay. Decline borrowing when the benefit is uncertain or
negative. This can improve practicality before multi-group allocation exists.

### 6. Adapter affinity does not implement request grouping

The paper groups same-snapshot requests while retaining FCFS within a group.
Current admission rotates users, then uses the last dispatched adapter as a
bounded routing preference (`scheduling.py:255`). It does not reorder a pending
set into adapter runs or know the full set of adapters resident in each engine.
Separate sessions using the same snapshot also receive different affinity keys.

Verified-zero handling occurs inside `VLLMSamplingBackend.sample`, after routing
(`sampling_backend.py:329`). Thus engine execution can merge onto the base path,
but the scheduler does not see a shared A0 identity. Additionally,
`adapter_equivalence.py:16` conservatively rejects parameter-targeted MoE
snapshots and other unsupported layouts. That is safe fallback, not universal
S1 coverage.

**Required:** canonical immutable snapshot identity, verified A0 classification
before scheduling, bounded same-snapshot run formation with aging/fairness, and
measured locality costs. Preserve per-user fairness rather than replacing it
with unbounded adapter affinity.

### 7. The slow controller and its execution machinery are absent

`DPSamplingBackend.__init__` permanently assigns `flex_index=0` and constructs a
fixed replica list (`sampling_backend.py:465`). HF has one training actor;
FSDP constructs one group of a fixed size. `config.py:264` requires FSDP group
size to equal sampler TP. Multiple FSDP ranks form **one** Flex group; they are
not multiple interchangeable Flex groups.

There is no periodic utilization controller, low/high threshold, minimum dwell,
conversion cost check, or actual conversion API. There is no mechanism to enlarge
data-parallel training with another Flex group or shrink it while keeping adapter
and optimizer state consistent. This is a resource-lifecycle change, not just an
extra `if` in routing.

**Required:** explicit group registry and roles; multiple training-capable groups;
training participation/state synchronization; allocation-window utilization
excluding overlapping wait time; transactional drain/convert/publish/rollback;
minimum persistent sampling and training capacity; then hysteretic control.
For FSDP, retain frozen-base sharding inside each group. Changing group roles
must not silently replicate the full frozen base on every training rank.

Independent Fixed quantization would also need separate Fixed/Flex engine
configuration. Current `config.py:258` rejects any LoopWeave quantization.
Host offload, prestarted conversions, and format-changing conversions described
in section 4.3.4 are also absent.

### 8. Some traffic and costs are invisible to the policy

OpenAI/streaming traffic uses Fixed-only URLs (`sampling_backend.py:513`) and
bypasses Tinker reservations. Mixed traffic therefore makes the scheduler's
Fixed-load estimate incomplete. The policy cannot enforce its holds, adapter
grouping, or completion estimates over that traffic.

Adapter add/remove operations share the exclusive gate but may wake the Flex
sampler (`sampling_backend.py:584`). Publication, checkpoint, and optimizer
operations can therefore produce transitions that a training-burst plan should
account for. Present aggregate `exclusive_busy_s` mixes these operations with
phase-transition time. It is not a clean training-utilization signal for the
paper's slow controller.

**Required:** common request accounting where APIs permit it, documented Fixed
fallback otherwise, and separate training, publication, switching, queueing,
idle, and useful sampling metrics. A planner should reason about the complete
update/publication boundary without violating existing request ordering.

## The prototype is not a complete implementation oracle

The prototype supplies useful algorithms, but source inspection reveals gaps
between its service wiring and the paper too:

- **Discovery differs:** the paper specifies a median; the prototype profiler
  uses exponential smoothing. Its runtime derives a delay window from observed
  lag or a fraction of the estimated period. `observe_consumed_staleness` has no
  caller in the inspected `src/tuft` tree, so real lag feedback is not established
  by that service wiring.
- **Sampling cores differ from the live path:** `SamplingSideScheduler` builds
  and routes whole runs but is only defined/exported in the inspected service
  tree. The live `SamplingPipelineBackend` uses a bounded same-adapter hold and
  can hold requests at a horizon instead of immediately routing them to Fixed.
  Algorithm-level parity must be distinguished from server-path parity.
- **Slow conversion is incomplete:** the runtime does invoke router conversions,
  but `sampling_router.py:712` removes a Fixed backend from a list without
  draining/shutting it down or adding its GPU to a trainer. Growth creates a new
  backend and suppresses adapter replay errors. The controller does not inspect
  the conversion method's boolean success value. Bookkeeping changes therefore
  do not establish safe resource conversion or elastic training. Its pressure
  observation also uses cumulative lane busy time/elapsed time, rather than a
  fresh control-interval measurement.
- **The learned Calibrator is not wired into the service:**
  [`corrector.py`](https://github.com/agentscope-ai/TuFT/blob/73232a7b5d8b62bc56e47df98b04fe5794d536dd/src/tuft/corrector.py#L34)
  provides a mock with optional constant bias/latency; `build_corrector` selects
  it. Research notes reference learned-predictor assets outside the released
  tree. They do not supply an integrated, trained service Calibrator here.

These are reasons to port concepts with explicit contracts and tests, rather
than copy the prototype wholesale. The audit does not infer that the paper's
external research experiments did not exist.

## What existing measurements do and do not establish

Inspection of the saved results for the current Qwen3-0.6B smoke pair and the
earlier dense-27B HF and MoE-35B FSDP comparisons shows **no window rejections in
the measured scheduler counters**. The sparse counter is absent when zero.
Separate GPU integration results do contain window rejections; those exercise
the heuristic's admission path, not a paper-style planner.

The timed workload uses three rounds by default and one backward call per
tenant per round (`scripts/benchmark_loopweave.py:612`). The predictor needs
four observations; only the first tenant receives an extra preflight backward
call, whose timing is unrelated to steady-state rounds. This limits how much
the timed workload can exercise learned horizons. More rounds could exercise
that heuristic, but **cannot activate missing fast/slow controllers**.

Existing measurements therefore demonstrate the tradeoffs of shared-base
borrowing, bounded admission, and short reactive alignment. They do not measure
the full paper policy, and neither the gains nor the regressions establish its
performance limit. Missing scheduling mechanisms are plausible improvement
opportunities, not evidence that a future implementation will necessarily win.

The current unit tests cover cancellation, queue bounds, FIFO fairness, phase
draining, an injected horizon, and alignment expiry. They do not cover paper
schedule feasibility, loop-aware holds, consumed-version budgets, composition
changes, or conversion rollback because those functions are absent.

## Implementation order and acceptance criteria

This is a proposed next sequence, not additional work executed by this audit.

1. **Correct loop telemetry and identity.** Separate user/run/snapshot identity
   and record full update boundaries. Acceptance: multiple backward calls per
   optimizer step yield one loop observation; concurrent runs under one user
   remain separate; maintenance and queue waits do not inflate demand.
2. **Implement the fast planner and budgeted delay gate.** Acceptance: a
   deterministic two/three-tenant trace produces the expected contiguous bursts;
   total delay never exceeds the tenant budget; infeasible, expired, cancelled,
   irregular, and concurrent-wave cases fall back safely. Include explicit
   snapshot-version semantics.
3. **Connect planned windows and a borrowing benefit gate.** Acceptance: decline
   work that cannot finish plus switch back before the deadline; decline gaps
   that cannot amortize switching; record overruns; keep Fixed serving and retain
   drain safety. Use workload-aware timing rather than fixed paper constants.
4. **Implement bounded adapter grouping.** Acceptance: same-snapshot FIFO,
   canonical A0 identity, aging under sustained competing tenants, and no
   adapter/version leakage. Preserve safety fallbacks for unsupported snapshots.
5. **Build multiple-group lifecycle support, then the slow controller.**
   Acceptance: decisions change real usable GPU roles, not counters alone;
   preserve FSDP sharding and optimizer state; drain before conversion; failed
   startup or adapter publication rolls back; sustained pressure changes
   allocation while hysteresis prevents oscillation.
6. **Treat calibration as a separate API decision.** Full paper parity requires
   a trained, path-specific Calibrator and its data/validation lifecycle.
   Practical TuFT should preserve raw behavior scores and expose any
   trainer-reference estimate separately rather than silently substituting it
   into existing importance-sampling inputs. The paper itself says calibrated
   scores are not behavior-policy probabilities.

The next performance work should follow functional acceptance and measure which
decisions actually fired. No additional experiments are authorized by this audit.
