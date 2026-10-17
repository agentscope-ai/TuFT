# Bounded GPU validation — 2026-10-17

**The shared-weight paths passed their functional checks, but the current policy
regressed on the eight-tenant HF workload.** No successful experiment was repeated.
Estimated total requested-resource cost is **$3.35**, including canceled startup
attempts. All four ephemeral apps have stopped. This is not a billing invoice.

## HF performance pair

Qwen3-0.6B, eight tenants, eight rounds, 64 optimizer updates, 56 published updates,
1,024 sampling requests and 131,072 generated tokens per policy. Both policies ran
on the same two NVIDIA H200 GPUs: Modal automatically upgraded the requested
H100s at the same price. Fixed vLLM capacity is 64 sequences, GPU LoRA capacity
is four, host adapter cache capacity is 128, and each sampler's KV budget is 4 GiB.

| Timed workload metric | Static | LoopWeave | Change |
| --- | ---: | ---: | ---: |
| Completion time | 26.076 s | 29.987 s | +15.0% |
| Training updates/second | 2.454 | 2.134 | -13.0% |
| Allocated GPU seconds/update | 0.815 | 0.937 | +15.0% |
| Mean sampling latency | 1.376 s | 1.266 s | -8.0% |
| Sampling p95 latency | 3.127 s | 2.206 s | -29.5% |
| Mean training latency | 1.282 s | 1.512 s | +18.0% |
| Observed total peak GPU memory | 11.140 GiB | 16.276 GiB | +46.1% |
| Mean observed GPU SM utilization | 25.5% | 32.4% | +6.9 points |

This is **one pair**, static first, on one small model. Initialization and
correctness probes are excluded from these metrics but included in spending.
The result establishes a regression in this case, not a general performance
ranking or a confidence interval. Training inputs and work counts match; this
is not a claim of identical floating-point trajectories.

- Physical alias checks passed with **zero copied base parameters** and 1.110 GiB
  of unique shared frozen storage. Base fingerprints remained unchanged.
- Trained adapter publication, tenant isolation, optimizer checkpoint restoration,
  and repeated switching passed. Numerical HF/vLLM calibration remains deferred.
- Borrowing was exercised: **293/1,024** measured samples ran on Flex, with one
  calibration probe and seven profitable-candidate decisions.
- The scheduler recorded **64 completed loop observations**, 15 infeasible-plan
  decisions, and **zero applied loop holds**. Eight rounds remove the previous
  three-round warmup limitation; they do not guarantee a feasible alignment plan.
- Batching grouped 601 measured requests within admitted dispatch sets. These
  counters describe submission order, not GPU cache misses or isolated speedup.
- Fourteen sleep/wake cycles consumed 0.902 s inside the measured workload. This
  alone does not explain the 3.911 s regression. Mean training latency increased
  while sampling latency fell; the counters do not separate training queueing
  from Flex drain time, so attributing the entire regression to draining would
  be premature.

**Next policy work:** expose training drain time separately from training queue
wait, then evaluate borrowing against end-to-end RL progress rather than only
predicted sampling completion. Inspect why plans are infeasible before relaxing
latency budgets. Do not tune against repeated runs of this one trace. Keep the
policy opt-in until a representative workload demonstrates resource efficiency.

## FSDP functional validation

One four-H100 integration test passed in 393.91 s including initialization.
The measured loop completed ten training updates, eight publications, and 10,240
sampled tokens in 9.360 s. This is a functional check, not a performance comparison.
Both Fixed and Flex completed requests (27 measured requests on Flex).

Each of the two training ranks exported 596,115,456 bytes of unique frozen
storage; both corresponding sampler ranks verified aliases, with zero copied
base parameters and unchanged base fingerprints. The test retained the existing
FSDP/TP shard layout, exercised adapter updates and switching, and ended with
empty queues and no active requests. No product sharding code changed in this
validation stage.

## Cost and experiment discipline

| Job | Allocation window | GPUs requested | Estimated compute |
| --- | ---: | ---: | ---: |
| Canceled HF startup | 108 s | 2 H100 | $0.275 |
| Canceled FSDP startup | 60 s | 4 H100 | $0.284 |
| Completed HF pair | 354 s | 2 H100 | $0.901 |
| Completed FSDP integration | 398 s | 4 H100 | $1.886 |
| **Total** | | | **$3.347** |

Windows run from observed container start to app stop, including startup and
shutdown. Each requested 16 CPU cores and 64 GiB RAM. Rates are from
[Modal pricing](https://modal.com/pricing); automatic H100-to-H200 upgrades have
[the same GPU price](https://modal.com/docs/guide/gpu#automatic-upgrades-to-h200s).
Actual billed CPU/RAM usage, credits and other charges can differ.

The first starts were canceled before measured work: review found the old
benchmark's Fixed admission limit swallowed the whole request wave, preventing
cold borrowing probes. The corrected performance workload limits admission to
the engine's 64-sequence capacity. The small FSDP correctness fixture scales
that limit down to eight, deliberately creating backlog; its timing is not used
as evidence of efficiency. New CPU preflights verify a bounded probe for both
workloads before any paid restart. Targeted CPU validation: **20 passed**.

The new launcher uses a 19-minute subprocess timeout, 20-minute function timeout,
no automatic retries, unique output labels, and partial-result preservation.
Reporting rejects failed or timed-out jobs. The image was cached; FSDP kernel
compilation still dominated startup. No large-model sweep or batch-invariant
numerical-control run was launched.

## Artifacts and scope

- [Performance figure (PNG)](budget-hf-backlog-20261017.png) and
  [vector figure (SVG)](budget-hf-backlog-20261017.svg).
- [Performance summary](budget-hf-backlog-summary.json),
  [cost ledger and functional results](budget-validation-20261017.json),
  [FSDP JUnit](budget-fsdp-backlog-junit.xml), and [CPU JUnit](budget-cpu-junit.xml).
- Raw local files: `budget-hf-backlog-20261017.json` and
  `budget-fsdp-backlog-20261017.json` retain all observations and diagnostics.
- Product code: `010ef24`; benchmark source digests and dependency versions are
  recorded in the HF raw file. The new harness/configuration changes accompany
  this report. Existing user changes to `pyproject.toml` are preserved.

This stage validates one base per deployment with HF and sharded FSDP. It does
not newly validate multiple simultaneous base models, large dense/MoE models,
HTTP streaming admission, cross-model GPU reallocation, or training/inference
numerical agreement. Earlier evidence for other configurations remains separate.
