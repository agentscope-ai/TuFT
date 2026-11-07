# Why borrowing stopped after the probes

Analysis of saved run `scheduler-fixes-20261103T230131Z`; no new GPU execution.

The immediate blocker was the five-second predicted occupancy cap. Sampling
demand remained available during long periods when the shared GPU was idle.

## What happened

1. At 0.14 seconds, four requests entered the borrowed-GPU sampler as a cold probe.
   Service times were 11.04–11.20 seconds. No service estimate was available at
   admission, so the token/count-bounded probe was allowed despite the time cap.
2. At 11.35 seconds, a fifth request entered with a predicted service time of
   2.81 seconds. This was still classified as a probe: the dedicated sampler's
   first aggregate rate observation did not arrive until 13.30 seconds.
3. That singleton actually took 10.80 seconds, 3.85 times its prediction.
   The history correctly refused to match a batch-of-four observation to a
   singleton, but its fallback used aggregate tokens/second from the four-request
   batch as though it were singleton latency. This is an estimator defect.
4. Once the singleton completed at 22.15 seconds, comparable future requests had
   an approximately 10.8-second service observation. The 1.5 safety factor alone
   makes that 16.2 seconds, before switching costs, versus a five-second budget.
   Borrowing stopped. There were 5,186 duration rejections and no profitable
   admission decisions. These are repeated scheduling checks, not unique requests.

The duration gate executes before the profitability comparison and applies even
when a training horizon is known. Therefore this run did not establish that the
dedicated sampler was always faster, nor that there was no useful idle window.
It established that the configured latency budget excluded this workload.

## Was sampling demand exhausted?

No, during the relevant idle windows:

| Saved snapshots | Shared active | Dedicated active | Queued | Training active |
| --- | ---: | ---: | ---: | --- |
| 30–150 s | 0 | 32 | 6–16 | No |
| 330–405 s | 0 | 32 | 10 | No |

There are 15 such snapshots. They are observations at 15-second intervals, not
an exact measurement of continuous idle duration. The progress logs omit
`training_pending`, and rejection events lack per-request estimates, so they
cannot support an exact counterfactual admission count.

Demand did become thin while tenants trained, and naturally disappeared at the
end. Each tenant starts three trajectories, waits for all four turns of each,
then trains and publishes before starting its next batch. With 16 tenants this
produces at most 48 simultaneous generation requests, not an unlimited stream.
That is a reasonable synchronous agent-RL pattern. It exercises contention but
does not represent the full diversity of independently arriving production jobs.
Task failure does not stop trajectories or reduce the fixed request count here.

## What the workload does not cover

- Only two updates per tenant: the loop predictor requires four completed
  readiness observations. All 16 tenants remained in `warming`, and every
  admission had an unknown horizon. Running longer would cover prediction, but
  would not remove the unconditional five-second cap.
- Similar tasks and synchronized starts create rollout/training waves. More
  independent tenants and staggered starts would be a useful separate workload,
  after correcting the admission policy; merely enlarging the queue is not useful.
- Early requests are not representative of later turns. Median output lengths
  were 198, 526, 674 and 923 tokens, yet all requested a 1,536-token maximum. The
  five shared observations cover only short first-turn outputs. The estimator
  records requested allowance, not actual output, and nearby prompt-length
  buckets can mix materially different generations.

## Recommended next changes

1. **Fix the throughput-to-latency fallback.** Preserve observed batch size and
   use a conservative request/batch service bound for an unmeasured concurrency.
   Add CPU regressions for the observed four-to-one error, batch growth, and
   early-to-late turn length changes. Keep the actual completion/drain barrier.
2. **Make the latency tradeoff explicit per model/deployment.** A five-second
   occupancy budget cannot accommodate the observed eleven-second generations.
   Report this incompatibility and distinguish a configured training-delay budget
   from a learned available idle window. A larger budget might admit early turns,
   but must still reject work exceeding the allowed drain risk. Do not raise it
   blindly or remove it based only on backlog: unexpected training must wait for
   an admitted non-preemptible request to finish. No readiness API is required
   for this correction; a strict five-second requirement may legitimately mean
   no borrowing for these requests.
3. **Save bounded rejection diagnostics.** Record reason, predicted service,
   concurrency, estimate source/confidence, horizon, training pending/active,
   queue depth, and fixed-versus-shared completion estimates. Record actual output
   lengths and link admission/completion with an opaque request ID. Current
   aggregate counters show the gate responsible but cannot audit every rejection.

The existing optimizer priority fix remains useful. There is no need to add
readiness signals or partial rollout support to address these estimator and
observability issues. More conservative estimation alone will not restore
borrowing under an incompatible five-second budget.

The evidence extraction script and JSON are saved beside the raw run as
`diagnose_borrowing.py` and `borrowing-diagnosis.json`. This diagnosis does not
predict the speedup or training delay of a relaxed policy; that requires a later,
separately authorized validation.
