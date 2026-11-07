# Why borrowing slowed the closed-loop workload

The saved trace supports this chain: borrowed generations delayed training;
training and publication then accumulated behind global queue barriers; tenants
could not start their next rollouts, leaving the dedicated sampler idle. This is
an implementation and scheduling problem exposed by the closed-loop workload.

## A concrete stall

All times are seconds since the borrowing workload started. Operation boundaries
are reconstructed from each completed trace event's queue, drain, phase and service
durations; client publication intervals are recorded directly.

| Time | Event |
| --- | --- |
| 517.5 | A backward call is ready, but borrowed sampling is still active. |
| 567.8 | Borrowing finishes draining: the trainer has waited 50.3 s. |
| 645.1 | Two tenants have completed their optimizer calls and request publication. |
| 735.5–736.2 | Their checkpoint saves finally run, after about 90 s waiting each. Each save takes about 0.4 s. |
| 750.0 | Four tenants are waiting in training and four are publishing; none are sampling. |
| 826.3–826.6 | The first two borrowed-replica registrations run after another roughly 90 s queue wait. Each takes about 0.11 s. |
| 831.1 | Dedicated sampling resumes after 153.8 s without an active request. |

Two tenants therefore spent approximately 181.5 seconds publishing after finishing
training. The sampler GPU was available, but the clients could not reach their
next rollout.

## What causes it

**Long non-preemptible borrowing.** The current policy sees an idle trainer and
sampling overflow, but has no knowledge of when training will become ready. The
four-request/8,192-token admission bounds allowed borrowed service p95 of 55.1 s.
Three training drain events lasted 15.9, 50.3 and 33.2 s. Closing admission when
training arrives works, but cannot shorten already-running generations.

**HF checkpoints accidentally act as global barriers.**
`HFTrainingBackend.save_state` passes `lora_id` to the model but omits
`scheduler_tenant`. `MutationQueue` therefore sees `tenant=None`: it cannot promote
that checkpoint and stops looking for priority optimizers behind it. The saved
adapter is tenant-specific (`selected_adapters=[lora_id]`). FSDP's scheduling wrapper
already supplies this identity for checkpoints; the observed missing identity is
in the HF path.

A CPU-only reproduction using the actual `MutationQueue` confirms the distinction:

- Global checkpoint: backward → checkpoint → optimizer.
- Tenant-identified checkpoint: checkpoint → optimizer → backward.

**Publication couples dedicated sampling to the borrowed GPU.**
`DPSamplingBackend._add_adapter` awaits registration on every replica. Borrowed
registration enters the same training gate with `kind="publication"` and no tenant
identity. A client waits for that replica even when its dedicated replica could
otherwise serve the next rollout. The global barrier also limits optimizer
priority for unrelated tenants.

These queue barriers exist in the static arm too: publication p95 was already
61.9 s. Borrowing added drain waits and bunched clients into training/publication,
amplifying the same coupling to 181.5 s p95. Initially independent arrival times
do not prevent synchronization later through a shared queue.

## What the evidence rules out or limits

- Request size barely changed: mean prompt tokens 2,153 → 2,159 and output tokens
  564 → 567. Generated text and rewards were not identical, so this is not an
  identical-trajectory replay.
- Median backward service stayed 30.47 → 30.41 s. Queueing, not a general slowdown
  of backward execution, dominates the regression.
- Recorded phase-switch time totals only 2.11 s static and 2.02 s borrowing across
  the timed runs. This is distinct from waiting for requests to drain.
- Frozen weights stayed unchanged, base parameter copies were zero, and there were
  no sampling failures or output-limit truncations.
- The sampler did run out of *ready* requests during the stall. Clients were blocked
  upstream, not finished with their allowed workload. This is a real consequence
  of train → publish → rollout dependencies.

Sampling p95 improved, but only 353 requests completed in the window versus 530
static. The lower latency does not establish an overall benefit. Completed output
throughput fell 29%, training p95 doubled, and average GPU utilization fell from
60.1% to 53.2%. Window update counts were equal, but cumulative updates by 840 s
were 17 static versus 16 borrowing.

## Fix order

1. Pass the HF checkpoint's tenant identity through to the scheduler, preserving
   per-tenant order and testing that unrelated work cannot suppress its priority.
2. Separate dedicated publication readiness from borrowed-replica readiness. Keep
   exclusion on the shared GPU, admit a borrowed request only after its snapshot is
   registered there, and preserve rollback/cancellation correctness. This common
   sampler path also needs FSDP coverage without changing frozen-base sharding.
3. Then address long borrowed generations with a tighter request bound or explicit
   client training-readiness information. Partial rollouts are not required to fix
   the two queueing problems above. This trace cannot identify an optimal token cap.

No production code was changed and no new GPU experiment was run for this diagnosis.
The trace establishes the waits and the CPU reproduction establishes the priority
barrier; the performance gain from each proposed fix still needs a controlled check.

[Reconstructed evidence](reactive-comparison-20261106-analysis/diagnosis.json) was
produced by [the offline diagnostic script](diagnose_reactive_comparison.py).

```bash
PYTHONPATH=src python benchmarks/borrowing/diagnose_reactive_comparison.py \
  benchmarks/borrowing/runs/reactive-20261106 \
  benchmarks/borrowing/reactive-comparison-20261106-analysis/diagnosis.json
```
