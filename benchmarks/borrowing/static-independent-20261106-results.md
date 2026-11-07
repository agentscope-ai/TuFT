# Static allocation with independently timed clients

More independent clients substantially increase trainer utilization in this workload. With eight tenants, the trainer GPU averages 76.3% utilization and has no ready training work for 19.1% of the observation window. Static allocation therefore fills most, but not all, of the trainer's time without borrowing. Sampling queueing increases latency as load grows.

## Results

All four arms use one GPU for training and a second GPU for the dedicated sampler, with borrowing disabled. The table covers the same ten-minute window, from 240 to 840 seconds after workload start. GPU utilization is the time-weighted NVML reading, not SM occupancy or percent of peak FLOPS. Latencies include queueing and are grouped by submission time; calls submitted in the window are followed through completion even if they finish afterward.

| Tenants | Trainer GPU | Sampler GPU | No ready training | Sampling p95 (s) | Training p95 (s; samples) | Updates completed in window |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 14.9% | 17.5% | 83.8% | 55.1 | 32.3 (3) | 3 |
| 2 | 31.1% | 24.8% | 66.8% | 57.7 | 32.5 (7) | 6 |
| 4 | 45.3% | 24.0% | 51.6% | 58.1 | 114.4 (9) | 11 |
| 8 | 76.3% | 39.6% | 19.1% | 71.7 | 64.1 (15) | 16 |

The latency sample counts are small for training, especially at one tenant. These p95 values are descriptive observations, not stable tail-latency estimates. Completion counts and submission-cohort counts differ because calls can cross either window boundary.

## What the timelines show

- **One tenant:** rollout and training alternate. The trainer is idle while that client's trajectories finish.
- **Two tenants:** independent phases persist across rounds. One client samples while the other trains; publication calls stay below 0.7 seconds. Training latency remains close to single-client service time.
- **Four tenants:** some clients bunch together after training and publication waits. Publication calls reach 62.9 seconds, and later rollout/training waves become visible. This run has a higher training tail than eight tenants; latency is not monotonic with tenant count because phase alignment and queueing differ.
- **Eight tenants:** the trainer receives a much steadier stream of work and the sampler stays active. The observation-window sampling queue averages about ten requests, versus effectively zero at the lower counts. Every tenant completes two updates within the window. Publication waits still reach 62.3 seconds, but the whole workload does not fall into the same large synchronized waves.

This supports the idea that enough independent tenants can occupy most trainer capacity under static allocation. It does not establish that idle trainer time disappears at larger scale, or that average utilization alone describes a good deployment. At eight tenants, sampling p95 rises to 71.7 seconds and training p95 is 64.1 seconds, compared with 55.1 and 32.3 seconds for one tenant. The sampler's 39.6% utilization also leaves a separate inference-efficiency question; this test does not isolate its cause.

## Workload and measurement

Qwen3.5-4B runs on two GPUs using the HF backend, eager inference, Triton GDN prefill, and the same pinned runtime and verified cached checkpoint as the prior sweep. The dedicated sampler admits 32 requests. Each tenant has eight concurrent four-turn coding trajectories, then submits a training update and publishes its LoRA adapter before continuing. Task feedback is produced by isolated CPU containers. Additional tool delays are modeled client/environment latency, not GPU work.

Client seed 20261106 produces independent initial delays uniformly within 0–240 seconds. Tenant IDs 0–7 start at approximately 27, 114, 44, 236, 153, 219, 134, and 79 seconds. Client think times are independently sampled within 0–20 seconds between rounds, and additional tool delays within 0–3 seconds between turns. Random draws are keyed by tenant and operation, so the schedule does not depend on asynchronous execution order or total tenant count. The tenant sets are nested: tenants 0–1 have the same schedule in every arm where they appear.

There is no cross-tenant barrier or forced rephasing. Arrivals are initially independent by construction; after arrival, shared resource queues can correlate client behavior. All tenants use the same task family and rollout batch size, so this is not a broad heterogeneous workload distribution. The seed is one finite realization, not proof of statistical independence in observed completion times.

The arrival ramp is excluded from the main averages. Clients stop starting new rounds at 840 seconds, then finish admitted work. Plots show the complete timeline and shade the observation window; final draining is excluded from utilization averages. Sampling points use request submission time; training points use the start of the forward/backward-plus-optimizer call sequence. Publication latency is plotted separately. Ten-second utilization bins are time-weighted; thirty-second sampling percentile bins require at least ten calls, and empty/undersized bins do not get interpolated percentile lines.

This is a fixed post-arrival observation window, not a proven stationary steady state. There is one run per count and one base model. The current reference causal-convolution fallback and eager inference affect absolute performance. Task success remains low (see validation JSON); completed scheduling work is not evidence of improved learning quality.

## Validation and scope

All four GPU arms completed once, without retries: **1,728 sampling requests and 54 training updates**. Frozen weights are unchanged, the borrowed-GPU sampler copies zero base parameters, and zero measured requests use it. Initial adapter fingerprints match across the nested tenant sets. Actual initial requests match the programmed arrival times within 0.05 seconds. No request hits the output limit.

Twenty CPU checks passed for timing and existing workload contracts. Offline checks validated observation-window boundaries, weighted utilization, latency cohorts finishing after the window, and all four GPU result files. Changes are limited to benchmark timing, publication timestamps, the static launcher, tests, and offline reporting. FSDP sharding and production scheduling are unchanged; this HF run does not validate FSDP performance. No readiness API or partial-rollout support was added.

Both assigned GPUs are released and the experiment's CPU tool service is stopped. Existing users' jobs were left running. No Modal experiments or new model downloads were performed; future downloads use ModelScope.

## Artifacts

- [Utilization and latency over time](static-independent-analysis/static-timelines.png) ([SVG](static-independent-analysis/static-timelines.svg))
- [Per-client phases and publication waits](static-independent-analysis/static-client-phases.png) ([SVG](static-independent-analysis/static-client-phases.svg))
- [Summary data](static-independent-analysis/static-latency-summary.json), [validation](static-independent-analysis/validation.json)
- Earlier, tightly staggered static traces are retained separately in `static-analysis/`; they are not used for the table above.
Raw telemetry and environment provenance are retained separately from this repository.
