# Shared weights for training and sampling

Agent RL alternates between generating rollouts and updating adapters. With many
tenants using the same base model, training GPUs can sit idle during generation,
while loading a second full base on those GPUs consumes memory needed by KV
caches and training. Sharing frozen storage lets those GPUs help serve rollouts
without another full base copy.

This opt-in mode lets a trainer and a borrowing vLLM sampler use the same frozen
base weights. Each tenant keeps its own LoRA adapter and optimizer state. Sampling
uses published adapter snapshots, so training cannot change an in-flight request.
The existing `round_robin` mode remains the default.

## Design

- The trainer owns the base storage. HF packs compatible tensors once at startup;
  FSDP keeps persistent base shards and maps each shard to the matching vLLM TP
  rank. Full weights gathered during training are temporary, not shared storage.
  Small normalization and router weights remain replicated to match vLLM.
- CUDA IPC gives the borrowing sampler views of that storage. Startup checks
  device placement, layouts and aliases. Unsupported layouts fail explicitly;
  vocabulary padding may need a separate allocation, reported by the loader.
- A dedicated sampler remains available. The borrowing sampler accepts bounded
  work only when its estimated benefit covers switching costs and the next known
  training window. Cold models use one bounded probe at a time.
- Training closes borrowing admission, waits for active requests, and sleeps the
  sampler to release KV memory. Sampling wakes it again. Cancellation and shutdown
  drain admitted work before storage is released.
- Fair admission, adapter grouping and queue limits serve multiple tenants.
  Telemetry separates queueing, draining, phase changes and execution. Predictive
  wave alignment is outside this milestone; requests are not delayed to pack
  future training bursts.

## Enable it

Start from `config/tuft_config.shared-weight.h100.yaml` for HF or
`config/tuft_config.shared-weight.fsdp.h100.yaml` for FSDP. Select
`scheduling.policy: shared_weight`, enable `colocate`, and set an explicit
`sampling_kv_cache_memory_bytes` budget that fits alongside training memory.

HF requires TP=1. FSDP requires `fsdp_num_gpus == tensor_parallel_size`, with the
shared group on one node. DP includes the borrowing sampler and must be at least
two. Thus HF DP=2 uses two GPUs; FSDP with two ranks and DP=2 uses four. Each base
model reserves its own training group. OpenAI-compatible serving stays on the
dedicated sampler; scheduled Tinker sampling uses both.

Supported layouts cover BF16 Qwen3 and Qwen3.5-family dense/MoE text models.
Shared mode checks the tested torch 2.11.0 / vLLM 0.24.0 pair. It uses released
packages, but CUDA IPC reducers and some model/PEFT integration are internal,
version-sensitive interfaces. An upgrade needs fresh compatibility validation.
The producer process must outlive its aliases; actor loss requires a restart.
An observed actor or phase failure disables borrowing. Already-published snapshots
remain servable on the dedicated sampler; training and new publications require a
restart. FSDP input rejections and known checkpoint I/O failures are recoverable
only when all ranks have finished safely. Uncertain collective or partial-mutation
failures disable the group.

## Evidence and limits

See the [experiment report](../../../../benchmarks/shared-weight/README.md) for
saved measurements and figures. An earlier revision of this code completed an
eight-tenant agent workload with verified shared storage and lower GPU time per
update, but worse startup p95. This extracted branch has CPU validation; the
saved GPU results are not a new measurement of this branch.

Rate calibration currently waits for a replica to drain. Sustained-load learning,
request-level engine timing, and representative warmup are follow-up work.
Training/inference numerical calibration is a separate problem. This mode does
not promise equal log probabilities across engines or universal speedups.
