# Cost of Switching and Static Partitioning: Section Data

This file contains only the data needed for the subsection:

```latex
\subsection{Cost of switching and static partitioning}
```

Primary model/config used for switching-cost comparison:

- Model: Qwen3-32B
- Model path: `/mnt/cpfs/shared/checkpoints/qwen/qwen3/qwen3-32B`
- Hardware: 4×A100 80GB
- Main parallel config: TP=4, DP=1
- Sampling workload for switching benchmark: `num_prompts=32`, `max_tokens=256`, `max_model_len=2048`
- Training workload for switching benchmark: `train_batch_size=2`, `train_seq_len=512`

## 1. Steady-State Switch Cost Summary

| Method | T→S hot switch | S→T hot switch | Round-trip hot switch | Notes |
|---|---:|---:|---:|---|
| Static coexistence | 0.0 ms | 0.0 ms | 0.0 ms | No explicit switch, but training OOM due to memory pressure |
| Process-level reload | 219749.8 ms | 80296.8 ms | 300046.6 ms | Kill old runtime and reload target runtime |
| vLLM sleep/wake + FSDP simulated reload | 6091.7 ms | 5325.0 ms | 11416.7 ms | vLLM sleep/wake real; FSDP offload/reload simulated |
| FlexGPU zero-copy | 370.3 ms | 205.5 ms | 575.8 ms | Real bidirectional zero-copy round-trip |

## 2. Process-Level Reload Breakdown

Source: `exp/results/qwen3-32B_tp4_process_switch.json`

| Direction | Stage | Time |
|---|---|---:|
| T→S | kill training / FSDP cleanup | 3162.0 ms |
| T→S | load vLLM sampling runtime | 216587.9 ms |
| T→S | total | 219749.8 ms |
| S→T | kill vLLM sampling runtime | 2096.1 ms |
| S→T | load FSDP training runtime | 78200.7 ms |
| S→T | total | 80296.8 ms |

Key observation:

- T→S is dominated by vLLM runtime loading.
- S→T is dominated by training runtime loading.
- Process-level switching is two to three orders of magnitude slower than FlexGPU hot switching.

## 3. vLLM Sleep/Wake + FSDP Simulated Reload Breakdown

Source: `exp/results/qwen3-32B_tp4_sleep_wake.json`

| Direction | Stage | Time |
|---|---|---:|
| T→S | FSDP offload/sleep simulation | 541.0 ms |
| T→S | vLLM wake | 5550.7 ms |
| T→S | total | 6091.7 ms |
| S→T | vLLM sleep | 5215.1 ms |
| S→T | FSDP reload/wake simulation | 109.9 ms |
| S→T | total | 5325.0 ms |

Important caveat:

- vLLM sleep/wake is real.
- FSDP has no native sleep/wake interface in this benchmark, so the FSDP-side offload/reload cost is simulated by moving model/optimizer state.

## 4. FlexGPU Zero-Copy Breakdown

Source: `exp/results/flexgpu_direct_32b_tp4.log` and `exp/results/qwen3-32B_tp4_flexgpu.json`

### 4.1 Per-round raw switch measurements

| Round | T→S total | S→T total | Training release | IPC descriptor | Inject alias | Training skeleton build | Base alias | Sampling throughput | Release alloc | Free after release |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 cold | 969.1 ms | 38517.9 ms | 678.3 ms | 231.6 ms | 59.2 ms | 38517.9 ms | 0.0 ms | 839.7 tok/s | 17.54 GB | 59.50 GB |
| 2 hot | 330.1 ms | 211.1 ms | 268.7 ms | 3.3 ms | 58.0 ms | 206.9 ms | 4.2 ms | 837.5 tok/s | 17.54 GB | 59.76 GB |
| 3 hot | 407.1 ms | 201.5 ms | 344.2 ms | 4.2 ms | 58.7 ms | 197.1 ms | 4.4 ms | 837.6 tok/s | 17.54 GB | 59.74 GB |
| 4 hot | 353.7 ms | 201.8 ms | 290.0 ms | 4.1 ms | 59.6 ms | 197.1 ms | 4.8 ms | 854.2 tok/s | 17.54 GB | 59.73 GB |
| 5 hot | 390.5 ms | 207.6 ms | 329.4 ms | 4.1 ms | 57.0 ms | 202.9 ms | 4.7 ms | 889.0 tok/s | 17.54 GB | 59.72 GB |

### 4.2 FlexGPU hot-path averages

Hot rounds: rounds 2–5.

| Direction | Component | Mean time |
|---|---|---:|
| T→S | training runtime release | ~308.1 ms |
| T→S | IPC descriptor generation | ~3.9 ms |
| T→S | vLLM alias injection | ~58.3 ms |
| T→S | total | 370.3 ms |
| S→T | training skeleton build | ~201.0 ms |
| S→T | base storage alias | ~4.5 ms |
| S→T | total | 205.5 ms |
| Round-trip | T→S + S→T | 575.8 ms |

Key observation:

- FlexGPU does not reload base weights.
- T→S hot path is dominated by training runtime release.
- S→T hot path is dominated by lightweight training skeleton construction.
- Actual base alias is only about 4–5 ms.

## 5. Cold vs Hot Switch Cost

| Method | Direction | Cold | Hot | Why cold is larger |
|---|---|---:|---:|---|
| FlexGPU | T→S | 969.1 ms | 370.3 ms | First-time vLLM/dummy/cache setup |
| FlexGPU | S→T | 38517.9 ms | 205.5 ms | First full training build; later skeleton+alias only |
| Process reload | T→S | 219749.8 ms | 219749.8 ms | Every switch reloads target runtime |
| Process reload | S→T | 80296.8 ms | 80296.8 ms | Every switch reloads target runtime |
| Sleep/wake | T→S | 6091.7 ms | 6091.7 ms | Measured as fixed sleep/wake/offload path |
| Sleep/wake | S→T | 5325.0 ms | 5325.0 ms | Measured as fixed sleep/wake/reload path |

Recommended paper choice:

- Use hot switch cost in main text for steady-state RL iterations.
- Mention cold initialization separately or in appendix.

## 6. Static Coexistence / Static Partitioning Memory Pressure

Static coexistence has zero switch latency but keeps both runtimes resident.

| Config | Training resident memory | Sampling memory budget | Total pressure | Training outcome | Sampling throughput |
|---|---:|---:|---:|---|---:|
| Qwen3-32B TP=2, DP=2 | 32.0 GB/GPU | 48.0 GB/GPU | 80.0 GB/GPU | OOM / not viable | 941.9 tok/s per TP group, 1883.8 tok/s total DP=2 |
| Qwen3-32B TP=4, DP=1 | 17.5 GB/GPU | 62.5 GB/GPU | 80.0 GB/GPU | OOM / not viable | 937.5 tok/s |

Key observation:

- Static coexistence removes explicit switching, but it consumes the full 80GB A100 budget.
- Training becomes infeasible even though sampling alone can run under the reduced vLLM memory budget.

## 7. Static Partitioning Idle Window Estimate

Representative aligned RL workload:

- Sampling: `num_prompts=128`, `max_tokens=1024`
- Training: `global_batch=128`, `seq_len=1024`
- Best feasible config: TP=2, DP=2

Measured/estimated times:

| Phase | Time |
|---|---:|
| Sampling | 20.35 s |
| Training | 32.69 s |
| Sample + train | 53.04 s |

Static partition implication:

| Static partition window | Idle resource |
|---|---|
| During sampling (~20.35 s) | Training partition idle |
| During training (~32.69 s) | Sampling partition idle |

This is an analytical idle-window estimate from phase times, not a direct GPU utilization trace.

## 8. Minimal data to cite in paper

If space is limited, cite only these numbers:

| Claim | Data |
|---|---|
| Process reload is expensive | 300.05 s round-trip switch |
| Sleep/wake is better but still seconds-level | 11.42 s round-trip switch |
| FlexGPU is sub-second | 0.58 s hot round-trip switch |
| FlexGPU T→S breakdown | release 308.1 ms, descriptor 3.9 ms, alias inject 58.3 ms |
| FlexGPU S→T breakdown | skeleton build 201.0 ms, base alias 4.5 ms |
| Static coexistence is not viable | 80GB/GPU pressure and training OOM |
| Static partition wastes phase-specific resources | sampling 20.35 s, training 32.69 s; inactive partition idle in each phase |

## 9. Suggested figures/tables

### Figure 1: Round-trip switch latency, log scale

| Method | Round-trip switch |
|---|---:|
| Process reload | 300.05 s |
| Sleep/wake | 11.42 s |
| FlexGPU | 0.58 s |

### Figure 2: FlexGPU hot switch breakdown

| Direction | Release/build | Descriptor | Alias/inject | Total |
|---|---:|---:|---:|---:|
| T→S | 308.1 ms | 3.9 ms | 58.3 ms | 370.3 ms |
| S→T | 201.0 ms | 0.0 ms | 4.5 ms | 205.5 ms |

### Table: Static coexistence memory pressure

| Config | Training | Sampling | Total pressure | Result |
|---|---:|---:|---:|---|
| 32B TP2/DP2 | 32.0 GB | 48.0 GB | 80.0 GB | Training OOM |
| 32B TP4/DP1 | 17.5 GB | 62.5 GB | 80.0 GB | Training OOM |
