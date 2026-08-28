# TuFT Paper Experiments — Final Result Data

> Model: Qwen3-4B (primary) / Qwen3-32B (estimated); hardware: 8×A100; workload: GSM8K/MATH/MBPP/Countdown
> single-turn generation tasks; 10 train steps per tenant, 32 samples per step; t8 = 8 tenants.
> Legend: measured = simulator-measured; estimated = derived from measurements/microbenchmarks.

## 1. Throughput (t8, steps/h)
| baseline | 4B | 32B |
|---|---|---|
| optimal (flex) | 541.9 | ≈342 |
| colocate | 381.5 | OOM |
| static (2:6) | 283.9 | ≈179 |
| serial | 184.3 | ≈116 |
| unified | 39.9 | ≈26 |

Additional: `static 1:1` measured = **471.8** (see §6 finding: 1:1 beats 2:6).

## 2. Scaling (4B, steps/h; colocate/unified low-tenant are estimates)
| tenant | optimal | static | serial | colocate | unified |
|---|---|---|---|---|---|
| t1 | 303 | 155 | 156 | 210* | 12* |
| t2 | 508 | 228 | 166 | 305* | 22* |
| t4 | 523 | 255 | 186 | 345* | 30* |
| t8 | 542 | 284 | 184 | 381 | 40 |
| t16 | 414 | 228 | 183* | 305* | 42* |

## 3. SLO (t8)
| baseline | training resp | sampling resp | logprob bias | staleness |
|---|---|---|---|---|
| optimal | 29.6s | 11.3s | 0.0181 | 2.58 |
| static | 72.8s | 16.1s | 0.1141 | 3.45 |
| serial (incl. queueing) | ~146.7s | ~140.3s | 0.1141 | 2.19 |
| colocate | 25.8s | 16.8s | 0.1141 | 1.84 |
| unified | ~20s | ~703s | 0.0195 | ≈0.5 |

## 4. Per-framework logprob bias and Corrector
- Per-framework bias: optimal **0.0181** / static **0.1141** / serial **0.1141** / colocate **0.1141** / unified **0.0195**.
- Corrector bias reduction (held-out): static token 26.5% / sequence 59.9% / adaptive **61.0%**; optimal ~0% (already at floor).

## 5. Key findings
1. **1:1 static beats 2:6** (471.8 vs 283.9): the bottleneck is training-GPU contention
   (8 tenants sharing 2 GPUs, 72.8s/step); 1:1 uses 4 GPUs (28.6s/step) and halves it. Sampling is
   pipelined and does not gate throughput.
2. **32B colocate = OOM** (two weight copies exceed GPU memory); 32B real throughput is blocked by host
   RAM (FSDP fp32 optimizer ≈101GB), so it is estimated.
3. **Slow loop does not convert at t8** (1:7 is already optimal); measured 2:6 optimal=525 < 1:7=541.9,
   showing the 1071 (2:6 what-if) is unrealizable.

## 6. Measured vs estimated
- Measured: 4B throughput/scaling/SLO/bias/corrector, static 1:1, 2:6 optimal, logprob dumps.
- Estimated: all 32B, colocate/unified low-tenant, unified staleness.
- Excluded (simulation, not runnable): component-progression ablation and GPU-utilization model.
