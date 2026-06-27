# Experiments

This directory contains all experiments for the TuFT framework mismatch and RLHF scheduling research. Each subdirectory is an independent, self-contained experiment with its own README.

## Paper Story Arc

```
[06] Multi-LoRA scheduling gives throughput gains
           │
           ▼
[01] But framework mismatch exists: same model, sampling ≠ training logprobs
           │
           ▼
[02] Mismatch has an intra-serving component: batch position/composition matters
           │
           ▼
[03] Build a bias predictor to estimate per-sequence mismatch
           │
           ▼
[04] Simulator: model multi-tenant scheduling to study staleness at scale
           │
           ▼
[05] End-to-end RLHF: predictor correction improves training on GSM8K/MATH
```

## Experiments Overview

| Dir | Name | Story Role |
|-----|------|------------|
| `01_mismatch_motivation/` | Training/Sampling Logprob Mismatch | **Core motivation**: demonstrates IS weight bias from framework mismatch |
| `02_request_batch_mismatch/` | vLLM Batching & Position Effect | **Mismatch decomposition**: intra-serving non-determinism from batch composition |
| `03_bias_predictor/` | Framework-Mismatch Bias Predictor | **Method**: learned MLP to predict per-sequence bias for IS weight correction |
| `04_simulator/` | Multi-Tenant RL Training Simulator | **Scheduling model**: simulate concurrent sampling+training to study staleness |
| `05_rlhf_bench/` | RLHF Benchmark with Bias Correction | **End-to-end eval**: IS-REINFORCE / GRPO × {baseline, predictor, global_mean} |
| `06_multilora_scheduling_bench/` | Multi-LoRA Scheduling Benchmark | **Motivation**: mixed-adapter batching throughput gain that creates the mismatch |

## Data Policy

Large data directories (results, logprobs, model checkpoints, mock LoRA weights) are **not tracked in git**. See each experiment's README for the original data location on NAS:
```
/mnt/nas/hanzhang.yhz/evaluation/       ← source for 01–05
/mnt/nas/hanzhang.yhz/multilora_bench/  ← source for 06
```

A `.gitignore` in this directory excludes all large data folders.
