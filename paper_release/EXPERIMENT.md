# TuFT Paper Experiments — Full Documentation

This directory (`TuFT/paper_release/`) gathers the **key configs, scripts and
documentation** for the paper experiments. The implementation source is NOT duplicated
here — it lives in the TuFT library at `TuFT/src/tuft/` (referenced in §7a). Large data
files and intermediate artifacts are excluded and archived to
`/mnt/nas/hanzhang.yhz/tuft_archive/`.

---

## 1. System overview

TuFT is a multi-tenant online RL training system. Its core is the **flex backend**, which
fuses training (torch-TP) and sampling (vLLM) on the same GPUs, uses IPC-aliased weights
for low-overhead switching, and rebalances the train/sample GPU split via two control loops.

- **Training backend**: FSDP / torch-TP (`backends/fsdp_training_backend.py`, ...).
- **Sampling backend**: vLLM replicas (fixed or flex).
- **Fast loop**: delay holds that keep sampling fresh.
- **Slow loop**: periodic rebalancing of the fixed↔flex GPU split.
- **Corrector**: corrects the sequence-logprob bias introduced by disaggregated pipelines.
- Source entry points: `src/tuft/server.py`, `cli.py`, `sampling_controller.py`,
  `training_controller.py`, `corrector.py`.

## 1a. Flex backend basics: startup & single-GPU RL (core primitive)

The flex backend (`FusedTorchTPVLLMFlexBackend`) fuses torch-TP training and vLLM sampling in
one process and switches between them via a zero-copy IPC weight alias. It is the core
primitive and runs on a **single GPU** without a server.

- Implementation: `../src/tuft/backends/flex/` (`torchtp.py` = `FusedTorchTPVLLMFlexBackend`,
  `torchtp_training.py`, `torchtp_zero_copy.py`, `training_worker_pool.py`, `vllm_worker.py`).
- Single-GPU RL example: `examples/countdown_rl/train_flex.py` (+ `env.py`). No server needed;
  it runs the `training → sampling → training` loop on one GPU with `training_backend: flex`,
  `tensor_parallel_size: 1`.

Run on one GPU:
```bash
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
TUFT_FLEX_COUNTDOWN_GPU=1 .venv/bin/python examples/countdown_rl/train_flex.py \
    --model-name qwen3-0.6b --num-steps 10 --batch-size 2 --group-size 4 --tp-size 1
```

Notes:
- Requires the model checkpoint (e.g. Qwen3-0.6B) and the Countdown dataset cached locally for
  offline use.
- Server-based multi-GPU deployments use the configs in §2 instead; the single-GPU flex path
  above is the minimal runnable form of the backend.
- **Verified on a single A100 (2026-08-28)**: `FlexBackend ready in ~120s`, then 3 RL steps with
  real gradient updates (rollouts 8/8/4, loss 0.81/-0.05/0.62, reward 0.1014→0.1065).

## 2. Deployment modes (5 baselines + optimal)

| Mode | GPU allocation (8 GPUs) | Config |
|---|---|---|
| **optimal** | 1 flex(train) + 7 fixed(sample), 1:7 | `tuft_config_eval_optimal_8gpu.yaml` |
| **static_disagg** | 2 FSDP(train) + 6 vLLM(sample), 2:6 | `tuft_config_eval_static_disagg_8gpu.yaml` |
| **serial_async** | 2 train + 6 sample, per-tenant time-sharing | `tuft_config_eval_serial_async_8gpu.yaml` |
| **colocate_2copies** | each GPU holds train+sample copies | `tuft_config_eval_colocate_2copies_8gpu.yaml` |
| **unified_engine** | 8×1 single engine alternating train/sample | `tuft_config_eval_unified_engine_8gpu.yaml` |
| static 1:1 | 4 FSDP(train) + 4 vLLM(sample) | `tuft_config_paper28_static_1to1_4b.yaml` |

`tuft_config.example.yaml` documents all available config options.

## 3. Workloads

`workloads/paper_tN.yaml`: N tenants, each with 4 single-turn generation tasks
(GSM8K / MATH / MBPP / Countdown), `max_tokens` 768/256/1024/512 (mean 640, actual
generation ~0.6 fill ≈384); 10 train steps per tenant, 32 samples per step, with a
per-tenant staleness limit.
Generator: `scripts/gen_eval_workload.py`.

## 4. Evaluation experiments

### (1) Throughput & SLO
End-to-end throughput (steps/h) and SLO (training/sampling response time, logprob bias,
staleness) for each deployment at t8. Runnable via `run_eval_matrix.py` with each mode
config; see `results/final_data.md §1, §3`.

> The component-progression ablation and the GPU-utilization model were simulation-only and
> have been **excluded** from this release (they are not runnable experiments).

### (2) Scaling
Throughput of each mode at t1/t2/t4/t8/t16.
- Measured: optimal/static t1–t16, serial t1–t8, colocate t8, unified t8.
- Estimated: colocate/unified low-tenant, parts of t16.
- Figure/tables: `scripts/qoder_plot_scaling.py`, `qoder_paper_tables.py`.

### Supplementary experiments (trimmed)
The logprob-dump and 32B-generality experiments appeared in the paper, but their
configs/scripts were **removed** to keep this release minimal — only the evaluation above is
runnable here. Reference 32B/switch-overhead numbers remain in `microbench/` and
`results/final_data.md`.

## 5. Running and reproduction

The evaluation drives the TuFT server through the bundled **simulator** (`simulator/`,
driver `simulator/run.py`, package `simulator/simulator/`). Point `run_eval_matrix.py` at it
with `--simulator-dir simulator` (its default is an external path).

Single entry point `scripts/run_eval_matrix.py` (`MODES` maps mode→config):
```bash
cd TuFT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TMPDIR=/root/tuft_local/tmp RAY_TMPDIR=/root/tuft_local/ray_tmp \
  RAY_memory_usage_threshold=0.99 TUFT_CHECKPOINT_DIR=/root/tuft_local/checkpoints \
  TUFT_ADAPTER_EXPORT_DIR=/root/tuft_local/adapter_exports
.venv/bin/python scripts/run_eval_matrix.py --modes <mode> \
  --workload workloads/paper_t8.yaml --results-dir <out> \
  --simulator-dir paper_release/simulator \
  --gpu-map "<mode>=0,1,2,3,4,5,6,7" --port 10651 \
  --health-timeout-s 2400 --sim-timeout-s 2400
```
Mode names match config stems, e.g. `optimal_8gpu`, `static_disagg_8gpu`, `paper28_static_1to1_4b`.
Analysis/plotting: `qoder_paper_tables.py`, `qoder_plot_scaling.py`, `qoder_dump_raw_data.py`.

Remaining external dependencies (not bundled): model checkpoints (e.g. Qwen3-4B-Base on CPFS)
and the HF datasets (GSM8K / MATH / MBPP / Countdown) for offline use.

## 6. Key findings (summary)

1. **1:1 static beats 2:6** (471.8 vs 283.9): the bottleneck is training-GPU contention
   (8 tenants sharing 2 GPUs, 72.8s/step → fsdp4 halves it to 28.6s); sampling is pipelined
   and does not gate throughput.
2. **32B colocate = OOM**; 32B real throughput is host-RAM-blocked → estimated.
3. **Slow loop does not convert at t8** (1:7 is already optimal); 2:6 optimal=525 < 1:7=541.9.

Full data in `results/final_data.md`.

## 7. File map

```
paper_release/
├── README.md                 # quick navigation
├── EXPERIMENT.md             # this document
├── configs/                  # 5 evaluation modes + 1:1 static + example (7)
├── workloads/                # paper_t1/t2/t4/t8/t16 (5)
├── scripts/                  # run_eval_matrix + workload gen + analysis/plot/export (7)
├── examples/countdown_rl/    # single-GPU flex RL example (train_flex.py + env.py)
├── simulator/                # bundled evaluation simulator (run.py + package + configs)
├── microbench/               # 32B/4B switch-overhead reference (7)
├── ref/                      # exp.py + paper.tex (paper source)
└── results/final_data.md     # final result tables
```
Implementation source: `../src/tuft/` (77 modules) — referenced, not copied.

## 7a. Implementation code map (in `../src/tuft/`)
- **Flex backend**: `backends/flex/` (`flex_backend.py`, `torchtp.py`, `torchtp_training.py`,
  `torchtp_zero_copy.py`, `training_worker_pool.py`, `vllm_runtime_proxy.py`, `vllm_worker.py`),
  `backends/torchtp_backend.py`.
- **Training backends**: `backends/fsdp_training_backend.py`, `backends/hf_training_model.py`,
  `backends/training_backend.py`.
- **Sampling**: `backends/sampling_backend.py`, `sampling_pipeline.py`, `sampling_router.py`.
- **Fast/slow loop schedulers**: `schedulers/rl_loop_scheduler.py`, `runtime_scheduler.py`,
  `switch_scheduler.py`.
- **Baselines**: serial_async → `runtime/serial_async_gate.py`; unified_engine →
  `runtime/unified_engine_gate.py`; static_disagg/colocate → fsdp+sampling backends via configs.
- **Controllers / server**: `server.py`, `backend.py`, `sampling_controller.py`,
  `training_controller.py`, `corrector.py`, `sequence_executor.py`.

## 7b. Archived data
Raw result data and intermediate configs were archived out of `TuFT/` to keep it clean:
`/mnt/nas/hanzhang.yhz/tuft_archive/` → `exp_data/{results,traces,logprob_dump}`,
`configs_intermediate/` (69), `misc/stray_tilde_cache`. Summarized results remain in
`results/final_data.md`.

## 8. Known limitations
- 32B end-to-end: requires a larger-memory node or a bf16 optimizer.
- Colocate low-tenant: FSDP world_size must be ≥ tenant count (single tenant needs
  batch/8 + gradient accumulation; the current `len(data)≥world` guard blocks this).
- This session's environment reaps long-running processes; run long static jobs in a stable terminal.
