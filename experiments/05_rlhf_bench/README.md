# 05. RLHF Benchmark with Bias Correction

## Motivation

Having characterized framework mismatch and built a predictor to estimate per-sequence bias, this experiment validates whether the correction actually **improves RLHF training** on a held-out task (GSM8K math reasoning).

It is the main end-to-end evaluation experiment of the paper.

## What the experiment does

Compares IS-REINFORCE and GRPO across three correction conditions:

| Condition | Description |
|-----------|-------------|
| `baseline` | No correction applied |
| `predictor` | Per-sequence bias estimated by the trained MLP predictor and subtracted |
| `global_mean` | Global mean bias (scalar) subtracted as ablation baseline |

Evaluations are run on GSM8K held-out test set (greedy decoding, exact-match accuracy) every `eval_interval` training steps. Training metrics (reward, approx_loss, IS weight distribution, clip rate) are logged per step.

Planned full experiment: 3 independent seeds × 2 algorithms × 3 correction conditions × {GSM8K, MATH, Countdown}.

## Files

| File | Description |
|------|-------------|
| `rlhf_bench.py` | Main training loop; combines algorithm + correction logic |
| `rlhf_compare.py` | Side-by-side comparison utilities for multiple runs |
| `plot_bench.py` | Per-run training curve plots |
| `plot_multi_seed.py` | Multi-seed aggregated plots with error bars |
| `plot_rlhf_compare.py` | Cross-condition comparison plots |
| `launch_math_multi_seed.sh` | Launch 3-seed MATH experiment |
| `launch_countdown_single_seed.sh` | Launch Countdown single-seed run |
| `launch_humaneval_single_seed.sh` | Launch HumanEval single-seed run |
| `launch_mbpp_single_seed.sh` | Launch MBPP single-seed run |
| `calibration_results.fit.json` | Mock backend calibration parameters |
| `calibration_results.jsonl` | Raw calibration measurements |
| `proposal.md` | Experiment design proposal |

## Results

Pre-computed results (jsonl/json) are in `results/`. File naming convention:
```
results/bench_<dataset>_seed<N>/  — single-run results
results/bench_<dataset>_multiseed/  — aggregated multi-seed results
```

Predictor checkpoint used:
```
/mnt/nas/hanzhang.yhz/evaluation/predictor/checkpoints/bench_v1/best.pt
```

## Running

```bash
# Multi-seed MATH experiment (recommended)
bash launch_math_multi_seed.sh

# Custom single run
python rlhf_bench.py \
  --groups all \
  --num_steps 50 \
  --eval_interval 5 --eval_n 30 \
  --grpo_g 8 --buffer_size 8 \
  --predictor_ckpt ../03_bias_predictor/predictor/checkpoints/best.pt \
  --output results/bench_gsm8k_seed42
```
