# TuFT Paper Release

Curated, mostly self-contained directory to run the paper **evaluation** experiments. It
bundles the configs, workloads, runner scripts, the flex-backend single-GPU RL example, and
the evaluation **simulator**. Only model checkpoints and HF datasets remain external. Large
data files, intermediate scripts and non-evaluation configs are excluded.

## Quick navigation
- **Full documentation**: `EXPERIMENT.md` (system / flex-backend single-GPU / deployments / experiments / reproduction / findings)
- **Flex backend single-GPU RL**: `EXPERIMENT.md §1a` + `examples/countdown_rl/train_flex.py`
- **Final data**: `results/final_data.md` (throughput / scaling / SLO / bias tables)
- **Paper source**: `ref/paper.tex`, `ref/exp.py`

## Layout
| Directory | Content | Count |
|---|---|---|
| `configs/` | 5 evaluation modes + 1:1 static + example | 7 |
| `workloads/` | paper_t1/t2/t4/t8/t16 | 5 |
| `scripts/` | run_eval_matrix + workload gen + analysis/plot/export | 7 |
| `examples/countdown_rl/` | single-GPU flex RL (train_flex.py + env.py) | 2 |
| `simulator/` | bundled evaluation simulator (run.py + package + configs) | – |
| `microbench/` | 32B/4B switch-overhead reference | 7 |
| `ref/` | paper tex + exp.py | 2 |
| `results/` | final_data.md | 1 |

Framework source: `../src/tuft/` (77 modules).

## Reproduce
See `EXPERIMENT.md §5`; the single entry point is `scripts/run_eval_matrix.py`.
