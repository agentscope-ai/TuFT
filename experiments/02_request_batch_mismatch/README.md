# 02. vLLM Batching & Position Effect on Logprobs

## Motivation

Framework mismatch has two components:
1. **Serving vs. training engine difference** (shown in `01_mismatch_motivation`)
2. **Intra-serving non-determinism** — does vLLM's own output change depending on batch composition?

This experiment decomposes the second component by asking:
- Does **batch size** affect the logprobs vLLM assigns to a fixed sequence?
- Does the **position** of a request within a batch matter?
- Do **co-batched adapters** (different LoRA weights) interfere with each other's logprob values?

All tests score the **same token sequence** under controlled batch configurations. Any differences beyond floating-point noise expose systematic, reproducible bias sources.

## Story Connection

This is a supporting experiment for the framework mismatch story. It shows that mismatch in RLHF is not just a training-vs-serving engine issue — it also has an intra-serving component driven by batch position. This motivates the need for a learned predictor that can condition on batch context.

## Files

| File | Description |
|------|-------------|
| `test_adapter_independence.py` | Scores fixed sequence in batches of same / different / mixed adapters; tests whether adapter co-scheduling affects logprobs |
| `test_batch_position.py` | Scores fixed sequence at varying batch sizes and positions within a batch; measures systematic logprob shift |
| `test_framework_mismatch.py` | End-to-end mismatch between vLLM scoring and TuFT training path |
| `utils.py` | Shared utilities (engine creation, adapter setup, diff computation) |
| `run_all.sh` | Runs all three tests in sequence |
| `plot_test1.py / plot_test2.py / plot_test3.py` | Visualization scripts for each test |

## Running

```bash
# Requires a running vLLM-compatible GPU environment
bash run_all.sh

# Or individual tests:
python test_batch_position.py --model Qwen/Qwen3-0.6B --num-adapters 4
python test_adapter_independence.py --model Qwen/Qwen3-0.6B --num-adapters 4
python test_framework_mismatch.py
```

## Data

Results data (~253 MB) is **not included** in the repo. Original results are at:
```
/mnt/nas/hanzhang.yhz/evaluation/microbench/02_requests_mismatch_moti/results/
```
