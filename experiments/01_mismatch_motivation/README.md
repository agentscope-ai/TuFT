# 01. Training/Sampling Logprob Mismatch (Motivation)

**Story role**: Core motivation experiment. Demonstrates that vLLM (sampling path) and HF/Tinker training forward pass produce systematically different logprobs on the same tokens and weights — causing IS weights to deviate from 1.0 even when the policy has not changed. This is the central problem the paper solves.

---

---

## 01. Training/Sampling Logprob Mismatch (`01_mismatch_moti/`)

### Motivation

In RLHF pipelines, log probabilities for the same sequence are computed along two different code paths inside the Tinker stack:

1. **Sampling path** — `compute_logprobs()` on a saved checkpoint (serving graph).
2. **Training path** — `forward()` on the live training graph.

This benchmark demonstrates that these two paths produce **different logprobs even for the exact same sequences on the exact same model weights**. Because the importance-sampling ratio is computed as

\[
\text{IS ratio} = \exp\bigl(\sum (\text{logprob}_{\text{training}} - \text{logprob}_{\text{sampling}})\bigr)
\]

a non-zero mismatch causes the IS ratio to deviate from 1.0 even when the policy has not changed, introducing systematic bias into RLHF gradient estimates.

### What the experiment does

1. **Connect** to a running Tinker service.
2. **Create** a LoRA training client on a base model.
3. **Sample** a fixed dataset (prompt + response tokens + their sampling logprobs) from the *initial* model weights.
4. **Run** multi-round training:
   - Before each training step, save current weights and compute logprobs for the fixed sequences via **both** the sampling path and the training path.
   - Record per-token and per-sequence differences.
   - Perform one importance-sampling training step on the fixed data.
5. **Save** raw metrics to JSON and plot the evolution of the mismatch.
6. **Clean up** all saved weight checkpoints on the Tinker server.

### Environment & Dependencies

- Python >= 3.9
- A running Tinker service (default: `http://localhost:10610`)
- `tinker` SDK (Tinker Python client)
- `torch`
- `numpy`
- `matplotlib`
- `transformers`
- `tqdm`

Install dependencies (example):

```bash
pip install tinker torch numpy matplotlib transformers tqdm
```

### Input Parameters

All parameters are passed as command-line flags to `verify_mismatch.py`:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--base-url` | `str` | `http://localhost:10610` | URL of the Tinker service. Can also be set via the `TINKER_BASE_URL` environment variable. |
| `--api-key` | `str` | `None` | API key for the Tinker service. Can also be set via the `TINKER_API_KEY` environment variable. |
| `--base-model` | `str` | `Qwen/Qwen3-0.6B` | Hugging Face model ID used as the base model for LoRA training. |
| `--lora-rank` | `int` | `8` | Rank of the LoRA adapters. |
| `--num-rounds` | `int` | `20` | Number of training rounds. Each round = compute mismatch + one training step. |
| `--learning-rate` | `float` | `1e-4` | Learning rate for the Adam optimizer. |
| `--max-tokens` | `int` | `32` | Maximum number of tokens to generate when sampling fixed responses. |
| `--temperature` | `float` | `0.7` | Sampling temperature for generating fixed responses. |
| `--seed` | `int` | `42` | Random seed. |
| `--output-dir` | `str` | `.` | Directory where `mismatch_results.json` and `logprob_mismatch.png` will be written. |
| `--num-prompts` | `int` | `5` | Number of fixed prompts to use (subset of the built-in 5 prompts). |
| `--num-samples-per-prompt` | `int` | `2` | Number of response sequences to sample per prompt. |

### Running the Experiment

```bash
cd microbench/01_mismatch_moti

# Run with default settings (requires Tinker at localhost:10610)
python verify_mismatch.py

# Run with custom settings
python3 verify_mismatch.py \
  --base-url http://127.0.0.1:10610 \
  --base-model Qwen/Qwen3-4B \
  --lora-rank 16 \
  --num-rounds 30 \
  --learning-rate 5e-5 \
  --max-tokens 64 \
  --temperature 0.7 \
  --output-dir ./results \
  --num-prompts 5 \
  --num-samples-per-prompt 4
```

### Output Files

| File | Description |
|------|-------------|
| `mismatch_results.json` | Per-round metrics (see table below). |
| `logprob_mismatch.png` | 4-panel plot showing how the mismatch evolves over training rounds. |

### Output Metrics (per round)

Each entry in `mismatch_results.json` contains the following fields:

#### 1. Per-Token Differences

| Metric | Variable | Computation | Purpose |
|--------|----------|-------------|---------|
| `mean_diff` | Mean difference | `mean(sampling_lp - training_lp)` | Direction of systematic bias (positive = sampling logprobs are higher). |
| `mean_abs_diff` | Mean absolute difference | `mean(\|sampling_lp - training_lp\|)` | Overall magnitude of the per-token mismatch. |
| `max_abs_diff` | Maximum absolute difference | `max(\|sampling_lp - training_lp\|)` | Worst-case single-token discrepancy. |
| `std_diff` | Standard deviation of differences | `std(sampling_lp - training_lp)` | Spread of the per-token mismatch distribution. |
| `num_tokens` | Total token count | — | Number of response tokens evaluated in this round. |

#### 2. Sequence-Level Cumulative Differences

| Metric | Variable | Computation | Purpose |
|--------|----------|-------------|---------|
| `mean_cum_diff` | Mean cumulative difference | `mean(sum(diff over sequence))` | Foundation for IS-weight bias; equals `mean(log(IS weight))`. |
| `max_cum_diff` | Maximum cumulative difference | `max(\|sum(diff)\|)` | Worst-case sequence-level discrepancy. |
| `std_cum_diff` | Std dev of cumulative differences | `std(sum(diff over sequence))` | Variability of sequence-level mismatch. |
| `mean_sampling_logprob` | Mean sampling logprob | `mean(sum(sampling_lp over sequence))` | Average accumulated logprob from the sampling path. |
| `mean_training_logprob` | Mean training logprob | `mean(sum(training_lp over sequence))` | Average accumulated logprob from the training path. |

#### 3. IS Weight Distribution

| Metric | Variable | Computation | Purpose |
|--------|----------|-------------|---------|
| `mean_is_weight` | Mean IS weight | `mean(exp(cum_diff))` | Systematic offset of IS weights from 1.0. |
| `std_is_weight` | Std dev of IS weights | `std(exp(cum_diff))` | Variability of IS weights. |
| `min_is_weight` | Minimum IS weight | `min(exp(cum_diff))` | Lower bound of IS weight outliers. |
| `max_is_weight` | Maximum IS weight | `max(exp(cum_diff))` | Upper bound of IS weight outliers. |
| `p99_is_weight` | P99 IS weight | `np.percentile(exp(cum_diff), 99)` | Extreme upper quantile of IS weights. |
| `p_out_clip_02` | Proportion outside clip ε=0.2 | `mean(\|exp(cum_diff) - 1\| > 0.2)` | **Core metric**: fraction of sequences whose IS weight exceeds the 0.2 clip threshold. |
| `p_out_clip_01` | Proportion outside clip ε=0.1 | `mean(\|exp(cum_diff) - 1\| > 0.1)` | Strict standard: fraction of sequences exceeding the 0.1 clip threshold. |

### Fixed Prompts

The experiment uses a built-in set of 5 fixed prompts. The `--num-prompts` flag controls how many of them are used:

1. `Q: What is the capital of France?\nA:`
2. `Q: Solve 2 + 3 * 4.\nA:`
3. `Q: Who wrote 'Romeo and Juliet'?\nA:`
4. `Q: What is the largest planet in our solar system?\nA:`
5. `Q: How many continents are there on Earth?\nA:`

Responses are sampled once at the beginning of the experiment and then held fixed for all training rounds. This isolates the mismatch from any data-distribution shift.

### How Logprobs Are Computed

**Sampling path** (`compute_sampling_logprobs`):
- Concatenates prompt tokens + full response tokens into a single sequence.
- Calls `sampling_client.compute_logprobs()` on the serving graph.
- Extracts the logprob values at the response-token positions.

**Training path** (`compute_training_logprobs`):
- Builds a `cross_entropy` datum where:
  - `model_input` = prompt tokens + response tokens (excluding the last one)
  - `target_tokens` = padding zeros followed by the full response tokens
  - `weights` = all zeros (so no gradient is produced, only logprobs are returned)
- Calls `training_client.forward(..., loss_fn="cross_entropy")` on the training graph.
- Extracts the logprob tensor from `loss_fn_outputs["logprobs"]` and slices out the response-token positions.

Both paths operate on **identical token sequences** and **identical model weights**, yet their returned logprobs differ.

### Interpreting Results

- **`mean_abs_diff` < 0.01**: Very small mismatch; unlikely to cause significant IS bias.
- **`mean_abs_diff` 0.01–0.05**: Moderate mismatch; may introduce noticeable gradient bias in long sequences.
- **`mean_abs_diff` > 0.05**: Large mismatch; IS weights will be heavily distorted, leading to unstable training.
- **`p_out_clip_02` > 0.1**: More than 10% of sequences have IS weights deviating from 1.0 by more than 0.2. In PPO this means >10% of data would be clipped to near-zero advantage, wasting compute.
- **`p_out_clip_01` > 0.1**: Strict standard — more than 10% of sequences exceed the tighter 0.1 clip threshold.
- **`mean_is_weight` > 1.05 or < 0.95**: Systematic shift in IS weights; the training and sampling paths have a consistent directional bias.
- **`max_is_weight` > 1.5 or `min_is_weight` < 0.67**: Extreme outliers in IS weights indicate severe mismatch on specific sequences.
- **`p99_is_weight` > 1.3**: The top 1% of sequences have highly inflated IS weights, suggesting a long tail of extreme mismatch.
- If `mean_diff` is consistently positive, the serving graph systematically assigns higher logprobs than the training graph (or vice versa if negative).

### Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `Connection refused` error | Tinker service not running at `--base-url`. | Start the Tinker service or check the URL/port. |
| `Sampling did not return logprobs` | The sampling client was created without logprob-returning configuration. | Check Tinker server config; ensure `compute_logprobs` is supported. |
| Very large `max_abs_diff` (>1.0) | Model divergence or tokenization mismatch between paths. | Verify `--base-model` is consistent with the Tinker server loaded model. |
| Empty `mismatch_results.json` | Script crashed before saving. | Check stderr for Python tracebacks; verify write permissions in `--output-dir`. |
| Out of memory | `--max-tokens` or `--num-prompts` too large for the GPU. | Reduce `--max-tokens`, `--num-prompts`, or `--num-samples-per-prompt`. |

### Expected Conclusion

The benchmark is designed to show that `mean_abs_diff` and `max_abs_diff` remain non-zero throughout training, confirming that **training and sampling logprobs are inherently inconsistent** in this stack. This validates the need for careful handling of importance-sampling ratios in RLHF loops.
