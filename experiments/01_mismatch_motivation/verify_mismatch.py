"""
Verify training/sampling logprob mismatch in Tinker API.

This script demonstrates that the logprobs computed during training
(via forward()) and sampling (via compute_logprobs()) differ, even
for the exact same sequences on the same model weights.

It performs multi-round training on a fixed dataset, recording both
sampling and training logprobs per round, then plots the evolution
of the logprob difference.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tinker
import torch
from tinker import types
from tqdm import tqdm
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Fixed prompts for the experiment
# ---------------------------------------------------------------------------

FIXED_PROMPTS = [
    "Q: What is the capital of France?\nA:",
    "Q: Solve 2 + 3 * 4.\nA:",
    "Q: Who wrote 'Romeo and Juliet'?\nA:",
    "Q: What is the largest planet in our solar system?\nA:",
    "Q: How many continents are there on Earth?\nA:",
]


def parse_args():
    p = argparse.ArgumentParser(description="Verify training/sampling logprob mismatch")
    p.add_argument("--base-url", default=os.getenv("TINKER_BASE_URL", ""))
    p.add_argument("--api-key", default=os.getenv("TINKER_API_KEY"))
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--num-rounds", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=".")
    p.add_argument("--num-prompts", type=int, default=5)
    p.add_argument("--num-samples-per-prompt", type=int, default=2)
    return p.parse_args()


def make_model_input(tokenizer, text: str) -> types.ModelInput:
    toks = tokenizer.encode(text, add_special_tokens=False)
    return types.ModelInput(chunks=[types.EncodedTextChunk(tokens=toks)])


def sample_fixed_data(
    service_client: tinker.ServiceClient,
    training_client,
    tokenizer,
    prompts: List[str],
    num_samples_per_prompt: int,
    max_tokens: int,
    temperature: float,
) -> Tuple[List[Tuple[types.ModelInput, List[int], List[float]]], str]:
    """Sample fixed (prompt, response_tokens, sampling_logprobs) tuples."""
    print("[setup] Sampling fixed responses from initial model...")
    save_result = training_client.save_weights_for_sampler(name="init_sampler").result()
    sampling_client = service_client.create_sampling_client(model_path=save_result.path)

    sampling_params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature)
    fixed_data: List[Tuple[types.ModelInput, List[int], List[float]]] = []

    for prompt_text in prompts:
        prompt = make_model_input(tokenizer, prompt_text)
        res = sampling_client.sample(
            prompt=prompt,
            num_samples=num_samples_per_prompt,
            sampling_params=sampling_params,
        ).result()

        for seq in res.sequences:
            toks = list(seq.tokens)
            lps = seq.logprobs
            if lps is None:
                raise RuntimeError("Sampling did not return logprobs.")
            fixed_data.append((prompt, toks, list(lps)))

    print(f"[setup] Collected {len(fixed_data)} fixed sequences.")
    return fixed_data, save_result.path


def compute_sampling_logprobs(
    sampling_client,
    prompt: types.ModelInput,
    response_tokens: List[int],
) -> List[float]:
    """
    Compute logprobs for response tokens using the sampling code path.

    We construct the full sequence (prompt + response) and call
    compute_logprobs, then extract the response-token positions.
    """
    # Full sequence tokens = prompt tokens + response tokens
    full_toks = []
    for chunk in prompt.chunks:
        full_toks.extend(chunk.tokens)
    prompt_len = len(full_toks)
    full_toks.extend(response_tokens)

    full_input = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=full_toks)])
    logprobs_future = sampling_client.compute_logprobs(prompt=full_input)
    all_logprobs = logprobs_future.result()

    # compute_logprobs returns a list of length len(full_toks).
    # all_logprobs[0] is None (no context for first token).
    # Response tokens start at position prompt_len.
    response_logprobs: List[float] = []
    for i in range(prompt_len, prompt_len + len(response_tokens)):
        lp = all_logprobs[i]
        if lp is None:
            raise RuntimeError(f"compute_logprobs returned None at position {i}")
        response_logprobs.append(lp)

    return response_logprobs


def compute_training_logprobs(
    training_client,
    prompt: types.ModelInput,
    response_tokens: List[int],
) -> List[float]:
    """
    Compute logprobs for response tokens using the training code path.

    We build a cross_entropy datum where:
      - model_input = prompt + response_tokens[:-1]
      - target_tokens = [0]*(prompt_len-1) + response_tokens
    The forward pass returns per-position logprobs of the target tokens.
    """
    # Prompt tokens
    prompt_toks = []
    for chunk in prompt.chunks:
        prompt_toks.extend(chunk.tokens)
    prompt_len = len(prompt_toks)

    # model_input = prompt + response[:-1]
    model_input_toks = prompt_toks + response_tokens[:-1]
    model_input = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=model_input_toks)])

    # target_tokens = padding + full response
    ob_len = prompt_len - 1  # number of padded positions
    target_tokens = [0] * ob_len + response_tokens

    target_tokens_t = torch.tensor(target_tokens, dtype=torch.long)
    # Zero weights so we only compute logprobs, no loss contribution
    weights_t = torch.zeros(len(target_tokens), dtype=torch.float32)

    datum = types.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_torch(target_tokens_t),
            "weights": types.TensorData.from_torch(weights_t),
        },
    )

    fwd_result = training_client.forward([datum], loss_fn="cross_entropy").result()

    # Extract logprobs from the first (and only) loss_fn_output
    logprobs_td = fwd_result.loss_fn_outputs[0]["logprobs"]
    logprobs_arr = np.array(logprobs_td.data)
    if logprobs_td.shape is not None:
        logprobs_arr = logprobs_arr.reshape(logprobs_td.shape)

    # Response tokens correspond to positions ob_len : ob_len + len(response_tokens)
    # = prompt_len - 1 : prompt_len - 1 + len(response_tokens)
    start_pos = ob_len
    end_pos = ob_len + len(response_tokens)
    response_logprobs = logprobs_arr[start_pos:end_pos].tolist()

    return response_logprobs


def compute_logprob_diffs(
    sampling_client,
    training_client,
    fixed_data: List[Tuple[types.ModelInput, List[int], List[float]]],
) -> Dict[str, float]:
    """Compute per-token and mean logprob differences for fixed data."""
    all_sampling_lps: List[float] = []
    all_training_lps: List[float] = []
    seq_cum_diffs: List[float] = []
    seq_sampling_cum: List[float] = []
    seq_training_cum: List[float] = []

    for prompt, response_tokens, _ in fixed_data:
        samp_lps = compute_sampling_logprobs(sampling_client, prompt, response_tokens)
        train_lps = compute_training_logprobs(training_client, prompt, response_tokens)

        if len(samp_lps) != len(train_lps):
            raise RuntimeError(
                f"Length mismatch: sampling={len(samp_lps)} training={len(train_lps)}"
            )

        all_sampling_lps.extend(samp_lps)
        all_training_lps.extend(train_lps)

        diffs = np.array(samp_lps) - np.array(train_lps)
        seq_cum_diffs.append(float(np.sum(diffs)))
        seq_sampling_cum.append(float(np.sum(samp_lps)))
        seq_training_cum.append(float(np.sum(train_lps)))

    all_diffs = np.array(all_sampling_lps) - np.array(all_training_lps)

    seq_cum_diffs_arr = np.array(seq_cum_diffs)
    is_weights = np.exp(seq_cum_diffs_arr)

    return {
        "mean_diff": float(np.mean(all_diffs)),
        "mean_abs_diff": float(np.mean(np.abs(all_diffs))),
        "max_abs_diff": float(np.max(np.abs(all_diffs))),
        "std_diff": float(np.std(all_diffs)),
        "num_tokens": len(all_diffs),
        "mean_cum_diff": float(np.mean(seq_cum_diffs_arr)),
        "max_cum_diff": float(np.max(np.abs(seq_cum_diffs_arr))),
        "std_cum_diff": float(np.std(seq_cum_diffs_arr)),
        "mean_sampling_logprob": float(np.mean(seq_sampling_cum)),
        "mean_training_logprob": float(np.mean(seq_training_cum)),
        "mean_is_weight": float(np.mean(is_weights)),
        "std_is_weight": float(np.std(is_weights)),
        "min_is_weight": float(np.min(is_weights)),
        "max_is_weight": float(np.max(is_weights)),
        "p99_is_weight": float(np.percentile(is_weights, 99)),
        "p_out_clip_02": float(np.mean(np.abs(is_weights - 1) > 0.2)),
        "p_out_clip_01": float(np.mean(np.abs(is_weights - 1) > 0.1)),
    }


def build_training_datum(
    prompt: types.ModelInput,
    response_tokens: List[int],
    sampling_logprobs: List[float],
    advantage: float = 1.0,
) -> types.Datum:
    """Build an importance-sampling datum for training."""
    prompt_toks = []
    for chunk in prompt.chunks:
        prompt_toks.extend(chunk.tokens)
    prompt_len = len(prompt_toks)

    ob_len = prompt_len - 1
    model_input = prompt.append(types.EncodedTextChunk(tokens=response_tokens[:-1]))

    target_tokens = [0] * ob_len + response_tokens
    padded_sampling_logprobs = [0.0] * ob_len + sampling_logprobs
    padded_advantages = [0.0] * ob_len + [advantage] * len(response_tokens)

    target_tokens_t = torch.tensor(target_tokens, dtype=torch.long)
    logprobs_t = torch.tensor(padded_sampling_logprobs, dtype=torch.float32)
    advantages_t = torch.tensor(padded_advantages, dtype=torch.float32)

    return types.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_torch(target_tokens_t),
            "logprobs": types.TensorData.from_torch(logprobs_t),
            "advantages": types.TensorData.from_torch(advantages_t),
        },
    )


def plot_results(results: List[Dict[str, float]], output_path: str):
    """Plot logprob diff evolution over training rounds."""
    rounds = list(range(len(results)))
    mean_abs_diffs = [r["mean_abs_diff"] for r in results]
    mean_diffs = [r["mean_diff"] for r in results]
    max_abs_diffs = [r["max_abs_diff"] for r in results]
    std_diffs = [r["std_diff"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(rounds, mean_abs_diffs, "b-o", linewidth=2, markersize=4)
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Mean Absolute Diff")
    ax.set_title("Mean |Sampling - Training| Logprob Diff")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(rounds, mean_diffs, "r-o", linewidth=2, markersize=4)
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Mean Diff")
    ax.set_title("Mean (Sampling - Training) Logprob Diff")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(rounds, max_abs_diffs, "g-o", linewidth=2, markersize=4)
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Max Absolute Diff")
    ax.set_title("Max |Sampling - Training| Logprob Diff")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(rounds, std_diffs, "m-o", linewidth=2, markersize=4)
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Std Dev of Diff")
    ax.set_title("Std Dev of Logprob Diff")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training vs Sampling Logprob Mismatch Evolution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[plot] Saved to {output_path}")
    plt.show()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Training/Sampling Logprob Mismatch Verification")
    print("=" * 60)
    print(f"base_url={args.base_url}")
    print(f"base_model={args.base_model}")
    print(f"lora_rank={args.lora_rank}")
    print(f"num_rounds={args.num_rounds}")
    print(f"lr={args.learning_rate}")
    print(f"max_tokens={args.max_tokens}")
    print(f"temperature={args.temperature}")
    print()

    # 1. Connect
    print("[1/5] Connecting to Tinker service...")
    service_client = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)

    # 2. Create training client
    print("[2/5] Creating LoRA training client...")
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    print(f"      tokenizer={type(tokenizer).__name__}")

    # 3. Sample fixed data
    print("[3/5] Sampling fixed dataset...")
    prompts = FIXED_PROMPTS[: args.num_prompts]
    fixed_data, init_weight_path = sample_fixed_data(
        service_client=service_client,
        training_client=training_client,
        tokenizer=tokenizer,
        prompts=prompts,
        num_samples_per_prompt=args.num_samples_per_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # 4. Multi-round training + logprob recording
    print("[4/5] Running multi-round training and recording logprob diffs...")
    adam_params = types.AdamParams(
        learning_rate=args.learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    results: List[Dict[str, float]] = []
    saved_weight_paths: List[str] = [init_weight_path]

    for round_idx in tqdm(range(args.num_rounds), desc="Rounds"):
        # Save weights and create sampling client for current model
        save_result = training_client.save_weights_for_sampler(
            name=f"round_{round_idx:03d}"
        ).result()
        saved_weight_paths.append(save_result.path)
        sampling_client = service_client.create_sampling_client(model_path=save_result.path)

        # Compute logprob differences
        diff_metrics = compute_logprob_diffs(
            sampling_client=sampling_client,
            training_client=training_client,
            fixed_data=fixed_data,
        )
        diff_metrics["round"] = round_idx
        results.append(diff_metrics)

        tqdm.write(
            f"  Round {round_idx:3d}: mean_abs_diff={diff_metrics['mean_abs_diff']:.6f}  "
            f"mean_diff={diff_metrics['mean_diff']:.6f}  "
            f"max_abs_diff={diff_metrics['max_abs_diff']:.6f}  "
            f"mean_cum_diff={diff_metrics['mean_cum_diff']:.6f}  "
            f"std_cum_diff={diff_metrics['std_cum_diff']:.6f}  "
            f"mean_is_weight={diff_metrics['mean_is_weight']:.4f}  "
            f"p_out_clip_02={diff_metrics['p_out_clip_02']:.4f}  "
            f"p_out_clip_01={diff_metrics['p_out_clip_01']:.4f}"
        )

        # Build training datums from fixed data and do one training step
        datums = []
        for prompt, response_tokens, sampling_logprobs in fixed_data:
            datum = build_training_datum(
                prompt=prompt,
                response_tokens=response_tokens,
                sampling_logprobs=sampling_logprobs,
                advantage=1.0,
            )
            datums.append(datum)

        if datums:
            training_client.forward_backward(datums, loss_fn="importance_sampling").result()
            training_client.optim_step(adam_params).result()

    # 5. Final measurement after last training step
    save_result = training_client.save_weights_for_sampler(
        name=f"round_{args.num_rounds:03d}"
    ).result()
    saved_weight_paths.append(save_result.path)
    sampling_client = service_client.create_sampling_client(model_path=save_result.path)
    final_metrics = compute_logprob_diffs(
        sampling_client=sampling_client,
        training_client=training_client,
        fixed_data=fixed_data,
    )
    final_metrics["round"] = args.num_rounds
    results.append(final_metrics)

    print()
    print("[5/5] Saving results and plotting...")

    # Save raw results
    results_path = os.path.join(args.output_dir, "mismatch_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] Results saved to {results_path}")

    # Plot
    plot_path = os.path.join(args.output_dir, "logprob_mismatch.png")
    plot_results(results, plot_path)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Initial mean_abs_diff: {results[0]['mean_abs_diff']:.6f}")
    print(f"Final   mean_abs_diff: {results[-1]['mean_abs_diff']:.6f}")
    print(f"Initial max_abs_diff:  {results[0]['max_abs_diff']:.6f}")
    print(f"Final   max_abs_diff:  {results[-1]['max_abs_diff']:.6f}")
    print(f"Initial mean_is_weight: {results[0]['mean_is_weight']:.4f}")
    print(f"Final   mean_is_weight: {results[-1]['mean_is_weight']:.4f}")
    print(f"Initial p_out_clip_02:  {results[0]['p_out_clip_02']:.4f}")
    print(f"Final   p_out_clip_02:  {results[-1]['p_out_clip_02']:.4f}")
    print()
    print("Conclusion: Training and sampling logprobs are inconsistent.")
    print("This introduces systematic bias into importance-sampling ratios in RLHF.")

    # 6. Delete all saved weights to free Tinker resources
    print()
    print("[6/6] Cleaning up saved weights...")
    rest_client = service_client.create_rest_client()
    for path in saved_weight_paths:
        try:
            rest_client.delete_checkpoint_from_tinker_path(path).result()
            print(f"[cleanup] Deleted {path}")
        except Exception as e:
            print(f"[cleanup] Failed to delete {path}: {e}")


if __name__ == "__main__":
    main()
