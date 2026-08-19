"""Measure training-vs-sampling logprob mismatch using vLLM-generated tokens.

Unlike the random-token probe, this script:
  1. Has vLLM sample real responses for fixed GSM8K prompts
  2. Computes logprobs for the SAME full sequence via both:
     - Training API (HuggingFace forward pass)
     - Sampling API (vLLM compute_logprobs)
  3. Compares them position-by-position

This reflects the actual mismatch seen during GRPO/PPO training.

Usage:
  TINKER_API_KEY=... python tests/tuft_mismatch_probe_real.py \
      --base-url http://127.0.0.1:10610 \
      --model Qwen/Qwen3.5-4B \
      --tokenizer /mnt/cpfs/shiweijie/hf_cache/qwen3.5-4b \
      --tag old_tuft \
      --output /tmp/tuft_mismatch_real_old_tuft.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import tinker
import torch
from tinker import types
from transformers import AutoTokenizer


PROMPTS = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as "
        "many clips in May. How many clips did Natalia sell altogether in April and May?"
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes "
        "of babysitting. How much did she earn?"
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has only half "
        "of the money she needs. Her parents decided to give her $15 for that purpose, "
        "and her grandparents twice as much as her parents. How much more money does "
        "Betty need to buy the wallet?"
    ),
    (
        "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages "
        "and today, she read twice as many. If she wants to read half of the remaining "
        "pages tomorrow, how many pages should she read?"
    ),
    (
        "James writes a 3-page letter to 2 different friends twice a week. How many "
        "pages does he write a year?"
    ),
    (
        "Mark has a garden with flowers. He planted plants of three different colors "
        "in it. Ten of them are yellow, and there are 80% more of those in purple. "
        "There are only 25% as many green flowers as there are yellow and purple "
        "flowers. How many flowers does Mark have in his garden?"
    ),
    (
        "Roberta started making pizza at 5:00 pm. She made 3 pizzas and each pizza "
        "took 15 minutes to make. What time did she finish making all the pizzas?"
    ),
    (
        "A store has 40 apples. They sell 15 in the morning and get a delivery of 20 "
        "more. Then they sell 10 more in the afternoon. How many apples do they have "
        "left?"
    ),
    (
        "Tom has 5 boxes with 8 marbles each. He gives 2 boxes to his friend. How "
        "many marbles does Tom have left?"
    ),
    (
        "A train travels at 60 km/h for 2 hours, then at 80 km/h for 1.5 hours. "
        "What is the total distance traveled?"
    ),
    (
        "Sarah has $50. She buys 3 books at $8 each and 2 pens at $3 each. How much "
        "money does she have left?"
    ),
    (
        "A farmer has 120 chickens. He sells 1/4 of them on Monday and 1/3 of the "
        "remaining on Tuesday. How many chickens does he have left?"
    ),
]


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default=os.getenv("TINKER_BASE_URL", "http://127.0.0.1:10610")
    )
    parser.add_argument("--api-key", default=os.getenv("TINKER_API_KEY", "tml-local-dev-key"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--tokenizer", default=None, help="Tokenizer path/name; defaults to --model"
    )
    parser.add_argument("--tag", default="tuft")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", default="/tmp/tuft_mismatch_real.json")
    return parser.parse_args()


def finite_values(values: List[Optional[float]]) -> List[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "abs_mean": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "abs_mean": float(tensor.abs().mean().item()),
    }


def mismatch_stats(diffs: List[float]) -> Dict[str, float]:
    base = stats(diffs)
    if diffs:
        tensor = torch.tensor(diffs, dtype=torch.float64)
        base["rmse"] = float(tensor.pow(2).mean().sqrt().item())
        base["max_abs"] = float(tensor.abs().max().item())
        # Relative error: |diff| / |sampling_logprob|
        base["median_abs"] = float(tensor.abs().median().item())
    else:
        base["rmse"] = 0.0
        base["max_abs"] = 0.0
        base["median_abs"] = 0.0
    return base


def main() -> None:
    args = make_args()
    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, fast=True)

    service_client = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)

    # Create training client (LoRA with rank, freshly initialized)
    print(
        f"[{args.tag}] creating training client model={args.model} "
        f"rank={args.rank} seed={args.seed}"
    )
    training_client = service_client.create_lora_training_client(
        base_model=args.model,
        rank=args.rank,
        seed=args.seed,
        train_mlp=True,
        train_attn=True,
        train_unembed=False,
    )

    # Create sampling client (base model, no LoRA)
    print(f"[{args.tag}] creating sampling client model={args.model}")
    sampling_client = service_client.create_sampling_client(base_model=args.model)

    # Step 1: Generate real responses via vLLM sampling
    print(
        f"[{args.tag}] generating {args.num_samples} responses via vLLM "
        f"(max_tokens={args.max_response_tokens}, temp={args.temperature})"
    )
    samples: List[Dict[str, Any]] = []
    for i in range(args.num_samples):
        prompt = PROMPTS[i % len(PROMPTS)]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        t0 = time.time()
        sample_result = sampling_client.sample(
            prompt=types.ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=args.max_response_tokens,
                temperature=args.temperature,
                seed=args.seed + i,
            ),
        ).result()
        gen_time = time.time() - t0

        seq = sample_result.sequences[0]
        response_ids = list(seq.tokens)
        # logprobs from generation (per-token, for the response only)
        gen_logprobs = list(seq.logprobs) if seq.logprobs else None

        full_ids = prompt_ids + response_ids
        samples.append(
            {
                "sample_id": i,
                "prompt_text": prompt,
                "prompt_length": len(prompt_ids),
                "response_length": len(response_ids),
                "full_ids": full_ids,
                "gen_logprobs": gen_logprobs,
                "gen_time_sec": gen_time,
            }
        )
        print(
            f"  sample {i}: prompt_len={len(prompt_ids)}, "
            f"resp_len={len(response_ids)}, gen_time={gen_time:.2f}s"
        )

    # Step 2: Get training logprobs via forward pass (all samples in one call)
    print(f"[{args.tag}] computing training logprobs via forward pass...")
    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(sample["full_ids"][:-1]),
            loss_fn_inputs={
                "target_tokens": sample["full_ids"][1:],
            },
        )
        for sample in samples
    ]

    t0 = time.time()
    forward_result = training_client.forward(data, "cross_entropy").result()
    training_time = time.time() - t0
    print(f"[{args.tag}] training forward done in {training_time:.2f}s")

    training_logprobs_by_sample: List[List[float]] = []
    for out in forward_result.loss_fn_outputs:
        logprob = torch.tensor(out["logprobs"].data, dtype=torch.float64)
        if out["logprobs"].shape is not None:
            logprob = logprob.reshape(out["logprobs"].shape)
        training_logprobs_by_sample.append(logprob.flatten().tolist())

    # Step 3: Get sampling logprobs via compute_logprobs for each full sequence
    print(f"[{args.tag}] computing sampling logprobs via compute_logprobs...")
    per_sample = []
    all_response_diffs: List[float] = []
    all_prompt_diffs: List[float] = []
    all_response_diffs_vs_gen: List[float] = []

    for sample_idx, sample in enumerate(samples):
        full_ids = sample["full_ids"]
        prompt_len = sample["prompt_length"]

        t1 = time.time()
        sampling_logprobs = sampling_client.compute_logprobs(
            types.ModelInput.from_ints(full_ids)
        ).result()
        sample_time = time.time() - t1

        # Align: training[j] = logprob of full_ids[j+1] given full_ids[:j+1]
        # sampling[k] = logprob of full_ids[k] given full_ids[:k]
        # So training[j] should match sampling[j+1]
        aligned_training: List[Optional[float]] = [None] * len(full_ids)
        for j, value in enumerate(training_logprobs_by_sample[sample_idx]):
            aligned_training[j + 1] = value

        response_diffs: List[float] = []
        prompt_diffs: List[float] = []
        response_diffs_vs_gen: List[float] = []
        token_rows = []

        for pos in range(len(full_ids)):
            train_lp = aligned_training[pos]
            sample_lp = sampling_logprobs[pos] if pos < len(sampling_logprobs) else None
            diff = None
            if (
                train_lp is not None
                and sample_lp is not None
                and math.isfinite(train_lp)
                and math.isfinite(float(sample_lp))
            ):
                diff = float(train_lp) - float(sample_lp)
                if pos >= prompt_len:
                    response_diffs.append(diff)
                else:
                    prompt_diffs.append(diff)

            # Compare training logprob vs generation logprob (for response tokens)
            gen_diff = None
            if (
                pos >= prompt_len
                and train_lp is not None
                and sample["gen_logprobs"] is not None
                and (pos - prompt_len) < len(sample["gen_logprobs"])
            ):
                gen_lp = sample["gen_logprobs"][pos - prompt_len]
                if gen_lp is not None and math.isfinite(train_lp) and math.isfinite(float(gen_lp)):
                    gen_diff = float(train_lp) - float(gen_lp)
                    response_diffs_vs_gen.append(gen_diff)

            token_rows.append(
                {
                    "pos": pos,
                    "token_id": int(full_ids[pos]),
                    "is_response": pos >= prompt_len,
                    "training_logprob": None if train_lp is None else round(float(train_lp), 6),
                    "sampling_logprob": None if sample_lp is None else round(float(sample_lp), 6),
                    "diff_train_minus_sample": None if diff is None else round(diff, 6),
                    "diff_train_minus_gen": None if gen_diff is None else round(gen_diff, 6),
                }
            )

        all_response_diffs.extend(response_diffs)
        all_prompt_diffs.extend(prompt_diffs)
        all_response_diffs_vs_gen.extend(response_diffs_vs_gen)

        # Keep worst tokens and boundary tokens
        token_rows_sorted = sorted(
            (row for row in token_rows if row["diff_train_minus_sample"] is not None),
            key=lambda row: abs(row["diff_train_minus_sample"]),
            reverse=True,
        )
        worst_tokens = token_rows_sorted[:10]
        edge_tokens = token_rows[:3] + token_rows[prompt_len - 1 : prompt_len + 5] + token_rows[-3:]

        per_sample.append(
            {
                "sample_id": sample["sample_id"],
                "prompt_text": sample["prompt_text"][:80],
                "prompt_length": prompt_len,
                "response_length": sample["response_length"],
                "full_length": len(full_ids),
                "gen_time_sec": sample["gen_time_sec"],
                "sampling_compute_time_sec": sample_time,
                "response_mismatch_train_vs_sample": mismatch_stats(response_diffs),
                "prompt_mismatch_train_vs_sample": mismatch_stats(prompt_diffs),
                "response_mismatch_train_vs_gen": mismatch_stats(response_diffs_vs_gen),
                "worst_tokens": worst_tokens,
                "edge_tokens": edge_tokens,
            }
        )

    result = {
        "tag": args.tag,
        "base_url": args.base_url,
        "model": args.model,
        "tokenizer": tokenizer_name,
        "rank": args.rank,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "max_response_tokens": args.max_response_tokens,
        "temperature": args.temperature,
        "method": "vllm_real_generation",
        "training_forward_time_sec": training_time,
        "overall_response_mismatch_train_vs_sample": mismatch_stats(all_response_diffs),
        "overall_prompt_mismatch_train_vs_sample": mismatch_stats(all_prompt_diffs),
        "overall_response_mismatch_train_vs_gen": mismatch_stats(all_response_diffs_vs_gen),
        "per_sample": per_sample,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[{args.tag}] wrote {args.output}")
    print(
        json.dumps(
            {
                "tag": result["tag"],
                "method": result["method"],
                "overall_response_mismatch_train_vs_sample": result[
                    "overall_response_mismatch_train_vs_sample"
                ],
                "overall_prompt_mismatch_train_vs_sample": result[
                    "overall_prompt_mismatch_train_vs_sample"
                ],
                "overall_response_mismatch_train_vs_gen": result[
                    "overall_response_mismatch_train_vs_gen"
                ],
                "training_forward_time_sec": result["training_forward_time_sec"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
