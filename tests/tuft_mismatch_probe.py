"""Measure training-vs-sampling logprob mismatch against a running TuFT service.

For each fixed input sequence, this script records:
  - training_logprobs: from TrainingClient.forward(cross_entropy)
  - sampling_logprobs: from SamplingClient.compute_logprobs(full_sequence)

Run it against old TuFT and new TuFT with the same --seed/--model/--tokenizer,
then compare the two JSON outputs.

Usage:
  TINKER_API_KEY=... python tests/tuft_mismatch_probe.py \
      --base-url http://127.0.0.1:10610 \
      --model Qwen/Qwen3.5-4B \
      --tokenizer /mnt/cpfs/shared/checkpoints/qwen/qwen3.5/Qwen3.5-4B/ \
      --tag old_tuft \
      --output /tmp/tuft_train_sample_mismatch_old.json
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
    parser.add_argument("--max-prompt-tokens", type=int, default=160)
    parser.add_argument("--response-tokens", type=int, default=64)
    parser.add_argument("--output", default="/tmp/tuft_train_sample_mismatch.json")
    return parser.parse_args()


def build_fixed_sequences(
    tokenizer: Any,
    num_samples: int,
    max_prompt_tokens: int,
    response_tokens: int,
    seed: int,
) -> List[Dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    samples: List[Dict[str, Any]] = []
    for i in range(num_samples):
        prompt = PROMPTS[i % len(PROMPTS)]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[:max_prompt_tokens]

        vocab = tokenizer.vocab_size
        low = 1000
        high = max(low + 1, vocab - 1000)
        resp_ids = torch.randint(
            low=low,
            high=high,
            size=(response_tokens,),
            generator=generator,
            dtype=torch.long,
        ).tolist()

        # Full sequence whose token i is predicted from prefix [:i].
        # compute_logprobs(full_sequence) returns logprob for each token given
        # previous tokens; training forward uses model_input=full[:-1],
        # target_tokens=full[1:], so target position j corresponds to full[j+1].
        full_ids = prompt_ids + resp_ids
        samples.append(
            {
                "sample_id": i,
                "prompt_text": prompt,
                "prompt_length": len(prompt_ids),
                "response_length": len(resp_ids),
                "full_ids": full_ids,
            }
        )
    return samples


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
    else:
        base["rmse"] = 0.0
        base["max_abs"] = 0.0
    return base


def main() -> None:
    args = make_args()
    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, fast=True)

    service_client = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)

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

    print(f"[{args.tag}] creating sampling client model={args.model}")
    sampling_client = service_client.create_sampling_client(base_model=args.model)

    samples = build_fixed_sequences(
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        max_prompt_tokens=args.max_prompt_tokens,
        response_tokens=args.response_tokens,
        seed=args.seed,
    )

    # Training logprobs for all samples in one forward call.
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

    training_logprobs_by_sample: List[List[float]] = []
    for out in forward_result.loss_fn_outputs:
        logprob = torch.tensor(out["logprobs"].data, dtype=torch.float64)
        if out["logprobs"].shape is not None:
            logprob = logprob.reshape(out["logprobs"].shape)
        training_logprobs_by_sample.append(logprob.flatten().tolist())

    per_sample = []
    all_response_diffs: List[float] = []
    all_prompt_diffs: List[float] = []

    for sample_idx, sample in enumerate(samples):
        full_ids = sample["full_ids"]
        prompt_len = sample["prompt_length"]

        # Sampling logprobs for the same full sequence.
        t1 = time.time()
        sampling_logprobs = sampling_client.compute_logprobs(
            types.ModelInput.from_ints(full_ids)
        ).result()
        sample_time = time.time() - t1

        # Align positions:
        # training_logprobs[j] = logprob of full_ids[j+1] given full_ids[:j+1]
        # sampling_logprobs[k] = logprob of full_ids[k] given full_ids[:k]
        # Therefore training[j] should match sampling[j+1].
        aligned_training: List[Optional[float]] = [None] * len(full_ids)
        for j, value in enumerate(training_logprobs_by_sample[sample_idx]):
            aligned_training[j + 1] = value

        response_diffs: List[float] = []
        prompt_diffs: List[float] = []
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

            # Keep output compact: only first/last few tokens plus worst mismatches.
            token_rows.append(
                {
                    "pos": pos,
                    "token_id": int(full_ids[pos]),
                    "is_response": pos >= prompt_len,
                    "training_logprob": None if train_lp is None else float(train_lp),
                    "sampling_logprob": None if sample_lp is None else float(sample_lp),
                    "diff_train_minus_sample": diff,
                }
            )

        all_response_diffs.extend(response_diffs)
        all_prompt_diffs.extend(prompt_diffs)

        token_rows_sorted = sorted(
            (row for row in token_rows if row["diff_train_minus_sample"] is not None),
            key=lambda row: abs(row["diff_train_minus_sample"]),
            reverse=True,
        )
        worst_tokens = token_rows_sorted[:10]
        edge_tokens = token_rows[:4] + token_rows[prompt_len - 2 : prompt_len + 4] + token_rows[-4:]

        per_sample.append(
            {
                "sample_id": sample["sample_id"],
                "prompt_text": sample["prompt_text"],
                "prompt_length": prompt_len,
                "response_length": sample["response_length"],
                "full_length": len(full_ids),
                "sampling_time_sec": sample_time,
                "response_mismatch": mismatch_stats(response_diffs),
                "prompt_mismatch": mismatch_stats(prompt_diffs),
                "worst_tokens": worst_tokens,
                "edge_tokens": edge_tokens,
                # Full arrays can be large; include only if needed via --full-arrays.
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
        "max_prompt_tokens": args.max_prompt_tokens,
        "response_tokens": args.response_tokens,
        "training_forward_time_sec": training_time,
        "overall_response_mismatch": mismatch_stats(all_response_diffs),
        "overall_prompt_mismatch": mismatch_stats(all_prompt_diffs),
        "per_sample": per_sample,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{args.tag}] wrote {args.output}")
    print(
        json.dumps(
            {
                "tag": result["tag"],
                "overall_response_mismatch": result["overall_response_mismatch"],
                "overall_prompt_mismatch": result["overall_prompt_mismatch"],
                "training_forward_time_sec": result["training_forward_time_sec"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
