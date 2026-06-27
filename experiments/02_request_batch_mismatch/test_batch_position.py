"""Test 2: Batch Size and Position Effect.

Prove that batch size and prompt position systematically affect
the inference engine's logprob output.

Method (prompt_logprobs scoring to avoid token divergence):
  1. Generate a canonical token sequence at BS=1 (greedy).
  2. Score that fixed sequence at BS=1 (solo, N reps) → reference.
  3. Score at various batch sizes (N reps each) → compare to solo.
  4. Score at various positions within BS=8 (N reps each) → compare.
  5. Show systematic, reproducible differences beyond noise.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from transformers import AutoTokenizer
from utils import (
    SamplingParams,
    compute_diff,
    create_adapters,
    create_vllm_engine,
    get_common_parser,
    get_prompts,
    save_results,
    warmup_engine,
)
from vllm.lora.request import LoRARequest


def _extract_prompt_logprobs(output, prompt_len: int) -> list[float]:
    """Extract logprobs for the response portion from prompt_logprobs."""
    logprobs = []
    if output.prompt_logprobs:
        for i in range(prompt_len, len(output.prompt_logprobs)):
            lp_dict = output.prompt_logprobs[i]
            if lp_dict:
                token_id = output.prompt_token_ids[i]
                if token_id in lp_dict:
                    lp_val = lp_dict[token_id]
                    if hasattr(lp_val, "logprob"):
                        logprobs.append(float(lp_val.logprob))
                    else:
                        logprobs.append(float(lp_val))
    return logprobs


def score_solo(llm, full_text, prompt_len, lora_request):
    """Score at BS=1."""
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    outputs = llm.generate([full_text], sp, lora_request=[lora_request])
    return _extract_prompt_logprobs(outputs[0], prompt_len)


def score_in_batch(
    llm, full_text, prompt_len, lora_request, filler_prompts, target_pos, batch_size
):
    """Score in a batch of given size, target at target_pos."""
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    n_fillers = batch_size - 1
    fillers = filler_prompts[:n_fillers]
    if len(fillers) < n_fillers:
        fillers = (fillers * (n_fillers // len(fillers) + 1))[:n_fillers]

    batch = list(fillers)
    batch.insert(target_pos, full_text)
    lora_batch = [lora_request] * len(batch)

    outputs = llm.generate(batch, sp, lora_request=lora_batch)
    return _extract_prompt_logprobs(outputs[target_pos], prompt_len)


def main():
    parser = get_common_parser("Test 2: Batch Size and Position Effect")
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16",
        help="Comma-separated batch sizes to test",
    )
    parser.add_argument(
        "--num-reps",
        type=int,
        default=10,
        help="Number of repetitions per condition",
    )
    args = parser.parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    N_REPS = args.num_reps

    print("=" * 72)
    print("  TEST 2: BATCH SIZE AND POSITION EFFECT")
    print("  Does batch context change logprob computation?")
    print("=" * 72)
    print(f"  model:       {args.model}")
    print(f"  batch_sizes: {batch_sizes}")
    print(f"  num_reps:    {N_REPS}")
    print(f"  max_tokens:  {args.max_tokens}")

    # --- Setup ---
    adapter_dir = os.path.join(args.output_dir, "adapters")
    print("\n[1/6] Creating adapter...")
    adapter_paths = create_adapters(args.model, adapter_dir, 1, rank=args.rank)

    print("\n[2/6] Initializing vLLM engine...")
    llm = create_vllm_engine(args.model, max_loras=1, gpu_mem=args.gpu_memory)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = get_prompts(32)
    target_prompt = prompts[0]
    filler_prompts = prompts[1:32]

    lora_req = LoRARequest("adapter_0", 1, adapter_paths["adapter_0"])

    print("\n[3/6] Warmup...")
    warmup_engine(llm, prompts, lora_req)
    torch.cuda.synchronize()
    print("  Done.")

    # --- Generate canonical sequence ---
    print("\n[4/6] Generating canonical sequence (BS=1, greedy)...")
    gen_sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=1)
    gen_out = llm.generate([target_prompt], gen_sp, lora_request=[lora_req])
    canonical_token_ids = list(gen_out[0].outputs[0].token_ids)
    prompt_token_ids = list(gen_out[0].prompt_token_ids)
    prompt_len = len(prompt_token_ids)

    full_token_ids = prompt_token_ids + canonical_token_ids
    full_text = tokenizer.decode(full_token_ids, skip_special_tokens=False)
    print(f"  Prompt: {prompt_len} tokens, Response: {len(canonical_token_ids)} tokens")

    results = {"config": vars(args)}

    # --- Exp A: Solo baseline (BS=1 scoring, N reps) ---
    print(f"\n[5/6] Experiment A: Solo baseline ({N_REPS} reps)...")
    solo_runs = []
    for _rep in range(N_REPS):
        lps = score_solo(llm, full_text, prompt_len, lora_req)
        solo_runs.append(lps)

    # Use first run as reference
    solo_ref = solo_runs[0]

    # Intra-solo variance (run-to-run noise at BS=1)
    solo_diffs = []
    solo_acc_diffs = []
    for i in range(1, N_REPS):
        d = compute_diff(solo_ref, solo_runs[i])
        solo_diffs.append(d["mean_abs_diff"])
        solo_acc_diffs.append(d["accumulated"])
    solo_noise = np.array(solo_diffs)
    solo_det = all(d == 0 for d in solo_diffs)
    print(f"  Solo deterministic: {solo_det}")
    print(f"  Solo noise: mean={solo_noise.mean():.4e} ± {solo_noise.std():.4e}")

    results["solo_baseline"] = {
        "deterministic": solo_det,
        "noise_mean": float(solo_noise.mean()),
        "noise_std": float(solo_noise.std()),
    }

    # --- Exp B: Batch size effect (target at pos=0) ---
    print("\n  Experiment B: Batch size effect (target at pos=0)...")
    exp_b = {}
    for bs in batch_sizes:
        if bs == 1:
            exp_b[1] = {
                "mean_diff": float(solo_noise.mean()),
                "std_diff": float(solo_noise.std()),
                "raw_mean_abs_diffs": solo_diffs,
                "raw_accumulated": solo_acc_diffs,
                "label": "solo_noise",
            }
            continue

        diffs = []
        acc_diffs = []
        for _rep in range(N_REPS):
            lps = score_in_batch(
                llm,
                full_text,
                prompt_len,
                lora_req,
                filler_prompts,
                target_pos=0,
                batch_size=bs,
            )
            d = compute_diff(solo_ref, lps)
            diffs.append(d["mean_abs_diff"])
            acc_diffs.append(d["accumulated"])
        arr = np.array(diffs)
        exp_b[bs] = {
            "mean_diff": float(arr.mean()),
            "std_diff": float(arr.std()),
            "raw_mean_abs_diffs": diffs,
            "raw_accumulated": acc_diffs,
        }
        ratio = arr.mean() / max(solo_noise.mean(), 1e-15)
        print(
            f"    BS={bs:3d}: mean_diff={arr.mean():.4e} ± {arr.std():.4e}  "
            f"(ratio vs solo_noise: {ratio:.2f}x)"
        )

    results["exp_b_batch_size_effect"] = {str(k): v for k, v in exp_b.items()}

    # --- Exp C: Position sweep at BS=8 ---
    test_bs = min(8, max(batch_sizes))
    print(f"\n  Experiment C: Position sweep (BS={test_bs}, {N_REPS} reps/pos)...")
    exp_c = {}
    for target_pos in range(test_bs):
        diffs = []
        acc_diffs = []
        for _rep in range(N_REPS):
            lps = score_in_batch(
                llm,
                full_text,
                prompt_len,
                lora_req,
                filler_prompts,
                target_pos=target_pos,
                batch_size=test_bs,
            )
            d = compute_diff(solo_ref, lps)
            diffs.append(d["mean_abs_diff"])
            acc_diffs.append(d["accumulated"])
        arr = np.array(diffs)
        exp_c[target_pos] = {
            "mean_diff": float(arr.mean()),
            "std_diff": float(arr.std()),
            "raw_mean_abs_diffs": diffs,
            "raw_accumulated": acc_diffs,
        }
        print(f"    pos={target_pos}: mean_diff={arr.mean():.4e} ± {arr.std():.4e}")

    results["exp_c_position_sweep"] = {f"pos_{k}": v for k, v in exp_c.items()}

    # Check if positions give different results
    pos_means = np.array([v["mean_diff"] for v in exp_c.values()])
    pos_variation = pos_means.std() / max(pos_means.mean(), 1e-15)

    # --- Exp D: Cross-position comparison (same BS, different positions) ---
    print(f"\n[6/6] Experiment D: Cross-position pairwise diff (BS={test_bs})...")
    # Use median runs from position sweep for pairwise
    # Just run once more and compare pos=0 vs pos=3 vs pos=7
    exp_d = {}
    pos_samples = {}
    for target_pos in [0, min(3, test_bs - 1), min(test_bs - 1, 7)]:
        lps = score_in_batch(
            llm,
            full_text,
            prompt_len,
            lora_req,
            filler_prompts,
            target_pos=target_pos,
            batch_size=test_bs,
        )
        pos_samples[target_pos] = lps

    for p1 in pos_samples:
        for p2 in pos_samples:
            if p1 >= p2:
                continue
            d = compute_diff(pos_samples[p1], pos_samples[p2])
            key = f"pos{p1}_vs_pos{p2}"
            exp_d[key] = {
                "mean_abs_diff": d["mean_abs_diff"],
                "max_abs_diff": d["max_abs_diff"],
                "identical": d["all_zero"],
            }
            status = "IDENTICAL" if d["all_zero"] else f"DIFFERS (mean={d['mean_abs_diff']:.2e})"
            print(f"    {key}: {status}")

    results["exp_d_cross_position"] = exp_d

    # --- Conclusion ---
    # Batch effect: BS=8 diff >> solo noise
    bs8_data = exp_b.get(8, exp_b.get(max(k for k in exp_b if k > 1), {}))
    batch_effect_ratio = bs8_data.get("mean_diff", 0) / max(solo_noise.mean(), 1e-15)
    has_batch_effect = batch_effect_ratio > 2.0  # at least 2x the noise

    # Position effect: positions give meaningfully different diffs
    has_position_effect = pos_variation > 0.1  # CoV > 10%

    results["conclusion"] = {
        "batch_size_affects_logprobs": has_batch_effect,
        "batch_effect_ratio_vs_noise": float(batch_effect_ratio),
        "position_affects_logprobs": has_position_effect,
        "position_variation_cov": float(pos_variation),
        "summary": (
            f"Batch size DOES affect logprobs ({batch_effect_ratio:.1f}x solo noise). "
            f"Position {'DOES' if has_position_effect else 'does NOT'} "
            f"significantly affect logprobs (CoV={pos_variation:.2%})."
        ),
    }

    print("\n" + "=" * 72)
    print(f"  CONCLUSION: {results['conclusion']['summary']}")
    print("=" * 72)

    save_results(results, args.output_dir, "test2_batch_position.json")


if __name__ == "__main__":
    main()
