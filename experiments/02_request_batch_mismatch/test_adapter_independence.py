"""Test 1: Adapter Independence (Statistical).

Prove that different LoRA adapters co-scheduled in the same batch do NOT
interfere with each other's logprob computation.

Method (statistical approach to handle batched non-determinism):
  1. Generate a canonical token sequence at BS=1 (greedy, deterministic).
  2. Score that sequence at BS=1 (solo) → deterministic reference logprobs.
  3. Repeatedly score in batch with same-adapter fillers (N reps).
  4. Repeatedly score in batch with different-adapter fillers (N reps).
  5. Repeatedly score in batch with mixed-adapter fillers (N reps).
  6. Compute per-run diff vs solo reference for each condition.
  7. Compare distributions: if adapters are independent, all three
     distributions of diffs should be statistically indistinguishable.

Rationale:
  Batched execution introduces small numerical noise (from FlashAttention,
  async scheduling, etc.), but this noise should be INDEPENDENT of which
  adapters the co-batched requests use. If adapter identities don't matter,
  the deviation from solo should have the same distribution regardless of
  filler adapter composition.
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


def score_solo(llm, full_text: str, prompt_len: int, lora_request):
    """Score a fixed sequence at BS=1 (no batching)."""
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    outputs = llm.generate([full_text], sp, lora_request=[lora_request])
    return _extract_prompt_logprobs(outputs[0], prompt_len)


def score_in_batch(
    llm,
    full_text: str,
    prompt_len: int,
    target_lora,
    filler_prompts: list[str],
    filler_loras: list,
    target_pos: int = 0,
):
    """Score a fixed sequence in a batch with fillers at other positions."""
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    batch = list(filler_prompts)
    batch.insert(target_pos, full_text)
    lora_batch = list(filler_loras)
    lora_batch.insert(target_pos, target_lora)

    outputs = llm.generate(batch, sp, lora_request=lora_batch)
    return _extract_prompt_logprobs(outputs[target_pos], prompt_len)


def main():
    parser = get_common_parser("Test 1: Adapter Independence")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-adapters", type=int, default=4)
    parser.add_argument(
        "--num-reps", type=int, default=20, help="Number of repetitions per condition"
    )
    args = parser.parse_args()

    N_REPS = args.num_reps
    BS = args.batch_size

    print("=" * 72)
    print("  TEST 1: ADAPTER INDEPENDENCE (Statistical)")
    print("  Do co-scheduled adapters interfere with each other?")
    print("=" * 72)
    print(f"  model:        {args.model}")
    print(f"  batch_size:   {BS}")
    print(f"  num_adapters: {args.num_adapters}")
    print(f"  num_reps:     {N_REPS}")
    print(f"  max_tokens:   {args.max_tokens}")

    # --- Setup ---
    adapter_dir = os.path.join(args.output_dir, "adapters")
    print("\n[1/6] Creating adapters...")
    adapter_paths = create_adapters(
        args.model,
        adapter_dir,
        args.num_adapters,
        rank=args.rank,
    )

    print("\n[2/6] Initializing vLLM engine (enforce_eager=True)...")
    llm = create_vllm_engine(
        args.model,
        max_loras=args.num_adapters,
        gpu_mem=args.gpu_memory,
        enforce_eager=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = get_prompts(32)
    target_prompt = prompts[0]
    filler_prompts = prompts[1:32]

    lora_reqs = {}
    for i in range(args.num_adapters):
        name = f"adapter_{i}"
        lora_reqs[name] = LoRARequest(name, i + 1, adapter_paths[name])

    print("\n[3/6] Warmup (all adapters)...")
    for _name, lr in lora_reqs.items():
        warmup_engine(llm, filler_prompts, lr)
    torch.cuda.synchronize()
    print("  Done.")

    # --- Generate canonical token sequence ---
    print("\n[4/6] Generating canonical sequence (BS=1, greedy)...")
    gen_sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=1)
    gen_out = llm.generate(
        [target_prompt],
        gen_sp,
        lora_request=[lora_reqs["adapter_0"]],
    )
    canonical_token_ids = list(gen_out[0].outputs[0].token_ids)
    prompt_token_ids = list(gen_out[0].prompt_token_ids)
    prompt_len = len(prompt_token_ids)

    full_token_ids = prompt_token_ids + canonical_token_ids
    full_text = tokenizer.decode(full_token_ids, skip_special_tokens=False)
    print(f"  Prompt: {prompt_len} tokens, Response: {len(canonical_token_ids)} tokens")

    # --- Solo reference (BS=1 scoring) ---
    print("\n[5/6] Solo reference (BS=1 scoring, verify determinism)...")
    solo_lps = score_solo(llm, full_text, prompt_len, lora_reqs["adapter_0"])
    solo_lps2 = score_solo(llm, full_text, prompt_len, lora_reqs["adapter_0"])
    solo_lps3 = score_solo(llm, full_text, prompt_len, lora_reqs["adapter_0"])
    solo_det = solo_lps == solo_lps2 == solo_lps3
    print(f"  Solo deterministic (3 runs identical): {solo_det}")
    print(f"  Response logprob tokens: {len(solo_lps)}")

    results = {
        "config": vars(args),
        "solo_deterministic": solo_det,
    }

    # --- Prepare filler batch ---
    fillers = filler_prompts[: BS - 1]
    if len(fillers) < BS - 1:
        fillers = (fillers * ((BS - 1) // len(fillers) + 1))[: BS - 1]

    target_pos = 0

    # --- Condition A: same adapter for all fillers ---
    print(f"\n[6/6] Statistical comparison ({N_REPS} reps/condition, BS={BS})...")
    print("\n  Condition A: all fillers = adapter_0 (same as target)")
    diffs_same = []
    acc_same = []
    for rep in range(N_REPS):
        filler_loras = [lora_reqs["adapter_0"]] * (BS - 1)
        lps = score_in_batch(
            llm,
            full_text,
            prompt_len,
            lora_reqs["adapter_0"],
            fillers,
            filler_loras,
            target_pos,
        )
        d = compute_diff(solo_lps, lps)
        diffs_same.append(d["mean_abs_diff"])
        acc_same.append(d["accumulated"])
        if rep < 3:
            print(
                f"    Rep {rep}: mean_abs_diff={d['mean_abs_diff']:.2e}, "
                f"max={d['max_abs_diff']:.2e}"
            )

    # --- Condition B: different adapter for fillers ---
    print("\n  Condition B: all fillers = adapter_1 (different from target)")
    diffs_diff = []
    acc_diff = []
    for rep in range(N_REPS):
        filler_loras = [lora_reqs["adapter_1"]] * (BS - 1)
        lps = score_in_batch(
            llm,
            full_text,
            prompt_len,
            lora_reqs["adapter_0"],
            fillers,
            filler_loras,
            target_pos,
        )
        d = compute_diff(solo_lps, lps)
        diffs_diff.append(d["mean_abs_diff"])
        acc_diff.append(d["accumulated"])
        if rep < 3:
            print(
                f"    Rep {rep}: mean_abs_diff={d['mean_abs_diff']:.2e}, "
                f"max={d['max_abs_diff']:.2e}"
            )

    # --- Condition C: mixed adapters for fillers ---
    print("\n  Condition C: fillers = mixed adapters (adapter_1,2,3,...)")
    diffs_mixed = []
    acc_mixed = []
    for rep in range(N_REPS):
        filler_loras = []
        for i in range(BS - 1):
            adapter_idx = (i % (args.num_adapters - 1)) + 1
            filler_loras.append(lora_reqs[f"adapter_{adapter_idx}"])
        lps = score_in_batch(
            llm,
            full_text,
            prompt_len,
            lora_reqs["adapter_0"],
            fillers,
            filler_loras,
            target_pos,
        )
        d = compute_diff(solo_lps, lps)
        diffs_mixed.append(d["mean_abs_diff"])
        acc_mixed.append(d["accumulated"])
        if rep < 3:
            print(
                f"    Rep {rep}: mean_abs_diff={d['mean_abs_diff']:.2e}, "
                f"max={d['max_abs_diff']:.2e}"
            )

    # --- Statistical analysis ---
    arr_same = np.array(diffs_same)
    arr_diff = np.array(diffs_diff)
    arr_mixed = np.array(diffs_mixed)

    print("\n  === Distribution Summary (mean_abs_diff vs solo) ===")
    print(f"    Same adapter fillers:  mean={arr_same.mean():.4e} ± std={arr_same.std():.4e}")
    print(f"    Diff adapter fillers:  mean={arr_diff.mean():.4e} ± std={arr_diff.std():.4e}")
    print(f"    Mixed adapter fillers: mean={arr_mixed.mean():.4e} ± std={arr_mixed.std():.4e}")

    # Ratio comparison (model-free)
    ratio_diff = arr_diff.mean() / max(arr_same.mean(), 1e-15)
    ratio_mixed = arr_mixed.mean() / max(arr_same.mean(), 1e-15)
    print(f"\n    Ratio (diff/same):  {ratio_diff:.4f}")
    print(f"    Ratio (mixed/same): {ratio_mixed:.4f}")

    # Statistical test
    stat_test = {
        "ratio_diff_same": float(ratio_diff),
        "ratio_mixed_same": float(ratio_mixed),
    }
    try:
        from scipy.stats import mannwhitneyu

        _, p_same_diff = mannwhitneyu(diffs_same, diffs_diff, alternative="two-sided")
        _, p_same_mixed = mannwhitneyu(diffs_same, diffs_mixed, alternative="two-sided")
        print("\n  Mann-Whitney U test (H0: distributions are equal):")
        print(
            f"    Same vs Diff:  p={p_same_diff:.4f}  {'(reject H0)' if p_same_diff < 0.05 else '(cannot reject H0)'}"  # noqa: E501
        )
        print(
            f"    Same vs Mixed: p={p_same_mixed:.4f}  {'(reject H0)' if p_same_mixed < 0.05 else '(cannot reject H0)'}"  # noqa: E501
        )
        stat_test["same_vs_diff_p"] = float(p_same_diff)
        stat_test["same_vs_mixed_p"] = float(p_same_mixed)
        independent = p_same_diff > 0.05 and p_same_mixed > 0.05
    except ImportError:
        print("\n  [scipy not available, using ratio heuristic]")
        # Heuristic: means within 2x ≈ no significant difference
        independent = 0.5 < ratio_diff < 2.0 and 0.5 < ratio_mixed < 2.0

    results["statistical_comparison"] = {
        "n_reps": N_REPS,
        "batch_size": BS,
        "target_position": target_pos,
        "diffs_same_adapter": diffs_same,
        "diffs_diff_adapter": diffs_diff,
        "diffs_mixed_adapter": diffs_mixed,
        "acc_same_adapter": acc_same,
        "acc_diff_adapter": acc_diff,
        "acc_mixed_adapter": acc_mixed,
        "stats": {
            "same": {"mean": float(arr_same.mean()), "std": float(arr_same.std())},
            "diff": {"mean": float(arr_diff.mean()), "std": float(arr_diff.std())},
            "mixed": {"mean": float(arr_mixed.mean()), "std": float(arr_mixed.std())},
        },
        "stat_test": stat_test,
    }

    results["conclusion"] = {
        "adapters_independent": bool(independent),
        "summary": (
            "Adapters are INDEPENDENT: co-scheduled adapter identity does not "
            "significantly affect target logprobs (distributions are "
            "statistically indistinguishable from same-adapter baseline)"
            if independent
            else "WARNING: Adapter interference detected — filler adapter "
            "identity significantly affects target logprob deviation"
        ),
    }

    print("\n" + "=" * 72)
    print(f"  CONCLUSION: {results['conclusion']['summary']}")
    print("=" * 72)

    save_results(results, args.output_dir, "test1_adapter_independence.json")


if __name__ == "__main__":
    main()
