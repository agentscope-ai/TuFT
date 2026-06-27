"""Test 3: Framework Mismatch — Sampling (vLLM) vs Training (PyTorch/peft).

Follows the real RL training flow:
  1. Sampling:  vLLM self-generates responses and collects **generation logprobs**
               (autoregressive decode, one token at a time)
  2. Training:  PyTorch/peft forward pass on (prompt + generated response)
               to compute logprobs (single prefill, all tokens at once)
  3. Compare:   generation logprobs vs forward logprobs = IS-ratio distortion

The mismatch between step 1 and step 2 directly translates to
importance-sampling ratio error in on-policy RL algorithms (PPO, GRPO, etc.).
"""

from __future__ import annotations

import gc
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer
from utils import (
    compute_diff,
    compute_pytorch_logprobs,
    create_adapters,
    create_vllm_engine,
    extract_logprobs,
    get_common_parser,
    get_prompts,
    save_results,
    warmup_engine,
)
from vllm import SamplingParams
from vllm.lora.request import LoRARequest


def main():
    parser = get_common_parser("Test 3: Framework Mismatch (vLLM vs PyTorch/peft)")
    parser.add_argument(
        "--vllm-batch-sizes",
        default="1,2,4,8",
        help="Comma-separated vLLM generation batch sizes",
    )
    parser.add_argument(
        "--pytorch-batch-sizes",
        default="1,2,4,8",
        help="Comma-separated PyTorch/peft forward-pass batch sizes",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=8,
        help="Number of sequences to generate and compare",
    )
    args = parser.parse_args()
    vllm_bs_list = [int(x) for x in args.vllm_batch_sizes.split(",")]
    pytorch_bs_list = [int(x) for x in args.pytorch_batch_sizes.split(",")]

    print("=" * 72)
    print("  TEST 3: FRAMEWORK MISMATCH — Sampling vs Training")
    print("  vLLM generation logprobs vs PyTorch/peft forward logprobs")
    print("=" * 72)
    print(f"  model:              {args.model}")
    print(f"  vllm_batch_sizes:   {vllm_bs_list}")
    print(f"  pytorch_batch_sizes: {pytorch_bs_list}")
    print(f"  num_sequences:      {args.num_sequences}")
    print(f"  max_tokens:         {args.max_tokens}")

    # --- Setup ---
    adapter_dir = os.path.join(args.output_dir, "adapters")
    print("\n[1/6] Creating adapter...")
    adapter_paths = create_adapters(args.model, adapter_dir, 1, rank=args.rank)
    adapter_path = adapter_paths["adapter_0"]

    print("\n[2/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = get_prompts(args.num_sequences)

    # ===================================================================
    # PHASE 1: vLLM self-generation (= RL sampling phase)
    #   Autoregressive decode -> tokens + generation logprobs
    # ===================================================================
    print("\n[3/6] Phase 1: vLLM self-generation (RL sampling)...")
    llm = create_vllm_engine(args.model, max_loras=1, gpu_mem=args.gpu_memory)
    lora_req = LoRARequest("adapter_0", 1, adapter_path)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=1)

    print("  Warmup...")
    warmup_engine(llm, prompts, lora_req)
    torch.cuda.synchronize()

    # Generate at each vLLM batch size, collecting tokens + generation logprobs
    vllm_gen = {}  # {bs: {sequences, prompt_lens, gen_logprobs}}

    for bs in vllm_bs_list:
        print(f"\n  Generating at vLLM BS={bs}...")
        sequences = []
        prompt_lens = []
        gen_logprobs = []
        n_seqs = len(prompts)

        if bs == 1:
            for prompt in prompts:
                out = llm.generate([prompt], sp, lora_request=[lora_req])
                prompt_ids = list(out[0].prompt_token_ids)
                response_ids = list(out[0].outputs[0].token_ids)
                sequences.append(prompt_ids + response_ids)
                prompt_lens.append(len(prompt_ids))
                gen_logprobs.append(extract_logprobs(out[0]))
        else:
            for batch_start in range(0, n_seqs, bs):
                batch_end = min(batch_start + bs, n_seqs)
                batch_prompts = list(prompts[batch_start:batch_end])
                actual_count = len(batch_prompts)

                # Pad to exact BS with filler prompts (from outside test set)
                if actual_count < bs:
                    filler_pool = get_prompts(32)[n_seqs:]
                    pad_count = bs - actual_count
                    fillers = (filler_pool * (pad_count // len(filler_pool) + 1))[:pad_count]
                    batch_prompts.extend(fillers)

                lora_batch = [lora_req] * len(batch_prompts)
                outs = llm.generate(batch_prompts, sp, lora_request=lora_batch)

                for j in range(actual_count):
                    prompt_ids = list(outs[j].prompt_token_ids)
                    response_ids = list(outs[j].outputs[0].token_ids)
                    sequences.append(prompt_ids + response_ids)
                    prompt_lens.append(len(prompt_ids))
                    gen_logprobs.append(extract_logprobs(outs[j]))

        vllm_gen[bs] = {
            "sequences": sequences,
            "prompt_lens": prompt_lens,
            "gen_logprobs": gen_logprobs,
        }
        avg_resp = np.mean([len(lp) for lp in gen_logprobs])
        print(f"    -> {len(sequences)} seqs, avg {avg_resp:.0f} response tokens each")

    # Check token agreement across vLLM batch sizes
    print("\n  Token agreement across vLLM batch sizes:")
    base_bs = vllm_bs_list[0]
    token_agreement = {}
    for bs in vllm_bs_list[1:]:
        agree_count = 0
        for i in range(args.num_sequences):
            base_resp = vllm_gen[base_bs]["sequences"][i][vllm_gen[base_bs]["prompt_lens"][i] :]
            other_resp = vllm_gen[bs]["sequences"][i][vllm_gen[bs]["prompt_lens"][i] :]
            if base_resp == other_resp:
                agree_count += 1
        token_agreement[f"bs{base_bs}_vs_bs{bs}"] = f"{agree_count}/{args.num_sequences}"
        print(f"    BS={base_bs} vs BS={bs}: {agree_count}/{args.num_sequences} sequences match")

    # Free vLLM engine
    print("\n  Freeing vLLM engine...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)

    # ===================================================================
    # PHASE 2: PyTorch/peft forward pass (= RL training phase)
    #   Single forward on full (prompt + response) -> training logprobs
    # ===================================================================
    print("\n[4/6] Phase 2: PyTorch/peft forward pass (RL training)...")

    # For each PyTorch BS, score sequences from ALL vLLM BS values in one
    # model load to avoid repeated loading.
    # pytorch_scores[vbs][pbs] = [response_logprobs_per_seq]
    pytorch_scores = {vbs: {} for vbs in vllm_bs_list}

    for pbs in pytorch_bs_list:
        print(f"\n  Computing PyTorch logprobs at BS={pbs}...")

        # Collect all sequences across all vLLM BS values
        all_sequences = []
        all_prompt_lens = []
        boundaries = {}  # {vbs: (start_idx, count)}
        offset = 0
        for vbs in vllm_bs_list:
            n = len(vllm_gen[vbs]["sequences"])
            boundaries[vbs] = (offset, n)
            all_sequences.extend(vllm_gen[vbs]["sequences"])
            all_prompt_lens.extend(vllm_gen[vbs]["prompt_lens"])
            offset += n

        # Single model load, score all sequences at this PyTorch BS
        all_lps = compute_pytorch_logprobs(
            model_path=args.model,
            adapter_path=adapter_path,
            tokenizer=tokenizer,
            token_sequences=all_sequences,
            batch_size=pbs,
        )

        # Split results back and extract response-only logprobs
        for vbs in vllm_bs_list:
            start, count = boundaries[vbs]
            response_lps = []
            for i in range(count):
                idx = start + i
                prompt_len = all_prompt_lens[idx]
                response_len = len(all_sequences[idx]) - prompt_len
                # PyTorch lps[j] = logprob(token_{j+1} | tokens_0..j)
                # Response tokens start at position prompt_len
                # -> lps[prompt_len-1 : prompt_len-1+response_len]
                pt_start = prompt_len - 1
                pt_end = pt_start + response_len
                response_lps.append(all_lps[idx][pt_start:pt_end])

            pytorch_scores[vbs][pbs] = response_lps
            avg_tok = np.mean([len(lp) for lp in response_lps])
            print(
                f"    vLLM_BS={vbs} seqs at PyTorch_BS={pbs}: "
                f"{count} seqs, avg {avg_tok:.0f} tokens"
            )

    # ===================================================================
    # PHASE 3: Cross-comparison
    # ===================================================================
    print("\n[5/6] Phase 3: Comparison...")

    results = {
        "config": vars(args),
        "token_agreement": token_agreement,
    }

    # --- Cross-framework: vLLM generation logprobs vs PyTorch forward ---
    print("\n  Cross-framework (vLLM generation vs PyTorch forward):")
    cross_framework = {}
    per_prompt_data = {}  # {key: [{mean_abs_diff, accumulated, num_tokens}, ...]}
    for vbs in vllm_bs_list:
        for pbs in pytorch_bs_list:
            diffs = []
            prompt_details = []
            for i in range(args.num_sequences):
                v_lps = vllm_gen[vbs]["gen_logprobs"][i]
                p_lps = pytorch_scores[vbs][pbs][i]
                d = compute_diff(v_lps, p_lps)
                diffs.append(d)
                prompt_details.append(
                    {
                        "mean_abs_diff": d["mean_abs_diff"],
                        "accumulated": d["accumulated"],
                        "num_tokens": d["num_tokens"],
                    }
                )

            key = f"vllm_bs{vbs}_vs_pytorch_bs{pbs}"
            avg_mean = float(np.mean([d["mean_abs_diff"] for d in diffs]))
            avg_max = float(np.max([d["max_abs_diff"] for d in diffs]))
            avg_acc = float(np.mean([d["accumulated"] for d in diffs]))
            cross_framework[key] = {
                "avg_mean_abs_diff": avg_mean,
                "max_abs_diff": avg_max,
                "avg_accumulated": avg_acc,
            }
            per_prompt_data[key] = prompt_details
            print(
                f"    vLLM(gen BS={vbs}) vs PyTorch(fwd BS={pbs}): "
                f"mean={avg_mean:.2e}, max={avg_max:.2e}, acc={avg_acc:.4f}"
            )

    results["cross_framework"] = cross_framework
    results["per_prompt"] = per_prompt_data

    # --- Within-vLLM: generation logprobs across BS ---
    # Only compare sequences where tokens agree across BS
    print("\n  Within vLLM (generation logprobs across BS):")
    vllm_internal = {}
    for bs in vllm_bs_list[1:]:
        diffs = []
        n_compared = 0
        for i in range(args.num_sequences):
            base_resp = vllm_gen[base_bs]["sequences"][i][vllm_gen[base_bs]["prompt_lens"][i] :]
            other_resp = vllm_gen[bs]["sequences"][i][vllm_gen[bs]["prompt_lens"][i] :]
            if base_resp == other_resp:
                d = compute_diff(
                    vllm_gen[base_bs]["gen_logprobs"][i],
                    vllm_gen[bs]["gen_logprobs"][i],
                )
                diffs.append(d)
                n_compared += 1

        if diffs:
            avg_diff = float(np.mean([d["mean_abs_diff"] for d in diffs]))
            all_zero = all(d["all_zero"] for d in diffs)
        else:
            avg_diff = float("nan")
            all_zero = False

        vllm_internal[f"bs{base_bs}_vs_bs{bs}"] = {
            "avg_mean_abs_diff": avg_diff,
            "n_compared": n_compared,
            "all_zero": all_zero,
        }
        if diffs:
            status = (
                "IDENTICAL"
                if all_zero
                else f"DIFFERS (avg_mean={avg_diff:.2e}, "
                f"{n_compared}/{args.num_sequences} token-matched)"
            )
        else:
            status = f"NO TOKEN MATCH (0/{args.num_sequences})"
        print(f"    vLLM BS={base_bs} vs BS={bs}: {status}")

    results["vllm_internal"] = vllm_internal

    # --- Within-PyTorch: forward logprobs across BS (same sequences) ---
    print("\n  Within PyTorch (forward logprobs across BS, using vLLM BS=1 sequences):")
    pytorch_internal = {}
    ref_vbs = vllm_bs_list[0]
    base_pbs = pytorch_bs_list[0]
    for pbs in pytorch_bs_list[1:]:
        diffs = []
        for i in range(args.num_sequences):
            d = compute_diff(
                pytorch_scores[ref_vbs][base_pbs][i],
                pytorch_scores[ref_vbs][pbs][i],
            )
            diffs.append(d)
        avg_diff = float(np.mean([d["mean_abs_diff"] for d in diffs]))
        all_zero = all(d["all_zero"] for d in diffs)
        pytorch_internal[f"bs{base_pbs}_vs_bs{pbs}"] = {
            "avg_mean_abs_diff": avg_diff,
            "all_zero": all_zero,
        }
        status = "IDENTICAL" if all_zero else f"DIFFERS (avg_mean={avg_diff:.2e})"
        print(f"    PyTorch BS={base_pbs} vs BS={pbs}: {status}")

    results["pytorch_internal"] = pytorch_internal

    # --- Conclusion ---
    print("\n[6/6] Summary...")

    gap_values = [v["avg_mean_abs_diff"] for v in cross_framework.values()]
    framework_gap = float(np.mean(gap_values))
    min_gap = float(np.min(gap_values))
    max_gap = float(np.max(gap_values))

    avg_acc = float(np.mean([v["avg_accumulated"] for v in cross_framework.values()]))

    vllm_has_batch_effect = any(not v.get("all_zero", True) for v in vllm_internal.values())
    pytorch_transparent = (
        all(v.get("all_zero", False) for v in pytorch_internal.values())
        if pytorch_internal
        else True
    )

    summary = (
        f"Framework mismatch: avg {framework_gap:.2e}/token "
        f"(range: {min_gap:.2e} ~ {max_gap:.2e}), "
        f"avg accumulated diff per seq: {avg_acc:.4f}. "
        f"PyTorch batching {'is' if pytorch_transparent else 'is NOT'} transparent. "
        f"vLLM generation logprobs "
        f"{'vary' if vllm_has_batch_effect else 'do not vary'} with BS."
    )

    results["conclusion"] = {
        "avg_framework_gap_per_token": framework_gap,
        "min_framework_gap_per_token": min_gap,
        "max_framework_gap_per_token": max_gap,
        "avg_accumulated_per_seq": avg_acc,
        "pytorch_batching_transparent": pytorch_transparent,
        "vllm_has_batch_effect": vllm_has_batch_effect,
        "summary": summary,
    }

    print("\n" + "=" * 72)
    print(f"  CONCLUSION: {summary}")
    print("=" * 72)

    save_results(results, args.output_dir, "test3_framework_mismatch.json")


if __name__ == "__main__":
    main()
