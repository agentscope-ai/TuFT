"""Shared utilities for microbenchmark tests."""

from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# ---------------------------------------------------------------------------
# Prompts (embedded for reproducibility, no external dataset dependency)
# ---------------------------------------------------------------------------

PROMPTS = [
    "What is nutrition science and why is it important?",
    "Compose a professional email about quantum computing.",
    "Categorize the following into relevant groups: web development.",
    "Draft a blog post introduction about augmented reality.",
    "Brainstorm 8 unique approaches to urban planning.",
    "Explain the concept of code refactoring as if talking to a 10-year-old.",
    "What are the main advantages and disadvantages of distributed systems?",
    "Describe how machine learning is used in healthcare applications.",
    "Write a short story about a robot discovering emotions.",
    "Explain the difference between TCP and UDP protocols.",
    "What factors contribute to climate change and how can we address them?",
    "Compare functional programming with object-oriented programming.",
    "Describe the process of photosynthesis in simple terms.",
    "What are the key principles of good software architecture?",
    "Explain how blockchain technology works at a fundamental level.",
    "Describe the history and impact of the printing press.",
    "What are best practices for writing secure web applications?",
    "Explain the concept of supply and demand in economics.",
    "How does the human immune system fight off infections?",
    "Describe the major differences between SQL and NoSQL databases.",
    "What role does artificial intelligence play in modern finance?",
    "Explain the principles behind renewable energy sources.",
    "How do compilers transform source code into machine code?",
    "Describe the cultural significance of the Renaissance period.",
    "What are the ethical considerations in genetic engineering?",
    "Explain how GPS navigation systems determine your location.",
    "What are the key challenges in building scalable web applications?",
    "Describe the relationship between sleep and cognitive performance.",
    "How do neural networks learn to recognize patterns in data?",
    "What are the main theories about the origin of the universe?",
    "Explain the concept of containerization in cloud computing.",
    "Describe how vaccines are developed and tested for safety.",
]


def get_prompts(n: int = 32) -> List[str]:
    """Return n diverse prompts from the embedded pool."""
    return PROMPTS[:n]


# ---------------------------------------------------------------------------
# Adapter creation
# ---------------------------------------------------------------------------


def create_adapters(
    base_model: str,
    output_dir: str,
    num_adapters: int,
    rank: int = 8,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Create LoRA adapters with random weights. Returns {name: path} dict."""
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    os.makedirs(output_dir, exist_ok=True)
    adapter_paths = {}

    for i in range(num_adapters):
        name = f"adapter_{i}"
        adapter_path = os.path.join(output_dir, name)

        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            print(f"  [skip] {name} already exists")
            adapter_paths[name] = adapter_path
            continue

        seed = 42 + i * 100
        torch.manual_seed(seed)
        print(f"  [create] {name} (seed={seed}, rank={rank})")

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )

        peft_model = get_peft_model(model, lora_config)

        for param_name, param in peft_model.named_parameters():
            if "lora_" in param_name and param.requires_grad:
                if "lora_A" in param_name:
                    torch.nn.init.kaiming_uniform_(param, a=5**0.5)
                elif "lora_B" in param_name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)

        peft_model.save_pretrained(adapter_path)
        print(f"  [saved] {adapter_path}")

        del model, peft_model
        gc.collect()

        adapter_paths[name] = adapter_path

    return adapter_paths


# ---------------------------------------------------------------------------
# vLLM engine utilities
# ---------------------------------------------------------------------------


def create_vllm_engine(
    model: str,
    max_loras: int = 4,
    max_lora_rank: int = 64,
    gpu_mem: float = 0.8,
    enforce_eager: bool = False,
) -> LLM:
    """Create a vLLM engine configured for LoRA experiments."""
    return LLM(
        model=model,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        max_loras=max_loras,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem,
        enable_prefix_caching=False,
        enforce_eager=enforce_eager,
    )


def warmup_engine(
    llm: LLM,
    prompts: List[str],
    lora_request: LoRARequest,
    n_calls: int = 10,
    max_tokens: int = 8,
) -> None:
    """Warm up CUDA graphs at target batch size and BS=1."""
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=1)
    batch = prompts[:8] if len(prompts) >= 8 else prompts + [prompts[0]] * (8 - len(prompts))
    lora_batch = [lora_request] * len(batch)

    for _ in range(n_calls):
        llm.generate(batch, sp, lora_request=lora_batch)

    # Also warmup BS=1
    for _ in range(5):
        llm.generate([prompts[0]], sp, lora_request=[lora_request])

    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Logprob extraction
# ---------------------------------------------------------------------------


def extract_logprobs(output) -> List[float]:
    """Extract per-token logprobs from a vLLM RequestOutput."""
    logprobs = []
    out = output.outputs[0]
    if out.logprobs:
        for lp_dict in out.logprobs:
            if lp_dict:
                lp_val = list(lp_dict.values())[0]
                if hasattr(lp_val, "logprob"):
                    logprobs.append(float(lp_val.logprob))
                else:
                    logprobs.append(float(lp_val))
    return logprobs


# ---------------------------------------------------------------------------
# PyTorch/peft logprob computation
# ---------------------------------------------------------------------------


def compute_pytorch_logprobs(
    model_path: str,
    adapter_path: str,
    tokenizer,
    token_sequences: List[List[int]],
    batch_size: int = 1,
) -> List[List[float]]:
    """Compute logprobs via PyTorch/peft forward pass.

    Args:
        model_path: Base model path
        adapter_path: LoRA adapter path
        tokenizer: Tokenizer instance
        token_sequences: List of full token sequences (prompt + response)
            Each is [tok0, tok1, ..., tokN]. Logprobs computed for tok1..tokN.
        batch_size: How many sequences to forward together.

    Returns:
        List of logprob lists (one per sequence, excluding first token).
    """
    import torch.nn.functional as F

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    peft_model.eval()

    all_logprobs = []

    for batch_start in range(0, len(token_sequences), batch_size):
        batch_seqs = token_sequences[batch_start : batch_start + batch_size]

        # Left-pad to same length
        max_len = max(len(s) for s in batch_seqs)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        input_ids_list = []
        attention_mask_list = []
        for seq in batch_seqs:
            pad_len = max_len - len(seq)
            input_ids_list.append([pad_token_id] * pad_len + seq)
            attention_mask_list.append([0] * pad_len + [1] * len(seq))

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device="cuda")
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long, device="cuda")

        with torch.no_grad():
            outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch, seq_len, vocab_size]
            log_probs = F.log_softmax(logits, dim=-1)

        # Extract logprob of actual next token at each position
        for i, seq in enumerate(batch_seqs):
            pad_len = max_len - len(seq)
            seq_logprobs = []
            for pos in range(len(seq) - 1):
                actual_pos = pad_len + pos
                next_token = seq[pos + 1]
                lp = float(log_probs[i, actual_pos, next_token])
                seq_logprobs.append(lp)
            all_logprobs.append(seq_logprobs)

    del peft_model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return all_logprobs


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------


def compute_diff(lps1: List[float], lps2: List[float]) -> Dict[str, Any]:
    """Compute difference metrics between two logprob lists."""
    arr1 = np.array(lps1, dtype=np.float64)
    arr2 = np.array(lps2, dtype=np.float64)
    min_len = min(len(arr1), len(arr2))
    arr1 = arr1[:min_len]
    arr2 = arr2[:min_len]

    diff = np.abs(arr1 - arr2)
    return {
        "mean_abs_diff": float(np.mean(diff)),
        "max_abs_diff": float(np.max(diff)) if len(diff) > 0 else 0.0,
        "accumulated": float(np.sum(diff)),
        "all_zero": bool(np.all(diff == 0)),
        "num_tokens": int(min_len),
    }


# ---------------------------------------------------------------------------
# CLI and I/O
# ---------------------------------------------------------------------------


def get_common_parser(description: str) -> argparse.ArgumentParser:
    """Create parser with shared arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        default="/mnt/cpfs/shared/qwen/Qwen3-4B",
        help="Base model path",
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Output directory for results JSON",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max tokens to generate",
    )
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--gpu-memory", type=float, default=0.8, help="GPU memory utilization")
    return parser


def save_results(data: Any, output_dir: str, filename: str) -> str:
    """Save results to JSON, handling numpy types."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

    print(f"\n  Results saved to: {path}")
    return path
