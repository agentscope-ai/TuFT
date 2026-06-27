"""
Multi-LoRA vs Sequential LoRA Inference Benchmark
===================================================
Compares two approaches:
  1. Multi-LoRA Batched: All requests (across different LoRA adapters) submitted in one batch
  2. Sequential per-Adapter: Requests processed one adapter at a time, serially

Base model: Qwen/Qwen3-8B (local)
LoRA adapters: 5 mock adapters (random weights, rank=16)
Dataset: HuggingFaceH4/no_robots
Metrics: Throughput (tokens/sec), Average Latency (sec/request)
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# ============================================================================
# Configuration
# ============================================================================
BASE_MODEL_PATH = "/mnt/cpfs/luyi/models/Qwen3-8B"
LORA_BASE_DIR = "/mnt/nas/hanzhang.yhz/multilora_bench/mock_loras"
DATA_DIR = "/mnt/nas/hanzhang.yhz/multilora_bench/data"
NUM_ADAPTERS = 5
SAMPLES_PER_ADAPTER = 50
MAX_OUTPUT_TOKENS = 256
OUTPUT_DIR = "/mnt/nas/hanzhang.yhz/multilora_bench/results"


# ============================================================================
# Data classes
# ============================================================================
@dataclass
class BenchmarkResult:
    mode: str  # "multi_lora_batched" or "sequential_per_adapter"
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_wall_time_sec: float = 0.0
    avg_latency_sec: float = 0.0  # average per-request latency
    throughput_req_per_sec: float = 0.0
    throughput_tok_per_sec: float = 0.0  # output tokens per second
    input_tok_per_sec: float = 0.0  # input throughput
    per_adapter_times: dict = field(default_factory=dict)


# ============================================================================
# Dataset loading
# ============================================================================
def load_prompts(num_adapters: int, samples_per_adapter: int) -> dict[int, list[str]]:
    """
    Load prompts from local JSON file (prepared by prepare_data.py).
    Falls back to HuggingFace dataset, then to synthetic generation.
    """
    # Try local prepared data first
    local_path = os.path.join(DATA_DIR, "prompts.json")
    if os.path.exists(local_path):
        print(f"Loading prompts from {local_path}...")
        with open(local_path) as f:
            data = json.load(f)
        prompts_by_adapter = {int(k): v for k, v in data["prompts_by_adapter"].items()}
        result = {}
        for idx in range(num_adapters):
            if idx in prompts_by_adapter and len(prompts_by_adapter[idx]) >= samples_per_adapter:
                result[idx] = prompts_by_adapter[idx][:samples_per_adapter]
            else:
                break
        else:
            total = sum(len(v) for v in result.values())
            print(f"Loaded {total} prompts (source: {data.get('source', 'unknown')})")
            return result

    # Try HuggingFace dataset
    try:
        print("Loading HuggingFaceH4/no_robots dataset...")
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/no_robots", split="train")
        total_needed = num_adapters * samples_per_adapter
        if len(ds) < total_needed:
            raise ValueError(f"Need {total_needed}, got {len(ds)}")
        prompts_by_adapter = {}
        for adapter_idx in range(num_adapters):
            start = adapter_idx * samples_per_adapter
            end = start + samples_per_adapter
            adapter_prompts = []
            for i in range(start, end):
                messages = ds[i]["messages"]
                user_msg = None
                for msg in messages:
                    if msg["role"] == "user":
                        user_msg = msg["content"]
                        break
                adapter_prompts.append(user_msg or messages[0]["content"])
            prompts_by_adapter[adapter_idx] = adapter_prompts
        total = sum(len(v) for v in prompts_by_adapter.values())
        print(f"Loaded {total} prompts from HuggingFace")
        return prompts_by_adapter
    except Exception as e:
        print(f"  HuggingFace failed: {e}")

    # Fallback: synthetic
    print("Generating synthetic prompts...")
    import random

    random.seed(42)
    templates = [
        "Explain the concept of {t} in detail.",
        "Write a comprehensive guide about {t}.",
        "What are the key principles of {t}?",
        "Describe how {t} works and its applications.",
        "Compare and contrast different approaches to {t}.",
    ]
    topics = [
        "machine learning",
        "web development",
        "cloud computing",
        "data science",
        "cybersecurity",
        "API design",
        "distributed systems",
        "DevOps",
        "design patterns",
    ]
    prompts_by_adapter = {}
    for idx in range(num_adapters):
        prompts_by_adapter[idx] = [
            random.choice(templates).format(t=random.choice(topics))
            for _ in range(samples_per_adapter)
        ]
    total = sum(len(v) for v in prompts_by_adapter.values())
    print(f"Generated {total} synthetic prompts")
    return prompts_by_adapter


# ============================================================================
# Engine initialization
# ============================================================================
def create_engine(enable_lora: bool = True, tp_size: int = 1) -> LLM:
    """Create a vLLM engine with optional LoRA support."""
    print(f"\nInitializing vLLM engine (enable_lora={enable_lora}, tp={tp_size})...")
    start = time.time()

    llm = LLM(
        model=BASE_MODEL_PATH,
        enable_lora=enable_lora,
        max_lora_rank=64,
        max_loras=NUM_ADAPTERS + 1,  # allow all adapters simultaneously
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )

    elapsed = time.time() - start
    print(f"Engine initialized in {elapsed:.1f}s")
    return llm


# ============================================================================
# Benchmark 1: Multi-LoRA Batched
# ============================================================================
def benchmark_multi_lora_batched(
    llm: LLM,
    prompts_by_adapter: dict[int, list[str]],
    sampling_params: SamplingParams,
) -> BenchmarkResult:
    """
    Submit ALL requests across ALL adapters in a single batch.
    vLLM schedules them together with multi-LoRA support.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Multi-LoRA Batched (all adapters, all requests in one batch)")
    print("=" * 70)

    result = BenchmarkResult(mode="multi_lora_batched")

    # Build combined request list
    all_prompts = []
    all_lora_requests = []

    for adapter_idx, prompts in prompts_by_adapter.items():
        adapter_path = os.path.join(LORA_BASE_DIR, f"adapter_{adapter_idx}")
        lora_req = LoRARequest(
            lora_name=f"adapter_{adapter_idx}",
            lora_int_id=adapter_idx + 1,  # must be > 0
            lora_path=adapter_path,
        )
        for prompt in prompts:
            all_prompts.append(prompt)
            all_lora_requests.append(lora_req)

    result.total_requests = len(all_prompts)
    print(f"Submitting {result.total_requests} requests in one batch...")

    # Run inference (timed)
    torch.cuda.synchronize()
    start_time = time.time()

    outputs = llm.generate(
        prompts=all_prompts,
        sampling_params=sampling_params,
        lora_request=all_lora_requests,
    )

    torch.cuda.synchronize()
    end_time = time.time()

    result.total_wall_time_sec = end_time - start_time

    # Compute token counts
    for output in outputs:
        result.total_input_tokens += len(output.prompt_token_ids)
        result.total_output_tokens += len(output.outputs[0].token_ids)

    # Compute metrics
    result.avg_latency_sec = result.total_wall_time_sec / result.total_requests
    result.throughput_req_per_sec = result.total_requests / result.total_wall_time_sec
    result.throughput_tok_per_sec = result.total_output_tokens / result.total_wall_time_sec
    result.input_tok_per_sec = result.total_input_tokens / result.total_wall_time_sec

    _print_result(result)
    return result


# ============================================================================
# Benchmark 2: Sequential per-Adapter
# ============================================================================
def benchmark_sequential_per_adapter(
    llm: LLM,
    prompts_by_adapter: dict[int, list[str]],
    sampling_params: SamplingParams,
) -> BenchmarkResult:
    """
    Process each adapter's requests separately in sequence.
    This simulates the pattern: load adapter A → process all A's requests →
    then adapter B → process all B's requests → etc.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Sequential per-Adapter (one adapter at a time)")
    print("=" * 70)

    result = BenchmarkResult(mode="sequential_per_adapter")
    result.total_requests = sum(len(p) for p in prompts_by_adapter.values())

    torch.cuda.synchronize()
    total_start = time.time()

    for adapter_idx, prompts in prompts_by_adapter.items():
        adapter_path = os.path.join(LORA_BASE_DIR, f"adapter_{adapter_idx}")
        lora_req = LoRARequest(
            lora_name=f"adapter_{adapter_idx}",
            lora_int_id=adapter_idx + 1,
            lora_path=adapter_path,
        )

        print(f"  Processing adapter_{adapter_idx} ({len(prompts)} requests)...")
        torch.cuda.synchronize()
        adapter_start = time.time()

        outputs = llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            lora_request=lora_req,
        )

        torch.cuda.synchronize()
        adapter_end = time.time()
        adapter_time = adapter_end - adapter_start

        # Count tokens for this adapter
        adapter_input_tokens = 0
        adapter_output_tokens = 0
        for output in outputs:
            adapter_input_tokens += len(output.prompt_token_ids)
            adapter_output_tokens += len(output.outputs[0].token_ids)

        result.total_input_tokens += adapter_input_tokens
        result.total_output_tokens += adapter_output_tokens
        result.per_adapter_times[f"adapter_{adapter_idx}"] = {
            "wall_time_sec": adapter_time,
            "input_tokens": adapter_input_tokens,
            "output_tokens": adapter_output_tokens,
            "throughput_tok_per_sec": adapter_output_tokens / adapter_time
            if adapter_time > 0
            else 0,
        }

        print(
            f"    -> {adapter_time:.2f}s, "
            f"{adapter_input_tokens} in / {adapter_output_tokens} out tokens, "
            f"{adapter_output_tokens / adapter_time:.1f} tok/s"
        )

    torch.cuda.synchronize()
    total_end = time.time()

    result.total_wall_time_sec = total_end - total_start
    result.avg_latency_sec = result.total_wall_time_sec / result.total_requests
    result.throughput_req_per_sec = result.total_requests / result.total_wall_time_sec
    result.throughput_tok_per_sec = result.total_output_tokens / result.total_wall_time_sec
    result.input_tok_per_sec = result.total_input_tokens / result.total_wall_time_sec

    _print_result(result)
    return result


# ============================================================================
# Display and analysis
# ============================================================================
def _print_result(result: BenchmarkResult):
    print(f"\n  --- {result.mode} Results ---")
    print(f"  Total requests:       {result.total_requests}")
    print(f"  Total input tokens:   {result.total_input_tokens:,}")
    print(f"  Total output tokens:  {result.total_output_tokens:,}")
    print(f"  Wall time:            {result.total_wall_time_sec:.2f}s")
    print(f"  Avg latency:          {result.avg_latency_sec:.4f}s / request")
    print(f"  Throughput (req/s):   {result.throughput_req_per_sec:.2f}")
    print(f"  Throughput (tok/s):   {result.throughput_tok_per_sec:.1f} output tokens/s")
    print(f"  Input throughput:     {result.input_tok_per_sec:.1f} input tokens/s")


def print_comparison(batched: BenchmarkResult, sequential: BenchmarkResult):
    """Print a comparison table of the two approaches."""
    print("\n")
    print("=" * 70)
    print("                    COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Metric':<35} {'Multi-LoRA Batched':>17} {'Sequential':>17}")
    print("-" * 70)

    rows = [
        ("Total requests", f"{batched.total_requests}", f"{sequential.total_requests}"),
        (
            "Total input tokens",
            f"{batched.total_input_tokens:,}",
            f"{sequential.total_input_tokens:,}",
        ),
        (
            "Total output tokens",
            f"{batched.total_output_tokens:,}",
            f"{sequential.total_output_tokens:,}",
        ),
        (
            "Wall time (s)",
            f"{batched.total_wall_time_sec:.2f}",
            f"{sequential.total_wall_time_sec:.2f}",
        ),
        (
            "Avg latency (s/req)",
            f"{batched.avg_latency_sec:.4f}",
            f"{sequential.avg_latency_sec:.4f}",
        ),
        (
            "Throughput (req/s)",
            f"{batched.throughput_req_per_sec:.2f}",
            f"{sequential.throughput_req_per_sec:.2f}",
        ),
        (
            "Output throughput (tok/s)",
            f"{batched.throughput_tok_per_sec:.1f}",
            f"{sequential.throughput_tok_per_sec:.1f}",
        ),
        (
            "Input throughput (tok/s)",
            f"{batched.input_tok_per_sec:.1f}",
            f"{sequential.input_tok_per_sec:.1f}",
        ),
    ]

    for label, v1, v2 in rows:
        print(f"  {label:<33} {v1:>17} {v2:>17}")

    print("-" * 70)

    # Speedup / comparison
    if sequential.total_wall_time_sec > 0:
        speedup = sequential.total_wall_time_sec / batched.total_wall_time_sec
        print(f"\n  Wall time speedup (Batched vs Sequential): {speedup:.2f}x")

    if sequential.throughput_tok_per_sec > 0:
        tput_ratio = batched.throughput_tok_per_sec / sequential.throughput_tok_per_sec
        print(f"  Output throughput ratio:                    {tput_ratio:.2f}x")

    if sequential.avg_latency_sec > 0:
        lat_ratio = batched.avg_latency_sec / sequential.avg_latency_sec
        print(f"  Avg latency ratio (lower is better):        {lat_ratio:.2f}x")

    print()
    if speedup > 1.0:
        print("  ✅ Multi-LoRA batched is FASTER overall (higher throughput).")
        print("     → Batch requests for multi-LoRA inference for better throughput.")
    else:
        print("  ⚠️ Sequential processing is comparable or faster.")
        print("     → Multi-LoRA batching may not provide significant benefit.")

    print("=" * 70)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Multi-LoRA vs Sequential Benchmark")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_OUTPUT_TOKENS, help="Max output tokens"
    )
    parser.add_argument(
        "--num-adapters", type=int, default=NUM_ADAPTERS, help="Number of LoRA adapters"
    )
    parser.add_argument(
        "--samples-per-adapter", type=int, default=SAMPLES_PER_ADAPTER, help="Samples per adapter"
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR, help="Directory to save results"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Multi-LoRA Batched vs Sequential per-Adapter Benchmark")
    print("=" * 70)
    print(f"  Base model:          {BASE_MODEL_PATH}")
    print(f"  LoRA adapters:       {args.num_adapters} (mock, rank=16)")
    print(f"  Samples per adapter: {args.samples_per_adapter}")
    print(f"  Total requests:      {args.num_adapters * args.samples_per_adapter}")
    print(f"  Max output tokens:   {args.max_tokens}")
    print(f"  Tensor parallel:     {args.tp_size}")
    print("=" * 70)

    # Load data
    prompts_by_adapter = load_prompts(args.num_adapters, args.samples_per_adapter)

    # Create engine (shared across both benchmarks to ensure fair comparison)
    llm = create_engine(enable_lora=True, tp_size=args.tp_size)

    sampling_params = SamplingParams(
        temperature=0.0,  # greedy for determinism
        max_tokens=args.max_tokens,
    )

    # Warmup: small batch to warm up the engine
    print("\nWarmup run (5 requests)...")
    warmup_lora = LoRARequest(
        lora_name="adapter_0",
        lora_int_id=1,
        lora_path=os.path.join(LORA_BASE_DIR, "adapter_0"),
    )
    llm.generate(
        prompts=prompts_by_adapter[0][:5],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=16),
        lora_request=warmup_lora,
    )
    print("Warmup complete.\n")

    # ---- Benchmark 1: Multi-LoRA Batched ----
    batched_result = benchmark_multi_lora_batched(llm, prompts_by_adapter, sampling_params)

    # ---- Benchmark 2: Sequential per-Adapter ----
    sequential_result = benchmark_sequential_per_adapter(llm, prompts_by_adapter, sampling_params)

    # ---- Comparison ----
    print_comparison(batched_result, sequential_result)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "config": {
            "base_model": BASE_MODEL_PATH,
            "num_adapters": args.num_adapters,
            "samples_per_adapter": args.samples_per_adapter,
            "max_output_tokens": args.max_tokens,
            "tp_size": args.tp_size,
            "lora_rank": 16,
            "lora_alpha": 32,
        },
        "multi_lora_batched": asdict(batched_result),
        "sequential_per_adapter": asdict(sequential_result),
    }
    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
