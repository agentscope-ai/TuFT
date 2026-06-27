"""
Multi-LoRA vs Sequential LoRA Benchmark (HTTP API Client)
==========================================================
Benchmarks vLLM multi-LoRA serving performance via the OpenAI-compatible API.

Two benchmark modes:
  1. Multi-LoRA Concurrent: All requests (across different LoRA adapters) sent
     concurrently. vLLM batches them internally, leveraging multi-LoRA scheduling.
  2. Sequential per-Adapter: Requests are sent one adapter at a time. Each adapter's
     batch completes before the next begins. Simulates "no multi-LoRA batching".

Usage:
  # First start the vLLM server:
  bash start_vllm_server.sh

  # Then run the benchmark:
  python benchmark_api.py --server-url http://localhost:8000

  # Custom settings:
  python benchmark_api.py --server-url http://localhost:8000 \
      --num-adapters 5 --samples-per-adapter 50 --max-tokens 256
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import aiohttp


# ============================================================================
# Configuration
# ============================================================================
DEFAULT_SERVER_URL = "http://localhost:8000"
NUM_ADAPTERS = 5
SAMPLES_PER_ADAPTER = 50
MAX_OUTPUT_TOKENS = 256
OUTPUT_DIR = "/mnt/nas/hanzhang.yhz/multilora_bench/results"
DATA_DIR = "/mnt/nas/hanzhang.yhz/multilora_bench/data"
BASE_MODEL_NAME = "Qwen3-8B"  # base model name registered in vLLM


# ============================================================================
# Data classes
# ============================================================================
@dataclass
class RequestResult:
    adapter_name: str
    prompt_len: int
    output_len: int
    latency_sec: float
    ttft_sec: float  # time to first token
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    mode: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_wall_time_sec: float = 0.0
    avg_latency_sec: float = 0.0
    p50_latency_sec: float = 0.0
    p90_latency_sec: float = 0.0
    p99_latency_sec: float = 0.0
    avg_ttft_sec: float = 0.0
    throughput_req_per_sec: float = 0.0
    throughput_tok_per_sec: float = 0.0
    per_adapter_stats: dict = field(default_factory=dict)


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
        # Validate
        for idx in range(num_adapters):
            if idx not in prompts_by_adapter:
                break
            if len(prompts_by_adapter[idx]) < samples_per_adapter:
                break
        else:
            # Trim to requested size
            result = {}
            for idx in range(num_adapters):
                result[idx] = prompts_by_adapter[idx][:samples_per_adapter]
            total = sum(len(v) for v in result.values())
            print(
                f"Loaded {total} prompts from local file (source: {data.get('source', 'unknown')})"
            )
            return result
        print("  Local data insufficient, falling back...")

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
                user_msg = next(
                    (m["content"] for m in messages if m["role"] == "user"), messages[0]["content"]
                )
                adapter_prompts.append(user_msg)
            prompts_by_adapter[adapter_idx] = adapter_prompts
        total = sum(len(v) for v in prompts_by_adapter.values())
        print(f"Loaded {total} prompts from HuggingFace dataset")
        return prompts_by_adapter
    except Exception as e:
        print(f"  HuggingFace failed: {e}")

    # Final fallback: generate synthetic prompts inline
    print("Generating synthetic prompts as fallback...")
    import random

    random.seed(42)
    templates = [
        "Explain the concept of {topic} in detail.",
        "Write a comprehensive guide about {topic}.",
        "What are the key principles of {topic}?",
        "Describe how {topic} works and its applications.",
        "Compare and contrast different approaches to {topic}.",
    ]
    topics = [
        "machine learning",
        "web development",
        "cloud computing",
        "data science",
        "cybersecurity",
        "database design",
        "API architecture",
        "distributed systems",
        "DevOps",
        "software testing",
        "agile methodology",
        "design patterns",
    ]
    prompts_by_adapter = {}
    for idx in range(num_adapters):
        prompts = []
        for _ in range(samples_per_adapter):
            prompt = random.choice(templates).format(topic=random.choice(topics))
            prompts.append(prompt)
        prompts_by_adapter[idx] = prompts
    total = sum(len(v) for v in prompts_by_adapter.values())
    print(f"Generated {total} synthetic prompts")
    return prompts_by_adapter


# ============================================================================
# HTTP request helper
# ============================================================================
async def send_completion_request(
    session: aiohttp.ClientSession,
    server_url: str,
    prompt: str,
    model_name: str,
    max_tokens: int,
    request_id: str,
) -> RequestResult:
    """Send a single completion request to the vLLM server and measure latency."""
    url = f"{server_url}/v1/completions"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }

    start_time = time.monotonic()
    ttft = 0.0

    try:
        async with session.post(url, json=payload) as resp:
            ttft = time.monotonic() - start_time
            if resp.status != 200:
                error_text = await resp.text()
                return RequestResult(
                    adapter_name=model_name,
                    prompt_len=len(prompt),
                    output_len=0,
                    latency_sec=time.monotonic() - start_time,
                    ttft_sec=ttft,
                    success=False,
                    error=f"HTTP {resp.status}: {error_text[:200]}",
                )
            data = await resp.json()
            end_time = time.monotonic()

            usage = data.get("usage", {})
            output_text = data["choices"][0]["text"]
            return RequestResult(
                adapter_name=model_name,
                prompt_len=usage.get("prompt_tokens", len(prompt) // 4),
                output_len=usage.get("completion_tokens", len(output_text) // 4),
                latency_sec=end_time - start_time,
                ttft_sec=ttft,
                success=True,
            )
    except Exception as e:
        return RequestResult(
            adapter_name=model_name,
            prompt_len=len(prompt),
            output_len=0,
            latency_sec=time.monotonic() - start_time,
            ttft_sec=ttft,
            success=False,
            error=str(e),
        )


# ============================================================================
# Benchmark helpers
# ============================================================================
def compute_stats(results: list[RequestResult], wall_time: float) -> BenchmarkResult:
    """Compute aggregate statistics from individual request results."""
    successful = [r for r in results if r.success]
    latencies = sorted([r.latency_sec for r in successful])
    ttfts = sorted([r.ttft_sec for r in successful])

    stats = BenchmarkResult(mode="")
    stats.total_requests = len(results)
    stats.successful_requests = len(successful)
    stats.failed_requests = len(results) - len(successful)
    stats.total_input_tokens = sum(r.prompt_len for r in successful)
    stats.total_output_tokens = sum(r.output_len for r in successful)
    stats.total_wall_time_sec = wall_time

    if successful:
        stats.avg_latency_sec = sum(latencies) / len(latencies)
        stats.p50_latency_sec = latencies[len(latencies) // 2]
        stats.p90_latency_sec = latencies[int(len(latencies) * 0.9)]
        stats.p99_latency_sec = latencies[int(len(latencies) * 0.99)]
        stats.avg_ttft_sec = sum(ttfts) / len(ttfts)

    if wall_time > 0:
        stats.throughput_req_per_sec = len(successful) / wall_time
        stats.throughput_tok_per_sec = stats.total_output_tokens / wall_time

    # Per-adapter breakdown
    from collections import defaultdict

    by_adapter = defaultdict(list)
    for r in successful:
        by_adapter[r.adapter_name].append(r)

    for name, adapter_results in by_adapter.items():
        lat = sorted([r.latency_sec for r in adapter_results])
        stats.per_adapter_stats[name] = {
            "count": len(adapter_results),
            "avg_latency": sum(lat) / len(lat),
            "total_output_tokens": sum(r.output_len for r in adapter_results),
        }

    return stats


# ============================================================================
# Benchmark 1: Multi-LoRA Concurrent (all adapters, all requests at once)
# ============================================================================
async def benchmark_multi_lora_concurrent(
    server_url: str,
    prompts_by_adapter: dict[int, list[str]],
    max_tokens: int,
    concurrency: int = 64,
) -> BenchmarkResult:
    """
    Send ALL requests across ALL adapters concurrently.
    vLLM's scheduler batches them together with multi-LoRA support.
    This represents the 'batched multi-LoRA' approach.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Multi-LoRA Concurrent")
    print("  (All adapters' requests sent concurrently → vLLM batches them)")
    print("=" * 70)

    # Build all requests
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:

        async def bounded_request(prompt, model_name, req_id):
            async with semaphore:
                return await send_completion_request(
                    session, server_url, prompt, model_name, max_tokens, req_id
                )

        all_tasks = []
        req_count = 0
        for adapter_idx, prompts in prompts_by_adapter.items():
            model_name = f"adapter_{adapter_idx}"
            for j, prompt in enumerate(prompts):
                req_id = f"multi_{adapter_idx}_{j}"
                all_tasks.append(bounded_request(prompt, model_name, req_id))
                req_count += 1

        print(f"Sending {req_count} requests concurrently (max_concurrency={concurrency})...")
        start_time = time.monotonic()
        results = await asyncio.gather(*all_tasks)
        end_time = time.monotonic()

    wall_time = end_time - start_time
    stats = compute_stats(results, wall_time)
    stats.mode = "multi_lora_concurrent"
    _print_stats(stats)
    return stats


# ============================================================================
# Benchmark 2: Sequential per-Adapter
# ============================================================================
async def benchmark_sequential_per_adapter(
    server_url: str,
    prompts_by_adapter: dict[int, list[str]],
    max_tokens: int,
    concurrency: int = 64,
) -> BenchmarkResult:
    """
    Process each adapter's requests one adapter at a time.
    Within each adapter, requests are sent concurrently for fair comparison.
    But the next adapter starts only after the current one is fully done.
    This simulates: "load adapter A, serve A's requests, then load adapter B, ..."
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Sequential per-Adapter")
    print("  (Process one adapter's requests fully, then move to next adapter)")
    print("=" * 70)

    all_results = []
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)

    start_time = time.monotonic()

    async with aiohttp.ClientSession(connector=connector) as session:

        async def bounded_request(prompt, model_name, req_id):
            async with semaphore:
                return await send_completion_request(
                    session, server_url, prompt, model_name, max_tokens, req_id
                )

        for adapter_idx, prompts in sorted(prompts_by_adapter.items()):
            model_name = f"adapter_{adapter_idx}"
            adapter_tasks = []
            for j, prompt in enumerate(prompts):
                req_id = f"seq_{adapter_idx}_{j}"
                adapter_tasks.append(bounded_request(prompt, model_name, req_id))

            print(f"  Processing {model_name} ({len(prompts)} requests)...")
            adapter_start = time.monotonic()
            adapter_results = await asyncio.gather(*adapter_tasks)
            adapter_time = time.monotonic() - adapter_start

            successful = [r for r in adapter_results if r.success]
            total_out = sum(r.output_len for r in successful)
            print(
                f"    -> {adapter_time:.2f}s, {len(successful)}/{len(adapter_results)} ok, "
                f"{total_out} output tokens, "
                f"{total_out / adapter_time:.1f} tok/s"
            )

            all_results.extend(adapter_results)

    end_time = time.monotonic()
    wall_time = end_time - start_time

    stats = compute_stats(all_results, wall_time)
    stats.mode = "sequential_per_adapter"
    _print_stats(stats)
    return stats


# ============================================================================
# Warmup
# ============================================================================
async def warmup(server_url: str, adapter_name: str = "adapter_0", n: int = 3):
    """Send a few warmup requests to prime the server."""
    print(f"\nWarmup: sending {n} requests to {adapter_name}...")
    connector = aiohttp.TCPConnector(limit=n)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(n):
            tasks.append(
                send_completion_request(
                    session,
                    server_url,
                    "Hello, tell me a short joke.",
                    adapter_name,
                    32,
                    f"warmup_{i}",
                )
            )
        results = await asyncio.gather(*tasks)
        ok = sum(1 for r in results if r.success)
        print(f"  Warmup done: {ok}/{n} successful")
        if ok == 0:
            print("  WARNING: All warmup requests failed! Check server status.")
            for r in results:
                if r.error:
                    print(f"    Error: {r.error}")
            return False
    return True


# ============================================================================
# Display
# ============================================================================
def _print_stats(stats: BenchmarkResult):
    print(f"\n  --- {stats.mode} Results ---")
    print(
        f"  Total requests:        {stats.total_requests} "
        f"({stats.successful_requests} ok, {stats.failed_requests} failed)"
    )
    print(f"  Total input tokens:    {stats.total_input_tokens:,}")
    print(f"  Total output tokens:   {stats.total_output_tokens:,}")
    print(f"  Wall time:             {stats.total_wall_time_sec:.2f}s")
    print(f"  Avg latency:           {stats.avg_latency_sec:.4f}s")
    print(f"  P50 latency:           {stats.p50_latency_sec:.4f}s")
    print(f"  P90 latency:           {stats.p90_latency_sec:.4f}s")
    print(f"  P99 latency:           {stats.p99_latency_sec:.4f}s")
    print(f"  Avg TTFT:              {stats.avg_ttft_sec:.4f}s")
    print(f"  Throughput (req/s):    {stats.throughput_req_per_sec:.2f}")
    print(f"  Throughput (tok/s):    {stats.throughput_tok_per_sec:.1f}")


def print_comparison(batched: BenchmarkResult, sequential: BenchmarkResult):
    """Print a comparison table."""
    print("\n\n")
    print("=" * 78)
    print("                         COMPARISON RESULTS")
    print("=" * 78)
    print(f"{'Metric':<35} {'Multi-LoRA Concurrent':>20} {'Sequential/Adapter':>20}")
    print("-" * 78)

    rows = [
        ("Total requests", f"{batched.total_requests}", f"{sequential.total_requests}"),
        (
            "Successful requests",
            f"{batched.successful_requests}",
            f"{sequential.successful_requests}",
        ),
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
        ("P50 latency (s)", f"{batched.p50_latency_sec:.4f}", f"{sequential.p50_latency_sec:.4f}"),
        ("P90 latency (s)", f"{batched.p90_latency_sec:.4f}", f"{sequential.p90_latency_sec:.4f}"),
        ("P99 latency (s)", f"{batched.p99_latency_sec:.4f}", f"{sequential.p99_latency_sec:.4f}"),
        ("Avg TTFT (s)", f"{batched.avg_ttft_sec:.4f}", f"{sequential.avg_ttft_sec:.4f}"),
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
    ]

    for label, v1, v2 in rows:
        print(f"  {label:<33} {v1:>20} {v2:>20}")

    print("-" * 78)

    # Speedups
    if sequential.total_wall_time_sec > 0 and batched.total_wall_time_sec > 0:
        speedup = sequential.total_wall_time_sec / batched.total_wall_time_sec
        print(f"\n  Wall time speedup (Concurrent / Sequential):  {speedup:.2f}x")

    if sequential.throughput_tok_per_sec > 0 and batched.throughput_tok_per_sec > 0:
        tput_ratio = batched.throughput_tok_per_sec / sequential.throughput_tok_per_sec
        print(f"  Output throughput ratio:                       {tput_ratio:.2f}x")

    if sequential.avg_latency_sec > 0 and batched.avg_latency_sec > 0:
        lat_ratio = batched.avg_latency_sec / sequential.avg_latency_sec
        print(f"  Avg latency ratio (lower is better):           {lat_ratio:.2f}x")

    print()
    if sequential.total_wall_time_sec > 0 and batched.total_wall_time_sec > 0:
        speedup = sequential.total_wall_time_sec / batched.total_wall_time_sec
        if speedup > 1.1:
            print("  ✅ CONCLUSION: Multi-LoRA concurrent batching is FASTER.")
            print("     → You SHOULD batch requests across different LoRA adapters.")
            print(f"     → {speedup:.1f}x throughput improvement over sequential processing.")
        elif speedup > 0.9:
            print("  ➡️  CONCLUSION: Performance is COMPARABLE.")
            print("     → Multi-LoRA batching provides marginal benefit in this setup.")
            print("     → Consider batching if you expect higher request volumes.")
        else:
            print("  ⚠️  CONCLUSION: Sequential processing is faster in this setup.")
            print("     → Multi-LoRA batching overhead exceeds the batching benefit.")
            print("     → Consider sequential processing for your workload.")
    print("=" * 78)


# ============================================================================
# Server health check
# ============================================================================
async def check_server(server_url: str) -> bool:
    """Check if the vLLM server is reachable and list available models."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{server_url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["id"] for m in data.get("data", [])]
                    print(f"Server OK. Available models: {models}")
                    return True
                else:
                    print(f"Server returned HTTP {resp.status}")
                    return False
    except Exception as e:
        print(f"Cannot reach server at {server_url}: {e}")
        return False


# ============================================================================
# Main
# ============================================================================
async def async_main(args):
    print("=" * 78)
    print("  Multi-LoRA Concurrent vs Sequential per-Adapter Benchmark")
    print("  (vLLM OpenAI API Client)")
    print("=" * 78)
    print(f"  Server:              {args.server_url}")
    print(f"  Adapters:            {args.num_adapters}")
    print(f"  Samples/adapter:     {args.samples_per_adapter}")
    print(f"  Total requests:      {args.num_adapters * args.samples_per_adapter}")
    print(f"  Max output tokens:   {args.max_tokens}")
    print(f"  Concurrency:         {args.concurrency}")
    print("=" * 78)

    # Health check
    print("\nChecking server...")
    if not await check_server(args.server_url):
        print("\nERROR: vLLM server is not reachable!")
        print("Start the server first:  bash start_vllm_server.sh")
        return

    # Load dataset
    prompts_by_adapter = load_prompts(args.num_adapters, args.samples_per_adapter)

    # Warmup
    warmup_ok = await warmup(args.server_url, "adapter_0", n=5)
    if not warmup_ok:
        print("WARNING: Warmup failed. Proceeding anyway...")

    # ---- Benchmark 1: Multi-LoRA Concurrent ----
    batched = await benchmark_multi_lora_concurrent(
        args.server_url, prompts_by_adapter, args.max_tokens, args.concurrency
    )

    # Short pause between benchmarks
    print("\nPausing 5s between benchmarks...")
    await asyncio.sleep(5)

    # ---- Benchmark 2: Sequential per-Adapter ----
    sequential = await benchmark_sequential_per_adapter(
        args.server_url, prompts_by_adapter, args.max_tokens, args.concurrency
    )

    # ---- Comparison ----
    print_comparison(batched, sequential)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "config": {
            "server_url": args.server_url,
            "num_adapters": args.num_adapters,
            "samples_per_adapter": args.samples_per_adapter,
            "max_output_tokens": args.max_tokens,
            "concurrency": args.concurrency,
        },
        "multi_lora_concurrent": asdict(batched),
        "sequential_per_adapter": asdict(sequential),
    }
    results_path = os.path.join(args.output_dir, "benchmark_api_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-LoRA vs Sequential Benchmark (vLLM API Client)"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=DEFAULT_SERVER_URL,
        help="vLLM server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--num-adapters",
        type=int,
        default=NUM_ADAPTERS,
        help="Number of LoRA adapters (default: 5)",
    )
    parser.add_argument(
        "--samples-per-adapter",
        type=int,
        default=SAMPLES_PER_ADAPTER,
        help="Samples per adapter (default: 50)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_OUTPUT_TOKENS,
        help="Max output tokens per request (default: 256)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=64, help="Max concurrent requests (default: 64)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for results"
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
