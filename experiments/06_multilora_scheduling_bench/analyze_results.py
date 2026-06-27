"""
Analyze and visualize benchmark results.
Reads JSON output from benchmark_api.py or benchmark.py and prints a summary.

Usage:
  python analyze_results.py results/benchmark_api_results.json
  python analyze_results.py results/benchmark_results.json
"""

import json
import os
import sys


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_header(title: str, width: int = 78):
    print("=" * width)
    padding = (width - len(title)) // 2
    print(" " * padding + title)
    print("=" * width)


def format_duration(sec: float) -> str:
    if sec < 1:
        return f"{sec * 1000:.1f}ms"
    elif sec < 60:
        return f"{sec:.2f}s"
    else:
        m, s = divmod(sec, 60)
        return f"{int(m)}m{s:.1f}s"


def analyze(results: dict):
    config = results.get("config", {})

    # Determine which benchmark type
    if "multi_lora_concurrent" in results:
        batched_key = "multi_lora_concurrent"
        sequential_key = "sequential_per_adapter"
        bench_type = "API-based (HTTP Client)"
    elif "multi_lora_batched" in results:
        batched_key = "multi_lora_batched"
        sequential_key = "sequential_per_adapter"
        bench_type = "Direct (vLLM Python API)"
    else:
        print("ERROR: Unrecognized result format")
        return

    batched = results[batched_key]
    sequential = results[sequential_key]

    print()
    print_header("BENCHMARK ANALYSIS REPORT")
    print()

    # Config
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"  benchmark_type: {bench_type}")
    print()

    # Main comparison table
    print_header("PERFORMANCE COMPARISON", 78)
    print()
    print(f"  {'Metric':<35} {'Multi-LoRA':>18} {'Sequential':>18}")
    print("  " + "-" * 73)

    metrics = [
        (
            "Wall Time",
            format_duration(batched["total_wall_time_sec"]),
            format_duration(sequential["total_wall_time_sec"]),
        ),
        ("Total Requests", str(batched["total_requests"]), str(sequential["total_requests"])),
        (
            "Total Output Tokens",
            f"{batched['total_output_tokens']:,}",
            f"{sequential['total_output_tokens']:,}",
        ),
        (
            "Avg Latency",
            format_duration(batched["avg_latency_sec"]),
            format_duration(sequential["avg_latency_sec"]),
        ),
        (
            "Throughput (req/s)",
            f"{batched['throughput_req_per_sec']:.2f}",
            f"{sequential['throughput_req_per_sec']:.2f}",
        ),
        (
            "Throughput (tok/s)",
            f"{batched['throughput_tok_per_sec']:.1f}",
            f"{sequential['throughput_tok_per_sec']:.1f}",
        ),
    ]

    # Add percentile latencies if present (API benchmark)
    if "p50_latency_sec" in batched:
        metrics.insert(
            4,
            (
                "P50 Latency",
                format_duration(batched["p50_latency_sec"]),
                format_duration(sequential["p50_latency_sec"]),
            ),
        )
        metrics.insert(
            5,
            (
                "P90 Latency",
                format_duration(batched["p90_latency_sec"]),
                format_duration(sequential["p90_latency_sec"]),
            ),
        )
        metrics.insert(
            6,
            (
                "P99 Latency",
                format_duration(batched["p99_latency_sec"]),
                format_duration(sequential["p99_latency_sec"]),
            ),
        )

    if "avg_ttft_sec" in batched:
        metrics.append(
            (
                "Avg TTFT",
                format_duration(batched["avg_ttft_sec"]),
                format_duration(sequential["avg_ttft_sec"]),
            )
        )

    for label, v1, v2 in metrics:
        print(f"  {label:<35} {v1:>18} {v2:>18}")

    print()

    # Speedup analysis
    print_header("SPEEDUP ANALYSIS", 78)
    print()

    b_wall = batched["total_wall_time_sec"]
    s_wall = sequential["total_wall_time_sec"]
    if b_wall > 0 and s_wall > 0:
        wall_speedup = s_wall / b_wall
        print(
            f"  Wall Time Speedup:      {wall_speedup:.2f}x "
            f"({'faster' if wall_speedup > 1 else 'slower'} with multi-LoRA)"
        )

    b_tput = batched["throughput_tok_per_sec"]
    s_tput = sequential["throughput_tok_per_sec"]
    if b_tput > 0 and s_tput > 0:
        tput_ratio = b_tput / s_tput
        print(
            f"  Throughput Ratio:        {tput_ratio:.2f}x "
            f"({'higher' if tput_ratio > 1 else 'lower'} with multi-LoRA)"
        )

    b_lat = batched["avg_latency_sec"]
    s_lat = sequential["avg_latency_sec"]
    if b_lat > 0 and s_lat > 0:
        lat_ratio = b_lat / s_lat
        print(
            f"  Latency Ratio:           {lat_ratio:.2f}x "
            f"({'worse' if lat_ratio > 1 else 'better'} with multi-LoRA)"
        )

    print()

    # Per-adapter breakdown (if available)
    seq_adapter_stats = sequential.get("per_adapter_stats") or sequential.get("per_adapter_times")
    if seq_adapter_stats:
        print_header("PER-ADAPTER BREAKDOWN (Sequential)", 78)
        print()
        print(f"  {'Adapter':<20} {'Time':>12} {'Output Tokens':>15} {'Tok/s':>12}")
        print("  " + "-" * 61)
        for name, stats in sorted(seq_adapter_stats.items()):
            t = stats.get("wall_time_sec", stats.get("avg_latency", 0) * stats.get("count", 1))
            toks = stats.get("output_tokens", stats.get("total_output_tokens", 0))
            tps = stats.get("throughput_tok_per_sec", toks / t if t > 0 else 0)
            print(f"  {name:<20} {format_duration(t):>12} {toks:>15,} {tps:>12.1f}")
        print()

    # Conclusion
    print_header("RECOMMENDATION", 78)
    print()
    if b_wall > 0 and s_wall > 0:
        wall_speedup = s_wall / b_wall
        if wall_speedup > 1.2:
            print("  ✅ Multi-LoRA batching provides SIGNIFICANT throughput improvement.")
            print(f"     {wall_speedup:.1f}x faster wall time with concurrent multi-LoRA requests.")
            print()
            print("  RECOMMENDATION: Use batch requests for multi-LoRA inference.")
            print("  Your system should aggregate requests across different LoRA adapters")
            print("  and submit them in batches to maximize GPU utilization.")
        elif wall_speedup > 1.0:
            print("  ➡️  Multi-LoRA batching provides MARGINAL improvement.")
            print(f"     {wall_speedup:.1f}x faster, but the benefit is small.")
            print()
            print("  RECOMMENDATION: Consider batching if request volume is high.")
            print("  For low-volume workloads, sequential processing may be simpler.")
        else:
            print("  ⚠️  Sequential processing is FASTER in this configuration.")
            print(f"     Multi-LoRA was {1 / wall_speedup:.1f}x slower.")
            print()
            print("  RECOMMENDATION: Stick with sequential per-adapter processing.")
            print("  Multi-LoRA overhead exceeds batching benefit for this workload.")
    print()
    print("=" * 78)


def main():
    if len(sys.argv) < 2:
        # Try default paths
        default_paths = [
            "results/benchmark_api_results.json",
            "results/benchmark_results.json",
        ]
        results_path = None
        base_dir = "/mnt/nas/hanzhang.yhz/multilora_bench"
        for p in default_paths:
            full = os.path.join(base_dir, p)
            if os.path.exists(full):
                results_path = full
                break
        if not results_path:
            print(f"Usage: python {sys.argv[0]} <results.json>")
            print("  Or run from the benchmark directory with results/ present.")
            sys.exit(1)
    else:
        results_path = sys.argv[1]

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)
    analyze(results)


if __name__ == "__main__":
    main()
