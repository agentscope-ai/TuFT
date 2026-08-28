"""Common utilities for mode-switch micro-benchmark.

Provides timing, memory monitoring, result formatting, and subprocess helpers.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator

import torch


# ─── Environment Setup ────────────────────────────────────────────────────────


def setup_env() -> None:
    """Set up environment variables for benchmark execution."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
    os.environ["PATH"] = "/usr/local/cuda-12.9/bin:" + os.environ.get("PATH", "")


def setup_sys_path() -> None:
    """Add lora_rl_bench and TuFT to sys.path."""
    bench_root = "/mnt/nas/hanzhang.yhz/lora_rl_bench"
    tuft_src = "/mnt/nas/hanzhang.yhz/flex_backend/TuFT/src"
    exp_root = str(Path(__file__).resolve().parent)
    for p in [bench_root, tuft_src, exp_root]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ─── Memory Monitoring ────────────────────────────────────────────────────────


@dataclass
class MemorySnapshot:
    """GPU memory snapshot in GB."""

    allocated_gb: float = 0.0
    reserved_gb: float = 0.0
    free_gb: float = 0.0
    total_gb: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def cuda_memory_snapshot(device: int | None = None) -> MemorySnapshot:
    """Capture current GPU memory state."""
    if not torch.cuda.is_available():
        return MemorySnapshot()
    if device is not None:
        torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return MemorySnapshot(
        allocated_gb=torch.cuda.memory_allocated() / (1024**3),
        reserved_gb=torch.cuda.memory_reserved() / (1024**3),
        free_gb=free_bytes / (1024**3),
        total_gb=total_bytes / (1024**3),
    )


def peak_memory_gb(device: int | None = None) -> float:
    """Get peak GPU memory allocated in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def reset_peak_memory(device: int | None = None) -> None:
    """Reset peak memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


# ─── Timing ───────────────────────────────────────────────────────────────────


@contextmanager
def timer(sync_cuda: bool = True) -> Generator[dict[str, float], None, None]:
    """Context manager for precise timing with optional CUDA sync.

    Usage:
        with timer() as t:
            do_work()
        print(f"Elapsed: {t['ms']:.2f} ms")
    """
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result: dict[str, float] = {}
    yield result
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result["ms"] = elapsed * 1000
    result["s"] = elapsed


def measure_fn(
    fn: Callable[[], Any],
    warmup: int = 1,
    rounds: int = 3,
    sync_cuda: bool = True,
) -> dict[str, Any]:
    """Run fn with warmup, measure over multiple rounds.

    Returns dict with mean_ms, min_ms, max_ms, all_ms, and last_result.
    """
    # Warmup
    for _ in range(warmup):
        fn()

    # Measure
    times_ms = []
    last_result = None
    for _ in range(rounds):
        with timer(sync_cuda=sync_cuda) as t:
            last_result = fn()
        times_ms.append(t["ms"])

    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "all_ms": times_ms,
        "last_result": last_result,
    }


# ─── Networking ───────────────────────────────────────────────────────────────


def free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ─── Result Data Structure ────────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run with cold/hot distinction.

    Cold = first measurement round (includes JIT, cache warmup, one-time costs)
    Hot  = average of rounds 2..N (steady-state performance)
    """

    model: str
    model_path: str
    tp_size: int
    dp_size: int
    approach: str
    approach_name: str
    # Switching overhead (hot = steady-state)
    switch_s2t_ms: float = 0.0  # sampling → training
    switch_t2s_ms: float = 0.0  # training → sampling
    switch_total_ms: float = 0.0  # full round-trip
    # Sampling performance (hot = steady-state average)
    sampling_throughput_toks: float = 0.0
    sampling_latency_ms: float = 0.0
    # Training performance (hot = steady-state average)
    training_step_ms: float = 0.0
    training_throughput_toks: float = 0.0  # tokens processed per second
    # ─── Cold/Hot distinction for SWITCH only ─────────────────────────────────
    # Training and sampling performance are steady workload metrics; cold/hot is
    # meaningful only for mode switching because first switch includes one-time
    # runtime load/cache/JIT costs while later switches are steady-state.
    cold_switch_t2s_ms: float = 0.0  # first training → sampling switch
    hot_switch_t2s_ms: float = 0.0  # avg subsequent training → sampling switches
    cold_switch_s2t_ms: float = 0.0  # first sampling → training switch
    hot_switch_s2t_ms: float = 0.0  # avg subsequent sampling → training switches
    cold_roundtrip_switch_ms: float = 0.0  # cold T→S + S→T
    hot_roundtrip_switch_ms: float = 0.0  # hot T→S + S→T
    # Backward-compatible deprecated performance cold/hot fields.
    cold_training_step_ms: float = 0.0
    hot_training_step_ms: float = 0.0
    cold_sampling_throughput_toks: float = 0.0
    hot_sampling_throughput_toks: float = 0.0
    cold_sampling_latency_ms: float = 0.0
    hot_sampling_latency_ms: float = 0.0
    # Per-round data for full transparency
    switch_t2s_times_ms: list[float] = field(default_factory=list)
    switch_s2t_times_ms: list[float] = field(default_factory=list)
    roundtrip_switch_times_ms: list[float] = field(default_factory=list)
    training_step_times_ms: list[float] = field(default_factory=list)
    sampling_throughputs: list[float] = field(default_factory=list)
    sampling_latencies_ms: list[float] = field(default_factory=list)
    # Memory
    gpu_memory_utilization: float = 0.0
    peak_memory_sampling_gb: float = 0.0
    peak_memory_training_gb: float = 0.0
    # Memory transition snapshots / deltas
    training_loaded_memory_gb: float = 0.0
    sampling_loaded_memory_gb: float = 0.0
    memory_after_t2s_release_gb: float = 0.0
    memory_after_s2t_release_gb: float = 0.0
    free_memory_after_t2s_release_gb: float = 0.0
    free_memory_after_s2t_release_gb: float = 0.0
    coexist_memory_pressure_gb: float = 0.0
    # Status
    oom: bool = False
    oom_phase: str = ""  # "sampling" or "training"
    error: str = ""
    notes: str = ""
    # Metadata
    workload: dict[str, Any] = field(default_factory=dict)
    raw_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_cold_hot(values: list[float]) -> tuple[float, float]:
    """Compute cold (first) and hot (avg of rest) from a list of measurements.

    Returns (cold, hot). If only 1 value, hot = cold.
    """
    if not values:
        return 0.0, 0.0
    cold = values[0]
    hot = sum(values[1:]) / len(values[1:]) if len(values) > 1 else cold
    return cold, hot


def switch_summary(t2s_times: list[float], s2t_times: list[float]) -> dict[str, Any]:
    """Summarize bidirectional switch times for T→S→T→S round-trip flows."""
    cold_t2s, hot_t2s = compute_cold_hot(t2s_times)
    cold_s2t, hot_s2t = compute_cold_hot(s2t_times)
    roundtrips = [a + b for a, b in zip(t2s_times, s2t_times)]
    cold_rt, hot_rt = compute_cold_hot(roundtrips)
    return {
        "switch_t2s_times_ms": t2s_times,
        "switch_s2t_times_ms": s2t_times,
        "roundtrip_switch_times_ms": roundtrips,
        "cold_switch_t2s_ms": cold_t2s,
        "hot_switch_t2s_ms": hot_t2s,
        "cold_switch_s2t_ms": cold_s2t,
        "hot_switch_s2t_ms": hot_s2t,
        "cold_roundtrip_switch_ms": cold_rt,
        "hot_roundtrip_switch_ms": hot_rt,
        "switch_t2s_ms": hot_t2s,
        "switch_s2t_ms": hot_s2t,
        "switch_total_ms": hot_rt,
    }


def save_result(result: BenchmarkResult, output_dir: Path) -> Path:
    """Save a single benchmark result to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{result.model}_tp{result.tp_size}_{result.approach}.json"
    path = output_dir / filename
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return path


# ─── Subprocess Runner ────────────────────────────────────────────────────────


def run_benchmark_subprocess(
    script: Path,
    args: list[str],
    timeout_s: int = 3600,
    env_extra: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a benchmark script as subprocess for GPU memory isolation.

    Returns (returncode, stdout, stderr).
    """
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
    env["PATH"] = "/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    if env_extra:
        env.update(env_extra)

    cmd = [sys.executable, str(script)] + args
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_json_from_output(output: str) -> dict[str, Any] | None:
    """Extract JSON result from subprocess output (last JSON block)."""
    lines = output.strip().split("\n")
    # Find the last line starting with '{' or containing __RESULT_JSON__
    for line in reversed(lines):
        if "__RESULT_JSON__" in line:
            json_str = line.split("__RESULT_JSON__")[1].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
        if line.strip().startswith("{"):
            try:
                return json.loads(line.strip())
            except json.JSONDecodeError:
                continue
    # Try to find multi-line JSON at the end
    json_start = output.rfind("\n__RESULT_JSON__\n")
    if json_start >= 0:
        json_str = output[json_start + len("\n__RESULT_JSON__\n"):]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    return None


# ─── Formatting ───────────────────────────────────────────────────────────────


def format_results_table(results: list[BenchmarkResult]) -> str:
    """Format results as a paper-friendly table."""
    header = (
        f"{'Model':<12} {'TP':>3} {'DP':>3} {'Approach':<20} "
        f"{'S→T(ms)':>10} {'T→S(ms)':>10} "
        f"{'Samp(tok/s)':>12} {'Train(ms)':>10} "
        f"{'MemUtil':>7} {'MemT':>7} {'MemS':>7} {'OOM':>5}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        oom_str = r.oom_phase if r.oom else "-"
        mem_t = r.peak_memory_training_gb
        mem_s = r.peak_memory_sampling_gb
        lines.append(
            f"{r.model:<12} {r.tp_size:>3} {r.dp_size:>3} {r.approach_name:<20} "
            f"{r.switch_s2t_ms:>10.1f} {r.switch_t2s_ms:>10.1f} "
            f"{r.sampling_throughput_toks:>12.1f} {r.training_step_ms:>10.1f} "
            f"{r.gpu_memory_utilization:>7.2f} {mem_t:>7.1f} {mem_s:>7.1f} {oom_str:>5}"
        )
    return "\n".join(lines)


def print_result_summary(result: BenchmarkResult) -> None:
    """Print a concise summary of one benchmark result."""
    print(f"\n{'='*60}")
    print(f"  {result.approach_name} | {result.model} | TP={result.tp_size} DP={result.dp_size}")
    print(f"{'='*60}")
    if result.oom:
        print(f"  OOM during: {result.oom_phase}")
        if result.error:
            print(f"  Error: {result.error[:200]}")
        return
    print(f"  Switch S→T: {result.switch_s2t_ms:.1f} ms")
    print(f"  Switch T→S: {result.switch_t2s_ms:.1f} ms")
    print(f"  Sampling:   {result.sampling_throughput_toks:.1f} tok/s")
    print(f"  Training:   {result.training_step_ms:.1f} ms/step")
    print(f"  Mem util:   {result.gpu_memory_utilization:.2f}")
    print(f"  Peak mem:   sampling={result.peak_memory_sampling_gb:.1f}GB, "
          f"training={result.peak_memory_training_gb:.1f}GB")
