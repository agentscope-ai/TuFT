#!/usr/bin/env python
"""Approach 2: Process-level mode-switch benchmark.

Training: FSDP (separate process, killed after training)
Sampling: vLLM (separate process, killed after sampling)
Switch: Kill process + reload model from disk

This measures the worst-case switching overhead where the entire runtime
is destroyed and recreated for each mode transition.

Usage:
    python bench_approach2_process.py --model qwen3-4b --tp-size 2
    python bench_approach2_process.py --model qwen3-32b --tp-size 4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Any

# ─── Path & Env Setup ─────────────────────────────────────────────────────────

EXP_ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH_ROOT = os.environ.get("TUFT_FLEX_BENCH_ROOT", "/mnt/nas/hanzhang.yhz/lora_rl_bench")
TUFT_ROOT = os.environ.get("TUFT_FLEX_TUFT_ROOT", "/mnt/nas/hanzhang.yhz/flex_backend/TuFT")
sys.path.insert(0, EXP_ROOT)
sys.path.insert(0, BENCH_ROOT)
sys.path.insert(0, os.path.join(TUFT_ROOT, "src"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
os.environ["PATH"] = "/usr/local/cuda-12.9/bin:" + os.environ.get("PATH", "")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from config import MODELS, WORKLOADS, get_exclusive_gpu_mem_util, get_prompts
from utils import BenchmarkResult, compute_cold_hot, free_port, save_result, switch_summary


# ─── vLLM Sampling Worker ─────────────────────────────────────────────────────


def _vllm_sampling_worker(
    model_path: str,
    tp_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_tokens: int,
    num_prompts: int,
    warmup_rounds: int,
    measure_rounds: int,
    conn,
) -> None:
    """vLLM sampling worker (runs in main process, signals results via conn)."""
    try:
        from vllm import LLM, SamplingParams

        # Load vLLM (measures load time = part of switch overhead)
        load_start = time.perf_counter()
        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=max_model_len,
        )
        torch.cuda.synchronize()
        load_ms = (time.perf_counter() - load_start) * 1000

        prompts = get_prompts(num_prompts)
        params = SamplingParams(
            max_tokens=max_tokens, temperature=0.0, top_p=1.0, seed=42,
        )

        # Warmup
        for _ in range(warmup_rounds):
            llm.generate(prompts[:4], params)

        # Measure
        throughputs = []
        latencies = []
        for _ in range(measure_rounds):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, params)
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000
            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            throughput = total_tokens / (latency_ms / 1000)
            throughputs.append(throughput)
            latencies.append(latency_ms)

        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

        conn.send({
            "type": "READY",
            "load_ms": load_ms,
            "mean_throughput": sum(throughputs) / len(throughputs),
            "mean_latency_ms": sum(latencies) / len(latencies),
            "throughputs": throughputs,
            "peak_mem_gb": peak_mem,
        })

    except Exception as exc:
        import traceback
        conn.send({"type": "ERROR", "error": repr(exc),
                   "traceback": traceback.format_exc()})


# ─── FSDP Training Worker ─────────────────────────────────────────────────────


def _fsdp_training_worker(
    rank: int,
    world_size: int,
    model_name: str,
    train_batch: int,
    train_seq_len: int,
    master_port: int,
    warmup_rounds: int,
    measure_rounds: int,
    conn,
) -> None:
    """FSDP training worker process."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    try:
        from datetime import timedelta
        dist.init_process_group("nccl", rank=rank, world_size=world_size,
                                timeout=timedelta(minutes=30))

        from common.config import load_bench_config, load_model_spec
        from common.model_utils import load_fsdp_model, load_tokenizer
        from common.training_utils import create_optimizer, make_synthetic_rl_batch, train_step_fsdp

        spec = load_model_spec(model_name)
        bench = load_bench_config()
        tokenizer = load_tokenizer(spec)

        # Load model (measures load time = part of switch overhead)
        load_start = time.perf_counter()
        fsdp_model, _ = load_fsdp_model(spec, bench, rank, world_size)
        optimizer = create_optimizer(fsdp_model, lr=1e-4)
        torch.cuda.synchronize()
        model_load_ms = (time.perf_counter() - load_start) * 1000

        rl_batch = make_synthetic_rl_batch(
            tokenizer, train_batch, train_seq_len, device=device, seed=bench.seed,
        )

        # Warmup
        for _ in range(warmup_rounds):
            train_step_fsdp(fsdp_model, optimizer, rl_batch)
        torch.cuda.synchronize()

        # Measure
        train_times = []
        for _ in range(measure_rounds):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss, step_ms = train_step_fsdp(fsdp_model, optimizer, rl_batch)
            torch.cuda.synchronize()
            train_times.append(step_ms)

        mean_train_ms = sum(train_times) / len(train_times)
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

        dist.destroy_process_group()

        conn.send({
            "type": "READY",
            "rank": rank,
            "loss": float(loss),
            "mean_train_ms": mean_train_ms,
            "train_times": train_times,
            "model_load_ms": model_load_ms,
            "peak_mem_gb": peak_mem,
        })

    except Exception as exc:
        import traceback
        try:
            conn.send({"type": "ERROR", "rank": rank, "error": repr(exc),
                       "traceback": traceback.format_exc()})
        finally:
            raise


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Process-level Switch Benchmark")
    parser.add_argument("--model", default="qwen3-4b", choices=["qwen3-4b", "qwen3-32b"])
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--measure-rounds", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    model_cfg = MODELS[args.model]
    workload = WORKLOADS[args.model]
    gpu_mem_util = get_exclusive_gpu_mem_util(args.model, args.tp_size)

    print("=" * 70)
    print("  Approach 2: Process-level Mode-Switch Benchmark")
    print("=" * 70)
    print(f"  Model: {model_cfg['name']} | TP={args.tp_size}")
    print(f"  Training: FSDP, batch={workload.train_batch_size}, seq_len={workload.train_seq_len}")
    print(f"  Sampling: vLLM, prompts={workload.num_prompts}, max_tokens={workload.max_tokens}")
    print(f"  GPU mem util: {gpu_mem_util:.2f}")
    print("=" * 70, flush=True)

    ctx = mp.get_context("spawn")

    # ─── Phase 1: vLLM Sampling ───────────────────────────────────────────────
    print("\n  [Phase 1] vLLM Sampling (will kill process after)...", flush=True)

    parent_conn, child_conn = ctx.Pipe()
    vllm_proc = ctx.Process(
        target=_vllm_sampling_worker,
        args=(
            model_cfg["path"], args.tp_size, gpu_mem_util,
            workload.max_model_len, workload.max_tokens,
            workload.num_prompts, args.warmup_rounds, args.measure_rounds,
            child_conn,
        ),
    )

    vllm_start = time.perf_counter()
    vllm_proc.start()
    vllm_msg = parent_conn.recv()

    if vllm_msg.get("type") == "ERROR":
        print(f"  [ERROR] vLLM failed: {vllm_msg['error']}")
        if "traceback" in vllm_msg:
            print(vllm_msg["traceback"])
        sys.exit(1)

    # Kill vLLM process and measure cleanup time
    kill_start = time.perf_counter()
    vllm_proc.terminate()
    vllm_proc.join(timeout=30)
    if vllm_proc.is_alive():
        vllm_proc.kill()
        vllm_proc.join(timeout=10)
    # Wait for GPU memory to be freed
    time.sleep(2)  # allow CUDA context cleanup
    torch.cuda.empty_cache()
    kill_vllm_ms = (time.perf_counter() - kill_start) * 1000

    vllm_total_startup_ms = vllm_msg["load_ms"]
    print(f"  [vLLM] load={vllm_msg['load_ms']:.0f}ms, "
          f"throughput={vllm_msg['mean_throughput']:.1f} tok/s")
    print(f"  [vLLM] kill_cleanup={kill_vllm_ms:.0f}ms")

    # ─── Phase 2: FSDP Training ───────────────────────────────────────────────
    print("\n  [Phase 2] FSDP Training (fresh process)...", flush=True)

    master_port = free_port()
    train_conns = []
    train_workers = []

    fsdp_spawn_start = time.perf_counter()
    for rank in range(args.tp_size):
        pc, cc = ctx.Pipe()
        proc = ctx.Process(
            target=_fsdp_training_worker,
            args=(
                rank, args.tp_size, args.model,
                workload.train_batch_size, workload.train_seq_len,
                master_port, args.warmup_rounds, args.measure_rounds,
                cc,
            ),
        )
        proc.start()
        train_conns.append(pc)
        train_workers.append(proc)

    # Collect results
    train_msgs = []
    for conn in train_conns:
        msg = conn.recv()
        if msg.get("type") == "ERROR":
            print(f"  [ERROR] Worker {msg.get('rank')}: {msg['error']}")
            if "traceback" in msg:
                print(msg["traceback"])
            sys.exit(1)
        train_msgs.append(msg)
    train_msgs.sort(key=lambda m: m["rank"])

    # Kill training processes
    kill_train_start = time.perf_counter()
    for proc in train_workers:
        proc.terminate()
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=10)
    time.sleep(2)
    torch.cuda.empty_cache()
    kill_train_ms = (time.perf_counter() - kill_train_start) * 1000

    mean_train_ms = sum(m["mean_train_ms"] for m in train_msgs) / len(train_msgs)
    mean_load_ms = sum(m["model_load_ms"] for m in train_msgs) / len(train_msgs)
    peak_train_mem = max(m["peak_mem_gb"] for m in train_msgs)
    all_train_times = train_msgs[0]["train_times"]
    cold_train_ms, hot_train_ms = compute_cold_hot(all_train_times)

    print(f"  [FSDP] load={mean_load_ms:.0f}ms, train_step={mean_train_ms:.1f}ms")
    print(f"  [FSDP] kill_cleanup={kill_train_ms:.0f}ms")

    # ─── Compute Switch Overhead ──────────────────────────────────────────────
    # S→T: kill vLLM + spawn FSDP processes + load FSDP model
    switch_s2t_ms = kill_vllm_ms + mean_load_ms
    # T→S: kill FSDP + spawn vLLM process + load vLLM model
    switch_t2s_ms = kill_train_ms + vllm_total_startup_ms
    t2s_times = [switch_t2s_ms for _ in range(args.measure_rounds)]
    s2t_times = [switch_s2t_ms for _ in range(args.measure_rounds)]
    switch_metrics = switch_summary(t2s_times, s2t_times)

    print(f"\n  [Switch] S→T = kill_vLLM({kill_vllm_ms:.0f}) + load_FSDP({mean_load_ms:.0f}) "
          f"= {switch_s2t_ms:.0f}ms")
    print(f"  [Switch] T→S = kill_FSDP({kill_train_ms:.0f}) + load_vLLM({vllm_total_startup_ms:.0f}) "
          f"= {switch_t2s_ms:.0f}ms")

    # Build result
    tokens_per_step = workload.train_batch_size * workload.train_seq_len
    train_throughput = tokens_per_step / (mean_train_ms / 1000)
    # For process switch, cold/hot sampling from vLLM worker
    vllm_throughputs = vllm_msg.get("throughputs", [])
    vllm_latencies = vllm_msg.get("latencies", [])
    cold_samp_tput, hot_samp_tput = compute_cold_hot(vllm_throughputs)
    cold_samp_lat, hot_samp_lat = compute_cold_hot(vllm_latencies)

    result = BenchmarkResult(
        model=model_cfg["name"],
        model_path=model_cfg["path"],
        tp_size=args.tp_size,
        dp_size=1,
        approach="process_switch",
        approach_name="Process-level Switch",
        switch_s2t_ms=switch_metrics["switch_s2t_ms"],
        switch_t2s_ms=switch_metrics["switch_t2s_ms"],
        switch_total_ms=switch_metrics["switch_total_ms"],
        sampling_throughput_toks=vllm_msg["mean_throughput"],
        sampling_latency_ms=vllm_msg["mean_latency_ms"],
        training_step_ms=mean_train_ms,
        training_throughput_toks=train_throughput,
        # Switch cold/hot and per-round sequence
        cold_switch_t2s_ms=switch_metrics["cold_switch_t2s_ms"],
        hot_switch_t2s_ms=switch_metrics["hot_switch_t2s_ms"],
        cold_switch_s2t_ms=switch_metrics["cold_switch_s2t_ms"],
        hot_switch_s2t_ms=switch_metrics["hot_switch_s2t_ms"],
        cold_roundtrip_switch_ms=switch_metrics["cold_roundtrip_switch_ms"],
        hot_roundtrip_switch_ms=switch_metrics["hot_roundtrip_switch_ms"],
        switch_t2s_times_ms=switch_metrics["switch_t2s_times_ms"],
        switch_s2t_times_ms=switch_metrics["switch_s2t_times_ms"],
        roundtrip_switch_times_ms=switch_metrics["roundtrip_switch_times_ms"],
        # Deprecated performance cold/hot fields retained for compatibility
        cold_training_step_ms=0.0,
        hot_training_step_ms=0.0,
        cold_sampling_throughput_toks=0.0,
        hot_sampling_throughput_toks=0.0,
        cold_sampling_latency_ms=0.0,
        hot_sampling_latency_ms=0.0,
        training_step_times_ms=all_train_times,
        sampling_throughputs=vllm_throughputs,
        sampling_latencies_ms=vllm_latencies,
        # Memory
        gpu_memory_utilization=gpu_mem_util,
        peak_memory_sampling_gb=vllm_msg["peak_mem_gb"],
        peak_memory_training_gb=peak_train_mem,
        training_loaded_memory_gb=peak_train_mem,
        sampling_loaded_memory_gb=vllm_msg["peak_mem_gb"],
        oom=False,
        workload={
            "train_batch_size": workload.train_batch_size,
            "train_seq_len": workload.train_seq_len,
            "num_prompts": workload.num_prompts,
            "max_tokens": workload.max_tokens,
            "max_model_len": workload.max_model_len,
        },
        raw_metrics={
            "vllm_load_ms": vllm_msg["load_ms"],
            "fsdp_load_ms": mean_load_ms,
            "kill_vllm_ms": kill_vllm_ms,
            "kill_fsdp_ms": kill_train_ms,
        },
    )

    print(f"\n__RESULT_JSON__{json.dumps(result.to_dict())}")
    print("\nDONE")

    if args.output_dir:
        from pathlib import Path
        save_result(result, Path(args.output_dir))


if __name__ == "__main__":
    main()
