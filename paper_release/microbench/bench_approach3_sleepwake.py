#!/usr/bin/env python
"""Approach 3: Sleep/Wake mode-switch benchmark.

Training: FSDP with simulated CPU offload/reload
Sampling: vLLM with native sleep(level=1)/wake_up()
Switch S→T: vLLM sleep + FSDP offload to CPU
Switch T→S: FSDP reload to GPU + vLLM wake

Usage:
    python bench_approach3_sleepwake.py --model qwen3-4b --tp-size 2
    python bench_approach3_sleepwake.py --model qwen3-32b --tp-size 4
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
from utils import BenchmarkResult, compute_cold_hot, cuda_memory_snapshot, free_port, save_result, switch_summary


# ─── FSDP Training Worker (Persistent) ────────────────────────────────────────


def _fsdp_offload_to_cpu(model, optimizer) -> None:
    """Simulate FSDP offload: move model params and optimizer states to CPU.

    Uses FSDP internal _all_handles for efficient flat-param offload if available,
    otherwise falls back to model-level CPU offload.
    """
    torch.cuda.synchronize()

    # Try FSDP handle-based offload (most efficient for FULL_SHARD)
    handles = getattr(model, '_all_handles', None)
    if handles:
        for handle in handles:
            handle.to_cpu()
    else:
        # Fallback: move entire model to CPU
        # With use_orig_params=True, this moves the underlying flat_param storage
        model.cpu()
        # Also move any ignored modules (embed_tokens, norm, lm_head)
        inner = getattr(model, '_fsdp_wrapped_module', None)
        if inner is not None:
            inner.cpu()

    # Move optimizer states to CPU
    if optimizer is not None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device.type == "cuda":
                    state[k] = v.to("cpu", non_blocking=False)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _fsdp_reload_to_gpu(model, optimizer, device) -> None:
    """Simulate FSDP reload: move model params and optimizer states back to GPU.

    Uses FSDP internal _all_handles for efficient flat-param reload if available,
    otherwise falls back to model.to(device).
    """
    # Try FSDP handle-based reload
    handles = getattr(model, '_all_handles', None)
    if handles:
        for handle in handles:
            handle.flat_param_to(device, non_blocking=False)
    else:
        model.to(device)
        # Also move ignored modules back
        inner = getattr(model, '_fsdp_wrapped_module', None)
        if inner is not None:
            inner.to(device)

    # Move optimizer states back to GPU
    if optimizer is not None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                    state[k] = v.to(device, non_blocking=False)
    torch.cuda.synchronize()


def _training_worker(
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
    """FSDP training worker with offload/reload capability."""
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

        # Load FSDP model
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
        torch.cuda.empty_cache()

        # Measure training
        train_times = []
        for _ in range(measure_rounds):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss, step_ms = train_step_fsdp(fsdp_model, optimizer, rl_batch)
            torch.cuda.synchronize()
            train_times.append(step_ms)

        mean_train_ms = sum(train_times) / len(train_times)
        peak_train_mem = torch.cuda.max_memory_allocated() / (1024**3)

        # Measure offload (S→T switch part)
        dist.barrier()
        torch.cuda.synchronize()
        offload_start = time.perf_counter()
        _fsdp_offload_to_cpu(fsdp_model, optimizer)
        torch.cuda.synchronize()
        offload_ms = (time.perf_counter() - offload_start) * 1000
        mem_after_offload = torch.cuda.memory_allocated() / (1024**3)

        dist.barrier()

        # Measure reload (T→S switch part)
        torch.cuda.synchronize()
        reload_start = time.perf_counter()
        _fsdp_reload_to_gpu(fsdp_model, optimizer, device)
        torch.cuda.synchronize()
        reload_ms = (time.perf_counter() - reload_start) * 1000

        dist.barrier()

        # Verify training still works after reload
        loss_verify, _ = train_step_fsdp(fsdp_model, optimizer, rl_batch)
        torch.cuda.synchronize()

        conn.send({
            "type": "READY",
            "rank": rank,
            "loss": float(loss),
            "loss_verify": float(loss_verify),
            "mean_train_ms": mean_train_ms,
            "train_times": train_times,
            "model_load_ms": model_load_ms,
            "offload_ms": offload_ms,
            "reload_ms": reload_ms,
            "peak_train_mem_gb": peak_train_mem,
            "mem_after_offload_gb": mem_after_offload,
        })

        # Wait for exit signal
        msg = conn.recv()
        dist.destroy_process_group()

    except Exception as exc:
        import traceback
        try:
            conn.send({"type": "ERROR", "rank": rank, "error": repr(exc),
                       "traceback": traceback.format_exc()})
        finally:
            raise


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Sleep/Wake Switch Benchmark")
    parser.add_argument("--model", default="qwen3-4b", choices=["qwen3-4b", "qwen3-32b"])
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--measure-rounds", type=int, default=5)
    parser.add_argument("--sleep-level", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    model_cfg = MODELS[args.model]
    workload = WORKLOADS[args.model]
    gpu_mem_util = get_exclusive_gpu_mem_util(args.model, args.tp_size)

    print("=" * 70)
    print("  Approach 3: Sleep/Wake Mode-Switch Benchmark")
    print("=" * 70)
    print(f"  Model: {model_cfg['name']} | TP={args.tp_size}")
    print(f"  Training: FSDP, batch={workload.train_batch_size}, seq_len={workload.train_seq_len}")
    print(f"  Sampling: vLLM sleep/wake, prompts={workload.num_prompts}, max_tokens={workload.max_tokens}")
    print(f"  GPU mem util: {gpu_mem_util:.2f}, sleep_level={args.sleep_level}")
    print("=" * 70, flush=True)

    # ─── Phase 1: FSDP Training + Offload/Reload measurement ──────────────────
    print("\n  [Phase 1] FSDP Training workers...", flush=True)
    ctx = mp.get_context("spawn")
    master_port = free_port()
    conns = []
    workers = []

    for rank in range(args.tp_size):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_training_worker,
            args=(
                rank, args.tp_size, args.model,
                workload.train_batch_size, workload.train_seq_len,
                master_port, args.warmup_rounds, args.measure_rounds,
                child_conn,
            ),
        )
        proc.start()
        conns.append(parent_conn)
        workers.append(proc)

    # Collect training results
    ready_msgs = []
    for conn in conns:
        msg = conn.recv()
        if msg.get("type") == "ERROR":
            print(f"  [ERROR] Worker {msg.get('rank')}: {msg['error']}")
            if "traceback" in msg:
                print(msg["traceback"])
            sys.exit(1)
        ready_msgs.append(msg)
    ready_msgs.sort(key=lambda m: m["rank"])

    mean_train_ms = sum(m["mean_train_ms"] for m in ready_msgs) / len(ready_msgs)
    mean_offload_ms = sum(m["offload_ms"] for m in ready_msgs) / len(ready_msgs)
    mean_reload_ms = sum(m["reload_ms"] for m in ready_msgs) / len(ready_msgs)
    peak_train_mem = max(m["peak_train_mem_gb"] for m in ready_msgs)
    max_mem_after_offload = max(float(m.get("mem_after_offload_gb", 0.0)) for m in ready_msgs)
    # Cold/hot for training
    all_train_times = ready_msgs[0]["train_times"]
    cold_train_ms, hot_train_ms = compute_cold_hot(all_train_times)

    print(f"  [Training] mean_step={mean_train_ms:.1f}ms")
    print(f"  [Offload] mean={mean_offload_ms:.1f}ms (FSDP → CPU)")
    print(f"  [Reload]  mean={mean_reload_ms:.1f}ms (CPU → FSDP)")
    print(f"  [Memory]  peak_train={peak_train_mem:.2f}GB")

    # Signal workers to exit
    for conn in conns:
        try:
            conn.send("EXIT")
        except Exception:
            pass
    for proc in workers:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)

    # ─── Phase 2: vLLM Sampling + Sleep/Wake measurement ──────────────────────
    print("\n  [Phase 2] vLLM Sampling with sleep/wake...", flush=True)

    # Clear distributed env
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        os.environ.pop(key, None)

    from vllm import LLM, SamplingParams

    # Load vLLM with sleep mode enabled
    vllm_load_start = time.perf_counter()
    llm = LLM(
        model=model_cfg["path"],
        dtype="bfloat16",
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=gpu_mem_util,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=workload.max_model_len,
        enable_sleep_mode=True,
    )
    torch.cuda.synchronize()
    vllm_load_ms = (time.perf_counter() - vllm_load_start) * 1000
    print(f"  [vLLM] load={vllm_load_ms:.0f}ms")

    prompts = get_prompts(workload.num_prompts)
    params = SamplingParams(
        max_tokens=workload.max_tokens, temperature=0.0, top_p=1.0, seed=42,
    )

    # Warmup sampling
    for _ in range(args.warmup_rounds):
        llm.generate(prompts[:4], params)

    # Measure sampling throughput
    throughputs = []
    latencies = []
    for _ in range(args.measure_rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        throughput = total_tokens / (latency_ms / 1000)
        throughputs.append(throughput)
        latencies.append(latency_ms)

    mean_throughput = sum(throughputs) / len(throughputs)
    mean_latency = sum(latencies) / len(latencies)
    peak_sampling_mem = torch.cuda.max_memory_allocated() / (1024**3)
    cold_samp_tput, hot_samp_tput = compute_cold_hot(throughputs)
    cold_samp_lat, hot_samp_lat = compute_cold_hot(latencies)

    print(f"  [Sampling] throughput={mean_throughput:.1f} tok/s, latency={mean_latency:.0f}ms")

    # Measure vLLM sleep
    torch.cuda.synchronize()
    sleep_start = time.perf_counter()
    llm.sleep(level=args.sleep_level)
    torch.cuda.synchronize()
    vllm_sleep_ms = (time.perf_counter() - sleep_start) * 1000

    # Measure vLLM wake
    torch.cuda.synchronize()
    wake_start = time.perf_counter()
    llm.wake_up()
    torch.cuda.synchronize()
    vllm_wake_ms = (time.perf_counter() - wake_start) * 1000

    print(f"  [vLLM] sleep={vllm_sleep_ms:.1f}ms, wake={vllm_wake_ms:.1f}ms")

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # ─── Compute Switch Overhead ──────────────────────────────────────────────
    # S→T: vLLM sleep + FSDP offload (both need to release GPU)
    # But in practice: vLLM sleeps first, then FSDP is already on GPU (no offload needed for T)
    # Actually the switch is:
    #   S→T: vLLM sleep (frees vLLM memory) → FSDP reload from CPU to GPU
    #   T→S: FSDP offload to CPU → vLLM wake (reloads vLLM weights)
    switch_s2t_ms = vllm_sleep_ms + mean_reload_ms
    switch_t2s_ms = mean_offload_ms + vllm_wake_ms
    t2s_times = [switch_t2s_ms for _ in range(args.measure_rounds)]
    s2t_times = [switch_s2t_ms for _ in range(args.measure_rounds)]
    switch_metrics = switch_summary(t2s_times, s2t_times)

    print(f"\n  [Switch] S→T = vLLM_sleep({vllm_sleep_ms:.0f}) + FSDP_reload({mean_reload_ms:.0f}) "
          f"= {switch_s2t_ms:.0f}ms")
    print(f"  [Switch] T→S = FSDP_offload({mean_offload_ms:.0f}) + vLLM_wake({vllm_wake_ms:.0f}) "
          f"= {switch_t2s_ms:.0f}ms")

    # Build result
    tokens_per_step = workload.train_batch_size * workload.train_seq_len
    train_throughput = tokens_per_step / (mean_train_ms / 1000)

    result = BenchmarkResult(
        model=model_cfg["name"],
        model_path=model_cfg["path"],
        tp_size=args.tp_size,
        dp_size=1,
        approach="sleep_wake",
        approach_name="Sleep/Wake Switch",
        switch_s2t_ms=switch_metrics["switch_s2t_ms"],
        switch_t2s_ms=switch_metrics["switch_t2s_ms"],
        switch_total_ms=switch_metrics["switch_total_ms"],
        sampling_throughput_toks=mean_throughput,
        sampling_latency_ms=mean_latency,
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
        sampling_throughputs=throughputs,
        sampling_latencies_ms=latencies,
        # Memory
        gpu_memory_utilization=gpu_mem_util,
        peak_memory_sampling_gb=peak_sampling_mem,
        peak_memory_training_gb=peak_train_mem,
        training_loaded_memory_gb=peak_train_mem,
        sampling_loaded_memory_gb=peak_sampling_mem,
        memory_after_t2s_release_gb=max_mem_after_offload,
        oom=False,
        workload={
            "train_batch_size": workload.train_batch_size,
            "train_seq_len": workload.train_seq_len,
            "num_prompts": workload.num_prompts,
            "max_tokens": workload.max_tokens,
            "max_model_len": workload.max_model_len,
        },
        raw_metrics={
            "vllm_sleep_ms": vllm_sleep_ms,
            "vllm_wake_ms": vllm_wake_ms,
            "fsdp_offload_ms": mean_offload_ms,
            "fsdp_reload_ms": mean_reload_ms,
            "fsdp_mem_after_offload_gb": max_mem_after_offload,
            "s2t_simulation_note": "FSDP has no native sleep/wake; sampling→training reload is simulated by moving FSDP params/optimizer back to GPU",
            "vllm_load_ms": vllm_load_ms,
            "sleep_level": args.sleep_level,
        },
    )

    print(f"\n__RESULT_JSON__{json.dumps(result.to_dict())}")
    print("\nDONE")

    if args.output_dir:
        from pathlib import Path
        save_result(result, Path(args.output_dir))


if __name__ == "__main__":
    main()
