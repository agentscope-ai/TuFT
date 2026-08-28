#!/usr/bin/env python
"""Approach 1: Coexistence (No Switch) mode-switch benchmark.

Training: FSDP (coexists in GPU memory with vLLM)
Sampling: vLLM (coexists in GPU memory with FSDP)
Switch: None (both always resident)

Both runtimes share GPU memory simultaneously. vLLM gets a configured
gpu_memory_utilization fraction; FSDP uses whatever remains.
Training may OOM if insufficient memory remains.

Usage:
    python bench_approach1_coexist.py --model qwen3-4b --tp-size 2
    python bench_approach1_coexist.py --model qwen3-32b --tp-size 4
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

from config import (
    GPU_TOTAL_GB,
    MODELS,
    WORKLOADS,
    get_coexistence_gpu_mem_util,
    get_prompts,
)
from utils import BenchmarkResult, compute_cold_hot, free_port, save_result, switch_summary


# ─── FSDP Training Worker (coexists with vLLM) ────────────────────────────────


def _fsdp_coexist_worker(
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
    """FSDP training worker that coexists with vLLM in GPU memory.

    This worker is spawned FIRST, loads the model, then waits for vLLM
    to be loaded in the main process before training.
    """
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

        mem_after_load = torch.cuda.memory_allocated() / (1024**3)

        rl_batch = make_synthetic_rl_batch(
            tokenizer, train_batch, train_seq_len, device=device, seed=bench.seed,
        )

        # Signal that model is loaded, wait for vLLM to be ready
        conn.send({
            "type": "LOADED",
            "rank": rank,
            "model_load_ms": model_load_ms,
            "mem_after_load_gb": mem_after_load,
        })

        # Wait for signal to start training (after vLLM is loaded)
        msg = conn.recv()
        if msg == "TRAIN":
            # Warmup
            oom = False
            oom_error = ""
            train_times = []
            loss = 0.0

            try:
                for _ in range(warmup_rounds):
                    train_step_fsdp(fsdp_model, optimizer, rl_batch)
                torch.cuda.synchronize()

                # Measure
                for _ in range(measure_rounds):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    loss, step_ms = train_step_fsdp(fsdp_model, optimizer, rl_batch)
                    torch.cuda.synchronize()
                    train_times.append(step_ms)

            except torch.cuda.OutOfMemoryError as e:
                oom = True
                oom_error = str(e)[:500]
                torch.cuda.empty_cache()

            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

            conn.send({
                "type": "TRAIN_DONE",
                "rank": rank,
                "oom": oom,
                "oom_error": oom_error,
                "loss": float(loss) if not oom else 0.0,
                "mean_train_ms": sum(train_times) / len(train_times) if train_times else 0.0,
                "train_times": train_times,
                "peak_mem_gb": peak_mem,
            })

        # Wait for exit
        msg = conn.recv()
        dist.destroy_process_group()

    except torch.cuda.OutOfMemoryError as e:
        try:
            conn.send({
                "type": "TRAIN_DONE",
                "rank": rank,
                "oom": True,
                "oom_error": str(e)[:500],
                "loss": 0.0,
                "mean_train_ms": 0.0,
                "train_times": [],
                "peak_mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
            })
        except Exception:
            pass
    except Exception as exc:
        import traceback
        try:
            conn.send({"type": "ERROR", "rank": rank, "error": repr(exc),
                       "traceback": traceback.format_exc()})
        finally:
            raise


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Coexistence (No Switch) Benchmark")
    parser.add_argument("--model", default="qwen3-4b", choices=["qwen3-4b", "qwen3-32b"])
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--measure-rounds", type=int, default=5)
    parser.add_argument("--gpu-mem-util", type=float, default=0.0,
                        help="Override gpu_memory_utilization (0=auto)")
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    model_cfg = MODELS[args.model]
    workload = WORKLOADS[args.model]

    # Compute gpu_memory_utilization for coexistence
    if args.gpu_mem_util > 0:
        gpu_mem_util = args.gpu_mem_util
    else:
        gpu_mem_util = get_coexistence_gpu_mem_util(args.model, args.tp_size)

    print("=" * 70)
    print("  Approach 1: Coexistence (No Switch) Benchmark")
    print("=" * 70)
    print(f"  Model: {model_cfg['name']} | TP={args.tp_size}")
    print(f"  Training: FSDP, batch={workload.train_batch_size}, seq_len={workload.train_seq_len}")
    print(f"  Sampling: vLLM, prompts={workload.num_prompts}, max_tokens={workload.max_tokens}")
    print(f"  GPU mem util (vLLM): {gpu_mem_util:.2f}")
    print(f"  NOTE: Both runtimes share GPU memory. Training may OOM.")
    print("=" * 70, flush=True)

    ctx = mp.get_context("spawn")
    master_port = free_port()

    # ─── Phase 1: Spawn FSDP workers first ────────────────────────────────────
    print("\n  [Phase 1] Loading FSDP training model...", flush=True)
    conns = []
    workers = []

    for rank in range(args.tp_size):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_fsdp_coexist_worker,
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

    # Wait for FSDP model load
    load_msgs = []
    for conn in conns:
        msg = conn.recv()
        if msg.get("type") == "ERROR":
            print(f"  [ERROR] FSDP worker {msg.get('rank')}: {msg['error']}")
            sys.exit(1)
        load_msgs.append(msg)

    mean_fsdp_load_ms = sum(m["model_load_ms"] for m in load_msgs) / len(load_msgs)
    max_fsdp_mem = max(m["mem_after_load_gb"] for m in load_msgs)
    print(f"  [FSDP] loaded in {mean_fsdp_load_ms:.0f}ms, mem={max_fsdp_mem:.2f}GB per GPU")

    # ─── Phase 2: Load vLLM in main process (shares GPU with FSDP) ────────────
    print(f"\n  [Phase 2] Loading vLLM (gpu_mem_util={gpu_mem_util:.2f})...", flush=True)

    # Clear distributed env for vLLM
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        os.environ.pop(key, None)

    from vllm import LLM, SamplingParams

    vllm_oom = False
    vllm_error = ""
    llm = None
    vllm_load_ms = 0.0
    mean_throughput = 0.0
    mean_latency = 0.0
    peak_sampling_mem = 0.0

    try:
        vllm_start = time.perf_counter()
        llm = LLM(
            model=model_cfg["path"],
            dtype="bfloat16",
            tensor_parallel_size=args.tp_size,
            gpu_memory_utilization=gpu_mem_util,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=workload.max_model_len,
        )
        torch.cuda.synchronize()
        vllm_load_ms = (time.perf_counter() - vllm_start) * 1000
        print(f"  [vLLM] loaded in {vllm_load_ms:.0f}ms")

        prompts = get_prompts(workload.num_prompts)
        params = SamplingParams(
            max_tokens=workload.max_tokens, temperature=0.0, top_p=1.0, seed=42,
        )

        # Warmup
        for _ in range(args.warmup_rounds):
            llm.generate(prompts[:4], params)

        # Measure sampling
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
        print(f"  [vLLM] throughput={mean_throughput:.1f} tok/s, latency={mean_latency:.0f}ms")

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        vllm_oom = True
        vllm_error = str(e)[:500]
        print(f"  [vLLM] OOM during loading/sampling: {vllm_error[:200]}")

    # ─── Phase 3: Signal FSDP to train (with vLLM still in memory) ────────────
    print(f"\n  [Phase 3] FSDP Training (with vLLM resident)...", flush=True)
    for conn in conns:
        conn.send("TRAIN")

    train_msgs = []
    for conn in conns:
        msg = conn.recv()
        if msg.get("type") == "ERROR":
            print(f"  [ERROR] FSDP worker {msg.get('rank')}: {msg.get('error', '')}")
            train_msgs.append({
                "type": "TRAIN_DONE", "rank": msg.get("rank", -1),
                "oom": True, "oom_error": msg.get("error", ""),
                "loss": 0.0, "mean_train_ms": 0.0, "train_times": [],
                "peak_mem_gb": 0.0,
            })
        else:
            train_msgs.append(msg)

    train_msgs.sort(key=lambda m: m.get("rank", -1))

    # Check for training OOM
    training_oom = any(m.get("oom", False) for m in train_msgs)
    oom_errors = [m.get("oom_error", "") for m in train_msgs if m.get("oom")]

    if training_oom:
        print(f"  [FSDP] OOM! Training cannot coexist with vLLM at util={gpu_mem_util:.2f}")
        if oom_errors:
            print(f"  [FSDP] Error: {oom_errors[0][:200]}")
        mean_train_ms = 0.0
        peak_train_mem = 0.0
    else:
        mean_train_ms = sum(m["mean_train_ms"] for m in train_msgs) / len(train_msgs)
        peak_train_mem = max(m["peak_mem_gb"] for m in train_msgs)
        print(f"  [FSDP] train_step={mean_train_ms:.1f}ms, peak_mem={peak_train_mem:.2f}GB")

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

    # Cleanup vLLM
    if llm is not None:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    # ─── Build Result ─────────────────────────────────────────────────────────
    tokens_per_step = workload.train_batch_size * workload.train_seq_len
    train_throughput = tokens_per_step / (mean_train_ms / 1000) if mean_train_ms > 0 else 0.0

    oom_phase = ""
    if training_oom:
        oom_phase = "training"
    elif vllm_oom:
        oom_phase = "sampling"

    # Cold/hot computation
    all_train_times = train_msgs[0].get("train_times", []) if not training_oom else []
    cold_train_ms, hot_train_ms = compute_cold_hot(all_train_times)
    cold_samp_tput, hot_samp_tput = compute_cold_hot(throughputs if not vllm_oom else [])
    cold_samp_lat, hot_samp_lat = compute_cold_hot(latencies if not vllm_oom else [])

    # Cold/hot switch computation: coexistence has no mode switch.
    t2s_times = [0.0 for _ in range(args.measure_rounds)]
    s2t_times = [0.0 for _ in range(args.measure_rounds)]
    switch_metrics = switch_summary(t2s_times, s2t_times)

    result = BenchmarkResult(
        model=model_cfg["name"],
        model_path=model_cfg["path"],
        tp_size=args.tp_size,
        dp_size=1,
        approach="coexistence",
        approach_name="Coexistence (No Switch)",
        switch_s2t_ms=switch_metrics["switch_s2t_ms"],  # No switch needed
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
        sampling_throughputs=throughputs if not vllm_oom else [],
        sampling_latencies_ms=latencies if not vllm_oom else [],
        # Memory
        gpu_memory_utilization=gpu_mem_util,
        peak_memory_sampling_gb=peak_sampling_mem,
        peak_memory_training_gb=peak_train_mem,
        training_loaded_memory_gb=max_fsdp_mem,
        sampling_loaded_memory_gb=peak_sampling_mem,
        coexist_memory_pressure_gb=max_fsdp_mem + gpu_mem_util * 80.0,
        oom=training_oom or vllm_oom,
        oom_phase=oom_phase,
        error="; ".join(oom_errors) if oom_errors else vllm_error,
        notes=f"vLLM gpu_memory_utilization={gpu_mem_util:.2f}, "
              f"computed after reserving estimated FSDP base/training memory; no switch in coexistence",
        workload={
            "train_batch_size": workload.train_batch_size,
            "train_seq_len": workload.train_seq_len,
            "num_prompts": workload.num_prompts,
            "max_tokens": workload.max_tokens,
            "max_model_len": workload.max_model_len,
        },
        raw_metrics={
            "vllm_load_ms": vllm_load_ms,
            "fsdp_load_ms": mean_fsdp_load_ms,
            "fsdp_mem_after_load_gb": max_fsdp_mem,
            "vllm_oom": vllm_oom,
            "training_oom": training_oom,
        },
    )

    print(f"\n__RESULT_JSON__{json.dumps(result.to_dict())}")
    print("\nDONE")

    if args.output_dir:
        from pathlib import Path
        save_result(result, Path(args.output_dir))


if __name__ == "__main__":
    main()
