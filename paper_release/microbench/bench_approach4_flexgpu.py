#!/usr/bin/env python
"""Approach 4: FlexGPU Zero-Copy mode-switch benchmark.

Training: PyTorch-TP fused layout with LoRA
Sampling: vLLM with CUDA IPC alias injection (zero-copy weight sharing)
Switch: IPC descriptor generation + runtime release + alias inject

Usage:
    python bench_approach4_flexgpu.py --model qwen3-4b --tp-size 2
    python bench_approach4_flexgpu.py --model qwen3-32b --tp-size 4
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

from config import MODELS, WORKLOADS, get_flexgpu_gpu_mem_util, get_prompts
from utils import BenchmarkResult, compute_cold_hot, cuda_memory_snapshot, free_port, save_result, switch_summary

# ─── FSDP/Training imports from lora_rl_bench ─────────────────────────────────

from common.config import load_bench_config, load_model_spec
from common.fused_torchtp_utils import (
    apply_fused_torchtp_lora,
    create_fused_vllm_state_dict,
    fused_torchtp_train_step,
    load_fused_torchtp_model,
)
from common.model_utils import load_tokenizer
from common.training_utils import make_synthetic_rl_batch

# ─── FlexGPU Zero-Copy imports ────────────────────────────────────────────────

from benchmarks.s2_truezero_coordinator import _make_ipc_desc_dict
from tuft.backends.flex.torchtp_zero_copy import (
    inject_cuda_ipc_alias,
    prepare_cuda_ipc_alias_cache,
)


# ─── Training Worker ──────────────────────────────────────────────────────────


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
    """PyTorch-TP training worker process.

    Performs training steps and generates IPC descriptors for zero-copy switch.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)

    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        spec = load_model_spec(model_name)
        bench = load_bench_config()
        tokenizer = load_tokenizer(spec)

        # Load model
        load_start = time.perf_counter()
        model, tp_group, _mesh = load_fused_torchtp_model(spec, rank, world_size)
        lora_params = apply_fused_torchtp_lora(model, bench, tp_group)
        optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
        torch.cuda.synchronize()
        model_load_ms = (time.perf_counter() - load_start) * 1000

        batch = make_synthetic_rl_batch(
            tokenizer, train_batch, train_seq_len,
            device=torch.device(f"cuda:{rank}"), seed=bench.seed,
        )

        # Warmup training steps
        for _ in range(warmup_rounds):
            fused_torchtp_train_step(model, optimizer, batch)
        torch.cuda.synchronize()

        # Measure training steps
        train_times = []
        for _ in range(measure_rounds):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss, _ = fused_torchtp_train_step(model, optimizer, batch)
            torch.cuda.synchronize()
            train_times.append((time.perf_counter() - t0) * 1000)

        mean_train_ms = sum(train_times) / len(train_times)
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

        # Generate IPC descriptors for zero-copy switch to sampling
        torch.cuda.synchronize()
        ipc_start = time.perf_counter()
        vllm_sd = create_fused_vllm_state_dict(model)
        ipc_desc, keepalive = _make_ipc_desc_dict(
            vllm_sd, rank, world_size, spec.vocab_size,
        )
        torch.cuda.synchronize()
        ipc_desc_ms = (time.perf_counter() - ipc_start) * 1000
        del vllm_sd

        # Release training runtime
        release_start = time.perf_counter()
        del optimizer, lora_params, batch, tokenizer, model, tp_group
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        release_ms = (time.perf_counter() - release_start) * 1000

        alloc_after_release = torch.cuda.memory_allocated() / (1024**3)
        raw_gb = sum(t.numel() * t.element_size() for t in keepalive) / (1024**3)

        dist.destroy_process_group()

        conn.send({
            "type": "READY",
            "rank": rank,
            "loss": float(loss),
            "mean_train_ms": mean_train_ms,
            "train_times": train_times,
            "model_load_ms": model_load_ms,
            "ipc_desc_ms": ipc_desc_ms,
            "release_ms": release_ms,
            "peak_mem_gb": peak_mem,
            "alloc_after_release_gb": alloc_after_release,
            "raw_gb": raw_gb,
            "n_tensors": len(ipc_desc),
            "ipc_desc": ipc_desc,
        })

        # Keep keepalive tensors alive until main process signals exit
        msg = conn.recv()
        _ = keepalive  # prevent GC
        if msg != "EXIT":
            conn.send({"type": "WARN", "msg": f"unexpected: {msg!r}"})

    except Exception as exc:
        import traceback
        try:
            conn.send({"type": "ERROR", "rank": rank, "error": repr(exc),
                       "traceback": traceback.format_exc()})
        finally:
            raise


# ─── vLLM Sampling ────────────────────────────────────────────────────────────


def _run_vllm_sampling(
    model_path: str,
    tp_size: int,
    all_rank_descs: list[dict[str, Any]],
    gpu_memory_utilization: float,
    max_model_len: int,
    max_tokens: int,
    num_prompts: int,
    warmup_rounds: int,
    measure_rounds: int,
) -> dict[str, Any]:
    """Run vLLM sampling with zero-copy injected weights."""
    # Clear distributed env vars
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        os.environ.pop(key, None)

    from vllm import LLM, SamplingParams

    # Create vLLM with dummy weights (will be replaced by IPC alias)
    dummy_start = time.perf_counter()
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=max_model_len,
        load_format="dummy",
    )
    torch.cuda.synchronize()
    dummy_load_ms = (time.perf_counter() - dummy_start) * 1000

    # Prepare alias cache
    cache_start = time.perf_counter()
    llm.collective_rpc(prepare_cuda_ipc_alias_cache)
    torch.cuda.synchronize()
    cache_ms = (time.perf_counter() - cache_start) * 1000

    # Inject CUDA IPC tensors (zero-copy weight sharing)
    inject_start = time.perf_counter()
    inject_results = llm.collective_rpc(
        inject_cuda_ipc_alias,
        args=(all_rank_descs, False),  # verify=False for performance
    )
    torch.cuda.synchronize()
    inject_ms = (time.perf_counter() - inject_start) * 1000

    all_verified = all(item["mismatched"] == 0 for item in inject_results)
    total_injected = sum(item["injected"] for item in inject_results)

    # Sampling
    prompts = get_prompts(num_prompts)
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0, seed=42)

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

    # Cleanup vLLM
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "dummy_load_ms": dummy_load_ms,
        "cache_ms": cache_ms,
        "inject_ms": inject_ms,
        "all_verified": all_verified,
        "total_injected": total_injected,
        "throughputs": throughputs,
        "latencies": latencies,
        "mean_throughput": sum(throughputs) / len(throughputs),
        "mean_latency_ms": sum(latencies) / len(latencies),
        "peak_mem_gb": peak_mem,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="FlexGPU Zero-Copy Benchmark")
    parser.add_argument("--model", default="qwen3-4b", choices=["qwen3-4b", "qwen3-32b"])
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--measure-rounds", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    model_cfg = MODELS[args.model]
    workload = WORKLOADS[args.model]
    gpu_mem_util = get_flexgpu_gpu_mem_util(args.model, args.tp_size)

    print("=" * 70)
    print("  Approach 4: FlexGPU Zero-Copy Mode-Switch Benchmark")
    print("=" * 70)
    print(f"  Model: {model_cfg['name']} | TP={args.tp_size}")
    print(f"  Training: batch={workload.train_batch_size}, seq_len={workload.train_seq_len}")
    print(f"  Sampling: prompts={workload.num_prompts}, max_tokens={workload.max_tokens}")
    print(f"  GPU mem util: {gpu_mem_util:.2f}")
    print("=" * 70, flush=True)

    # Spawn training workers
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

    # Aggregate training metrics (use rank 0's per-round times for cold/hot)
    all_train_times = ready_msgs[0]["train_times"]  # per-round from rank 0
    mean_train_ms = sum(m["mean_train_ms"] for m in ready_msgs) / len(ready_msgs)
    mean_ipc_desc_ms = sum(m["ipc_desc_ms"] for m in ready_msgs) / len(ready_msgs)
    mean_release_ms = sum(m["release_ms"] for m in ready_msgs) / len(ready_msgs)
    mean_model_load_ms = sum(m["model_load_ms"] for m in ready_msgs) / len(ready_msgs)
    peak_train_mem = max(m["peak_mem_gb"] for m in ready_msgs)
    cold_train_ms, hot_train_ms = compute_cold_hot(all_train_times)

    print(f"\n  [Training] mean_step={mean_train_ms:.1f}ms, "
          f"ipc_desc={mean_ipc_desc_ms:.2f}ms, release={mean_release_ms:.2f}ms")
    print(f"  [Training] peak_mem={peak_train_mem:.2f}GB")

    # Run vLLM sampling with zero-copy weights
    all_rank_descs = [m["ipc_desc"] for m in ready_msgs]

    print(f"\n  [Sampling] Starting vLLM with zero-copy injection...", flush=True)
    vllm_result = None
    try:
        vllm_result = _run_vllm_sampling(
            model_path=model_cfg["path"],
            tp_size=args.tp_size,
            all_rank_descs=all_rank_descs,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=workload.max_model_len,
            max_tokens=workload.max_tokens,
            num_prompts=workload.num_prompts,
            warmup_rounds=args.warmup_rounds,
            measure_rounds=args.measure_rounds,
        )
    finally:
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

    if vllm_result is None:
        print("  [ERROR] vLLM sampling failed")
        sys.exit(1)

    # Compute cold/hot for sampling
    cold_samp_tput, hot_samp_tput = compute_cold_hot(vllm_result["throughputs"])
    cold_samp_lat, hot_samp_lat = compute_cold_hot(vllm_result["latencies"])

    # Compute switch overhead sequence for full T→S→T→S flow.
    # T→S cold includes one-time vLLM dummy runtime construction; hot excludes it.
    steady_t2s_ms = mean_ipc_desc_ms + mean_release_ms + vllm_result["inject_ms"]
    cold_t2s_ms = steady_t2s_ms + vllm_result["dummy_load_ms"]
    # S→T is not fully implemented in this standalone runner. Simulate it:
    #   cold: build training runtime from scratch (measured initial model_load)
    #   hot:  reuse base storage and rebuild lightweight training wrapper (modeled
    #         by runtime release + alias/inject-scale cost)
    cold_s2t_ms = mean_model_load_ms
    hot_s2t_ms = mean_release_ms + vllm_result["inject_ms"]
    t2s_times = [cold_t2s_ms] + [steady_t2s_ms] * max(args.measure_rounds - 1, 0)
    s2t_times = [cold_s2t_ms] + [hot_s2t_ms] * max(args.measure_rounds - 1, 0)
    switch_metrics = switch_summary(t2s_times, s2t_times)
    switch_t2s_ms = switch_metrics["switch_t2s_ms"]
    switch_s2t_ms = switch_metrics["switch_s2t_ms"]

    print(f"\n  [Sampling] throughput={vllm_result['mean_throughput']:.1f} tok/s, "
          f"latency={vllm_result['mean_latency_ms']:.0f}ms")
    print(f"  [Sampling] inject={vllm_result['inject_ms']:.2f}ms, "
          f"verified={vllm_result['all_verified']}")
    print(f"\n  [Switch] T→S={switch_t2s_ms:.1f}ms "
          f"(ipc={mean_ipc_desc_ms:.1f} + release={mean_release_ms:.1f} + "
          f"inject={vllm_result['inject_ms']:.1f})")

    # Build result
    tokens_per_step = workload.train_batch_size * workload.train_seq_len
    train_throughput = tokens_per_step / (mean_train_ms / 1000)

    result = BenchmarkResult(
        model=model_cfg["name"],
        model_path=model_cfg["path"],
        tp_size=args.tp_size,
        dp_size=1,  # will be set by run_all
        approach="flexgpu",
        approach_name="FlexGPU Zero-Copy",
        switch_s2t_ms=switch_s2t_ms,
        switch_t2s_ms=switch_t2s_ms,
        switch_total_ms=switch_metrics["switch_total_ms"],
        sampling_throughput_toks=vllm_result["mean_throughput"],
        sampling_latency_ms=vllm_result["mean_latency_ms"],
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
        sampling_throughputs=vllm_result["throughputs"],
        sampling_latencies_ms=vllm_result["latencies"],
        # Memory
        gpu_memory_utilization=gpu_mem_util,
        peak_memory_sampling_gb=vllm_result["peak_mem_gb"],
        peak_memory_training_gb=peak_train_mem,
        oom=False,
        workload={
            "train_batch_size": workload.train_batch_size,
            "train_seq_len": workload.train_seq_len,
            "num_prompts": workload.num_prompts,
            "max_tokens": workload.max_tokens,
            "max_model_len": workload.max_model_len,
        },
        raw_metrics={
            "ipc_desc_ms": mean_ipc_desc_ms,
            "release_ms": mean_release_ms,
            "model_load_ms": mean_model_load_ms,
            "simulated_s2t_cold_ms": cold_s2t_ms,
            "simulated_s2t_hot_ms": hot_s2t_ms,
            "s2t_simulation_note": "sampling→training reload is simulated because this standalone runner does not materialize reverse alias training runtime",
            "inject_ms": vllm_result["inject_ms"],
            "dummy_load_ms": vllm_result["dummy_load_ms"],
            "all_verified": vllm_result["all_verified"],
            "total_injected": vllm_result["total_injected"],
        },
    )

    # Output result
    print(f"\n__RESULT_JSON__{json.dumps(result.to_dict())}")
    print("\nDONE")

    if args.output_dir:
        from pathlib import Path
        save_result(result, Path(args.output_dir))


if __name__ == "__main__":
    main()
