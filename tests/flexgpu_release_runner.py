#!/usr/bin/env python
# ruff: noqa: E402,I001
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any


BENCH_ROOT = os.environ.get("TUFT_FLEX_BENCH_ROOT", "/mnt/nas/hanzhang.yhz/lora_rl_bench")
TUFT_ROOT = os.environ.get("TUFT_FLEX_TUFT_ROOT", "/mnt/nas/hanzhang.yhz/flex_backend/TuFT")
sys.path.insert(0, str(os.path.join(TUFT_ROOT, "src")))
sys.path.insert(0, BENCH_ROOT)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
os.environ["PATH"] = "/usr/local/cuda-12.9/bin:" + os.environ.get("PATH", "")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmarks.s2_truezero_coordinator import _free_port, _make_ipc_desc_dict
from common.config import load_bench_config, load_model_spec
from common.fused_torchtp_utils import (
    apply_fused_torchtp_lora,
    create_fused_vllm_state_dict,
    fused_torchtp_train_step,
    load_fused_torchtp_model,
)
from common.model_utils import load_tokenizer
from common.training_utils import make_synthetic_rl_batch
from tuft.backends.flex.torchtp_zero_copy import inject_cuda_ipc_alias, prepare_cuda_ipc_alias_cache


def _prepare_ipc_tensor(
    key: str,
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    vocab_size: int,
) -> torch.Tensor:
    if key in ("model.embed_tokens.weight", "lm_head.weight"):
        vocab_chunk = vocab_size // world_size
        tensor = tensor[rank * vocab_chunk : (rank + 1) * vocab_chunk]
    return tensor.contiguous()


def _tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        int(tensor.data_ptr()),
        int(tensor.storage_offset()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


def _state_dict_signatures(
    state_dict: dict[str, torch.Tensor],
    rank: int,
    world_size: int,
    vocab_size: int,
) -> dict[str, tuple[Any, ...]]:
    return {
        key: _tensor_signature(_prepare_ipc_tensor(key, tensor, rank, world_size, vocab_size))
        for key, tensor in state_dict.items()
    }


def _storage_stability(
    before: dict[str, tuple[Any, ...]],
    after: dict[str, tuple[Any, ...]],
) -> tuple[bool, list[str]]:
    changed = [key for key, signature in before.items() if after.get(key) != signature]
    return not changed, changed[:10]


def _training_worker_release_runtime(
    rank: int,
    world_size: int,
    model_name: str,
    train_batch: int,
    train_seq_len: int,
    master_port: int,
    release_mode: str,
    descriptor_mode: str,
    conn,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.cuda.set_device(rank)

    keepalive: list[torch.Tensor] = []
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        spec = load_model_spec(model_name)
        bench = load_bench_config()
        tokenizer = load_tokenizer(spec)

        model, tp_group, _mesh = load_fused_torchtp_model(spec, rank, world_size)
        lora_params = apply_fused_torchtp_lora(model, bench, tp_group)
        optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
        batch = make_synthetic_rl_batch(
            tokenizer,
            train_batch,
            train_seq_len,
            device=torch.device(f"cuda:{rank}"),
            seed=bench.seed,
        )

        descriptor_reused = descriptor_mode == "reuse"
        storage_stable = True
        changed_storage_examples: list[str] = []
        descriptor_init_ms = 0.0
        ipc_desc: dict[str, Any] = {}
        vllm_sd: dict[str, Any] = {}
        if descriptor_reused:
            descriptor_start = time.perf_counter()
            initial_vllm_sd = create_fused_vllm_state_dict(model)
            initial_signatures = _state_dict_signatures(
                initial_vllm_sd,
                rank,
                world_size,
                spec.vocab_size,
            )
            ipc_desc, keepalive = _make_ipc_desc_dict(
                initial_vllm_sd,
                rank,
                world_size,
                spec.vocab_size,
            )
            torch.cuda.synchronize()
            descriptor_init_ms = (time.perf_counter() - descriptor_start) * 1000
            del initial_vllm_sd
        else:
            initial_signatures = {}

        fused_torchtp_train_step(model, optimizer, batch)
        torch.cuda.synchronize()
        train_start = time.perf_counter()
        loss, _ = fused_torchtp_train_step(model, optimizer, batch)
        torch.cuda.synchronize()
        train_ms = (time.perf_counter() - train_start) * 1000

        if descriptor_reused:
            ipc_start = time.perf_counter()
            final_vllm_sd = create_fused_vllm_state_dict(model)
            final_signatures = _state_dict_signatures(
                final_vllm_sd,
                rank,
                world_size,
                spec.vocab_size,
            )
            storage_stable, changed_storage_examples = _storage_stability(
                initial_signatures,
                final_signatures,
            )
            torch.cuda.synchronize()
            ipc_ms = (time.perf_counter() - ipc_start) * 1000
            del final_vllm_sd
        else:
            ipc_start = time.perf_counter()
            vllm_sd = create_fused_vllm_state_dict(model)
            ipc_desc, keepalive = _make_ipc_desc_dict(vllm_sd, rank, world_size, spec.vocab_size)
            torch.cuda.synchronize()
            ipc_ms = (time.perf_counter() - ipc_start) * 1000
        raw_gb = sum(t.numel() * t.element_size() for t in keepalive) / (1024**3)

        runtime_release_start = time.perf_counter()
        if descriptor_reused:
            del optimizer, lora_params, batch, tokenizer, model, tp_group
        else:
            del optimizer, lora_params, batch, tokenizer, vllm_sd, model, tp_group
        torch.cuda.empty_cache()
        if release_mode == "strict":
            torch.cuda.synchronize()
        runtime_release_ms = (time.perf_counter() - runtime_release_start) * 1000
        alloc_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)

        dist.destroy_process_group()

        conn.send(
            {
                "type": "READY",
                "rank": rank,
                "loss": float(loss),
                "train_ms": float(train_ms),
                "fuse_ipc_ms": float(ipc_ms),
                "descriptor_init_ms": float(descriptor_init_ms),
                "descriptor_reused": bool(descriptor_reused),
                "storage_stable": bool(storage_stable),
                "changed_storage_examples": changed_storage_examples,
                "runtime_release_ms": float(runtime_release_ms),
                "source_released": True,
                "raw_gb": float(raw_gb),
                "alloc_gb_after_release": float(alloc_gb),
                "reserved_gb_after_release": float(reserved_gb),
                "n_tensors": len(ipc_desc),
                "ipc_desc": ipc_desc,
            }
        )

        msg = conn.recv()
        if msg != "EXIT":
            conn.send({"type": "WARN", "rank": rank, "msg": f"unexpected command {msg!r}"})
        _ = keepalive
    except Exception as exc:
        try:
            conn.send({"type": "ERROR", "rank": rank, "error": repr(exc)})
        finally:
            raise


def _run_vllm_flex(
    model_path: str,
    tp_size: int,
    all_rank_descs: list[dict[str, Any]],
    gpu_memory_utilization: float,
    max_model_len: int,
    max_tokens: int,
    sample_rounds: int,
    verify_inject: bool,
) -> dict[str, Any]:
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
    ):
        os.environ.pop(key, None)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
    os.environ["PATH"] = "/usr/local/cuda-12.9/bin:" + os.environ.get("PATH", "")

    from vllm import LLM, SamplingParams

    print("[3/5] Creating vLLM dummy workers...", flush=True)
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
    dummy_ms = (time.perf_counter() - dummy_start) * 1000
    print(f"      dummy_load: {dummy_ms:.0f}ms (one-time)", flush=True)

    cache_start = time.perf_counter()
    cache_results = llm.collective_rpc(prepare_cuda_ipc_alias_cache)
    cache_ms = (time.perf_counter() - cache_start) * 1000
    print(f"      alias_cache_prepare: {cache_ms:.2f}ms results={cache_results}", flush=True)

    print("[4/5] Injecting CUDA IPC tensors via storage alias...", flush=True)
    inject_start = time.perf_counter()
    inject_results = llm.collective_rpc(
        inject_cuda_ipc_alias,
        args=(all_rank_descs, verify_inject),
    )
    inject_ms = (time.perf_counter() - inject_start) * 1000
    for item in inject_results:
        print(
            f"      rank {item['rank']}: injected={item['injected']} "
            f"verified={item['verified']} mismatch={item['mismatched']} "
            f"max_diff={item['max_diff']:.1e} skipped={item['skipped']}",
            flush=True,
        )
        if item["examples"]:
            print(f"        examples={item['examples']}", flush=True)
    print(f"      inject_alias: {inject_ms:.2f}ms", flush=True)

    prompts = ["What is machine learning?", "Explain quantum computing simply."]
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0, seed=42)
    _ = llm.generate(prompts[:1], params)
    throughputs = []
    latencies = []
    for idx in range(sample_rounds):
        start = time.perf_counter()
        outputs = llm.generate(prompts, params)
        latency = (time.perf_counter() - start) * 1000
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = total_tokens / (latency / 1000)
        throughputs.append(throughput)
        latencies.append(latency)
        print(
            f"      round {idx + 1}: {throughput:.1f} tok/s, "
            f"{latency:.0f}ms, {total_tokens} tokens",
            flush=True,
        )

    return {
        "dummy_ms": dummy_ms,
        "alias_cache_ms": cache_ms,
        "inject_ms": inject_ms,
        "inject_results": inject_results,
        "throughputs": throughputs,
        "latencies": latencies,
    }


def _mean(items: list[dict[str, Any]], key: str) -> float:
    return sum(float(item[key]) for item in items) / max(len(items), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-4b", choices=["qwen3-4b", "qwen3-32b"])
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--train-batch", type=int, default=2)
    parser.add_argument("--train-seq-len", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--sample-rounds", type=int, default=3)
    parser.add_argument("--release-mode", choices=["strict", "fast"], default="fast")
    parser.add_argument("--descriptor-mode", choices=["fresh", "reuse"], default="fresh")
    parser.add_argument("--verify-inject", action="store_true")
    args = parser.parse_args()

    spec = load_model_spec(args.model)
    ctx = mp.get_context("spawn")
    master_port = _free_port()

    print("=== FlexGPU Released-Runtime True-Zero Benchmark ===", flush=True)
    print(
        f"  model={spec.name}, tp={args.tp_size}, train_batch={args.train_batch}, "
        f"seq={args.train_seq_len}, max_tokens={args.max_tokens}, verify={args.verify_inject}",
        flush=True,
    )

    conns = []
    workers = []
    for rank in range(args.tp_size):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_training_worker_release_runtime,
            args=(
                rank,
                args.tp_size,
                args.model,
                args.train_batch,
                args.train_seq_len,
                master_port,
                args.release_mode,
                args.descriptor_mode,
                child_conn,
            ),
        )
        proc.start()
        conns.append(parent_conn)
        workers.append(proc)

    ready_msgs = []
    for conn in conns:
        msg = conn.recv()
        if msg.get("type") != "READY":
            raise RuntimeError(f"Training worker failed: {msg}")
        ready_msgs.append(msg)
    ready_msgs.sort(key=lambda item: item["rank"])

    all_rank_descs = [msg["ipc_desc"] for msg in ready_msgs]
    mean_train = _mean(ready_msgs, "train_ms")
    mean_ipc = _mean(ready_msgs, "fuse_ipc_ms")
    mean_descriptor_init = _mean(ready_msgs, "descriptor_init_ms")
    mean_release = _mean(ready_msgs, "runtime_release_ms")
    source_released = all(bool(msg.get("source_released")) for msg in ready_msgs)
    descriptor_reused = all(bool(msg.get("descriptor_reused")) for msg in ready_msgs)
    storage_stable = all(bool(msg.get("storage_stable")) for msg in ready_msgs)
    changed_examples = [
        example for msg in ready_msgs for example in msg.get("changed_storage_examples", [])
    ]
    print(
        f"[2/5] Training runtime released: mean_train={mean_train:.1f}ms, "
        f"mean_ipc_desc={mean_ipc:.2f}ms, descriptor_init={mean_descriptor_init:.2f}ms, "
        f"mean_release={mean_release:.2f}ms, source_released={source_released}, "
        f"descriptor_reused={descriptor_reused}, storage_stable={storage_stable}",
        flush=True,
    )
    for msg in ready_msgs:
        print(
            f"      rank {msg['rank']}: loss={msg['loss']:.4f}, train={msg['train_ms']:.1f}ms, "
            f"ipc={msg['fuse_ipc_ms']:.2f}ms, desc_init={msg['descriptor_init_ms']:.2f}ms, "
            f"release={msg['runtime_release_ms']:.2f}ms, "
            f"alloc_after_release={msg['alloc_gb_after_release']:.2f}GB, "
            f"raw={msg['raw_gb']:.2f}GB, tensors={msg['n_tensors']}",
            flush=True,
        )
    if changed_examples:
        print(f"      changed_storage_examples={changed_examples[:10]}", flush=True)

    vllm_result = None
    try:
        vllm_result = _run_vllm_flex(
            model_path=spec.path,
            tp_size=args.tp_size,
            all_rank_descs=all_rank_descs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            sample_rounds=args.sample_rounds,
            verify_inject=args.verify_inject,
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
        raise RuntimeError("vLLM run failed before producing results")

    all_verified = all(item["mismatched"] == 0 for item in vllm_result["inject_results"])
    mean_tput = sum(vllm_result["throughputs"]) / len(vllm_result["throughputs"])
    mean_latency = sum(vllm_result["latencies"]) / len(vllm_result["latencies"])
    switch_core_ms = mean_ipc + mean_release + vllm_result["inject_ms"]

    print("\n=== FlexGPU Summary ===", flush=True)
    print(f"  mean_train_ms: {mean_train:.1f}", flush=True)
    print(f"  mean_ipc_desc_ms: {mean_ipc:.2f}", flush=True)
    print(f"  descriptor_init_ms(one-time): {mean_descriptor_init:.2f}", flush=True)
    print(f"  mean_runtime_release_ms: {mean_release:.2f}", flush=True)
    print(f"  alias_cache_ms(one-time): {vllm_result['alias_cache_ms']:.2f}", flush=True)
    print(f"  inject_alias_ms: {vllm_result['inject_ms']:.2f}", flush=True)
    print(f"  switch_core_ms: {switch_core_ms:.2f}", flush=True)
    print(f"  dummy_load_ms(one-time): {vllm_result['dummy_ms']:.0f}", flush=True)
    print(f"  sampling_throughput: {mean_tput:.1f} tok/s", flush=True)
    print(f"  sampling_latency_ms: {mean_latency:.0f}", flush=True)
    print(f"  readback_exact: {all_verified}", flush=True)
    print(f"  source_runtime_released: {source_released}", flush=True)
    print(f"  descriptor_reused: {descriptor_reused}", flush=True)
    print(f"  storage_stable: {storage_stable}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
