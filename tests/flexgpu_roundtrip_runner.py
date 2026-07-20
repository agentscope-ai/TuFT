#!/usr/bin/env python
# ruff: noqa: E402,I001
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from types import MethodType
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
from torch import nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from common.config import load_bench_config, load_model_spec
from common.fused_torchtp_utils import (
    FusedGateUpLinear,
    FusedQKVLinear,
    FusedQKVLoRA,
    RowwiseLoRALinear,
    apply_fused_torchtp_lora,
    fused_attention_forward,
    fused_mlp_forward,
    fused_torchtp_train_step,
    load_fused_torchtp_model,
)
from common.model_utils import load_tokenizer
from common.training_utils import make_synthetic_rl_batch
from flexgpu_release_runner import _free_port, _run_vllm_flex  # pyright: ignore[reportAttributeAccessIssue]
from tuft.backends.flex.torchtp_zero_copy import (
    create_fused_vllm_state_dict,
    tensor_to_cuda_ipc_descriptor,
)


def _cuda_snapshot() -> dict[str, float]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "free_gb": free_bytes / (1024**3),
        "total_gb": total_bytes / (1024**3),
    }


def _init_pg(rank: int, world_size: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


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


def _make_ipc_desc_and_keepalive_by_key(
    state_dict: dict[str, torch.Tensor],
    rank: int,
    world_size: int,
    vocab_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, torch.Tensor]]:
    descriptors: dict[str, dict[str, Any]] = {}
    keepalive_by_key: dict[str, torch.Tensor] = {}
    for key, original_tensor in state_dict.items():
        desc_tensor = _prepare_ipc_tensor(key, original_tensor, rank, world_size, vocab_size)
        if not desc_tensor.is_cuda:
            raise RuntimeError(f"{key} is not a CUDA tensor")
        keepalive_by_key[key] = original_tensor.contiguous()
        descriptors[key] = {
            "shape": tuple(desc_tensor.shape),
            "dtype": str(desc_tensor.dtype),
            "ipc": tensor_to_cuda_ipc_descriptor(desc_tensor),
        }
    return descriptors, keepalive_by_key


def _weight_target_map(model: Any) -> dict[str, tuple[Any, str, Any]]:
    targets: dict[str, tuple[Any, str, Any]] = {}
    for index, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        mlp = layer.mlp
        prefix = f"model.layers.{index}"
        qkv_base = attn.qkv_proj.base if hasattr(attn.qkv_proj, "base") else attn.qkv_proj
        o_base = attn.o_proj.base if hasattr(attn.o_proj, "base") else attn.o_proj
        targets[f"{prefix}.self_attn.qkv_proj.weight"] = (qkv_base, "weight", qkv_base.weight)
        targets[f"{prefix}.self_attn.o_proj.weight"] = (o_base, "weight", o_base.weight)
        targets[f"{prefix}.mlp.gate_up_proj.weight"] = (
            mlp.gate_up_proj,
            "weight",
            mlp.gate_up_proj.weight,
        )
        targets[f"{prefix}.mlp.down_proj.weight"] = (mlp.down_proj, "weight", mlp.down_proj.weight)
        targets[f"{prefix}.input_layernorm.weight"] = (
            layer.input_layernorm,
            "weight",
            layer.input_layernorm.weight,
        )
        targets[f"{prefix}.post_attention_layernorm.weight"] = (
            layer.post_attention_layernorm,
            "weight",
            layer.post_attention_layernorm.weight,
        )
        if hasattr(attn, "q_norm"):
            targets[f"{prefix}.self_attn.q_norm.weight"] = (
                attn.q_norm,
                "weight",
                attn.q_norm.weight,
            )
        if hasattr(attn, "k_norm"):
            targets[f"{prefix}.self_attn.k_norm.weight"] = (
                attn.k_norm,
                "weight",
                attn.k_norm.weight,
            )
    targets["model.embed_tokens.weight"] = (
        model.model.embed_tokens,
        "weight",
        model.model.embed_tokens.weight,
    )
    targets["model.norm.weight"] = (model.model.norm, "weight", model.model.norm.weight)
    if (
        hasattr(model, "lm_head")
        and model.lm_head.weight is not None
        and not getattr(model.config, "tie_word_embeddings", False)
    ):
        targets["lm_head.weight"] = (model.lm_head, "weight", model.lm_head.weight)
    return targets


def _local_tensor_ref(value: Any) -> torch.Tensor:
    if hasattr(value, "_local_tensor"):
        return value._local_tensor
    if hasattr(value, "to_local"):
        return value.to_local()
    return value


def _set_local_tensor_data(owner: Any, attr: str, value: Any, source: torch.Tensor) -> torch.Tensor:
    if hasattr(value, "_local_tensor"):
        value._local_tensor = source
        value.requires_grad_(False)
        return value._local_tensor
    parameter = nn.Parameter(source, requires_grad=False)
    setattr(owner, attr, parameter)
    return parameter


def _alias_training_model_to_keepalive(
    model: Any,
    keepalive_by_key: dict[str, torch.Tensor],
    rank: int,
    world_size: int,
    vocab_size: int,
) -> dict[str, Any]:
    targets = _weight_target_map(model)
    aliased = 0
    skipped = 0
    mismatched = 0
    examples: list[Any] = []
    for key, (owner, attr, target) in targets.items():
        source = keepalive_by_key.get(key)
        if source is None:
            skipped += 1
            continue
        updated = _set_local_tensor_data(owner, attr, target, source)
        aliased += 1
        if int(_local_tensor_ref(updated).data_ptr()) != int(source.data_ptr()):
            mismatched += 1
            if len(examples) < 5:
                examples.append(
                    (key, int(_local_tensor_ref(updated).data_ptr()), int(source.data_ptr()), "ptr")
                )
    return {"aliased": aliased, "skipped": skipped, "mismatched": mismatched, "examples": examples}


def _build_empty_fused_torchtp_model(spec: Any, world_size: int) -> tuple[Any, Any, Any]:
    dtype = torch.bfloat16 if spec.dtype == "bfloat16" else torch.float16
    config = AutoConfig.from_pretrained(spec.path, trust_remote_code=True)
    config.use_cache = False
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        for layer in model.model.layers:
            attn = layer.self_attn
            mlp = layer.mlp
            attn.q_size = attn.q_proj.out_features
            attn.kv_size = attn.k_proj.out_features
            attn.qkv_proj = FusedQKVLinear(
                attn.q_proj,
                attn.k_proj,
                attn.v_proj,
                tp_size=world_size,
            )
            del attn.q_proj, attn.k_proj, attn.v_proj
            attn.forward = MethodType(fused_attention_forward, attn)
            mlp.gate_size = mlp.gate_proj.out_features
            mlp.up_size = mlp.up_proj.out_features
            mlp.gate_up_proj = FusedGateUpLinear(mlp.gate_proj, mlp.up_proj, tp_size=world_size)
            del mlp.gate_proj, mlp.up_proj
            mlp.forward = MethodType(fused_mlp_forward, mlp)
    tp_group = dist.group.WORLD
    mesh = None
    num_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
    model.config.num_attention_heads = num_heads // world_size
    model.config.num_key_value_heads = num_kv_heads // world_size
    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        attn.q_size = attn.q_size // world_size
        attn.kv_size = attn.kv_size // world_size
        mlp.gate_size = mlp.gate_size // world_size
        mlp.up_size = mlp.up_size // world_size
    return model, tp_group, mesh


def _materialize_remaining_meta_tensors(model: Any) -> dict[str, Any]:
    materialized = 0
    examples: list[str] = []
    for module_name, module in model.named_modules():
        for name, param in list(module._parameters.items()):
            if param is None or not getattr(param, "is_meta", False):
                continue
            replacement = nn.Parameter(
                torch.zeros(
                    tuple(param.shape),
                    dtype=param.dtype,
                    device=torch.cuda.current_device(),
                ),
                requires_grad=False,
            )
            module._parameters[name] = replacement
            materialized += 1
            if len(examples) < 10:
                examples.append(f"{module_name}.{name}")
        for name, buffer in list(module._buffers.items()):
            if buffer is None or not getattr(buffer, "is_meta", False):
                continue
            module._buffers[name] = torch.zeros(
                tuple(buffer.shape),
                dtype=buffer.dtype,
                device=torch.cuda.current_device(),
            )
            materialized += 1
            if len(examples) < 10:
                examples.append(f"{module_name}.{name}")
    return {"materialized_meta_tensors": materialized, "examples": examples}


def _apply_local_fused_lora(model: Any, bench: Any, tp_group: Any) -> list[Any]:
    params = []
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.qkv_proj = FusedQKVLoRA(
            attn.qkv_proj,
            rank=bench.lora_rank,
            alpha=bench.lora_alpha,
            q_out=attn.q_size,
            k_out=attn.kv_size,
            v_out=attn.kv_size,
            tp_group=tp_group,
        )
        o_local_in = attn.o_proj.weight.shape[1]
        o_out = attn.o_proj.weight.shape[0]
        attn.o_proj = RowwiseLoRALinear(
            attn.o_proj,
            rank=bench.lora_rank,
            alpha=bench.lora_alpha,
            local_in_features=o_local_in,
            out_features=o_out,
            tp_group=tp_group,
        )
        params.extend([param for param in attn.qkv_proj.parameters() if param.requires_grad])
        params.extend([attn.o_proj.lora_A, attn.o_proj.lora_B])
    return params


def _build_training_runtime(
    model_name: str,
    rank: int,
    world_size: int,
    train_batch: int,
    train_seq_len: int,
    keepalive_by_key: dict[str, torch.Tensor] | None,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, dict[str, Any]]:
    spec = load_model_spec(model_name)
    bench = load_bench_config()
    tokenizer = load_tokenizer(spec)
    load_start = time.perf_counter()
    if keepalive_by_key is None:
        model, tp_group, mesh = load_fused_torchtp_model(spec, rank, world_size)
    else:
        model, tp_group, mesh = _build_empty_fused_torchtp_model(spec, world_size)
    load_ms = (time.perf_counter() - load_start) * 1000
    alias_ms = 0.0
    alias_result = {"aliased": 0, "skipped": 0, "mismatched": 0, "examples": []}
    if keepalive_by_key is not None:
        alias_start = time.perf_counter()
        alias_result = _alias_training_model_to_keepalive(
            model,
            keepalive_by_key,
            rank,
            world_size,
            spec.vocab_size,
        )
        materialize_result = _materialize_remaining_meta_tensors(model)
        alias_result["materialized_meta_tensors"] = materialize_result["materialized_meta_tensors"]
        alias_result["materialized_meta_examples"] = materialize_result["examples"]
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        alias_ms = (time.perf_counter() - alias_start) * 1000
        for param in model.parameters():
            param.requires_grad_(False)
        lora_params = _apply_local_fused_lora(model, bench, tp_group)
    else:
        lora_params = apply_fused_torchtp_lora(model, bench, tp_group)
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
    batch = make_synthetic_rl_batch(
        tokenizer,
        train_batch,
        train_seq_len,
        device=torch.device(f"cuda:{rank}"),
        seed=bench.seed,
    )
    metrics = {
        "load_ms": load_ms,
        "alias_ms": alias_ms,
        "alias_result": alias_result,
        "memory_after_build": _cuda_snapshot(),
    }
    return spec, tokenizer, model, tp_group, mesh, optimizer, lora_params, batch, metrics


def _release_training_runtime(
    model: Any,
    tp_group: Any,
    mesh: Any,
    optimizer: Any,
    lora_params: Any,
    batch: Any,
    tokenizer: Any,
    release_mode: str,
) -> tuple[float, dict[str, float]]:
    start = time.perf_counter()
    del optimizer, lora_params, batch, tokenizer, model, tp_group, mesh
    gc.collect()
    torch.cuda.empty_cache()
    if release_mode == "strict":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000, _cuda_snapshot()


def _training_worker_roundtrip(
    rank: int,
    world_size: int,
    model_name: str,
    train_batch: int,
    train_seq_len: int,
    release_mode: str,
    descriptor_mode: str,
    initial_master_port: int,
    conn,
) -> None:
    torch.cuda.set_device(rank)
    keepalive_by_key: dict[str, torch.Tensor] | None = None
    ipc_desc: dict[str, dict[str, Any]] | None = None
    initial_signatures: dict[str, tuple[Any, ...]] | None = None
    try:
        master_port = initial_master_port
        round_index = 0
        while True:
            _init_pg(rank, world_size, master_port)
            spec, tokenizer, model, tp_group, mesh, optimizer, lora_params, batch, build_metrics = (
                _build_training_runtime(
                    model_name,
                    rank,
                    world_size,
                    train_batch,
                    train_seq_len,
                    keepalive_by_key,
                )
            )
            fused_torchtp_train_step(model, optimizer, batch)
            torch.cuda.synchronize()
            train_start = time.perf_counter()
            loss, _ = fused_torchtp_train_step(model, optimizer, batch)
            torch.cuda.synchronize()
            train_ms = (time.perf_counter() - train_start) * 1000

            descriptor_reused = descriptor_mode == "reuse" and ipc_desc is not None
            storage_stable = True
            changed_storage_examples: list[str] = []
            descriptor_init_ms = 0.0
            ipc_start = time.perf_counter()
            current_state_dict = create_fused_vllm_state_dict(model)
            current_signatures = _state_dict_signatures(
                current_state_dict,
                rank,
                world_size,
                spec.vocab_size,
            )
            if descriptor_reused and initial_signatures is not None:
                alias_result = build_metrics.get("alias_result", {})
                storage_stable = bool(alias_result.get("aliased", 0)) and not bool(
                    alias_result.get("mismatched", 0)
                )
                changed_storage_examples = list(alias_result.get("examples", []))[:10]
            else:
                ipc_desc, keepalive_by_key = _make_ipc_desc_and_keepalive_by_key(
                    current_state_dict,
                    rank,
                    world_size,
                    spec.vocab_size,
                )
                initial_signatures = current_signatures
                descriptor_init_ms = (time.perf_counter() - ipc_start) * 1000
            torch.cuda.synchronize()
            ipc_ms = (time.perf_counter() - ipc_start) * 1000
            raw_gb = (
                sum(t.numel() * t.element_size() for t in keepalive_by_key.values()) / (1024**3)
                if keepalive_by_key is not None
                else 0.0
            )
            release_ms, release_memory = _release_training_runtime(
                model,
                tp_group,
                mesh,
                optimizer,
                lora_params,
                batch,
                tokenizer,
                release_mode,
            )
            dist.destroy_process_group()
            conn.send(
                {
                    "type": "READY",
                    "round": round_index,
                    "rank": rank,
                    "loss": float(loss),
                    "train_ms": float(train_ms),
                    "fuse_ipc_ms": float(ipc_ms),
                    "descriptor_init_ms": float(descriptor_init_ms),
                    "descriptor_reused": bool(descriptor_reused),
                    "storage_stable": bool(storage_stable),
                    "changed_storage_examples": changed_storage_examples,
                    "runtime_release_ms": float(release_ms),
                    "source_released": True,
                    "raw_gb": float(raw_gb),
                    "alloc_gb_after_release": float(release_memory["allocated_gb"]),
                    "reserved_gb_after_release": float(release_memory["reserved_gb"]),
                    "free_gb_after_release": float(release_memory["free_gb"]),
                    "build_metrics": build_metrics,
                    "n_tensors": len(ipc_desc or {}),
                    "ipc_desc": ipc_desc,
                }
            )
            msg = conn.recv()
            if msg == "EXIT":
                break
            if not isinstance(msg, dict) or msg.get("type") != "REBUILD_TRAINING":
                conn.send({"type": "WARN", "rank": rank, "msg": f"unexpected command {msg!r}"})
                break
            master_port = int(msg["master_port"])
            round_index += 1
    except Exception as exc:
        try:
            conn.send({"type": "ERROR", "rank": rank, "error": repr(exc)})
        finally:
            raise


def _mean(items: list[dict[str, Any]], key: str) -> float:
    return sum(float(item[key]) for item in items) / max(len(items), 1)


def _collect_ready(conns: list[Any], round_index: int) -> list[dict[str, Any]]:
    ready = []
    for conn in conns:
        msg = conn.recv()
        if msg.get("type") != "READY":
            raise RuntimeError(f"training worker failed in round {round_index}: {msg}")
        ready.append(msg)
    ready.sort(key=lambda item: item["rank"])
    return ready


def _print_round_training(round_index: int, ready: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "mean_train_ms": _mean(ready, "train_ms"),
        "mean_ipc_desc_ms": _mean(ready, "fuse_ipc_ms"),
        "mean_descriptor_init_ms": _mean(ready, "descriptor_init_ms"),
        "mean_runtime_release_ms": _mean(ready, "runtime_release_ms"),
        "source_released": all(bool(msg.get("source_released")) for msg in ready),
        "descriptor_reused": all(bool(msg.get("descriptor_reused")) for msg in ready),
        "storage_stable": all(bool(msg.get("storage_stable")) for msg in ready),
        "max_alloc_after_release_gb": max(float(msg["alloc_gb_after_release"]) for msg in ready),
        "max_reserved_after_release_gb": max(
            float(msg["reserved_gb_after_release"]) for msg in ready
        ),
        "min_free_after_release_gb": min(float(msg["free_gb_after_release"]) for msg in ready),
        "max_build_alloc_gb": max(
            float(msg["build_metrics"]["memory_after_build"]["allocated_gb"]) for msg in ready
        ),
    }
    print(
        f"[round {round_index + 1}] training released: "
        f"train={metrics['mean_train_ms']:.1f}ms ipc={metrics['mean_ipc_desc_ms']:.2f}ms "
        f"release={metrics['mean_runtime_release_ms']:.2f}ms "
        f"alloc_after_release_max={metrics['max_alloc_after_release_gb']:.2f}GB "
        f"free_after_release_min={metrics['min_free_after_release_gb']:.2f}GB "
        f"storage_stable={metrics['storage_stable']}",
        flush=True,
    )
    for msg in ready:
        alias = msg["build_metrics"].get("alias_result", {})
        print(
            f"      rank {msg['rank']}: loss={msg['loss']:.4f}, "
            f"build_alloc={msg['build_metrics']['memory_after_build']['allocated_gb']:.2f}GB, "
            f"release_alloc={msg['alloc_gb_after_release']:.2f}GB, "
            f"raw={msg['raw_gb']:.2f}GB, alias={alias.get('aliased', 0)} "
            f"skipped={alias.get('skipped', 0)} mismatch={alias.get('mismatched', 0)}",
            flush=True,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-4b", choices=["qwen3-4b", "qwen3-32b"])
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--train-batch", type=int, default=2)
    parser.add_argument("--train-seq-len", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--sample-rounds", type=int, default=1)
    parser.add_argument("--release-mode", choices=["strict", "fast"], default="fast")
    parser.add_argument("--descriptor-mode", choices=["fresh", "reuse"], default="reuse")
    parser.add_argument("--verify-inject", action="store_true")
    parser.add_argument("--memory-breakdown", action="store_true")
    args = parser.parse_args()

    spec = load_model_spec(args.model)
    ctx = mp.get_context("spawn")
    master_port = _free_port()
    print("=== FlexGPU Round-Trip True-Zero Benchmark ===", flush=True)
    print(
        f"  model={spec.name}, tp={args.tp_size}, rounds={args.rounds}, "
        f"train_batch={args.train_batch}, seq={args.train_seq_len}, "
        f"num_prompts={args.num_prompts}, max_tokens={args.max_tokens}, "
        f"verify={args.verify_inject}",
        flush=True,
    )

    conns = []
    workers = []
    for rank in range(args.tp_size):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_training_worker_roundtrip,
            args=(
                rank,
                args.tp_size,
                args.model,
                args.train_batch,
                args.train_seq_len,
                args.release_mode,
                args.descriptor_mode,
                master_port,
                child_conn,
            ),
        )
        proc.start()
        conns.append(parent_conn)
        workers.append(proc)

    round_metrics: list[dict[str, Any]] = []
    try:
        for round_index in range(args.rounds):
            ready = _collect_ready(conns, round_index)
            training_metrics = _print_round_training(round_index, ready)
            all_rank_descs = [msg["ipc_desc"] for msg in ready]
            vllm_result = _run_vllm_flex(
                model_path=spec.path,
                tp_size=args.tp_size,
                all_rank_descs=all_rank_descs,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_tokens=args.max_tokens,
                sample_rounds=args.sample_rounds,
                num_prompts=args.num_prompts,
                inject_rounds=2,
                verify_inject=args.verify_inject,
                memory_breakdown=args.memory_breakdown,
            )
            steady_inject = vllm_result["inject_ms_list"][1]
            sampling_metrics = {
                "sampling_throughput": sum(vllm_result["throughputs"])
                / len(vllm_result["throughputs"]),
                "sampling_latency_ms": sum(vllm_result["latencies"])
                / len(vllm_result["latencies"]),
                "inject_alias_ms": vllm_result["inject_ms"],
                "steady_inject_alias_ms": steady_inject,
                "readback_exact": all(
                    item["mismatched"] == 0 for item in vllm_result["inject_results"]
                ),
            }
            steady_t2s_ms = (
                training_metrics["mean_ipc_desc_ms"]
                + training_metrics["mean_runtime_release_ms"]
                + sampling_metrics["steady_inject_alias_ms"]
            )
            round_summary = {
                "round": round_index + 1,
                **training_metrics,
                **sampling_metrics,
                "steady_training_to_sampling_ms": steady_t2s_ms,
            }
            round_metrics.append(round_summary)
            print(
                f"[round {round_index + 1}] sampling: "
                f"throughput={sampling_metrics['sampling_throughput']:.1f} tok/s "
                f"latency={sampling_metrics['sampling_latency_ms']:.0f}ms "
                f"steady_inject={steady_inject:.2f}ms "
                f"steady_t2s={steady_t2s_ms:.2f}ms",
                flush=True,
            )
            if round_index + 1 < args.rounds:
                next_port = _free_port()
                for conn in conns:
                    conn.send({"type": "REBUILD_TRAINING", "master_port": next_port})
            else:
                for conn in conns:
                    conn.send("EXIT")
    finally:
        for proc in workers:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)

    print("\n=== FlexGPU Round-Trip Summary ===", flush=True)
    for item in round_metrics:
        print(
            f"  round {item['round']}: train_ms={item['mean_train_ms']:.1f}, "
            f"release_ms={item['mean_runtime_release_ms']:.2f}, "
            f"steady_t2s_ms={item['steady_training_to_sampling_ms']:.2f}, "
            f"sampling_tput={item['sampling_throughput']:.1f}, "
            f"alloc_after_release_max={item['max_alloc_after_release_gb']:.2f}GB, "
            f"free_after_release_min={item['min_free_after_release_gb']:.2f}GB, "
            f"storage_stable={item['storage_stable']}, readback={item['readback_exact']}",
            flush=True,
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
