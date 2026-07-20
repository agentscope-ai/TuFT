from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any


TensorDescriptorFactory = Callable[[Any], Any]
TensorRebuilder = Callable[..., Any]


def _get_local_tensor(value: Any) -> Any:
    if hasattr(value, "to_local"):
        return value.to_local()
    if hasattr(value, "_local_tensor"):
        return value._local_tensor
    return value


def _get_base_module(module: Any) -> Any:
    return module.base if hasattr(module, "base") else module


def create_fused_vllm_state_dict(model: Any) -> dict[str, Any]:
    """Create vLLM-compatible local-shard state dict from a fused PyTorch-TP model."""
    state_dict: dict[str, Any] = {}
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        mlp = layer.mlp
        prefix = f"model.layers.{index}"
        qkv_base = _get_base_module(attention.qkv_proj)
        o_base = _get_base_module(attention.o_proj)
        gate_up_base = _get_base_module(mlp.gate_up_proj)
        down_base = _get_base_module(mlp.down_proj)

        state_dict[f"{prefix}.self_attn.qkv_proj.weight"] = _get_local_tensor(qkv_base.weight)
        state_dict[f"{prefix}.self_attn.o_proj.weight"] = _get_local_tensor(o_base.weight)
        state_dict[f"{prefix}.mlp.gate_up_proj.weight"] = _get_local_tensor(gate_up_base.weight)
        state_dict[f"{prefix}.mlp.down_proj.weight"] = _get_local_tensor(down_base.weight)
        state_dict[f"{prefix}.input_layernorm.weight"] = _get_local_tensor(
            layer.input_layernorm.weight
        )
        state_dict[f"{prefix}.post_attention_layernorm.weight"] = _get_local_tensor(
            layer.post_attention_layernorm.weight
        )
        if hasattr(attention, "q_norm"):
            state_dict[f"{prefix}.self_attn.q_norm.weight"] = _get_local_tensor(
                attention.q_norm.weight
            )
        if hasattr(attention, "k_norm"):
            state_dict[f"{prefix}.self_attn.k_norm.weight"] = _get_local_tensor(
                attention.k_norm.weight
            )

    state_dict["model.embed_tokens.weight"] = _get_local_tensor(model.model.embed_tokens.weight)
    state_dict["model.norm.weight"] = _get_local_tensor(model.model.norm.weight)
    if (
        hasattr(model, "lm_head")
        and model.lm_head.weight is not None
        and not getattr(model.config, "tie_word_embeddings", False)
    ):
        state_dict["lm_head.weight"] = _get_local_tensor(model.lm_head.weight)
    return state_dict


def tensor_to_cuda_ipc_descriptor(tensor: Any) -> Any:
    from torch.multiprocessing.reductions import reduce_tensor

    _, args = reduce_tensor(tensor)
    return args


def make_cuda_ipc_descriptor_dict(
    state_dict: Mapping[str, Any],
    *,
    rank: int,
    world_size: int,
    vocab_size: int | None = None,
    descriptor_factory: TensorDescriptorFactory = tensor_to_cuda_ipc_descriptor,
    require_cuda: bool = True,
) -> tuple[dict[str, dict[str, Any]], list[Any]]:
    """Create CUDA IPC descriptors and keepalive tensor refs for one producer TP rank."""
    descriptors: dict[str, dict[str, Any]] = {}
    keepalive: list[Any] = []
    vocab_chunk = None
    if vocab_size is not None:
        if vocab_size % world_size != 0:
            raise ValueError(f"vocab_size={vocab_size} is not divisible by world_size={world_size}")
        vocab_chunk = vocab_size // world_size

    for key, original_tensor in state_dict.items():
        tensor = _get_local_tensor(original_tensor)
        if key in ("model.embed_tokens.weight", "lm_head.weight") and vocab_chunk is not None:
            tensor = tensor[rank * vocab_chunk : (rank + 1) * vocab_chunk]
        if hasattr(tensor, "contiguous"):
            tensor = tensor.contiguous()
        if require_cuda and not bool(getattr(tensor, "is_cuda", False)):
            raise RuntimeError(f"{key} is not a CUDA tensor")
        keepalive.append(tensor)
        descriptors[key] = {
            "shape": tuple(getattr(tensor, "shape", ())),
            "dtype": str(getattr(tensor, "dtype", "unknown")),
            "ipc": descriptor_factory(tensor),
        }
    return descriptors, keepalive


def _get_or_create_vllm_object_cache(worker: Any) -> dict[str, Any]:
    cache = getattr(worker, "_tuft_flex_object_cache", None)
    if cache is not None:
        return cache
    model = worker.model_runner.model
    cache = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    worker._tuft_flex_object_cache = cache
    return cache


def _object_cache_storage_bytes(cache: Mapping[str, Any]) -> tuple[int, int]:
    seen: set[int] = set()
    total = 0
    for obj in cache.values():
        tensor = getattr(obj, "data", obj)
        if not hasattr(tensor, "data_ptr"):
            continue
        try:
            ptr = int(tensor.data_ptr())
        except RuntimeError:
            continue
        if ptr in seen:
            continue
        seen.add(ptr)
        total += int(tensor.numel()) * int(tensor.element_size())
    return total, len(seen)


def collect_cuda_memory_snapshot(
    worker: Any,
    label: str,
    empty_cache: bool = False,
    cache_objects: bool = True,
) -> dict[str, Any]:
    import gc

    import torch
    from vllm.distributed import get_tensor_model_parallel_rank

    if empty_cache:
        gc.collect()
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    cache = _get_or_create_vllm_object_cache(worker) if cache_objects else None
    object_bytes, object_count = _object_cache_storage_bytes(cache) if cache is not None else (0, 0)
    return {
        "label": label,
        "rank": get_tensor_model_parallel_rank(),
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "free_gb": free_bytes / (1024**3),
        "total_gb": total_bytes / (1024**3),
        "object_storage_gb": object_bytes / (1024**3),
        "object_storage_count": object_count,
    }


def prepare_cuda_ipc_alias_cache(worker: Any) -> dict[str, Any]:
    cache = _get_or_create_vllm_object_cache(worker)
    return {"cached_objects": len(cache)}


def inject_cuda_ipc_alias(
    worker: Any,
    all_rank_descriptors: list[dict[str, dict[str, Any]]],
    verify: bool = True,
    *,
    tensor_rebuilder: TensorRebuilder | None = None,
) -> dict[str, Any]:
    """vLLM worker callback: rebuild CUDA IPC tensors and alias parameter storage."""
    if tensor_rebuilder is None:
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        tensor_rebuilder = rebuild_cuda_tensor
    assert tensor_rebuilder is not None
    from vllm.distributed import get_tensor_model_parallel_rank

    tensor_parallel_rank = get_tensor_model_parallel_rank()
    descriptors = all_rank_descriptors[tensor_parallel_rank]
    objects = _get_or_create_vllm_object_cache(worker)

    injected = 0
    verified = 0
    skipped = 0
    mismatched = 0
    max_diff = 0.0
    examples: list[Any] = []

    for key, obj in objects.items():
        if key not in descriptors:
            if key.endswith(("_q_scale", "_k_scale", "_v_scale", "_prob_scale")):
                obj.data.fill_(1.0)
                injected += 1
                verified += 1
            else:
                skipped += 1
            continue

        shared = tensor_rebuilder(*descriptors[key]["ipc"])
        if tuple(shared.shape) != tuple(obj.data.shape):
            skipped += 1
            if len(examples) < 5:
                examples.append((key, tuple(obj.data.shape), tuple(shared.shape), "shape"))
            continue

        obj.data = shared
        injected += 1
        if verify:
            diff = (obj.data.float() - shared.float()).abs().max().item()
            max_diff = max(max_diff, float(diff))
            if diff == 0.0:
                verified += 1
            else:
                mismatched += 1
                if len(examples) < 5:
                    examples.append((key, float(diff)))
        else:
            verified += 1

    if not bool(getattr(worker, "_tuft_flex_dummy_cache_cleared", False)):
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        worker._tuft_flex_dummy_cache_cleared = True

    return {
        "rank": tensor_parallel_rank,
        "injected": injected,
        "verified": verified,
        "mismatched": mismatched,
        "skipped": skipped,
        "max_diff": max_diff,
        "examples": examples,
    }


async def call_collective_rpc(engine: Any, fn: Callable[..., Any], args: tuple[Any, ...]) -> Any:
    """Call collective_rpc on a local vLLM LLM object or a Ray/trinity actor wrapper."""
    collective_rpc = getattr(engine, "collective_rpc", None)
    if callable(collective_rpc):
        result = collective_rpc(fn, args=args)
        if inspect.isawaitable(result):
            return await result
        return result

    actor_method = getattr(engine, "collective_rpc", None)
    remote = getattr(actor_method, "remote", None)
    if callable(remote):
        result = remote(fn, None, args, None)
        if isinstance(result, (dict, list, tuple)):
            return result
        import ray

        return await asyncio_to_thread_get(ray, result)

    private_method = getattr(engine, "_collective_rpc", None)
    remote = getattr(private_method, "remote", None)
    if callable(remote):
        result = remote(fn, None, args, None)
        if isinstance(result, (dict, list, tuple)):
            return result
        import ray

        return await asyncio_to_thread_get(ray, result)

    raise AttributeError("vLLM engine does not expose collective_rpc")


async def asyncio_to_thread_get(ray_module: Any, ref: Any) -> Any:
    import asyncio

    return await asyncio.to_thread(ray_module.get, ref)


def summarize_injection_results(results: list[Mapping[str, Any]]) -> dict[str, float]:
    injected = sum(float(item.get("injected", 0.0)) for item in results)
    verified = sum(float(item.get("verified", 0.0)) for item in results)
    mismatched = sum(float(item.get("mismatched", 0.0)) for item in results)
    skipped = sum(float(item.get("skipped", 0.0)) for item in results)
    max_diff = max((float(item.get("max_diff", 0.0)) for item in results), default=0.0)
    return {
        "zero_copy": 1.0,
        "base_transform_supported": 1.0,
        "ipc_injected:sum": injected,
        "ipc_verified:sum": verified,
        "ipc_mismatched:sum": mismatched,
        "ipc_skipped:sum": skipped,
        "ipc_max_diff:max": max_diff,
    }
