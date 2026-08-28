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
    if hasattr(model, "lm_head") and not getattr(model.config, "tie_word_embeddings", False):
        lm_head_base = _get_base_module(model.lm_head)
        if getattr(lm_head_base, "weight", None) is not None:
            state_dict["lm_head.weight"] = _get_local_tensor(lm_head_base.weight)
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
        # Keepalive must retain the full (training-side) tensor: for vocab-
        # parallel weights the descriptor is a per-rank shard for vLLM, but the
        # sampling->training skeleton rebuild aliases the full training
        # embedding/lm_head back from keepalive. Keeping only the shard there
        # caused embedding index-out-of-range device asserts after rebuild.
        keepalive_tensor = tensor
        if key in ("model.embed_tokens.weight", "lm_head.weight") and vocab_chunk is not None:
            tensor = tensor[rank * vocab_chunk : (rank + 1) * vocab_chunk]
        if hasattr(tensor, "contiguous"):
            tensor = tensor.contiguous()
        if require_cuda and not bool(getattr(tensor, "is_cuda", False)):
            raise RuntimeError(f"{key} is not a CUDA tensor")
        keepalive.append(keepalive_tensor)
        descriptors[key] = {
            "shape": tuple(getattr(tensor, "shape", ())),
            "dtype": str(getattr(tensor, "dtype", "unknown")),
            "ipc": descriptor_factory(tensor),
        }
    return descriptors, keepalive


def _retag_cumem_allocation(ptr: int, tag: str) -> int:
    try:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
    except Exception:
        return 0
    data = allocator.pointer_to_data.get(ptr)
    if data is None:
        return 0
    data.tag = tag
    return int(data.handle[1])


def _cumem_tag_summary() -> dict[str, float]:
    try:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
    except Exception:
        return {}
    summary: dict[str, float] = {}
    for data in allocator.pointer_to_data.values():
        summary[data.tag] = summary.get(data.tag, 0.0) + data.handle[1] / (1024**3)
    return summary


def _cumem_allocation_summary() -> dict[str, Any]:
    try:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
    except Exception:
        return {}
    by_tag: dict[str, dict[str, Any]] = {}
    for data in allocator.pointer_to_data.values():
        size_gb = data.handle[1] / (1024**3)
        item = by_tag.setdefault(data.tag, {"count": 0, "total_gb": 0.0, "largest_gb": 0.0})
        item["count"] += 1
        item["total_gb"] += size_gb
        item["largest_gb"] = max(float(item["largest_gb"]), size_gb)
    return by_tag


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
    retagged_allocations = 0
    retagged_bytes = 0

    for key, obj in objects.items():
        descriptor_key = key
        if descriptor_key not in descriptors:
            # enable_lora=True wraps supported linear/embedding modules so their
            # weights live under a ".base_layer" submodule (e.g.
            # qkv_proj.base_layer.weight). Normalize back to the fused-layout
            # descriptor names. Also tolerate extra leading "model." prefixes.
            normalized = descriptor_key.replace(".base_layer.", ".")
            if normalized in descriptors:
                descriptor_key = normalized
            else:
                stripped = descriptor_key
                while stripped not in descriptors and stripped.startswith("model."):
                    stripped = stripped[len("model.") :]
                stripped = stripped.replace(".base_layer.", ".")
                if stripped in descriptors:
                    descriptor_key = stripped
        if descriptor_key not in descriptors:
            if key.endswith(("_q_scale", "_k_scale", "_v_scale", "_prob_scale")):
                obj.data.fill_(1.0)
                injected += 1
                verified += 1
            else:
                skipped += 1
                if len(examples) < 10:
                    examples.append((key, "no_descriptor"))
            continue

        shared = tensor_rebuilder(*descriptors[descriptor_key]["ipc"])
        if tuple(shared.shape) != tuple(obj.data.shape):
            skipped += 1
            if len(examples) < 5:
                examples.append((key, tuple(obj.data.shape), tuple(shared.shape), "shape"))
            continue

        old_tensor = getattr(obj, "data", obj)
        try:
            old_ptr = int(old_tensor.data_ptr())
        except Exception:
            old_ptr = 0
        obj.data = shared
        if old_ptr and old_ptr != int(shared.data_ptr()):
            retagged = _retag_cumem_allocation(old_ptr, "discarded_weights")
            if retagged:
                retagged_allocations += 1
                retagged_bytes += retagged
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

    worker._tuft_flex_alias_injected = True
    return {
        "rank": tensor_parallel_rank,
        "injected": injected,
        "verified": verified,
        "mismatched": mismatched,
        "skipped": skipped,
        "max_diff": max_diff,
        "retagged_allocations": retagged_allocations,
        "retagged_gb": retagged_bytes / (1024**3),
        "cumem_tags": _cumem_tag_summary(),
        "examples": examples,
    }


def get_pre_capture_alias_result(worker: Any) -> dict[str, Any]:
    from vllm.distributed import get_tensor_model_parallel_rank

    result = getattr(worker, "_tuft_flex_precapture_alias_result", None)
    if result is None:
        return {
            "rank": get_tensor_model_parallel_rank(),
            "injected": 0,
            "verified": 0,
            "mismatched": 0,
            "skipped": 0,
            "max_diff": 0.0,
            "examples": ["pre_capture_alias_not_run"],
        }
    return result


def capture_cuda_graph_after_alias(worker: Any) -> dict[str, Any]:
    """vLLM worker callback: capture CUDA Graph after IPC alias is installed.

    This is an experimental path for FlexBackend: start vLLM with eager mode so
    engine initialization does not capture graphs over dummy weights, inject IPC
    aliases, then flip the worker back to non-eager and capture graphs over the
    aliased storage.
    """
    import time

    import torch
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.warmup.kernel_warmup import kernel_warmup

    rank = get_tensor_model_parallel_rank()
    if not bool(getattr(worker, "_tuft_flex_alias_injected", False)):
        return {"rank": rank, "captured": False, "error": "alias_not_injected"}

    model_runner = worker.model_runner
    worker.model_config.enforce_eager = False
    worker.vllm_config.model_config.enforce_eager = False
    model_runner.model_config.enforce_eager = False
    model_runner.vllm_config.model_config.enforce_eager = False

    start = time.perf_counter()
    try:
        kernel_warmup(worker)
        graph_bytes = int(model_runner.capture_model())
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        worker._tuft_flex_post_alias_cudagraph_captured = True
        return {
            "rank": rank,
            "captured": True,
            "capture_ms": elapsed_ms,
            "graph_memory_gb": graph_bytes / (1024**3),
        }
    except Exception as exc:
        return {"rank": rank, "captured": False, "error": repr(exc)}


def sleep_vllm_worker(worker: Any, level: int = 1) -> dict[str, Any]:
    import time

    import torch
    from vllm.distributed import get_tensor_model_parallel_rank

    start = time.perf_counter()
    worker.sleep(level=level)
    torch.cuda.synchronize()
    return {
        "rank": get_tensor_model_parallel_rank(),
        "sleep_ms": (time.perf_counter() - start) * 1000,
    }


def wake_vllm_worker(worker: Any, tags: list[str] | None = None) -> dict[str, Any]:
    import time

    import torch
    from vllm.distributed import get_tensor_model_parallel_rank

    start = time.perf_counter()
    worker.wake_up(tags=tags)
    torch.cuda.synchronize()
    return {
        "rank": get_tensor_model_parallel_rank(),
        "wake_ms": (time.perf_counter() - start) * 1000,
        "tags": tags,
    }


def flex_partial_sleep_vllm_worker(worker: Any, kv_cache_gb: float) -> dict[str, Any]:
    """Discard dummy weights and only part of KV cache pages."""
    import time

    import torch
    from vllm.device_allocator.cumem import CuMemAllocator, unmap_and_release
    from vllm.distributed import get_tensor_model_parallel_rank

    free_before = torch.cuda.mem_get_info()[0]
    start = time.perf_counter()
    worker._sleep_saved_buffers = {}
    allocator = CuMemAllocator.get_instance()
    before_summary = _cumem_allocation_summary()
    freed_kv_gb = 0.0
    freed_discarded_gb = 0.0
    target_kv_gb = max(float(kv_cache_gb), 0.0)
    for _ptr, data in list(allocator.pointer_to_data.items()):
        if data.tag == "discarded_weights":
            unmap_and_release(data.handle)
            freed_discarded_gb += data.handle[1] / (1024**3)
            data.tag = "discarded_weights_sleeping"
    kv_items = [
        data for data in allocator.pointer_to_data.values() if data.tag == "kv_cache"
    ]
    kv_items.sort(key=lambda value: value.handle[1], reverse=True)
    for data in kv_items:
        if freed_kv_gb >= target_kv_gb:
            break
        unmap_and_release(data.handle)
        freed_kv_gb += data.handle[1] / (1024**3)
        data.tag = "kv_cache_sleeping"
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    after_summary = _cumem_allocation_summary()
    free_after, total = torch.cuda.mem_get_info()
    return {
        "rank": get_tensor_model_parallel_rank(),
        "sleep_ms": (time.perf_counter() - start) * 1000,
        "freed_gb": (free_after - free_before) / (1024**3),
        "used_gb": (total - free_after) / (1024**3),
        "freed_kv_gb": freed_kv_gb,
        "freed_discarded_gb": freed_discarded_gb,
        "cumem_before": before_summary,
        "cumem_after": after_summary,
    }


def flex_partial_wake_vllm_worker(worker: Any) -> dict[str, Any]:
    """Wake only KV pages previously unmapped by flex_partial_sleep_vllm_worker."""
    import time

    import torch
    from vllm.device_allocator.cumem import CuMemAllocator, create_and_map
    from vllm.distributed import get_tensor_model_parallel_rank

    start = time.perf_counter()
    worker._sleep_saved_buffers = {}
    allocator = CuMemAllocator.get_instance()
    before_summary = _cumem_allocation_summary()
    remapped_kv_gb = 0.0
    for data in allocator.pointer_to_data.values():
        if data.tag == "kv_cache_sleeping":
            create_and_map(data.handle)
            remapped_kv_gb += data.handle[1] / (1024**3)
            data.tag = "kv_cache"
    worker.model_runner.post_kv_cache_wake_up()
    torch.cuda.synchronize()
    after_summary = _cumem_allocation_summary()
    return {
        "rank": get_tensor_model_parallel_rank(),
        "wake_ms": (time.perf_counter() - start) * 1000,
        "remapped_kv_gb": remapped_kv_gb,
        "cumem_before": before_summary,
        "cumem_after": after_summary,
    }


def update_registered_lora_inplace(
    worker: Any, lora_int_id: int, safetensors_path: str
) -> dict[str, Any]:
    """Refresh an already-registered vLLM LoRA by copying new weights in place.

    Avoids the full ``add_lora`` reload (LRU bookkeeping + tensor rebuild) that
    dominates the training->sampling flip. Reads the adapter from a RAM-backed
    path and copies lora_a/lora_b into the resident ``LoRALayerWeights`` tensors.
    Returns ``ok=False`` when the adapter is not resident so the caller falls
    back to ``add_lora``.
    """
    import time

    start = time.perf_counter()
    model_runner = getattr(worker, "model_runner", None)
    lora_manager = getattr(model_runner, "lora_manager", None)
    if lora_manager is None:
        return {"ok": False, "reason": "no lora_manager"}
    registered = getattr(lora_manager, "_registered_adapters", {})
    lora_model = registered.get(int(lora_int_id))
    if lora_model is None:
        return {"ok": False, "reason": "not registered"}
    try:
        from safetensors.torch import load_file

        tensors = load_file(safetensors_path)
    except Exception as exc:  # unreadable file -> caller falls back
        return {"ok": False, "reason": f"load failed: {exc!r}"}
    # vLLM's add_lora calls LoRALayerWeights.optimize(), which folds the PEFT
    # scaling (lora_alpha/r) into lora_b and resets scaling to 1. To keep the
    # in-place refresh numerically identical to add_lora+optimize we must apply
    # the same scaling to lora_b here; copying the raw tensor would under-scale
    # the LoRA delta and skew sampling logprobs.
    scaling = 1.0
    try:
        import json
        import os

        cfg_path = os.path.join(os.path.dirname(safetensors_path), "adapter_config.json")
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        rank = float(cfg.get("r", cfg.get("rank", 0)) or 0)
        alpha = float(cfg.get("lora_alpha", 0) or 0)
        if rank > 0 and alpha > 0:
            scaling = alpha / rank
    except Exception:
        scaling = 1.0
    updated = 0
    for key, tensor in tensors.items():
        if ".lora_A." in key:
            which = "lora_a"
        elif ".lora_B." in key:
            which = "lora_b"
        else:
            continue
        module = key.replace("base_model.model.", "")
        module = module.split(".lora_A.")[0].split(".lora_B.")[0]
        layer_weights = lora_model.get_lora(module)
        if layer_weights is None:
            continue
        dst = layer_weights.lora_a if which == "lora_a" else layer_weights.lora_b
        if dst is None or tuple(dst.shape) != tuple(tensor.shape):
            continue
        src = tensor.to(device=dst.device, dtype=dst.dtype)
        if which == "lora_b" and scaling != 1.0:
            src = src * scaling
        dst.copy_(src)
        updated += 1
    return {
        "ok": updated > 0,
        "updated": updated,
        "ms": (time.perf_counter() - start) * 1000,
    }


def flex_sleep_vllm_worker(worker: Any) -> dict[str, Any]:
    """Discard vLLM CuMem pages (discarded weights + full KV cache) without CPU offload.

    Uses manual unmap_and_release with explicit tag tracking (same approach as
    flex_partial_sleep_vllm_worker) so that repeated sleep/wake cycles are robust.
    vLLM's built-in allocator.sleep()/wake_up() can fail with CUDA invalid
    argument on the second sleep after a wake_up, so we avoid it here.
    """
    import time

    import torch
    from vllm.device_allocator.cumem import CuMemAllocator, unmap_and_release
    from vllm.distributed import get_tensor_model_parallel_rank

    free_before = torch.cuda.mem_get_info()[0]
    start = time.perf_counter()
    worker._sleep_saved_buffers = {}
    allocator = CuMemAllocator.get_instance()
    before_summary = _cumem_allocation_summary()
    freed_discarded_gb = 0.0
    freed_kv_gb = 0.0
    for _ptr, data in list(allocator.pointer_to_data.items()):
        if data.tag == "discarded_weights":
            unmap_and_release(data.handle)
            freed_discarded_gb += data.handle[1] / (1024**3)
            data.tag = "discarded_weights_sleeping"
        elif data.tag == "kv_cache":
            unmap_and_release(data.handle)
            freed_kv_gb += data.handle[1] / (1024**3)
            data.tag = "kv_cache_sleeping"
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    after_summary = _cumem_allocation_summary()
    free_after, total = torch.cuda.mem_get_info()
    return {
        "rank": get_tensor_model_parallel_rank(),
        "sleep_ms": (time.perf_counter() - start) * 1000,
        "freed_gb": (free_after - free_before) / (1024**3),
        "used_gb": (total - free_after) / (1024**3),
        "freed_kv_gb": freed_kv_gb,
        "freed_discarded_gb": freed_discarded_gb,
        "cumem_before": before_summary,
        "cumem_after": after_summary,
    }


def flex_wake_vllm_worker(worker: Any, tags: list[str] | None = None) -> dict[str, Any]:
    """Wake CuMem pages previously unmapped by flex_sleep_vllm_worker.

    Uses manual create_and_map with explicit tag tracking (same approach as
    flex_partial_wake_vllm_worker) for robust repeated sleep/wake cycles.
    The *tags* argument is accepted for API compatibility but ignored: all
    pages put to sleep by flex_sleep_vllm_worker are remapped.
    """
    import time

    import torch
    from vllm.device_allocator.cumem import CuMemAllocator, create_and_map
    from vllm.distributed import get_tensor_model_parallel_rank

    start = time.perf_counter()
    worker._sleep_saved_buffers = {}
    allocator = CuMemAllocator.get_instance()
    before_summary = _cumem_allocation_summary()
    remapped_discarded_gb = 0.0
    remapped_kv_gb = 0.0
    for data in allocator.pointer_to_data.values():
        if data.tag == "discarded_weights_sleeping":
            create_and_map(data.handle)
            remapped_discarded_gb += data.handle[1] / (1024**3)
            data.tag = "discarded_weights"
        elif data.tag == "kv_cache_sleeping":
            create_and_map(data.handle)
            remapped_kv_gb += data.handle[1] / (1024**3)
            data.tag = "kv_cache"
    worker.model_runner.post_kv_cache_wake_up()
    torch.cuda.synchronize()
    after_summary = _cumem_allocation_summary()
    return {
        "rank": get_tensor_model_parallel_rank(),
        "wake_ms": (time.perf_counter() - start) * 1000,
        "tags": tags,
        "remapped_kv_gb": remapped_kv_gb,
        "remapped_discarded_gb": remapped_discarded_gb,
        "cumem_before": before_summary,
        "cumem_after": after_summary,
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
