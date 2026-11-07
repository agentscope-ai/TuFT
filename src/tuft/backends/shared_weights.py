"""Frozen HF/vLLM weight storage for the opt-in Qwen TP=1 flex backend.

HF keeps its normal modules and LoRA/optimizer implementation. Only frozen
q/k/v and gate/up storage is packed once at startup; HF parameters become views
of those slabs. vLLM receives CUDA IPC aliases, not copies of the slabs.

The explicit PyTorch reducers are an internal interface, isolated here because
Ray's normal tensor serialization would copy the data. Only trusted actor-to-
actor startup messages may supply descriptors. The producer must outlive vLLM.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .shared_weight_layout import (
    canonical_name,
    fused_groups,
    is_recurrent_scale,
    serving_name,
    text_config,
)


def tensor_fingerprint(tensor: Any) -> str:
    """Validation digest, including scalar buffers and BF16 values."""
    import hashlib

    import torch

    data = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def check_shared_weight_versions(torch_version: str, vllm_version: str) -> None:
    from packaging.version import Version

    if Version(torch_version).public != "2.11.0" or Version(vllm_version).public != "0.24.0":
        raise RuntimeError("Shared weights require the tested torch 2.11.0 / vLLM 0.24.0 pair")


def pack_hf_base_weights(model: Any) -> dict[str, Any]:
    """Pack a frozen Qwen base, preserving Parameter identities and values.

    Call before serving requests or creating optimizers. This function accepts
    CPU tensors too, to allow numerical/storage tests without a CUDA runtime.
    """
    import torch

    text_config(model.config)
    params = {
        canonical_name(name): param
        for name, param in model.named_parameters(remove_duplicate=False)
        if ".lora_" not in name
    }
    if any(p.requires_grad for p in params.values()):
        raise ValueError("Only frozen base weights may be shared")
    if any(
        p.dtype != torch.bfloat16 and not (is_recurrent_scale(n) and p.dtype == torch.float32)
        for n, p in params.items()
    ):
        raise ValueError("Shared weights require a BF16 base model")
    if len({p.device for p in params.values()}) != 1:
        raise ValueError("Shared weights require the entire base model on one device")
    result = {}
    consumed: set[str] = set()
    for name, param in params.items():
        # vLLM computes recurrent decay in FP32, as does the HF forward path.
        if is_recurrent_scale(name):
            param.data = param.data.float()
    for parent, names, fused in fused_groups(model.config):
        keys = [f"{parent}.{name}.weight" for name in names]
        pieces = [params[key] for key in keys]
        slab = torch.cat([p.detach() for p in pieces], dim=0)
        offset = 0
        for param in pieces:
            size = param.shape[0]
            param.data = slab[offset : offset + size]
            offset += size
        result[f"{parent}.{fused}.weight"] = slab
        consumed.update(keys)
    for name, param in params.items():
        if name not in consumed:
            result[serving_name(name)] = param.detach()
    return result


def export_cuda_weights(weights: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from torch.multiprocessing.reductions import reduce_tensor

    if any(not t.is_cuda or not t.is_contiguous() for t in weights.values()):
        raise ValueError("CUDA IPC export requires contiguous CUDA tensors")
    torch.cuda.synchronize()
    return {name: reduce_tensor(tensor)[1] for name, tensor in weights.items()}


def resolve_shared_weight_aliases(model: Any, weights: Mapping[str, Any]) -> dict[str, Any]:
    """Expand repeated module paths only when they refer to one Parameter."""
    params = dict(model.named_parameters(remove_duplicate=False))
    result = dict(weights)
    known = {id(params[n]): tensor for n, tensor in weights.items() if n in params}
    for name, param in params.items():
        if name not in result and id(param) in known:
            result[name] = known[id(param)]
    return result


def bind_shared_weights(model: Any, weights: Mapping[str, Any]) -> dict[str, int]:
    """Bind all base parameters, failing closed on unknown layouts.

    Vocabulary padding can differ between HF and vLLM. Only those tensors may
    be copied, with padding explicitly zeroed. All other tensors must alias.
    """
    import torch

    params = dict(model.named_parameters(remove_duplicate=False))
    # vLLM's MoE runner registers the same router/shared expert under several
    # paths. Resolve only actual Parameter identities, never name heuristics.
    weights = resolve_shared_weight_aliases(model, weights)
    if params.keys() != weights.keys():
        raise ValueError(
            f"Shared weight keys differ: missing={params.keys() - weights.keys()}, "
            f"extra={weights.keys() - params.keys()}"
        )
    copied = set()
    for name, param in params.items():
        shared = weights[name]
        if param.dtype != shared.dtype or param.device != shared.device:
            raise ValueError(f"Shared weight dtype/device mismatch: {name}")
        if param.shape != shared.shape:
            if not (
                name.removeprefix("language_model.")
                in {"model.embed_tokens.weight", "lm_head.weight"}
                and param.ndim == shared.ndim == 2
                and param.shape[0] >= shared.shape[0]
                and param.shape[1] == shared.shape[1]
            ):
                raise ValueError(f"Shared weight shape mismatch: {name}")
            copied.add(name)
    shared_bytes = copied_bytes = 0
    shared_storages = {}
    with torch.no_grad():
        for name, param in params.items():
            shared = weights[name]
            if name in copied:
                param.zero_()
                param[: shared.shape[0]].copy_(shared)
                copied_bytes += param.numel() * param.element_size()
            else:
                param.data = shared
                shared_bytes += param.numel() * param.element_size()
                storage = shared.untyped_storage()
                shared_storages[storage.data_ptr()] = storage.nbytes()
    return {
        "shared_parameter_bytes": shared_bytes,
        "shared_storage_bytes": sum(shared_storages.values()),
        "copied_parameter_bytes": copied_bytes,
        "shared_parameters": len(params) - len(copied),
        "copied_parameters": len(copied),
    }
