"""Match persistent FSDP2 base shards to a colocated vLLM TP group.

Uses public shard placement and load_state_dict(assign=True) before optimizers
exist. FSDP still gathers full layers for training and frees them afterwards;
only its persistent local base shards are exported to the sampler.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any

import torch
from torch.distributed.tensor import DTensor, Shard
from torch.nn.utils import parametrize

from .shared_weight_layout import (
    canonical_name,
    fused_groups,
    is_recurrent_scale,
    serving_name,
    text_config,
)


class SegmentedShardLayout(torch.nn.Module):
    """Store TP segments rank-first; expose ordinary HF order during training.

    The public parametrization API reconstructs the logical weight only after
    FSDP gathers its containing layer. Persistent storage stays evenly sharded.
    These frozen-weight permutations never alter LoRA coordinates.
    """

    def __init__(self, sizes, world_size: int, dim: int, device):
        super().__init__()
        self.dim = dim
        pieces = torch.arange(sum(sizes), device=device).split(sizes)
        order = torch.cat([p.chunk(world_size)[r] for r in range(world_size) for p in pieces])
        self.register_buffer("order", order, persistent=False)
        self.register_buffer("inverse", order.argsort(), persistent=False)

    def forward(self, weight):
        return weight.index_select(self.dim, self.inverse)

    def right_inverse(self, weight):
        return weight.index_select(self.dim, self.order)


def _uncached_expert_parameter(wrapper):
    """Avoid priming PEFT's nested cache before it installs the LoRA proxy.

    ParamWrapper.get_param is a PEFT integration boundary. Directly evaluating
    the public ParametrizationList preserves its transforms without caching the
    base-only value that would otherwise hide the subsequently added LoRA delta.
    """
    base = wrapper.get_base_layer()
    return base.parametrizations[wrapper.parameter_name]()


@contextmanager
def _activate_layout_lora(wrapper, active_adapters):
    """PEFT compatibility: append without caching the pre-LoRA weight.

    Nested ParamWrappers enter torch's parametrization cache before installing
    the inner expert delta. Validation by register_parametrization would cache
    the existing layout's output too early. The proxy preserves shape/dtype;
    unsafe=True skips that validation, not any weight or adapter computation.
    """
    from peft.tuners.lora.layer import _LoraParameterProxy

    deltas = [wrapper.get_delta_weight(a) for a in active_adapters if a in wrapper.lora_A]
    if not deltas:
        yield
        return
    delta = deltas[0]
    for other in deltas[1:]:
        delta = delta + other
    base = wrapper.get_base_layer()
    parametrize.register_parametrization(
        base, wrapper.parameter_name, _LoraParameterProxy(delta), unsafe=True
    )
    try:
        with parametrize.cached():
            yield
    finally:
        wrapper._remove_parametrizations()


def prepare_fsdp_shared_layout(model: Any, world_size: int) -> None:
    config = text_config(model.config)
    # Validate partitions before mutating any storage.
    shared_fsdp_placement(model, world_size)
    for name, param in list(model.named_parameters()):
        logical = canonical_name(name)
        if is_recurrent_scale(logical):
            param.data = param.data.float()
        if world_size == 1 or config.model_type == "qwen3" or ".lora_" in name:
            continue
        dim, sizes = 0, None
        if logical.endswith((".linear_attn.in_proj_qkv.weight", ".linear_attn.conv1d.weight")):
            key = config.linear_num_key_heads * config.linear_key_head_dim
            value = config.linear_num_value_heads * config.linear_value_head_dim
            sizes = [key, key, value]
        elif logical.endswith(".experts.gate_up_proj"):
            dim, sizes = 1, [config.moe_intermediate_size] * 2
        if sizes is not None:
            module_name, parameter_name = name.rsplit(".", 1)
            parametrize.register_parametrization(
                model.get_submodule(module_name),
                parameter_name,
                SegmentedShardLayout(sizes, world_size, dim, param.device),
            )
    # PEFT wraps each expert parameter separately, so the outer wrapper can
    # already be inside parametrize.cached() when the inner wrapper reads it.
    from peft.tuners.lora.layer import ParamWrapper

    for module in model.modules():
        if isinstance(module, ParamWrapper) and parametrize.is_parametrized(
            module.get_base_layer(), module.parameter_name
        ):
            module.get_param = MethodType(_uncached_expert_parameter, module)
            module._activate_lora = MethodType(_activate_layout_lora, module)


def shared_fsdp_placement(model: Any, world_size: int) -> tuple[dict, set]:
    config = text_config(model.config)
    # TP replicated KV heads and padded vocabulary partitions need different
    # mappings. Reject them rather than binding a plausible but wrong shard.
    dimensions = [config.num_attention_heads, config.num_key_value_heads]
    if config.model_type == "qwen3_5_moe_text":
        dimensions += [config.moe_intermediate_size, config.shared_expert_intermediate_size]
    else:
        dimensions += [config.intermediate_size]
    if config.model_type != "qwen3":
        dimensions += [config.linear_num_key_heads, config.linear_num_value_heads]
    for value in dimensions:
        if value % world_size:
            raise ValueError("Shared FSDP requires evenly partitioned attention heads and MLP")
    if config.vocab_size % (64 * world_size):
        raise ValueError("Shared FSDP requires vocabulary divisible by 64 * TP size")
    placements, ignored = {}, set()
    for name, param in model.named_parameters():
        name = canonical_name(name)
        if ".lora_" in name:
            continue
        if param.requires_grad or (
            param.dtype != torch.bfloat16
            and not (is_recurrent_scale(name) and param.dtype == torch.float32)
        ):
            raise ValueError("Shared FSDP requires frozen BF16 base parameters")
        if name.endswith((".linear_attn.A_log", ".linear_attn.dt_bias")):
            placements[param] = Shard(0)
        elif param.ndim == 1 or name.endswith(
            (".mlp.gate.weight", ".mlp.shared_expert_gate.weight")
        ):
            # vLLM replicates normalization and router weights; these are small
            # and frozen. Large expert matrices remain sharded.
            ignored.add(param)
        elif param.ndim == 2:
            dim = 1 if any(f".{p}." in name for p in ("o_proj", "down_proj", "out_proj")) else 0
            if param.shape[dim] % world_size:
                raise ValueError(f"Shared FSDP cannot evenly shard {name}")
            placements[param] = Shard(dim)
        elif param.ndim == 3 and name.endswith(".linear_attn.conv1d.weight"):
            placements[param] = Shard(0)
        elif param.ndim == 3 and name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
            placements[param] = Shard(1 if name.endswith("gate_up_proj") else 2)
        else:
            raise ValueError(f"Unsupported shared FSDP parameter: {name}")
    return placements, ignored


def pack_fsdp_base_weights(model: Any) -> dict[str, torch.Tensor]:
    """Pack local q/k/v and gate/up shards without changing full tensor values."""
    params = {
        canonical_name(name): (name, param)
        for name, param in model.named_parameters(remove_duplicate=False)
        if ".lora_" not in name
    }
    # PEFT prefix appears here because assign must target the FSDP root.
    prefix = "base_model.model."
    result, updates, consumed = {}, {}, set()
    for parent, names, fused in fused_groups(model.config):
        keys = [f"{prefix}{parent}.{name}.weight" for name in names]
        pieces = [params[key][1] for key in keys]
        if not all(isinstance(p, DTensor) for p in pieces):
            raise ValueError("Expected FSDP DTensor base shards before packing")
        local = [p.to_local().detach() for p in pieces]
        slab = torch.cat(local, dim=0)
        offset = 0
        for key, param, tensor in zip(keys, pieces, local, strict=True):
            view = slab[offset : offset + tensor.shape[0]]
            offset += tensor.shape[0]
            updates[params[key][0]] = DTensor.from_local(
                view,
                device_mesh=param.device_mesh,
                placements=param.placements,
                run_check=False,
                shape=param.shape,
                stride=param.stride(),
            )
        result[f"{parent}.{fused}.weight"] = slab
        consumed.update(keys)
    # FSDP's registered state-load hooks refresh its local all-gather inputs.
    # Mutating only DTensor.to_local().data would leave stale internal storage.
    model.load_state_dict(updates, strict=False, assign=True)
    for key, (_name, param) in params.items():
        if key not in consumed:
            tensor = param.to_local() if isinstance(param, DTensor) else param
            result[serving_name(key.removeprefix(prefix))] = tensor.detach()
    return result
