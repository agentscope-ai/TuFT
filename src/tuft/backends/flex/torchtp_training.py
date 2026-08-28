from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


@dataclass(frozen=True)
class FlexModelSpec:
    name: str
    path: str
    dtype: str
    vocab_size: int


def resolve_flex_model_spec(name: str, *, model_root: str | None = None) -> FlexModelSpec:
    root = Path(
        model_root
        or os.getenv("TUFT_QWEN3_MODEL_ROOT", "/mnt/cpfs/shared/checkpoints/qwen/qwen3")
    )
    paths = {
        "qwen3-0.6b": root / "Qwen3-0.6B",
        "qwen3-4b": root / "Qwen3-4B-Base",
        "qwen3-32b": root / "qwen3-32B",
    }
    if name not in paths:
        raise ValueError(f"Unknown flex model {name!r}; available={sorted(paths)}")
    path = paths[name]
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return FlexModelSpec(
        name=name,
        path=str(path),
        dtype="bfloat16",
        vocab_size=int(config.vocab_size),
    )


def load_tokenizer_for_spec(spec: FlexModelSpec):
    return AutoTokenizer.from_pretrained(spec.path, trust_remote_code=True)


def make_synthetic_rl_batch(
    tokenizer: Any,
    batch_size: int,
    seq_len: int,
    *,
    device: torch.device | str,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    vocab_size = int(tokenizer.vocab_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, : seq_len // 4] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def _rank_major_pack(parts: list[torch.Tensor], tp_size: int) -> torch.Tensor:
    if tp_size == 1:
        return torch.cat(parts, dim=0)
    chunks_per_part = [torch.chunk(part, tp_size, dim=0) for part in parts]
    rank_chunks = []
    for rank in range(tp_size):
        rank_chunks.extend([chunks[rank] for chunks in chunks_per_part])
    return torch.cat(rank_chunks, dim=0)


class FusedQKVLinear(nn.Linear):
    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, tp_size: int = 1):
        self.q_out = q_proj.out_features
        self.k_out = k_proj.out_features
        self.v_out = v_proj.out_features
        super().__init__(
            q_proj.in_features,
            self.q_out + self.k_out + self.v_out,
            bias=q_proj.bias is not None,
            device=q_proj.weight.device,
            dtype=q_proj.weight.dtype,
        )
        if not q_proj.weight.is_meta:
            self.weight.data.copy_(
                _rank_major_pack(
                    [q_proj.weight.data, k_proj.weight.data, v_proj.weight.data],
                    tp_size,
                )
            )
        if self.bias is not None and q_proj.bias is not None and not q_proj.bias.is_meta:
            self.bias.data.copy_(
                _rank_major_pack([q_proj.bias.data, k_proj.bias.data, v_proj.bias.data], tp_size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class FusedGateUpLinear(nn.Linear):
    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, tp_size: int = 1):
        self.gate_out = gate_proj.out_features
        self.up_out = up_proj.out_features
        super().__init__(
            gate_proj.in_features,
            self.gate_out + self.up_out,
            bias=gate_proj.bias is not None,
            device=gate_proj.weight.device,
            dtype=gate_proj.weight.dtype,
        )
        if not gate_proj.weight.is_meta:
            self.weight.data.copy_(
                _rank_major_pack([gate_proj.weight.data, up_proj.weight.data], tp_size)
            )
        if self.bias is not None and gate_proj.bias is not None and not gate_proj.bias.is_meta:
            self.bias.data.copy_(
                _rank_major_pack([gate_proj.bias.data, up_proj.bias.data], tp_size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def _all_reduce_grad(param: nn.Parameter, group: dist.ProcessGroup | None) -> None:
    if group is not None and param.grad is not None:
        dist.all_reduce(param.grad, group=group)


class _AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
        ctx.group = group
        if group is not None:
            dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


class _FusedQKVLoRAAdapter(nn.Module):
    def __init__(
        self,
        *,
        rank: int,
        alpha: int,
        in_features: int,
        q_out: int,
        k_out: int,
        v_out: int,
        dtype: torch.dtype,
        device: torch.device | int,
        tp_group: dist.ProcessGroup | None,
    ) -> None:
        super().__init__()
        self.scaling = alpha / rank
        self.tp_group = tp_group
        self.q_A = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
        self.q_B = nn.Parameter(torch.empty(q_out, rank, dtype=dtype, device=device))
        self.k_A = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
        self.k_B = nn.Parameter(torch.empty(k_out, rank, dtype=dtype, device=device))
        self.v_A = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
        self.v_B = nn.Parameter(torch.empty(v_out, rank, dtype=dtype, device=device))
        for param in (self.q_A, self.k_A, self.v_A):
            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            param.register_post_accumulate_grad_hook(
                lambda p, group=tp_group: _all_reduce_grad(p, group)
            )
        for param in (self.q_B, self.k_B, self.v_B):
            torch.nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dq = F.linear(F.linear(x, self.q_A), self.q_B)
        dk = F.linear(F.linear(x, self.k_A), self.k_B)
        dv = F.linear(F.linear(x, self.v_A), self.v_B)
        return torch.cat([dq, dk, dv], dim=-1) * self.scaling


class FusedQKVLoRA(nn.Module):
    def __init__(
        self,
        base: FusedQKVLinear,
        rank: int,
        alpha: int,
        q_out: int,
        k_out: int,
        v_out: int,
        tp_group: dist.ProcessGroup | None = None,
        adapter_id: str = "default",
    ) -> None:
        super().__init__()
        self.base = base
        self.q_out = q_out
        self.k_out = k_out
        self.v_out = v_out
        self.tp_group = tp_group
        self.adapters = nn.ModuleDict()
        self._adapter_key_by_id: dict[str, str] = {}
        self.active_adapter: str | None = None
        self._next_adapter_index = 0
        self.add_adapter(adapter_id, rank=rank, alpha=alpha)
        self.set_adapter(adapter_id)

    def _new_adapter_key(self, adapter_id: str) -> str:
        key = f"adapter_{self._next_adapter_index}"
        self._next_adapter_index += 1
        self._adapter_key_by_id[adapter_id] = key
        return key

    def add_adapter(self, adapter_id: str, *, rank: int, alpha: int) -> list[nn.Parameter]:
        if adapter_id in self._adapter_key_by_id:
            raise ValueError(f"Adapter {adapter_id} already exists.")
        base_weight = (
            self.base.weight.to_local()
            if hasattr(self.base.weight, "to_local")
            else self.base.weight
        )
        dtype = base_weight.dtype
        device = torch.cuda.current_device() if base_weight.is_cuda else base_weight.device
        in_features = base_weight.shape[1] if not base_weight.is_meta else self.base.in_features
        key = self._new_adapter_key(adapter_id)
        adapter = _FusedQKVLoRAAdapter(
            rank=rank,
            alpha=alpha,
            in_features=in_features,
            q_out=self.q_out,
            k_out=self.k_out,
            v_out=self.v_out,
            dtype=dtype,
            device=device,
            tp_group=self.tp_group,
        )
        self.adapters[key] = adapter
        return list(adapter.parameters())

    def set_adapter(self, adapter_id: str) -> None:
        if adapter_id not in self._adapter_key_by_id:
            raise ValueError(f"Adapter {adapter_id} not found.")
        self.active_adapter = adapter_id

    def remove_adapter(self, adapter_id: str) -> None:
        key = self._adapter_key_by_id.pop(adapter_id, None)
        if key is None:
            return
        del self.adapters[key]
        if self.active_adapter == adapter_id:
            self.active_adapter = next(iter(self._adapter_key_by_id), None)

    def adapter_parameters(self, adapter_id: str) -> list[nn.Parameter]:
        if adapter_id not in self._adapter_key_by_id:
            raise ValueError(f"Adapter {adapter_id} not found.")
        return list(self.adapters[self._adapter_key_by_id[adapter_id]].parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self.active_adapter is None:
            return base_out
        adapter = self.adapters[self._adapter_key_by_id[self.active_adapter]]
        return base_out + adapter(x)


class _ReplicatedLoRAAdapter(nn.Module):
    def __init__(
        self,
        *,
        rank: int,
        alpha: int,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        device: torch.device | int,
    ) -> None:
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, dtype=dtype, device=device))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


class ReplicatedLoRALinear(nn.Module):
    """LoRA wrapper for replicated full-output linear layers such as lm_head."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: int,
        adapter_id: str = "default",
    ) -> None:
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self._adapter_key_by_id: dict[str, str] = {}
        self.active_adapter: str | None = None
        self._next_adapter_index = 0
        self.add_adapter(adapter_id, rank=rank, alpha=alpha)
        self.set_adapter(adapter_id)

    def _new_adapter_key(self, adapter_id: str) -> str:
        key = f"adapter_{self._next_adapter_index}"
        self._next_adapter_index += 1
        self._adapter_key_by_id[adapter_id] = key
        return key

    def add_adapter(self, adapter_id: str, *, rank: int, alpha: int) -> list[nn.Parameter]:
        if adapter_id in self._adapter_key_by_id:
            raise ValueError(f"Adapter {adapter_id} already exists.")
        base_weight = (
            self.base.weight.to_local()
            if hasattr(self.base.weight, "to_local")
            else self.base.weight
        )
        dtype = base_weight.dtype
        device = torch.cuda.current_device() if base_weight.is_cuda else base_weight.device
        key = self._new_adapter_key(adapter_id)
        adapter = _ReplicatedLoRAAdapter(
            rank=rank,
            alpha=alpha,
            in_features=base_weight.shape[1],
            out_features=base_weight.shape[0],
            dtype=dtype,
            device=device,
        )
        self.adapters[key] = adapter
        return list(adapter.parameters())

    def set_adapter(self, adapter_id: str) -> None:
        if adapter_id not in self._adapter_key_by_id:
            raise ValueError(f"Adapter {adapter_id} not found.")
        self.active_adapter = adapter_id

    def remove_adapter(self, adapter_id: str) -> None:
        key = self._adapter_key_by_id.pop(adapter_id, None)
        if key is None:
            return
        del self.adapters[key]
        if self.active_adapter == adapter_id:
            self.active_adapter = next(iter(self._adapter_key_by_id), None)

    def adapter_parameters(self, adapter_id: str) -> list[nn.Parameter]:
        if adapter_id not in self._adapter_key_by_id:
            raise ValueError(f"Adapter {adapter_id} not found.")
        return list(self.adapters[self._adapter_key_by_id[adapter_id]].parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.active_adapter is None:
            return out
        adapter = self.adapters[self._adapter_key_by_id[self.active_adapter]]
        return out + adapter(x)


class _FusedGateUpLoRAAdapter(nn.Module):
    def __init__(
        self,
        *,
        rank: int,
        alpha: int,
        in_features: int,
        gate_out: int,
        up_out: int,
        dtype: torch.dtype,
        device: torch.device | int,
        tp_group: dist.ProcessGroup | None,
    ) -> None:
        super().__init__()
        self.scaling = alpha / rank
        self.tp_group = tp_group
        self.gate_A = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
        self.gate_B = nn.Parameter(torch.empty(gate_out, rank, dtype=dtype, device=device))
        self.up_A = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
        self.up_B = nn.Parameter(torch.empty(up_out, rank, dtype=dtype, device=device))
        for param in (self.gate_A, self.up_A):
            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            param.register_post_accumulate_grad_hook(
                lambda p, group=tp_group: _all_reduce_grad(p, group)
            )
        for param in (self.gate_B, self.up_B):
            torch.nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_gate = F.linear(F.linear(x, self.gate_A), self.gate_B)
        d_up = F.linear(F.linear(x, self.up_A), self.up_B)
        return torch.cat([d_gate, d_up], dim=-1) * self.scaling


class FusedGateUpLoRA(nn.Module):
    def __init__(
        self,
        base: FusedGateUpLinear,
        rank: int,
        alpha: int,
        gate_out: int,
        up_out: int,
        tp_group: dist.ProcessGroup | None = None,
        adapter_id: str = "default",
    ) -> None:
        super().__init__()
        self.base = base
        self.gate_out = gate_out
        self.up_out = up_out
        self.tp_group = tp_group
        self.adapters = nn.ModuleDict()
        self._adapter_key_by_id: dict[str, str] = {}
        self.active_adapter: str | None = None
        self._next_adapter_index = 0
        self.add_adapter(adapter_id, rank=rank, alpha=alpha)
        self.set_adapter(adapter_id)

    def _new_adapter_key(self, adapter_id: str) -> str:
        key = f"adapter_{self._next_adapter_index}"
        self._next_adapter_index += 1
        self._adapter_key_by_id[adapter_id] = key
        return key

    def add_adapter(self, adapter_id: str, *, rank: int, alpha: int) -> list[nn.Parameter]:
        if adapter_id in self._adapter_key_by_id:
            raise ValueError(f"Adapter {adapter_id} already exists.")
        base_weight = (
            self.base.weight.to_local()
            if hasattr(self.base.weight, "to_local")
            else self.base.weight
        )
        dtype = base_weight.dtype
        device = torch.cuda.current_device() if base_weight.is_cuda else base_weight.device
        in_features = base_weight.shape[1] if not base_weight.is_meta else self.base.in_features
        key = self._new_adapter_key(adapter_id)
        adapter = _FusedGateUpLoRAAdapter(
            rank=rank,
            alpha=alpha,
            in_features=in_features,
            gate_out=self.gate_out,
            up_out=self.up_out,
            dtype=dtype,
            device=device,
            tp_group=self.tp_group,
        )
        self.adapters[key] = adapter
        return list(adapter.parameters())

    def set_adapter(self, adapter_id: str) -> None:
        if adapter_id not in self._adapter_key_by_id:
            raise ValueError(f"Adapter {adapter_id} not found.")
        self.active_adapter = adapter_id

    def remove_adapter(self, adapter_id: str) -> None:
        key = self._adapter_key_by_id.pop(adapter_id, None)
        if key is None:
            return
        del self.adapters[key]
        if self.active_adapter == adapter_id:
            self.active_adapter = next(iter(self._adapter_key_by_id), None)

    def adapter_parameters(self, adapter_id: str) -> list[nn.Parameter]:
        if adapter_id not in self._adapter_key_by_id:
            raise ValueError(f"Adapter {adapter_id} not found.")
        return list(self.adapters[self._adapter_key_by_id[adapter_id]].parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self.active_adapter is None:
            return base_out
        adapter = self.adapters[self._adapter_key_by_id[self.active_adapter]]
        return base_out + adapter(x)


class _RowwiseLoRAAdapter(nn.Module):
    def __init__(
        self,
        *,
        rank: int,
        alpha: int,
        local_in_features: int,
        out_features: int,
        dtype: torch.dtype,
        device: torch.device | int,
        tp_group: dist.ProcessGroup | None,
    ) -> None:
        super().__init__()
        self.scaling = alpha / rank
        self.tp_group = tp_group
        self.lora_A = nn.Parameter(torch.empty(rank, local_in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, dtype=dtype, device=device))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        self.lora_B.register_post_accumulate_grad_hook(
            lambda p, group=tp_group: _all_reduce_grad(p, group)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return _AllReduce.apply(delta, self.tp_group)


class RowwiseLoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: int,
        local_in_features: int,
        out_features: int,
        tp_group: dist.ProcessGroup | None = None,
        adapter_id: str = "default",
    ) -> None:
        super().__init__()
        self.base = base
        self.local_in_features = local_in_features
        self.out_features = out_features
        self.tp_group = tp_group
        self.adapters = nn.ModuleDict()
        self._adapter_key_by_id: dict[str, str] = {}
        self.active_adapter: str | None = None
        self._next_adapter_index = 0
        self.add_adapter(adapter_id, rank=rank, alpha=alpha)
        self.set_adapter(adapter_id)

    def _new_adapter_key(self, adapter_id: str) -> str:
        key = f"adapter_{self._next_adapter_index}"
        self._next_adapter_index += 1
        self._adapter_key_by_id[adapter_id] = key
        return key

    def add_adapter(self, adapter_id: str, *, rank: int, alpha: int) -> list[nn.Parameter]:
        if adapter_id in self._adapter_key_by_id:
            raise ValueError(f"Adapter {adapter_id} already exists.")
        base_weight = (
            self.base.weight.to_local()
            if hasattr(self.base.weight, "to_local")
            else self.base.weight
        )
        dtype = base_weight.dtype
        device = torch.cuda.current_device() if base_weight.is_cuda else base_weight.device
        key = self._new_adapter_key(adapter_id)
        adapter = _RowwiseLoRAAdapter(
            rank=rank,
            alpha=alpha,
            local_in_features=self.local_in_features,
            out_features=self.out_features,
            dtype=dtype,
            device=device,
            tp_group=self.tp_group,
        )
        self.adapters[key] = adapter
        return list(adapter.parameters())

    def set_adapter(self, adapter_id: str) -> None:
        if adapter_id not in self._adapter_key_by_id:
            raise ValueError(f"Adapter {adapter_id} not found.")
        self.active_adapter = adapter_id

    def remove_adapter(self, adapter_id: str) -> None:
        key = self._adapter_key_by_id.pop(adapter_id, None)
        if key is None:
            return
        del self.adapters[key]
        if self.active_adapter == adapter_id:
            self.active_adapter = next(iter(self._adapter_key_by_id), None)

    def adapter_parameters(self, adapter_id: str) -> list[nn.Parameter]:
        if adapter_id not in self._adapter_key_by_id:
            raise ValueError(f"Adapter {adapter_id} not found.")
        return list(self.adapters[self._adapter_key_by_id[adapter_id]].parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.active_adapter is None:
            return out
        adapter = self.adapters[self._adapter_key_by_id[self.active_adapter]]
        return out + adapter(x)


def fused_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    qkv = self.qkv_proj(hidden_states)
    q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
    query_states = self.q_norm(q.view(*input_shape, -1, self.head_dim)).transpose(1, 2)
    key_states = self.k_norm(k.view(*input_shape, -1, self.head_dim)).transpose(1, 2)
    value_states = v.view(*input_shape, -1, self.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def fused_mlp_forward(self, x):
    gate_up = self.gate_up_proj(x)
    gate, up = torch.split(gate_up, [self.gate_size, self.up_size], dim=-1)
    return self.down_proj(self.act_fn(gate) * up)


def fuse_qwen3_model_inplace(model: nn.Module, tp_size: int = 1) -> nn.Module:
    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        attn.q_size = attn.q_proj.out_features
        attn.kv_size = attn.k_proj.out_features
        attn.qkv_proj = FusedQKVLinear(attn.q_proj, attn.k_proj, attn.v_proj, tp_size=tp_size)
        del attn.q_proj, attn.k_proj, attn.v_proj
        attn.forward = MethodType(fused_attention_forward, attn)
        mlp.gate_size = mlp.gate_proj.out_features
        mlp.up_size = mlp.up_proj.out_features
        mlp.gate_up_proj = FusedGateUpLinear(mlp.gate_proj, mlp.up_proj, tp_size=tp_size)
        del mlp.gate_proj, mlp.up_proj
        mlp.forward = MethodType(fused_mlp_forward, mlp)
    return model


def _ensure_distributed_for_tp() -> None:
    """Initialize torch.distributed and set CUDA device for TP when launched via torchrun.

    The TuFT server can be launched with ``torchrun --nproc_per_node=N`` to enable
    TP > 1 for the training model. Each process gets ``RANK``, ``WORLD_SIZE`` and
    ``LOCAL_RANK`` env vars from torchrun. This helper sets the CUDA device and
    initializes the process group so that ``init_device_mesh`` succeeds.
    """
    import os

    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        return
    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")
    if rank is None or world_size is None:
        return  # not launched via torchrun
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        torch.cuda.set_device(int(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29555")
    dist.init_process_group(backend="nccl")


def _materialize_meta_tensors(model: Any) -> int:
    """Replace any meta-device params/buffers with real (zero-initialized) tensors.

    transformers can leave freshly-initialized weights (e.g. an untied lm_head
    whose data is missing from the checkpoint) on the meta device, and that
    surfaces non-deterministically across repeated loads in one process. A
    .to(cuda) on a meta tensor raises NotImplementedError, so materialize them
    first. Tied weights are copied from their source instead of being
    re-initialized so lm_head stays identical to embed_tokens.
    """
    import torch.nn as nn

    tied = getattr(model.config, "tie_word_embeddings", False)
    embed = getattr(getattr(model, "model", None), "embed_tokens", None)
    lm_head = getattr(model, "lm_head", None)
    fixed = 0

    def _fix(module: nn.Module) -> None:
        nonlocal fixed
        for name, param in list(module._parameters.items()):
            if param is not None and param.device.type == "meta":
                is_tied_head = (
                    tied
                    and module is lm_head
                    and name == "weight"
                    and embed is not None
                    and embed.weight.device.type != "meta"
                )
                module._parameters[name] = (
                    nn.Parameter(embed.weight.data.clone(), requires_grad=param.requires_grad)
                    if is_tied_head
                    else nn.Parameter(
                        torch.empty(param.shape, dtype=param.dtype).requires_grad_(
                            param.requires_grad
                        ),
                        requires_grad=param.requires_grad,
                    )
                )
                fixed += 1
        for name, buf in list(module._buffers.items()):
            if buf is not None and buf.device.type == "meta":
                module._buffers[name] = torch.empty(buf.shape, dtype=buf.dtype)
                fixed += 1

    model.apply(_fix)
    return fixed


def load_fused_torchtp_model(
    spec: FlexModelSpec,
    rank: int,
    world_size: int,
    attn_implementation: str = "sdpa",
):
    dtype = torch.bfloat16 if spec.dtype == "bfloat16" else torch.float16
    # When launched via torchrun (or the CLI's internal distributed relaunch),
    # initialize distributed before moving the model to CUDA so each rank uses
    # LOCAL_RANK. If no distributed env exists, fall back to a single-process
    # TP=1 training model (used by unit tests / single-GPU mode).
    import os
    import torch.distributed as dist

    distributed_env = os.environ.get("RANK") is not None and os.environ.get("WORLD_SIZE") is not None
    # low_cpu_mem_usage leaves newly-initialized tied weights (e.g. lm_head for
    # Qwen3-4B-Base) as meta tensors, and .to(cuda) then fails with "Cannot
    # copy out of meta tensor" on later loads in the same process (DP unified
    # engine loads one replica per GPU). Always use the eager load path; one
    # copy fits comfortably in host RAM.
    model = AutoModelForCausalLM.from_pretrained(
        spec.path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        attn_implementation=attn_implementation,
    )
    model.config.use_cache = False

    if world_size > 1 and distributed_env:
        _ensure_distributed_for_tp()
    single_process = not (dist.is_available() and dist.is_initialized())
    effective_tp = 1 if single_process else world_size

    # Defensive: materialize any meta tensors left behind by transformers'
    # missing-weight init (observed on repeated loads in one process) before
    # the CUDA move, otherwise .to() raises NotImplementedError.
    if _materialize_meta_tensors(model) > 0 and (rank == 0 or single_process):
        print("  [FusedTorchTP] materialized meta tensors before CUDA move", flush=True)

    model = model.to(torch.cuda.current_device())
    fuse_qwen3_model_inplace(model, tp_size=effective_tp)

    if effective_tp <= 1:
        # Single-GPU path: no distributed init or parallelize_module needed.
        for layer in model.model.layers:
            attn = layer.self_attn
            mlp = layer.mlp
            attn.q_size = attn.qkv_proj.q_out
            attn.kv_size = attn.qkv_proj.k_out
            mlp.gate_size = mlp.gate_up_proj.gate_out
            mlp.up_size = mlp.gate_up_proj.up_out
        for param in model.parameters():
            param.requires_grad_(False)
        mem_gb = torch.cuda.memory_allocated() / (1024**3)
        if single_process and world_size > 1:
            print(
                f"  [FusedTorchTP] loaded TP=1 (single-process, vLLM TP={world_size}), "
                f"VRAM={mem_gb:.1f}GB",
                flush=True,
            )
        else:
            print(f"  [FusedTorchTP] loaded TP=1 (single-GPU), VRAM={mem_gb:.1f}GB", flush=True)
        return model, None, None

    _ensure_distributed_for_tp()
    mesh = init_device_mesh("cuda", (world_size,))
    tp_group = mesh.get_group()
    num_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        parallelize_module(attn.qkv_proj, mesh, ColwiseParallel())
        parallelize_module(attn.o_proj, mesh, RowwiseParallel())
        parallelize_module(mlp.gate_up_proj, mesh, ColwiseParallel())
        parallelize_module(mlp.down_proj, mesh, RowwiseParallel())
        attn.q_size = attn.q_size // world_size
        attn.kv_size = attn.kv_size // world_size
        mlp.gate_size = mlp.gate_size // world_size
        mlp.up_size = mlp.up_size // world_size
    model.config.num_attention_heads = num_heads // world_size
    model.config.num_key_value_heads = num_kv_heads // world_size
    for param in model.parameters():
        param.requires_grad_(False)
    if rank == 0:
        mem_gb = torch.cuda.memory_allocated() / (1024**3)
        print(f"  [FusedTorchTP] loaded TP={world_size}, VRAM={mem_gb:.1f}GB", flush=True)
    return model, tp_group, mesh


def apply_fused_torchtp_lora(
    model: nn.Module,
    *,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    tp_group: dist.ProcessGroup | None = None,
    lora_id: str = "default",
    train_attn: bool = True,
    train_mlp: bool = True,
    train_unembed: bool = False,
) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for layer in model.model.layers:
        attn = layer.self_attn
        if train_attn:
            q_local = attn.q_size
            kv_local = attn.kv_size
            if isinstance(attn.qkv_proj, FusedQKVLoRA):
                params.extend(
                    attn.qkv_proj.add_adapter(lora_id, rank=lora_rank, alpha=lora_alpha)
                )
                attn.qkv_proj.set_adapter(lora_id)
            else:
                attn.qkv_proj = FusedQKVLoRA(
                    attn.qkv_proj,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    q_out=q_local,
                    k_out=kv_local,
                    v_out=kv_local,
                    tp_group=tp_group,
                    adapter_id=lora_id,
                )
                params.extend(attn.qkv_proj.adapter_parameters(lora_id))
            if isinstance(attn.o_proj, RowwiseLoRALinear):
                params.extend(
                    attn.o_proj.add_adapter(lora_id, rank=lora_rank, alpha=lora_alpha)
                )
                attn.o_proj.set_adapter(lora_id)
            else:
                o_weight = (
                    attn.o_proj.weight.to_local()
                    if hasattr(attn.o_proj.weight, "to_local")
                    else attn.o_proj.weight
                )
                attn.o_proj = RowwiseLoRALinear(
                    attn.o_proj,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    local_in_features=o_weight.shape[1],
                    out_features=o_weight.shape[0],
                    tp_group=tp_group,
                    adapter_id=lora_id,
                )
                params.extend(attn.o_proj.adapter_parameters(lora_id))

        mlp = getattr(layer, "mlp", None)
        if train_mlp and mlp is not None:
            if isinstance(mlp.gate_up_proj, FusedGateUpLoRA):
                params.extend(
                    mlp.gate_up_proj.add_adapter(lora_id, rank=lora_rank, alpha=lora_alpha)
                )
                mlp.gate_up_proj.set_adapter(lora_id)
            else:
                mlp.gate_up_proj = FusedGateUpLoRA(
                    mlp.gate_up_proj,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    gate_out=mlp.gate_size,
                    up_out=mlp.up_size,
                    tp_group=tp_group,
                    adapter_id=lora_id,
                )
                params.extend(mlp.gate_up_proj.adapter_parameters(lora_id))
            if isinstance(mlp.down_proj, RowwiseLoRALinear):
                params.extend(
                    mlp.down_proj.add_adapter(lora_id, rank=lora_rank, alpha=lora_alpha)
                )
                mlp.down_proj.set_adapter(lora_id)
            else:
                down_weight = (
                    mlp.down_proj.weight.to_local()
                    if hasattr(mlp.down_proj.weight, "to_local")
                    else mlp.down_proj.weight
                )
                mlp.down_proj = RowwiseLoRALinear(
                    mlp.down_proj,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    local_in_features=down_weight.shape[1],
                    out_features=down_weight.shape[0],
                    tp_group=tp_group,
                    adapter_id=lora_id,
                )
                params.extend(mlp.down_proj.adapter_parameters(lora_id))

    if train_unembed and hasattr(model, "lm_head") and isinstance(model.lm_head, ReplicatedLoRALinear):
        params.extend(model.lm_head.add_adapter(lora_id, rank=lora_rank, alpha=lora_alpha))
        model.lm_head.set_adapter(lora_id)
    elif train_unembed and hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        model.lm_head = ReplicatedLoRALinear(
            model.lm_head,
            rank=lora_rank,
            alpha=lora_alpha,
            adapter_id=lora_id,
        )
        params.extend(model.lm_head.adapter_parameters(lora_id))
    return params


def set_fused_torchtp_lora_adapter(model: nn.Module, lora_id: str) -> None:
    for module in model.modules():
        if isinstance(
            module,
            (FusedQKVLoRA, FusedGateUpLoRA, RowwiseLoRALinear, ReplicatedLoRALinear),
        ):
            module.set_adapter(lora_id)


def fused_adapter_peft_state_dict(model: nn.Module, lora_id: str) -> dict[str, torch.Tensor]:
    """Export one fused LoRA adapter as PEFT-format tensors.

    Produces the standard PEFT key layout
    ``base_model.model.model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_{A,B}.weight``
    so the result can be written as ``adapter_model.safetensors`` and loaded by
    vLLM's native multi-LoRA serving path (``LoRARequest``). The base weights
    are never modified: LoRA deltas stay separate, matching the semantics of
    HF/FSDP training backends + VLLMSamplingBackend.

    Only valid for TP=1 (full, unsharded adapter matrices).
    """
    tensors: dict[str, torch.Tensor] = {}
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        return tensors
    for index, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        prefix = f"base_model.model.model.layers.{index}.self_attn"
        qkv = attn.qkv_proj
        if isinstance(qkv, FusedQKVLoRA) and lora_id in qkv._adapter_key_by_id:
            adapter = qkv.adapters[qkv._adapter_key_by_id[lora_id]]
            tensors[f"{prefix}.q_proj.lora_A.weight"] = adapter.q_A.detach().cpu().contiguous()
            tensors[f"{prefix}.q_proj.lora_B.weight"] = adapter.q_B.detach().cpu().contiguous()
            tensors[f"{prefix}.k_proj.lora_A.weight"] = adapter.k_A.detach().cpu().contiguous()
            tensors[f"{prefix}.k_proj.lora_B.weight"] = adapter.k_B.detach().cpu().contiguous()
            tensors[f"{prefix}.v_proj.lora_A.weight"] = adapter.v_A.detach().cpu().contiguous()
            tensors[f"{prefix}.v_proj.lora_B.weight"] = adapter.v_B.detach().cpu().contiguous()
        o_proj = attn.o_proj
        if isinstance(o_proj, RowwiseLoRALinear) and lora_id in o_proj._adapter_key_by_id:
            adapter = o_proj.adapters[o_proj._adapter_key_by_id[lora_id]]
            tensors[f"{prefix}.o_proj.lora_A.weight"] = adapter.lora_A.detach().cpu().contiguous()
            tensors[f"{prefix}.o_proj.lora_B.weight"] = adapter.lora_B.detach().cpu().contiguous()
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        mlp_prefix = f"base_model.model.model.layers.{index}.mlp"
        gate_up = mlp.gate_up_proj
        if isinstance(gate_up, FusedGateUpLoRA) and lora_id in gate_up._adapter_key_by_id:
            adapter = gate_up.adapters[gate_up._adapter_key_by_id[lora_id]]
            tensors[f"{mlp_prefix}.gate_proj.lora_A.weight"] = adapter.gate_A.detach().cpu().contiguous()
            tensors[f"{mlp_prefix}.gate_proj.lora_B.weight"] = adapter.gate_B.detach().cpu().contiguous()
            tensors[f"{mlp_prefix}.up_proj.lora_A.weight"] = adapter.up_A.detach().cpu().contiguous()
            tensors[f"{mlp_prefix}.up_proj.lora_B.weight"] = adapter.up_B.detach().cpu().contiguous()
        down_proj = mlp.down_proj
        if isinstance(down_proj, RowwiseLoRALinear) and lora_id in down_proj._adapter_key_by_id:
            adapter = down_proj.adapters[down_proj._adapter_key_by_id[lora_id]]
            tensors[f"{mlp_prefix}.down_proj.lora_A.weight"] = adapter.lora_A.detach().cpu().contiguous()
            tensors[f"{mlp_prefix}.down_proj.lora_B.weight"] = adapter.lora_B.detach().cpu().contiguous()
    if hasattr(model, "lm_head") and isinstance(model.lm_head, ReplicatedLoRALinear):
        lm_head = model.lm_head
        if lora_id in lm_head._adapter_key_by_id:
            adapter = lm_head.adapters[lm_head._adapter_key_by_id[lora_id]]
            tensors["base_model.model.lm_head.lora_A.weight"] = (
                adapter.lora_A.detach().cpu().contiguous()
            )
            tensors["base_model.model.lm_head.lora_B.weight"] = (
                adapter.lora_B.detach().cpu().contiguous()
            )
    return tensors


def _target_modules_from_peft_tensors(tensors: dict[str, torch.Tensor]) -> list[str]:
    ordered = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ]
    present = []
    for module in ordered:
        marker = f".{module}."
        if any(marker in key or key.endswith(f".{module}.lora_A.weight") for key in tensors):
            present.append(module)
    return present


def write_peft_adapter_dir(
    adapter_dir: Any,
    tensors: dict[str, torch.Tensor],
    *,
    rank: int,
    alpha: int,
    base_model_name_or_path: str,
) -> None:
    """Write PEFT-format adapter files (safetensors + adapter_config.json)."""
    import json
    from pathlib import Path

    from safetensors.torch import save_file

    directory = Path(adapter_dir)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: tensor.contiguous() for key, tensor in tensors.items()},
        directory / "adapter_model.safetensors",
    )
    adapter_config = {
        "base_model_name_or_path": base_model_name_or_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": int(alpha),
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": int(rank),
        "rank_pattern": None,
        "alpha_pattern": None,
        "target_modules": _target_modules_from_peft_tensors(tensors),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    (directory / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))


def _all_gather_cat(
    tensor: torch.Tensor,
    tp_group: dist.ProcessGroup,
    world_size: int,
    dim: int,
) -> torch.Tensor:
    """All-gather a sharded tensor across TP ranks and concatenate along dim."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    src = tensor.detach().to(device).contiguous()
    chunks = [torch.empty_like(src) for _ in range(world_size)]
    dist.all_gather(chunks, src, group=tp_group)
    return torch.cat(chunks, dim=dim).cpu()


def gather_fused_adapter_peft_state_dict(
    model: nn.Module,
    lora_id: str,
    tp_group: dist.ProcessGroup | None = None,
    world_size: int = 1,
) -> dict[str, torch.Tensor]:
    """PEFT export that works for any TP size.

    For TP=1 this equals ``fused_adapter_peft_state_dict``. For TP>1 the
    column-sharded ``q/k/v_proj.lora_B`` shards are all-gathered along dim 0
    and the row-sharded ``o_proj.lora_A`` shards along dim 1, producing full
    PEFT matrices. Replicated tensors (lora_A of q/k/v, lora_B of o) are taken
    from the local rank. This is a collective: all TP ranks must call it with
    the same adapter id.
    """
    local = fused_adapter_peft_state_dict(model, lora_id)
    if world_size <= 1 or tp_group is None:
        return local
    if not (dist.is_available() and dist.is_initialized()):
        return local
    column_sharded_suffixes = (
        "q_proj.lora_B.weight",
        "k_proj.lora_B.weight",
        "v_proj.lora_B.weight",
        "gate_proj.lora_B.weight",
        "up_proj.lora_B.weight",
    )
    gathered: dict[str, torch.Tensor] = {}
    for key, tensor in local.items():
        if key.endswith(column_sharded_suffixes):
            gathered[key] = _all_gather_cat(tensor, tp_group, world_size, dim=0)
        elif key.endswith("o_proj.lora_A.weight") or key.endswith("down_proj.lora_A.weight"):
            gathered[key] = _all_gather_cat(tensor, tp_group, world_size, dim=1)
        else:
            gathered[key] = tensor
    return gathered


def remove_fused_torchtp_lora_adapter(model: nn.Module, lora_id: str) -> None:
    for module in list(model.modules()):
        if isinstance(
            module,
            (FusedQKVLoRA, FusedGateUpLoRA, RowwiseLoRALinear, ReplicatedLoRALinear),
        ):
            module.remove_adapter(lora_id)


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


def _set_parameter_storage(owner: Any, attr: str, value: Any, source: torch.Tensor) -> torch.Tensor:
    if hasattr(value, "_local_tensor"):
        value._local_tensor = source
        value.requires_grad_(False)
        return value._local_tensor
    parameter = nn.Parameter(source, requires_grad=False)
    setattr(owner, attr, parameter)
    return parameter


def alias_training_model_to_keepalive(
    model: Any,
    keepalive_by_key: dict[str, torch.Tensor],
) -> dict[str, Any]:
    aliased = 0
    skipped = 0
    mismatched = 0
    examples: list[Any] = []
    for key, (owner, attr, target) in _weight_target_map(model).items():
        source = keepalive_by_key.get(key)
        if source is None:
            skipped += 1
            continue
        updated = _set_parameter_storage(owner, attr, target, source)
        aliased += 1
        if int(_local_tensor_ref(updated).data_ptr()) != int(source.data_ptr()):
            mismatched += 1
            if len(examples) < 5:
                examples.append(
                    (key, int(_local_tensor_ref(updated).data_ptr()), int(source.data_ptr()), "ptr")
                )
    return {"aliased": aliased, "skipped": skipped, "mismatched": mismatched, "examples": examples}


def materialize_remaining_meta_tensors(model: Any) -> dict[str, Any]:
    materialized = 0
    examples: list[str] = []
    for module_name, module in model.named_modules():
        for name, param in list(module._parameters.items()):
            if param is None or not getattr(param, "is_meta", False):
                continue
            module._parameters[name] = nn.Parameter(
                torch.zeros(
                    tuple(param.shape),
                    dtype=param.dtype,
                    device=torch.cuda.current_device(),
                ),
                requires_grad=False,
            )
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


def build_empty_fused_torchtp_model(spec: FlexModelSpec, world_size: int) -> tuple[Any, Any, None]:
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
        fuse_qwen3_model_inplace(model, tp_size=world_size)
    # In single-process mode (no torch.distributed), tp_group is None.
    if dist.is_available() and dist.is_initialized():
        tp_group = dist.group.WORLD
    else:
        tp_group = None
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
    return model, tp_group, None


def build_training_runtime_from_keepalive(
    spec: FlexModelSpec,
    keepalive_by_key: dict[str, torch.Tensor],
    world_size: int,
    *,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    apply_lora: bool = True,
) -> tuple[Any, Any, None, list[nn.Parameter], dict[str, Any]]:
    model, tp_group, mesh = build_empty_fused_torchtp_model(spec, world_size)
    alias_result = alias_training_model_to_keepalive(model, keepalive_by_key)
    materialize_result = materialize_remaining_meta_tensors(model)
    alias_result["materialized_meta_tensors"] = materialize_result["materialized_meta_tensors"]
    alias_result["materialized_meta_examples"] = materialize_result["examples"]
    # Re-tie lm_head to embed_tokens when tie_word_embeddings=True.
    # alias_training_model_to_keepalive replaces embed_tokens.weight via setattr,
    # which breaks the tie: lm_head.weight still points to the old meta tensor
    # and gets materialized as zeros. Fix by re-binding after alias.
    if getattr(model.config, "tie_word_embeddings", False) and hasattr(model, "lm_head"):
        model.lm_head.weight = model.model.embed_tokens.weight
    for param in model.parameters():
        param.requires_grad_(False)
    lora_params = []
    if apply_lora:
        lora_params = apply_fused_torchtp_lora(
            model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            tp_group=tp_group,
        )
    return model, tp_group, mesh, lora_params, alias_result


def fused_torchtp_train_step(model: nn.Module, optimizer, batch: dict, sync: bool = True):
    if sync:
        torch.cuda.synchronize()
    start = time.perf_counter()
    model.train()
    optimizer.zero_grad()
    out = model(**batch)
    loss = out.loss
    loss.backward()
    optimizer.step()
    if sync:
        torch.cuda.synchronize()
    return float(loss.item()), (time.perf_counter() - start) * 1000
