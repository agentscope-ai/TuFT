"""FSDP2 utility functions for TuFT multi-GPU training."""

from __future__ import annotations

import logging
from abc import ABC
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from packaging import version
from torch.distributed.device_mesh import init_device_mesh


# ---------------------------------------------------------------------------
# Version-aware imports
# ---------------------------------------------------------------------------
if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import (  # noqa: F401 — re-exported
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )
    from torch.distributed.fsdp._fully_shard import _fully_shard as _fully_shard_module
    from torch.distributed.tensor import Shard
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import (  # noqa: F401
        CPUOffloadPolicy,  # pyright: ignore[reportPrivateImportUsage]
        FSDPModule,  # pyright: ignore[reportPrivateImportUsage]
        MixedPrecisionPolicy,  # pyright: ignore[reportPrivateImportUsage]
        fully_shard,  # pyright: ignore[reportPrivateImportUsage]
    )

    _fully_shard_module = torch.distributed._composable.fsdp  # type: ignore[attr-defined]
    from torch.distributed.tensor import Shard
else:
    raise RuntimeError("FSDP2 requires PyTorch >= 2.4")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handles ABC compatibility when applying fully_shard to HuggingFace models.
# ---------------------------------------------------------------------------
@contextmanager
def maybe_patch_fsdp_module(model: nn.Module):
    orig_fsdp_module = _fully_shard_module.FSDPModule

    class _FSDPModuleABC(ABC, orig_fsdp_module):  # type: ignore[misc]
        pass

    try:
        if isinstance(model, ABC):
            _fully_shard_module.FSDPModule = _FSDPModuleABC
        yield
    finally:
        _fully_shard_module.FSDPModule = orig_fsdp_module


# ---------------------------------------------------------------------------
# Wraps transformer sub-modules then root with fully_shard.
# ---------------------------------------------------------------------------
def apply_fsdp2(model: nn.Module, fsdp_kwargs: dict) -> None:
    """Apply FSDP2 sharding to a HuggingFace-style model.

    Wraps each transformer layer (identified by ``_no_split_modules``) and
    standalone ``nn.Embedding`` layers, then wraps the root module.
    """
    no_split_modules = _resolve_no_split_modules(model)

    tie_word_embeddings = getattr(getattr(model, "config", None), "tie_word_embeddings", True)

    modules_to_shard: list[nn.Module] = []
    for _name, module in model.named_modules():
        if module.__class__.__name__ in no_split_modules:
            modules_to_shard.append(module)
        elif isinstance(module, nn.Embedding) and not tie_word_embeddings:
            modules_to_shard.append(module)

    for module in modules_to_shard:
        with maybe_patch_fsdp_module(module):
            fully_shard(module, **fsdp_kwargs)

    with maybe_patch_fsdp_module(model):
        fully_shard(model, **fsdp_kwargs)


def _resolve_no_split_modules(model: nn.Module) -> list[str]:
    """Walk through possible wrapper layers (e.g. PeFT) to find _no_split_modules."""
    no_split = getattr(model, "_no_split_modules", None)
    if no_split:
        return list(no_split)

    inner: object = model
    while hasattr(inner, "model"):
        inner = inner.model  # type: ignore[union-attr]
        no_split = getattr(inner, "_no_split_modules", None)
        if no_split:
            return list(no_split)

    return []


# ---------------------------------------------------------------------------
# Loads a full state dict (rank 0 only) into FSDP2-sharded model via DCP.
# ---------------------------------------------------------------------------
def fsdp2_load_full_state_dict(
    model: nn.Module,
    full_state: dict,
    device_mesh=None,
    cpu_offload=None,
) -> None:
    """Broadcast full state dict from rank 0 into FSDP2-sharded model.

    Directly calls ``set_model_state_dict`` on the already-FSDP2-wrapped model
    with ``broadcast_from_rank0=True`` — rank 0 provides the full state dict
    while other ranks pass an empty dict.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
    )

    if dist.get_rank() == 0:
        set_model_state_dict(model, full_state, options=options)
    else:
        set_model_state_dict(model, {}, options=options)

    for _name, buf in model.named_buffers():
        if buf.device.type != "cuda":
            buf.data = buf.data.to(torch.cuda.current_device())
        dist.broadcast(buf, src=0)


# ---------------------------------------------------------------------------
# Collect the full state dict from an FSDP2 model.
# ---------------------------------------------------------------------------
def get_fsdp2_full_state_dict(
    model: nn.Module, offload_to_cpu: bool = True, rank0_only: bool = True
) -> dict:
    """Collect the full state dict from an FSDP2 model via DCP."""
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=offload_to_cpu,
        broadcast_from_rank0=not rank0_only,
    )
    return get_model_state_dict(model, options=options)


# ---------------------------------------------------------------------------
# Gradient clipping compatible with FSDP2 DTensor parameters.
# ---------------------------------------------------------------------------
def fsdp2_clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach=None,
) -> torch.Tensor:
    """Gradient clipping that works with FSDP2 DTensor parameters."""
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = total_norm.to(torch.cuda.current_device(), non_blocking=True)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


# ---------------------------------------------------------------------------
# DeviceMesh helpers.
# ---------------------------------------------------------------------------
def create_device_mesh(world_size: int, fsdp_size: int = -1):
    """Create a 1-D or 2-D device mesh for FSDP2.

    Args:
        world_size: total number of ranks.
        fsdp_size: FSDP group size. If -1 or >= world_size, use pure FSDP
            (all ranks in one dimension). Otherwise use HSDP (DDP x FSDP).
    """
    if fsdp_size < 0 or fsdp_size >= world_size:
        return init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    return init_device_mesh(
        "cuda",
        mesh_shape=(world_size // fsdp_size, fsdp_size),
        mesh_dim_names=("ddp", "fsdp"),
    )


# ---------------------------------------------------------------------------
# Model offload helpers.
# ---------------------------------------------------------------------------
@torch.no_grad()
def offload_fsdp2_model_to_cpu(model: nn.Module, empty_cache: bool = True) -> None:
    model.cpu()
    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp2_model_to_gpu(model: nn.Module) -> None:
    model.to(torch.cuda.current_device())


# ---------------------------------------------------------------------------
# Optimizer offload helpers.
# ---------------------------------------------------------------------------
@torch.no_grad()
def offload_fsdp_optimizer(optimizer: torch.optim.Optimizer) -> None:
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_fsdp_optimizer(optimizer: torch.optim.Optimizer, device) -> None:
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# Shard placement function.
# ---------------------------------------------------------------------------
def get_shard_placement_fn(fsdp_size: int):
    """Return a placement function that shards on a dimension divisible by fsdp_size."""

    def _fn(param: torch.Tensor):
        for i, s in enumerate(param.shape):
            if s % fsdp_size == 0:
                return Shard(i)
        return Shard(0)

    return _fn


# ---------------------------------------------------------------------------
# FSDP version detection.
# ---------------------------------------------------------------------------
def fsdp_version(model: nn.Module) -> int:
    """Return 1 for FSDP1, 2 for FSDP2, 0 for neither."""
    if isinstance(model, FSDPModule):
        return 2
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(model, FSDP):
            return 1
    except ImportError as exc:
        logger.debug("FSDP1 import unavailable: %s", exc)
    return 0
