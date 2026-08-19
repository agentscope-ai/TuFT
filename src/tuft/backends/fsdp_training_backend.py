"""Multi-node/multi-GPU training via FSDP2 and multi-adapter LoRA."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from packaging import version
from peft import LoraConfig, TaskType, get_peft_model
from tinker import types
from torch.distributed.tensor import DTensor


# FSDP v2 imports (requires PyTorch >= 2.4)
# PyTorch 2.6+ exports from public module; 2.4/2.5 use private _composable.fsdp
if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
elif version.parse(torch.__version__) >= version.parse("2.4"):
    # pyright: ignore[reportPrivateImportUsage]
    from torch.distributed._composable.fsdp import (
        MixedPrecisionPolicy,  # type: ignore[attr-defined]
        fully_shard,  # type: ignore[attr-defined]
    )
else:
    raise ImportError(
        f"FSDP v2 requires PyTorch >= 2.4, but got {torch.__version__}. "
        "Please upgrade PyTorch or use training_backend='hf' instead."
    )
from tuft.backends.base_backend import BaseTrainingBackend
from tuft.backends.fsdp_engine import (
    FSDPModelConfig,
    build_base_model,
    forward_backward as fsdp_forward_backward,
)
from tuft.backends.lora_modules import (
    achievable_target_module_sets,
    gated_deltanet_mismatch_hint,
    get_target_modules,
    resolve_target_modules,
)
from tuft.backends.loss_inputs import (
    FSDP_BACKEND_OWNED_LOSS_INPUTS,
    validate_client_loss_fn_inputs,
)
from tuft.backends.vllm_lora_compat import (
    add_language_model_aliases,
    vllm_nests_language_model,
)
from tuft.checkpoints import CheckpointRecord
from tuft.config import (
    DEFAULT_LORA_ALPHA_RATIO,
    FSDP_QV_TARGET_MODULES,
    ModelConfig,
    compute_lora_alpha,
)
from tuft.exceptions import InvalidRequestException, ResourceExhaustedException


# Matches the slot/adapter name embedded in a PEFT LoRA parameter key, e.g.
# '...lora_A.adapter_r8_3.weight'. Used to canonicalize keys so checkpoints are
# independent of which slot saved/loads them.
_LORA_AB_NAME_RE = re.compile(r"\.lora_(A|B)\.[^.]+\.weight$")


def _canonical_lora_key(key: str) -> str:
    """Strip the embedded slot name from a LoRA parameter key.

    '...lora_A.adapter_r8_3.weight' -> '...lora_A.weight'. A key that is already
    canonical ('...lora_A.weight') is returned unchanged.
    """
    return _LORA_AB_NAME_RE.sub(r".lora_\1.weight", key)


def _copy_full_tensor_into_dtensor(param: DTensor, full_tensor: torch.Tensor) -> None:
    """Copy a full (unsharded) tensor into an FSDP2 DTensor parameter.

    Uses ``distribute_tensor`` with the parameter's own device_mesh/placements so the
    sharding matches FSDP2 semantics exactly (handles non-divisible dim0 and
    dim0 < world_size correctly, unlike manual equal-chunk slicing).
    """
    from torch.distributed.tensor import distribute_tensor

    local_device = param.to_local().device
    full = full_tensor.to(device=local_device, dtype=param.dtype)
    sharded = distribute_tensor(full, device_mesh=param.device_mesh, placements=param.placements)
    param.to_local().copy_(sharded.to_local())


def _shard_list(xs: list[Any], n_shards: int) -> list[list[Any]]:
    """Split xs into n_shards contiguous shards (order-preserving)."""
    if n_shards <= 0:
        raise ValueError(f"n_shards must be > 0, got {n_shards}")
    total = len(xs)
    base = total // n_shards
    rem = total % n_shards
    shards = []
    start = 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        shards.append(xs[start : start + size])
        start += size
    return shards


def _merge_metrics(
    results: list[dict[str, Any]], weights: list[int] | None = None
) -> dict[str, Any]:
    """Merge per-shard metrics.

    ``:sum`` metrics are summed; ``:mean`` metrics are averaged weighted by the
    corresponding shard size (``weights``) so unequal shards do not skew the mean
    (matches the HF backend, which weights by micro-batch size).
    """
    merged: dict[str, Any] = {}
    mean_acc: dict[str, list[float]] = {}

    for idx, out in enumerate(results):
        w = float(weights[idx]) if weights is not None and idx < len(weights) else 1.0
        metrics = out.get("metrics", {}) or {}
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue

            if k.endswith(":sum"):
                merged[k] = merged.get(k, 0.0) + float(v)
            elif k.endswith(":mean"):
                acc = mean_acc.setdefault(k, [0.0, 0.0])
                acc[0] += float(v) * w
                acc[1] += w
            else:
                merged[k] = merged.get(k, 0.0) + float(v)

    for k, (weighted_sum, total_weight) in mean_acc.items():
        merged[k] = weighted_sum / total_weight if total_weight else 0.0

    return merged


# Default port for torch.distributed init (multi-GPU). ModelConfig.fsdp_master_port should match.
DEFAULT_MASTER_PORT = 29500

_DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _fsdp_logprobs_to_loss_fn_outputs(
    engine_output: Dict[str, Any],
    data: list[types.Datum],
) -> list[Dict[str, Any]]:
    """Convert engine log-prob tensors to per-datum Tinker outputs."""

    model_output = (engine_output or {}).get("model_output") or {}
    per_sample = model_output.get("log_probs") or []
    if len(per_sample) != len(data):
        raise RuntimeError(
            f"FSDP engine returned {len(per_sample)} log-prob rows for {len(data)} datums"
        )
    return [
        {"logprobs": types.TensorData.from_torch(t.detach().cpu().float().clone())}
        for t in per_sample
    ]


def _write_adapter_weights_file(
    logger: logging.Logger, adapter_name: str, path: Path, peft_state: dict
) -> None:
    """Write adapter weights as safetensors, falling back to ``.bin`` on failure.

    Always removes the other format first so vLLM (which prefers safetensors
    over .bin) can never pick a stale file left behind by a previous save at
    the same checkpoint path.
    """
    try:
        from safetensors.torch import save_file

        (path / "adapter_model.bin").unlink(missing_ok=True)
        save_file(peft_state, path / "adapter_model.safetensors")
    except Exception as e:
        logger.warning(
            "safetensors save failed for adapter '%s', falling back to .bin: %s",
            adapter_name,
            e,
        )
        (path / "adapter_model.safetensors").unlink(missing_ok=True)
        torch.save(peft_state, path / "adapter_model.bin")


# =============================================================================
# Slot configuration and worker-internal data structures
# =============================================================================


@dataclass
class SlotPoolConfig:
    """Multi-adapter slot pool configuration (rank -> number of slots)."""

    rank_slots: Dict[int, int] = field(default_factory=lambda: {8: 16, 16: 8})
    lora_alpha_ratio: int = DEFAULT_LORA_ALPHA_RATIO
    target_modules: List[str] = field(default_factory=lambda: list(_DEFAULT_TARGET_MODULES))

    def get_lora_alpha(self, rank: int) -> int:
        return compute_lora_alpha(rank, self.lora_alpha_ratio)


@dataclass
class AdapterInfo:
    """Per-adapter metadata and optimizer."""

    name: str
    rank: int
    lora_alpha: int
    target_modules: List[str]
    optimizer: Any = None
    step_count: int = 0


# =============================================================================
# Serializable config for Ray actors
# =============================================================================


def _get_rank_slots_from_config(config: ModelConfig) -> Dict[int, int]:
    """Get rank_slots from ModelConfig (config preferred; otherwise default).

    rank_slots: LoRA rank -> number of adapter slots (concurrent adapters of that rank).
    The default capacity is the same for every target geometry. Widening the
    geometry does multiply per-slot adapter memory, but that memory is a small
    fraction of the base model a slot is attached to (a few percent of the
    frozen weights for the widest supported set), so scaling capacity down to
    hold it constant costs far more tenancy than it saves. Operators who need a
    different trade set ``fsdp_rank_slots`` after measuring their own model.
    """
    max_rank = getattr(config, "max_lora_rank", 8)
    raw = getattr(config, "fsdp_rank_slots", None)
    if raw and len(raw) > 0:
        return {int(k): v for k, v in raw.items()}

    if max_rank <= 8:
        return {max_rank: 16}
    return {8: 16, max_rank: 8}


def _get_target_modules_from_config(config: ModelConfig) -> List[str]:
    """Resolve the homogeneous target geometry allocated by the FSDP slot pool."""

    if config.fsdp_qv_only:
        return list(FSDP_QV_TARGET_MODULES)
    explicit = getattr(config, "fsdp_target_modules", None)
    if explicit is not None:
        return list(explicit)

    try:
        modules = resolve_target_modules(
            str(config.model_path),
            train_attn=getattr(config, "fsdp_train_attn", True),
            train_mlp=getattr(config, "fsdp_train_mlp", True),
            train_unembed=getattr(config, "fsdp_train_unembed", True),
        )
    except ValueError as exc:
        raise ValueError(
            "Cannot infer FSDP LoRA target modules for model "
            f"{config.model_path}: {exc}. FSDP must allocate its adapter geometry "
            "before sharding; configure fsdp_target_modules explicitly."
        ) from exc
    if not modules:
        raise ValueError(
            "FSDP LoRA slot geometry resolved to no target modules; enable at least one "
            "of fsdp_train_attn/fsdp_train_mlp/fsdp_train_unembed or set "
            "fsdp_target_modules explicitly."
        )
    return modules


def _validate_explicit_target_modules(config: ModelConfig) -> None:
    """Stop startup when no client request can match the configured module list.

    ``fsdp_target_modules`` is still the escape hatch for model families the
    resolver does not know; those skip this check. For known families,
    ``_validate_lora_config`` requires the client's module list to equal the
    slot pool's list exactly. An outdated list — for example one from before
    issue #149 added the Qwen3.5/3.8 ``linear_attn.*`` modules — would
    otherwise boot a server that rejects every create_training_run.
    """

    if config.fsdp_qv_only:
        return
    explicit = getattr(config, "fsdp_target_modules", None)
    if explicit is None:
        return
    achievable = achievable_target_module_sets(str(config.model_path))
    if achievable is None:
        return
    explicit_set = set(explicit)
    if any(explicit_set == set(modules) for modules in achievable):
        return
    hint = next(
        (
            notice
            for modules in achievable
            if (notice := gated_deltanet_mismatch_hint(explicit, modules)) is not None
        ),
        None,
    )
    achievable_summary = " or ".join(str(sorted(set(modules))) for modules in achievable)
    raise ValueError(
        f"fsdp_target_modules={sorted(explicit_set)} for model "
        f"'{config.model_name}' ({config.model_path}) cannot be requested by any client: "
        f"client LoRA flags can only produce {achievable_summary}, so every "
        "create_training_run would fail. Set fsdp_target_modules to one of those "
        "lists, or remove it to use the default, then restart the server."
        + (f" {hint}" if hint else "")
    )


def _config_to_worker_dict(config: ModelConfig) -> dict:
    """Convert ModelConfig to the serializable subset needed by Ray workers."""

    target_modules = _get_target_modules_from_config(config)
    rank_slots = _get_rank_slots_from_config(config)
    return {
        "model_path": str(config.model_path),
        "max_model_len": config.max_model_len,
        "fsdp_override_config": dict(getattr(config, "fsdp_override_config", None) or {}),
        "attn_implementation": getattr(config, "attn_implementation", None),
        "slot_config": {
            "rank_slots": rank_slots,
            "lora_alpha_ratio": int(getattr(config, "lora_alpha_ratio", DEFAULT_LORA_ALPHA_RATIO)),
            "target_modules": target_modules,
        },
    }


def _worker_dict_to_configs(config_dict: dict) -> tuple[FSDPModelConfig, SlotPoolConfig]:
    """Build the torch-native model and slot configurations inside an actor."""

    override = dict(config_dict.get("fsdp_override_config") or {})
    attn_implementation = override.pop(
        "attn_implementation",
        config_dict.get("attn_implementation") or "sdpa",
    )
    logging.getLogger(__name__).info(
        "[FSDPTrainingBackend] Loading %s with attn_implementation=%s",
        config_dict.get("model_path"),
        attn_implementation,
    )
    model_config = FSDPModelConfig(
        path=config_dict["model_path"],
        max_model_len=int(config_dict["max_model_len"]),
        attn_implementation=attn_implementation,
        override_config=override,
    )
    sc = config_dict.get("slot_config") or {}
    slot_config = SlotPoolConfig(
        rank_slots=dict(sc.get("rank_slots", {8: 16})),
        lora_alpha_ratio=int(sc.get("lora_alpha_ratio", DEFAULT_LORA_ALPHA_RATIO)),
        target_modules=list(sc.get("target_modules", _DEFAULT_TARGET_MODULES)),
    )
    return model_config, slot_config


# =============================================================================
# MultiAdapterFSDPWorker
# =============================================================================
# Owns the sharded PEFT module, adapter slots, per-adapter optimizers, and checkpoints.
# =============================================================================


class MultiAdapterFSDPWorker:
    """Own a multi-LoRA FSDP2 module and its per-adapter training state."""

    def __init__(
        self,
        model_config: FSDPModelConfig,
        slot_config: SlotPoolConfig,
    ):
        self.model_config = model_config
        self.slot_config = slot_config
        self.module: Any = None
        self._adapters: Dict[str, AdapterInfo] = {}
        self._adapters_by_rank: Dict[int, List[str]] = {}
        self._name_counter: Dict[int, int] = {}
        self._allocated: Dict[str, bool] = {}
        self._initialized = False
        self.logger = logging.getLogger(f"{__name__}.MultiAdapterFSDPWorker")

    def _generate_adapter_name(self, rank: int) -> str:
        if rank not in self._name_counter:
            self._name_counter[rank] = 0
        idx = self._name_counter[rank]
        self._name_counter[rank] += 1
        return f"adapter_r{rank}_{idx}"

    def initialize(self) -> None:
        """Build the base model, PEFT adapter pool, FSDP2 module, and first optimizer."""

        if self._initialized:
            return
        base_model = build_base_model(self.model_config)

        peft_model = None
        for rank, count in self.slot_config.rank_slots.items():
            lora_alpha = self.slot_config.get_lora_alpha(rank)
            for _ in range(count):
                name = self._generate_adapter_name(rank)
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=rank,
                    lora_alpha=lora_alpha,
                    target_modules=list(self.slot_config.target_modules),
                )
                if peft_model is None:
                    peft_model = get_peft_model(
                        base_model,
                        lora_config,
                        adapter_name=name,
                        autocast_adapter_dtype=False,
                    )
                else:
                    peft_model.add_adapter(name, lora_config)
                self._adapters[name] = AdapterInfo(
                    name=name,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    target_modules=list(self.slot_config.target_modules),
                )
                self._adapters_by_rank.setdefault(rank, []).append(name)
                self._allocated[name] = False
        if peft_model is None or not self._adapters:
            raise RuntimeError("slot_config.rank_slots must define at least one slot")

        first = next(iter(self._adapters))
        peft_model.set_adapter(first)
        model_bf16 = peft_model.to(torch.bfloat16)
        model_cuda = model_bf16.cuda()

        # FSDP v2: fully_shard (same as fsdp_standalone_reference.py)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        device_mesh = init_device_mesh("cuda", (world_size,)) if world_size > 1 else None

        transformer_layer_cls_names = getattr(model_cuda, "_no_split_modules", None) or [
            "DecoderLayer",
            "TransformerBlock",
            "LlamaDecoderLayer",
            "Qwen2DecoderLayer",
            "Qwen3DecoderLayer",
        ]
        wrapped_modules = []
        for _name, module in model_cuda.named_modules():
            if module.__class__.__name__ in transformer_layer_cls_names:
                wrapped_modules.append(module)
        for module in wrapped_modules:
            fully_shard(module, mesh=device_mesh, mp_policy=mp_policy)
        fully_shard(model_cuda, mesh=device_mesh, mp_policy=mp_policy)
        self.module = model_cuda
        self._create_optimizer_for_adapter(first)
        self._initialized = True

    def _create_optimizer_for_adapter(self, adapter_name: str) -> None:
        info = self._adapters[adapter_name]
        if info.optimizer is not None:
            return
        self._activate_adapter(adapter_name)
        params = [p for p in self.module.parameters() if p.requires_grad]
        info.optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)

    def _activate_adapter(self, adapter_name: str) -> None:
        """Set PEFT active adapter before computation; same as HFTrainingModel._activate_adapter."""
        module = self.module
        # Handle both FSDP v1 (wrapped) and FSDP v2 (not wrapped) cases
        if hasattr(module, "set_adapter"):
            module.set_adapter(adapter_name)
        elif hasattr(module, "module") and hasattr(module.module, "set_adapter"):
            module.module.set_adapter(adapter_name)
        else:
            raise RuntimeError(f"Cannot find set_adapter method on module: {type(module)}")

    def _reinit_adapter_weights(self, adapter_name: str) -> None:
        """Re-initialize an adapter's LoRA weights to PEFT's fresh-init state.

        lora_A -> kaiming_uniform (a=sqrt(5)); lora_B -> zeros. With lora_B zero the
        adapter contributes nothing on top of the base model (identity delta), i.e. a
        clean start. Works for both plain tensors and FSDP2 DTensors (operates on the
        local shard; every rank resets its own shard).
        """
        _match = f".{adapter_name}."
        with torch.no_grad():
            for name, param in self.module.named_parameters():
                if _match not in name:
                    continue
                data = param.to_local() if isinstance(param, DTensor) else param.data
                if ".lora_B." in name:
                    data.zero_()
                elif ".lora_A." in name:
                    torch.nn.init.kaiming_uniform_(data, a=math.sqrt(5))

    def _reset_adapter_slot(self, adapter_name: str) -> None:
        """Reset a slot to a fresh-adapter state.

        A reused slot would otherwise inherit the previous run's trained LoRA weights,
        Adam momentum / step count, and any unconsumed accumulated gradients. The HF
        backend creates a fresh adapter + fresh AdamW per create_adapter; this keeps
        the two backends consistent.
        """
        info = self._adapters.get(adapter_name)
        if info is None:
            return
        # Drop the optimizer so a fresh AdamW (no stale momentum/step) is recreated.
        info.optimizer = None
        info.step_count = 0
        self._reinit_adapter_weights(adapter_name)
        # Clear any unconsumed accumulated gradients on this adapter's parameters.
        _match = f".{adapter_name}."
        for name, param in self.module.named_parameters():
            if _match in name and param.grad is not None:
                param.grad = None

    def forward_backward(
        self,
        adapter_name: str,
        data: list[types.Datum],
        loss_fn_name: str,
        loss_fn_config: dict[str, float] | None,
        micro_batch_size: int,
        forward_only: bool = False,
        client_keys: list[str] | None = None,
    ) -> Dict[str, Any]:
        """Run forward/backward without stepping or clearing accumulated gradients.

        NOTE: We deliberately do NOT call optimizer.zero_grad() here. Multiple
        consecutive forward_backward calls between two optim_step calls will
        accumulate gradients (standard PyTorch grad-accumulation semantics).
        zero_grad happens inside optim_step() at the end of each training step
        (see optim_step below).

        This is required for two cases:
          1. Per-call internal micro-batch accumulation (callers split a large
             batch into chunks of ModelConfig.micro_batch_size and rely on grads
             persisting across the loop).
          2. Cross-call grad accumulation (e.g. trinity v22: 32 forward_backward
             calls + 1 optim_step), which removes mini-batch SGD intra-step
             off-policy drift in PPO-style RL training.
        """
        self._activate_adapter(adapter_name)
        info = self._adapters[adapter_name]
        if info.optimizer is None:
            self._create_optimizer_for_adapter(adapter_name)
        self.module.train()
        return fsdp_forward_backward(
            self.module,
            data,
            loss_fn_name,
            loss_fn_config,
            micro_batch_size,
            forward_only=forward_only,
            client_keys=client_keys,
        )

    def optim_step(
        self,
        adapter_name: str,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        betas: Optional[tuple[float, float]] = None,
        eps: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._activate_adapter(adapter_name)
        info = self._adapters[adapter_name]
        if info.optimizer is None:
            self._create_optimizer_for_adapter(adapter_name)
        opt = info.optimizer
        if learning_rate is not None:
            for pg in opt.param_groups:
                pg["lr"] = learning_rate
        if weight_decay is not None:
            for pg in opt.param_groups:
                pg["weight_decay"] = weight_decay
        if betas is not None:
            for pg in opt.param_groups:
                pg["betas"] = tuple(betas)
        if eps is not None:
            for pg in opt.param_groups:
                pg["eps"] = eps
        if grad_clip_norm is not None and grad_clip_norm > 0:
            # Clip only this adapter's parameters (those in its optimizer). Clipping
            # self.module.parameters() would also rescale pending accumulated gradients
            # of other adapters/runs sharing the base model (cross-contamination), and
            # would skew this run's own global norm.
            clip_params = [p for pg in opt.param_groups for p in pg["params"]]
            torch.nn.utils.clip_grad_norm_(clip_params, grad_clip_norm)
        opt.step()
        opt.zero_grad()
        info.step_count += 1
        return {"step_count": info.step_count, "adapter": adapter_name}

    def save_checkpoint(self, adapter_name: str, path: str | Path, optimizer: bool = True) -> None:
        """Save adapter.pt (training load_state) + PEFT format (sampling); optional optimizer."""
        self._activate_adapter(adapter_name)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # FSDP v2: parameters may be DTensor, need to call full_tensor() to get full param
        # Collect full state dict for the adapter keyed by slot-independent canonical
        # names, so load works even when the slot name differs (restart + fallback).
        state = {}
        _match = f".{adapter_name}."
        for name, param in self.module.named_parameters():
            if _match in name:
                key = _canonical_lora_key(name)
                if isinstance(param, DTensor):
                    # FSDP v2: gather full tensor from all ranks
                    state[key] = param.full_tensor().cpu().clone()
                else:
                    state[key] = param.data.cpu().clone()

        # Only rank 0 saves to avoid duplicate writes
        import torch.distributed as dist

        is_rank_0 = not dist.is_initialized() or dist.get_rank() == 0

        # Move any existing path/adapter_name/ contents to path/ first, so our writes below
        # are not overwritten by unstripped PEFT files (vLLM requires keys like .lora_A.weight).
        lora_subdir = path / adapter_name
        if is_rank_0 and lora_subdir.exists() and lora_subdir.is_dir():
            for item in lora_subdir.iterdir():
                dest = path / item.name
                if dest.exists():
                    if dest.is_file():
                        dest.unlink()
                    elif dest.is_dir():
                        shutil.rmtree(dest)
                shutil.move(str(item), str(dest))
            lora_subdir.rmdir()

        if is_rank_0:
            # 1) Internal format for FSDP load_state
            torch.save(state, path / "adapter.pt")

            if optimizer:
                info = self._adapters[adapter_name]
                if info.optimizer is not None:
                    torch.save(info.optimizer.state_dict(), path / "optimizer.pt")

            # 2) PEFT format (adapter_config.json, adapter_model.safetensors)
            # so sampling/VLLM can load
            # FSDP v2: PEFT's save_pretrained cannot handle DTensor directly.
            # We manually save the adapter config and weights in PEFT format.
            import json

            # Save adapter_config.json
            # Prefer the runtime peft_config attribute when accessible, but always
            # fall back to constructing a minimal config from AdapterInfo so the
            # file is ALWAYS written regardless of FSDP v2 attribute visibility.
            module = self.module
            has_nested_peft = hasattr(module, "module") and hasattr(module.module, "peft_config")
            peft_model = module.module if has_nested_peft else module

            config_dict: dict | None = None
            if hasattr(peft_model, "peft_config") and adapter_name in peft_model.peft_config:
                runtime_cfg = peft_model.peft_config[adapter_name]
                config_dict = runtime_cfg.to_dict()
                assert config_dict is not None
                for key, value in config_dict.items():
                    if isinstance(value, set):
                        config_dict[key] = list(value)

            if config_dict is None:
                # Fallback: construct a minimal but vLLM-compatible adapter_config.json
                # from the AdapterInfo that was stored during initialize().
                info = self._adapters[adapter_name]
                config_dict = {
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "base_model_name_or_path": self.model_config.path,
                    "r": info.rank,
                    "lora_alpha": info.lora_alpha,
                    "target_modules": list(info.target_modules),
                    "lora_dropout": 0.0,
                    "fan_in_fan_out": False,
                    "bias": "none",
                    "modules_to_save": None,
                    "init_lora_weights": True,
                    "layers_to_transform": None,
                    "layers_pattern": None,
                    "inference_mode": True,
                }
                self.logger.warning(
                    "peft_config not accessible on module after FSDP v2 wrap; "
                    "writing adapter_config.json from AdapterInfo for adapter '%s'",
                    adapter_name,
                )

            with open(path / "adapter_config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

            # Save adapter weights in safetensors format for sampling/vLLM.
            # vLLM lora/utils.py expects keys ending in ".lora_A.weight" or ".lora_B.weight"
            # (parts[-2] in ["lora_A","lora_B"]). `state` is already keyed canonically
            # (slot name stripped), which is exactly the layout vLLM expects.
            peft_state = dict(state)
            # Only emit `language_model.` alias keys for models whose vLLM
            # implementation nests the text backbone under `language_model`
            # (e.g. Qwen3.5); emitting them unconditionally would double the
            # checkpoint size and vLLM GPU load-time memory for every model.
            if vllm_nests_language_model(self.model_config.path):
                peft_state = add_language_model_aliases(peft_state)

            _write_adapter_weights_file(self.logger, adapter_name, path, peft_state)

    def load_checkpoint(
        self,
        adapter_name: str,
        path: str | Path,
        checkpoint_modules: List[str],
        optimizer: bool = True,
    ) -> None:
        path = Path(path)
        expected_modules = self._adapters[adapter_name].target_modules
        if set(checkpoint_modules) != set(expected_modules):
            raise RuntimeError(
                f"Cannot load FSDP checkpoint targeting modules {sorted(checkpoint_modules)} "
                f"into a slot targeting {sorted(expected_modules)}."
            )
        state = torch.load(path / "adapter.pt", map_location="cpu", weights_only=True)
        # Build a slot-name-independent lookup: canonicalize saved keys so a checkpoint
        # saved under ANY slot name loads into the current slot. Without this, a slot
        # name mismatch (common after restart + create_adapter fallback) would silently
        # match zero parameters and continue training from stale weights.
        canonical_state = {_canonical_lora_key(k): v for k, v in state.items()}
        self._activate_adapter(adapter_name)
        matched = 0
        with torch.no_grad():
            _match = f".{adapter_name}."
            for name, param in self.module.named_parameters():
                if _match not in name:
                    continue
                key = _canonical_lora_key(name)
                if key not in canonical_state:
                    continue
                matched += 1
                loaded_tensor = canonical_state[key]
                if isinstance(param, DTensor):
                    _copy_full_tensor_into_dtensor(param, loaded_tensor)
                else:
                    param.data.copy_(loaded_tensor.to(param.device))
        if matched == 0:
            raise RuntimeError(
                f"load_checkpoint matched 0 parameters for adapter '{adapter_name}' from "
                f"{path / 'adapter.pt'}; the checkpoint key layout is incompatible."
            )
        if optimizer:
            opt_path = path / "optimizer.pt"
            if opt_path.exists():
                if self._adapters[adapter_name].optimizer is None:
                    self._create_optimizer_for_adapter(adapter_name)
                opt_state = torch.load(opt_path, map_location="cpu", weights_only=True)
                self._adapters[adapter_name].optimizer.load_state_dict(opt_state)

    def allocate_slot(self, rank: int) -> Optional[str]:
        """Allocate an unused slot for rank, reset it to a fresh state, return name."""
        for name in self._adapters_by_rank.get(rank, []):
            if not self._allocated.get(name, False):
                self._allocated[name] = True
                self._reset_adapter_slot(name)
                return name
        return None

    def reserve_slot(self, adapter_name: str) -> None:
        """Mark a specific slot allocated and reset it.

        Used to synchronize non-lead ranks to the slot chosen by the lead rank, so
        every rank resets weights/optimizer identically for the same adapter name.
        """
        if adapter_name in self._adapters:
            self._allocated[adapter_name] = True
            self._reset_adapter_slot(adapter_name)

    def release_slot(self, adapter_name: str) -> None:
        # Reset on release so a freed slot does not retain the run's trained weights,
        # optimizer state, or gradients (allocate_slot also resets defensively).
        self._reset_adapter_slot(adapter_name)
        self._allocated.pop(adapter_name, None)

    def list_adapters(self) -> List[str]:
        return list(self._adapters.keys())


# =============================================================================
# FSDPWorkerActor: Ray actor, one GPU per process, forms torch.distributed with peers
# =============================================================================


class FSDPWorkerActor:
    """Single-GPU Ray actor; N form process group via init_dist, each holds one worker."""

    def __init__(self, rank: int, world_size: int, config_dict: dict) -> None:
        self.rank = rank
        self.world_size = world_size
        self.config_dict = config_dict
        self._worker: Optional[MultiAdapterFSDPWorker] = None
        self._dist_initialized = False
        self.logger = logging.getLogger(f"{__name__}.FSDPWorkerActor")

    def get_node_ip(self) -> str:
        import ray

        return ray.util.get_node_ip_address()

    def init_dist(self, master_addr: str, master_port: int = DEFAULT_MASTER_PORT) -> None:
        import torch.distributed as dist

        if self._dist_initialized:
            return
        import os

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        # Must set CUDA device before init_process_group to avoid DeviceMesh
        # picking the wrong GPU (PyTorch 2.6+ creates DeviceMesh internally).
        # Each actor is a Ray num_gpus=1 process: Ray sets CUDA_VISIBLE_DEVICES
        # to a single physical GPU that torch sees as cuda:0. Do NOT use the
        # global rank (self.rank) here — rank>=1 would select a nonexistent
        # local device and fail before NCCL init. Exactly one device is visible,
        # so always pin index 0.
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"tcp://{master_addr}:{master_port}",
        )
        self._dist_initialized = True

    def build_worker(self) -> None:
        if self._worker is not None:
            return
        model_config, slot_config = _worker_dict_to_configs(self.config_dict)
        self._worker = MultiAdapterFSDPWorker(
            model_config=model_config,
            slot_config=slot_config,
        )
        logging.info("[SERVER][Actor] build_worker 调用 initialize 前 rank=%s", self.rank)
        self._worker.initialize()
        logging.info("[SERVER][Actor] build_worker initialize 返回 rank=%s", self.rank)

    def allocate_slot(self, rank: int) -> Optional[str]:
        if self._worker is None:
            return None
        return self._worker.allocate_slot(rank)

    def reserve_slot(self, adapter_name: str) -> None:
        if self._worker is not None:
            self._worker.reserve_slot(adapter_name)

    def release_slot(self, adapter_name: str) -> None:
        if self._worker is not None:
            self._worker.release_slot(adapter_name)

    def forward_backward(
        self,
        data: list,
        adapter_name: str,
        loss_fn_name: str,
        loss_fn_config: Optional[dict] = None,
        forward_only: bool = False,
        micro_batch_size: Optional[int] = None,
        client_keys: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """Run forward (+backward) on this actor's data shard.

        data: list[types.Datum] (or dicts after Ray serialization).

        The torch-native engine splits this shard into contiguous micro-batches
        and accumulates gradients without stepping or clearing them. The caller
        guarantees every rank runs the same number of micro-batches so FSDP2
        collectives remain symmetric.

        Caller is responsible for invoking `optim_step` afterwards
        (which will step + zero_grad).
        """
        if data and isinstance(data[0], dict):
            data = [types.Datum(**d) for d in data]

        if not data or self._worker is None:
            return {
                "metrics": {},
                "loss_fn_outputs": [],
            }

        # Keep a single micro-batch when the configured size does not divide
        # the shard. The backend applies the same fallback on every rank.
        if micro_batch_size and micro_batch_size > 0 and len(data) % micro_batch_size == 0:
            mb = micro_batch_size
        else:
            mb = len(data)
        n_micro = len(data) // mb
        out = self._worker.forward_backward(
            adapter_name,
            data,
            loss_fn_name,
            loss_fn_config,
            mb,
            forward_only=forward_only,
            client_keys=client_keys,
        )
        metrics = dict(out.get("metrics") or {})
        metrics["actor/num_micro_batches"] = float(n_micro)
        all_outputs = _fsdp_logprobs_to_loss_fn_outputs(out, data)

        return {
            "metrics": metrics,
            "loss_fn_outputs": all_outputs,
        }

    def optim_step(
        self,
        adapter_name: str,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        betas: Optional[tuple[float, float]] = None,
        eps: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self._worker is None:
            return {}
        return self._worker.optim_step(
            adapter_name, learning_rate, weight_decay, grad_clip_norm, betas, eps
        )

    def save_checkpoint(self, adapter_name: str, path: str, optimizer: bool = True) -> None:
        # FSDP v2: all ranks must participate in full_tensor() collective operation
        if self._worker is None:
            return
        self._worker.save_checkpoint(adapter_name, Path(path), optimizer)

    def load_checkpoint(
        self,
        adapter_name: str,
        path: str,
        checkpoint_modules: List[str],
        optimizer: bool = True,
    ) -> None:
        if self._worker is None:
            return
        self._worker.load_checkpoint(adapter_name, Path(path), checkpoint_modules, optimizer)


# =============================================================================
# FSDPTrainingBackend (implements BaseTrainingBackend)
# Multi-GPU: N GPUs = N FSDPWorkerActor processes, forming torch.distributed.
# =============================================================================


class FSDPTrainingBackend(BaseTrainingBackend):
    """
    Multi-node/multi-GPU training backend; peer to HFTrainingBackend.
    Selected via ModelConfig.training_backend = "fsdp".
    Uses one Ray actor per GPU, forming a process group for FSDP2.
    """

    def __init__(
        self,
        config: ModelConfig,
        fsdp_index: Optional[int] = None,
        worker_venv_path: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self._fsdp_index = fsdp_index  # Index among FSDP models; port = base + fsdp_index
        self._worker_venv_path = worker_venv_path
        self._worker: Optional[MultiAdapterFSDPWorker] = None
        self._actors: List[Any] = []
        self._world_size: int = 0
        self._lora_id_to_adapter_name: Dict[str, str] = {}
        self._adapter_name_to_lora_id: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        # Both the Ray and the single-process path derive their slot pool from this
        # one dict, so they cannot disagree on rank slots, lora_alpha_ratio, or
        # target modules.
        self._config_dict = _config_to_worker_dict(config)
        _, self._slot_config = _worker_dict_to_configs(self._config_dict)
        # A fsdp_target_modules list no client can match should stop the
        # server here, before any create_training_run arrives.
        _validate_explicit_target_modules(config)
        self.logger = logging.getLogger(f"{__name__}.FSDPTrainingBackend")

    async def shutdown(self) -> None:
        """Kill all FSDP worker Ray actors and release GPU resources."""
        import ray

        for actor in self._actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        self._actors = []
        self._worker = None
        self._world_size = 0
        self._lora_id_to_adapter_name.clear()
        self._adapter_name_to_lora_id.clear()

    async def async_init(self) -> None:
        if self._world_size > 0 or self._worker is not None:
            return
        n_gpus = getattr(self.config, "fsdp_num_gpus", 1)
        n_gpus = max(1, int(n_gpus))
        use_ray = os.environ.get("TUFT_FSDP_NO_RAY") != "1"
        if not use_ray and n_gpus != 1:
            raise ValueError(
                "TUFT_FSDP_NO_RAY=1 (no Ray) requires fsdp_num_gpus=1. "
                f"Got fsdp_num_gpus={n_gpus}. Multi-GPU requires Ray."
            )
        if not use_ray:
            # Local single-process: no Ray actors; for standalone tests (train/save logic)
            import torch.distributed as dist

            if not dist.is_available() or not dist.is_initialized():
                import socket

                os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                if "MASTER_PORT" not in os.environ:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(("", 0))
                        os.environ["MASTER_PORT"] = str(s.getsockname()[1])
                os.environ.setdefault("RANK", "0")
                os.environ.setdefault("WORLD_SIZE", "1")
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    rank=0,
                    world_size=1,
                )
            model_config, slot_config = _worker_dict_to_configs(self._config_dict)
            self._worker = MultiAdapterFSDPWorker(
                model_config=model_config,
                slot_config=slot_config,
            )
            await asyncio.to_thread(self._worker.initialize)
            self._world_size = 1
            return
        import ray

        config_dict = self._config_dict
        _venv = self._worker_venv_path
        if not _venv or not _venv.strip():
            self.logger.warning(
                "worker_venv_path is not set. Recommend using a virtual environment for Ray FSDP; "
                "set worker_venv_path in config if all nodes use the same venv. "
                "Proceeding with empty runtime_env (relying on node-installed packages)."
            )
            _runtime_env = {}
        else:
            _path = os.environ.get("PATH", "")
            # Ray uses py_executable so worker uses venv Python; else node Ray may not find tuft
            _venv_python = str(Path(_venv) / "bin" / "python")
            _runtime_env = {
                "py_executable": _venv_python,
                "env_vars": {
                    "VIRTUAL_ENV": _venv,
                    "PATH": f"{_venv}/bin:{_path}",
                },
            }

        actors = []
        for r in range(n_gpus):
            actor = (
                ray.remote(FSDPWorkerActor)
                .options(
                    num_gpus=1,
                    runtime_env=_runtime_env,
                )
                .remote(r, n_gpus, config_dict)
            )
            actors.append(actor)
        # Set _world_size / _actors only after all succeed; else next create_adapter retries init
        # get_node_ip should return quickly; timeout avoids hang when actor not scheduled (e.g. GPU)
        _GET_NODE_IP_TIMEOUT = 120
        self.logger.info("[FSDP] async_init: created %d actors, calling get_node_ip...", n_gpus)
        try:
            master_addr = await asyncio.to_thread(
                ray.get, actors[0].get_node_ip.remote(), timeout=_GET_NODE_IP_TIMEOUT
            )
        except Exception as e:
            self.logger.error("[FSDP] get_node_ip FAILED: %s", e)
            raise
        self.logger.info("[FSDP] get_node_ip OK: %s, calling init_dist...", master_addr)
        base_port = getattr(self.config, "fsdp_master_port", DEFAULT_MASTER_PORT)
        master_port = base_port + self._fsdp_index if self._fsdp_index is not None else base_port
        try:
            await asyncio.gather(
                *[
                    asyncio.to_thread(ray.get, a.init_dist.remote(master_addr, master_port))
                    for a in actors
                ]
            )
        except Exception as e:
            self.logger.error("[FSDP] init_dist FAILED: %s", e)
            raise
        self.logger.info("[FSDP] init_dist OK, calling build_worker...")
        try:
            await asyncio.gather(
                *[asyncio.to_thread(ray.get, a.build_worker.remote()) for a in actors]
            )
        except Exception as e:
            self.logger.error("[FSDP] build_worker FAILED: %s", e)
            raise
        self.logger.info("[FSDP] build_worker OK, FSDP backend ready")
        self._actors = actors
        self._world_size = n_gpus

    def _get_adapter_name(self, lora_id: str) -> str:
        if lora_id not in self._lora_id_to_adapter_name:
            raise ValueError(f"Unknown lora_id: {lora_id}; call create_adapter first.")
        return self._lora_id_to_adapter_name[lora_id]

    def _validate_lora_config(self, lora_config: types.LoraConfig) -> None:
        """Require the request to match the geometry allocated before sharding."""

        if self.config.fsdp_qv_only:
            return
        explicit_modules = self.config.fsdp_target_modules
        try:
            requested_modules = get_target_modules(str(self.config.model_path), lora_config)
        except ValueError as exc:
            # An explicit operator-provided geometry is the escape hatch for
            # unsupported model families, where client modifiers cannot be
            # resolved into a concrete set for comparison.
            if explicit_modules is not None:
                return
            raise InvalidRequestException(str(exc)) from exc
        slot_modules = self._slot_config.target_modules
        if set(requested_modules) == set(slot_modules):
            return
        modifier_summary = (
            f"train_attn={lora_config.train_attn}, train_mlp={lora_config.train_mlp}, "
            f"train_unembed={lora_config.train_unembed}"
        )
        slot_description = (
            f"the server's explicit fsdp_target_modules={sorted(slot_modules)}"
            if explicit_modules is not None
            else f"the preallocated slot pool targets {sorted(slot_modules)}"
        )
        raise InvalidRequestException(
            "FSDP LoRA target-module mismatch: client modifiers "
            f"({modifier_summary}) resolve to {sorted(requested_modules)}, but "
            f"{slot_description}. The training run was not created. Change the client modifiers, "
            "or reconfigure the server's FSDP target geometry and restart it so both module "
            "sets match."
        )

    def _validate_checkpoint_geometry(self, checkpoint_record: CheckpointRecord) -> List[str]:
        """Return validated checkpoint geometry for the worker load."""

        checkpoint_modules = checkpoint_record.saved_target_modules
        if checkpoint_modules is None:
            raise InvalidRequestException(
                f"Cannot load FSDP checkpoint {checkpoint_record.checkpoint_id}: neither "
                "adapter_config.json nor metadata provides a supported explicit "
                "target_modules list. PEFT regex-string targets cannot be validated for "
                "FSDP slot loading."
            )
        slot_modules = self._slot_config.target_modules
        if set(checkpoint_modules) != set(slot_modules):
            hint = gated_deltanet_mismatch_hint(checkpoint_modules, slot_modules)
            raise InvalidRequestException(
                f"Cannot load FSDP checkpoint {checkpoint_record.checkpoint_id} targeting "
                f"modules {sorted(checkpoint_modules)} into a slot targeting "
                f"{sorted(slot_modules)}." + (f" {hint}" if hint else "")
            )
        return checkpoint_modules

    async def create_adapter(self, lora_id: str, lora_config: types.LoraConfig) -> None:
        self._validate_lora_config(lora_config)
        rank = getattr(lora_config, "rank", 8)
        if rank not in self._slot_config.rank_slots:
            raise InvalidRequestException(
                f"FSDP LoRA rank {rank} is not configured. This server preallocates slots "
                f"for ranks {sorted(self._slot_config.rank_slots)}. Choose a configured rank "
                "or update fsdp_rank_slots and restart the server."
            )
        async with self._lock:
            if self._world_size == 0 and self._worker is None and not self._actors:
                await self.async_init()
            if self._worker is not None:
                adapter_name = await asyncio.to_thread(self._worker.allocate_slot, rank)
            elif self._actors:
                import ray

                adapter_name: str | None = await asyncio.to_thread(
                    ray.get, self._actors[0].allocate_slot.remote(rank)
                )
                if adapter_name is not None and len(self._actors) > 1:
                    # Every rank shares the base model; sync the remaining ranks to the
                    # same slot and reset weights/optimizer identically on each of them.
                    await asyncio.gather(
                        *[
                            asyncio.to_thread(ray.get, a.reserve_slot.remote(adapter_name))
                            for a in self._actors[1:]
                        ]
                    )
            else:
                raise RuntimeError("FSDPTrainingBackend not initialized.")
            if adapter_name is None:
                capacity = self._slot_config.rank_slots[rank]
                raise ResourceExhaustedException(
                    f"All {capacity} preallocated FSDP LoRA slots for rank {rank} are in use. "
                    "Remove a training run or retry after a slot is released. To increase "
                    "capacity, update fsdp_rank_slots and restart the server."
                )
            self._lora_id_to_adapter_name[lora_id] = adapter_name
            self._adapter_name_to_lora_id[adapter_name] = lora_id

    async def remove_adapter(self, lora_id: str) -> None:
        async with self._lock:
            adapter_name = self._lora_id_to_adapter_name.pop(lora_id, None)
            if adapter_name:
                self._adapter_name_to_lora_id.pop(adapter_name, None)
                if self._worker is not None:
                    await asyncio.to_thread(self._worker.release_slot, adapter_name)
                elif self._actors:
                    import ray

                    # Release (and reset) the slot on every rank, not just the lead,
                    # so no rank retains the run's weights/optimizer state.
                    await asyncio.gather(
                        *[
                            asyncio.to_thread(ray.get, a.release_slot.remote(adapter_name))
                            for a in self._actors
                        ]
                    )

    async def forward(
        self,
        data: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool = False,
    ) -> types.ForwardBackwardOutput:
        adapter_name = self._get_adapter_name(lora_id)
        loss_fn_name = (
            loss_fn if isinstance(loss_fn, str) else getattr(loss_fn, "__name__", "cross_entropy")
        )
        client_keys = validate_client_loss_fn_inputs(
            data,
            ignored_keys=FSDP_BACKEND_OWNED_LOSS_INPUTS,
        )

        # Per-call internal micro-batch grad accumulation.
        #
        # When ModelConfig.micro_batch_size < len(data) (or, in multi-GPU mode,
        # < shard length), each call splits its (sharded) batch into
        # ceil(len/mb) micro-batches. Each micro-batch runs a full
        # forward+backward, and gradients accumulate across micro-batches
        # because MultiAdapterFSDPWorker.forward_backward never zero_grad's
        # at entry. The final optim_step (called by the user) consumes the
        # accumulated gradients and zero_grad's.
        #
        # This makes the FSDP backend behave like the HF backend's built-in
        # micro-batching (see hf_training_model.HFTrainingModel.forward), and
        # lets callers (e.g. Trinity-RFT v22) pass a large train batch +
        # 1 optim_step instead of N (forward_backward + optim_step) loops --
        # eliminating mini-batch SGD intra-step off-policy drift in PPO/GRPO.
        mb = int(getattr(self.config, "micro_batch_size", 0) or 0)

        if self._worker is not None:
            # NO_RAY single-process mode: serialize GPU work across runs. Ray mode is
            # safe because each actor serializes its own method calls, but here two
            # runs' asyncio.to_thread calls could otherwise race on set_adapter.
            async with self._lock:
                if mb > 0 and len(data) % mb == 0:
                    eff_mb = mb
                else:
                    eff_mb = max(len(data), 1)
                n_micro = max(len(data) // eff_mb, 1) if data else 0
                out = await asyncio.to_thread(
                    self._worker.forward_backward,
                    adapter_name,
                    data,
                    loss_fn_name,
                    loss_fn_config,
                    eff_mb,
                    forward_only=not backward,
                    client_keys=client_keys,
                )
            metrics = dict(out.get("metrics") or {})
            metrics["actor/num_micro_batches"] = float(n_micro)
            loss_fn_outputs = _fsdp_logprobs_to_loss_fn_outputs(out, data)
        else:
            import ray

            n_actors = len(self._actors)
            if not data:
                return types.ForwardBackwardOutput(
                    loss_fn_output_type=loss_fn_name,
                    loss_fn_outputs=[],
                    metrics={},
                )

            # NCCL deadlock guard: each actor must receive at least one datum,
            # otherwise idle actors block forever on FSDP-2 collectives.
            if len(data) < n_actors:
                raise ValueError(
                    f"FSDP forward requires len(data) >= fsdp_num_gpus (world_size). "
                    f"Got len(data)={len(data)}, world_size={n_actors}. "
                    f"Sending fewer datums than ranks leaves some ranks idle and causes "
                    f"NCCL collectives in other ranks to hang permanently, deadlocking "
                    f"the entire training_run record's execution lock. Increase batch "
                    f"size or upstream chunking, or set fsdp_num_gpus=1 in tuft_config.yaml."
                )

            shards = _shard_list(data, n_actors)

            # In multi-actor mode every actor must issue the same number of
            # micro-batches, otherwise FSDP-2 NCCL collectives deadlock
            # (one rank finishes early while others are still iterating).
            # Only use micro-batching when mb evenly divides ALL shard sizes;
            # otherwise fall back to single-batch per shard (mb=None).
            if mb > 0 and all(len(s) % mb == 0 for s in shards if s):
                # Still need same micro-batch count: check that all non-empty
                # shards produce the same n_micro.
                micro_counts = {len(s) // mb for s in shards if s}
                eff_mb = mb if len(micro_counts) == 1 else None
            else:
                eff_mb = None

            self.logger.info(
                "FSDP multi-actor forward: batch=%d actors=%d mb=%s eff_mb=%s",
                len(data),
                n_actors,
                mb,
                eff_mb,
            )

            refs = []
            ref_weights = []
            for actor, shard in zip(self._actors, shards, strict=False):
                if not shard:
                    continue
                refs.append(
                    actor.forward_backward.remote(
                        list(shard),
                        adapter_name,
                        loss_fn_name,
                        loss_fn_config,
                        not backward,
                        eff_mb,
                        client_keys,
                    )
                )
                ref_weights.append(len(shard))

            results = await asyncio.to_thread(ray.get, refs) if refs else []

            metrics = _merge_metrics(results, ref_weights)
            loss_fn_outputs = []
            for out in results:
                loss_fn_outputs.extend(out.get("loss_fn_outputs", []))

        # Tinker expects every metric key to be "name:reduction" (e.g. loss:sum)
        metrics = {k: v for k, v in metrics.items() if ":" in k}

        return types.ForwardBackwardOutput(
            loss_fn_output_type=loss_fn_name,
            loss_fn_outputs=loss_fn_outputs,
            metrics=metrics,
        )

    async def optim_step(
        self,
        adam_params: types.AdamParams,
        lora_id: str,
    ) -> types.OptimStepResponse:
        adapter_name = self._get_adapter_name(lora_id)
        betas = (adam_params.beta1, adam_params.beta2)
        if self._worker is not None:
            async with self._lock:
                result = await asyncio.to_thread(
                    self._worker.optim_step,
                    adapter_name,
                    adam_params.learning_rate,
                    adam_params.weight_decay,
                    adam_params.grad_clip_norm,
                    betas,
                    adam_params.eps,
                )
        else:
            import ray

            refs = [
                a.optim_step.remote(
                    adapter_name,
                    adam_params.learning_rate,
                    adam_params.weight_decay,
                    adam_params.grad_clip_norm,
                    betas,
                    adam_params.eps,
                )
                for a in self._actors
            ]
            results = await asyncio.to_thread(ray.get, refs)
            result = results[0] if results else {}
        metrics = {k: float(v) for k, v in (result or {}).items() if isinstance(v, (int, float))}
        return types.OptimStepResponse(metrics=metrics or None)

    async def save_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer: bool,
    ) -> None:
        adapter_name = self._get_adapter_name(lora_id)
        path = checkpoint_record.adapter_path
        if self._worker is not None:
            async with self._lock:
                await asyncio.to_thread(self._worker.save_checkpoint, adapter_name, path, optimizer)
        else:
            import ray

            refs = [
                a.save_checkpoint.remote(adapter_name, str(path), optimizer) for a in self._actors
            ]
            await asyncio.to_thread(ray.get, refs)

    async def load_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer: bool,
    ) -> None:
        checkpoint_modules = self._validate_checkpoint_geometry(checkpoint_record)
        adapter_name = self._get_adapter_name(lora_id)
        path = checkpoint_record.adapter_path
        if self._worker is not None:
            async with self._lock:
                await asyncio.to_thread(
                    self._worker.load_checkpoint,
                    adapter_name,
                    path,
                    checkpoint_modules,
                    optimizer,
                )
        else:
            import ray

            refs = [
                a.load_checkpoint.remote(
                    adapter_name,
                    str(path),
                    checkpoint_modules,
                    optimizer,
                )
                for a in self._actors
            ]
            await asyncio.to_thread(ray.get, refs)
