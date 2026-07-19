"""Multi-node/multi-GPU training via FSDP2 and multi-adapter LoRA."""

from __future__ import annotations

import asyncio
import logging
import os
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
from tuft.checkpoints import CheckpointRecord
from tuft.config import ModelConfig


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


def _merge_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for out in results:
        metrics = out.get("metrics", {}) or {}
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue

            if k.endswith(":sum"):
                merged[k] = merged.get(k, 0.0) + float(v)
            elif k.endswith(":mean"):
                merged.setdefault(k, []).append(float(v))
            else:
                merged[k] = merged.get(k, 0.0) + float(v)

    for k, v in list(merged.items()):
        if isinstance(v, list):
            merged[k] = sum(v) / len(v) if v else 0.0

    return merged


# Default port for torch.distributed init (multi-GPU). ModelConfig.fsdp_master_port should match.
DEFAULT_MASTER_PORT = 29500


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


# =============================================================================
# Slot configuration and worker-internal data structures
# =============================================================================


@dataclass
class SlotPoolConfig:
    """Multi-adapter slot pool configuration (rank -> number of slots)."""

    rank_slots: Dict[int, int] = field(default_factory=lambda: {8: 5, 16: 2})
    lora_alpha_ratio: int = 2
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    def get_lora_alpha(self, rank: int) -> int:
        return rank * self.lora_alpha_ratio


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
    Defaults (override via ModelConfig.fsdp_rank_slots):
    - rank 8: 16 slots (common case; more slots for lower memory per adapter).
    - other max_lora_rank: 8 slots (fewer slots for higher rank due to memory).
    """
    max_rank = getattr(config, "max_lora_rank", 8)
    raw = getattr(config, "fsdp_rank_slots", None)
    if raw and len(raw) > 0:
        return {int(k): v for k, v in raw.items()}
    # default slots
    if max_rank == 8:
        return {8: 16}
    return {8: 16, max_rank: 8}


def _config_to_worker_dict(config: ModelConfig) -> dict:
    """Convert ModelConfig to the serializable subset needed by Ray workers."""

    rank_slots = _get_rank_slots_from_config(config)
    return {
        "model_path": str(config.model_path),
        "max_model_len": config.max_model_len,
        "fsdp_override_config": dict(getattr(config, "fsdp_override_config", None) or {}),
        "attn_implementation": getattr(config, "attn_implementation", None),
        "slot_config": {
            "rank_slots": rank_slots,
            "lora_alpha_ratio": 2,
            "target_modules": ["q_proj", "v_proj"],
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
        rank_slots=dict(sc.get("rank_slots", {8: 4})),
        lora_alpha_ratio=int(sc.get("lora_alpha_ratio", 2)),
        target_modules=list(sc.get("target_modules", ["q_proj", "v_proj"])),
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

    def forward_backward(
        self,
        adapter_name: str,
        data: list[types.Datum],
        loss_fn_name: str,
        loss_fn_config: dict[str, float] | None,
        micro_batch_size: int,
        forward_only: bool = False,
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
        )

    def optim_step(
        self,
        adapter_name: str,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
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
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), grad_clip_norm)
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
        # Collect full state dict for the adapter
        state = {}
        _match = f".{adapter_name}."
        for name, param in self.module.named_parameters():
            if _match in name:
                if isinstance(param, DTensor):
                    # FSDP v2: gather full tensor from all ranks
                    state[name] = param.full_tensor().cpu().clone()
                else:
                    state[name] = param.data.cpu().clone()

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
            module = self.module
            has_nested_peft = hasattr(module, "module") and hasattr(module.module, "peft_config")
            peft_model = module.module if has_nested_peft else module

            # Save adapter_config.json
            if hasattr(peft_model, "peft_config") and adapter_name in peft_model.peft_config:
                adapter_config = peft_model.peft_config[adapter_name]
                config_dict = adapter_config.to_dict()
                # Convert sets to lists for JSON serialization
                for key, value in config_dict.items():
                    if isinstance(value, set):
                        config_dict[key] = list(value)
                import json

                with open(path / "adapter_config.json", "w") as f:
                    json.dump(config_dict, f, indent=2)

            # Save adapter weights in safetensors format for sampling/vLLM.
            # vLLM lora/utils.py expects keys ending in ".lora_A.weight" or ".lora_B.weight"
            # (parts[-2] in ["lora_A","lora_B"]). PEFT uses ".lora_A.<adapter_name>.weight";
            # strip the adapter name so vLLM does not raise "unsupported LoRA weight".
            peft_state = {}
            for name, tensor in state.items():
                key = name
                if adapter_name != "default":
                    key = key.replace(".lora_A." + adapter_name + ".", ".lora_A.")
                    key = key.replace(".lora_B." + adapter_name + ".", ".lora_B.")
                    key = key.replace(".lora_A." + adapter_name + ".weight", ".lora_A.weight")
                    key = key.replace(".lora_B." + adapter_name + ".weight", ".lora_B.weight")
                peft_state[key] = tensor

            try:
                from safetensors.torch import save_file

                save_file(peft_state, path / "adapter_model.safetensors")
            except Exception:
                # Fallback to torch.save if safetensors fails
                torch.save(peft_state, path / "adapter_model.bin")

    def load_checkpoint(self, adapter_name: str, path: str | Path, optimizer: bool = True) -> None:
        path = Path(path)
        state = torch.load(path / "adapter.pt", map_location="cpu", weights_only=True)
        self._activate_adapter(adapter_name)
        with torch.no_grad():
            _match = f".{adapter_name}."
            for name, param in self.module.named_parameters():
                if _match in name and name in state:
                    loaded_tensor = state[name]
                    if isinstance(param, DTensor):
                        # FSDP v2: param is DTensor, need to copy to local shard
                        # Get the local shard and copy data to it
                        local_param = param.to_local()
                        # The loaded state is full tensor, we need to shard it
                        # For dim-0 sharding, slice the tensor
                        import torch.distributed as dist

                        if dist.is_initialized():
                            world_size = dist.get_world_size()
                            rank = dist.get_rank()
                            chunk_size = loaded_tensor.size(0) // world_size
                            start_idx = rank * chunk_size
                            end_idx = start_idx + chunk_size
                            local_param.copy_(
                                loaded_tensor[start_idx:end_idx].to(local_param.device)
                            )
                        else:
                            local_param.copy_(loaded_tensor.to(local_param.device))
                    else:
                        param.data.copy_(loaded_tensor.to(param.device))
        if optimizer:
            opt_path = path / "optimizer.pt"
            if opt_path.exists():
                if self._adapters[adapter_name].optimizer is None:
                    self._create_optimizer_for_adapter(adapter_name)
                opt_state = torch.load(opt_path, map_location="cpu", weights_only=True)
                self._adapters[adapter_name].optimizer.load_state_dict(opt_state)

    def allocate_slot(self, rank: int) -> Optional[str]:
        """Allocate an unused slot for rank; return adapter_name or None."""
        for name in self._adapters_by_rank.get(rank, []):
            if not self._allocated.get(name, False):
                self._allocated[name] = True
                return name
        return None

    def release_slot(self, adapter_name: str) -> None:
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
    ) -> Dict[str, Any]:
        if self._worker is None:
            return {}
        return self._worker.optim_step(adapter_name, learning_rate, weight_decay, grad_clip_norm)

    def save_checkpoint(self, adapter_name: str, path: str, optimizer: bool = True) -> None:
        # FSDP v2: all ranks must participate in full_tensor() collective operation
        if self._worker is None:
            return
        self._worker.save_checkpoint(adapter_name, Path(path), optimizer)

    def load_checkpoint(self, adapter_name: str, path: str, optimizer: bool = True) -> None:
        if self._worker is None:
            return
        self._worker.load_checkpoint(adapter_name, Path(path), optimizer)


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
        rank_slots = _get_rank_slots_from_config(config)
        self._slot_config = SlotPoolConfig(
            rank_slots=rank_slots,
        )
        self._config_dict = _config_to_worker_dict(config)
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
            model_config, _slot = _worker_dict_to_configs(self._config_dict)
            self._worker = MultiAdapterFSDPWorker(
                model_config=model_config,
                slot_config=self._slot_config,
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

    async def create_adapter(self, lora_id: str, lora_config: types.LoraConfig) -> None:
        async with self._lock:
            if self._world_size == 0 and self._worker is None and not self._actors:
                await self.async_init()
            rank = getattr(lora_config, "rank", 8)
            if self._worker is not None:
                adapter_name = await asyncio.to_thread(self._worker.allocate_slot, rank)
            elif self._actors:
                import ray

                adapter_name: str | None = await asyncio.to_thread(
                    ray.get, self._actors[0].allocate_slot.remote(rank)
                )
            else:
                raise RuntimeError("FSDPTrainingBackend not initialized.")
            if adapter_name is None:
                raise ValueError(f"No free slot for rank={rank}; all slots allocated.")
            self._lora_id_to_adapter_name[lora_id] = adapter_name
            self._adapter_name_to_lora_id[adapter_name] = lora_id

    async def remove_adapter(self, lora_id: str) -> None:
        async with self._lock:
            adapter_name = self._lora_id_to_adapter_name.pop(lora_id, None)
            if adapter_name:
                self._adapter_name_to_lora_id.pop(adapter_name, None)
                if self._worker is not None:
                    self._worker.release_slot(adapter_name)
                elif self._actors:
                    import ray

                    await asyncio.to_thread(
                        ray.get, self._actors[0].release_slot.remote(adapter_name)
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
                    )
                )

            results = await asyncio.to_thread(ray.get, refs) if refs else []

            metrics = _merge_metrics(results)
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
        if self._worker is not None:
            result = await asyncio.to_thread(
                self._worker.optim_step,
                adapter_name,
                adam_params.learning_rate,
                adam_params.weight_decay,
                adam_params.grad_clip_norm,
            )
        else:
            import ray

            refs = [
                a.optim_step.remote(
                    adapter_name,
                    adam_params.learning_rate,
                    adam_params.weight_decay,
                    adam_params.grad_clip_norm,
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
        adapter_name = self._get_adapter_name(lora_id)
        path = checkpoint_record.adapter_path
        if self._worker is not None:
            await asyncio.to_thread(self._worker.load_checkpoint, adapter_name, path, optimizer)
        else:
            import ray

            refs = [
                a.load_checkpoint.remote(adapter_name, str(path), optimizer) for a in self._actors
            ]
            await asyncio.to_thread(ray.get, refs)
