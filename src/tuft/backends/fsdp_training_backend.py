"""
FSDP training backend: multi-node/multi-GPU training via FSDP and multi-adapter LoRA.

Peer to HFTrainingBackend; selected by ModelConfig.training_backend = "fsdp".
Implements BaseTrainingBackend; uses MultiAdapterVerlWorker and adapts data/loss for the engine.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from packaging import version
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from tinker import types
from torch.distributed.tensor import DTensor
from verl.utils import tensordict_utils as tu


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
from transformers import AutoModelForCausalLM
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config.engine import TrainingWorkerConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

from tuft.backends.base_backend import BaseTrainingBackend
from tuft.checkpoints import CheckpointRecord
from tuft.config import ModelConfig
from tuft.loss_fn import get_loss_fn


def _shard_list(xs: list[Any], n_shards: int) -> list[list[Any]]:
    """Split xs into n_shards contiguous shards."""
    if n_shards <= 0:
        raise ValueError(f"n_shards must be > 0, got {n_shards}")

    total = len(xs)
    base = total // n_shards
    rem = total % n_shards

    shards = []
    start = 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        end = start + size
        shards.append(xs[start:end])
        start = end
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


def _chunk_tensordict_allow_2d_nested(td: TensorDict, chunks: int) -> list | tuple:
    """Chunk TensorDict like engine's chunk_tensordict; 2D nested use unbind."""
    assert isinstance(td, TensorDict) and len(td) % chunks == 0, (
        f"expecting td with length divisible by chunks, but got {len(td)} and {chunks}"
    )
    chunk_size = len(td) // chunks
    # 2D nested (e.g. input_ids) do not support chunk/slice; use unbind
    keys = {
        key
        for key, val in td.items()
        if isinstance(val, torch.Tensor) and getattr(val, "is_nested", False) and val.dim() >= 2
    }
    new_td = TensorDict(
        {k: v for k, v in td.items() if k not in keys},
        batch_size=td.batch_size,
        device=td.device,
    )
    tds = new_td.chunk(chunks=chunks)
    for key in keys:
        tensors = td[key].unbind(dim=0)
        for i, sub_td in enumerate(tds):
            sub_td[key] = torch.nested.as_nested_tensor(
                tensors[i * chunk_size : (i + 1) * chunk_size], layout=torch.jagged
            )
    return tds


# Monkey-patch so prepare_micro_batches uses 2D-nested-safe chunk (avoids NestedTensor slice)
tu.chunk_tensordict = _chunk_tensordict_allow_2d_nested

# Default port for torch.distributed init (multi-GPU). ModelConfig.fsdp_master_port should match.
DEFAULT_MASTER_PORT = 29500


# =============================================================================
# Data and loss adaptation: Datum -> TensorDict, tuft loss -> engine loss_function
# =============================================================================


def _datum_list_to_tensordict(
    data: list[types.Datum],
    adapter_id: str,
    device: str = "cuda",
) -> "TensorDict":
    """Convert list[types.Datum] to TensorDict (input_ids, position_ids, weights, etc.)."""
    from torch.nn.utils.rnn import pad_sequence

    input_ids_list = [torch.tensor(datum.model_input.to_ints(), dtype=torch.long) for datum in data]
    global_token_num = [x.size(0) for x in input_ids_list]
    # NO_PADDING path expects 2D nested (batch, seq_len) for to_padded_tensor
    # output_size=(batch_size, max_seq_len)
    input_ids_nested = torch.nested.nested_tensor(
        [t.to(device) for t in input_ids_list], dtype=torch.long, device=device
    )
    # position_ids must be 2D nested for prepare_model_inputs to_padded_tensor
    position_ids_list = [
        torch.arange(seq_len, dtype=torch.long, device=device)
        for seq_len in (x.size(0) for x in input_ids_list)
    ]
    position_ids_nested = torch.nested.nested_tensor(
        position_ids_list, dtype=torch.long, device=device
    )

    # target_tokens, weights for loss from loss_fn_inputs
    target_tokens_list = []
    weights_list = []
    # RLHF-specific fields (logprobs, advantages)
    logprobs_list = []
    advantages_list = []
    has_rlhf_fields = False

    for datum in data:
        inp = datum.loss_fn_inputs or {}
        toks = inp.get("target_tokens")
        target_tokens_list.append(
            toks.to_torch() if toks is not None else torch.zeros(1, dtype=torch.long)
        )
        w = inp.get("weights")
        default_weights = torch.ones(len(datum.model_input.to_ints()), dtype=torch.float32)
        weights_list.append(w.to_torch() if w is not None else default_weights)

        # Extract RLHF fields if present
        logprobs = inp.get("logprobs")
        if logprobs is not None:
            has_rlhf_fields = True
            logprobs_list.append(logprobs.to_torch())
        else:
            logprobs_list.append(torch.zeros(1, dtype=torch.float32))

        advantages = inp.get("advantages")
        if advantages is not None:
            advantages_list.append(advantages.to_torch())
        else:
            advantages_list.append(torch.zeros(1, dtype=torch.float32))

    target_tokens = pad_sequence(
        [t.squeeze() if t.dim() > 1 else t for t in target_tokens_list],
        batch_first=True,
        padding_value=0,
    )
    weights = pad_sequence(weights_list, batch_first=True, padding_value=0.0)
    batch_size = len(input_ids_list)
    weights_device = weights.to(device)

    td_dict = {
        "target_tokens": target_tokens.to(device),
        "weights": weights_device,
        "loss_mask": weights_device,
        "temperature": torch.full((batch_size, 1, 1), 1.0, device=device, dtype=torch.float32),
        "adapter_id": adapter_id,
    }

    # Add RLHF fields if any datum has them
    if has_rlhf_fields:
        logprobs_padded = pad_sequence(logprobs_list, batch_first=True, padding_value=0.0)
        advantages_padded = pad_sequence(advantages_list, batch_first=True, padding_value=0.0)
        td_dict["logprobs"] = logprobs_padded.to(device)
        td_dict["advantages"] = advantages_padded.to(device)

    td = TensorDict(td_dict, batch_size=batch_size)
    td["input_ids"] = input_ids_nested
    td["position_ids"] = position_ids_nested
    tu.assign_non_tensor(td, global_token_num=global_token_num)
    tu.assign_non_tensor(
        td,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=batch_size,
        use_remove_padding=False,
        pad_mode=DatasetPadMode.NO_PADDING,
    )
    return td


def _make_verl_loss_fn(
    loss_fn_name: str,
    loss_fn_config: dict[str, float] | None,
) -> Callable[..., Any]:
    """Wrap tuft get_loss_fn to engine signature (model_output, data) -> (loss, metrics)."""
    loss_fn_config = loss_fn_config or {}
    tuft_loss = get_loss_fn(loss_fn_name)

    # Check if this is an RLHF loss function that requires additional fields
    rlhf_loss_fns = {"ppo", "grpo", "cispo", "importance_sampling", "dro"}
    is_rlhf = loss_fn_name.lower() in rlhf_loss_fns

    def _loss_function(
        model_output: Dict[str, Any],
        data: TensorDict,
        dp_group: Any = None,
        **kwargs: Any,
    ) -> Any:
        # model_output["log_probs"] is nested (B, seq); data["weights"] is (B, max_len)
        log_probs_nt = model_output["log_probs"]
        weights = data["weights"]
        batch_size, max_len = weights.shape
        target_logprobs = torch.nested.to_padded_tensor(
            log_probs_nt, padding=0.0, output_size=(batch_size, max_len)
        )

        # Build loss_fn_inputs based on loss function type
        if is_rlhf:
            # RLHF losses (PPO, GRPO, etc.) need logprobs and advantages
            # Check if data has required fields
            has_logprobs = "logprobs" in data.keys()
            has_advantages = "advantages" in data.keys()

            if not has_logprobs or not has_advantages:
                import logging

                logging.warning(
                    f"RLHF loss '{loss_fn_name}' requires 'logprobs' and 'advantages' in data. "
                    f"Available keys: {list(data.keys())}. "
                    f"Using fallback values."
                )

            loss_fn_inputs = {
                "target_logprobs": target_logprobs,
                "logprobs": data.get("logprobs", target_logprobs)
                if has_logprobs
                else target_logprobs,
                "advantages": data.get("advantages", torch.zeros_like(target_logprobs))
                if has_advantages
                else torch.zeros_like(target_logprobs),
            }
        else:
            # Standard losses (cross_entropy) only need target_logprobs and weights
            loss_fn_inputs = {"target_logprobs": target_logprobs, "weights": weights}

        loss, metrics = tuft_loss(loss_fn_inputs, loss_fn_config)
        return loss, metrics

    return _loss_function


def _fsdp_logprobs_to_loss_fn_outputs(
    engine_output: Dict[str, Any],
    data: list[types.Datum],
) -> list[Dict[str, Any]]:
    """Extract log_probs from engine.forward_backward_batch model_output and convert to
    per-datum loss_fn_outputs."""
    model_output = (engine_output or {}).get("model_output") or {}
    log_probs_nt = model_output.get("log_probs")
    if log_probs_nt is None or not hasattr(log_probs_nt, "unbind"):
        return [
            {"logprobs": types.TensorData(data=[0.0], dtype="float32", shape=[1])} for _ in data
        ]
    try:
        # Verl returns log_probs as nested tensor (batch, variable_len); unbind(0) gives per-sample
        per_sample = log_probs_nt.unbind(dim=0)
        return [
            {"logprobs": types.TensorData.from_torch(t.detach().cpu().float().clone())}
            for t in per_sample
        ]
    except Exception:
        return [
            {"logprobs": types.TensorData(data=[0.0], dtype="float32", shape=[1])} for _ in data
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
    """ModelConfig -> serializable dict for TrainingWorkerConfig and SlotPoolConfig in actors."""
    rank_slots = _get_rank_slots_from_config(config)
    return {
        "model_path": str(config.model_path),
        "max_model_len": config.max_model_len,
        "use_remove_padding": getattr(config, "use_remove_padding", False),
        "fsdp_override_config": dict(getattr(config, "fsdp_override_config", None) or {}),
        "slot_config": {
            "rank_slots": rank_slots,
            "lora_alpha_ratio": 2,
            "target_modules": ["q_proj", "v_proj"],
        },
    }


def _worker_dict_to_training_config(config_dict: dict) -> tuple:
    """Build TrainingWorkerConfig and SlotPoolConfig from config_dict inside an actor."""
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config import HFModelConfig
    from verl.workers.config.engine import FSDPEngineConfig
    from verl.workers.config.optimizer import OptimizerConfig

    override = dict(config_dict.get("fsdp_override_config") or {})
    if "attn_implementation" not in override:
        override["attn_implementation"] = "eager"
    hf_model_config = HFModelConfig(
        path=config_dict["model_path"],
        use_remove_padding=config_dict.get("use_remove_padding", False),
        override_config=override,
    )
    engine_config = FSDPEngineConfig(
        strategy="fsdp2",
        use_dynamic_bsz=False,
        max_token_len_per_gpu=config_dict["max_model_len"],
        micro_batch_size_per_gpu=1,
        forward_only=False,
    )
    training_config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=hf_model_config,
        engine_config=engine_config,
        optimizer_config=OptimizerConfig(),
        checkpoint_config=CheckpointConfig(),
    )
    sc = config_dict.get("slot_config") or {}
    slot_config = SlotPoolConfig(
        rank_slots=dict(sc.get("rank_slots", {8: 4})),
        lora_alpha_ratio=int(sc.get("lora_alpha_ratio", 2)),
        target_modules=list(sc.get("target_modules", ["q_proj", "v_proj"])),
    )
    return training_config, slot_config


# =============================================================================
# MultiAdapterVerlWorker
# =============================================================================
# Holds FSDPEngineWithLMHead; manages LoRA slot allocation/release; before forward_backward
# calls set_adapter and engine.forward_backward_batch; provides per-adapter optim_step/save/load.
# No Ray dependency. Backend holds one instance in single-process mode; in multi-GPU mode,
# N VerlWorkerActor (Ray) instances each hold one; N processes form torch.distributed FSDP;
# Ray handles process creation and .remote() scheduling.
# =============================================================================


class MultiAdapterVerlWorker:
    """
    Multi-LoRA slots: holds engine, calls forward_backward_batch under train_mode;
    set_adapter is called before each call to select the current PEFT active adapter.
    """

    def __init__(
        self,
        model_config: Any,
        engine_config: Any,
        optimizer_config: Any,
        checkpoint_config: Any,
        slot_config: SlotPoolConfig,
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.slot_config = slot_config
        self.engine: Any = None
        self._adapters: Dict[str, AdapterInfo] = {}
        self._adapters_by_rank: Dict[int, List[str]] = {}
        self._name_counter: Dict[int, int] = {}
        self._allocated: Dict[str, bool] = {}
        self._initialized = False
        # train_mode: enter on first forward_backward, exit in leave_train_mode() (e.g. release)
        self._train_mode_ctx: Any = None
        self.logger = logging.getLogger(f"{__name__}.MultiAdapterVerlWorker")

    def _generate_adapter_name(self, rank: int) -> str:
        if rank not in self._name_counter:
            self._name_counter[rank] = 0
        idx = self._name_counter[rank]
        self._name_counter[rank] += 1
        return f"adapter_r{rank}_{idx}"

    def initialize(self) -> None:
        """Build engine (base + materialize + PEFT adapters + FSDP v2) and create
        per-adapter optimizers."""
        if self._initialized:
            return
        base_model = self.engine._build_module()

        # Materialize: meta model has no storage; load pretrained weights before .cuda()
        model_path = getattr(self.engine.model_config, "local_path", None) or getattr(
            self.engine.model_config, "path", None
        )
        if not model_path:
            raise RuntimeError("engine.model_config has no local_path or path for materialization")
        trust_remote = getattr(self.engine.model_config, "trust_remote_code", True)
        full_cpu = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote,
        )
        state_dict = full_cpu.state_dict()
        del full_cpu
        base_model.load_state_dict(state_dict, assign=True, strict=False)
        del state_dict

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
                    base_model.enable_input_require_grads()
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
        self.engine.module = model_cuda
        self._create_optimizer_for_adapter(first)
        self._initialized = True

    def _create_optimizer_for_adapter(self, adapter_name: str) -> None:
        info = self._adapters[adapter_name]
        if info.optimizer is not None:
            return
        self._activate_adapter(adapter_name)
        params = [p for p in self.engine.module.parameters() if p.requires_grad]
        info.optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)

    def _activate_adapter(self, adapter_name: str) -> None:
        """Set PEFT active adapter before computation; same as HFTrainingModel._activate_adapter."""
        # FSDP v2 does not wrap the module, so we access the PEFT model directly
        # For FSDP v1: self.engine.module.module.set_adapter(adapter_name)
        # For FSDP v2: self.engine.module is the PEFT model itself
        module = self.engine.module
        # Handle both FSDP v1 (wrapped) and FSDP v2 (not wrapped) cases
        if hasattr(module, "set_adapter"):
            module.set_adapter(adapter_name)
        elif hasattr(module, "module") and hasattr(module.module, "set_adapter"):
            module.module.set_adapter(adapter_name)
        else:
            raise RuntimeError(f"Cannot find set_adapter method on module: {type(module)}")

    def _ensure_train_mode_entered(self) -> None:
        """Enter train_mode once and store context; exit in leave_train_mode() (e.g. release).

        We store the context instead of using 'with engine.train_mode():' because the context's
        __exit__ (or teardown) automatically clears engine state (e.g. optimizer_zero_grad). With a
        fine-grained SDK where forward_backward and optim_step are separate calls, exiting the
        context after each forward would clear gradients before optim_step runs. So we enter once,
        keep the context, and only exit explicitly in leave_train_mode() when the model is released.
        """
        if self._train_mode_ctx is not None:
            return
        ctx = self.engine.train_mode()
        ctx.__enter__()
        self._train_mode_ctx = ctx

    def leave_train_mode(self) -> None:
        """Exit the stored train_mode context. Call on model release to clean up Verl engine."""
        if self._train_mode_ctx is not None:
            self._train_mode_ctx.__exit__(None, None, None)
            self._train_mode_ctx = None

    def forward_backward(
        self,
        adapter_name: str,
        data: TensorDict,
        loss_function: Callable[..., Any],
        forward_only: bool = False,
    ) -> Dict[str, Any]:
        """Set PEFT active adapter, then call engine.forward_backward_batch (no step)."""
        self._activate_adapter(adapter_name)
        info = self._adapters[adapter_name]
        if info.optimizer is None:
            self._create_optimizer_for_adapter(adapter_name)
        self.engine.optimizer = info.optimizer
        if not forward_only:
            info.optimizer.zero_grad()
        self._ensure_train_mode_entered()
        output = self.engine.forward_backward_batch(
            data=data,
            loss_function=loss_function,
            forward_only=forward_only,
        )
        return output

    def _zero_grad_for_adapter(self, adapter_name: str) -> None:
        _match = f".{adapter_name}."
        for name, param in self.engine.module.named_parameters():
            if _match in name and param.grad is not None:
                param.grad = None

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
            torch.nn.utils.clip_grad_norm_(self.engine.module.parameters(), grad_clip_norm)
        opt.step()
        opt.zero_grad()
        info.step_count += 1
        self.engine.to(
            "cpu",
            model=self.engine.is_param_offload_enabled,
            optimizer=self.engine.is_optimizer_offload_enabled,
            grad=self.engine.is_param_offload_enabled,
        )
        self.engine.mode = None
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
        for name, param in self.engine.module.named_parameters():
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
            module = self.engine.module
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
            for name, param in self.engine.module.named_parameters():
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
# VerlWorkerActor: Ray actor, one GPU per process, forms torch.distributed with peers
# =============================================================================


class VerlWorkerActor:
    """Single-GPU Ray actor; N form process group via init_dist, each holds one worker."""

    def __init__(self, rank: int, world_size: int, config_dict: dict) -> None:
        self.rank = rank
        self.world_size = world_size
        self.config_dict = config_dict
        self._worker: Optional[MultiAdapterVerlWorker] = None
        self._dist_initialized = False
        self.logger = logging.getLogger(f"{__name__}.VerlWorkerActor")

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
        training_config, slot_config = _worker_dict_to_training_config(self.config_dict)
        engine = FSDPEngineWithLMHead(
            model_config=training_config.model_config,  # pyright: ignore[reportCallIssue]
            engine_config=training_config.engine_config,  # pyright: ignore[reportCallIssue]
            optimizer_config=training_config.optimizer_config,  # pyright: ignore[reportCallIssue]
            checkpoint_config=training_config.checkpoint_config,  # pyright: ignore[reportCallIssue]
        )
        self._worker = MultiAdapterVerlWorker(
            model_config=training_config.model_config,
            engine_config=training_config.engine_config,
            optimizer_config=training_config.optimizer_config,
            checkpoint_config=training_config.checkpoint_config,
            slot_config=slot_config,
        )
        self._worker.engine = engine
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
    ) -> Dict[str, Any]:
        """data: list[types.Datum] (or dicts after Ray serialization). Returns shard result."""
        if data and isinstance(data[0], dict):
            data = [types.Datum(**d) for d in data]

        if not data:
            return {
                "metrics": {},
                "loss_fn_outputs": [],
            }

        td = _datum_list_to_tensordict(data, adapter_name, "cuda")
        loss_fn = _make_verl_loss_fn(loss_fn_name, loss_fn_config)

        if self._worker is None:
            return {
                "metrics": {},
                "loss_fn_outputs": [],
            }

        out = self._worker.forward_backward(
            adapter_name,
            td,
            loss_fn,
            forward_only=forward_only,
        )

        metrics = out.get("metrics", {}) or {}

        def _to_scalar(v: Any) -> Any:
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                return sum(v) if len(v) > 1 else float(v[0])
            return v

        metrics = {k: _to_scalar(v) for k, v in metrics.items()}

        loss_fn_outputs = _fsdp_logprobs_to_loss_fn_outputs(out, data)

        return {
            "metrics": metrics,
            "loss_fn_outputs": loss_fn_outputs,
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

    def leave_train_mode(self) -> None:
        """Exit train_mode context on this actor; call on model release."""
        if self._worker is not None:
            self._worker.leave_train_mode()


# =============================================================================
# FSDPTrainingBackend (implements BaseTrainingBackend)
# Multi-GPU: N GPUs = N VerlWorkerActor (Ray), forming torch.distributed.
# =============================================================================


class FSDPTrainingBackend(BaseTrainingBackend):
    """
    Multi-node/multi-GPU training backend; peer to HFTrainingBackend.
    Selected via ModelConfig.training_backend = "fsdp".
    Uses N VerlWorkerActor (Ray), one per GPU, forming a process group for FSDP.
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
        self._worker: Optional[MultiAdapterVerlWorker] = None
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
            training_config, _slot = _worker_dict_to_training_config(self._config_dict)
            engine = FSDPEngineWithLMHead(
                model_config=training_config.model_config,  # pyright: ignore[reportCallIssue]
                engine_config=training_config.engine_config,  # pyright: ignore[reportCallIssue]  # pyright: ignore[reportCallIssue]
                optimizer_config=training_config.optimizer_config,  # pyright: ignore[reportCallIssue]  # pyright: ignore[reportCallIssue]
                checkpoint_config=training_config.checkpoint_config,  # pyright: ignore[reportCallIssue]  # pyright: ignore[reportCallIssue]
            )
            self._worker = MultiAdapterVerlWorker(
                model_config=training_config.model_config,
                engine_config=training_config.engine_config,
                optimizer_config=training_config.optimizer_config,
                checkpoint_config=training_config.checkpoint_config,
                slot_config=self._slot_config,
            )
            self._worker.engine = engine
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
                ray.remote(VerlWorkerActor)
                .options(
                    num_gpus=1,
                    runtime_env=_runtime_env,
                )
                .remote(r, n_gpus, config_dict)
            )
            actors.append(actor)
        # Set _world_size / _actors only after all succeed; else next create_adapter retries init
        # get_node_ip should return quickly; timeout avoids hang when actor not scheduled (e.g. GPU)
        _GET_NODE_IP_TIMEOUT = 30
        master_addr = await asyncio.to_thread(
            ray.get, actors[0].get_node_ip.remote(), timeout=_GET_NODE_IP_TIMEOUT
        )
        base_port = getattr(self.config, "fsdp_master_port", DEFAULT_MASTER_PORT)
        master_port = base_port + self._fsdp_index if self._fsdp_index is not None else base_port
        await asyncio.gather(
            *[
                asyncio.to_thread(ray.get, a.init_dist.remote(master_addr, master_port))
                for a in actors
            ]
        )
        await asyncio.gather(*[asyncio.to_thread(ray.get, a.build_worker.remote()) for a in actors])
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
                    self._worker.leave_train_mode()
                elif self._actors:
                    import ray

                    await asyncio.to_thread(
                        ray.get, self._actors[0].release_slot.remote(adapter_name)
                    )
                    await asyncio.to_thread(
                        ray.get, [a.leave_train_mode.remote() for a in self._actors]
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

        def _to_scalar(v: Any) -> Any:
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                return sum(v) if len(v) > 1 else float(v[0])
            return v

        if self._worker is not None:
            td = await asyncio.to_thread(_datum_list_to_tensordict, data, adapter_name, "cuda")
            verl_loss_fn = _make_verl_loss_fn(loss_fn_name, loss_fn_config)
            out = await asyncio.to_thread(
                self._worker.forward_backward,
                adapter_name,
                td,
                verl_loss_fn,
                not backward,
            )
            metrics = out.get("metrics", {}) or {}
            metrics = {k: _to_scalar(v) for k, v in metrics.items()}
            loss_fn_outputs = _fsdp_logprobs_to_loss_fn_outputs(out, data)
        else:
            import ray

            shards = _shard_list(data, len(self._actors))

            refs = []
            for actor, shard in zip(self._actors, shards, strict=False):
                if not shard:
                    continue
                refs.append(
                    actor.forward_backward.remote(
                        shard,
                        adapter_name,
                        loss_fn_name,
                        loss_fn_config,
                        not backward,
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
