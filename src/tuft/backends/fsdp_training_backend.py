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
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from tinker import types
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config.engine import TrainingWorkerConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

from tuft.backends.base_backend import BaseTrainingBackend
from tuft.checkpoints import CheckpointRecord
from tuft.config import ModelConfig
from tuft.loss_fn import get_loss_fn


def _chunk_tensordict_allow_2d_nested(td: TensorDict, chunks: int) -> list:
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
    for datum in data:
        inp = datum.loss_fn_inputs or {}
        toks = inp.get("target_tokens")
        target_tokens_list.append(
            toks.to_torch() if toks is not None else torch.zeros(1, dtype=torch.long)
        )
        w = inp.get("weights")
        default_weights = torch.ones(len(datum.model_input.to_ints()), dtype=torch.float32)
        weights_list.append(w.to_torch() if w is not None else default_weights)
    target_tokens = pad_sequence(
        [t.squeeze() if t.dim() > 1 else t for t in target_tokens_list],
        batch_first=True,
        padding_value=0,
    )
    weights = pad_sequence(weights_list, batch_first=True, padding_value=0.0)
    batch_size = len(input_ids_list)
    weights_device = weights.to(device)
    td = TensorDict(
        {
            "target_tokens": target_tokens.to(device),
            "weights": weights_device,
            "loss_mask": weights_device,
            "temperature": torch.full((batch_size, 1, 1), 1.0, device=device, dtype=torch.float32),
            "adapter_id": adapter_id,
        },
        batch_size=batch_size,
    )
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
        loss_fn_inputs = {"target_logprobs": target_logprobs, "weights": weights}
        loss, metrics = tuft_loss(loss_fn_inputs, loss_fn_config)
        return loss, metrics

    return _loss_function


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


def _config_to_worker_dict(config: ModelConfig) -> dict:
    """ModelConfig -> serializable dict for TrainingWorkerConfig and SlotPoolConfig in actors."""
    max_rank = getattr(config, "max_lora_rank", 8)
    return {
        "model_path": str(config.model_path),
        "max_model_len": config.max_model_len,
        "use_remove_padding": getattr(config, "use_remove_padding", False),
        "fsdp_override_config": dict(getattr(config, "fsdp_override_config", None) or {}),
        "slot_config": {
            "rank_slots": {max_rank: 4},
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
        use_dynamic_bsz=False,
        max_token_len_per_gpu=config_dict["max_model_len"],
        micro_batch_size_per_gpu=1,
        forward_only=True,
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
        self._train_mode_ctx: Any = None
        self.logger = logging.getLogger(f"{__name__}.MultiAdapterVerlWorker")

    def _generate_adapter_name(self, rank: int) -> str:
        if rank not in self._name_counter:
            self._name_counter[rank] = 0
        idx = self._name_counter[rank]
        self._name_counter[rank] += 1
        return f"adapter_r{rank}_{idx}"

    def initialize(self) -> None:
        """Build engine (base + PEFT adapters + FSDP) and create per-adapter optimizers."""
        if self._initialized:
            return
        base_model = self.engine._build_module()
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
        if peft_model is not None:
            first = next(iter(self._adapters))
            peft_model.set_adapter(first)
            model_bf16 = peft_model.to(torch.bfloat16)
            model_cuda = model_bf16.cuda()
            fsdp_model = FSDP(model_cuda, use_orig_params=True)
            self.engine.module = fsdp_model
            self._create_optimizer_for_adapter(first)
        self._initialized = True
        self.logger.info(
            "MultiAdapterVerlWorker initialized with adapters: %s",
            list(self._adapters.keys()),
        )

    def _create_optimizer_for_adapter(self, adapter_name: str) -> None:
        info = self._adapters[adapter_name]
        if info.optimizer is not None:
            return
        self._activate_adapter(adapter_name)
        params = [p for p in self.engine.module.parameters() if p.requires_grad]
        info.optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)

    def _activate_adapter(self, adapter_name: str) -> None:
        """Set PEFT active adapter before computation; same as HFTrainingModel._activate_adapter."""
        self.engine.module.module.set_adapter(adapter_name)

    def train_mode(self) -> Any:
        """Return engine.train_mode() context; one 'with', multiple forward_backward inside."""
        return self.engine.train_mode()

    def forward_backward(
        self,
        adapter_name: str,
        data: TensorDict,
        loss_function: Callable[..., Any],
    ) -> Dict[str, Any]:
        """Set PEFT active adapter by adapter_name, then call engine.forward_backward_batch."""
        self._activate_adapter(adapter_name)
        output = self.engine.forward_backward_batch(
            data=data,
            loss_function=loss_function,
            forward_only=False,
        )
        return output

    def _zero_grad_for_adapter(self, adapter_name: str) -> None:
        for name, param in self.engine.module.named_parameters():
            if adapter_name in name and param.grad is not None:
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
        params = [
            p
            for n, p in self.engine.module.named_parameters()
            if adapter_name in n and p.requires_grad and p.grad is not None
        ]
        if grad_clip_norm is not None and grad_clip_norm > 0 and params:
            torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
        opt.step()
        self._zero_grad_for_adapter(adapter_name)
        info.step_count += 1
        return {"step_count": info.step_count, "adapter": adapter_name}

    def save_checkpoint(self, adapter_name: str, path: str | Path, optimizer: bool = True) -> None:
        """Save adapter.pt (training load_state) + PEFT format (sampling); optional optimizer."""
        self._activate_adapter(adapter_name)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # 1) Internal format for FSDP load_state
        state = {
            name: param.data.cpu().clone()
            for name, param in self.engine.module.named_parameters()
            if adapter_name in name
        }
        torch.save(state, path / "adapter.pt")
        if optimizer:
            info = self._adapters[adapter_name]
            if info.optimizer is not None:
                torch.save(info.optimizer.state_dict(), path / "optimizer.pt")
        # 2) PEFT format (adapter_config.json, adapter_model.safetensors) so sampling/VLLM can load
        peft_model = self.engine.module.module
        peft_model.save_pretrained(str(path), selected_adapters=[adapter_name])
        lora_subdir = path / adapter_name
        if lora_subdir.exists() and lora_subdir.is_dir():
            for item in lora_subdir.iterdir():
                dest = path / item.name
                if dest.exists():
                    if dest.is_file():
                        dest.unlink()
                    elif dest.is_dir():
                        shutil.rmtree(dest)
                shutil.move(str(item), str(dest))
            lora_subdir.rmdir()

    def load_checkpoint(self, adapter_name: str, path: str | Path, optimizer: bool = True) -> None:
        path = Path(path)
        state = torch.load(path / "adapter.pt", map_location="cpu")
        self._activate_adapter(adapter_name)
        with torch.no_grad():
            for name, param in self.engine.module.named_parameters():
                if adapter_name in name and name in state:
                    param.data.copy_(state[name].to(param.device))
        if optimizer:
            opt_path = path / "optimizer.pt"
            if opt_path.exists():
                if self._adapters[adapter_name].optimizer is None:
                    self._create_optimizer_for_adapter(adapter_name)
                opt_state = torch.load(opt_path, map_location="cpu")
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
            model_config=training_config.model_config,
            engine_config=training_config.engine_config,
            optimizer_config=training_config.optimizer_config,
            checkpoint_config=training_config.checkpoint_config,
        )
        self._worker = MultiAdapterVerlWorker(
            model_config=training_config.model_config,
            engine_config=training_config.engine_config,
            optimizer_config=training_config.optimizer_config,
            checkpoint_config=training_config.checkpoint_config,
            slot_config=slot_config,
        )
        self._worker.engine = engine
        self._worker.initialize()

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
    ) -> Dict[str, Any]:
        """data: list[types.Datum] (or dicts after Ray serialization). Returns metrics."""
        if self._worker is None:
            return {}
        # Ray may deserialize Datum as dict
        if data and isinstance(data[0], dict):
            data = [types.Datum(**d) for d in data]
        td = _datum_list_to_tensordict(data, adapter_name, "cuda")
        loss_fn = _make_verl_loss_fn(loss_fn_name, loss_fn_config)
        if getattr(self._worker.engine, "optimizer", None) is not None:
            with self._worker.train_mode():
                out = self._worker.forward_backward(adapter_name, td, loss_fn)
        else:
            self._worker.engine.module.train()
            out = self._worker.forward_backward(adapter_name, td, loss_fn)
        metrics = out.get("metrics", {}) or {}

        def _to_scalar(v: Any) -> Any:
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                return sum(v) if len(v) > 1 else float(v[0])
            return v

        return {k: _to_scalar(v) for k, v in metrics.items()}

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
        import torch.distributed as dist

        if self._worker is None:
            return
        if dist.get_rank() != 0:
            return
        self._worker.save_checkpoint(adapter_name, Path(path), optimizer)

    def load_checkpoint(self, adapter_name: str, path: str, optimizer: bool = True) -> None:
        if self._worker is None:
            return
        self._worker.load_checkpoint(adapter_name, Path(path), optimizer)


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

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._worker: Optional[MultiAdapterVerlWorker] = None
        self._actors: List[Any] = []
        self._world_size: int = 0
        self._lora_id_to_adapter_name: Dict[str, str] = {}
        self._adapter_name_to_lora_id: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        max_rank = getattr(config, "max_lora_rank", 8)
        self._slot_config = SlotPoolConfig(
            rank_slots={max_rank: 4},
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
                model_config=training_config.model_config,
                engine_config=training_config.engine_config,
                optimizer_config=training_config.optimizer_config,
                checkpoint_config=training_config.checkpoint_config,
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
            self.logger.info("FSDPTrainingBackend async_init done: local single process (no Ray)")
            return
        import ray

        self._world_size = n_gpus
        config_dict = self._config_dict
        # No runtime_env pip: Ray venv may lack pip. Workers use torch/peft/verl from pyproject.
        actors = []
        for r in range(n_gpus):
            actor = (
                ray.remote(VerlWorkerActor)
                .options(
                    num_gpus=1,
                    runtime_env={},
                )
                .remote(r, n_gpus, config_dict)
            )
            actors.append(actor)
        self._actors = actors
        master_addr = await asyncio.to_thread(ray.get, actors[0].get_node_ip.remote())
        master_port = getattr(self.config, "fsdp_master_port", DEFAULT_MASTER_PORT)
        await asyncio.gather(
            *[
                asyncio.to_thread(ray.get, a.init_dist.remote(master_addr, master_port))
                for a in actors
            ]
        )
        await asyncio.gather(*[asyncio.to_thread(ray.get, a.build_worker.remote()) for a in actors])
        self.logger.info("FSDPTrainingBackend async_init done: %d Ray actors (FSDP)", n_gpus)

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

                adapter_name = await asyncio.to_thread(
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
        if self._worker is not None:
            td = await asyncio.to_thread(_datum_list_to_tensordict, data, adapter_name, "cuda")
            verl_loss_fn = _make_verl_loss_fn(loss_fn_name, loss_fn_config)
            if getattr(self._worker.engine, "optimizer", None) is not None:
                with self._worker.train_mode():
                    out = await asyncio.to_thread(
                        self._worker.forward_backward,
                        adapter_name,
                        td,
                        verl_loss_fn,
                    )
            else:
                self._worker.engine.module.train()
                out = await asyncio.to_thread(
                    self._worker.forward_backward,
                    adapter_name,
                    td,
                    verl_loss_fn,
                )
            metrics = out.get("metrics", {}) or {}

            def _to_scalar(v: Any) -> Any:
                if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                    return sum(v) if len(v) > 1 else float(v[0])
                return v

            metrics = {k: _to_scalar(v) for k, v in metrics.items()}
        else:
            import ray

            refs = [
                a.forward_backward.remote(data, adapter_name, loss_fn_name, loss_fn_config)
                for a in self._actors
            ]
            results = await asyncio.to_thread(ray.get, refs)
            metrics = results[0] if results else {}
        if "loss" not in metrics and "loss:sum" in metrics:
            metrics["loss"] = metrics["loss:sum"]
        loss_fn_outputs = [
            {"logprobs": types.TensorData(data=[0.0], dtype="float32", shape=[1])} for _ in data
        ]
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
