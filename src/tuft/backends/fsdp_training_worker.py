"""FSDP2 Training Worker — one instance per GPU.

Handles model construction, FSDP2 wrapping, and multi-tenant adapter
management with forward/backward/optimizer operations.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import TYPE_CHECKING, Any, Dict

import torch
import torch.distributed
from peft import LoraConfig, get_peft_model
from tinker import types
from tinker.types import LoraConfig as TinkerLoraConfig
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM

from tuft.backends.fsdp_utils import (
    FSDPModule,  # type: ignore[reportPrivateImportUsage]
    MixedPrecisionPolicy,  # type: ignore[reportPrivateImportUsage]
    apply_fsdp2,
    create_device_mesh,
    fsdp2_clip_grad_norm_,
)
from tuft.backends.hf_training_model import get_target_modules
from tuft.checkpoints import CheckpointRecord
from tuft.config import ModelConfig
from tuft.loss_fn import get_loss_fn, metrics_reduction


if TYPE_CHECKING:
    from torch.distributed.tensor import DTensor
else:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import DTensor  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


def _detect_decoder_layer_linears(model: torch.nn.Module) -> list[str] | str:
    """Return the short names of all ``nn.Linear`` modules that live inside
    transformer decoder layers (identified via ``_no_split_modules``).

    This gives a comprehensive target_modules list for the "default" LoRA
    adapter so that all possible target modules are pre-wrapped *before*
    FSDP2 sharding.
    """
    no_split = set()
    for m in (model, getattr(model, "model", None)):
        ns = getattr(m, "_no_split_modules", None)
        if ns:
            no_split.update(ns)
    if not no_split:
        no_split = {"DecoderLayer"}

    in_decoder_layer = set()
    for _name, mod in model.named_modules():
        if mod.__class__.__name__ in no_split:
            for sub_name, sub_mod in mod.named_modules():
                if isinstance(sub_mod, torch.nn.Linear):
                    in_decoder_layer.add(sub_name.split(".")[-1])
            break

    return sorted(in_decoder_layer) if in_decoder_layer else "all-linear"


class FSDPTrainingWorker:
    """Single-GPU FSDP2 training worker.

    Each Ray actor wraps one instance.  All workers in the same
    :class:`FSDPWorkerGroup` share an NCCL process group and jointly hold
    one FSDP2-sharded model.

    Multi-tenant design
    -------------------
    *   Each adapter has its own ``AdamW`` optimizer and gradient buffer.
    *   When switching adapters, the *entire* gradient state of the active
        adapter (all ``requires_grad`` params) is saved and restored so that
        gradient accumulation across ``forward_backward`` calls is seamless.
    *   Optimizer states (momentum / variance) live inside per-adapter
        ``AdamW`` instances and are never affected by adapter switches.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(
        self,
        config: ModelConfig,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ) -> None:
        self.config = config
        self.rank = rank
        self.world_size = world_size

        # --- 1. distributed setup ---
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="cpu:gloo,cuda:nccl",
                rank=rank,
                world_size=world_size,
            )

        # --- 2. device mesh ---
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=-1)

        # --- 3. build model ---
        self.model: torch.nn.Module = self._build_fsdp2_model(config)

        # --- 4. multi-tenant state ---
        self.adapter_optimizer: Dict[str, torch.optim.AdamW] = {}
        self.saved_grads: Dict[str, Dict[str, torch.Tensor]] = {}
        self.has_pending_grads: Dict[str, bool] = {}
        self.active_adapter: str | None = None
        self.micro_batch_size = config.micro_batch_size

        logger.info("FSDPTrainingWorker rank=%d ready (world_size=%d)", rank, world_size)

    def cleanup(self) -> None:
        """Release GPU resources and destroy the distributed process group.

        Must be called before killing the actor to allow the next worker group
        to initialise a new process group on the same GPUs.
        """
        del self.model
        self.adapter_optimizer.clear()
        self.saved_grads.clear()
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("FSDPTrainingWorker rank=%d cleaned up", self.rank)

    @property
    def _peft(self) -> Any:
        """Access PEFT-specific methods (add_adapter, set_adapter, etc.)
        on the FSDP2-wrapped PeftModel without pyright complaints about
        nn.Module's dynamic __getattr__ return type.
        """
        return self.model

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _build_fsdp2_model(self, config: ModelConfig) -> torch.nn.Module:
        model = AutoModelForCausalLM.from_pretrained(
            str(config.model_path),
            torch_dtype=torch.bfloat16,
        )
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Wrap ALL decoder-layer linear modules with a placeholder LoRA
        # *before* FSDP2 sharding.  This ensures later add_adapter calls
        # only invoke PEFT's update_layer (adding weights to existing
        # ModuleDicts) instead of _replace_module, which would break
        # FSDP2's parameter tracking.
        #
        # autocast_adapter_dtype=False prevents PEFT from up-casting
        # bfloat16 LoRA weights to float32, keeping all parameters in a
        # consistent dtype for mixed-precision training.
        model.enable_input_require_grads()
        default_targets = _detect_decoder_layer_linears(model)
        peft_config = LoraConfig(target_modules=default_targets)
        model = get_peft_model(
            model,
            peft_config=peft_config,
            adapter_name="default",
            autocast_adapter_dtype=False,
        )

        torch.distributed.barrier()

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        fsdp_kwargs = {
            "mesh": self.device_mesh,
            "mp_policy": mp_policy,
            "reshard_after_forward": True,
        }
        apply_fsdp2(model, fsdp_kwargs)

        if not isinstance(model, FSDPModule):
            raise RuntimeError("FSDP2 wrapping failed — model is not FSDPModule")
        torch.distributed.barrier()
        result: torch.nn.Module = model  # type: ignore[assignment]
        return result

    # ------------------------------------------------------------------
    # Adapter management
    # ------------------------------------------------------------------
    def create_adapter(
        self,
        lora_id: str,
        lora_config: TinkerLoraConfig,
    ) -> None:
        if lora_id in self.adapter_optimizer:
            raise ValueError(f"Adapter {lora_id} already exists.")

        target_mods = get_target_modules(str(self.config.model_path), lora_config)
        peft_config = LoraConfig(
            r=lora_config.rank,
            target_modules=target_mods,
            lora_alpha=lora_config.rank,
        )
        self._peft.add_adapter(adapter_name=lora_id, peft_config=peft_config)
        self._peft.set_adapter(lora_id)

        # PEFT's _move_adapter_to_device_of_base_layer infers dtype from the
        # base weight (DTensor bfloat16).  Enforce bfloat16 explicitly in case
        # PEFT's internal logic changes, and broadcast from rank 0 so every
        # rank starts with identical LoRA weights (FSDP2 does not manage these
        # dynamically-added parameters).
        torch.distributed.barrier()
        for p in self.model.parameters():
            if p.requires_grad and not isinstance(p.data, DTensor):
                if p.data.dtype != torch.bfloat16:
                    p.data = p.data.to(torch.bfloat16)
                torch.distributed.broadcast(p.data, src=0)
        torch.distributed.barrier()

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.adapter_optimizer[lora_id] = torch.optim.AdamW(params)
        self.active_adapter = lora_id

    def remove_adapter(self, lora_id: str) -> None:
        if lora_id in self.adapter_optimizer:
            self._peft.delete_adapter(lora_id)
            del self.adapter_optimizer[lora_id]
            self.saved_grads.pop(lora_id, None)
            self.has_pending_grads.pop(lora_id, None)
            if self.active_adapter == lora_id:
                self.active_adapter = None
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Gradient save / restore for multi-tenant isolation
    # ------------------------------------------------------------------
    def _save_adapter_grads(self, adapter_id: str) -> None:
        """Snapshot the gradient of every trainable param for *adapter_id*."""
        grads: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads[name] = param.grad.detach().clone()
        self.saved_grads[adapter_id] = grads

    def _restore_adapter_grads(self, adapter_id: str) -> None:
        """Restore previously saved gradients for *adapter_id*."""
        saved = self.saved_grads.get(adapter_id)
        if saved is None:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in saved:
                param.grad = saved[name]

    def _zero_active_grads(self) -> None:
        """Zero out gradients of the currently trainable parameters."""
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = None

    def _clear_saved_grads(self, adapter_id: str) -> None:
        self.saved_grads.pop(adapter_id, None)
        self.has_pending_grads[adapter_id] = False

    def _switch_adapter(self, target_adapter_id: str) -> None:
        """Switch the active adapter, persisting / restoring gradients."""
        if self.active_adapter == target_adapter_id:
            return

        # save current adapter's grads (if any pending)
        if self.active_adapter is not None and self.has_pending_grads.get(
            self.active_adapter, False
        ):
            self._save_adapter_grads(self.active_adapter)

        # activate target
        self._peft.set_adapter(target_adapter_id)
        self.active_adapter = target_adapter_id

        # restore target adapter's grads
        if self.has_pending_grads.get(target_adapter_id, False):
            self._restore_adapter_grads(target_adapter_id)
        else:
            self._zero_active_grads()

    # ------------------------------------------------------------------
    # Training: forward / backward
    # ------------------------------------------------------------------
    def forward(
        self,
        data_shard: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool,
    ) -> types.ForwardBackwardOutput:
        """Forward (and optionally backward) on a data shard."""
        self._switch_adapter(lora_id)

        loss_fn_callable = get_loss_fn(loss_fn)
        batch_size = len(data_shard)
        micro_batch_size = self.micro_batch_size
        num_micro_batches = max(1, (batch_size + micro_batch_size - 1) // micro_batch_size)

        all_loss_fn_outputs: list[dict] = []
        micro_batch_weights: list[float] = []
        metric_list: list[dict] = []
        total_loss = 0.0

        for micro_idx in range(num_micro_batches):
            start = micro_idx * micro_batch_size
            end = min(start + micro_batch_size, batch_size)
            micro_data = data_shard[start:end]

            micro_loss, micro_metrics, micro_outputs = self._forward_micro_batch(
                micro_data, loss_fn_callable, loss_fn_config, backward
            )
            total_loss += micro_loss
            all_loss_fn_outputs.extend(micro_outputs)
            micro_batch_weights.append(len(micro_outputs))
            metric_list.append(micro_metrics)
            torch.cuda.empty_cache()

        if backward:
            self.has_pending_grads[lora_id] = True

        reduced_metrics = metrics_reduction(metric_list, micro_batch_weights)

        return types.ForwardBackwardOutput(
            loss_fn_output_type=loss_fn,
            loss_fn_outputs=all_loss_fn_outputs,
            metrics=reduced_metrics or {},
        )

    def _forward_micro_batch(
        self,
        data: list[types.Datum],
        loss_fn_callable,
        loss_fn_config: dict[str, float] | None,
        backward: bool,
    ) -> tuple[float, dict, list[dict]]:
        """Process one micro-batch."""
        input_ids = [torch.tensor(d.model_input.to_ints(), dtype=torch.long) for d in data]
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = (input_ids_padded != 0).long()
        position_ids = (
            torch.arange(input_ids_padded.size(1), dtype=torch.long)
            .unsqueeze(0)
            .expand(input_ids_padded.size(0), -1)
        )

        device = torch.cuda.current_device()
        input_ids_padded = input_ids_padded.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)

        outputs = self.model(
            input_ids=input_ids_padded,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )

        if loss_fn_config is None:
            loss_fn_config = {}

        logits = outputs.logits
        del outputs
        torch.cuda.empty_cache()

        if "temperature" in loss_fn_config:
            logits = logits / loss_fn_config["temperature"]

        loss_fn_inputs = self._prepare_loss_fn_inputs(data)
        target_tokens = loss_fn_inputs["target_tokens"]
        target_logprobs = self._compute_logprobs_from_target_tokens(logits, target_tokens)
        del logits
        torch.cuda.empty_cache()

        loss_fn_inputs["target_logprobs"] = target_logprobs
        loss, metric = loss_fn_callable(loss_fn_inputs, loss_fn_config)

        if backward:
            loss.backward(retain_graph=False)
            torch.cuda.empty_cache()

        unpadded = self._unpad_tensor(
            target_logprobs.detach(),
            [len(d.model_input.to_ints()) for d in data],
        )
        loss_fn_outputs = [
            {"logprobs": types.TensorData.from_torch(lp.cpu().clone())} for lp in unpadded
        ]

        loss_value = loss.detach().item()
        del target_logprobs, unpadded, loss_fn_inputs, loss
        torch.cuda.empty_cache()

        return loss_value, metric, loss_fn_outputs

    # ------------------------------------------------------------------
    # Training: optimizer step
    # ------------------------------------------------------------------
    def optim_step(
        self,
        adam_params: types.AdamParams,
        lora_id: str,
    ) -> types.OptimStepResponse:
        """Optimizer step with manual gradient synchronization for LoRA params."""
        self._switch_adapter(lora_id)

        optimizer = self.adapter_optimizer[lora_id]
        for pg in optimizer.param_groups:
            pg["lr"] = adam_params.learning_rate
            pg["betas"] = (adam_params.beta1, adam_params.beta2)
            pg["eps"] = adam_params.eps
            pg["weight_decay"] = adam_params.weight_decay

        # LoRA adapter parameters live outside FSDP2 sharding (regular
        # tensors, not DTensors).  FSDP2 only reduces gradients of sharded
        # parameters.  We must AllReduce the LoRA gradients so every rank
        # applies the same optimizer update and parameters stay in sync.
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None and not isinstance(p.grad, DTensor):
                torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if adam_params.grad_clip_norm > 0:
            grad_norm = fsdp2_clip_grad_norm_(trainable, max_norm=adam_params.grad_clip_norm)
        else:
            grads = [p.grad for p in trainable if p.grad is not None]
            if grads:
                grad_norm = torch.stack([g.detach().norm(2) for g in grads]).norm(2)
            else:
                grad_norm = torch.tensor(0.0)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        if not torch.isfinite(grad_norm):
            logger.warning("grad_norm is not finite (%s), skipping update", grad_norm)
            optimizer.zero_grad()
        else:
            optimizer.step()

        optimizer.zero_grad()
        self._clear_saved_grads(lora_id)
        torch.cuda.empty_cache()
        return types.OptimStepResponse()

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------
    def save_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer_flag: bool,
    ) -> None:
        if self.rank == 0:
            adapter_dir = checkpoint_record.adapter_path
            adapter_dir.mkdir(parents=True, exist_ok=True)
            self._peft.save_pretrained(str(adapter_dir), selected_adapters=[lora_id])
            lora_subdir = adapter_dir / lora_id
            if lora_subdir.exists() and lora_subdir.is_dir():
                for item in lora_subdir.iterdir():
                    dest = adapter_dir / item.name
                    if dest.exists():
                        if dest.is_file():
                            dest.unlink()
                        elif dest.is_dir():
                            shutil.rmtree(dest)
                    shutil.move(str(item), str(dest))
                lora_subdir.rmdir()

        if optimizer_flag and lora_id in self.adapter_optimizer:
            opt_dir = checkpoint_record.optimizer_path
            opt_dir.mkdir(parents=True, exist_ok=True)
            opt_path = opt_dir / f"optimizer_rank_{self.rank}.pt"
            torch.save(self.adapter_optimizer[lora_id].state_dict(), opt_path)

        torch.distributed.barrier()

    def load_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer_flag: bool,
    ) -> None:
        adapter_dir = checkpoint_record.adapter_path
        self._peft.load_adapter(str(adapter_dir), adapter_name=lora_id)

        self._peft.set_adapter(lora_id)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params)

        if optimizer_flag:
            opt_path = checkpoint_record.optimizer_path / f"optimizer_rank_{self.rank}.pt"
            if not opt_path.exists():
                legacy = checkpoint_record.optimizer_path / f"{lora_id}_rank_{self.rank}.pt"
                if legacy.exists():
                    opt_path = legacy
            if opt_path.exists():
                optimizer.load_state_dict(
                    torch.load(opt_path, map_location="cpu", weights_only=True)
                )
        self.adapter_optimizer[lora_id] = optimizer
        self.active_adapter = lora_id

        torch.distributed.barrier()

    # ------------------------------------------------------------------
    # GPU memory reporting
    # ------------------------------------------------------------------
    def get_memory_stats(self) -> dict:
        """Return GPU memory statistics for this worker's device."""
        device = torch.cuda.current_device()
        return {
            "rank": self.rank,
            "device": device,
            "allocated_mb": round(torch.cuda.memory_allocated(device) / 1024**2, 1),
            "reserved_mb": round(torch.cuda.memory_reserved(device) / 1024**2, 1),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
        }

    def get_adapter_param_fingerprint(self, lora_id: str) -> dict:
        """Fingerprint of adapter params for cross-rank consistency checks."""
        self._switch_adapter(lora_id)
        total = 0.0
        count = 0
        for p in self.model.parameters():
            if p.requires_grad:
                t = p.data
                if isinstance(t, DTensor):
                    t = t.full_tensor()
                total += t.float().sum().item()
                count += t.numel()
        return {"rank": self.rank, "param_sum": total, "param_count": count}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_loss_fn_inputs(self, data: list[types.Datum]) -> dict[str, torch.Tensor]:
        device = torch.cuda.current_device()
        loss_fn_input_dict: dict[str, torch.Tensor] = {}
        loss_fn_input_keys = data[0].loss_fn_inputs.keys()
        for key in loss_fn_input_keys:
            tensors = [datum.loss_fn_inputs[key].to_torch() for datum in data]
            if all(t.dim() == 1 for t in tensors):
                padded = pad_sequence(tensors, batch_first=True, padding_value=0)
                loss_fn_input_dict[key] = padded.to(device)
            else:
                try:
                    stacked = torch.stack(tensors)
                    loss_fn_input_dict[key] = stacked.to(device)
                except Exception:
                    max_shape = list(tensors[0].shape)
                    for t in tensors:
                        for i, s in enumerate(t.shape):
                            if s > max_shape[i]:
                                max_shape[i] = s
                    padded_tensors = []
                    for t in tensors:
                        pad_width = [(0, m - s) for s, m in zip(t.shape, max_shape, strict=False)]
                        pad_args: list[int] = []
                        for p in reversed(pad_width):
                            pad_args.extend(p)
                        padded_tensors.append(torch.nn.functional.pad(t, pad_args, value=0))
                    stacked = torch.stack(padded_tensors)
                    loss_fn_input_dict[key] = stacked.to(device)
        return loss_fn_input_dict

    def _compute_logprobs_from_target_tokens(
        self, logits: torch.Tensor, target_tokens: torch.Tensor
    ) -> torch.Tensor:
        if logits.dtype in (torch.float32, torch.float64):
            logits_labels = torch.gather(logits, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(
                -1
            )
            logsumexp_values = torch.stack([torch.logsumexp(logit, dim=-1) for logit in logits])
            return logits_labels - logsumexp_values
        log_probs_labels = []
        for row_logits, row_labels in zip(logits, target_tokens, strict=True):
            row_lp = torch.nn.functional.log_softmax(row_logits, dim=-1)
            row_lp_labels = row_lp.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            log_probs_labels.append(row_lp_labels)
        return torch.stack(log_probs_labels)

    def _unpad_tensor(self, padded: torch.Tensor, lengths: list[int]) -> list[torch.Tensor]:
        return [padded[i, :length] for i, length in enumerate(lengths)]
