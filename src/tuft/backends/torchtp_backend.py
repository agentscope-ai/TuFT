from __future__ import annotations

import asyncio
import threading
import concurrent.futures
import contextlib
import json
from logging import getLogger
from pathlib import Path
from typing import Any, Optional

import torch
from safetensors.torch import load_file, save_file
from tinker import types

from tuft.backends.base_backend import BaseSamplingBackend, BaseTrainingBackend
from tuft.checkpoints import CheckpointRecord
from tuft.config import ModelConfig
from tuft.loss_fn import get_loss_fn, metrics_reduction

logger = getLogger(__name__)

# Dedicated pool for autoregressive decoding. asyncio.to_thread uses the loop's
# DEFAULT executor, which training_controller/sampling_controller also use for
# run_in_executor bookkeeping (_save_training_run etc.). A decoupled measurement
# (unified t8, 2026-08-27) showed 30+ concurrent ~50s decodes saturating that
# pool, so eight _save_training_run calls queued behind them for hours, every
# create_model hung after "create_adapter returned", and training never started.
# Decodes get their own pool so control-plane executor work is never starved.
_DECODE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=64, thread_name_prefix="tuft-decode"
)


LORA_ID_TO_KEY_ATTR = "_adapter_key_by_id"


def _is_qwen_model(config: ModelConfig) -> bool:
    return "qwen" in str(config.model_path).lower() or "qwen" in config.model_name.lower()


@contextlib.contextmanager
def _pinned_device(device_index: int | None):
    """Temporarily set the CUDA device so freshly initialized tensors (e.g.
    LoRA A/B params in apply_fused_torchtp_lora) land on the replica's GPU
    instead of the default device 0. set_device is thread-local."""
    if device_index is None or not torch.cuda.is_available():
        yield
        return
    prev = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    try:
        yield
    finally:
        torch.cuda.set_device(prev)


def _lora_options(config: ModelConfig, lora_config: types.LoraConfig | None) -> dict[str, Any]:
    rank = int(getattr(lora_config, "rank", 8) or 8) if lora_config is not None else 8
    train_attn = bool(getattr(lora_config, "train_attn", True)) if lora_config is not None else True
    train_mlp = bool(getattr(lora_config, "train_mlp", True)) if lora_config is not None else True
    train_unembed = bool(getattr(lora_config, "train_unembed", True)) if lora_config is not None else True
    if _is_qwen_model(config):
        train_unembed = False
    return {
        "rank": rank,
        "alpha": rank,
        "train_attn": train_attn,
        "train_mlp": train_mlp,
        "train_unembed": train_unembed,
    }


def _adapter_for(module: Any, lora_id: str) -> Any | None:
    key_by_id = getattr(module, LORA_ID_TO_KEY_ATTR, None)
    adapters = getattr(module, "adapters", None)
    if not key_by_id or adapters is None or lora_id not in key_by_id:
        return None
    return adapters[key_by_id[lora_id]]


def _copy_if_present(adapter: Any, attr: str, tensors: dict[str, torch.Tensor], key: str) -> None:
    tensor = tensors.get(key)
    if tensor is None:
        return
    param = getattr(adapter, attr)
    param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))


def _load_peft_tensors_into_fused_model(model: Any, lora_id: str, tensors: dict[str, torch.Tensor]) -> None:
    from tuft.backends.flex.torchtp_training import (
        FusedGateUpLoRA,
        FusedQKVLoRA,
        ReplicatedLoRALinear,
        RowwiseLoRALinear,
    )

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is not None:
        for index, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                prefix = f"base_model.model.model.layers.{index}.self_attn"
                qkv = getattr(attn, "qkv_proj", None)
                if isinstance(qkv, FusedQKVLoRA):
                    adapter = _adapter_for(qkv, lora_id)
                    if adapter is not None:
                        _copy_if_present(adapter, "q_A", tensors, f"{prefix}.q_proj.lora_A.weight")
                        _copy_if_present(adapter, "q_B", tensors, f"{prefix}.q_proj.lora_B.weight")
                        _copy_if_present(adapter, "k_A", tensors, f"{prefix}.k_proj.lora_A.weight")
                        _copy_if_present(adapter, "k_B", tensors, f"{prefix}.k_proj.lora_B.weight")
                        _copy_if_present(adapter, "v_A", tensors, f"{prefix}.v_proj.lora_A.weight")
                        _copy_if_present(adapter, "v_B", tensors, f"{prefix}.v_proj.lora_B.weight")
                o_proj = getattr(attn, "o_proj", None)
                if isinstance(o_proj, RowwiseLoRALinear):
                    adapter = _adapter_for(o_proj, lora_id)
                    if adapter is not None:
                        _copy_if_present(adapter, "lora_A", tensors, f"{prefix}.o_proj.lora_A.weight")
                        _copy_if_present(adapter, "lora_B", tensors, f"{prefix}.o_proj.lora_B.weight")
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                prefix = f"base_model.model.model.layers.{index}.mlp"
                gate_up = getattr(mlp, "gate_up_proj", None)
                if isinstance(gate_up, FusedGateUpLoRA):
                    adapter = _adapter_for(gate_up, lora_id)
                    if adapter is not None:
                        _copy_if_present(adapter, "gate_A", tensors, f"{prefix}.gate_proj.lora_A.weight")
                        _copy_if_present(adapter, "gate_B", tensors, f"{prefix}.gate_proj.lora_B.weight")
                        _copy_if_present(adapter, "up_A", tensors, f"{prefix}.up_proj.lora_A.weight")
                        _copy_if_present(adapter, "up_B", tensors, f"{prefix}.up_proj.lora_B.weight")
                down = getattr(mlp, "down_proj", None)
                if isinstance(down, RowwiseLoRALinear):
                    adapter = _adapter_for(down, lora_id)
                    if adapter is not None:
                        _copy_if_present(adapter, "lora_A", tensors, f"{prefix}.down_proj.lora_A.weight")
                        _copy_if_present(adapter, "lora_B", tensors, f"{prefix}.down_proj.lora_B.weight")
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, ReplicatedLoRALinear):
        adapter = _adapter_for(lm_head, lora_id)
        if adapter is not None:
            _copy_if_present(adapter, "lora_A", tensors, "base_model.model.lm_head.lora_A.weight")
            _copy_if_present(adapter, "lora_B", tensors, "base_model.model.lm_head.lora_B.weight")


def _peft_rank_alpha(adapter_path: Path) -> tuple[int, int]:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return 8, 8
    data = json.loads(config_path.read_text())
    rank = int(data.get("r", 8) or 8)
    alpha = int(data.get("lora_alpha", rank) or rank)
    return rank, alpha


class TorchTPTrainingBackend(BaseTrainingBackend):
    """Independent TorchTP training backend using the fused TorchTP model path."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        model: Any | None = None,
        device_index: int | None = None,
    ) -> None:
        super().__init__(config)
        self.model = model
        self._device_index = device_index
        self._tp_group: Any | None = None
        self._tp_mesh: Any | None = None
        self._adapter_params: dict[str, list[Any]] = {}
        self._adapter_optimizers: dict[str, torch.optim.Optimizer] = {}
        self._adapter_configs: dict[str, types.LoraConfig] = {}

    async def async_init(self) -> None:
        if self.model is not None:
            return
        # Load in a worker thread: from_pretrained + .to(cuda) is synchronous
        # and would otherwise stall the event loop (heartbeats, other tenants).
        await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self) -> None:
        from tuft.backends.flex.torchtp_training import load_fused_torchtp_model, resolve_flex_model_spec

        spec = resolve_flex_model_spec(self.config.model_name)
        # Pin DP replicas to distinct GPUs: load_fused_torchtp_model moves the
        # model to torch.cuda.current_device(), so set (and restore) the device
        # around the load. set_device is thread-local, safe under to_thread.
        prev_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        if self._device_index is not None and torch.cuda.is_available():
            torch.cuda.set_device(self._device_index)
        try:
            self.model, self._tp_group, self._tp_mesh = load_fused_torchtp_model(
                spec,
                rank=0,
                world_size=int(self.config.tensor_parallel_size or 1),
                attn_implementation=self.config.attn_implementation or "sdpa",
            )
        finally:
            if prev_device is not None and self._device_index is not None:
                torch.cuda.set_device(prev_device)

    async def create_adapter(self, lora_id: str, lora_config: types.LoraConfig) -> None:
        await self.async_init()
        if lora_id in self._adapter_optimizers:
            raise ValueError(f"Adapter {lora_id} already exists.")
        from tuft.backends.flex.torchtp_training import apply_fused_torchtp_lora

        opts = _lora_options(self.config, lora_config)
        seed = getattr(lora_config, "seed", None)
        if seed is not None:
            torch.manual_seed(int(seed))
        # New LoRA params initialize on the current device; pin the replica's
        # GPU so DP replicas (device_index > 0) don't get cuda:0 params.
        with _pinned_device(self._device_index):
            params = apply_fused_torchtp_lora(
                self.model,
                lora_rank=opts["rank"],
                lora_alpha=opts["alpha"],
                tp_group=self._tp_group,
                lora_id=lora_id,
                train_attn=opts["train_attn"],
                train_mlp=opts["train_mlp"],
                train_unembed=opts["train_unembed"],
            )
        self._adapter_params[lora_id] = params
        self._adapter_optimizers[lora_id] = torch.optim.AdamW(params)
        self._adapter_configs[lora_id] = lora_config

    async def remove_adapter(self, lora_id: str) -> None:
        from tuft.backends.flex.torchtp_training import remove_fused_torchtp_lora_adapter

        if self.model is not None:
            remove_fused_torchtp_lora_adapter(self.model, lora_id)
        self._adapter_params.pop(lora_id, None)
        self._adapter_optimizers.pop(lora_id, None)
        self._adapter_configs.pop(lora_id, None)

    def _prepare_loss_fn_inputs(self, data: list[types.Datum], device: torch.device) -> dict[str, torch.Tensor]:
        from torch.nn.utils.rnn import pad_sequence

        result: dict[str, torch.Tensor] = {}
        for key in data[0].loss_fn_inputs.keys():
            tensors = [datum.loss_fn_inputs[key].to_torch() for datum in data]
            if all(tensor.dim() == 1 for tensor in tensors):
                result[key] = pad_sequence(tensors, batch_first=True, padding_value=0).to(device)
            else:
                result[key] = torch.stack(tensors).to(device)
        return result

    def _compute_logprobs_from_target_tokens(self, logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        rows = []
        for row_logits, row_labels in zip(logits, target_tokens, strict=True):
            row_log_probs = torch.nn.functional.log_softmax(row_logits, dim=-1)
            rows.append(row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1))
        return torch.stack(rows)

    async def forward(
        self,
        data: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool = False,
    ) -> types.ForwardBackwardOutput:
        await self.async_init()
        if lora_id not in self._adapter_optimizers:
            raise ValueError(f"Adapter {lora_id} not found.")
        # Run the GPU forward/backward in a worker thread: the fused TorchTP model
        # forward is synchronous and would otherwise block the asyncio event loop,
        # stalling heartbeats and every other in-flight request.
        return await asyncio.to_thread(
            self._forward_sync, data, lora_id, loss_fn, loss_fn_config, backward
        )

    def _forward_sync(
        self,
        data: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool,
    ) -> types.ForwardBackwardOutput:
        from torch.nn.utils.rnn import pad_sequence
        from tuft.backends.flex.torchtp_training import set_fused_torchtp_lora_adapter

        set_fused_torchtp_lora_adapter(self.model, lora_id)
        if hasattr(self.model, "train"):
            self.model.train()
        loss_fn_callable = get_loss_fn(loss_fn)
        micro_batch_size = max(int(self.config.micro_batch_size or 1), 1)
        num_micro_batches = (len(data) + micro_batch_size - 1) // micro_batch_size
        outputs: list[dict[str, types.TensorData]] = []
        metrics_list: list[dict[str, float]] = []
        weights: list[float] = []
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.set_device(device.index or 0)
        for index in range(num_micro_batches):
            micro = data[index * micro_batch_size : (index + 1) * micro_batch_size]
            input_ids = [torch.tensor(d.model_input.to_ints(), dtype=torch.long) for d in micro]
            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0).to(device)
            attention_mask = (input_ids_padded != 0).long().to(device)
            position_ids = (
                torch.arange(input_ids_padded.size(1), dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(input_ids_padded.size(0), -1)
            )
            try:
                result = self.model(
                    input_ids=input_ids_padded,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_dict=True,
                )
            except RuntimeError as exc:
                # Diagnostic + self-heal for the intermittent float32 drift:
                # log every non-bf16 parameter, cast it back, and retry once.
                if "same dtype" not in str(exc):
                    raise
                offenders = [
                    (name, str(p.dtype))
                    for name, p in self.model.named_parameters()
                    if p.dtype != torch.bfloat16
                ]
                logger.error(
                    "dtype drift on forward (lora=%s): %s", lora_id, offenders[:20]
                )
                for _name, p in self.model.named_parameters():
                    if p.dtype != torch.bfloat16:
                        p.data = p.data.to(torch.bfloat16)
                result = self.model(
                    input_ids=input_ids_padded,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_dict=True,
                )
            logits = result.logits
            inputs = self._prepare_loss_fn_inputs(micro, device)
            target_logprobs = self._compute_logprobs_from_target_tokens(logits, inputs["target_tokens"])
            inputs["target_logprobs"] = target_logprobs
            loss, metric = loss_fn_callable(inputs, loss_fn_config or {})
            if backward:
                loss.backward(retain_graph=False)
            lengths = [len(datum.loss_fn_inputs["target_tokens"].tolist()) for datum in micro]
            for row, length in zip(target_logprobs.detach(), lengths, strict=True):
                outputs.append({"logprobs": types.TensorData.from_torch(row[:length].cpu().clone())})
            metrics_list.append(metric)
            weights.append(float(len(micro)))
        return types.ForwardBackwardOutput(
            loss_fn_output_type=loss_fn,
            loss_fn_outputs=outputs,
            metrics=metrics_reduction(metrics_list, weights),
        )

    async def optim_step(self, adam_params: types.AdamParams, lora_id: str) -> types.OptimStepResponse:
        return await asyncio.to_thread(self._optim_step_sync, adam_params, lora_id)

    def _optim_step_sync(self, adam_params: types.AdamParams, lora_id: str) -> types.OptimStepResponse:
        optimizer = self._adapter_optimizers[lora_id]
        device = next(iter(optimizer.param_groups[0]["params"])).device
        if device.type == "cuda":
            torch.cuda.set_device(device.index or 0)
        for group in optimizer.param_groups:
            group["lr"] = adam_params.learning_rate
            group["betas"] = (adam_params.beta1, adam_params.beta2)
            group["eps"] = adam_params.eps
            group["weight_decay"] = adam_params.weight_decay
        optimizer.step()
        optimizer.zero_grad()
        return types.OptimStepResponse()

    async def save_state(self, lora_id: str, checkpoint_record: CheckpointRecord, optimizer: bool) -> None:
        if lora_id not in self._adapter_configs:
            raise ValueError(f"Adapter {lora_id} not found.")
        from tuft.backends.flex.torchtp_training import fused_adapter_peft_state_dict, write_peft_adapter_dir

        opts = _lora_options(self.config, self._adapter_configs[lora_id])
        tensors = fused_adapter_peft_state_dict(self.model, lora_id)
        write_peft_adapter_dir(
            checkpoint_record.adapter_path,
            tensors,
            rank=opts["rank"],
            alpha=opts["alpha"],
            base_model_name_or_path=str(self.config.model_path),
        )
        if optimizer:
            checkpoint_record.optimizer_path.mkdir(parents=True, exist_ok=True)
            torch.save(self._adapter_optimizers[lora_id].state_dict(), checkpoint_record.optimizer_path / f"{lora_id}.pt")

    async def load_state(self, lora_id: str, checkpoint_record: CheckpointRecord, optimizer: bool) -> None:
        rank, alpha = _peft_rank_alpha(checkpoint_record.adapter_path)
        config = types.LoraConfig(rank=rank)
        await self.create_adapter(lora_id, config)
        tensors = load_file(checkpoint_record.adapter_path / "adapter_model.safetensors")
        _load_peft_tensors_into_fused_model(self.model, lora_id, tensors)
        if optimizer:
            opt_path = checkpoint_record.optimizer_path / f"{lora_id}.pt"
            if opt_path.exists():
                self._adapter_optimizers[lora_id].load_state_dict(torch.load(opt_path))


class TorchTPSamplingBackend(BaseSamplingBackend):
    """Torch autoregressive sampling backend with per-sequence KV cache."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        model: Any | None = None,
        device_index: int | None = None,
    ) -> None:
        super().__init__(config)
        self.model = model
        self._device_index = device_index
        self._tp_group: Any | None = None
        self._tp_mesh: Any | None = None
        self._adapter_configs: dict[str, types.LoraConfig] = {}
        # Unified-engine sampling hangs when several asyncio.to_thread workers decode
        # on the same TorchTP model at once (measured 2026-08-27: 8 replicas x 4
        # samples all entered _sample_sync, none returned, GPUs at 3%). Serialize
        # decoding per replica: drain semantics already give sampling exclusive
        # engine time between training steps, so one decode at a time loses nothing.
        self._decode_lock = threading.Lock()

    async def async_init(self) -> None:
        if self.model is not None:
            return
        await asyncio.to_thread(self._load_model_sync)
        self.model.config.use_cache = True
        if hasattr(self.model, "eval"):
            self.model.eval()

    def _load_model_sync(self) -> None:
        from tuft.backends.flex.torchtp_training import load_fused_torchtp_model, resolve_flex_model_spec

        spec = resolve_flex_model_spec(self.config.model_name)
        prev_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        if self._device_index is not None and torch.cuda.is_available():
            torch.cuda.set_device(self._device_index)
        try:
            self.model, self._tp_group, self._tp_mesh = load_fused_torchtp_model(
                spec,
                rank=0,
                world_size=int(self.config.tensor_parallel_size or 1),
                attn_implementation=self.config.attn_implementation or "sdpa",
            )
        finally:
            if prev_device is not None and self._device_index is not None:
                torch.cuda.set_device(prev_device)

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        await self.async_init()
        from tuft.backends.flex.torchtp_training import apply_fused_torchtp_lora

        rank, alpha = _peft_rank_alpha(adapter_path)
        lora_config = types.LoraConfig(rank=rank)
        opts = _lora_options(self.config, lora_config)
        # Pin the replica's GPU so the freshly created LoRA params and the
        # copied PEFT tensors stay on the same device as the model.
        with _pinned_device(self._device_index):
            apply_fused_torchtp_lora(
                self.model,
                lora_rank=rank,
                lora_alpha=alpha,
                tp_group=self._tp_group,
                lora_id=lora_id,
                train_attn=opts["train_attn"],
                train_mlp=opts["train_mlp"],
                train_unembed=opts["train_unembed"],
            )
            tensors = load_file(adapter_path / "adapter_model.safetensors")
            _load_peft_tensors_into_fused_model(self.model, lora_id, tensors)
        self._adapter_configs[lora_id] = lora_config

    async def remove_adapter(self, lora_id: str) -> None:
        from tuft.backends.flex.torchtp_training import remove_fused_torchtp_lora_adapter

        if self.model is not None:
            remove_fused_torchtp_lora_adapter(self.model, lora_id)
        self._adapter_configs.pop(lora_id, None)

    def _sample_next(self, logits: torch.Tensor, params: types.SamplingParams) -> tuple[int, float]:
        temperature = float(params.temperature or 1.0)
        scores = logits.float() / max(temperature, 1e-6)
        top_k = int(params.top_k if params.top_k is not None else -1)
        if top_k and top_k > 0 and top_k < scores.numel():
            values, indices = torch.topk(scores, top_k)
            filtered = torch.full_like(scores, float("-inf"))
            filtered.scatter_(0, indices, values)
            scores = filtered
        probs = torch.softmax(scores, dim=-1)
        top_p = float(params.top_p if params.top_p is not None else 1.0)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > top_p
            remove[0] = False
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum()
            sampled_pos = torch.multinomial(sorted_probs, 1)
            token = int(sorted_indices[sampled_pos].item())
        elif temperature <= 1e-6:
            token = int(torch.argmax(probs).item())
        else:
            token = int(torch.multinomial(probs, 1).item())
        logprob = float(torch.log(probs[token].clamp_min(1e-30)).item())
        return token, logprob

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        await self.async_init()
        # Autoregressive generation is synchronous and long-running. It runs on the
        # dedicated decode pool, NOT the loop's default executor: asyncio.to_thread
        # saturated the default pool with ~50s decodes and starved the controllers'
        # run_in_executor bookkeeping (measured 2026-08-27: eight create_model calls
        # hung in _save_training_run for hours, training never started).
        from tuft.backends.flex.torchtp_training import set_fused_torchtp_lora_adapter

        logger.info("[unified-probe] sample dispatch lora=%s n=%s", lora_id, num_samples)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _DECODE_EXECUTOR,
            self._decode_locked,
            set_fused_torchtp_lora_adapter,
            prompt,
            num_samples,
            sampling_params,
            lora_id,
        )

    def _decode_locked(
        self,
        set_fused_torchtp_lora_adapter,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        lora_id: Optional[str],
    ) -> types.SampleResponse:
        # One decode at a time per replica: concurrent torch decodes on the same
        # TorchTP model hung outright (measured: 8 replicas x 4 samples entered,
        # none returned, GPUs idle). Drain semantics already give sampling
        # exclusive engine time between training steps, so this loses nothing.
        with self._decode_lock:
            return self._decode_inner(
                set_fused_torchtp_lora_adapter, prompt, num_samples, sampling_params, lora_id
            )

    def _decode_inner(
        self,
        set_fused_torchtp_lora_adapter,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        lora_id: Optional[str],
    ) -> types.SampleResponse:
        if lora_id is not None:
            if lora_id not in self._adapter_configs:
                raise ValueError(f"LoRA adapter {lora_id} not found in TorchTPSamplingBackend.")
            set_fused_torchtp_lora_adapter(self.model, lora_id)
        if hasattr(self.model, "eval"):
            self.model.eval()
        max_tokens = int(sampling_params.max_tokens or self.config.default_max_tokens or 16)
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.set_device(device.index or 0)
        sequences: list[types.SampledSequence] = []
        base_ids = prompt.to_ints()
        with torch.inference_mode():
            for sample_index in range(num_samples):
                if sampling_params.seed is not None:
                    torch.manual_seed(int(sampling_params.seed) + sample_index)
                input_ids = torch.tensor([base_ids], dtype=torch.long, device=device)
                outputs = self.model(input_ids=input_ids, use_cache=True, return_dict=True)
                past_key_values = getattr(outputs, "past_key_values", None)
                logits = outputs.logits[:, -1, :].squeeze(0)
                tokens: list[int] = []
                logprobs: list[float] = []
                stop_reason = "length"
                for _ in range(max_tokens):
                    token, logprob = self._sample_next(logits, sampling_params)
                    tokens.append(token)
                    logprobs.append(logprob)
                    if sampling_params.stop and token in [s for s in sampling_params.stop if isinstance(s, int)]:
                        stop_reason = "stop"
                        break
                    next_input = torch.tensor([[token]], dtype=torch.long, device=device)
                    outputs = self.model(
                        input_ids=next_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = getattr(outputs, "past_key_values", None)
                    logits = outputs.logits[:, -1, :].squeeze(0)
                sequences.append(
                    types.SampledSequence(
                        stop_reason=stop_reason,
                        _tokens_list=tokens,
                        _logprobs_list=logprobs,
                    )
                )
        return types.SampleResponse(sequences=sequences)


_UNIFIED_REPLICA_GATES: dict[str, list] = {}


def _unified_replica_gates(config: ModelConfig, dp_size: int):
    """Per-replica UnifiedEngineGate list shared by the training and sampling
    DP wrappers of one model. Returns None outside unified_engine mode (the
    single-replica path keeps using the server-level gate instead)."""
    ev = getattr(config, "evaluation", None)
    if ev is None or getattr(ev, "deployment_mode", None) != "unified_engine":
        return None
    gates = _UNIFIED_REPLICA_GATES.get(config.model_name)
    if gates is None:
        from tuft.runtime.unified_engine_gate import UnifiedEngineGate

        window = float(getattr(ev, "unified_engine_sampling_min_window_s", 3.0) or 3.0)
        gates = [UnifiedEngineGate(sampling_min_window_s=window) for _ in range(dp_size)]
        _UNIFIED_REPLICA_GATES[config.model_name] = gates
    return gates


class DPTorchTPTrainingBackend(BaseTrainingBackend):
    """Data-parallel unified TorchTP training backend (e.g. DP4 TP1 for 4B).

    Holds N independent single-GPU TorchTP replicas pinned to GPUs 0..N-1.
    Adapters are assigned round-robin to replicas at creation; every training
    op for a tenant routes to the replica owning its adapter. In unified_engine
    mode each replica alternates train/sample independently via its own
    UnifiedEngineGate (shared with the sampling wrapper through a per-model
    registry), so DP replicas do not serialize on one global gate.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._dp_size = max(int(config.data_parallel_size or 1), 1)
        self._replicas = [
            TorchTPTrainingBackend(config, device_index=i) for i in range(self._dp_size)
        ]
        self._adapter_replica: dict[str, int] = {}
        self._rr_counter = 0
        self._gates = _unified_replica_gates(config, self._dp_size)

    def _replica_for(self, lora_id: str) -> TorchTPTrainingBackend:
        index = self._adapter_replica.get(lora_id)
        if index is None:
            raise ValueError(f"Adapter {lora_id} not found in DPTorchTPTrainingBackend.")
        return self._replicas[index]

    async def async_init(self) -> None:
        # Load replicas sequentially: concurrent CUDA context initialization on
        # different devices from multiple threads hits driver-level races
        # ("CUDA driver error: operation not permitted").
        for replica in self._replicas:
            await replica.async_init()

    async def create_adapter(self, lora_id: str, lora_config: types.LoraConfig) -> None:
        index = self._rr_counter % self._dp_size
        self._rr_counter += 1
        self._adapter_replica[lora_id] = index
        await self._replicas[index].create_adapter(lora_id, lora_config)

    async def remove_adapter(self, lora_id: str) -> None:
        index = self._adapter_replica.pop(lora_id, None)
        if index is not None:
            await self._replicas[index].remove_adapter(lora_id)

    async def forward(
        self,
        data: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool = False,
    ) -> types.ForwardBackwardOutput:
        index = self._adapter_replica.get(lora_id)
        if index is None:
            raise ValueError(f"Adapter {lora_id} not found in DPTorchTPTrainingBackend.")
        if self._gates is not None:
            gate = self._gates[index]
            await gate.acquire_training()
            try:
                return await self._replicas[index].forward(
                    data, lora_id, loss_fn, loss_fn_config, backward
                )
            finally:
                await gate.release_training()
        return await self._replicas[index].forward(
            data, lora_id, loss_fn, loss_fn_config, backward
        )

    async def optim_step(self, adam_params: types.AdamParams, lora_id: str) -> types.OptimStepResponse:
        index = self._adapter_replica.get(lora_id)
        if index is None:
            raise ValueError(f"Adapter {lora_id} not found in DPTorchTPTrainingBackend.")
        if self._gates is not None:
            gate = self._gates[index]
            await gate.acquire_training()
            try:
                return await self._replicas[index].optim_step(adam_params, lora_id)
            finally:
                await gate.release_training()
        return await self._replicas[index].optim_step(adam_params, lora_id)

    async def save_state(self, lora_id: str, checkpoint_record: CheckpointRecord, optimizer: bool) -> None:
        await self._replica_for(lora_id).save_state(lora_id, checkpoint_record, optimizer)

    async def load_state(self, lora_id: str, checkpoint_record: CheckpointRecord, optimizer: bool) -> None:
        # Assign the restored adapter to a replica the same way create would.
        index = self._rr_counter % self._dp_size
        self._rr_counter += 1
        self._adapter_replica[lora_id] = index
        await self._replicas[index].load_state(lora_id, checkpoint_record, optimizer)

    def unified_gate_snapshot(self) -> dict[str, float]:
        """Aggregate per-replica gate counters for the evaluation snapshot."""
        if not self._gates:
            return {}
        agg = {
            "replicas": float(len(self._gates)),
            "switches_to_training": 0.0,
            "switches_to_sampling": 0.0,
            "sampling_hold_seconds": 0.0,
        }
        for gate in self._gates:
            snap = gate.snapshot()
            agg["switches_to_training"] += snap["switches_to_training"]
            agg["switches_to_sampling"] += snap["switches_to_sampling"]
            agg["sampling_hold_seconds"] += snap["sampling_hold_seconds"]
        return agg


class DPTorchTPSamplingBackend(BaseSamplingBackend):
    """Data-parallel TorchTP sampling backend mirroring DPTorchTPTrainingBackend.

    Adapters are broadcast to every replica (weight sync is cheap for LoRA
    deltas), so any tenant's sample can land on any replica; base-model
    samples are distributed round-robin.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._dp_size = max(int(config.data_parallel_size or 1), 1)
        self._replicas = [
            TorchTPSamplingBackend(config, device_index=i) for i in range(self._dp_size)
        ]
        self._rr_counter = 0
        self._gates = _unified_replica_gates(config, self._dp_size)

    async def async_init(self) -> None:
        # Sequential loads: see DPTorchTPTrainingBackend.async_init for why
        # concurrent per-device CUDA init is unsafe here.
        for replica in self._replicas:
            await replica.async_init()

    def _next_replica(self) -> TorchTPSamplingBackend:
        replica = self._replicas[self._rr_counter % self._dp_size]
        self._rr_counter += 1
        return replica

    def _pick_sampling_replica(self) -> int:
        """Round-robin start, prefer a replica with no training in flight so
        samples do not wait behind another tenant's train step on one engine."""
        start = self._rr_counter % self._dp_size
        self._rr_counter += 1
        if self._gates is not None:
            for offset in range(self._dp_size):
                idx = (start + offset) % self._dp_size
                if self._gates[idx].sampling_idle():
                    return idx
        return start

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        idx = self._pick_sampling_replica()
        logger.info("[unified-probe] DP sample: picked replica %s lora=%s", idx, lora_id)
        if self._gates is not None:
            await self._gates[idx].acquire_sampling()
        logger.info("[unified-probe] DP sample: entering replica %s engine", idx)
        resp = await self._replicas[idx].sample(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
            lora_id=lora_id,
        )
        logger.info("[unified-probe] DP sample: replica %s returned", idx)
        return resp

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        await asyncio.gather(*[r.add_adapter(lora_id, adapter_path) for r in self._replicas])

    async def remove_adapter(self, lora_id: str) -> None:
        await asyncio.gather(*[r.remove_adapter(lora_id) for r in self._replicas])
