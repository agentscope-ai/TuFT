import asyncio
import shutil
from typing import Dict

import ray
import torch
from opentelemetry.trace import StatusCode
from peft import LoraConfig, get_peft_model
from ray.actor import ActorProxy
from tinker import types
from tinker.types import LoraConfig as TinkerLoraConfig
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM

from tuft.checkpoints import CheckpointRecord
from tuft.config import ModelConfig
from tuft.loss_fn import get_loss_fn
from tuft.telemetry.tracing import extract_context, get_tracer


_get_tracer = lambda: get_tracer("tuft.hf_training_model")  # noqa: E731


MODULE_MAP = {
    "llama": {
        "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "unembed": ["lm_head"],
    },
    "qwen": {
        "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "unembed": [],  # set unembed will cause warning in Qwen models
    },
}


def get_target_modules(model_path: str, lora_config: TinkerLoraConfig) -> list[str]:
    if "qwen" in model_path.lower():
        mode_series = "qwen"
    elif "llama" in model_path.lower():
        mode_series = "llama"
    else:
        raise ValueError(f"Unsupported model series: {model_path}")
    target_modules = []
    if lora_config.train_attn:
        target_modules.extend(MODULE_MAP[mode_series]["attn"])
    if lora_config.train_mlp:
        target_modules.extend(MODULE_MAP[mode_series]["mlp"])
    if lora_config.train_unembed:
        target_modules.extend(MODULE_MAP[mode_series]["unembed"])
    return target_modules


class HFTrainingModel:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model = self._init_peft_model(config)
        self.adapter_optimizer: Dict[str, torch.optim.AdamW] = {}
        self._lock = asyncio.Lock()

    async def async_init(self) -> None:
        """Do nothing for now. Just used to make sure the actor is ready."""
        pass

    # --------------------------------
    # LoRA adapter management methods
    # --------------------------------
    async def create_adapter(
        self,
        lora_id: str,
        lora_config: TinkerLoraConfig,
        trace_context: dict[str, str] | None = None,
    ):
        ctx = extract_context(trace_context or {})
        with _get_tracer().start_as_current_span("hf_model.create_adapter", context=ctx) as span:
            span.set_attribute("tuft.lora_id", lora_id)
            try:
                if lora_id in self.adapter_optimizer:
                    raise ValueError(f"Adapter {lora_id} already exists.")
                peft_config = LoraConfig(
                    r=lora_config.rank,
                    target_modules=get_target_modules(str(self.config.model_path), lora_config),
                    # TODO: here we set lora_alpha equal to rank for common practice,
                    # but we may expose it in the future if needed.
                    lora_alpha=lora_config.rank,
                )

                self.model.add_adapter(adapter_name=lora_id, peft_config=peft_config)
                async with self._lock:
                    self.model.set_adapter(lora_id)
                    params = [p for p in self.model.parameters() if p.requires_grad]
                    self.adapter_optimizer[lora_id] = torch.optim.AdamW(params)
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR)
                raise

    async def save_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer: bool,
        trace_context: dict[str, str] | None = None,
    ):
        """
        Save LoRA adapter and optimizer state.
        Args:
            lora_id: The LoRA adapter ID to save.
            checkpoint_record: The CheckpointRecord containing paths to save to.
            optimizer: Whether to save the optimizer state.
            trace_context: Optional trace context for distributed tracing.
        """
        ctx = extract_context(trace_context or {})
        with _get_tracer().start_as_current_span("hf_model.save_state", context=ctx) as span:
            span.set_attribute("tuft.lora_id", lora_id)
            span.set_attribute("tuft.optimizer", optimizer)
            try:
                if lora_id not in self.adapter_optimizer:
                    raise ValueError(f"Adapter {lora_id} not found.")

                # 1. Save adapter (LoRA weights)
                adapter_dir = checkpoint_record.adapter_path
                adapter_dir.mkdir(parents=True, exist_ok=True)
                # peft automatically creates a subdirectory with adapter name inside the given path
                self.model.save_pretrained(str(adapter_dir), selected_adapters=[lora_id])
                # move the files out of the subdirectory
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

                # 2. Save optimizer state
                if optimizer:
                    opt_dir = checkpoint_record.optimizer_path
                    opt_dir.mkdir(parents=True, exist_ok=True)
                    opt_state = self.adapter_optimizer[lora_id].state_dict()
                    opt_path = opt_dir / (f"{lora_id}.pt")
                    torch.save(opt_state, opt_path)
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR)
                raise

    async def load_state(
        self,
        lora_id: str,
        checkpoint_record: CheckpointRecord,
        optimizer: bool,
        trace_context: dict[str, str] | None = None,
    ):
        """
        Load LoRA adapter and optimizer state (standard format).
        Args:
            lora_id: The LoRA adapter ID to load.
            checkpoint_record: The CheckpointRecord containing paths to load from.
            optimizer: Whether to load the optimizer state.
            trace_context: Optional trace context for distributed tracing.
        """
        ctx = extract_context(trace_context or {})
        with _get_tracer().start_as_current_span("hf_model.load_state", context=ctx) as span:
            span.set_attribute("tuft.lora_id", lora_id)
            span.set_attribute("tuft.optimizer", optimizer)
            # 1. Load adapter
            # find lora adapter name from the directory
            self.model.load_adapter(
                model_id=str(checkpoint_record.adapter_path), adapter_name=lora_id
            )

            # 2. Load optimizer state if needed
            async with self._lock:
                self.model.set_adapter(lora_id)
                params = [p for p in self.model.parameters() if p.requires_grad]
                optimizer_obj = torch.optim.AdamW(params)
                if optimizer:
                    opt_dir = checkpoint_record.optimizer_path
                    opt_path = opt_dir / f"{lora_id}.pt"
                    state_dict = None
                    if opt_path.exists():
                        state_dict = torch.load(opt_path)
                    if state_dict is not None:
                        optimizer_obj.load_state_dict(state_dict)
                self.adapter_optimizer[lora_id] = optimizer_obj

    async def remove_adapter(self, lora_id: str):
        async with self._lock:
            if lora_id in self.adapter_optimizer:
                self.model.delete_adapter(lora_id)
                self.adapter_optimizer.pop(lora_id)

    # --------------------------------
    # Training methods
    # --------------------------------
    async def forward(
        self,
        data: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool = False,
        trace_context: dict[str, str] | None = None,
    ) -> types.ForwardBackwardOutput:
        """Forward pass (and backward if specified).

        Args:
            data: List of Datum objects containing input data.
            lora_id: The LoRA adapter ID to use.
            loss_fn: The loss function to apply.
            loss_fn_config: Optional configuration for the loss function.
            backward: Whether to perform backward pass.
            trace_context: Optional trace context for distributed tracing.

        Returns:
            ForwardBackwardOutput: The output of the forward (and backward) pass.
        """
        ctx = extract_context(trace_context or {})
        span_name = "hf_model.forward_backward" if backward else "hf_model.forward"
        with _get_tracer().start_as_current_span(span_name, context=ctx) as span:
            span.set_attribute("tuft.lora_id", lora_id)
            span.set_attribute("tuft.backward", backward)
            span.set_attribute("tuft.data_count", len(data))
            # Prepare input tensors
            input_ids = [
                torch.tensor(datum.model_input.to_ints(), dtype=torch.long) for datum in data
            ]
            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
            attention_mask = (input_ids_padded != 0).long()
            position_ids = (
                torch.arange(input_ids_padded.size(1), dtype=torch.long)
                .unsqueeze(0)
                .expand(input_ids_padded.size(0), -1)
            )
            # Move tensors to model device
            device = next(self.model.parameters()).device
            input_ids_padded = input_ids_padded.to(device)
            attention_mask = attention_mask.to(device)
            position_ids = position_ids.to(device)

            # Activate the correct adapter
            async with self._lock:
                self._activate_adapter(lora_id)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids_padded,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    return_dict=True,
                )

                # Compute loss
                if loss_fn_config is None:
                    loss_fn_config = {}
                loss_fn_callable = get_loss_fn(loss_fn)
                logits = outputs.logits
                if "temperature" in loss_fn_config:
                    temperature = loss_fn_config["temperature"]
                    logits.div_(temperature)

                loss_fn_inputs = self._prepare_loss_fn_inputs(data)

                ## compute target_logprobs from logits and target_tokens
                target_tokens = loss_fn_inputs["target_tokens"]
                target_logprobs = self._compute_logprobs_from_target_tokens(logits, target_tokens)
                loss_fn_inputs["target_logprobs"] = target_logprobs

                loss, metric = loss_fn_callable(loss_fn_inputs, loss_fn_config)

                # Backward pass if needed
                if backward:
                    loss.backward()

            unpaded_logprobs = self._unpad_tensor(
                target_logprobs, [len(datum.model_input.to_ints()) for datum in data]
            )
            return types.ForwardBackwardOutput(
                loss_fn_output_type=loss_fn,
                loss_fn_outputs=[
                    {"logprobs": types.TensorData.from_torch(logprobs.detach())}
                    for logprobs in unpaded_logprobs
                ],
                metrics=metric,
            )

    async def optim_step(
        self,
        adam_params: types.AdamParams,
        lora_id: str,
        trace_context: dict[str, str] | None = None,
    ) -> types.OptimStepResponse:
        """Perform an optimization step using Adam optimizer.

        Args:
            adam_params: Parameters for the Adam optimizer.
            lora_id: The LoRA adapter ID to use.
            trace_context: Optional trace context for distributed tracing.

        Returns:
            OptimStepResponse: The response containing optimization metrics.
        """
        ctx = extract_context(trace_context or {})
        with _get_tracer().start_as_current_span("hf_model.optim_step", context=ctx) as span:
            span.set_attribute("tuft.lora_id", lora_id)
            optimizer = self.adapter_optimizer[lora_id]
            for param_group in optimizer.param_groups:
                param_group["lr"] = adam_params.learning_rate
                param_group["betas"] = (adam_params.beta1, adam_params.beta2)
                param_group["eps"] = adam_params.eps
                param_group["weight_decay"] = adam_params.weight_decay
            optimizer.step()
            optimizer.zero_grad()
            return types.OptimStepResponse()

    # --------------------------------
    # Helper methods
    # --------------------------------
    def _prepare_loss_fn_inputs(self, data: list[types.Datum]) -> Dict[str, torch.Tensor]:
        """Prepare input tensors from Datum list."""
        device = next(self.model.parameters()).device

        loss_fn_input_dict = {}
        # prepare loss_fn_inputs tensors
        loss_fn_input_keys = data[0].loss_fn_inputs.keys()
        for key in loss_fn_input_keys:
            tensors = [datum.loss_fn_inputs[key].to_torch() for datum in data]
            # If tensor is 1D, pad to max length; if already same shape, stack directly
            if all(t.dim() == 1 for t in tensors):
                padded = pad_sequence(tensors, batch_first=True, padding_value=0)
                loss_fn_input_dict[key] = padded.to(device)
            else:
                # Try to stack, if shape mismatch, pad last dim
                try:
                    stacked = torch.stack(tensors)
                    loss_fn_input_dict[key] = stacked.to(device)
                except Exception:
                    # Pad last dim to max length
                    max_shape = list(tensors[0].shape)
                    for t in tensors:
                        for i, s in enumerate(t.shape):
                            if s > max_shape[i]:
                                max_shape[i] = s
                    padded_tensors = []
                    for t in tensors:
                        pad_width = [(0, m - s) for s, m in zip(t.shape, max_shape, strict=False)]
                        pad_args = []
                        for p in reversed(pad_width):
                            pad_args.extend(p)
                        padded = torch.nn.functional.pad(t, pad_args, value=0)
                        padded_tensors.append(padded)
                    stacked = torch.stack(padded_tensors)
                    loss_fn_input_dict[key] = stacked.to(device)

        return loss_fn_input_dict

    def _compute_logprobs_from_target_tokens(
        self, logits: torch.Tensor, target_tokens: torch.Tensor
    ) -> torch.Tensor:
        logits_labels = torch.gather(logits, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(logit, dim=-1) for logit in logits])
        logprobs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
        return logprobs_labels

    def _unpad_tensor(
        self, padded_tensor: torch.Tensor, original_lengths: list[int]
    ) -> list[torch.Tensor]:
        """Unpad a padded tensor back to list of tensors with original lengths."""
        tensors = []
        for i, length in enumerate(original_lengths):
            tensors.append(padded_tensor[i, :length])
        return tensors

    def _init_peft_model(self, config: ModelConfig):
        model = AutoModelForCausalLM.from_pretrained(
            str(config.model_path),
            dtype="auto",
            device_map="auto",
        )
        peft_config = LoraConfig()
        peft_model = get_peft_model(model, peft_config=peft_config, adapter_name="default")
        return peft_model

    def _activate_adapter(self, lora_id: str):
        if lora_id not in self.adapter_optimizer:
            raise ValueError(f"Adapter {lora_id} not found.")
        self.model.set_adapter(lora_id)

    @classmethod
    def get_actor(cls, config: ModelConfig) -> "ActorProxy":
        return (
            ray.remote(cls)
            .options(
                name="training_model_" + config.model_name,
                num_gpus=1 if not config.colocate else 1 - config.sampling_memory_fraction,
            )
            .remote(config)
        )
