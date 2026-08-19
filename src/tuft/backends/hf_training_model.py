import asyncio
import logging
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict

import ray
import torch
from opentelemetry.trace import StatusCode
from peft import LoraConfig, get_peft_model
from ray.actor import ActorProxy
from tinker import types
from tinker.types import LoraConfig as TinkerLoraConfig
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM

from tuft.backends.lora_modules import MODULE_MAP, get_target_modules
from tuft.backends.loss_inputs import (
    MODEL_DERIVED_LOSS_INPUTS,
    batch_loss_fn_input,
    validate_client_loss_fn_inputs,
)
from tuft.backends.vllm_lora_compat import (
    add_language_model_aliases,
    resolve_model_series,
    vllm_nests_language_model,
)
from tuft.checkpoints import CheckpointRecord
from tuft.config import DEFAULT_LORA_ALPHA_RATIO, ModelConfig, compute_lora_alpha
from tuft.loss_fn import get_loss_fn, metrics_reduction
from tuft.telemetry.tracing import extract_context, get_tracer


_get_tracer = lambda: get_tracer("tuft.hf_training_model")  # noqa: E731

OPTIMIZER_STATE_FILENAME = "optimizer.pt"


def _resolve_optimizer_state_path(checkpoint_record: CheckpointRecord) -> Path | None:
    """Locate the optimizer state file of a checkpoint, or None if it has none.

    Prefers the run-independent ``optimizer.pt``. Older checkpoints named the
    file after the run that saved them, so fall back to that; keying the
    fallback on the checkpoint's own training_run_id (rather than the adapter
    loading it) keeps legacy cross-run restores working too.
    """
    opt_dir = checkpoint_record.optimizer_path
    candidates = (
        opt_dir / OPTIMIZER_STATE_FILENAME,
        opt_dir / f"{checkpoint_record.training_run_id}.pt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _export_vllm_compatible_lora_aliases(adapter_dir: Path, model_path: str) -> None:
    """Export alias keys with a `language_model.` prefix in the saved adapter.

    peft saves LoRA keys relative to the text backbone (e.g.
    `base_model.model.model.layers.*`), but some vLLM model implementations
    (e.g. Qwen3.5's Qwen3_5ForConditionalGeneration) wrap the text backbone
    under `language_model`, and vLLM's hf_to_vllm_mapper only rewrites keys
    prefixed with `model.language_model.`. Without matching keys, vLLM
    silently ignores ALL adapter weights and serves the base model instead.

    Aliases are only emitted for architectures that actually need them (see
    ``vllm_nests_language_model``). Failures propagate: a checkpoint written
    without the aliases would silently degrade into the exact bug this export
    fixes, so ``save_state`` must fail loudly instead of returning success.
    """
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        return
    if not vllm_nests_language_model(model_path):
        return

    from safetensors.torch import load_file, save_file

    state = load_file(str(weights_path))
    aliased = add_language_model_aliases(state)
    if len(aliased) == len(state):
        return

    # Atomic in-place rewrite: write to a temp file in the same directory and
    # os.replace over the original, so a crash mid-write can never leave a
    # truncated adapter_model.safetensors behind a valid checkpoint record.
    tmp_path = adapter_dir / "adapter_model.safetensors.tmp"
    save_file(aliased, str(tmp_path))
    os.replace(tmp_path, weights_path)


def build_peft_lora_config(
    model_path: str,
    lora_config: TinkerLoraConfig,
    lora_alpha_ratio: int = DEFAULT_LORA_ALPHA_RATIO,
) -> LoraConfig:
    """Translate a Tinker ``LoraConfig`` into the peft config for one adapter.

    The Tinker config carries no alpha, so it comes from
    ``ModelConfig.lora_alpha_ratio`` via ``compute_lora_alpha`` — the same helper
    the FSDP backend's slot pool uses, so both backends scale a given rank
    identically.
    """
    return LoraConfig(
        r=lora_config.rank,
        target_modules=get_target_modules(model_path, lora_config),
        lora_alpha=compute_lora_alpha(lora_config.rank, lora_alpha_ratio),
    )


def get_default_target_modules(model_path: str) -> list[str] | None:
    """Minimal target modules for the placeholder ``default`` adapter.

    The placeholder only wraps the base model as a PeftModel so real adapters
    can be added later; it is never trained, so a single module keeps its
    permanently resident parameter overhead minimal. Returns None for unknown
    series so callers fall back to peft auto-inference instead of failing at
    server startup (the hard failure stays in create_adapter).
    """
    series = resolve_model_series(model_path)
    if series is None:
        return None
    return [MODULE_MAP[series]["attn"][0]]


class HFTrainingModel:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model = self._init_peft_model(config)
        self.adapter_optimizer: Dict[str, torch.optim.AdamW] = {}
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger()
        self.micro_batch_size = config.micro_batch_size

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
                peft_config = build_peft_lora_config(
                    str(self.config.model_path),
                    lora_config,
                    self.config.lora_alpha_ratio,
                )
                span.set_attribute("tuft.lora_alpha", peft_config.lora_alpha)

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

                # Export `language_model.`-prefixed alias keys so vLLM can
                # match them for models whose LM is nested under
                # `language_model` (e.g. Qwen3.5); no-op for other models.
                _export_vllm_compatible_lora_aliases(adapter_dir, str(self.config.model_path))

                # 2. Save optimizer state
                if optimizer:
                    opt_dir = checkpoint_record.optimizer_path
                    opt_dir.mkdir(parents=True, exist_ok=True)
                    opt_state = self.adapter_optimizer[lora_id].state_dict()
                    # Named independently of the run that saved it so a restore
                    # into a DIFFERENT run still finds it, matching the FSDP
                    # backend. Keying it on lora_id made optimizer=True a silent
                    # no-op for cross-run restores.
                    torch.save(opt_state, opt_dir / OPTIMIZER_STATE_FILENAME)
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
                    opt_path = _resolve_optimizer_state_path(checkpoint_record)
                    if opt_path is None:
                        # Silently starting from a fresh optimizer costs the next
                        # optim step a full-size, bias-uncorrected update, so say so.
                        self.logger.warning(
                            "No optimizer state found under %s; adapter %s resumes "
                            "with a fresh optimizer.",
                            checkpoint_record.optimizer_path,
                            lora_id,
                        )
                    else:
                        optimizer_obj.load_state_dict(torch.load(opt_path))
                self.adapter_optimizer[lora_id] = optimizer_obj

    async def remove_adapter(self, lora_id: str):
        async with self._lock:
            if lora_id in self.adapter_optimizer:
                self.model.delete_adapter(lora_id)
                optimizer = self.adapter_optimizer.pop(lora_id)
                del optimizer
                torch.cuda.empty_cache()

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
        """Forward pass with micro-batch gradient accumulation.

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

            batch_size = len(data)
            if batch_size == 0:
                span.set_attribute("tuft.num_micro_batches", 0)
                return types.ForwardBackwardOutput(
                    loss_fn_output_type=loss_fn,
                    loss_fn_outputs=[],
                    metrics={},
                )

            micro_batch_size = self.config.micro_batch_size
            client_keys = validate_client_loss_fn_inputs(
                data,
                ignored_keys=MODEL_DERIVED_LOSS_INPUTS,
                required_keys=frozenset({"target_tokens"}),
            )

            num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size
            span.set_attribute("tuft.num_micro_batches", num_micro_batches)

            if num_micro_batches > 1:
                self.logger.debug(
                    f"[MICRO_BATCH] Splitting batch_size={batch_size} into "
                    f"{num_micro_batches} micro-batches of size {micro_batch_size}"
                )

            loss_fn_callable = get_loss_fn(loss_fn)
            all_loss_fn_outputs = []
            micro_batch_weights = []
            metric_list = []
            total_loss = 0.0

            async with self._lock:
                self._activate_adapter(lora_id)

                for micro_idx in range(num_micro_batches):
                    start_idx = micro_idx * micro_batch_size
                    end_idx = min(start_idx + micro_batch_size, batch_size)
                    micro_data = data[start_idx:end_idx]

                    torch.cuda.reset_peak_memory_stats()
                    self.logger.debug(
                        f"[GPU-micro_batch_{micro_idx}] before_forward: "
                        f"allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB, "
                        f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB"
                    )

                    micro_loss, micro_metrics, micro_outputs = await self._forward_micro_batch(
                        micro_data,
                        loss_fn_callable,
                        loss_fn_config,
                        backward=backward,
                        client_keys=client_keys,
                    )

                    total_loss += micro_loss
                    all_loss_fn_outputs.extend(micro_outputs)
                    micro_batch_weights.append(len(micro_outputs))

                    metric_list.append(micro_metrics)

                    self.logger.debug(
                        f"[GPU-micro_batch_{micro_idx}] after_forward: "
                        f"allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB, "
                        f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB, "
                        f"max_allocated={torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
                    )

            # Let the caching allocator reuse released blocks across micro-batches.
            # Emptying the cache inside the loop forces repeated CUDA allocations
            # and synchronizations; release unused blocks once at the request boundary.
            torch.cuda.empty_cache()

            avg_loss = total_loss / num_micro_batches
            self.logger.debug(f"Average loss: {avg_loss}")
            metric_list = metrics_reduction(metric_list, micro_batch_weights)

            self.logger.debug(
                f"[GPU-after_micro_batches] allocated={torch.cuda.memory_allocated() / 1e9:.2f}GB"
                f", reserved={torch.cuda.memory_reserved() / 1e9:.2f}GB"
            )

            return types.ForwardBackwardOutput(
                loss_fn_output_type=loss_fn,
                loss_fn_outputs=all_loss_fn_outputs,
                metrics=metric_list or {},
            )

    async def _forward_micro_batch(
        self,
        data: list[types.Datum],
        loss_fn_callable: Callable,
        loss_fn_config: dict[str, float] | None,
        backward: bool,
        *,
        client_keys: list[str] | None = None,
    ) -> tuple[float, dict[str, float], list[dict]]:
        """Process a single micro-batch.

        Returns:
            tuple: (loss_value, metrics_dict, loss_fn_outputs_list)
        """
        # Prepare input tensors
        input_ids = [torch.tensor(datum.model_input.to_ints(), dtype=torch.long) for datum in data]
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
        # Length-based mask: a genuine mid-sequence token id 0 must not be masked
        # out (matches the FSDP engine's length-based mask).
        seq_lens = torch.tensor([len(datum.model_input.to_ints()) for datum in data])
        positions = torch.arange(input_ids_padded.size(1))
        attention_mask = (positions.unsqueeze(0) < seq_lens.unsqueeze(1)).long()
        position_ids = (
            torch.arange(input_ids_padded.size(1), dtype=torch.long)
            .unsqueeze(0)
            .expand(input_ids_padded.size(0), -1)
        )

        device = next(self.model.parameters()).device
        input_ids_padded = input_ids_padded.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)

        # Forward-only requests (Tinker `forward`, including the first pass of the
        # SDK's forward_backward_custom) never call backward, so skip building the
        # autograd graph. Matches the FSDP engine's forward_only path.
        grad_context = nullcontext() if backward else torch.no_grad()
        with grad_context:
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

            if "temperature" in loss_fn_config:
                temperature = loss_fn_config["temperature"]
                logits = logits / temperature

            loss_fn_inputs = self._prepare_loss_fn_inputs(data, client_keys=client_keys)
            target_tokens = loss_fn_inputs["target_tokens"]

            target_logprobs = self._compute_logprobs_from_target_tokens(logits, target_tokens)
            del logits

            loss_fn_inputs["target_logprobs"] = target_logprobs
            loss, metric = loss_fn_callable(loss_fn_inputs, loss_fn_config)

            # Backward with gradient accumulation
            if backward:
                loss.backward(retain_graph=False)

        unpaded_logprobs = self._unpad_tensor(
            target_logprobs.detach(),
            [len(datum.model_input.to_ints()) for datum in data],
        )
        loss_fn_outputs = [
            {"logprobs": types.TensorData.from_torch(logprobs.cpu().clone())}
            for logprobs in unpaded_logprobs
        ]

        loss_value = loss.detach().item()

        del target_logprobs
        del unpaded_logprobs
        del loss_fn_inputs
        del loss

        return loss_value, metric, loss_fn_outputs

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
            if adam_params.grad_clip_norm:
                clip_params = [p for group in optimizer.param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(clip_params, adam_params.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.empty_cache()
        return types.OptimStepResponse()

    # --------------------------------
    # Helper methods
    # --------------------------------
    def _prepare_loss_fn_inputs(
        self,
        data: list[types.Datum],
        *,
        client_keys: list[str] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare input tensors from Datum list."""
        device = next(self.model.parameters()).device
        if client_keys is None:
            client_keys = validate_client_loss_fn_inputs(
                data,
                ignored_keys=MODEL_DERIVED_LOSS_INPUTS,
                required_keys=frozenset({"target_tokens"}),
            )
        return {
            key: batch_loss_fn_input(data, key, device=device)
            for key in client_keys
            if key not in MODEL_DERIVED_LOSS_INPUTS
        }

    def _compute_logprobs_from_target_tokens(
        self, logits: torch.Tensor, target_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities of target tokens from logits with low memory usage.
        https://github.com/OpenRLHF/OpenRLHF/pull/718
        """
        if logits.dtype in [torch.float32, torch.float64]:
            logits_labels = torch.gather(logits, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(
                -1
            )
            logsumexp_values = torch.stack(
                [
                    torch.logsumexp(logit, dim=-1) for logit in logits
                ]  # loop to reduce peak mem consumption
            )
            log_probs_labels = (
                logits_labels - logsumexp_values
            )  # log_softmax(x_i) = x_i - logsumexp(x)
        else:
            log_probs_labels = []
            for row_logits, row_labels in zip(
                logits, target_tokens, strict=True
            ):  # loop to reduce peak mem consumption
                row_log_probs = torch.nn.functional.log_softmax(row_logits, dim=-1)
                row_log_probs_labels = row_log_probs.gather(
                    dim=-1, index=row_labels.unsqueeze(-1)
                ).squeeze(-1)
                log_probs_labels.append(row_log_probs_labels)
            log_probs_labels = torch.stack(log_probs_labels)
        return log_probs_labels

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
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable({"use_reentrant": False})
        default_modules = get_default_target_modules(str(config.model_path))
        if default_modules is not None:
            peft_config = LoraConfig(target_modules=default_modules)
        else:
            # Unknown series: fall back to peft auto-inference so the server
            # can still start; create_adapter keeps the hard failure.
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
                num_gpus=(1 if not config.colocate else 1 - config.sampling_memory_fraction),
            )
            .remote(config)
        )
