"""Torch-native forward/backward utilities for the FSDP training backend.

This module intentionally owns only the small model-execution surface TuFT needs:
model construction, contiguous micro-batching, next-token log-prob extraction, and
loss/backward execution. FSDP wrapping, adapter management, optimizers, checkpoints,
and distributed orchestration remain in :mod:`fsdp_training_backend`.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from tinker import types
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelForCausalLM

from tuft.loss_fn import get_loss_fn


_RLHF_LOSS_FNS = {"ppo", "grpo", "cispo", "importance_sampling", "dro"}


@dataclass
class FSDPModelConfig:
    """Minimal serializable model configuration used by FSDP workers."""

    path: str
    max_model_len: int
    attn_implementation: str | None = None
    override_config: dict[str, Any] = field(default_factory=dict)
    trust_remote_code: bool = True


@dataclass
class MicroBatch:
    """Padded model inputs/labels plus the true per-sample sequence lengths."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    lengths: list[int]


def _explicit_target_tokens(datum: types.Datum, device: torch.device | str) -> torch.Tensor | None:
    value = (datum.loss_fn_inputs or {}).get("target_tokens")
    if value is None:
        return None
    return value.to_torch().to(device=device, dtype=torch.long).reshape(-1)


def build_base_model(config: FSDPModelConfig) -> Any:
    """Load a single CPU copy of the base model and enable checkpointed training."""

    override = dict(config.override_config)
    attn_implementation = override.pop("attn_implementation", config.attn_implementation)
    hf_config = AutoConfig.from_pretrained(
        config.path,
        trust_remote_code=config.trust_remote_code,
    )
    for key, value in override.items():
        setattr(hf_config, key, value)

    model_kwargs: dict[str, Any] = {
        "config": hf_config,
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": config.trust_remote_code,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(config.path, **model_kwargs)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable({"use_reentrant": False})
    return model


def _prepare_micro_batch(data: list[types.Datum], device: torch.device | str) -> MicroBatch:
    """Pad model inputs and labels, honoring explicit Tinker target tokens."""

    if not data:
        raise ValueError("A micro-batch must contain at least one datum")

    sequences = [
        torch.tensor(datum.model_input.to_ints(), dtype=torch.long, device=device) for datum in data
    ]
    lengths = [int(sequence.numel()) for sequence in sequences]
    if any(length == 0 for length in lengths):
        raise ValueError("FSDP forward does not support empty token sequences")

    input_ids = pad_sequence(sequences, batch_first=True, padding_value=0)
    max_len = input_ids.size(1)
    token_positions = torch.arange(max_len, dtype=torch.long, device=device)
    length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
    attention_mask = (token_positions.unsqueeze(0) < length_tensor.unsqueeze(1)).long()
    position_ids = token_positions.unsqueeze(0).expand(len(data), -1)

    # Standard TuFT/Tinker training data carries already-shifted target_tokens
    # in loss_fn_inputs. Use them when present so the final supervised token of
    # each sample is preserved; fall back to the prior flat-roll convention only
    # for legacy/RLHF-style rows that omit explicit targets.
    flat_labels = torch.roll(torch.cat(sequences), shifts=-1, dims=0)
    labels = []
    offset = 0
    for datum, length in zip(data, lengths, strict=True):
        explicit_targets = _explicit_target_tokens(datum, device)
        if explicit_targets is not None:
            if int(explicit_targets.numel()) != length:
                raise ValueError(
                    "target_tokens length must match model_input length: "
                    f"got {int(explicit_targets.numel())} vs {length}"
                )
            labels.append(explicit_targets)
        else:
            labels.append(flat_labels[offset : offset + length])
        offset += length
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    return MicroBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels_padded,
        lengths=lengths,
    )


def _compute_target_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Gather label log-probabilities without materializing a full fp32 log-softmax."""

    if logits.dtype in (torch.float32, torch.float64):
        label_logits = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        logsumexp = torch.stack([torch.logsumexp(row, dim=-1) for row in logits])
        return label_logits - logsumexp

    rows = []
    for row_logits, row_labels in zip(logits, labels, strict=True):
        row_logprobs = torch.nn.functional.log_softmax(row_logits, dim=-1)
        rows.append(row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1))
    return torch.stack(rows)


def _datum_field(
    datum: types.Datum,
    key: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    value = (datum.loss_fn_inputs or {}).get(key)
    if value is None:
        return None
    return value.to_torch().to(device=device, dtype=dtype).reshape(-1)


def _copy_row(destination: torch.Tensor, row: int, value: torch.Tensor) -> None:
    width = min(destination.size(1), value.numel())
    if width:
        destination[row, :width] = value[:width]


def _prepare_loss_fn_inputs(
    data: list[types.Datum],
    target_logprobs: torch.Tensor,
    loss_fn_name: str,
) -> dict[str, torch.Tensor]:
    """Build padded TuFT loss inputs directly from Datum objects."""

    batch_size, max_len = target_logprobs.shape
    device = target_logprobs.device
    if loss_fn_name.lower() in _RLHF_LOSS_FNS:
        sampling_logprobs = target_logprobs.clone()
        advantages = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
        for row, datum in enumerate(data):
            old_logprobs = _datum_field(
                datum,
                "logprobs",
                device=device,
                dtype=torch.float32,
            )
            if old_logprobs is not None:
                _copy_row(sampling_logprobs, row, old_logprobs)
            advantage = _datum_field(
                datum,
                "advantages",
                device=device,
                dtype=torch.float32,
            )
            if advantage is not None:
                _copy_row(advantages, row, advantage)
        return {
            "target_logprobs": target_logprobs,
            "logprobs": sampling_logprobs,
            "advantages": advantages,
        }

    weights = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
    for row, datum in enumerate(data):
        value = _datum_field(datum, "weights", device=device, dtype=torch.float32)
        if value is None:
            value = torch.ones(len(datum.model_input.to_ints()), dtype=torch.float32, device=device)
        _copy_row(weights, row, value)
    return {"target_logprobs": target_logprobs, "weights": weights}


def _merge_micro_metrics(metric_list: list[dict[str, Any]]) -> dict[str, float]:
    """Combine numeric micro-batch metrics using their declared reduction."""

    grouped: dict[str, list[float]] = {}
    for metrics in metric_list:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                grouped.setdefault(key, []).append(float(value))

    merged: dict[str, float] = {}
    for key, values in grouped.items():
        if key.endswith(":mean"):
            merged[key] = sum(values) / len(values)
        else:
            merged[key] = sum(values)
    return merged


def forward_backward(
    module: Any,
    data: list[types.Datum],
    loss_fn_name: str,
    loss_fn_config: dict[str, float] | None,
    micro_batch_size: int,
    *,
    forward_only: bool = False,
) -> dict[str, Any]:
    """Run contiguous micro-batches while preserving summed gradient accumulation."""

    if not data:
        return {"model_output": {"log_probs": []}, "metrics": {}}
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")

    device = next(module.parameters()).device
    loss_callable = get_loss_fn(loss_fn_name)
    config = loss_fn_config or {}
    per_sample_logprobs: list[torch.Tensor] = []
    metric_list: list[dict[str, Any]] = []

    grad_context = torch.no_grad() if forward_only else nullcontext()
    with grad_context:
        for start in range(0, len(data), micro_batch_size):
            micro_data = data[start : start + micro_batch_size]
            batch = _prepare_micro_batch(micro_data, device)
            autocast = torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            )
            with autocast:
                outputs = module(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    position_ids=batch.position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
                target_logprobs = _compute_target_logprobs(logits, batch.labels)

            loss_inputs = _prepare_loss_fn_inputs(micro_data, target_logprobs, loss_fn_name)
            loss, metrics = loss_callable(loss_inputs, config)
            if not forward_only:
                loss.backward()
            metric_list.append(metrics)
            per_sample_logprobs.extend(
                target_logprobs[row, :length].detach() for row, length in enumerate(batch.lengths)
            )

    return {
        "model_output": {"log_probs": per_sample_logprobs},
        "metrics": _merge_micro_metrics(metric_list),
    }
