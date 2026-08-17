"""Batching rules for client-supplied ``Datum.loss_fn_inputs`` fields.

Both training backends forward arbitrary client tensors to the loss function, so
the key-discovery and padding rules live here rather than being re-implemented
per backend. Keeping one implementation makes HF/FSDP parity a property of the
code instead of something a cross-backend test has to keep rediscovering.
"""

from __future__ import annotations

import torch
from tinker import types
from torch.nn.utils.rnn import pad_sequence


MODEL_DERIVED_LOSS_INPUTS = frozenset({"target_logprobs"})

# FSDP constructs these fields from model outputs or well-defined per-row
# defaults. They are deliberately excluded from the generic all-rows contract.
FSDP_BACKEND_OWNED_LOSS_INPUTS = frozenset(
    {"target_tokens", "target_logprobs", "weights", "logprobs", "advantages"}
)


def client_loss_fn_input_keys(data: list[types.Datum]) -> list[str]:
    """Union of the ``loss_fn_inputs`` keys across ``data``, in first-seen order.

    Taking the union rather than the first datum's keys makes the key set a
    property of the whole request: slicing ``data`` into micro-batches cannot
    change which fields reach the loss function.
    """

    return list(dict.fromkeys(key for datum in data for key in (datum.loss_fn_inputs or {})))


def validate_client_loss_fn_inputs(
    data: list[types.Datum],
    *,
    ignored_keys: frozenset[str] = frozenset(),
    required_keys: frozenset[str] = frozenset(),
) -> list[str]:
    """Validate one request's generic client fields and return its ordered keys.

    Arbitrary fields have no schema describing how an absent row should be
    synthesized, so every non-ignored field must be present on every datum.
    Rank and dtype must also be stable across the complete request; shapes may
    vary and are padded independently inside each micro-batch.
    """

    keys = client_loss_fn_input_keys(data)
    key_set = set(keys)
    for key in required_keys:
        if key not in key_set:
            raise ValueError(f"loss_fn_inputs field {key!r} must be present for every datum")

    for key in keys:
        if key in ignored_keys:
            continue

        tensors = []
        for datum in data:
            value = (datum.loss_fn_inputs or {}).get(key)
            if value is None:
                raise ValueError(f"loss_fn_inputs field {key!r} must be present for every datum")
            tensors.append(value.to_torch())

        ndim = tensors[0].dim()
        if any(tensor.dim() != ndim for tensor in tensors):
            raise ValueError(
                f"loss_fn_inputs field {key!r} must have the same rank for every datum"
            )
        dtype = tensors[0].dtype
        if any(tensor.dtype != dtype for tensor in tensors):
            raise ValueError(
                f"loss_fn_inputs field {key!r} must have the same dtype for every datum"
            )

    return keys


def batch_loss_fn_input(
    data: list[types.Datum],
    key: str,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Stack or pad one client-supplied loss input across a batch.

    Rank and dtype are validated up front so a mismatch reports the offending
    key instead of surfacing as a bare ``torch.stack`` error. Padding runs on CPU
    so the batched result costs a single host-to-device transfer rather than one
    per row.
    """

    tensors = []
    for datum in data:
        value = (datum.loss_fn_inputs or {}).get(key)
        if value is None:
            raise ValueError(f"loss_fn_inputs field {key!r} must be present for every datum")
        tensors.append(value.to_torch())

    ndim = tensors[0].dim()
    if any(tensor.dim() != ndim for tensor in tensors):
        raise ValueError(f"loss_fn_inputs field {key!r} must have the same rank for every datum")
    dtype = tensors[0].dtype
    if any(tensor.dtype != dtype for tensor in tensors):
        raise ValueError(f"loss_fn_inputs field {key!r} must have the same dtype for every datum")

    if ndim == 0:
        return torch.stack(tensors).to(device)
    if ndim == 1:
        return pad_sequence(tensors, batch_first=True, padding_value=0).to(device)

    max_shape = [max(tensor.size(dim) for tensor in tensors) for dim in range(ndim)]
    padded = []
    for tensor in tensors:
        pad: list[int] = []
        for size, maximum in reversed(list(zip(tensor.shape, max_shape, strict=True))):
            pad.extend((0, maximum - size))
        padded.append(torch.nn.functional.pad(tensor, pad, value=0))
    return torch.stack(padded).to(device)
