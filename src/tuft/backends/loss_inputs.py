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


def client_loss_fn_input_keys(data: list[types.Datum]) -> list[str]:
    """Union of the ``loss_fn_inputs`` keys across ``data``, in first-seen order.

    Taking the union rather than the first datum's keys makes the key set a
    property of the whole request: slicing ``data`` into micro-batches cannot
    change which fields reach the loss function.
    """

    return list(dict.fromkeys(key for datum in data for key in (datum.loss_fn_inputs or {})))


def batch_loss_fn_input(
    data: list[types.Datum],
    key: str,
    *,
    device: torch.device | str,
    reference_data: list[types.Datum] | None = None,
) -> torch.Tensor:
    """Stack or pad one client-supplied loss input across a batch.

    Rows that omit ``key`` are zero-filled, the same way rows shorter than the
    batch maximum are zero-padded, so a field supplied by only part of a batch
    does not abort training. ``reference_data`` (defaulting to ``data``) supplies
    the rank and dtype when no row of ``data`` carries the field at all, which
    happens when the caller is batching one micro-batch of a larger request.

    Rank and dtype are validated up front so a mismatch reports the offending
    key instead of surfacing as a bare ``torch.stack`` error. Padding runs on CPU
    so the batched result costs a single host-to-device transfer rather than one
    per row.
    """

    tensors: list[torch.Tensor | None] = [
        None if (value := (datum.loss_fn_inputs or {}).get(key)) is None else value.to_torch()
        for datum in data
    ]
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        present = [
            value.to_torch()
            for datum in reference_data or data
            if (value := (datum.loss_fn_inputs or {}).get(key)) is not None
        ]
    if not present:
        raise ValueError(f"loss_fn_inputs field {key!r} is missing from every datum")

    ndim = present[0].dim()
    if any(tensor.dim() != ndim for tensor in present):
        raise ValueError(f"loss_fn_inputs field {key!r} must have the same rank for every datum")
    dtype = present[0].dtype
    if any(tensor.dtype != dtype for tensor in present):
        raise ValueError(f"loss_fn_inputs field {key!r} must have the same dtype for every datum")

    max_shape = [max(tensor.size(dim) for tensor in present) for dim in range(ndim)]
    filler = torch.zeros(max_shape, dtype=dtype)
    rows = [filler if tensor is None else tensor for tensor in tensors]

    if ndim == 0:
        return torch.stack(rows).to(device)
    if ndim == 1:
        return pad_sequence(rows, batch_first=True, padding_value=0).to(device)

    padded = []
    for tensor in rows:
        pad: list[int] = []
        for size, maximum in reversed(list(zip(tensor.shape, max_shape, strict=True))):
            pad.extend((0, maximum - size))
        padded.append(torch.nn.functional.pad(tensor, pad, value=0))
    return torch.stack(padded).to(device)
