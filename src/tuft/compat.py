"""Compatibility helpers for tinker 0.7 → 0.18.2 migration.

Provides serialization of tinker 0.18.2 dataclass types to the JSON wire format
expected by the tinker SDK client.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tinker.proto import tinker_public_pb2 as public_pb
from tinker.types.sample_response import SampleResponse


def serialize_sample_response(response: SampleResponse) -> dict[str, Any]:
    """Serialize a SampleResponse dataclass to the JSON wire format.

    The tinker SDK client expects the old Pydantic field names:
    - sequences[].tokens (list[int])
    - sequences[].logprobs (list[float] | None)
    - sequences[].stop_reason (str)
    - prompt_logprobs (list[float|None] | None)
    - topk_prompt_logprobs (list[list[tuple[int,float]]|None] | None)
    - type: "sample"
    """
    sequences = []
    for seq in response.sequences:
        seq_dict: dict[str, Any] = {
            "stop_reason": seq.stop_reason,
            "tokens": seq.tokens,  # uses @cached_property (lazy conversion from np)
        }
        if seq.logprobs is not None:
            seq_dict["logprobs"] = seq.logprobs
        else:
            seq_dict["logprobs"] = None
        sequences.append(seq_dict)

    result: dict[str, Any] = {
        "type": "sample",
        "sequences": sequences,
        "prompt_logprobs": response.prompt_logprobs,
        "topk_prompt_logprobs": response.topk_prompt_logprobs,
    }
    return result


def _serialize_tensor_data(td: Any) -> dict[str, Any]:
    """Serialize a TensorData dataclass to a JSON-safe dict.

    Uses ``td.data`` (returns list[int] | list[float]) instead of the
    internal ``_numpy`` field which Pydantic cannot serialize.
    """
    d: dict[str, Any] = {
        "data": td.data,
        "dtype": td.dtype,
    }
    if td.shape is not None:
        d["shape"] = td.shape
    if td.sparse_crow_indices is not None:
        d["sparse_crow_indices"] = td.sparse_crow_indices
    if td.sparse_col_indices is not None:
        d["sparse_col_indices"] = td.sparse_col_indices
    return d


def _serialize_forward_backward_output(payload: Any) -> dict[str, Any]:
    """Serialize a ForwardBackwardOutput dataclass to a JSON-safe dict."""
    loss_fn_outputs = [
        {k: _serialize_tensor_data(v) for k, v in datum.items()}
        for datum in payload.loss_fn_outputs
    ]
    return {
        "loss_fn_output_type": payload.loss_fn_output_type,
        "loss_fn_outputs": loss_fn_outputs,
        "metrics": payload.metrics,
    }


def maybe_serialize_payload(payload: Any) -> Any:
    """If payload is a SampleResponse or ForwardBackwardOutput dataclass, serialize it to dict.

    These dataclass types contain numpy arrays (TensorData._numpy) that
    Pydantic cannot serialize, so we convert them to JSON-safe dicts.
    Other Pydantic-based types (OptimStepResponse, etc.) are handled
    natively by FastAPI and need no conversion.
    """
    from tinker.types.forward_backward_output import ForwardBackwardOutput

    if isinstance(payload, SampleResponse):
        return serialize_sample_response(payload)
    if isinstance(payload, ForwardBackwardOutput):
        return _serialize_forward_backward_output(payload)
    return payload


# Proto enum mapping: SDK string -> proto enum value
_STOP_REASON_TO_PROTO: dict[str, int] = {
    "stop": public_pb.STOP_REASON_STOP,
    "length": public_pb.STOP_REASON_LENGTH,
}


def serialize_sample_response_proto(response: SampleResponse) -> bytes:
    """Serialize a SampleResponse to protobuf wire format.

    Proto schema (from tinker_public_pb2):
    - SampledSequence: stop_reason (enum), tokens (bytes=int32[]), logprobs (bytes=float32[])
    - SampleResponse: sequences[], prompt_logprobs (bytes=float32[]), topk_prompt_logprobs
    """
    proto = public_pb.SampleResponse()

    for seq in response.sequences:
        proto_seq = proto.sequences.add()
        proto_seq.stop_reason = _STOP_REASON_TO_PROTO.get(  # type: ignore[assignment]
            seq.stop_reason, public_pb.STOP_REASON_LENGTH
        )
        # Convert tokens to int32 bytes
        tokens = seq.tokens  # @cached_property, returns list[int]
        proto_seq.tokens = np.array(tokens, dtype=np.int32).tobytes()
        # Convert logprobs to float32 bytes (optional)
        logprobs = seq.logprobs
        if logprobs is not None:
            proto_seq.logprobs = np.array(logprobs, dtype=np.float32).tobytes()

    # Prompt logprobs: float32 array with NaN for None positions
    prompt_lp = response.prompt_logprobs
    if prompt_lp is not None:
        lp_array = np.array(
            [v if v is not None else float("nan") for v in prompt_lp],
            dtype=np.float32,
        )
        proto.prompt_logprobs = lp_array.tobytes()

    # Top-k prompt logprobs: dense N*K matrices
    topk_lp = response.topk_prompt_logprobs
    if topk_lp is not None:
        # Determine k from first non-None entry
        k = 0
        for entry in topk_lp:
            if entry is not None:
                k = max(k, len(entry))
                break
        if k > 0:
            n = len(topk_lp)
            token_ids = np.zeros((n, k), dtype=np.int32)
            logprobs_matrix = np.full((n, k), -99999.0, dtype=np.float32)
            for i, entry in enumerate(topk_lp):
                if entry is not None:
                    for j, (tid, lp) in enumerate(entry[:k]):
                        token_ids[i, j] = tid
                        logprobs_matrix[i, j] = lp
            topk_msg = proto.topk_prompt_logprobs
            topk_msg.token_ids = token_ids.tobytes()
            topk_msg.logprobs = logprobs_matrix.tobytes()
            topk_msg.k = k
            topk_msg.prompt_length = n

    return proto.SerializeToString()
