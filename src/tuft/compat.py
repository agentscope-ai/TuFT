"""Compatibility helpers for the Tinker JSON and protobuf wire formats."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from tinker.proto import tinker_public_pb2 as public_pb
from tinker.types._pydantic_types.forward_backward_input import (
    ForwardBackwardInput as PydanticForwardBackwardInput,
)
from tinker.types.forward_backward_output import ForwardBackwardOutput
from tinker.types.model_id import ModelID
from tinker.types.sample_response import SampleResponse


class ForwardRequest(BaseModel):
    """TuFT-owned JSON model retained after Tinker 0.25 removed its mirror."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    forward_input: PydanticForwardBackwardInput
    model_id: ModelID
    seq_id: int | None = None


class ForwardBackwardRequest(BaseModel):
    """TuFT-owned JSON model retained after Tinker 0.25 removed its mirror."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    forward_backward_input: PydanticForwardBackwardInput
    model_id: ModelID
    seq_id: int | None = None


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

_PROTO_DTYPE_TO_NUMPY: dict[int, np.dtype[Any]] = {
    public_pb.DTYPE_FLOAT32: np.dtype(np.float32),
    public_pb.DTYPE_INT64: np.dtype(np.int64),
    public_pb.DTYPE_INT32: np.dtype(np.int32),
    public_pb.DTYPE_BFLOAT16: np.dtype(np.uint16),
}

_PROTO_DTYPE_TO_TENSOR_DTYPE: dict[int, str] = {
    public_pb.DTYPE_FLOAT32: "float32",
    public_pb.DTYPE_BFLOAT16: "float32",
    public_pb.DTYPE_INT64: "int64",
    public_pb.DTYPE_INT32: "int64",
}

_TENSOR_DTYPE_TO_PROTO: dict[str, int] = {
    "float32": public_pb.DTYPE_FLOAT32,
    "int64": public_pb.DTYPE_INT64,
}

_TENSOR_DTYPE_TO_NUMPY: dict[str, np.dtype[Any]] = {
    "float32": np.dtype(np.float32),
    "int64": np.dtype(np.int64),
}


def _decode_proto_array(data: bytes, dtype: int) -> np.ndarray[Any, Any]:
    np_dtype = _PROTO_DTYPE_TO_NUMPY.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported protobuf tensor dtype: {dtype}")

    array = np.frombuffer(data, dtype=np_dtype)
    if dtype == public_pb.DTYPE_BFLOAT16:
        return (array.astype(np.uint32) << 16).view(np.float32)
    if dtype == public_pb.DTYPE_INT32:
        return array.astype(np.int64)
    return array.copy()


def _deserialize_tensor_proto(tensor: public_pb.Tensor) -> dict[str, Any]:
    tensor_dtype = _PROTO_DTYPE_TO_TENSOR_DTYPE.get(tensor.dtype)
    if tensor_dtype is None:
        raise ValueError(f"Unsupported protobuf tensor dtype: {tensor.dtype}")

    encoding = tensor.WhichOneof("encoding")
    if encoding == "dense":
        values = _decode_proto_array(tensor.dense, tensor.dtype)
        sparse_crow_indices = None
        sparse_col_indices = None
    elif encoding == "sparse_csr":
        values = _decode_proto_array(tensor.sparse_csr.values, tensor.dtype)
        sparse_crow_indices = np.frombuffer(tensor.sparse_csr.crow_indices, dtype=np.int64).tolist()
        sparse_col_indices = np.frombuffer(tensor.sparse_csr.col_indices, dtype=np.int64).tolist()
    else:
        raise ValueError("Protobuf tensor is missing a dense or sparse_csr encoding")

    return {
        "data": values.tolist(),
        "dtype": tensor_dtype,
        "shape": list(tensor.shape) or None,
        "sparse_crow_indices": sparse_crow_indices,
        "sparse_col_indices": sparse_col_indices,
    }


def _deserialize_chunk_proto(chunk: public_pb.Chunk) -> dict[str, Any]:
    chunk_type = chunk.WhichOneof("chunk")
    if chunk_type == "encoded_text":
        return {
            "type": "encoded_text",
            "tokens": np.frombuffer(chunk.encoded_text.tokens, dtype=np.int32).tolist(),
        }
    if chunk_type == "image":
        result: dict[str, Any] = {
            "type": "image",
            "data": chunk.image.data,
            "format": chunk.image.format,
        }
        if chunk.image.HasField("expected_tokens"):
            result["expected_tokens"] = chunk.image.expected_tokens
        return result
    if chunk_type == "dmel":
        return {"type": "dmel", "dmel": chunk.dmel.dmel}
    raise ValueError("Protobuf model input contains an unsupported chunk")


def deserialize_forward_backward_request_proto(
    proto_bytes: bytes,
) -> tuple[ForwardBackwardRequest, bool]:
    """Decode a Tinker 0.25 forward/backward protobuf request."""
    proto = public_pb.ForwardBackwardRequest()
    proto.ParseFromString(proto_bytes)

    data = []
    for datum in proto.data:
        data.append(
            {
                "model_input": {
                    "chunks": [_deserialize_chunk_proto(chunk) for chunk in datum.model_input]
                },
                "loss_fn_inputs": {
                    name: _deserialize_tensor_proto(tensor)
                    for name, tensor in datum.loss_fn_inputs.items()
                },
            }
        )

    forward_backward_input = PydanticForwardBackwardInput.model_validate(
        {
            "data": data,
            "loss_fn": proto.loss_fn,
            "loss_fn_config": dict(proto.loss_fn_config) or None,
        }
    )
    request = ForwardBackwardRequest(
        forward_backward_input=forward_backward_input,
        model_id=proto.model_id,
        seq_id=proto.seq_id,
    )
    return request, proto.forward_only


def serialize_forward_backward_output_proto(response: ForwardBackwardOutput) -> bytes:
    """Serialize per-datum loss outputs as Tinker's batched protobuf tensors."""
    proto = public_pb.ForwardBackwardOutput(
        loss_fn_output_type=response.loss_fn_output_type,
        metrics=response.metrics,
    )
    if not response.loss_fn_outputs:
        return proto.SerializeToString()

    expected_fields = set(response.loss_fn_outputs[0])
    if any(set(datum) != expected_fields for datum in response.loss_fn_outputs[1:]):
        raise ValueError("All loss_fn_outputs must contain the same tensor fields")

    record = proto.loss_fn_outputs.add()
    record.type_tag = response.loss_fn_output_type
    record.num_datums = len(response.loss_fn_outputs)
    for name in sorted(expected_fields):
        tensors = [datum[name] for datum in response.loss_fn_outputs]
        dtype = tensors[0].dtype
        np_dtype = _TENSOR_DTYPE_TO_NUMPY.get(dtype)
        proto_dtype = _TENSOR_DTYPE_TO_PROTO.get(dtype)
        if np_dtype is None or proto_dtype is None:
            raise ValueError(f"Unsupported TensorData dtype for protobuf response: {dtype}")

        arrays: list[np.ndarray[Any, Any]] = []
        trailing_shape: list[int] | None = None
        offsets = [0]
        for tensor in tensors:
            if tensor.dtype != dtype:
                raise ValueError(f"Mismatched tensor dtypes for loss output field {name}")
            if tensor.sparse_crow_indices is not None:
                raise ValueError("Sparse loss outputs cannot be encoded as BatchedTensor")

            array = np.asarray(tensor.data, dtype=np_dtype)
            shape = list(tensor.shape) if tensor.shape else [array.size]
            if int(np.prod(shape, dtype=np.int64)) != array.size:
                raise ValueError(f"Invalid shape for loss output field {name}: {shape}")
            current_trailing_shape = shape[1:]
            if trailing_shape is None:
                trailing_shape = current_trailing_shape
            elif current_trailing_shape != trailing_shape:
                raise ValueError(f"Mismatched trailing shapes for loss output field {name}")

            array = np.ascontiguousarray(array.reshape(shape))
            arrays.append(array)
            offsets.append(offsets[-1] + array.nbytes)

        batched = record.fields[name]
        batched.data = b"".join(array.tobytes() for array in arrays)
        batched.offsets = np.asarray(offsets, dtype=np.int64).tobytes()
        batched.dtype = proto_dtype
        batched.trailing_shape.extend(trailing_shape or [])

    return proto.SerializeToString()


def serialize_sample_response_proto(response: SampleResponse) -> bytes:
    """Serialize a SampleResponse to protobuf wire format.

    Proto schema (from tinker_public_pb2):
    - SampledSequence: stop_reason (enum), tokens (bytes=int32[]), logprobs (bytes=float32[])
    - SampleResponse: sequences[], prompt_logprobs (bytes=float32[]), topk_prompt_logprobs
    """
    proto = public_pb.SampleResponse()
    proto.prompt_cache_hit_tokens = response.prompt_cache_hit_tokens

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
