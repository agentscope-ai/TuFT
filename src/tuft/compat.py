"""Serialization helpers for the Tinker wire formats.

TuFT targets tinker >= 0.25, where ``/forward_backward`` is protobuf-only in both
directions. The response side is not merely an optimization: since tinker 0.22
``SampleResponse`` and ``ForwardBackwardOutput`` are plain dataclasses, and the
SDK's ``deserialize_json_response`` only revives pydantic models, so a JSON body
reaches the caller as a bare dict. Everything the SDK can decode as protobuf must
therefore go back as protobuf.

JSON serialization is still needed for the payload types FastAPI cannot handle on
its own (the same dataclasses hold numpy arrays that pydantic will not encode).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, cast, get_args

import numpy as np
import zstandard as zstd
from google.protobuf.message import DecodeError
from tinker import types
from tinker.proto import tinker_public_pb2 as public_pb
from tinker.proto.response_conv import (
    deserialize_forward_backward_output,
    deserialize_sample_response,
)
from tinker.types.forward_backward_output import ForwardBackwardOutput
from tinker.types.sample_response import SampleResponse


#: Payload types the SDK decodes as protobuf. Anything else falls back to JSON.
PROTO_PAYLOAD_TYPES: tuple[type, ...] = (SampleResponse, ForwardBackwardOutput)

_LOSS_FN_NAMES = frozenset(get_args(types.LossFnType))


@dataclass(frozen=True)
class ForwardBackwardRequest:
    """A decoded ``/api/v1/forward_backward`` request."""

    model_id: str
    seq_id: int
    data: list[types.Datum]
    loss_fn: types.LossFnType
    loss_fn_config: dict[str, float] | None
    forward_only: bool


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

_PROTO_DTYPE_TO_TENSOR_DTYPE: dict[int, types.TensorDtype] = {
    public_pb.DTYPE_FLOAT32: "float32",
    public_pb.DTYPE_BFLOAT16: "float32",
    public_pb.DTYPE_INT64: "int64",
    public_pb.DTYPE_INT32: "int64",
}

_TENSOR_DTYPE_TO_PROTO: dict[str, public_pb.DType.ValueType] = {
    "float32": public_pb.DTYPE_FLOAT32,
    "int64": public_pb.DTYPE_INT64,
}

_TENSOR_DTYPE_TO_NUMPY: dict[str, np.dtype[Any]] = {
    "float32": np.dtype(np.float32),
    "int64": np.dtype(np.int64),
}


def _decode_proto_array(data: bytes, dtype: int) -> np.ndarray[Any, Any]:
    """Read a packed tensor buffer, widening to the dtype the public wire declares.

    The returned view may be read-only; ``TensorData`` copies non-writable input.
    """
    np_dtype = _PROTO_DTYPE_TO_NUMPY.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported protobuf tensor dtype: {dtype}")

    array = np.frombuffer(data, dtype=np_dtype)
    if dtype == public_pb.DTYPE_BFLOAT16:
        # bfloat16 is the upper 16 bits of a float32.
        return (array.astype(np.uint32) << 16).view(np.float32)
    if dtype == public_pb.DTYPE_INT32:
        return array.astype(np.int64)
    return array


def _tensor_from_proto(tensor: public_pb.Tensor) -> types.TensorData:
    """Decode a proto Tensor, keeping the payload in numpy the whole way."""
    tensor_dtype = _PROTO_DTYPE_TO_TENSOR_DTYPE.get(tensor.dtype)
    if tensor_dtype is None:
        raise ValueError(f"Unsupported protobuf tensor dtype: {tensor.dtype}")

    shape = list(tensor.shape) or None
    encoding = tensor.WhichOneof("encoding")
    if encoding == "dense":
        return types.TensorData(
            data=_decode_proto_array(tensor.dense, tensor.dtype),
            dtype=tensor_dtype,
            shape=shape,
        )
    if encoding == "sparse_csr":
        return types.TensorData(
            data=_decode_proto_array(tensor.sparse_csr.values, tensor.dtype),
            dtype=tensor_dtype,
            shape=shape,
            sparse_crow_indices=np.frombuffer(
                tensor.sparse_csr.crow_indices, dtype=np.int64
            ).tolist(),
            sparse_col_indices=np.frombuffer(
                tensor.sparse_csr.col_indices, dtype=np.int64
            ).tolist(),
        )
    raise ValueError("Protobuf tensor is missing a dense or sparse_csr encoding")


def _chunk_from_proto(chunk: public_pb.Chunk) -> types.ModelInputChunk:
    chunk_type = chunk.WhichOneof("chunk")
    if chunk_type == "encoded_text":
        return types.EncodedTextChunk(
            tokens=np.frombuffer(chunk.encoded_text.tokens, dtype=np.int32).tolist()
        )
    if chunk_type == "image":
        return types.ImageChunk(
            data=chunk.image.data,
            format=cast(Any, chunk.image.format),
            expected_tokens=(
                chunk.image.expected_tokens if chunk.image.HasField("expected_tokens") else None
            ),
        )
    if chunk_type == "dmel":
        return types.DmelChunk(dmel=chunk.dmel.dmel)
    raise ValueError("Protobuf model input contains an unsupported chunk")


def _datum_from_proto(datum: public_pb.Datum) -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput(
            chunks=[_chunk_from_proto(chunk) for chunk in datum.model_input]
        ),
        loss_fn_inputs={
            name: _tensor_from_proto(tensor) for name, tensor in datum.loss_fn_inputs.items()
        },
    )


def decode_forward_backward_request(
    body: bytes, *, content_encoding: str = ""
) -> ForwardBackwardRequest:
    """Decode a ``/forward_backward`` protobuf request body.

    Every malformed input surfaces as ``ValueError`` -- including protobuf's
    ``DecodeError`` and pydantic's ``ValidationError``, which callers would
    otherwise have to catch separately -- so the caller maps one type onto 422.
    """
    if content_encoding == "zstd":
        # The SDK compresses the body when the server advertises
        # proto_compress_fwdbwd; nothing in the ASGI stack decompresses it.
        try:
            body = zstd.ZstdDecompressor().decompressobj().decompress(body)
        except zstd.ZstdError as exc:
            raise ValueError(f"Malformed zstd request body: {exc}") from exc
    elif content_encoding:
        raise ValueError(f"Unsupported Content-Encoding: {content_encoding}")

    proto = public_pb.ForwardBackwardRequest()
    try:
        proto.ParseFromString(body)
    except DecodeError as exc:
        raise ValueError(f"Malformed protobuf request body: {exc}") from exc
    if proto.loss_fn not in _LOSS_FN_NAMES:
        raise ValueError(f"Unsupported loss_fn: {proto.loss_fn!r}")

    return ForwardBackwardRequest(
        model_id=proto.model_id,
        seq_id=proto.seq_id,
        data=[_datum_from_proto(datum) for datum in proto.data],
        loss_fn=cast(types.LossFnType, proto.loss_fn),
        loss_fn_config=dict(proto.loss_fn_config) or None,
        forward_only=proto.forward_only,
    )


def _tensor_to_array(
    name: str, tensor: types.TensorData
) -> tuple[np.ndarray[Any, Any], public_pb.DType.ValueType]:
    """Return a contiguous array in its declared shape, plus the proto dtype."""
    np_dtype = _TENSOR_DTYPE_TO_NUMPY.get(tensor.dtype)
    proto_dtype = _TENSOR_DTYPE_TO_PROTO.get(tensor.dtype)
    if np_dtype is None or proto_dtype is None:
        raise ValueError(f"Unsupported TensorData dtype for protobuf response: {tensor.dtype}")
    if tensor.sparse_crow_indices is not None:
        raise ValueError("Sparse loss outputs cannot be encoded as BatchedTensor")

    # to_numpy() hands back the backing array; tensor.data would round-trip the
    # whole batch through Python lists.
    array = np.asarray(tensor.to_numpy(), dtype=np_dtype)
    shape = list(tensor.shape) if tensor.shape else [array.size]
    if int(np.prod(shape, dtype=np.int64)) != array.size:
        raise ValueError(f"Invalid shape for loss output field {name}: {shape}")
    return np.ascontiguousarray(array.reshape(shape)), proto_dtype


def _add_array_record(
    proto: public_pb.ForwardBackwardOutput,
    type_tag: str,
    fields: list[str],
    datums: list[dict[str, tuple[np.ndarray[Any, Any], public_pb.DType.ValueType]]],
) -> None:
    record = proto.loss_fn_outputs.add()
    record.type_tag = type_tag
    record.num_datums = len(datums)
    for name in fields:
        arrays = [datum[name][0] for datum in datums]
        batched = record.fields[name]
        batched.data = b"".join(array.tobytes() for array in arrays)
        batched.offsets = np.cumsum(
            [0, *(array.nbytes for array in arrays)], dtype=np.int64
        ).tobytes()
        batched.dtype = datums[0][name][1]
        batched.trailing_shape.extend(arrays[0].shape[1:])


def serialize_forward_backward_output_proto(response: ForwardBackwardOutput) -> bytes:
    """Serialize per-datum loss outputs as Tinker's batched protobuf tensors."""
    proto = public_pb.ForwardBackwardOutput(
        loss_fn_output_type=response.loss_fn_output_type,
        metrics=response.metrics,
    )
    if not response.loss_fn_outputs:
        return proto.SerializeToString()

    # Field names are invariant across records; the SDK indexes each field's
    # per-datum arrays by position across the whole response.
    fields = sorted(response.loss_fn_outputs[0])
    if any(sorted(datum) != fields for datum in response.loss_fn_outputs[1:]):
        raise ValueError("All loss_fn_outputs must contain the same tensor fields")

    encoded = [
        {name: _tensor_to_array(name, datum[name]) for name in fields}
        for datum in response.loss_fn_outputs
    ]

    def _layout(
        datum: dict[str, tuple[np.ndarray[Any, Any], public_pb.DType.ValueType]],
    ) -> tuple[Any, ...]:
        return tuple((datum[name][0].shape[1:], datum[name][1]) for name in fields)

    # A BatchedTensor carries a single dtype and trailing shape, so start a new
    # ArrayRecord whenever either changes. The SDK concatenates records in order.
    start = 0
    for index in range(1, len(encoded) + 1):
        if index < len(encoded) and _layout(encoded[index]) == _layout(encoded[start]):
            continue
        _add_array_record(proto, response.loss_fn_output_type, fields, encoded[start:index])
        start = index

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


def serialize_payload_proto(payload: Any) -> bytes | None:
    """Serialize a proto-capable payload, or return None for every other type."""
    if isinstance(payload, SampleResponse):
        return serialize_sample_response_proto(payload)
    if isinstance(payload, ForwardBackwardOutput):
        return serialize_forward_backward_output_proto(payload)
    return None


_STORED_PROTO_KIND = "__tuft_proto__"
_STORED_PROTO_DATA = "data"


def encode_payload_for_storage(payload: Any) -> dict[str, str] | None:
    """Encode a proto-capable payload for persistence, or None for other types.

    Persisting these as JSON would lose the dataclass: a restored payload comes
    back as a dict, which then fails the protobuf check on retrieval and reaches
    the caller as a bare dict. Storing the same protobuf we would have sent keeps
    a restarted server byte-identical to a live one.
    """
    proto_bytes = serialize_payload_proto(payload)
    if proto_bytes is None:
        return None
    kind = "sample_response" if isinstance(payload, SampleResponse) else "forward_backward_output"
    return {
        _STORED_PROTO_KIND: kind,
        _STORED_PROTO_DATA: base64.b64encode(proto_bytes).decode("ascii"),
    }


def decode_stored_payload(payload: Any) -> Any:
    """Revive a payload written by :func:`encode_payload_for_storage`."""
    if not isinstance(payload, dict) or _STORED_PROTO_KIND not in payload:
        return payload
    proto_bytes = base64.b64decode(payload[_STORED_PROTO_DATA])
    if payload[_STORED_PROTO_KIND] == "sample_response":
        return deserialize_sample_response(proto_bytes)
    return deserialize_forward_backward_output(proto_bytes)


__all__ = [
    "ForwardBackwardRequest",
    "PROTO_PAYLOAD_TYPES",
    "decode_forward_backward_request",
    "decode_stored_payload",
    "encode_payload_for_storage",
    "maybe_serialize_payload",
    "serialize_forward_backward_output_proto",
    "serialize_payload_proto",
    "serialize_sample_response",
    "serialize_sample_response_proto",
]
