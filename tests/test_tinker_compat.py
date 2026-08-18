from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import zstandard as zstd
from tinker import types
from tinker.proto import tinker_public_pb2 as public_pb
from tinker.proto.request_conv import forward_backward_request_to_proto
from tinker.proto.response_conv import (
    deserialize_forward_backward_output,
    deserialize_sample_response,
)

from tuft.auth import User
from tuft.compat import (
    decode_stored_payload,
    encode_payload_for_storage,
    serialize_forward_backward_output_proto,
)
from tuft.config import AppConfig, ModelConfig
from tuft.futures import FutureRecord
from tuft.server import create_root_app


class _ImmediateFutureStore:
    def __init__(self) -> None:
        self.payload: Any = None
        self.operation_args: dict[str, Any] | None = None
        self.operation_type: str | None = None

    async def enqueue(
        self,
        operation,
        *,
        model_id: str | None,
        operation_args: dict[str, Any] | None,
        operation_type: str | None = None,
        **_: Any,
    ) -> types.UntypedAPIFuture:
        self.operation_args = operation_args
        self.operation_type = operation_type
        self.payload = await operation()
        return types.UntypedAPIFuture(request_id="test-request", model_id=model_id)

    async def retrieve(self, **_: Any) -> Any:
        return self.payload


class _FakeState:
    def __init__(self) -> None:
        self.future_store = _ImmediateFutureStore()
        self.backward: bool | None = None
        self.data: list[Any] = []

    def get_user(self, api_key: str) -> User | None:
        return User("tester") if api_key == "test-key" else None

    async def run_forward(self, *args: Any, backward: bool, **kwargs: Any):
        self.backward = backward
        self.data = args[2]
        return types.ForwardBackwardOutput(
            loss_fn_output_type="cross_entropy",
            loss_fn_outputs=[
                {"logprobs": types.TensorData(data=[-0.1, -0.2], dtype="float32", shape=[2])},
                {"logprobs": types.TensorData(data=[-0.3], dtype="float32", shape=[1])},
            ],
            metrics={"loss:sum": 0.6},
        )


def _proto_request(*, forward_only: bool) -> public_pb.ForwardBackwardRequest:
    request = public_pb.ForwardBackwardRequest(
        model_id="model-1",
        seq_id=1,
        loss_fn="cross_entropy",
        loss_fn_config={"scale": 0.5},
        forward_only=forward_only,
    )

    first = request.data.add()
    first.model_input.add().encoded_text.tokens = np.asarray([11, 12], dtype=np.int32).tobytes()
    first_weights = first.loss_fn_inputs["weights"]
    first_weights.dtype = public_pb.DTYPE_FLOAT32
    first_weights.shape.extend([2])
    first_weights.dense = np.asarray([1.0, 0.5], dtype=np.float32).tobytes()

    second = request.data.add()
    second.model_input.add().encoded_text.tokens = np.asarray([21], dtype=np.int32).tobytes()
    second_weights = second.loss_fn_inputs["weights"]
    second_weights.dtype = public_pb.DTYPE_FLOAT32
    second_weights.shape.extend([1, 2])
    second_weights.sparse_csr.values = np.asarray([1.0], dtype=np.float32).tobytes()
    second_weights.sparse_csr.crow_indices = np.asarray([0, 1], dtype=np.int64).tobytes()
    second_weights.sparse_csr.col_indices = np.asarray([1], dtype=np.int64).tobytes()
    return request


def _create_test_app():
    config = AppConfig()
    config.supported_models = [
        ModelConfig(
            model_name="test-model",
            model_path=Path("/dummy/test-model"),
            max_model_len=128,
            tensor_parallel_size=1,
            sampling_memory_fraction=0.5,
        )
    ]
    return create_root_app(config)


@pytest.fixture
def compatibility_app():
    app = _create_test_app()
    state = _FakeState()
    app.state.server_state = state
    return app, state


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _post_proto(client: httpx.AsyncClient, body: bytes, **headers: str):
    return client.post(
        "/api/v1/forward_backward",
        content=body,
        headers={
            "X-API-Key": "test-key",
            "Content-Type": "application/x-protobuf",
            **headers,
        },
    )


@pytest.mark.asyncio
async def test_protobuf_forward_only_request_and_response(compatibility_app) -> None:
    app, state = compatibility_app
    async with _client(app) as client:
        response = await _post_proto(client, _proto_request(forward_only=True).SerializeToString())
        assert response.status_code == 202
        assert state.backward is False
        assert state.future_store.operation_args["backward"] is False
        assert state.future_store.operation_type == "forward"
        assert state.data[0].model_input.to_ints() == [11, 12]
        assert state.data[0].loss_fn_inputs["weights"].data == pytest.approx([1.0, 0.5])
        assert state.data[1].loss_fn_inputs["weights"].sparse_crow_indices == [0, 1]

        result = await client.post(
            "/api/v1/retrieve_future",
            json={"request_id": response.json()["request_id"]},
            headers={"X-API-Key": "test-key", "Accept": "application/x-protobuf"},
        )

    assert result.status_code == 200
    assert result.headers["content-type"].startswith("application/x-protobuf")
    decoded = deserialize_forward_backward_output(result.content)
    assert [output["logprobs"].data for output in decoded.loss_fn_outputs] == [
        pytest.approx([-0.1, -0.2]),
        pytest.approx([-0.3]),
    ]
    assert decoded.metrics["loss:sum"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_protobuf_forward_backward_runs_backward(compatibility_app) -> None:
    app, state = compatibility_app
    async with _client(app) as client:
        response = await _post_proto(client, _proto_request(forward_only=False).SerializeToString())

    assert response.status_code == 202
    assert state.backward is True
    assert state.future_store.operation_type == "forward_backward"


@pytest.mark.asyncio
async def test_sdk_encoded_request_round_trips(compatibility_app) -> None:
    """Decode a body produced by the SDK's own encoder, not a hand-built proto."""
    app, state = compatibility_app
    request = types.ForwardBackwardRequest(
        forward_backward_input=types.ForwardBackwardInput(
            data=[
                types.Datum(
                    model_input=types.ModelInput.from_ints([5, 6, 7]),
                    loss_fn_inputs={
                        "target_tokens": types.TensorData(
                            data=[8, 9, 10], dtype="int64", shape=[3]
                        ),
                        "weights": types.TensorData(
                            data=[1.0, 1.0, 0.0], dtype="float32", shape=[3]
                        ),
                    },
                )
            ],
            loss_fn="ppo",
            loss_fn_config={"clip": 0.2},
        ),
        model_id="model-1",
        seq_id=3,
    )
    body = forward_backward_request_to_proto(request).SerializeToString()

    async with _client(app) as client:
        response = await _post_proto(client, body)

    assert response.status_code == 202
    args = state.future_store.operation_args
    assert args["seq_id"] == 3
    assert args["loss_fn"] == "ppo"
    assert args["loss_fn_config"] == {"clip": pytest.approx(0.2)}
    assert state.data[0].model_input.to_ints() == [5, 6, 7]
    assert state.data[0].loss_fn_inputs["target_tokens"].data == [8, 9, 10]
    assert state.data[0].loss_fn_inputs["weights"].data == pytest.approx([1.0, 1.0, 0.0])


@pytest.mark.asyncio
async def test_zstd_compressed_request_body(compatibility_app) -> None:
    """The SDK zstd-compresses the body when proto_compress_fwdbwd is enabled."""
    app, state = compatibility_app
    body = zstd.ZstdCompressor().compress(_proto_request(forward_only=False).SerializeToString())

    async with _client(app) as client:
        response = await _post_proto(client, body, **{"Content-Encoding": "zstd"})

    assert response.status_code == 202
    assert state.data[0].model_input.to_ints() == [11, 12]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,headers,status_code",
    [
        (b"not-a-proto-message", {}, 422),
        (b"not-zstd", {"Content-Encoding": "zstd"}, 422),
        (b"", {"Content-Encoding": "gzip"}, 422),
        (b"{}", {"Content-Type": "application/json"}, 415),
        (b"{}", {"Content-Type": ""}, 415),
    ],
)
async def test_malformed_request_bodies_are_rejected(
    compatibility_app, body: bytes, headers: dict[str, str], status_code: int
) -> None:
    app, _ = compatibility_app
    async with _client(app) as client:
        response = await _post_proto(client, body, **headers)
    assert response.status_code == status_code


@pytest.mark.asyncio
async def test_unknown_loss_fn_is_rejected(compatibility_app) -> None:
    app, _ = compatibility_app
    request = _proto_request(forward_only=False)
    request.loss_fn = "not_a_loss_fn"
    async with _client(app) as client:
        response = await _post_proto(client, request.SerializeToString())
    assert response.status_code == 422
    assert "not_a_loss_fn" in response.json()["detail"]


@pytest.mark.asyncio
async def test_sample_response_uses_protobuf_when_requested(compatibility_app) -> None:
    app, state = compatibility_app
    state.future_store.payload = types.SampleResponse(
        sequences=[
            types.SampledSequence(
                stop_reason="stop",
                tokens_np=np.asarray([7, 8], dtype=np.int32),
                logprobs_np=np.asarray([-0.4, -0.5], dtype=np.float32),
            )
        ],
        prompt_cache_hit_tokens=3,
    )

    async with _client(app) as client:
        result = await client.post(
            "/api/v1/retrieve_future",
            json={"request_id": "sample-request"},
            headers={"X-API-Key": "test-key", "Accept": "application/x-protobuf"},
        )

    assert result.status_code == 200
    decoded = deserialize_sample_response(result.content)
    assert decoded.sequences[0].tokens == [7, 8]
    assert decoded.prompt_cache_hit_tokens == 3


def test_openapi_documents_the_protobuf_request_body() -> None:
    schema = _create_test_app().openapi()
    # /api/v1/forward is gone: 0.25 routes forward() through forward_backward
    # with forward_only set.
    assert "/api/v1/forward" not in schema["paths"]
    body = schema["paths"]["/api/v1/forward_backward"]["post"]["requestBody"]
    assert list(body["content"]) == ["application/x-protobuf"]


def _tensor(size: int, shape: list[int]) -> types.TensorData:
    return types.TensorData(data=np.arange(size, dtype=np.float32), dtype="float32", shape=shape)


def test_ragged_loss_outputs_split_into_array_records() -> None:
    """Trailing shapes may differ per datum; each run becomes its own record."""
    response = types.ForwardBackwardOutput(
        loss_fn_output_type="ppo",
        loss_fn_outputs=[
            {"x": _tensor(6, [3, 2])},
            {"x": _tensor(4, [2, 2])},
            {"x": _tensor(3, [1, 3])},
        ],
        metrics={},
    )
    proto_bytes = serialize_forward_backward_output_proto(response)

    parsed = public_pb.ForwardBackwardOutput()
    parsed.ParseFromString(proto_bytes)
    assert [record.num_datums for record in parsed.loss_fn_outputs] == [2, 1]

    decoded = deserialize_forward_backward_output(proto_bytes)
    assert [output["x"].shape for output in decoded.loss_fn_outputs] == [[3, 2], [2, 2], [1, 3]]
    assert decoded.loss_fn_outputs[2]["x"].to_numpy().tolist() == [[0.0, 1.0, 2.0]]


def test_mismatched_loss_output_fields_are_rejected() -> None:
    response = types.ForwardBackwardOutput(
        loss_fn_output_type="ppo",
        loss_fn_outputs=[
            {"x": types.TensorData(data=[1.0], dtype="float32", shape=[1])},
            {"y": types.TensorData(data=[1.0], dtype="float32", shape=[1])},
        ],
        metrics={},
    )
    with pytest.raises(ValueError, match="same tensor fields"):
        serialize_forward_backward_output_proto(response)


@pytest.mark.parametrize(
    "payload",
    [
        types.ForwardBackwardOutput(
            loss_fn_output_type="cross_entropy",
            loss_fn_outputs=[
                {"logprobs": types.TensorData(data=[-0.1, -0.2], dtype="float32", shape=[2])}
            ],
            metrics={"loss:sum": 0.3},
        ),
        types.SampleResponse(
            sequences=[
                types.SampledSequence(
                    stop_reason="length",
                    tokens_np=np.asarray([1, 2, 3], dtype=np.int32),
                    logprobs_np=np.asarray([-0.1, -0.2, -0.3], dtype=np.float32),
                )
            ],
        ),
    ],
    ids=["forward_backward_output", "sample_response"],
)
def test_persisted_payloads_survive_as_dataclasses(payload: Any) -> None:
    """A payload restored from Redis must still serialize back to protobuf."""
    restored = decode_stored_payload(encode_payload_for_storage(payload))
    assert type(restored) is type(payload)
    if isinstance(payload, types.ForwardBackwardOutput):
        assert restored.loss_fn_outputs[0]["logprobs"].data == pytest.approx([-0.1, -0.2])
        assert restored.metrics == pytest.approx(payload.metrics)
    else:
        assert restored.sequences[0].tokens == [1, 2, 3]
        assert restored.sequences[0].stop_reason == "length"


def test_non_proto_payloads_are_left_alone() -> None:
    payload = types.OptimStepResponse(metrics={"lr": 1e-4})
    assert encode_payload_for_storage(payload) is None
    assert decode_stored_payload(payload) is payload


@pytest.mark.parametrize(
    "payload",
    [
        types.ForwardBackwardOutput(
            loss_fn_output_type="cross_entropy",
            loss_fn_outputs=[
                {"logprobs": types.TensorData(data=[-0.1, -0.2], dtype="float32", shape=[2])}
            ],
            metrics={"loss:sum": 0.3},
        ),
        types.SampleResponse(
            sequences=[
                types.SampledSequence(
                    stop_reason="stop",
                    tokens_np=np.asarray([4, 5], dtype=np.int32),
                    logprobs_np=np.asarray([-0.1, -0.2], dtype=np.float32),
                )
            ],
        ),
    ],
    ids=["forward_backward_output", "sample_response"],
)
def test_future_record_survives_a_redis_round_trip(payload: Any) -> None:
    """A future restored after a restart must still hold the dataclass payload."""
    record = FutureRecord(request_id="r", status="ready", payload=payload)
    restored = FutureRecord.model_validate_json(record.model_dump_json())
    assert type(restored.payload) is type(payload)
