from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from tinker import types
from tinker.proto import tinker_public_pb2 as public_pb
from tinker.proto.response_conv import (
    deserialize_forward_backward_output,
    deserialize_sample_response,
)

from tuft.auth import User
from tuft.config import AppConfig, ModelConfig
from tuft.server import create_root_app


class _ImmediateFutureStore:
    def __init__(self) -> None:
        self.payload: Any = None
        self.operation_args: dict[str, Any] | None = None

    async def enqueue(
        self,
        operation,
        *,
        model_id: str | None,
        operation_args: dict[str, Any] | None,
        **_: Any,
    ) -> types.UntypedAPIFuture:
        self.operation_args = operation_args
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


def _json_request() -> dict[str, Any]:
    return {
        "model_id": "model-1",
        "seq_id": 1,
        "forward_backward_input": {
            "data": [
                {
                    "model_input": {"chunks": [{"type": "encoded_text", "tokens": [11, 12]}]},
                    "loss_fn_inputs": {
                        "weights": {"data": [1.0, 0.5], "dtype": "float32", "shape": [2]}
                    },
                },
                {
                    "model_input": {"chunks": [{"type": "encoded_text", "tokens": [21]}]},
                    "loss_fn_inputs": {
                        "weights": {"data": [1.0], "dtype": "float32", "shape": [1]}
                    },
                },
            ],
            "loss_fn": "cross_entropy",
            "loss_fn_config": {"scale": 0.5},
        },
    }


@pytest.fixture
def compatibility_app():
    app = _create_test_app()
    state = _FakeState()
    app.state.server_state = state
    return app, state


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


@pytest.mark.asyncio
async def test_protobuf_forward_only_request_and_response(compatibility_app) -> None:
    app, state = compatibility_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/forward_backward",
            content=_proto_request(forward_only=True).SerializeToString(),
            headers={
                "X-API-Key": "test-key",
                "Content-Type": "application/x-protobuf",
            },
        )
        assert response.status_code == 202
        assert state.backward is False
        assert state.future_store.operation_args["backward"] is False
        assert state.data[0].model_input.to_ints() == [11, 12]
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
async def test_json_forward_backward_request_and_response(compatibility_app) -> None:
    app, state = compatibility_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/forward_backward",
            json=_json_request(),
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 202
        assert state.backward is True

        result = await client.post(
            "/api/v1/retrieve_future",
            json={"request_id": response.json()["request_id"]},
            headers={"X-API-Key": "test-key", "Accept": "application/json"},
        )

    assert result.status_code == 200
    assert result.headers["content-type"].startswith("application/json")
    assert result.json()["loss_fn_outputs"][0]["logprobs"]["data"] == pytest.approx([-0.1, -0.2])


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

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        result = await client.post(
            "/api/v1/retrieve_future",
            json={"request_id": "sample-request"},
            headers={"X-API-Key": "test-key", "Accept": "application/x-protobuf"},
        )

    assert result.status_code == 200
    decoded = deserialize_sample_response(result.content)
    assert decoded.sequences[0].tokens == [7, 8]
    assert decoded.prompt_cache_hit_tokens == 3


def test_openapi_builds_with_tuft_owned_request_models() -> None:
    schema = _create_test_app().openapi()
    assert "/api/v1/forward" in schema["paths"]
    assert "/api/v1/forward_backward" in schema["paths"]
