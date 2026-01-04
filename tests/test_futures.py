from __future__ import annotations

import time

from fastapi import HTTPException

from llm_rpc.futures import FutureStore
from tinker import types
from tinker.types.try_again_response import TryAgainResponse


def _wait_for_result(store: FutureStore, request_id: str, timeout: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = store.retrieve(request_id)
        if not isinstance(result, TryAgainResponse):
            return result
        time.sleep(0.01)
    raise AssertionError("future did not complete in time")


def test_future_store_returns_try_again_until_ready():
    store = FutureStore(max_workers=1)

    def _operation() -> types.SaveWeightsResponse:
        time.sleep(0.05)
        return types.SaveWeightsResponse(path="tinker://run/weights/ckpt")

    future = store.enqueue(_operation, model_id="run")
    first_response = store.retrieve(future.request_id)
    assert isinstance(first_response, TryAgainResponse)

    final = _wait_for_result(store, future.request_id)
    assert isinstance(final, types.SaveWeightsResponse)
    assert final.path.endswith("ckpt")
    store.shutdown()


def test_future_store_records_failures_as_request_failed():
    store = FutureStore(max_workers=1)

    def _operation() -> types.SaveWeightsResponse:
        raise HTTPException(status_code=400, detail="bad request")

    future = store.enqueue(_operation)
    result = _wait_for_result(store, future.request_id)
    assert isinstance(result, types.RequestFailedResponse)
    assert result.error == "bad request"
    assert result.category == types.RequestErrorCategory.User
    store.shutdown()


def test_future_store_handles_unexpected_errors():
    store = FutureStore(max_workers=1)

    def _operation() -> types.SaveWeightsResponse:  # pragma: no cover - executed in thread
        raise RuntimeError("boom")

    future = store.enqueue(_operation)
    result = _wait_for_result(store, future.request_id)
    assert isinstance(result, types.RequestFailedResponse)
    assert result.category == types.RequestErrorCategory.Server
    assert "boom" in result.error
    store.shutdown()
