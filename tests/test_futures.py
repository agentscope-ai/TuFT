from __future__ import annotations

import asyncio
import time

import pytest
from tinker.types.try_again_response import TryAgainResponse

from tinker import types
from tuft.exceptions import UnknownModelException, UserMismatchException
from tuft.futures import FutureStore


async def _wait_for_result(store: FutureStore, request_id: str, user_id: str, timeout: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = await store.retrieve(request_id, user_id=user_id, timeout=timeout)
        if not isinstance(result, TryAgainResponse):
            return result
        await asyncio.sleep(0.01)
    raise AssertionError("future did not complete in time")


@pytest.mark.asyncio
async def test_future_store_mismatches_user():
    store = FutureStore()

    def _operation() -> types.SaveWeightsResponse:
        return types.SaveWeightsResponse(path="tinker://run/weights/ckpt")

    future = await store.enqueue(_operation, model_id="run", user_id="tester")
    with pytest.raises(UserMismatchException) as exc_info:
        await store.retrieve(future.request_id, user_id="wrong_user", timeout=1.0)
    assert "You do not have permission" in str(exc_info.value)
    await store.shutdown()


@pytest.mark.asyncio
async def test_future_store_returns_try_again_until_ready():
    store = FutureStore()

    def _operation() -> types.SaveWeightsResponse:
        time.sleep(1)
        return types.SaveWeightsResponse(path="tinker://run/weights/ckpt")

    future = await store.enqueue(_operation, model_id="run", user_id="tester")
    first_response = await store.retrieve(future.request_id, user_id="tester", timeout=0.1)
    assert isinstance(first_response, TryAgainResponse)

    final = await _wait_for_result(store, future.request_id, user_id="tester")
    assert isinstance(final, types.SaveWeightsResponse)
    assert final.path.endswith("ckpt")
    await store.shutdown()


@pytest.mark.asyncio
async def test_future_store_records_failures_as_request_failed():
    store = FutureStore()

    def _operation() -> types.SaveWeightsResponse:
        raise UnknownModelException("unknown")

    future = await store.enqueue(_operation, user_id="tester")
    result = await _wait_for_result(store, future.request_id, user_id="tester")
    assert isinstance(result, types.RequestFailedResponse)
    assert result.error == "Unknown model: unknown"
    assert result.category == types.RequestErrorCategory.User
    await store.shutdown()


@pytest.mark.asyncio
async def test_future_store_handles_unexpected_errors():
    store = FutureStore()

    def _operation() -> types.SaveWeightsResponse:  # pragma: no cover - executed in thread
        raise RuntimeError("boom")

    future = await store.enqueue(_operation, user_id="tester")
    result = await _wait_for_result(store, future.request_id, user_id="tester")
    assert isinstance(result, types.RequestFailedResponse)
    assert result.category == types.RequestErrorCategory.Server
    assert "boom" in result.error
    await store.shutdown()
