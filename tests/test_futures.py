from __future__ import annotations

import asyncio
import time

import pytest
from tinker import types
from tinker.types.try_again_response import TryAgainResponse

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


@pytest.mark.asyncio
async def test_mark_pending_sample_futures_failed():
    """Test that mark_pending_sample_futures_failed only marks sample futures as failed."""
    store = FutureStore()

    async def _sample_operation() -> types.SampleResponse:
        await asyncio.sleep(10)  # Long enough to stay pending
        return types.SampleResponse(sequences=[])

    async def _training_operation() -> types.SaveWeightsResponse:
        await asyncio.sleep(10)
        return types.SaveWeightsResponse(path="tinker://run/weights/ckpt")

    sample_future = await store.enqueue(
        _sample_operation, user_id="tester", operation_type="sample"
    )
    training_future = await store.enqueue(
        _training_operation, user_id="tester", model_id="model-1", operation_type="save_weights"
    )

    sample_result = await store.retrieve(sample_future.request_id, user_id="tester", timeout=0.1)
    training_result = await store.retrieve(
        training_future.request_id, user_id="tester", timeout=0.1
    )
    assert isinstance(sample_result, TryAgainResponse)
    assert isinstance(training_result, TryAgainResponse)

    # Mark sample futures as failed
    count = store.mark_pending_sample_futures_failed()
    assert count == 1  # Only the sample future should be marked

    # Sample future should now be failed
    sample_result = await store.retrieve(sample_future.request_id, user_id="tester", timeout=0.1)
    assert isinstance(sample_result, types.RequestFailedResponse)
    assert sample_result.category == types.RequestErrorCategory.Server

    training_result = await store.retrieve(
        training_future.request_id, user_id="tester", timeout=0.1
    )
    # It should NOT be a RequestFailedResponse from our mark call
    assert not isinstance(training_result, types.RequestFailedResponse)
    assert isinstance(training_result, TryAgainResponse)

    await store.shutdown()
