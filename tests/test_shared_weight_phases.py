import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tuft.backends.vllm_engine import VLLMEngine, VLLMEngineConfig


def engine():
    result = VLLMEngine.__new__(VLLMEngine)
    result.config = VLLMEngineConfig(model_path="test", share_base_weights=True)
    result._phase_lock = asyncio.Lock()
    result._sleeping = False
    result._phase_failed = False
    result._phase_stats = {}
    result.async_llm = SimpleNamespace(sleep=AsyncMock(), wake_up=AsyncMock())
    return result


async def test_repeated_and_concurrent_phase_requests_coalesce():
    model = engine()
    await asyncio.gather(*(model.set_training_phase(True) for _ in range(4)))
    model.async_llm.sleep.assert_awaited_once_with(level=1)
    assert model._sleeping
    await asyncio.gather(*(model.set_training_phase(False) for _ in range(4)))
    model.async_llm.wake_up.assert_awaited_once()
    assert not model._sleeping
    assert model._phase_stats["sleep_count"] == model._phase_stats["wake_count"] == 1


async def test_uncertain_sleep_state_fails_closed():
    model = engine()
    model.async_llm.sleep.side_effect = RuntimeError("GPU lost")
    with pytest.raises(RuntimeError, match="GPU lost"):
        await model.set_training_phase(True)
    with pytest.raises(RuntimeError, match="previously failed"):
        await model.set_training_phase(False)
    model.async_llm.wake_up.assert_not_awaited()
