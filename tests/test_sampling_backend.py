from __future__ import annotations

import os
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from tinker import types


@pytest.fixture(scope="function", autouse=True)
def ray_cluster():
    import ray

    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_sampling_backend():
    from tuft.backends.sampling_backend import VLLMSamplingBackend
    from tuft.config import ModelConfig

    assert (
        "TUFT_TEST_MODEL" in os.environ
    ), "Environment variable TUFT_TEST_MODEL must be set for this test."

    model_path = Path(os.environ.get("TUFT_TEST_MODEL", "Qwen/Qwen3-0.6B"))
    model_config = ModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        model_path=model_path,
        max_model_len=2048,
        tensor_parallel_size=1,
    )
    backend = VLLMSamplingBackend(model_config)
    await backend.async_init()
    assert backend.base_model == "Qwen/Qwen3-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_special_tokens=False
    )["input_ids"][0].tolist()
    # generate without prompt logprobs
    response = await backend.sample(
        prompt=types.ModelInput.from_ints(input_ids),
        num_samples=3,
        sampling_params=types.SamplingParams(max_tokens=256, temperature=0.7),
    )
    assert response.sequences is not None
    assert len(response.sequences) == 3
    assert response.prompt_logprobs is None
    assert response.topk_prompt_logprobs is None
    for seq in response.sequences:
        assert seq.tokens is not None
        assert seq.logprobs is not None
        assert len(seq.tokens) > 0
        assert len(seq.logprobs) == len(seq.tokens)
        assert seq.stop_reason == "stop"

    # generate with prompt logprobs
    response_with_logprobs = await backend.sample(
        prompt=types.ModelInput.from_ints(input_ids),
        num_samples=2,
        sampling_params=types.SamplingParams(max_tokens=3, temperature=0.7),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=3,
    )
    assert response_with_logprobs.sequences is not None
    assert len(response_with_logprobs.sequences) == 2
    assert response_with_logprobs.prompt_logprobs is not None
    assert response_with_logprobs.topk_prompt_logprobs is not None
    assert len(response_with_logprobs.prompt_logprobs) == len(input_ids)
    assert len(response_with_logprobs.topk_prompt_logprobs) == len(input_ids)
    # first token should have no top-k logprobs
    assert response_with_logprobs.topk_prompt_logprobs[0] is None
    # each subsequent token should have top-k logprobs
    for topk in response_with_logprobs.topk_prompt_logprobs[1:]:
        assert topk is not None
        assert len(topk) == 3
    # check sequences
    for seq in response_with_logprobs.sequences:
        assert seq.tokens is not None
        assert seq.logprobs is not None
        assert len(seq.tokens) > 0
        assert len(seq.logprobs) == len(seq.tokens)
        assert seq.stop_reason == "length"  # stop because of max_tokens=3

    # compute logprobs only
    logprobs_response = await backend.sample(
        prompt=types.ModelInput.from_ints(input_ids),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
    )
    assert len(logprobs_response.sequences) == 1
    assert logprobs_response.prompt_logprobs is not None
    assert len(logprobs_response.prompt_logprobs) == len(input_ids)
    assert logprobs_response.topk_prompt_logprobs is None
