import os
from pathlib import Path
from typing import List

import numpy as np
import pytest
import transformers

from llm_rpc.backends.training_backend import HFTrainingBackend
from llm_rpc.config import ModelConfig
from tinker import types


def _construct_data() -> List[types.Datum]:
    assert (
        "LLM_RPC_TEST_MODEL" in os.environ
    ), "Environment variable LLM_RPC_TEST_MODEL must be set for this test."

    model_path = Path(os.environ.get("LLM_RPC_TEST_MODEL", "Qwen/Qwen3-0.6B"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    examples = [
        {"input": "banana split", "output": "anana-bay plit-say"},
        {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
        {"input": "donut shop", "output": "onut-day op-shay"},
        {"input": "pickle jar", "output": "ickle-pay ar-jay"},
        {"input": "space exploration", "output": "ace-spay exploration-way"},
        {"input": "rubber duck", "output": "ubber-ray uck-day"},
        {"input": "coding wizard", "output": "oding-cay izard-way"},
    ]
    data = []
    for example in examples:
        prompt = f"English: {example['input']}\nPig Latin:"

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_weights = [0] * len(prompt_tokens)
        # Add a space before the output string, and finish with double newline
        completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
        completion_weights = [1] * len(completion_tokens)

        tokens = prompt_tokens + completion_tokens
        weights = prompt_weights + completion_weights

        input_tokens = tokens[:-1]
        target_tokens = tokens[
            1:
        ]  # We're predicting the next token, so targets need to be shifted.
        weights = weights[1:]
        data.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
            )
        )
    return data


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_training_backend():
    assert (
        "LLM_RPC_TEST_MODEL" in os.environ
    ), "Environment variable LLM_RPC_TEST_MODEL must be set for this test."

    model_path = Path(os.environ.get("LLM_RPC_TEST_MODEL", "Qwen/Qwen3-0.6B"))
    model_config = ModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        model_path=model_path,
        max_model_len=2048,
        tensor_parallel_size=1,
    )
    backend = HFTrainingBackend(model_config)
    await backend.async_init()
    assert backend.model is not None

    await backend.create_lora_adapter("test_lora_1", types.LoraConfig(rank=8, seed=42))
    await backend.create_lora_adapter("test_lora_2", types.LoraConfig(rank=8, seed=42))

    data = _construct_data()
    loss_per_tokens_1 = []
    loss_per_tokens_2 = []
    for step in range(6):
        outputs_1 = await backend.forward(
            data=data,
            lora_id="test_lora_1",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )
        outputs_2 = await backend.forward(
            data=data,
            lora_id="test_lora_2",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )
        await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="test_lora_1")
        await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="test_lora_2")
        logprobs_1 = np.concatenate(
            [output["logprobs"].tolist() for output in outputs_1.loss_fn_outputs]
        )
        logprobs_2 = np.concatenate(
            [output["logprobs"].tolist() for output in outputs_2.loss_fn_outputs]
        )
        weights = np.concatenate([example.loss_fn_inputs["weights"].tolist() for example in data])
        loss_per_token_1 = -np.dot(logprobs_1, weights) / weights.sum()
        loss_per_token_2 = -np.dot(logprobs_2, weights) / weights.sum()
        loss_per_tokens_1.append(loss_per_token_1)
        loss_per_tokens_2.append(loss_per_token_2)
        print(f"(1) Loss per token at step {step}: {loss_per_token_1:.4f}")
        print(f"(2) Loss per token at step {step}: {loss_per_token_2:.4f}")
    # Verify that the loss is decreasing
    for i in range(1, len(loss_per_tokens_1)):
        assert (
            loss_per_tokens_1[i] < loss_per_tokens_1[i - 1]
        ), "Loss did not decrease for lora_id test_lora_1"
        assert (
            loss_per_tokens_2[i] < loss_per_tokens_2[i - 1]
        ), "Loss did not decrease for lora_id test_lora_2"
        assert (
            abs(loss_per_tokens_1[i] - loss_per_tokens_2[i]) < 0.1
        ), "Losses for both LoRAs diverged unexpectedly"


# From Tinker official test output:
# Loss per token at step 0: 4.2463
# Loss per token at step 1: 3.8110
# Loss per token at step 2: 2.6812
# Loss per token at step 3: 1.8330
# Loss per token at step 4: 1.0662
# Loss per token at step 5: 0.5573
