import os
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
import transformers

from llm_rpc.backends.training_backend import HFTrainingBackend
from llm_rpc.checkpoints import CheckpointRecord
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
                loss_fn_inputs=dict(
                    weights=types.TensorData(data=weights, dtype="float32"),
                    target_tokens=types.TensorData(data=target_tokens, dtype="int64"),
                ),
            )
        )
    return data


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_training_backend():
    assert (
        "LLM_RPC_TEST_MODEL" in os.environ
    ), "Environment variable LLM_RPC_TEST_MODEL must be set for this test."

    model_path = Path(os.environ.get("LLM_RPC_TEST_MODEL", "Qwen/Qwen3-8B"))
    model_config = ModelConfig(
        model_name="Qwen/Qwen3-8B",
        model_path=model_path,
        max_model_len=2048,
        tensor_parallel_size=1,
    )
    backend = HFTrainingBackend(model_config)
    await backend.async_init()
    assert backend.model is not None

    await backend.create_adapter("test_lora_1", types.LoraConfig(rank=8, seed=42))
    await backend.create_adapter("test_lora_2", types.LoraConfig(rank=8, seed=42))

    data = _construct_data()
    weights = np.concatenate([example.loss_fn_inputs["weights"].tolist() for example in data])
    loss_per_tokens_1 = []
    loss_per_tokens_2 = []
    for step in range(3):
        # test two separate lora training in turn
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
            abs(loss_per_tokens_1[i] - loss_per_tokens_2[i]) < 0.2
        ), "Losses for both LoRAs diverged unexpectedly"
    # test saving and loading adapter
    # use a temp directory to save and load
    loss_per_tokens_loaded_1 = []
    loss_per_tokens_loaded_2 = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)
        checkpoint_1 = CheckpointRecord(
            checkpoint_id="test_lora_1",
            checkpoint_type="training",
            path=temp_dir / "test_lora_1",
            training_run_id="test_run_1",
            size_bytes=0,
        )
        checkpoint_2 = CheckpointRecord(
            checkpoint_id="test_lora_2",
            checkpoint_type="training",
            path=temp_dir / "test_lora_2",
            training_run_id="test_run_2",
            size_bytes=0,
        )
        await backend.save_state(
            lora_id="test_lora_1", checkpoint_record=checkpoint_1, optimizer=True
        )
        await backend.save_state(
            lora_id="test_lora_2", checkpoint_record=checkpoint_2, optimizer=False
        )
        # create a new backend and load the saved adapter
        # run forward with the loaded adapter and verify the loss is similar
        await backend.load_state(
            lora_id="test_lora_3", checkpoint_record=checkpoint_1, optimizer=True
        )
        await backend.load_state(
            lora_id="test_lora_4", checkpoint_record=checkpoint_2, optimizer=False
        )

        for step in range(3, 6):
            outputs_1 = await backend.forward(
                data=data,
                lora_id="test_lora_3",
                loss_fn="cross_entropy",
                loss_fn_config=None,
                backward=True,
            )
            outputs_2 = await backend.forward(
                data=data,
                lora_id="test_lora_4",
                loss_fn="cross_entropy",
                loss_fn_config=None,
                backward=True,
            )
            await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="test_lora_3")
            await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="test_lora_4")
            logprobs_loaded_1 = np.concatenate(
                [output["logprobs"].tolist() for output in outputs_1.loss_fn_outputs]
            )
            loss_per_token_loaded_1 = -np.dot(logprobs_loaded_1, weights) / weights.sum()
            loss_per_tokens_loaded_1.append(loss_per_token_loaded_1)
            logprobs_loaded_2 = np.concatenate(
                [output["logprobs"].tolist() for output in outputs_2.loss_fn_outputs]
            )
            loss_per_token_loaded_2 = -np.dot(logprobs_loaded_2, weights) / weights.sum()
            loss_per_tokens_loaded_2.append(loss_per_token_loaded_2)
            print(f"(1) Loss per token at step {step}: {loss_per_token_loaded_1:.4f}")
            print(f"(2) Loss per token at step {step}: {loss_per_token_loaded_2:.4f}")
    assert (
        loss_per_tokens_loaded_1[0] < loss_per_tokens_1[-1]
    ), "Loaded lora_id test_lora_3 did not improve over saved state"
    assert (
        loss_per_tokens_loaded_2[0] < loss_per_tokens_2[-1]
    ), "Loaded lora_id test_lora_4 did not improve over saved state"
    for i in range(1, len(loss_per_tokens_loaded_1)):
        assert (
            loss_per_tokens_loaded_1[i] < loss_per_tokens_loaded_1[i - 1]
        ), "Loss did not decrease for loaded lora_id test_lora_3"
        assert (
            loss_per_tokens_loaded_2[i] < loss_per_tokens_loaded_2[i - 1]
        ), "Loss did not decrease for loaded lora_id test_lora_4"


# From offical Tinker on  Qwen/Qwen3-8B:
# Loss per token: 4.2681
# Forward Backward Metrics: {'clock_cycle:unique': 8649633.0, 'loss:sum': 230.4754238128662}
# Optimization Step Metrics: None
# Loss per token: 3.8261
# Forward Backward Metrics: {'clock_cycle:unique': 8649635.0, 'loss:sum': 206.60849380493164}
# Optimization Step Metrics: None
# Loss per token: 2.7188
# Forward Backward Metrics: {'clock_cycle:unique': 8649637.0, 'loss:sum': 146.8175163269043}
# Optimization Step Metrics: None
# Loss per token: 1.8391
# Forward Backward Metrics: {'clock_cycle:unique': 8649639.0, 'loss:sum': 99.31112432479858}
# Optimization Step Metrics: None
# Loss per token: 1.0804
# Forward Backward Metrics: {'clock_cycle:unique': 8649641.0, 'loss:sum': 58.34414064884186}
# Optimization Step Metrics: None
# Loss per token: 0.5762
# Forward Backward Metrics: {'clock_cycle:unique': 8649643.0, 'loss:sum': 31.11434930562973}
# Optimization Step Metrics: None
