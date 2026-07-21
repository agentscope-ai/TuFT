import os
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
import ray
import transformers
from tinker import types

from tuft.backends.training_backend import HFTrainingBackend
from tuft.checkpoints import CheckpointRecord
from tuft.config import ModelConfig

from .helpers import (
    PIG_LATIN_EXAMPLES,
    PIG_LATIN_EXAMPLES_EXTENDED,
    TEST_PROMPTS,
    _normalize_text,
    clear_ray_state,
)


@pytest.fixture(scope="function")
def ray_cluster(request):
    if request.config.getoption("--gpu"):
        # make sure we start with a fresh ray instance
        clear_ray_state()

        ray.init(ignore_reinit_error=True)
        yield
        clear_ray_state()
    else:
        yield


def _construct_data(name: str = "extended") -> List[types.Datum]:
    assert "TUFT_TEST_MODEL" in os.environ, (
        "Environment variable TUFT_TEST_MODEL must be set for this test."
    )

    model_path = Path(os.environ.get("TUFT_TEST_MODEL", "Qwen/Qwen3-0.6B"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    examples = PIG_LATIN_EXAMPLES_EXTENDED if name == "extended" else PIG_LATIN_EXAMPLES
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


LOGPROB_MEAN_REL_DIFF_PCT_TOL = 10.0


def _construct_single_lora_logprob_datum(tokenizer) -> tuple[list[types.Datum], list[int]]:
    prompt = "English: banana split\nPig Latin:"
    completion = " anana-bay plit-say\n\n"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=target_tokens, dtype="int64"),
                "weights": types.TensorData(
                    data=[1.0] * len(target_tokens),
                    dtype="float32",
                ),
            },
        )
    ]
    return data, tokens


def _checkpoint_record(temp_dir: Path, checkpoint_id: str) -> CheckpointRecord:
    return CheckpointRecord(
        checkpoint_id=checkpoint_id,
        owner_name="default",
        checkpoint_type="training",
        path=temp_dir / checkpoint_id,
        training_run_id="test_logprob_run",
        size_bytes=0,
    )


async def _create_training_backend_for_logprob_check(model_config: ModelConfig, backend_name: str):
    if backend_name == "hfpeft":
        return HFTrainingBackend(model_config)
    if backend_name == "fsdp":
        from tuft.backends.fsdp_training_backend import FSDPTrainingBackend

        return FSDPTrainingBackend(model_config)
    raise ValueError(f"Unknown backend: {backend_name}")


async def _trained_lora_training_logprobs(training_backend, data: list[types.Datum]) -> np.ndarray:
    await training_backend.create_adapter("test_lora", types.LoraConfig(rank=8, seed=42))
    for _step in range(2):
        await training_backend.forward(
            data=data,
            lora_id="test_lora",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )
        await training_backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="test_lora")
    outputs = await training_backend.forward(
        data=data,
        lora_id="test_lora",
        loss_fn="cross_entropy",
        loss_fn_config=None,
        backward=False,
    )
    return np.asarray(outputs.loss_fn_outputs[0]["logprobs"].tolist(), dtype=np.float32)


async def _vllm_lora_prompt_logprobs(
    model_config: ModelConfig,
    checkpoint: CheckpointRecord,
    tokens: list[int],
):
    from tuft.backends.sampling_backend import VLLMSamplingBackend

    sampling_backend = VLLMSamplingBackend(model_config)
    try:
        await sampling_backend.async_init()
        await sampling_backend.add_adapter("test_lora", checkpoint.adapter_path)
        response = await sampling_backend.sample(
            prompt=types.ModelInput.from_ints(tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=1,
                temperature=1.0,
                top_p=1.0,
            ),
            include_prompt_logprobs=True,
            lora_id="test_lora",
        )
    finally:
        await sampling_backend.shutdown()

    assert response.prompt_logprobs is not None
    prompt_logprobs = response.prompt_logprobs[1:]
    assert all(logprob is not None for logprob in prompt_logprobs)
    return np.asarray(prompt_logprobs, dtype=np.float32)


def _logprob_mismatch_stats(
    training_logprobs: np.ndarray,
    sampling_logprobs: np.ndarray,
) -> dict[str, float]:
    assert training_logprobs.shape == sampling_logprobs.shape
    abs_diff = np.abs(training_logprobs - sampling_logprobs)
    rel_diff_pct = abs_diff / np.maximum(np.abs(sampling_logprobs), 1e-6) * 100.0
    return {
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
        "mean_rel_diff_pct": float(rel_diff_pct.mean()),
        "max_rel_diff_pct": float(rel_diff_pct.max()),
    }


def _assert_logprobs_close(
    training_logprobs: np.ndarray,
    sampling_logprobs: np.ndarray,
    *,
    label: str,
) -> None:
    stats = _logprob_mismatch_stats(training_logprobs, sampling_logprobs)
    print(
        "[lora-logprob-mismatch] "
        f"{label}: "
        f"mean_abs={stats['mean_abs_diff']:.6f}, "
        f"max_abs={stats['max_abs_diff']:.6f}, "
        f"mean_pct={stats['mean_rel_diff_pct']:.4f}%, "
        f"max_pct={stats['max_rel_diff_pct']:.4f}%",
        flush=True,
    )
    assert stats["mean_rel_diff_pct"] <= LOGPROB_MEAN_REL_DIFF_PCT_TOL, (
        "training/vLLM LoRA prompt-average logprob mismatch exceeds "
        f"{LOGPROB_MEAN_REL_DIFF_PCT_TOL:.1f}% for {label}: "
        f"mean pct={stats['mean_rel_diff_pct']:.4f}%; "
        f"max pct={stats['max_rel_diff_pct']:.4f}%; "
        f"mean abs diff={stats['mean_abs_diff']:.4f}; "
        f"max abs diff={stats['max_abs_diff']:.4f}"
    )


def _fsdp_logprob_gpu_counts(device_count: int) -> list[int]:
    gpu_counts = [1]
    fsdp_num_gpus = 2
    while fsdp_num_gpus <= device_count:
        gpu_counts.append(fsdp_num_gpus)
        fsdp_num_gpus *= 2
    return gpu_counts


async def _run_lora_logprob_match_case(
    *,
    backend_name: str,
    fsdp_num_gpus: int,
    model_path: Path,
    tokenizer,
):
    import torch

    label = f"backend={backend_name}, fsdp_num_gpus={fsdp_num_gpus}"
    model_config = ModelConfig(
        model_name=f"Qwen/Qwen3-0.6B-logprob-{backend_name}-{fsdp_num_gpus}gpu",
        model_path=model_path,
        max_model_len=512,
        tensor_parallel_size=1,
        max_lora_rank=8,
        training_backend="fsdp" if backend_name == "fsdp" else "hf",
        fsdp_num_gpus=fsdp_num_gpus,
        sampling_memory_fraction=0.5,
    )
    single_data, tokens = _construct_single_lora_logprob_datum(tokenizer)
    data = single_data * fsdp_num_gpus if backend_name == "fsdp" else single_data

    prev_no_ray = os.environ.get("TUFT_FSDP_NO_RAY")
    if backend_name == "fsdp" and fsdp_num_gpus == 1:
        os.environ["TUFT_FSDP_NO_RAY"] = "1"
    elif backend_name == "fsdp":
        os.environ.pop("TUFT_FSDP_NO_RAY", None)

    training_backend = None
    try:
        training_backend = await _create_training_backend_for_logprob_check(
            model_config,
            backend_name,
        )
        await training_backend.async_init()
        training_logprobs = await _trained_lora_training_logprobs(training_backend, data)

        with tempfile.TemporaryDirectory() as tmpdirname:
            checkpoint = _checkpoint_record(
                Path(tmpdirname),
                f"{backend_name}_{fsdp_num_gpus}gpu_test_lora",
            )
            await training_backend.save_state(
                lora_id="test_lora",
                checkpoint_record=checkpoint,
                optimizer=False,
            )
            await training_backend.shutdown()
            training_backend = None
            torch.cuda.empty_cache()

            sampling_logprobs = await _vllm_lora_prompt_logprobs(
                model_config,
                checkpoint,
                tokens,
            )

        _assert_logprobs_close(training_logprobs, sampling_logprobs, label=label)

    finally:
        if training_backend is not None:
            await training_backend.shutdown()
        if backend_name == "fsdp":
            if prev_no_ray is None:
                os.environ.pop("TUFT_FSDP_NO_RAY", None)
            else:
                os.environ["TUFT_FSDP_NO_RAY"] = prev_no_ray
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", ["hfpeft", "fsdp"])
async def test_lora_training_logprobs_match_vllm_sampling(backend_name: str, ray_cluster):
    import torch

    if "TUFT_TEST_MODEL" not in os.environ:
        pytest.skip("Set TUFT_TEST_MODEL for LoRA logprob consistency test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping vLLM logprob consistency test")

    model_path = Path(os.environ.get("TUFT_TEST_MODEL", "Qwen/Qwen3-0.6B"))
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    cuda_device_count = torch.cuda.device_count()

    if backend_name == "hfpeft":
        await _run_lora_logprob_match_case(
            backend_name=backend_name,
            fsdp_num_gpus=1,
            model_path=model_path,
            tokenizer=tokenizer,
        )
        return

    fsdp_gpu_counts = _fsdp_logprob_gpu_counts(cuda_device_count)
    print(
        "[lora-logprob-mismatch] "
        f"detected_cuda_gpus={cuda_device_count}; fsdp_test_gpu_counts={fsdp_gpu_counts}",
        flush=True,
    )
    for fsdp_num_gpus in fsdp_gpu_counts:
        await _run_lora_logprob_match_case(
            backend_name=backend_name,
            fsdp_num_gpus=fsdp_num_gpus,
            model_path=model_path,
            tokenizer=tokenizer,
        )


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_training_backend():
    assert "TUFT_TEST_MODEL" in os.environ, (
        "Environment variable TUFT_TEST_MODEL must be set for this test."
    )

    model_path = Path(os.environ.get("TUFT_TEST_MODEL", "Qwen/Qwen3-8B"))
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
        assert loss_per_tokens_1[i] < loss_per_tokens_1[i - 1], (
            "Loss did not decrease for lora_id test_lora_1"
        )
        assert loss_per_tokens_2[i] < loss_per_tokens_2[i - 1], (
            "Loss did not decrease for lora_id test_lora_2"
        )
        assert abs(loss_per_tokens_1[i] - loss_per_tokens_2[i]) < 0.2, (
            "Losses for both LoRAs diverged unexpectedly"
        )
    # test saving and loading adapter
    # use a temp directory to save and load
    loss_per_tokens_loaded_1 = []
    loss_per_tokens_loaded_2 = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)
        checkpoint_1 = CheckpointRecord(
            checkpoint_id="test_lora_1",
            owner_name="default",
            checkpoint_type="training",
            path=temp_dir / "test_lora_1",
            training_run_id="test_run_1",
            size_bytes=0,
        )
        checkpoint_2 = CheckpointRecord(
            checkpoint_id="test_lora_2",
            owner_name="default",
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
    assert loss_per_tokens_loaded_1[0] < loss_per_tokens_1[-1], (
        "Loaded lora_id test_lora_3 did not improve over saved state"
    )
    assert loss_per_tokens_loaded_2[0] < loss_per_tokens_2[-1], (
        "Loaded lora_id test_lora_4 did not improve over saved state"
    )
    for i in range(1, len(loss_per_tokens_loaded_1)):
        assert loss_per_tokens_loaded_1[i] < loss_per_tokens_loaded_1[i - 1], (
            "Loss did not decrease for loaded lora_id test_lora_3"
        )
        assert loss_per_tokens_loaded_2[i] < loss_per_tokens_loaded_2[i - 1], (
            "Loss did not decrease for loaded lora_id test_lora_4"
        )


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


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_colocate_sampling_and_training():
    from tuft.backends.sampling_backend import VLLMSamplingBackend
    from tuft.backends.training_backend import HFTrainingBackend

    assert "TUFT_TEST_MODEL" in os.environ, (
        "Environment variable TUFT_TEST_MODEL must be set for this test."
    )

    model_path = Path(os.environ.get("TUFT_TEST_MODEL", "Qwen/Qwen3-0.6B"))
    model_config = ModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        model_path=model_path,
        max_model_len=2048,
        tensor_parallel_size=1,
        colocate=True,
        sampling_memory_fraction=0.25,
    )
    training_backend = HFTrainingBackend(model_config)
    sampling_backend = VLLMSamplingBackend(model_config)
    await training_backend.async_init()
    await sampling_backend.async_init()

    await training_backend.create_adapter("test_lora", types.LoraConfig(rank=8, seed=42))

    data = _construct_data(name="default")
    weights = np.concatenate([example.loss_fn_inputs["weights"].tolist() for example in data])
    for step in range(20):
        # test two separate lora training in turn
        outputs = await training_backend.forward(
            data=data,
            lora_id="test_lora",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )

        await training_backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="test_lora")
        logprobs = np.concatenate(
            [output["logprobs"].tolist() for output in outputs.loss_fn_outputs]
        )
        loss_per_token = -np.dot(logprobs, weights) / weights.sum()
        print(f"(1) Loss per token at step {step}: {loss_per_token:.4f}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)
        checkpoint = CheckpointRecord(
            checkpoint_id="test_lora",
            owner_name="default",
            checkpoint_type="training",
            path=temp_dir / "test_lora",
            training_run_id="test_run",
            size_bytes=0,
        )
        await training_backend.save_state(
            lora_id="test_lora", checkpoint_record=checkpoint, optimizer=False
        )
        await sampling_backend.add_adapter("test_lora", checkpoint.adapter_path)

        for prompt_text, example in zip(TEST_PROMPTS, PIG_LATIN_EXAMPLES, strict=True):
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            sample_res = await sampling_backend.sample(
                prompt=types.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=16,
                    temperature=0.1,
                    top_p=1.0,
                    stop=["\n"],
                ),
                lora_id="test_lora",
            )
            assert sample_res.sequences and sample_res.sequences[0].tokens
            output_text = tokenizer.decode(sample_res.sequences[0].tokens, skip_special_tokens=True)
            assert _normalize_text(output_text) == _normalize_text(example["output"]), (
                f"Expected {_normalize_text(example['output'])}, got {_normalize_text(output_text)}"
            )
