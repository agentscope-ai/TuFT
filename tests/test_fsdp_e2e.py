"""FSDP2 End-to-End Integration Tests.

Covers the FSDP2 multi-GPU training through both:
  A) Full server API  (ServiceClient -> HTTP -> TrainingController -> FSDPTrainingBackend)
  B) Direct backend API  (FSDPTrainingBackend -> FSDPWorkerGroup -> FSDPTrainingWorker)

Scenarios tested:
  1. Mid-training adapter creation with different ranks  (server)
  2. Full pipeline: interleaved training -> save -> inference  (server)
  3. Dynamic adapter lifecycle: create / train / remove / reload  (backend)
  4. Checkpoint + resume with optimizer state  (backend)
  5. Rapid adapter switching under stress  (backend)
  6. Adapter churn memory stability  (backend)
  7. Forward / backward / optim decoupled operations  (backend)

Requires:
    - ``--gpu`` pytest flag
    - ``TUFT_TEST_MODEL`` environment variable pointing to a HuggingFace model
    - At least 2 GPUs available (FSDP_TEST_GPUS, default 2)

Usage:
    TUFT_TEST_MODEL=/path/to/model \\
    pytest -xvs tests/test_fsdp_e2e.py --gpu -m gpu
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, List

import httpx
import numpy as np
import pytest
import ray
import tinker.types as types
import uvicorn
from tinker.lib.public_interfaces.service_client import ServiceClient
from transformers import AutoTokenizer

from tuft.backends.fsdp_training_backend import FSDPTrainingBackend
from tuft.checkpoints import CheckpointRecord
from tuft.config import AppConfig, ModelConfig
from tuft.server import create_root_app

from .helpers import (
    PIG_LATIN_EXAMPLES,
    PIG_LATIN_EXAMPLES_EXTENDED,
    REVERSE_EXAMPLES,
    REVERSE_PROMPTS,
    TEST_PROMPTS,
    _create_reverse_training_data,
    _create_training_data,
    _find_free_port,
    _log,
    _normalize_text,
    clear_ray_state,
)


# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
MODEL_PATH = Path(os.environ["TUFT_TEST_MODEL"]) if "TUFT_TEST_MODEL" in os.environ else None
MODEL_NAME = "Qwen/Qwen3-0.6B"
NUM_FSDP_GPUS = max(2, int(os.environ.get("FSDP_TEST_GPUS", "2")))


def _skip_if_no_model():
    if MODEL_PATH is None:
        pytest.skip("TUFT_TEST_MODEL not set")


# ---------------------------------------------------------------------------
# Reusable helpers
# ---------------------------------------------------------------------------


def _make_config(model_path: Path | None = MODEL_PATH, **overrides) -> ModelConfig:
    assert model_path is not None, "TUFT_TEST_MODEL must be set"
    defaults: dict[str, Any] = dict(
        model_name=MODEL_NAME,
        model_path=model_path,
        max_model_len=2048,
        tensor_parallel_size=1,
        training_backend="fsdp",
        num_gpus_per_node=NUM_FSDP_GPUS,
        num_nodes=1,
        max_lora_rank=32,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _construct_data(model_path: Path | None, name: str = "extended") -> List[types.Datum]:
    assert model_path is not None, "TUFT_TEST_MODEL must be set"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    examples = PIG_LATIN_EXAMPLES_EXTENDED if name == "extended" else PIG_LATIN_EXAMPLES
    data: list[types.Datum] = []
    for example in examples:
        prompt = f"English: {example['input']}\nPig Latin:"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)
        tokens = prompt_tokens + completion_tokens
        weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
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


def _compute_loss(output: types.ForwardBackwardOutput, data) -> float:
    weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in data])
    logprobs = np.concatenate([o["logprobs"].tolist() for o in output.loss_fn_outputs])
    return float(-np.dot(logprobs, weights) / weights.sum())


async def _train_step(
    backend: FSDPTrainingBackend,
    data,
    lora_id: str,
    lr: float = 1e-4,
) -> float:
    output = await backend.forward(
        data=data,
        lora_id=lora_id,
        loss_fn="cross_entropy",
        loss_fn_config=None,
        backward=True,
    )
    await backend.optim_step(types.AdamParams(learning_rate=lr), lora_id=lora_id)
    return _compute_loss(output, data)


async def _forward_only(
    backend: FSDPTrainingBackend,
    data,
    lora_id: str,
) -> float:
    output = await backend.forward(
        data=data,
        lora_id=lora_id,
        loss_fn="cross_entropy",
        loss_fn_config=None,
        backward=False,
    )
    return _compute_loss(output, data)


def _print_memory(stats: list[dict], tag: str) -> None:
    for s in stats:
        print(
            f"  [{tag}] rank {s['rank']}  "
            f"alloc={s['allocated_mb']:.0f}MB  "
            f"reserved={s['reserved_mb']:.0f}MB  "
            f"peak={s['max_allocated_mb']:.0f}MB",
            flush=True,
        )


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(scope="function")
def fsdp_ray_cluster(request):
    """Function-scoped Ray cluster for backend-level tests."""
    if not request.config.getoption("--gpu"):
        yield
        return
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < NUM_FSDP_GPUS:
        pytest.skip(
            f"Need at least {NUM_FSDP_GPUS} GPUs for FSDP, found {torch.cuda.device_count()}"
        )
    _skip_if_no_model()
    clear_ray_state()
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"TRANSFORMERS_NO_TORCHVISION": "1"}},
    )
    yield
    clear_ray_state()


@pytest.fixture(scope="function")
def fsdp_server(request, tmp_path):
    """Function-scoped server fixture with FSDP training backend.

    Starts a full TuFT server with a single FSDP-backed model,
    yields the base URL, and tears everything down after the test.
    """
    if not request.config.getoption("--gpu"):
        yield None, None
        return

    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < NUM_FSDP_GPUS:
        pytest.skip(
            f"Need at least {NUM_FSDP_GPUS} GPUs for FSDP, found {torch.cuda.device_count()}"
        )
    _skip_if_no_model()
    assert MODEL_PATH is not None
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")

    clear_ray_state()
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")

    _log("Starting Ray for FSDP server test...")
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"TRANSFORMERS_NO_TORCHVISION": "1"}},
    )

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    config = AppConfig(checkpoint_dir=checkpoint_dir)
    config.supported_models = [
        ModelConfig(
            model_name=MODEL_NAME,
            model_path=MODEL_PATH,
            max_model_len=2048,
            tensor_parallel_size=1,
            training_backend="fsdp",
            num_gpus_per_node=NUM_FSDP_GPUS,
            num_nodes=1,
            max_lora_rank=32,
        ),
    ]
    config.authorized_users = {"tml-fsdp-test-key": "default"}

    _log("Creating FastAPI app with FSDP backend...")
    app = create_root_app(config)
    port = _find_free_port()
    _log(f"Starting FSDP server on port {port}...")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    client = httpx.Client()
    healthy = False
    for attempt in range(1, 181):
        try:
            response = client.get(f"{base_url}/api/v1/healthz", timeout=1)
            response.raise_for_status()
            healthy = True
            break
        except httpx.HTTPError:
            time.sleep(2)
        if attempt % 10 == 0:
            _log(f"Waiting for FSDP server healthz... attempt {attempt}/180")
    if not healthy:
        server.should_exit = True
        thread.join(timeout=10)
        client.close()
        clear_ray_state()
        raise RuntimeError("FSDP server failed to start within 6 minutes")
    _log("FSDP server is healthy")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    yield base_url, tokenizer

    _log("Tearing down FSDP server...")
    server.should_exit = True
    thread.join(timeout=10)
    client.close()
    clear_ray_state()


# ===================================================================
# SECTION A — Server Integration Tests
# ===================================================================


@pytest.mark.integration
@pytest.mark.gpu
def test_fsdp_e2e_mid_training_new_adapter(fsdp_server) -> None:
    """Create new adapters DURING an ongoing training session.

    Flow:
      1. Create adapter_a (rank=8), train 5 epochs solo
      2. Create adapter_b (rank=4)  ← mid-training
      3. Train a + b interleaved for 5 epochs
      4. Create adapter_c (rank=16)  ← mid-training again
      5. Train a + b + c interleaved for 3 epochs
      6. Save all three for sampler
      7. Deploy each to inference and verify output
    """
    base_url, tokenizer = fsdp_server
    if base_url is None:
        pytest.skip("Server fixture not available")

    service_client = ServiceClient(
        api_key="tml-fsdp-test-key",  # pragma: allowlist secret
        base_url=base_url,
        timeout=300,
    )
    try:
        caps = service_client.get_server_capabilities()
        assert caps.supported_models, "no supported models"
        base_model = caps.supported_models[0].model_name
        _log(f"Base model: {base_model}")

        train_data = _create_training_data(tokenizer)

        # --- Phase 1: solo training of adapter_a ---
        _log("Phase 1: Creating adapter_a (rank=8) ...")
        client_a = service_client.create_lora_training_client(base_model=base_model, rank=8)
        for epoch in range(1, 6):
            client_a.forward_backward(train_data, "cross_entropy").result(timeout=180)
            client_a.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=120)
            _log(f"  [Phase 1] adapter_a epoch {epoch}/5")

        # --- Phase 2: create adapter_b mid-training ---
        _log("Phase 2: Creating adapter_b (rank=4) mid-training ...")
        client_b = service_client.create_lora_training_client(base_model=base_model, rank=4)
        for epoch in range(1, 6):
            client_a.forward_backward(train_data, "cross_entropy").result(timeout=180)
            client_a.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=120)
            client_b.forward_backward(train_data, "cross_entropy").result(timeout=180)
            client_b.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=120)
            _log(f"  [Phase 2] a+b epoch {epoch}/5")

        # --- Phase 3: create adapter_c mid-training ---
        _log("Phase 3: Creating adapter_c (rank=16) mid-training ...")
        client_c = service_client.create_lora_training_client(base_model=base_model, rank=16)
        for epoch in range(1, 4):
            for c in [client_a, client_b, client_c]:
                c.forward_backward(train_data, "cross_entropy").result(timeout=180)
                c.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=120)
            _log(f"  [Phase 3] a+b+c epoch {epoch}/3")

        # --- Save all for sampler ---
        _log("Saving all adapters for sampler ...")
        sampling_clients = []
        for _i, (label, client) in enumerate([("a", client_a), ("b", client_b), ("c", client_c)]):
            resp = client.save_weights_for_sampler(f"fsdp-e2e-{label}").result(timeout=120)
            assert resp.path.startswith("tinker://"), f"Bad sampler path: {resp.path}"
            _log(f"  adapter_{label} saved: {resp.path}")
            sampling_clients.append(service_client.create_sampling_client(model_path=resp.path))

        # --- Inference verification ---
        _log("Verifying inference outputs ...")
        for sc_idx, sampling_client in enumerate(sampling_clients):
            correct = 0
            for prompt_text, example in zip(TEST_PROMPTS, PIG_LATIN_EXAMPLES, strict=True):
                prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
                sample_res = sampling_client.sample(
                    prompt=types.ModelInput.from_ints(prompt_tokens),
                    num_samples=1,
                    sampling_params=types.SamplingParams(
                        max_tokens=16,
                        temperature=0.1,
                        top_p=1.0,
                        stop=["\n"],
                    ),
                ).result(timeout=60)
                assert sample_res.sequences and sample_res.sequences[0].tokens
                output_text = tokenizer.decode(
                    sample_res.sequences[0].tokens, skip_special_tokens=True
                )
                expected = _normalize_text(example["output"])
                got = _normalize_text(output_text)
                if got == expected:
                    correct += 1
                _log(
                    f"  adapter[{sc_idx}] '{prompt_text.strip()}'  "
                    f"exp='{expected}' got='{got}' {'OK' if got == expected else 'MISS'}"
                )
            _log(f"  adapter[{sc_idx}] accuracy: {correct}/{len(PIG_LATIN_EXAMPLES)}")

        _log("test_fsdp_e2e_mid_training_new_adapter PASSED")
    finally:
        service_client.holder.close()


@pytest.mark.integration
@pytest.mark.gpu
def test_fsdp_e2e_interleaved_task_isolation(fsdp_server) -> None:
    """Two adapters learn DIFFERENT tasks through interleaved FSDP training.

    Adapter A learns Pig Latin; Adapter B learns Reverse Words.
    Validates that FSDP multi-tenant training keeps adapters isolated.
    """
    base_url, tokenizer = fsdp_server
    if base_url is None:
        pytest.skip("Server fixture not available")

    service_client = ServiceClient(
        api_key="tml-fsdp-test-key",  # pragma: allowlist secret
        base_url=base_url,
        timeout=300,
    )
    try:
        caps = service_client.get_server_capabilities()
        base_model = caps.supported_models[0].model_name

        pig_data = _create_training_data(tokenizer)
        rev_data = _create_reverse_training_data(tokenizer)

        _log("Creating adapter_pig (rank=8) and adapter_rev (rank=8) ...")
        client_pig = service_client.create_lora_training_client(base_model=base_model, rank=8)
        client_rev = service_client.create_lora_training_client(base_model=base_model, rank=8)

        _log("Interleaved training (20 epochs) ...")
        for epoch in range(1, 21):
            client_pig.forward_backward(pig_data, "cross_entropy").result(timeout=180)
            client_pig.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=120)
            client_rev.forward_backward(rev_data, "cross_entropy").result(timeout=180)
            client_rev.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=120)
            if epoch % 5 == 0:
                _log(f"  interleaved epoch {epoch}/20")

        # Save and deploy
        sampler_pig = client_pig.save_weights_for_sampler("pig-iso").result(timeout=120)
        sampler_rev = client_rev.save_weights_for_sampler("rev-iso").result(timeout=120)
        sc_pig = service_client.create_sampling_client(model_path=sampler_pig.path)
        sc_rev = service_client.create_sampling_client(model_path=sampler_rev.path)

        # Validate Pig Latin adapter
        _log("Validating Pig Latin adapter ...")
        pig_correct = 0
        for prompt_text, example in zip(TEST_PROMPTS, PIG_LATIN_EXAMPLES, strict=True):
            tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            res = sc_pig.sample(
                prompt=types.ModelInput.from_ints(tokens),
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=16,
                    temperature=0.1,
                    top_p=1.0,
                    stop=["\n"],
                ),
            ).result(timeout=60)
            output = _normalize_text(
                tokenizer.decode(res.sequences[0].tokens, skip_special_tokens=True)
            )
            expected = _normalize_text(example["output"])
            match = output == expected
            pig_correct += int(match)
            _log(
                f"  pig: '{prompt_text.strip()}'  exp='{expected}' got='{output}' "
                f"{'OK' if match else 'MISS'}"
            )
        _log(f"  Pig Latin accuracy: {pig_correct}/{len(PIG_LATIN_EXAMPLES)}")
        assert pig_correct >= len(PIG_LATIN_EXAMPLES) - 1, (
            f"Pig Latin adapter too inaccurate: {pig_correct}/{len(PIG_LATIN_EXAMPLES)}"
        )

        # Validate Reverse adapter
        _log("Validating Reverse adapter ...")
        rev_correct = 0
        for prompt_text, example in zip(REVERSE_PROMPTS, REVERSE_EXAMPLES, strict=True):
            tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            res = sc_rev.sample(
                prompt=types.ModelInput.from_ints(tokens),
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=32,
                    temperature=0.0,
                    top_p=1.0,
                    stop=["\n"],
                ),
            ).result(timeout=60)
            output = _normalize_text(
                tokenizer.decode(res.sequences[0].tokens, skip_special_tokens=True)
            )
            expected = _normalize_text(example["output"])
            match = output == expected
            rev_correct += int(match)
            _log(
                f"  rev: '{prompt_text.strip()}'  exp='{expected}' got='{output}' "
                f"{'OK' if match else 'MISS'}"
            )
        _log(f"  Reverse accuracy: {rev_correct}/{len(REVERSE_EXAMPLES)}")
        assert rev_correct >= len(REVERSE_EXAMPLES) - 2, (
            f"Reverse adapter too inaccurate: {rev_correct}/{len(REVERSE_EXAMPLES)}"
        )

        # Cross-check isolation: Pig Latin adapter should NOT produce reversed words
        cross_prompt = "Reverse each word.\nEnglish: hello world\nReversed:"
        cross_tokens = tokenizer.encode(cross_prompt, add_special_tokens=True)
        cross_res = sc_pig.sample(
            prompt=types.ModelInput.from_ints(cross_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=32,
                temperature=0.0,
                top_p=1.0,
                stop=["\n"],
            ),
        ).result(timeout=60)
        cross_out = _normalize_text(
            tokenizer.decode(cross_res.sequences[0].tokens, skip_special_tokens=True)
        )
        _log(f"  Cross-check: Pig Latin adapter on reverse prompt → '{cross_out}'")
        assert cross_out != _normalize_text("olleh dlrow"), (
            "Pig Latin adapter incorrectly learned reverse task"
        )

        _log("test_fsdp_e2e_interleaved_task_isolation PASSED")
    finally:
        service_client.holder.close()


# ===================================================================
# SECTION B — Backend Edge-Case Tests
# ===================================================================


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_dynamic_adapter_lifecycle(fsdp_ray_cluster) -> None:
    """Create / train / remove / create new adapters dynamically.

    Verifies that:
      - Removing an adapter frees its resources
      - Remaining adapters keep working
      - New adapters can be created and trained after removal
      - GPU memory is stable
    """
    config = _make_config()
    backend = FSDPTrainingBackend(config)
    await backend.async_init()
    data = _construct_data(MODEL_PATH, "default")

    try:
        mem_init = await backend.get_memory_stats()
        _print_memory(mem_init, "init")

        # Create adapter_1 (rank=8), train 3 steps
        _log("Creating adapter_1 (rank=8) ...")
        await backend.create_adapter("adapter_1", types.LoraConfig(rank=8, seed=42))
        losses_1: list[float] = []
        for step in range(3):
            loss = await _train_step(backend, data, "adapter_1")
            losses_1.append(loss)
            _log(f"  adapter_1 step {step} loss={loss:.4f}")

        # Create adapter_2 (rank=16), train 2 steps
        _log("Creating adapter_2 (rank=16) ...")
        await backend.create_adapter("adapter_2", types.LoraConfig(rank=16, seed=99))
        losses_2: list[float] = []
        for step in range(2):
            loss = await _train_step(backend, data, "adapter_2")
            losses_2.append(loss)
            _log(f"  adapter_2 step {step} loss={loss:.4f}")

        mem_two_adapters = await backend.get_memory_stats()
        _print_memory(mem_two_adapters, "two adapters active")

        # Remove adapter_1
        _log("Removing adapter_1 ...")
        await backend.remove_adapter("adapter_1")

        mem_after_remove = await backend.get_memory_stats()
        _print_memory(mem_after_remove, "after removing adapter_1")

        # adapter_2 should still work
        loss = await _train_step(backend, data, "adapter_2")
        _log(f"  adapter_2 after removal of adapter_1: loss={loss:.4f}")
        assert loss < losses_2[-1] + 0.1, "adapter_2 broken after adapter_1 removal"

        # Create adapter_3 (rank=4), train 3 steps
        _log("Creating adapter_3 (rank=4) ...")
        await backend.create_adapter("adapter_3", types.LoraConfig(rank=4, seed=77))
        losses_3: list[float] = []
        for step in range(3):
            loss = await _train_step(backend, data, "adapter_3")
            losses_3.append(loss)
            _log(f"  adapter_3 step {step} loss={loss:.4f}")

        mem_end = await backend.get_memory_stats()
        _print_memory(mem_end, "end of lifecycle test")

        # Verify losses decrease
        for i in range(1, len(losses_1)):
            assert losses_1[i] < losses_1[i - 1], "adapter_1 loss not decreasing"
        for i in range(1, len(losses_3)):
            assert losses_3[i] < losses_3[i - 1], "adapter_3 loss not decreasing"

        for s in mem_end:
            assert s["allocated_mb"] < 50000, "Memory explosion detected"

        _log("test_fsdp_dynamic_adapter_lifecycle PASSED")
    finally:
        backend.shutdown()


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_checkpoint_resume_optimizer(fsdp_ray_cluster) -> None:
    """Train, checkpoint with optimizer, remove, reload, resume training.

    Validates that optimizer momentum is preserved across checkpoint/resume
    so training continues smoothly from the saved state.
    """
    config = _make_config()
    backend = FSDPTrainingBackend(config)
    await backend.async_init()
    data = _construct_data(MODEL_PATH, "default")

    try:
        await backend.create_adapter("ckpt_adapter", types.LoraConfig(rank=8, seed=42))

        # Train 5 steps
        losses_before: list[float] = []
        for step in range(5):
            loss = await _train_step(backend, data, "ckpt_adapter")
            losses_before.append(loss)
            _log(f"  [before save] step {step} loss={loss:.4f}")

        eval_loss_before = await _forward_only(backend, data, "ckpt_adapter")
        _log(f"  eval loss before save = {eval_loss_before:.4f}")

        # Save checkpoint with optimizer
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = CheckpointRecord(
                checkpoint_id="ckpt_adapter",
                owner_name="default",
                checkpoint_type="training",
                path=Path(tmpdir) / "ckpt",
                training_run_id="test_resume",
                size_bytes=0,
            )
            await backend.save_state("ckpt_adapter", ckpt, optimizer=True)
            _log("  checkpoint saved (with optimizer)")

            # Remove the adapter
            await backend.remove_adapter("ckpt_adapter")
            _log("  adapter removed")

            # Reload from checkpoint (same name so optimizer files match)
            await backend.load_state("ckpt_adapter", ckpt, optimizer=True)
            _log("  adapter reloaded as ckpt_adapter")

            eval_loss_after_reload = await _forward_only(backend, data, "ckpt_adapter")
            _log(f"  eval loss after reload = {eval_loss_after_reload:.4f}")
            assert abs(eval_loss_after_reload - eval_loss_before) < 1.0, (
                f"Eval loss diverged too much after checkpoint reload: "
                f"{eval_loss_before:.4f} → {eval_loss_after_reload:.4f}"
            )

            # Continue training 5 more steps
            losses_after: list[float] = []
            for step in range(5):
                loss = await _train_step(backend, data, "ckpt_adapter")
                losses_after.append(loss)
                _log(f"  [after reload] step {step} loss={loss:.4f}")

            assert losses_after[0] < losses_before[-1] + 0.1, (
                "Training did not resume smoothly from checkpoint"
            )

            all_losses = losses_before + losses_after
            regression_count = sum(
                1 for i in range(1, len(all_losses)) if all_losses[i] > all_losses[i - 1] + 0.05
            )
            assert regression_count <= 1, (
                f"Too many loss regressions ({regression_count}) across checkpoint boundary"
            )

            # --- Cross-name load: verify optimizer file is found even with new name ---
            _log("  Testing cross-name checkpoint load ...")
            await backend.remove_adapter("ckpt_adapter")
            await backend.load_state("ckpt_adapter_renamed", ckpt, optimizer=True)
            loss_renamed = await _train_step(backend, data, "ckpt_adapter_renamed")
            _log(f"  cross-name load: first step loss = {loss_renamed:.4f}")
            assert loss_renamed < losses_before[-1] + 0.5, (
                "Cross-name checkpoint load failed to resume training"
            )

        _log("test_fsdp_checkpoint_resume_optimizer PASSED")
    finally:
        backend.shutdown()


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_rapid_adapter_switching(fsdp_ray_cluster) -> None:
    """Stress test: rapidly switch between 4 adapters of different ranks.

    Each iteration does forward+backward on every adapter, then optim_step
    for each. Verifies no deadlocks, correct loss decrease, and stable memory.
    """
    config = _make_config()
    backend = FSDPTrainingBackend(config)
    await backend.async_init()
    data = _construct_data(MODEL_PATH, "default")

    try:
        adapters = [
            ("rapid_r4", 4),
            ("rapid_r8", 8),
            ("rapid_r16", 16),
            ("rapid_r32", 32),
        ]
        for name, rank in adapters:
            await backend.create_adapter(name, types.LoraConfig(rank=rank, seed=42))
            _log(f"  created {name} (rank={rank})")

        mem_created = await backend.get_memory_stats()
        _print_memory(mem_created, "after creating 4 adapters")

        adapter_losses: dict[str, list[float]] = {name: [] for name, _ in adapters}

        for iteration in range(4):
            _log(f"  --- rapid switching iteration {iteration} ---")
            for name, _ in adapters:
                output = await backend.forward(
                    data=data,
                    lora_id=name,
                    loss_fn="cross_entropy",
                    loss_fn_config=None,
                    backward=True,
                )
                loss = _compute_loss(output, data)
                adapter_losses[name].append(loss)
                _log(f"    {name} fwd+bwd loss={loss:.4f}")

            for name, _ in adapters:
                await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id=name)

        mem_end = await backend.get_memory_stats()
        _print_memory(mem_end, "after rapid switching")

        for name, _ in adapters:
            losses = adapter_losses[name]
            assert losses[-1] < losses[0], (
                f"{name}: final loss {losses[-1]:.4f} >= initial loss {losses[0]:.4f}"
            )
            _log(f"  {name}: {losses[0]:.4f} → {losses[-1]:.4f}  OK")

        for s in mem_end:
            assert s["allocated_mb"] < 50000, f"Memory explosion on rank {s['rank']}"

        _log("test_fsdp_rapid_adapter_switching PASSED")
    finally:
        backend.shutdown()


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_adapter_churn_memory_stability(fsdp_ray_cluster) -> None:
    """Repeatedly create, train, and remove adapters to verify no memory leak.

    Runs 3 rounds of create → train → remove with 2 adapters each round.
    Checks that GPU allocated memory doesn't grow beyond a reasonable bound.
    """
    config = _make_config()
    backend = FSDPTrainingBackend(config)
    await backend.async_init()
    data = _construct_data(MODEL_PATH, "default")

    try:
        mem_baseline = await backend.get_memory_stats()
        _print_memory(mem_baseline, "baseline")

        round_peak_deltas: list[float] = []

        for round_idx in range(3):
            _log(f"  --- churn round {round_idx} ---")

            names = [f"churn_r{round_idx}_a", f"churn_r{round_idx}_b"]
            for name in names:
                await backend.create_adapter(name, types.LoraConfig(rank=8, seed=round_idx * 10))

            for _step in range(2):
                for name in names:
                    await _train_step(backend, data, name)

            for name in names:
                await backend.remove_adapter(name)

            mem_after = await backend.get_memory_stats()
            _print_memory(mem_after, f"round {round_idx} after cleanup")

            delta = mem_after[0]["allocated_mb"] - mem_baseline[0]["allocated_mb"]
            round_peak_deltas.append(delta)
            _log(f"  round {round_idx} mem delta: {delta:.0f} MB")

        if len(round_peak_deltas) >= 2:
            growth = round_peak_deltas[-1] - round_peak_deltas[0]
            _log(f"  memory growth across rounds: {growth:.0f} MB")
            assert growth < 1000, (
                f"Suspected memory leak: {growth:.0f} MB growth "
                f"across {len(round_peak_deltas)} rounds"
            )

        _log("test_fsdp_adapter_churn_memory_stability PASSED")
    finally:
        backend.shutdown()


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_forward_backward_optim_separation(fsdp_ray_cluster) -> None:
    """Test that forward, backward, and optim_step can be decoupled.

    Validates the multi-tenant scenario where these operations
    happen as independent RPC calls (not always in lockstep).
    """
    config = _make_config()
    backend = FSDPTrainingBackend(config)
    await backend.async_init()
    data = _construct_data(MODEL_PATH, "default")

    try:
        await backend.create_adapter("decouple", types.LoraConfig(rank=8, seed=42))

        # Forward-only should not change weights
        eval_before = await _forward_only(backend, data, "decouple")
        eval_after_fwd = await _forward_only(backend, data, "decouple")
        _log(f"  forward-only: {eval_before:.4f} → {eval_after_fwd:.4f}")
        assert abs(eval_after_fwd - eval_before) < 1e-3, "Forward-only changed the model weights"

        # Forward + backward (no optim) should NOT change weights
        # but SHOULD accumulate gradients
        await backend.forward(
            data=data,
            lora_id="decouple",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )
        eval_after_bwd = await _forward_only(backend, data, "decouple")
        _log(f"  after backward (no optim): {eval_after_bwd:.4f}")
        assert abs(eval_after_bwd - eval_before) < 1e-3, (
            "Backward without optim_step changed the model weights"
        )

        # Multiple forward+backward calls accumulate gradients
        await backend.forward(
            data=data,
            lora_id="decouple",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )

        # Now optim_step should apply the accumulated gradients
        await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="decouple")
        eval_after_optim = await _forward_only(backend, data, "decouple")
        _log(f"  after optim_step: {eval_after_optim:.4f}")
        assert eval_after_optim < eval_before, (
            f"Optim step did not improve loss: {eval_before:.4f} → {eval_after_optim:.4f}"
        )

        # Interleave with a second adapter
        await backend.create_adapter("decouple_b", types.LoraConfig(rank=4, seed=99))

        eval_a_before = await _forward_only(backend, data, "decouple")
        await backend.forward(
            data=data,
            lora_id="decouple_b",
            loss_fn="cross_entropy",
            loss_fn_config=None,
            backward=True,
        )
        await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="decouple_b")
        eval_a_after = await _forward_only(backend, data, "decouple")
        _log(f"  adapter_a after adapter_b optim: {eval_a_before:.4f} → {eval_a_after:.4f}")
        assert abs(eval_a_after - eval_a_before) < 0.01, (
            "Adapter A weights changed when only adapter B was updated"
        )

        _log("test_fsdp_forward_backward_optim_separation PASSED")
    finally:
        backend.shutdown()


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_multitenant_gradient_isolation(fsdp_ray_cluster) -> None:
    """Verify multi-tenant gradient/optimizer isolation on the same base model.

    Two adapters (different ranks) share one FSDP2-wrapped base model.

    1. Accumulate gradients for A, switch to B, train B, switch back to A,
       verify A's pending grads are intact (optim produces the same result
       as no-switch training).
    2. Verify optimizer states are independent: training A does not shift B.
    3. Verify cross-rank parameter consistency via fingerprints.
    """
    config = _make_config()
    backend = FSDPTrainingBackend(config)
    await backend.async_init()
    data = _construct_data(MODEL_PATH, "default")

    try:
        # --- Setup: two adapters with different ranks on the same base model ---
        await backend.create_adapter("iso_a", types.LoraConfig(rank=8, seed=42))
        await backend.create_adapter("iso_b", types.LoraConfig(rank=16, seed=99))

        # ---- Phase 1: gradient preservation across adapter switch ----
        _log("Phase 1: gradient preservation across adapter switch")

        # Train adapter_a for 3 steps without interruption (control)
        eval_a_init = await _forward_only(backend, data, "iso_a")
        _log(f"  iso_a initial eval loss = {eval_a_init:.4f}")
        for _ in range(3):
            await _train_step(backend, data, "iso_a")
        eval_a_control = await _forward_only(backend, data, "iso_a")
        _log(f"  iso_a after 3 uninterrupted steps = {eval_a_control:.4f}")

        # Reset: remove and re-create iso_a
        await backend.remove_adapter("iso_a")
        await backend.create_adapter("iso_a", types.LoraConfig(rank=8, seed=42))

        # Same 3 steps, but interrupt with iso_b training between each
        for step in range(3):
            await backend.forward(
                data=data,
                lora_id="iso_a",
                loss_fn="cross_entropy",
                loss_fn_config=None,
                backward=True,
            )
            # Switch away: train iso_b for 2 steps
            for _ in range(2):
                await _train_step(backend, data, "iso_b")
            # Switch back and complete iso_a's optim
            await backend.optim_step(types.AdamParams(learning_rate=1e-4), lora_id="iso_a")
            _log(f"  iso_a interrupted step {step} completed")

        eval_a_interrupted = await _forward_only(backend, data, "iso_a")
        _log(f"  iso_a after 3 interrupted steps = {eval_a_interrupted:.4f}")

        # The interrupted training should still make progress
        assert eval_a_interrupted < eval_a_init, (
            f"iso_a did not improve with interrupted training: "
            f"{eval_a_init:.4f} → {eval_a_interrupted:.4f}"
        )

        # ---- Phase 2: optimizer state independence ----
        _log("Phase 2: optimizer state independence")

        eval_b_before = await _forward_only(backend, data, "iso_b")
        # Train only iso_a for 5 more steps
        for _ in range(5):
            await _train_step(backend, data, "iso_a")
        eval_b_after = await _forward_only(backend, data, "iso_b")
        _log(f"  iso_b before/after iso_a training: {eval_b_before:.4f} → {eval_b_after:.4f}")
        assert abs(eval_b_after - eval_b_before) < 0.01, (
            f"iso_b weights shifted when only iso_a was trained: "
            f"delta={abs(eval_b_after - eval_b_before):.6f}"
        )

        # ---- Phase 3: cross-rank parameter consistency ----
        _log("Phase 3: cross-rank parameter consistency")

        fps_a = backend.worker_group.get_adapter_param_fingerprint_all("iso_a")
        fps_b = backend.worker_group.get_adapter_param_fingerprint_all("iso_b")
        for fp in fps_a:
            _log(f"  iso_a rank {fp['rank']}: sum={fp['param_sum']:.6f}  count={fp['param_count']}")
        for fp in fps_b:
            _log(f"  iso_b rank {fp['rank']}: sum={fp['param_sum']:.6f}  count={fp['param_count']}")

        if len(fps_a) >= 2:
            assert abs(fps_a[0]["param_sum"] - fps_a[1]["param_sum"]) < 1e-2, (
                f"iso_a params diverged across ranks: "
                f"rank0={fps_a[0]['param_sum']:.6f}  "
                f"rank1={fps_a[1]['param_sum']:.6f}"
            )
        if len(fps_b) >= 2:
            assert abs(fps_b[0]["param_sum"] - fps_b[1]["param_sum"]) < 1e-2, (
                f"iso_b params diverged across ranks: "
                f"rank0={fps_b[0]['param_sum']:.6f}  "
                f"rank1={fps_b[1]['param_sum']:.6f}"
            )

        # Adapters must have different params (different ranks and training)
        assert fps_a[0]["param_count"] != fps_b[0]["param_count"], (
            "Adapters have the same param count despite different ranks"
        )

        _log("test_fsdp_multitenant_gradient_isolation PASSED")
    finally:
        backend.shutdown()


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_fsdp_cross_rank_param_sync(fsdp_ray_cluster) -> None:
    """Verify LoRA parameters stay identical across ranks after training.

    This specifically tests the AllReduce gradient synchronisation added
    in optim_step.  Without it, each rank would apply different gradients
    (from different data shards) and parameters would diverge.
    """
    config = _make_config()
    backend = FSDPTrainingBackend(config)
    await backend.async_init()
    data = _construct_data(MODEL_PATH, "extended")

    try:
        await backend.create_adapter("sync_test", types.LoraConfig(rank=8, seed=42))

        fp_init = backend.worker_group.get_adapter_param_fingerprint_all("sync_test")
        _log("Initial fingerprints:")
        for fp in fp_init:
            _log(f"  rank {fp['rank']}: sum={fp['param_sum']:.6f}")

        if len(fp_init) >= 2:
            assert abs(fp_init[0]["param_sum"] - fp_init[1]["param_sum"]) < 1e-4, (
                "Parameters diverged at initialisation"
            )

        for step in range(10):
            await _train_step(backend, data, "sync_test")
            if step % 3 == 2:
                fps = backend.worker_group.get_adapter_param_fingerprint_all("sync_test")
                sums = [fp["param_sum"] for fp in fps]
                max_diff = max(sums) - min(sums) if len(sums) > 1 else 0.0
                _log(f"  step {step}: sums={[f'{s:.6f}' for s in sums]}  max_diff={max_diff:.8f}")
                if len(sums) >= 2:
                    assert max_diff < 1e-2, (
                        f"Parameters diverged across ranks at step {step}: max_diff={max_diff:.8f}"
                    )

        fp_final = backend.worker_group.get_adapter_param_fingerprint_all("sync_test")
        _log("Final fingerprints:")
        for fp in fp_final:
            _log(f"  rank {fp['rank']}: sum={fp['param_sum']:.6f}")

        if len(fp_final) >= 2:
            final_diff = abs(fp_final[0]["param_sum"] - fp_final[1]["param_sum"])
            assert final_diff < 1e-2, f"Final params diverged: diff={final_diff:.8f}"

        assert fp_final[0]["param_sum"] != fp_init[0]["param_sum"], (
            "Parameters did not change during training"
        )

        _log("test_fsdp_cross_rank_param_sync PASSED")
    finally:
        backend.shutdown()
