from __future__ import annotations

import os
import re
import socket
import threading
import time
import warnings
from pathlib import Path

import httpx
import pytest
import ray
import uvicorn
from transformers import AutoTokenizer

import tinker.types as types
from tinker._exceptions import RequestFailedError
from tinker.lib.public_interfaces.service_client import ServiceClient
from tuft.config import AppConfig, ModelConfig
from tuft.persistence import PersistenceConfig
from tuft.server import create_root_app

"""
How to run this test (GPU required):
    TUFT_TEST_MODEL=/path/dir/qwen/Qwen3-0.6B \\
    pytest -s tests/test_integration_lora_train_sample.py --gpu -m gpu
    # -s prints real-time progress logs (server startup, training, sampling).

Notes:
    - The test is marked with @pytest.mark.gpu and will be skipped unless --gpu is set.
    - In CI without GPUs or without TUFT_TEST_MODEL, the test will skip and not fail.
"""

PIG_LATIN_EXAMPLES = [
    {"input": "banana split", "output": "anana-bay plit-say"},
    {"input": "hello world", "output": "ello-hay orld-way"},
    {"input": "donut shop", "output": "onut-day op-shay"},
]

TEST_PROMPTS = [
    "English: banana split\nPig Latin:",
    "English: hello world\nPig Latin:",
    "English: donut shop\nPig Latin:",
]

REVERSE_EXAMPLES = [
    {"input": "banana split", "output": "ananab tilps"},
    {"input": "hello world", "output": "olleh dlrow"},
    {"input": "donut shop", "output": "tunod pohs"},
    {"input": "deep learning", "output": "peed gninrael"},
    {"input": "paper plane", "output": "repap enalp"},
]

REVERSE_PROMPTS = [
    "Reverse each word.\nEnglish: banana split\nReversed:",
    "Reverse each word.\nEnglish: hello world\nReversed:",
    "Reverse each word.\nEnglish: donut shop\nReversed:",
    "Reverse each word.\nEnglish: deep learning\nReversed:",
    "Reverse each word.\nEnglish: paper plane\nReversed:",
]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _log(message: str) -> None:
    print(message, flush=True)


def _run_with_seq_sync(training_client, action):
    while True:
        try:
            return action()
        except RequestFailedError as exc:
            match = re.search(r"Sequence conflict: expected (\d+), got (\d+)\.", str(exc))
            if not match:
                raise
            expected = int(match.group(1))
            training_client._request_id_counter = expected - 1


def _start_server(
    config: AppConfig, port: int
) -> tuple[uvicorn.Server, threading.Thread, str, httpx.Client]:
    app = create_root_app(config)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    client = httpx.Client()
    healthy = False
    for attempt in range(1, 121):
        try:
            response = client.get(f"{base_url}/api/v1/healthz", timeout=1)
            response.raise_for_status()
            healthy = True
            break
        except httpx.HTTPError:
            time.sleep(2)
        if attempt % 5 == 0:
            _log(f"Waiting for server healthz... attempt {attempt}/120")
    if not healthy:
        server.should_exit = True
        thread.join(timeout=5)
        client.close()
        raise RuntimeError("Server failed to start")
    _log("Server is healthy")
    return server, thread, base_url, client


def _stop_server(server: uvicorn.Server, thread: threading.Thread, client: httpx.Client) -> None:
    server.should_exit = True
    thread.join(timeout=5)
    client.close()


def _create_training_data(tokenizer) -> list[types.Datum]:
    data: list[types.Datum] = []
    for example in PIG_LATIN_EXAMPLES:
        prompt = f"English: {example['input']}\nPig Latin:"
        completion = f" {example['output']}\n"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

        tokens = prompt_tokens + completion_tokens
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)

        datum = types.Datum(
            model_input=types.ModelInput.from_ints(input_tokens),
            loss_fn_inputs={
                "target_tokens": types.TensorData(
                    data=target_tokens,
                    dtype="int64",
                    shape=[len(target_tokens)],
                ),
                "weights": types.TensorData(
                    data=weights,
                    dtype="float32",
                    shape=[len(weights)],
                ),
            },
        )
        data.append(datum)
    return data


def _create_reverse_training_data(tokenizer) -> list[types.Datum]:
    data: list[types.Datum] = []
    for example in REVERSE_EXAMPLES:
        prompt = f"Reverse each word.\nEnglish: {example['input']}\nReversed:"
        completion = f" {example['output']}\n"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

        tokens = prompt_tokens + completion_tokens
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)

        datum = types.Datum(
            model_input=types.ModelInput.from_ints(input_tokens),
            loss_fn_inputs={
                "target_tokens": types.TensorData(
                    data=target_tokens,
                    dtype="int64",
                    shape=[len(target_tokens)],
                ),
                "weights": types.TensorData(
                    data=weights,
                    dtype="float32",
                    shape=[len(weights)],
                ),
            },
        )
        data.append(datum)
    return data


@pytest.fixture(scope="module")
def server_endpoint(tmp_path_factory: pytest.TempPathFactory):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")

    model_envs = ["TUFT_TEST_MODEL_1", "TUFT_TEST_MODEL_2"]

    models = []
    for env in model_envs:
        if env not in os.environ:
            warnings.warn(
                f"Skipping GPU integration test because {env} is not set.",
                RuntimeWarning,
                stacklevel=2,
            )
            pytest.skip(f"{env} is not set, skipping GPU integration test")
        models.append(Path(os.environ[env]))

    _log(f"Using model path: {models}")

    _log("Starting Ray...")
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"TRANSFORMERS_NO_TORCHVISION": "1"}},
    )
    checkpoint_dir = tmp_path_factory.mktemp("checkpoints")
    _log(f"Checkpoint dir: {checkpoint_dir}")
    config = AppConfig(checkpoint_dir=Path(checkpoint_dir))
    config.supported_models = [
        ModelConfig(
            model_name="Qwen/Qwen3-0.6B",
            model_path=models[0],
            max_model_len=4096,
            tensor_parallel_size=1,
        ),
        ModelConfig(
            model_name="Qwen/Qwen3-1.7B",
            model_path=models[1],
            max_model_len=4096,
            tensor_parallel_size=1,
        ),
    ]
    config.authorized_users = {
        "tml-test-key": "default",
    }
    _log("Creating FastAPI app...")
    app = create_root_app(config)
    port = _find_free_port()
    _log(f"Starting server on port {port}...")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    client = httpx.Client()
    healthy = False
    for attempt in range(1, 121):
        try:
            response = client.get(f"{base_url}/api/v1/healthz", timeout=1)
            response.raise_for_status()
            healthy = True
            break
        except httpx.HTTPError:
            time.sleep(2)
        if attempt % 5 == 0:
            _log(f"Waiting for server healthz... attempt {attempt}/120")
    if not healthy:
        server.should_exit = True
        thread.join(timeout=5)
        client.close()
        raise RuntimeError("Server failed to start")
    _log("Server is healthy")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)
    client.close()
    ray.shutdown(_exiting_interpreter=True)


@pytest.mark.integration
@pytest.mark.gpu
def test_auth_and_pig_latin_training_flow(server_endpoint: str) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU integration test")
    http_client = httpx.Client()
    try:
        _log(f"Server endpoint: {server_endpoint}")
        missing_key = http_client.post(f"{server_endpoint}/api/v1/create_session", json={})
        assert missing_key.status_code == 401

        invalid_key = http_client.post(
            f"{server_endpoint}/api/v1/create_session",
            headers={"X-API-Key": "invalid-key"},
            json={},
        )
        assert invalid_key.status_code == 403
    finally:
        http_client.close()

    service_client = ServiceClient(
        api_key="tml-test-key",  # pragma: allowlist secret
        base_url=server_endpoint,
        timeout=120,
    )
    # here we assume the model has the same tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.environ["TUFT_TEST_MODEL"])
    try:
        capabilities = service_client.get_server_capabilities()
        assert capabilities.supported_models, "server did not report supported models"
        assert len(capabilities.supported_models) == 2, "expected 2 supported models"
        supported_model_names = [m.model_name for m in capabilities.supported_models]
        base_model_1 = "Qwen/Qwen3-0.6B"
        base_model_2 = "Qwen/Qwen3-1.7B"
        assert base_model_1 in supported_model_names, f"{base_model_1} not reported as supported"
        assert base_model_2 in supported_model_names, f"{base_model_2} not reported as supported"

        _log(f"Base model: {base_model_1}")

        _log("Creating LoRA training client...")
        training_client_1 = service_client.create_lora_training_client(
            base_model=base_model_1, rank=8
        )
        training_client_2 = service_client.create_lora_training_client(
            base_model=base_model_1, rank=16
        )
        training_client_3 = service_client.create_lora_training_client(
            base_model=base_model_2, rank=16
        )
        training_clients = [training_client_1, training_client_2, training_client_3]
        train_data = _create_training_data(tokenizer)
        _log(f"Training samples: {len(train_data)}")

        for epoch in range(1, 21):
            if epoch == 1:
                _log("Running training loop...")
            for training_client in training_clients:
                training_client.forward_backward(train_data, "cross_entropy").result(timeout=60)
            for training_client in training_clients:
                training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=60)
            if epoch % 5 == 0:
                _log(f"Training progress: epoch {epoch}/20")
        _log("Training complete")

        sampling_clients = []

        for i, training_client in enumerate(training_clients, start=1):
            _log(f"Saving sampler weights for training client {i}...")
            sampler_response = training_client.save_weights_for_sampler(
                f"sampler-client-{i}"
            ).result(timeout=60)
            assert sampler_response.path.startswith("tinker://")
            _log(f"Sampler path for client {i}: {sampler_response.path}")
            sampling_clients.append(
                service_client.create_sampling_client(model_path=sampler_response.path)
            )

        _log("Running sampling assertions...")
        for prompt_text, example in zip(TEST_PROMPTS, PIG_LATIN_EXAMPLES, strict=True):
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            for sampling_client in sampling_clients:
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
                _log(f"Prompt: {prompt_text!r}")
                _log(f"Output: {output_text!r}")
                assert _normalize_text(output_text) == _normalize_text(example["output"])
    finally:
        service_client.holder.close()


@pytest.mark.integration
@pytest.mark.gpu
def test_multi_lora_adapters(server_endpoint: str) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU integration test")
    service_client = ServiceClient(
        api_key="tml-test-key",  # pragma: allowlist secret
        base_url=server_endpoint,
        timeout=120,
    )
    tokenizer = AutoTokenizer.from_pretrained(os.environ["TUFT_TEST_MODEL"])
    try:
        capabilities = service_client.get_server_capabilities()
        assert capabilities.supported_models, "server did not report supported models"
        base_model = capabilities.supported_models[0].model_name or "Qwen/Qwen3-0.6B"
        _log(f"Base model: {base_model}")

        _log("Training LoRA A (Pig Latin)...")
        training_client_a = service_client.create_lora_training_client(
            base_model=base_model, rank=8
        )
        pig_latin_data = _create_training_data(tokenizer)
        _log("Training LoRA B (Reverse Words)...")
        training_client_b = service_client.create_lora_training_client(
            base_model=base_model, rank=8
        )
        reverse_data = _create_reverse_training_data(tokenizer)

        _log("Running interleaved training loop...")
        for epoch in range(1, 31):
            training_client_a.forward_backward(pig_latin_data, "cross_entropy").result(timeout=60)
            training_client_a.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=60)
            training_client_b.forward_backward(reverse_data, "cross_entropy").result(timeout=60)
            training_client_b.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=60)
            if epoch % 5 == 0:
                _log(f"Interleaved progress: epoch {epoch}/30")
        _log("Interleaved training complete")

        sampler_a = training_client_a.save_weights_for_sampler("sampler-pig-latin-a").result(
            timeout=60
        )
        assert sampler_a.path.startswith("tinker://")
        _log(f"Sampler A path: {sampler_a.path}")

        sampler_b = training_client_b.save_weights_for_sampler("sampler-reverse-b").result(
            timeout=60
        )
        assert sampler_b.path.startswith("tinker://")
        _log(f"Sampler B path: {sampler_b.path}")

        sampling_client_a = service_client.create_sampling_client(model_path=sampler_a.path)
        sampling_client_b = service_client.create_sampling_client(model_path=sampler_b.path)

        _log("Validating LoRA A (Pig Latin) outputs...")
        for prompt_text, example in zip(TEST_PROMPTS, PIG_LATIN_EXAMPLES, strict=True):
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            sample_res = sampling_client_a.sample(
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
            output_text = tokenizer.decode(sample_res.sequences[0].tokens, skip_special_tokens=True)
            _log(f"LoRA A prompt: {prompt_text!r}")
            _log(f"LoRA A output: {output_text!r}")
            assert _normalize_text(output_text) == _normalize_text(example["output"])

        _log("Validating LoRA B (Reverse Words) outputs...")
        for prompt_text, example in zip(REVERSE_PROMPTS, REVERSE_EXAMPLES, strict=True):
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            sample_res = sampling_client_b.sample(
                prompt=types.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=32,
                    temperature=0.0,
                    top_p=1.0,
                    stop=["\n"],
                ),
            ).result(timeout=60)
            assert sample_res.sequences and sample_res.sequences[0].tokens
            output_text = tokenizer.decode(sample_res.sequences[0].tokens, skip_special_tokens=True)
            _log(f"LoRA B prompt: {prompt_text!r}")
            _log(f"LoRA B output: {output_text!r}")
            assert _normalize_text(output_text) == _normalize_text(example["output"])

        _log("Validating LoRA A/B separation...")
        cross_prompt = "Reverse each word.\nEnglish: hello world\nReversed:"
        cross_tokens = tokenizer.encode(cross_prompt, add_special_tokens=True)
        cross_res_a = sampling_client_a.sample(
            prompt=types.ModelInput.from_ints(cross_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=32,
                temperature=0.0,
                top_p=1.0,
                stop=["\n"],
            ),
        ).result(timeout=60)
        assert cross_res_a.sequences and cross_res_a.sequences[0].tokens
        cross_text_a = tokenizer.decode(cross_res_a.sequences[0].tokens, skip_special_tokens=True)
        _log(f"LoRA A on Reverse prompt output: {cross_text_a!r}")
        assert _normalize_text(cross_text_a) != _normalize_text("olleh dlrow")
    finally:
        service_client.holder.close()
        ray.shutdown(_exiting_interpreter=True)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.persistence
def test_checkpoint_resume_persistence(tmp_path: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU integration test")

    model_env = os.environ.get("TUFT_TEST_MODEL")
    if not model_env:
        warnings.warn(
            "Skipping GPU integration test because TUFT_TEST_MODEL is not set.",
            RuntimeWarning,
            stacklevel=2,
        )
        pytest.skip("TUFT_TEST_MODEL is not set, skipping GPU integration test")

    file_redis_path = tmp_path / "file_redis.json"
    if file_redis_path.exists():
        file_redis_path.unlink()
    _log(f"FileRedis path: {file_redis_path}")

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")

    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"TRANSFORMERS_NO_TORCHVISION": "1"}},
    )
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config = AppConfig(checkpoint_dir=checkpoint_dir)
    config.supported_models = [
        ModelConfig(
            model_name="Qwen/Qwen3-0.6B",
            model_path=Path(model_env),
            max_model_len=4096,
            tensor_parallel_size=1,
        )
    ]
    config.authorized_users = {
        "tml-test-key": "default",
    }
    config.persistence = PersistenceConfig.from_file_redis(
        file_path=file_redis_path,
        namespace="tuft_test",
    )

    port = _find_free_port()
    server = thread = client = None
    service_client = None
    try:
        _log("Starting server...")
        server, thread, base_url, client = _start_server(config, port)

        service_client = ServiceClient(api_key="tml-test-key", base_url=base_url, timeout=120)
        tokenizer = AutoTokenizer.from_pretrained(model_env)

        capabilities = service_client.get_server_capabilities()
        assert capabilities.supported_models, "server did not report supported models"
        base_model = capabilities.supported_models[0].model_name or "Qwen/Qwen3-0.6B"
        _log(f"Base model: {base_model}")

        training_client = service_client.create_lora_training_client(base_model=base_model, rank=2)
        train_data = _create_training_data(tokenizer)
        _log("Running training loop...")
        training_client.forward_backward(train_data, "cross_entropy").result(timeout=60)
        training_client.optim_step(types.AdamParams(learning_rate=5e-4)).result(timeout=60)
        _log("Training complete")

        checkpoint_name = "persistence-ckpt"
        checkpoint = training_client.save_state(checkpoint_name).result(timeout=60)
        checkpoint_path = checkpoint.path
        assert checkpoint_path.startswith("tinker://")
        _log(f"Checkpoint path: {checkpoint_path}")

        sampler_response = training_client.save_weights_for_sampler("persistence-sampler").result(
            timeout=60
        )
        sampler_path = sampler_response.path
        assert sampler_path.startswith("tinker://")
        _log(f"Sampler path: {sampler_path}")
        sampling_client = service_client.create_sampling_client(model_path=sampler_path)
        sample_res = sampling_client.sample(
            prompt=types.ModelInput.from_ints(
                tokenizer.encode(TEST_PROMPTS[0], add_special_tokens=True)
            ),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=4, temperature=0.1, top_p=1.0),
        ).result(timeout=60)
        assert sample_res.sequences and sample_res.sequences[0].tokens
        _log(f"Checkpoint dir contents: {list(checkpoint_dir.rglob('*'))}")
        _log(
            f"FileRedis exists after save: {file_redis_path.exists()} size="
            f"{file_redis_path.stat().st_size if file_redis_path.exists() else 'n/a'}"
        )

        session_id = service_client.holder.get_session_id()
        rest_client = service_client.create_rest_client()
        session_before = rest_client.get_session(session_id).result(timeout=30)
        checkpoints_before = rest_client.list_checkpoints(training_client.model_id).result(
            timeout=30
        )
        checkpoint_ids_before = [c.checkpoint_id for c in checkpoints_before.checkpoints]
        assert checkpoint_name in checkpoint_ids_before

        _log("Restarting server...")
        _stop_server(server, thread, client)
        ray.shutdown(_exiting_interpreter=True)
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"TRANSFORMERS_NO_TORCHVISION": "1"}},
        )
        server, thread, base_url, client = _start_server(config, port)

        # Create a new service_client with the new server URL
        service_client = ServiceClient(
            api_key="tml-test-key",  # pragma: allowlist secret
            base_url=base_url,
            timeout=120,
        )
        rest_client = service_client.create_rest_client()
        sessions = rest_client.list_sessions().result(timeout=30)
        assert session_id in sessions.sessions

        session_after = rest_client.get_session(session_id).result(timeout=30)
        assert session_after.training_run_ids == session_before.training_run_ids
        assert session_after.sampler_ids == session_before.sampler_ids

        checkpoints_after = rest_client.list_checkpoints(training_client.model_id).result(
            timeout=30
        )
        checkpoint_ids_after = [c.checkpoint_id for c in checkpoints_after.checkpoints]
        assert checkpoint_name in checkpoint_ids_after

        # Use existing training_client to load state from checkpoint
        _log("Loading checkpoint into existing training client...")
        training_client.load_state_with_optimizer(checkpoint_path).result(timeout=60)
        _log(
            "Training client state loaded from checkpoint. "
            f"turn={training_client._turn_counter} "
            f"next_request={training_client._request_id_counter}"
        )
        training_client.forward_backward(train_data, "cross_entropy").result(timeout=60)
        training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=60)
        resumed_weights = training_client.save_weights_for_sampler("resume-sampler").result(
            timeout=60
        )
        assert resumed_weights.path.startswith("tinker://")
        resumed_sampling = service_client.create_sampling_client(model_path=resumed_weights.path)
        resumed_res = resumed_sampling.sample(
            prompt=types.ModelInput.from_ints(
                tokenizer.encode(TEST_PROMPTS[1], add_special_tokens=True)
            ),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=4, temperature=0.1, top_p=1.0),
        ).result(timeout=60)
        assert resumed_res.sequences and resumed_res.sequences[0].tokens
    finally:
        if service_client is not None:
            service_client.holder.close()
        if server and thread and client:
            _stop_server(server, thread, client)
        ray.shutdown(_exiting_interpreter=True)
        if file_redis_path.exists():
            file_redis_path.unlink()
