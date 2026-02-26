"""Sampling backend implementated using vLLM"""

import asyncio
import json
import socket
import threading
import time
from logging import getLogger
from pathlib import Path
from typing import Optional

from opentelemetry.trace import StatusCode
from tinker import types

from ..config import ModelConfig
from ..telemetry.tracing import get_tracer
from .base_backend import BaseSamplingBackend


_get_tracer = lambda: get_tracer("tuft.sampling_backend")  # noqa: E731


logger = getLogger(__name__)


class VLLMSamplingBackend(BaseSamplingBackend):
    """A sampling backend using vLLM.

    User side `sample`, `sample_async`, `compute_logprobs` and
    `compute_logprobs_async` are all supported by the sample method.
    """

    def __init__(self, config: ModelConfig) -> None:
        from vllm.lora.request import LoRARequest

        super().__init__(config)
        self.engine = self._create_engine(config)
        self.lora_adapters: dict[str, LoRARequest] = {}
        self._counter = 1
        self._lock = asyncio.Lock()
        self._openai_api_url: Optional[str] = None

    def _create_engine(self, config: ModelConfig):
        if config.colocate:
            return self._create_colocated_engine(config)
        else:
            return self._create_standalone_engine(config)

    def _create_colocated_engine(self, config: ModelConfig):
        import ray
        from trinity.common.config import InferenceModelConfig
        from trinity.common.models.vllm_model import vLLMRolloutModel

        return (
            ray.remote(vLLMRolloutModel)
            .options(
                name="sampling_model_" + self.base_model,
                num_gpus=config.sampling_memory_fraction,
            )
            .remote(
                config=InferenceModelConfig(
                    model_path=str(config.model_path),
                    tensor_parallel_size=1,
                    max_model_len=config.max_model_len,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    logprobs=config.logprobs,
                    min_response_tokens=config.min_response_tokens,
                    repetition_penalty=1.0,
                    enable_lora=True,
                    enable_runtime_lora_updating=True,
                    enable_openai_api=True,
                    lora_kwargs={
                        "max_lora_rank": config.max_lora_rank,
                        "max_loras": config.max_loras,
                    },
                    # sampling use less memory than training
                    gpu_memory_utilization=config.sampling_memory_fraction,
                )
            )
        )

    def _create_standalone_engine(self, config: ModelConfig):
        import ray
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        from trinity.common.config import InferenceModelConfig
        from trinity.common.models.vllm_model import vLLMRolloutModel

        # create a placement group for this model
        pg = placement_group(
            [{"CPU": 1, "GPU": 1} for _ in range(config.tensor_parallel_size)],
            strategy="PACK",
        )
        ray.get(pg.ready(), timeout=10)
        return (
            ray.remote(vLLMRolloutModel)
            .options(
                name="sampling_model_" + self.base_model,
                num_gpus=0 if config.tensor_parallel_size > 1 else 1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                ),
            )
            .remote(
                config=InferenceModelConfig(
                    model_path=str(config.model_path),
                    tensor_parallel_size=config.tensor_parallel_size,
                    max_model_len=config.max_model_len,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    logprobs=config.logprobs,
                    min_response_tokens=config.min_response_tokens,
                    repetition_penalty=1.0,
                    enable_lora=True,
                    enable_runtime_lora_updating=True,
                    enable_openai_api=True,
                    lora_kwargs={
                        "max_lora_rank": config.max_lora_rank,
                        "max_loras": config.max_loras,
                    },
                )
            )
        )

    async def async_init(self) -> None:
        """Initialize the backend for sampling."""
        # Ray @ray.remote decorator adds .remote() method dynamically
        await self.engine.prepare.remote()  # type: ignore[attr-defined]
        self._openai_api_url = await self.engine.get_api_server_url.remote()  # type: ignore[attr-defined]
        logger.info(
            f"SamplingBackend for model {self.base_model} initialized. "
            f"OpenAI API URL: {self._openai_api_url}"
        )
        # Wait for the OpenAI API server to be ready to accept connections
        if self._openai_api_url:
            await self._wait_for_openai_server(self._openai_api_url)

    async def _wait_for_openai_server(self, url: str, timeout: float = 120.0) -> None:
        """Poll the vLLM OpenAI server until it accepts connections."""
        import asyncio

        import httpx as _httpx

        deadline = asyncio.get_event_loop().time() + timeout
        check_url = f"{url}/v1/models"
        attempt = 0
        async with _httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() < deadline:
                attempt += 1
                try:
                    resp = await client.get(check_url, timeout=3.0)
                    if resp.status_code == 200:
                        logger.info(f"vLLM OpenAI server at {url} is ready (attempt {attempt})")
                        return
                except (_httpx.ConnectError, _httpx.TimeoutException, OSError):
                    pass
                if attempt % 10 == 0:
                    logger.info(f"Waiting for vLLM OpenAI server at {url}... attempt {attempt}")
                await asyncio.sleep(2.0)
        logger.warning(f"vLLM OpenAI server at {url} not ready after {timeout}s")

    def get_openai_api_url(self) -> Optional[str]:
        """Return the vLLM OpenAI API base URL."""
        return self._openai_api_url

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        """Sampling using vLLM engine."""
        with _get_tracer().start_as_current_span("sampling_backend.sample") as span:
            span.set_attribute("tuft.num_samples", num_samples)
            span.set_attribute("tuft.has_lora", lora_id is not None)
            try:
                async with self._lock:
                    if lora_id is not None and lora_id not in self.lora_adapters:
                        raise ValueError(f"LoRA adapter {lora_id} not found in backend.")
                    lora_request = self.lora_adapters[lora_id] if lora_id is not None else None
                # Ray @ray.remote decorator adds .remote() method dynamically
                return await self.engine.sample.remote(  # type: ignore[attr-defined]
                    prompt=prompt,
                    num_samples=num_samples,
                    sampling_params=sampling_params,
                    include_prompt_logprobs=include_prompt_logprobs,
                    topk_prompt_logprobs=topk_prompt_logprobs,
                    lora_request=lora_request,
                )
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR)
                raise

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        from vllm.lora.request import LoRARequest

        with _get_tracer().start_as_current_span("sampling_backend.add_adapter") as span:
            span.set_attribute("tuft.lora_id", lora_id)
            try:
                async with self._lock:
                    self._counter += 1
                    self.lora_adapters[lora_id] = LoRARequest(
                        lora_int_id=self._counter + 1,
                        lora_name=lora_id,
                        lora_path=str(adapter_path),
                    )
                    if not adapter_path.exists():
                        raise ValueError(f"LoRA adapter path {adapter_path} does not exist.")
                    await self.engine.add_lora_adapter.remote(self.lora_adapters[lora_id])  # type: ignore[attr-defined]
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR)
                raise

    async def remove_adapter(self, lora_id: str) -> None:
        with _get_tracer().start_as_current_span("sampling_backend.remove_adapter") as span:
            span.set_attribute("tuft.lora_id", lora_id)
            async with self._lock:
                if lora_id in self.lora_adapters:
                    await self.engine.remove_lora_adapter.remote(lora_id)  # type: ignore[attr-defined]
                    del self.lora_adapters[lora_id]


def _build_mock_openai_app(model_name: str):
    """Build a minimal FastAPI app that mimics vLLM's OpenAI-compatible endpoints."""
    import itertools

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    mock_app = FastAPI()
    _id_counter = itertools.count(1)

    def _make_completion_response(body: dict) -> dict:
        model = body.get("model", model_name)
        raw_prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 16)
        n = body.get("n", 1)
        logprobs_n = body.get("logprobs")
        stop = body.get("stop")

        # Handle batch prompts (list of strings)
        prompts = raw_prompt if isinstance(raw_prompt, list) else [raw_prompt]
        choices = []
        idx = 0
        for prompt in prompts:
            prompt_str = str(prompt)[:20]
            for _j in range(n):
                text = f" dummy completion output for '{prompt_str}'"
                finish = "length"
                # Respect stop sequences
                if stop:
                    for s in stop if isinstance(stop, list) else [stop]:
                        pos = text.find(s)
                        if pos != -1:
                            text = text[:pos]
                            finish = "stop"
                            break
                logprobs_data = None
                if logprobs_n is not None:
                    logprobs_data = {
                        "tokens": [" dummy"],
                        "token_logprobs": [-0.5],
                        "top_logprobs": [{"dummy": -0.5}] if logprobs_n > 0 else None,
                        "text_offset": [0],
                    }
                choices.append(
                    {
                        "index": idx,
                        "text": text,
                        "logprobs": logprobs_data,
                        "finish_reason": finish,
                    }
                )
                idx += 1
        prompt_tokens = sum(max(1, len(str(p).split())) for p in prompts)
        return {
            "id": f"cmpl-dummy-{next(_id_counter)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": max_tokens * len(choices),
                "total_tokens": prompt_tokens + max_tokens * len(choices),
            },
        }

    def _make_chat_response(body: dict) -> dict:
        model = body.get("model", model_name)
        n = body.get("n", 1)
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 16
        choices = []
        for i in range(n):
            choices.append(
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": "This is a dummy response from the mock OpenAI server.",
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            )
        return {
            "id": f"chatcmpl-dummy-{next(_id_counter)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": max_tokens,
                "total_tokens": 10 + max_tokens,
            },
        }

    async def _stream_chat_chunks(body: dict):
        model = body.get("model", model_name)
        chunk_id = f"chatcmpl-dummy-{next(_id_counter)}"
        # First chunk with role
        first = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(first)}\n\n"
        # Content chunks
        words = ["This", " is", " a", " dummy", " streamed", " response."]
        for word in words:
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        # Final chunk
        final = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": len(words),
                "total_tokens": 10 + len(words),
            },
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    async def _stream_completion_chunks(body: dict):
        model = body.get("model", model_name)
        chunk_id = f"cmpl-dummy-{next(_id_counter)}"
        words = [" dummy", " completion", " output"]
        for word in words:
            chunk = {
                "id": chunk_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "text": word, "logprobs": None, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        final = {
            "id": chunk_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "text": "", "logprobs": None, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": len(words),
                "total_tokens": 5 + len(words),
            },
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    @mock_app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        if body.get("stream"):
            return StreamingResponse(
                _stream_completion_chunks(body), media_type="text/event-stream"
            )
        return JSONResponse(_make_completion_response(body))

    @mock_app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        if body.get("stream"):
            return StreamingResponse(_stream_chat_chunks(body), media_type="text/event-stream")
        return JSONResponse(_make_chat_response(body))

    @mock_app.get("/v1/models")
    async def list_models():
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "dummy",
                    }
                ],
            }
        )

    return mock_app


class DummySamplingBackend(BaseSamplingBackend):
    """A dummy sampling backend that returns fixed responses for unittest."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.lora_adapters: dict[str, Path] = {}
        self._counter = 1
        self._lock = asyncio.Lock()
        self._openai_api_url: Optional[str] = None
        self._mock_server: object | None = None
        self._mock_thread: threading.Thread | None = None

    async def async_init(self) -> None:
        """Start an embedded mock OpenAI server for testing."""
        self._start_mock_openai_server()

    def _start_mock_openai_server(self) -> None:
        """Start a lightweight mock OpenAI-compatible HTTP server in a background thread."""
        import uvicorn

        mock_app = _build_mock_openai_app(str(self.config.model_path))

        # Find a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        server = uvicorn.Server(
            uvicorn.Config(mock_app, host="127.0.0.1", port=port, log_level="error")
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait for the server to become ready
        import httpx

        for _ in range(50):
            try:
                resp = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=0.5)
                if resp.status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)

        self._openai_api_url = f"http://127.0.0.1:{port}"
        self._mock_server = server
        self._mock_thread = thread
        logger.info(f"DummySamplingBackend mock OpenAI server started at {self._openai_api_url}")

    def get_openai_api_url(self) -> Optional[str]:
        """Return the mock OpenAI API base URL."""
        return self._openai_api_url

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        """Return a fixed dummy response."""
        prompt_tokens = prompt.to_ints()
        max_tokens = sampling_params.max_tokens or 16
        sequences: list[types.SampledSequence] = []
        for _ in range(num_samples):
            generated = self._generate_tokens(prompt_tokens, max_tokens)
            seq = types.SampledSequence(
                stop_reason="length",
                tokens=generated,
                logprobs=[-0.3 for _ in generated],
            )
            sequences.append(seq)
        prompt_logprobs = None
        topk_prompt = None
        if include_prompt_logprobs:
            prompt_logprobs = [-0.1 if tok is not None else None for tok in prompt_tokens]
        if topk_prompt_logprobs > 0:
            topk_prompt = [
                [
                    (token, round(-0.05 - idx * 0.01, 4))
                    for idx, token in enumerate(prompt_tokens[:topk_prompt_logprobs])
                ]
                if token is not None
                else None
                for token in prompt_tokens
            ]
        return types.SampleResponse(
            sequences=sequences,
            prompt_logprobs=prompt_logprobs,
            topk_prompt_logprobs=topk_prompt,
        )

    def _generate_tokens(self, prompt_tokens: list[int], max_tokens: int) -> list[int]:
        start = prompt_tokens[-1] if prompt_tokens else (abs(self.config.seed) % 32000) + 1
        return [(start + i) % 32000 for i in range(1, max_tokens + 1)]

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        self.lora_adapters[lora_id] = adapter_path

    async def remove_adapter(self, lora_id: str) -> None:
        if lora_id in self.lora_adapters:
            del self.lora_adapters[lora_id]
