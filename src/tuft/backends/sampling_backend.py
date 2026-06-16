"""Sampling backend implementated using vLLM"""

import asyncio
import json
import os
import socket
import threading
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Optional

from opentelemetry.trace import StatusCode
from tinker import types

from ..config import ModelConfig
from ..telemetry.tracing import get_tracer
from .base_backend import BaseSamplingBackend


_get_tracer = lambda: get_tracer("tuft.sampling_backend")  # noqa: E731


logger = getLogger(__name__)


def _build_sample_response(
    req_output: Any,
    include_prompt_logprobs: bool = False,
    topk_prompt_logprobs: int = 0,
) -> types.SampleResponse:
    """Build a tinker 0.18.2 SampleResponse from vLLM RequestOutput.

    WHY THIS EXISTS:
      trinity-rft 0.5.1 declares ``tinker>=0.10.0`` as a dependency but its
      ``vLLMRolloutModel.sample()`` constructs ``SampledSequence`` and
      ``SampleResponse`` using the old tinker 0.7 keyword arguments
      (``tokens=``, ``logprobs=``, ``prompt_logprobs=``,
      ``topk_prompt_logprobs=``).  In tinker 0.18.2 these types are frozen
      dataclasses and the old names are no longer valid constructor parameters,
      causing ``TypeError: SampledSequence.__init__() got an unexpected keyword
      argument 'tokens'``.

      Because the trinity model runs as a **Ray remote actor in a separate
      process**, we cannot monkey-patch tinker's constructors from our main
      process.  The only self-contained workaround is to bypass trinity's
      ``sample()`` entirely, call its lower-level ``_generate_internal()``
      (which returns the raw vLLM ``RequestOutput`` without touching tinker
      types), and build the ``SampleResponse`` ourselves here using the new
      tinker 0.18.2 constructor API.

    HOW TO REVERT WHEN TRINITY IS FIXED:
      The recommended first step is to upgrade trinity-rft to its latest
      version (``pip install --upgrade trinity-rft``).  If the new version
      constructs ``SampledSequence`` / ``SampleResponse`` with tinker
      0.18.2-compatible keyword arguments (``_tokens_list=``,
      ``_logprobs_list=``, etc.), then do the following:

      1. In ``VLLMSamplingBackend.sample()``, replace the call to
         ``engine._generate_internal.remote()`` + ``_build_sample_response()``
         with a direct call to ``engine.sample.remote()``.
      2. Delete this ``_build_sample_response()`` function.
      3. Optionally delete ``_normalize_sample_response()`` if no longer needed.
      4. Remove the ``skip_reading_prefix_cache`` workaround in
         ``VLLMSamplingBackend.sample()`` (trinity handles it internally).

    The logic below mirrors trinity's ``vllm_model.py::sample()`` but uses
    the new constructor API (``_tokens_list=``, ``_logprobs_list=``, etc.).
    """
    sequences: list[types.SampledSequence] = []
    topk_prompt_logprobs_list: list[list[tuple[int, float]] | None] = [None]
    prompt_logprobs: list[float | None] = [None]

    # collect prompt logprobs
    if include_prompt_logprobs:
        for logprob_dict in req_output.prompt_logprobs[1:]:
            prompt_logprobs.append(next(iter(logprob_dict.values())).logprob)
            if topk_prompt_logprobs > 0:
                logprob_items = sorted(logprob_dict.items(), key=lambda x: x[1].rank)
                topk = logprob_items[:topk_prompt_logprobs]
                topk_prompt_logprobs_list.append(
                    [(token_id, logprob.logprob) for token_id, logprob in topk]
                )

    # collect response sequences
    for seq_output in req_output.outputs:
        seq = types.SampledSequence(
            stop_reason="length" if seq_output.finish_reason == "length" else "stop",
            _tokens_list=seq_output.token_ids,
            _logprobs_list=[
                next(iter(logprob_dict.values())).logprob for logprob_dict in seq_output.logprobs
            ],
        )
        sequences.append(seq)

    return types.SampleResponse(
        sequences=sequences,
        _prompt_logprobs_list=prompt_logprobs if include_prompt_logprobs else None,
        _topk_prompt_logprobs_list=(
            topk_prompt_logprobs_list
            if include_prompt_logprobs and topk_prompt_logprobs > 0
            else None
        ),
    )


def _normalize_sample_response(raw: Any) -> types.SampleResponse:
    """Normalize engine sample response to tinker 0.18.2 SampleResponse dataclass.

    Handles responses from engines that may use older tinker versions:
    - If already a SampleResponse dataclass: pass through
    - If dict (JSON-like): construct from dict fields
    - If Pydantic-like object with .sequences attribute: extract and convert
    """
    if isinstance(raw, types.SampleResponse):
        return raw

    # Handle dict response (e.g., from JSON serialization)
    if isinstance(raw, dict):
        sequences = []
        for seq_data in raw.get("sequences", []):
            if isinstance(seq_data, dict):
                sequences.append(
                    types.SampledSequence(
                        stop_reason=seq_data["stop_reason"],
                        _tokens_list=seq_data.get("tokens", []),
                        _logprobs_list=seq_data.get("logprobs"),
                    )
                )
            else:
                # Already a SampledSequence-like object
                sequences.append(
                    types.SampledSequence(
                        stop_reason=seq_data.stop_reason,
                        _tokens_list=list(seq_data.tokens) if hasattr(seq_data, "tokens") else [],
                        _logprobs_list=list(seq_data.logprobs)
                        if hasattr(seq_data, "logprobs") and seq_data.logprobs is not None
                        else None,
                    )
                )
        return types.SampleResponse(
            sequences=sequences,
            _prompt_logprobs_list=raw.get("prompt_logprobs"),
            _topk_prompt_logprobs_list=raw.get("topk_prompt_logprobs"),
        )

    # Handle old Pydantic-like object (has .sequences attribute)
    if hasattr(raw, "sequences"):
        sequences = []
        for seq in raw.sequences:
            tokens = list(seq.tokens) if hasattr(seq, "tokens") else []
            logprobs = (
                list(seq.logprobs)
                if hasattr(seq, "logprobs") and seq.logprobs is not None
                else None
            )
            sequences.append(
                types.SampledSequence(
                    stop_reason=seq.stop_reason,
                    _tokens_list=tokens,
                    _logprobs_list=logprobs,
                )
            )
        prompt_lp = getattr(raw, "prompt_logprobs", None)
        topk_lp = getattr(raw, "topk_prompt_logprobs", None)
        return types.SampleResponse(
            sequences=sequences,
            _prompt_logprobs_list=prompt_lp,
            _topk_prompt_logprobs_list=topk_lp,
        )

    raise TypeError(f"Cannot normalize sample response of type {type(raw)}")


class VLLMSamplingBackend(BaseSamplingBackend):
    """A sampling backend using vLLM.

    User side `sample`, `sample_async`, `compute_logprobs` and
    `compute_logprobs_async` are all supported by the sample method.
    """

    def __init__(
        self,
        config: ModelConfig,
        worker_venv_path: Optional[str] = None,
        *,
        instance_index: int = 0,
    ) -> None:
        from vllm.lora.request import LoRARequest

        super().__init__(config)
        self._worker_venv_path = worker_venv_path
        self._instance_index = instance_index
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

    def _build_inference_model_config(self, config: ModelConfig, **extra_kwargs):
        from trinity.common.config import InferenceModelConfig

        return InferenceModelConfig(
            model_path=str(config.model_path),
            tensor_parallel_size=extra_kwargs.pop("tensor_parallel_size"),
            max_model_len=(
                config.sampling_max_model_len
                if config.sampling_max_model_len is not None
                else config.max_model_len
            ),
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            logprobs=config.logprobs,
            min_response_tokens=config.min_response_tokens,
            repetition_penalty=1.0,
            enable_lora=True,
            enable_runtime_lora_updating=True,
            enable_openai_api=True,
            enable_auto_tool_choice=config.enable_auto_tool_choice,
            tool_call_parser=config.tool_call_parser,
            reasoning_parser=config.reasoning_parser,
            lora_kwargs={
                "max_lora_rank": config.max_lora_rank,
                "max_loras": config.max_loras,
                **({"quantization": config.quantization} if config.quantization else {}),
            },
            gpu_memory_utilization=extra_kwargs.pop(
                "gpu_memory_utilization", config.sampling_memory_fraction
            ),
            **extra_kwargs,
        )

    def _create_colocated_engine(self, config: ModelConfig):
        import ray
        from trinity.common.models.vllm_model import vLLMRolloutModel

        if not self._worker_venv_path or not self._worker_venv_path.strip():
            _runtime_env = {}
        else:
            _path = os.environ.get("PATH", "")
            _venv_python = str(Path(self._worker_venv_path) / "bin" / "python")
            _runtime_env = {
                "py_executable": _venv_python,
                "env_vars": {
                    "VIRTUAL_ENV": self._worker_venv_path,
                    "PATH": f"{self._worker_venv_path}/bin:{_path}",
                },
            }
        return (
            ray.remote(vLLMRolloutModel)
            .options(
                name="sampling_model_" + self.base_model,
                num_gpus=config.sampling_memory_fraction,
                runtime_env=_runtime_env,
            )
            .remote(
                config=self._build_inference_model_config(
                    config,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=config.sampling_memory_fraction,
                )
            )
        )

    def _create_standalone_engine(self, config: ModelConfig):
        import ray
        from trinity.common.models.vllm_model import vLLMRolloutModel

        # Assign tensor_parallel_size GPUs to the actor itself
        # so that Ray populates CUDA_VISIBLE_DEVICES correctly.  vLLM then
        # creates its own placement group inside the EngineCore process where
        # the GPUs are visible.
        num_gpus = config.tensor_parallel_size
        bundle_indices = ",".join(str(i) for i in range(config.tensor_parallel_size))

        if not self._worker_venv_path or not self._worker_venv_path.strip():
            _runtime_env = {}
        else:
            _path = os.environ.get("PATH", "")
            _venv_python = str(Path(self._worker_venv_path) / "bin" / "python")
            _runtime_env = {
                "py_executable": _venv_python,
                "env_vars": {
                    "VIRTUAL_ENV": self._worker_venv_path,
                    "PATH": f"{self._worker_venv_path}/bin:{_path}",
                },
            }

        # Use instance_index to differentiate Ray actor names for DP replicas
        actor_name = f"sampling_model_{self.base_model}"
        if self._instance_index > 0:
            actor_name = f"{actor_name}_dp{self._instance_index}"

        # In standalone/DP mode, each vLLM instance has a dedicated GPU.
        # Use 0.9 (vLLM default) for max KV cache, not sampling_memory_fraction
        # which is designed for colocate mode (shared GPU with training).
        standalone_gpu_memory_utilization = 0.9

        return (
            ray.remote(vLLMRolloutModel)
            .options(
                name=actor_name,
                num_gpus=num_gpus,
                runtime_env=_runtime_env,
            )
            .remote(
                config=self._build_inference_model_config(
                    config,
                    tensor_parallel_size=config.tensor_parallel_size,
                    bundle_indices=bundle_indices,
                    gpu_memory_utilization=standalone_gpu_memory_utilization,
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

                # -----------------------------------------------------------------
                # WORKAROUND: bypass trinity's engine.sample.remote()
                #
                # trinity-rft 0.5.1 uses old tinker 0.7 constructor keywords
                # (tokens=, logprobs=) inside its sample() method, which crash
                # with tinker 0.18.2 frozen dataclasses.  The actor runs in a
                # separate Ray worker process, so monkey-patching from here
                # won't help.
                #
                # Instead we call the lower-level _generate_internal() which
                # returns raw vLLM RequestOutput, then build SampleResponse
                # ourselves via _build_sample_response() using the new API.
                #
                # TODO(trinity): First try upgrading trinity-rft to latest
                # (pip install --upgrade trinity-rft). If the new version is
                # compatible with tinker 0.18.2, replace this block with:
                #   raw_response = await self.engine.sample.remote(
                #       prompt=prompt,
                #       num_samples=num_samples,
                #       sampling_params=sampling_params,
                #       include_prompt_logprobs=include_prompt_logprobs,
                #       topk_prompt_logprobs=topk_prompt_logprobs,
                #       lora_request=lora_request,
                #   )
                #   return _normalize_sample_response(raw_response)
                # -----------------------------------------------------------------
                prompt_token_ids = prompt.to_ints()
                params = {
                    "max_tokens": (
                        sampling_params.max_tokens if sampling_params.max_tokens is not None else 16
                    ),
                    "seed": sampling_params.seed,
                    "top_k": sampling_params.top_k,
                    "top_p": sampling_params.top_p,
                    "temperature": sampling_params.temperature,
                    "n": num_samples,
                    "prompt_logprobs": (topk_prompt_logprobs if include_prompt_logprobs else None),
                    "logprobs": 0,
                }
                # Avoid prefix cache corruption when computing prompt logprobs.
                # Trinity sets this for vLLM >= 0.12.0 to prevent OverflowError
                # in vLLM's _update_prompt_logprobs when prefix cache is active.
                if include_prompt_logprobs:
                    params["skip_reading_prefix_cache"] = True
                if sampling_params.stop is not None:
                    params["stop"] = sampling_params.stop

                # Ray @ray.remote decorator adds .remote() method dynamically
                req_output = await self.engine._generate_internal.remote(  # type: ignore[attr-defined]
                    prompt={"prompt_token_ids": prompt_token_ids},
                    lora_request=lora_request,
                    **params,
                )
                return _build_sample_response(
                    req_output=req_output,
                    include_prompt_logprobs=include_prompt_logprobs,
                    topk_prompt_logprobs=topk_prompt_logprobs,
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


class DPSamplingBackend(BaseSamplingBackend):
    """Data-Parallel sampling backend: N independent vLLM instances with round-robin LB.

    Each instance runs on its own GPU(s) (tensor_parallel_size per instance).
    Requests are distributed across instances using round-robin for uniform load.
    All instances share the same LoRA adapters.
    """

    def __init__(
        self,
        config: ModelConfig,
        worker_venv_path: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self._dp_size = config.data_parallel_size
        self._instances: list[VLLMSamplingBackend] = []
        for i in range(self._dp_size):
            instance = VLLMSamplingBackend(
                config, worker_venv_path=worker_venv_path, instance_index=i
            )
            self._instances.append(instance)
        # Atomic round-robin counter
        self._rr_counter = 0
        self._rr_lock = asyncio.Lock()
        logger.info(
            "DPSamplingBackend: created %d instances for model %s",
            self._dp_size,
            config.model_name,
        )

    def _next_instance(self) -> VLLMSamplingBackend:
        """Round-robin selection (no lock needed for simple modular arithmetic)."""
        idx = self._rr_counter % self._dp_size
        self._rr_counter += 1
        return self._instances[idx]

    async def async_init(self) -> None:
        """Initialize all DP instances in parallel."""
        await asyncio.gather(*[inst.async_init() for inst in self._instances])
        logger.info(
            "DPSamplingBackend: all %d instances initialized for model %s",
            self._dp_size,
            self.base_model,
        )

    def get_openai_api_url(self) -> Optional[str]:
        """Return a single URL (first instance) for backward compat.

        For DP-aware routing, use get_openai_api_urls() instead.
        """
        return self._instances[0].get_openai_api_url() if self._instances else None

    def get_openai_api_urls(self) -> list[str]:
        """Return all vLLM instance URLs for DP-aware load balancing."""
        urls = []
        for inst in self._instances:
            url = inst.get_openai_api_url()
            if url:
                urls.append(url)
        return urls

    def get_next_openai_api_url(self) -> Optional[str]:
        """Return the next URL via round-robin for load-balanced proxying."""
        inst = self._next_instance()
        return inst.get_openai_api_url()

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        """Route sampling to a DP instance via round-robin."""
        instance = self._next_instance()
        return await instance.sample(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
            lora_id=lora_id,
        )

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        """Add LoRA adapter to ALL DP instances."""
        await asyncio.gather(
            *[inst.add_adapter(lora_id, adapter_path) for inst in self._instances]
        )

    async def remove_adapter(self, lora_id: str) -> None:
        """Remove LoRA adapter from ALL DP instances."""
        await asyncio.gather(
            *[inst.remove_adapter(lora_id) for inst in self._instances]
        )


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
                _tokens_list=generated,
                _logprobs_list=[-0.3 for _ in generated],
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
            _prompt_logprobs_list=prompt_logprobs,
            _topk_prompt_logprobs_list=topk_prompt,
        )

    def _generate_tokens(self, prompt_tokens: list[int], max_tokens: int) -> list[int]:
        start = prompt_tokens[-1] if prompt_tokens else (abs(self.config.seed) % 32000) + 1
        return [(start + i) % 32000 for i in range(1, max_tokens + 1)]

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        self.lora_adapters[lora_id] = adapter_path

    async def remove_adapter(self, lora_id: str) -> None:
        if lora_id in self.lora_adapters:
            del self.lora_adapters[lora_id]
