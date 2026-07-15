"""Ray-actor wrapper around a vLLM async engine with an embedded OpenAI server.

This is TuFT's direct vLLM integration, replacing the previously used
``vLLMRolloutModel`` from trinity-rft (the behavior mirrors trinity 0.6.0 for
the vLLM range pinned in pyproject.toml; see
docs/sphinx_doc/source/development/vllm-backend.md for the full design and
maintenance notes). One actor instance owns:

- a ``vllm.AsyncLLMEngine`` created in ``prepare()``, used directly by the
  Tinker-compatible sampling path (``generate()`` returns raw vLLM
  ``RequestOutput``; the tinker response types are built by the caller in
  ``sampling_backend.py``), and
- vLLM's own OpenAI-compatible API server, started in-process against that
  same engine (see ``vllm_api_server.py``) so HTTP traffic proxied by
  ``tuft.oai`` shares the engine and its LoRA adapters.

This module intentionally imports vllm/ray only inside methods so it can be
imported on CPU-only machines (unit tests, config validation).
"""

import asyncio
import itertools
import os
import socket
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Optional


logger = getLogger(__name__)


@dataclass
class VLLMEngineConfig:
    """Engine-facing configuration, resolved from ``tuft.config.ModelConfig``.

    ``sampling_backend.py`` builds one of these per engine instance; colocate
    and standalone modes differ only in ``tensor_parallel_size``,
    ``gpu_memory_utilization`` and ``bundle_indices``.
    """

    model_path: str
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    seed: int = 42
    enforce_eager: bool = False

    # Default sampling parameters (per-request params override via clone).
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    logprobs: int = 0
    min_response_tokens: int = 0
    repetition_penalty: float = 1.0

    # LoRA
    enable_lora: bool = True
    max_lora_rank: int = 16
    max_loras: int = 1
    quantization: Optional[str] = None
    # Gates vLLM's /v1/load_lora_adapter HTTP endpoint (used by tuft.oai to
    # serve trained adapters by name through the OpenAI API).
    enable_runtime_lora_updating: bool = True

    # OpenAI-compatible API server
    enable_openai_api: bool = True
    enable_auto_tool_choice: bool = False
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    chat_template: Optional[str] = None
    enable_log_requests: bool = False

    # Ray placement-group bundle indices ("" outside standalone TP mode).
    bundle_indices: str = ""


class VLLMEngine:
    """vLLM engine + OpenAI API server, run as a Ray actor.

    Wrapped with ``ray.remote(VLLMEngine)`` by the sampling backend; all
    public methods are invoked via ``.remote()``.
    """

    def __init__(self, config: VLLMEngineConfig) -> None:
        import vllm
        from packaging.version import parse as parse_version
        from vllm.sampling_params import RequestOutputKind

        from .vllm_worker import get_vllm_version

        self.config = config
        self.vllm_version = get_vllm_version()

        # Engine environment. Mirrors the trinity-rft 0.6.0 setup that TuFT
        # previously ran on for vLLM 0.19-0.23; revisit when bumping vLLM.
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = config.bundle_indices
        if self.vllm_version >= parse_version("0.22.0"):
            # Stay on the V1 model runner for now.
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        os.environ["VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE"] = "shm"
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = "1"
        # Keep the engine core in-process so collective_rpc and fractional-GPU
        # colocation work from within the Ray actor.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
        if config.enable_runtime_lora_updating:
            os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"

        self.default_sampling_params = vllm.SamplingParams(
            n=1,
            temperature=config.temperature,
            max_tokens=None,
            min_tokens=config.min_response_tokens,
            skip_special_tokens=True,
            include_stop_str_in_output=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
            logprobs=config.logprobs,
            top_p=config.top_p,
            top_k=config.top_k,
        )

        self.async_llm: Any = None
        self.api_server_host: Optional[str] = None
        self.api_server_port: Optional[int] = None
        self._api_server_task: Optional[asyncio.Task] = None
        self._request_counter = itertools.count(1)
        self._prepared = False
        self._prepare_lock = asyncio.Lock()

    async def prepare(self) -> None:
        """Create the vLLM engine, apply worker patches, start the API server."""
        import vllm

        async with self._prepare_lock:
            if self._prepared:
                return

            override_generation_config = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repetition_penalty": self.config.repetition_penalty,
            }

            engine_args = vllm.AsyncEngineArgs(
                model=self.config.model_path,
                worker_cls="tuft.backends.vllm_worker.TuFTGPUWorker",
                tensor_parallel_size=self.config.tensor_parallel_size,
                seed=self.config.seed,
                distributed_executor_backend="mp",
                max_model_len=self.config.max_model_len,  # type: ignore[arg-type]
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                dtype=self.config.dtype,  # type: ignore[arg-type]
                trust_remote_code=True,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                enforce_eager=self.config.enforce_eager,
                override_generation_config=override_generation_config,
                reasoning_parser=self.config.reasoning_parser,  # type: ignore[arg-type]
                disable_log_stats=True,
                enable_log_requests=self.config.enable_log_requests,
                enable_lora=self.config.enable_lora,
                max_lora_rank=self.config.max_lora_rank,  # type: ignore[arg-type]
                max_loras=self.config.max_loras,
                # Return logprobs of the actual sampling distribution (after
                # temperature scaling) -- required for RL importance ratios.
                logprobs_mode="processed_logprobs",
                async_scheduling=True,
            )
            if self.config.quantization:
                engine_args.quantization = self.config.quantization

            self.async_llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
            # Apply TuFT's worker-side patches (prompt-logprobs temperature
            # scaling) on every worker process; see vllm_worker.py.
            await self.async_llm.collective_rpc("apply_patches")
            await self._run_api_server()
            self._prepared = True

    async def _run_api_server(self) -> None:
        """Start vLLM's OpenAI-compatible API server in this actor's event loop."""
        if not self.config.enable_openai_api:
            logger.info("OpenAI API server is not enabled. Skipping...")
            return
        if self._api_server_task is not None:
            return

        from .vllm_api_server import run_api_server

        host, port = self._get_available_address()
        self._api_server_task = asyncio.create_task(
            run_api_server(
                self.async_llm,
                host=host,
                port=port,
                model_path=self.config.model_path,
                logger=logger,
                chat_template=self.config.chat_template,
                enable_auto_tool_choice=self.config.enable_auto_tool_choice,
                tool_call_parser=self.config.tool_call_parser,
                reasoning_parser=self.config.reasoning_parser,
                enable_log_requests=self.config.enable_log_requests,
            )
        )
        self.api_server_host = host
        self.api_server_port = port

    @staticmethod
    def _get_available_address() -> tuple[str, int]:
        """Node IP plus an ephemeral port bound on this actor's node."""
        import ray

        address = ray.util.get_node_ip_address()
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        return address, port

    def get_api_server_url(self) -> Optional[str]:
        """URL of the embedded OpenAI API server (None if not enabled)."""
        if not self._prepared:
            raise RuntimeError("Engine is not prepared. Please call `prepare()` first.")
        if self.api_server_host is None or self.api_server_port is None:
            return None
        return f"http://{self.api_server_host}:{self.api_server_port}"

    async def generate(self, prompt: Any, lora_request: Any = None, **kwargs: Any) -> Any:
        """Generate and return the raw vLLM ``RequestOutput``.

        ``prompt`` is a vLLM prompt dict (e.g. ``{"prompt_token_ids": [...]}``);
        ``kwargs`` override fields of the default ``SamplingParams``.
        """
        stream = self.async_llm.generate(
            request_id=str(next(self._request_counter)),
            prompt=prompt,
            sampling_params=self._create_sampling_params(**kwargs),
            lora_request=lora_request,
        )
        async for request_output in stream:
            if request_output.finished:
                return request_output
        raise RuntimeError("[vLLM] The request is not finished. This should not happen.")

    def _create_sampling_params(self, **kwargs: Any):
        """Clone the default sampling params and apply per-request overrides.

        Unknown keys are dropped (hasattr guard) so version-dependent fields
        like ``skip_reading_prefix_cache`` degrade gracefully.
        """
        if len(kwargs) == 0:
            return self.default_sampling_params
        params = self.default_sampling_params.clone()
        for k, v in kwargs.items():
            if hasattr(params, k):
                setattr(params, k, v)
        return params

    async def add_lora(self, lora_request: Any) -> int:
        """Register a LoRA adapter with the engine (direct generate path)."""
        return await self.async_llm.add_lora(lora_request)

    async def remove_lora(self, lora_int_id: int) -> None:
        """Remove a LoRA adapter from the engine by its integer id."""
        await self.async_llm.remove_lora(lora_int_id)

    async def shutdown(self) -> None:
        """Stop the API server and shut down the engine (kills child procs)."""
        if self._api_server_task is not None:
            self._api_server_task.cancel()
            try:
                await self._api_server_task
            except asyncio.CancelledError:
                pass
            self._api_server_task = None
        if self.async_llm is not None:
            logger.info("Shutting down vLLM engine")
            self.async_llm.shutdown()
