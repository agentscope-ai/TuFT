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
import tempfile
import time
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
    share_base_weights: bool = False
    kv_cache_memory_bytes: int | None = None
    language_model_only: bool = False
    moe_backend: str = "auto"
    gdn_prefill_backend: str | None = None
    max_num_seqs: int | None = None
    # Trusted physical devices reserved by the colocated FSDP actors.
    shared_gpu_ids: list[str] | None = None


class VLLMEngine:
    """vLLM engine + OpenAI API server, run as a Ray actor.

    Wrapped with ``ray.remote(VLLMEngine)`` by the sampling backend; all
    public methods are invoked via ``.remote()``.
    """

    def __init__(self, config: VLLMEngineConfig) -> None:
        if config.shared_gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(config.shared_gpu_ids)
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
        self._shared_manifest_dir: Any = None
        self._phase_lock = asyncio.Lock()
        self._sleeping = False
        self._phase_failed = False
        self._phase_stats: dict[str, float] = {}

    def configure_shared_weights(self, descriptors: dict | list[dict]) -> None:
        """Accept trusted trainer IPC metadata before engine initialization."""
        import pickle
        from pathlib import Path

        import torch
        import vllm

        from .shared_weights import check_shared_weight_versions

        check_shared_weight_versions(torch.__version__, vllm.__version__)
        if not self.config.share_base_weights or self._prepared or self._shared_manifest_dir:
            raise RuntimeError("Invalid shared-weight initialization order")
        self._shared_manifest_dir = tempfile.TemporaryDirectory(prefix="tuft-shared-")
        path = Path(self._shared_manifest_dir.name) / "weights.pkl"
        ranks = descriptors if isinstance(descriptors, list) else [descriptors]
        if len(ranks) != self.config.tensor_parallel_size:
            raise ValueError("Shared weight rank count must equal sampler TP size")
        path.write_bytes(pickle.dumps(ranks))

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
                kv_cache_memory_bytes=self.config.kv_cache_memory_bytes,
                language_model_only=self.config.language_model_only,
                moe_backend=self.config.moe_backend,
                gdn_prefill_backend=self.config.gdn_prefill_backend,
                max_num_seqs=self.config.max_num_seqs,
            )
            if self.config.quantization:
                engine_args.quantization = self.config.quantization
            if self.config.share_base_weights:
                if self._shared_manifest_dir is None:
                    raise RuntimeError("Shared sampler requires trainer IPC descriptors")
                engine_args.load_format = "tuft_shared"
                engine_args.model_loader_extra_config = {
                    "manifest": self._shared_manifest_dir.name + "/weights.pkl"
                }
                engine_args.enable_sleep_mode = True
                # Trainers load text backbones only. Do not allocate an unused
                # vision encoder in the consumer. Triton MoE retains the BF16
                # expert matrix layout rather than creating shuffled copies.
                engine_args.language_model_only = True
                engine_args.moe_backend = "triton"

            self.async_llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
            # Apply TuFT's worker-side patches (prompt-logprobs temperature
            # scaling) on every worker process; see vllm_worker.py.
            await self.async_llm.collective_rpc("apply_patches")
            if self.config.share_base_weights:
                await self.async_llm.collective_rpc("shared_weight_report")
                self._shared_manifest_dir.cleanup()
            await self._run_api_server()
            self._prepared = True

    async def reset_prefix_cache(self) -> bool:
        """Diagnostic cache-reset control; callers must first drain requests."""
        return await self.async_llm.reset_prefix_cache()

    async def set_training_phase(self, training: bool) -> None:
        """Called only with the scheduler's shared-GPU reservation held.

        Imported base storage is owned by HF, outside vLLM's sleep allocator.
        Level 1 preserves any private weights/LoRA buffers and discards KV data.
        """
        if not self.config.share_base_weights:
            return
        async with self._phase_lock:
            if self._phase_failed:
                raise RuntimeError("Shared sampler phase transition previously failed")
            if training == self._sleeping:
                return
            start = time.monotonic()
            try:
                if training:
                    await self.async_llm.sleep(level=1)
                else:
                    await self.async_llm.wake_up()
                self._sleeping = training
            except BaseException:
                self._phase_failed = True
                raise
            key = "sleep" if training else "wake"
            self._phase_stats[key + "_s"] = (
                self._phase_stats.get(key + "_s", 0) + time.monotonic() - start
            )
            self._phase_stats[key + "_count"] = self._phase_stats.get(key + "_count", 0) + 1

    async def shared_weight_report(self) -> dict:
        rows = await self.async_llm.collective_rpc("shared_weight_report")
        totals = {key: sum(row[key] for row in rows) for key in rows[0]}
        return {**totals, "ranks": rows, **self._phase_stats, "sleeping": self._sleeping}

    async def _rank_fingerprints(self, method: str) -> dict:
        rows = await self.async_llm.collective_rpc(method)
        if len(rows) == 1:
            return rows[0]
        return {
            f"rank_{rank}/{key}": value
            for rank, row in enumerate(rows)
            for key, value in row.items()
        }

    async def base_weight_fingerprints(self) -> dict:
        return await self._rank_fingerprints("base_weight_fingerprints")

    async def model_buffer_fingerprints(self) -> dict:
        return await self._rank_fingerprints("model_buffer_fingerprints")

    async def adapter_weight_fingerprints(self) -> dict:
        return await self._rank_fingerprints("adapter_weight_fingerprints")

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

    def get_device_assignment(self) -> dict:
        """Ray's physical GPU assignment, before starting the engine workers."""
        import ray

        return {"node": ray.get_runtime_context().get_node_id(), "gpus": ray.get_gpu_ids()}

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
        if self._shared_manifest_dir is not None:
            self._shared_manifest_dir.cleanup()
