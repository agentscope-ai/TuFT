"""Configuration helpers for the TuFT service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field, model_validator

from .persistence import PersistenceConfig


def _default_checkpoint_dir() -> Path | None:
    """Return None to let CLI set the default based on TUFT_HOME."""
    return None


class TelemetryConfig(BaseModel):
    """Configuration for OpenTelemetry integration.

    Attributes:
        enabled: Whether telemetry is enabled.
        service_name: Name of the service for tracing.
        otlp_endpoint: OTLP exporter endpoint. If None, uses TUFT_OTLP_ENDPOINT env var.
        resource_attributes: Additional resource attributes as key-value pairs.
    """

    enabled: bool = False
    service_name: str = "tuft"
    otlp_endpoint: str | None = None
    resource_attributes: dict[str, str] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """Evaluation-mode deployment configuration.

    Selects one of the five paper deployment modes and switches on the
    corresponding scheduling components. All logic lives server-side; the
    simulator/Tinker client remains deployment-agnostic.
    """

    # serial_async | unified_engine | colocate_2copies | static_disagg | optimal
    deployment_mode: str = "static_disagg"

    # RL-loop scheduler (fast loop + discovery), optimal mode only.
    rl_scheduler_enabled: bool = False
    rl_scheduler_replan_every_s: float = 30.0

    # Sampling-side pipeline L1/L2/L3 (A0 merge, adapter grouping, run routing).
    sampling_pipeline_enabled: bool = False

    # Mock corrector: simulated per-sample correction overhead. bias defaults to 0
    # so logprobs are unchanged until the real Corrector is trained.
    mock_corrector_enabled: bool = False
    mock_corrector_latency_ms: float = 0.0
    mock_corrector_bias: float = 0.0

    # Slow loop Fixed<->Flex composition control.
    slow_loop_enabled: bool = False
    # identity_tag: fixed group is a relabeled flex-sampling group (near-zero cost).
    # quantized: fixed group loads an independently quantized base (reload cost).
    slow_loop_conversion_mode: str = "identity_tag"
    # Measured 0.6B conversion costs used as the cost model for the slow loop.
    slow_loop_fixed_to_flex_cost_s: float = 3.8
    slow_loop_flex_to_fixed_cost_s: float = 0.95
    # Slow-loop hysteresis band on training pressure (busy fraction of the single
    # training lane). Exposed here because the useful band depends on the
    # deployment: a 7-replica sampling group on 8 GPUs sits near 0.3 while a
    # training-bound arm sits near 1.0, and a threshold the run never crosses
    # makes the slow loop inert (conversion_count stays 0).
    slow_loop_high_pressure_threshold: float = 0.75
    slow_loop_low_pressure_threshold: float = 0.35
    slow_loop_smoothing_alpha: float = 0.2
    slow_loop_min_dwell_s: float = 60.0

    # Flex duty-cycle (optimal): opportunistically flip the flex GPU into
    # sampling mode while the training lane is idle and sampling has backlog,
    # then flip back to drain training. This is what makes the RL-loop
    # scheduler's freed training windows actually relieve sampling pressure.
    flex_duty_cycle_enabled: bool = False
    flex_sampling_min_window_s: float = 3.0
    flex_switch_backlog_threshold: int = 4
    flex_duty_cycle_tick_s: float = 0.5
    flex_duty_cycle_min_dwell_s: float = 2.0
    # Upper bound on one duty-cycle sampling window, and how far it may
    # stretch when the sampling queue is deep. 0 disables the bound. These
    # were read by RLRuntimeScheduler but never plumbed from config, so the
    # yaml value was dropped and every arm used the dataclass default 10s -
    # the uncapped control then still reported cap_exits>0.
    # How many sampling requests the flex group may hold while it is in the
    # sampling pool. 0 keeps the old round-robin 1/N share, which left the flex
    # GPU at 0.213-1.068 req/s with 66% of its window idle.
    flex_target_inflight: int = 0
    flex_duty_window_cap_s: float = 10.0
    flex_duty_window_cap_max_scale: float = 6.0
    # Only treat the training lane as idle (worth filling with sampling) after it
    # has been quiet this long. Measured gap distribution: the median gap is ~1s
    # of intra-step bubble, while gaps >=2s hold 90% of the idle seconds, so a
    # threshold near 1.5s fills the real gaps and skips the bubbles.
    flex_block_quiet_s: float = 1.5
    # Fill a training gap with sampling only if the observed gap is at least
    # this many round-trip switch costs. 1.0 = fill every gap that can pay for
    # its own switch (reactive "no training -> sample" policy).
    flex_gap_gain_factor: float = 2.0
    # L2 adapter-coherent coalescing: a request that would switch the active
    # adapter run is held while same-adapter requests are still in flight, up to
    # this bound, so downstream batches stay adapter-coherent (fewer LoRA
    # swaps per vLLM batch). 0 disables it (pure arrival order).
    l2_coalesce_max_hold_s: float = 0.0
    # L3 short-request admission: while the flex GPU is opportunistically
    # sampling, only requests whose max_tokens is <= this may land on it, so a
    # long decode never blocks the flip back to training (paper: only short R_s
    # go to the flex group). 0 disables the filter (any request may use flex).
    flex_max_sample_tokens: int = 0

    # Phase-locked periodic flex windows (optimal): the flex GPU alternates on a
    # schedule (training window then sampling window) instead of reactively. The
    # fast loop aligns each tenant's training readiness into the training window
    # via its one-shot sampling delay, so training forms contiguous blocks and
    # the sampling window is one long stretch (2 zero-copy flips per period).
    flex_window_mode: bool = False
    flex_window_period_s: float = 60.0
    flex_training_window_s: float = 20.0
    # Max time a (misaligned) training request may be held waiting for the next
    # training window before the scheduler flips early for it.
    flex_max_training_hold_s: float = 5.0
    # Floor for the per-tenant delay budget W_i as a fraction of the tenant
    # period. A pre-sampling shift moves the whole loop, so it does NOT make
    # rollouts stale; the bound is a latency budget, not a staleness one.
    rl_align_slack_frac: float = 0.5

    # Fast-loop training packing (optimal): hold early-ready training bursts so
    # bursts consolidate into fewer contiguous blocks, merging scattered idle
    # into larger flex-sampling windows. Bounded by per-tenant staleness window.
    rl_pack_training: bool = False
    # Fast-loop one-shot sampling delay (phi): master switch. The delay only pays
    # off when something consumes the contiguity it buys (the flex duty cycle).
    # With the flex GPU dedicated to training nothing consumes it, so the delay is
    # pure added sampling latency; turning it off measures that cost.
    rl_fast_loop_holds_enabled: bool = True
    # The packer stops at the first tenant whose required delay exceeds W_i. By
    # default the whole plan is then dropped (no holds at all), so the fast loop
    # goes inert once one tenant's budget is exceeded -- measured as replans
    # 26 / infeasible 25 / hold 0s at 24 tenants. Enabling this applies the
    # feasible prefix instead; every prefix hold still respects its own W_i.
    rl_fast_loop_partial_plan: bool = False

    # Serial-Async time-slicing: how many tenants run per slice and how many RL
    # iterations each tenant owns the cluster before yielding.
    serial_async_tenants_per_slice: int = 1
    serial_async_iterations_per_slice: int = 4

    # Unified-Engine mode alternation: minimum sampling window after the
    # training queue drains before training can pull the engine back.
    unified_engine_sampling_min_window_s: float = 3.0


class ModelConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model_name: str  # name used in APIs
    model_path: Path  # path to model checkpoint
    max_model_len: int  # maximum context length supported by the model
    tensor_parallel_size: int = 1  # tensor parallel size
    # Data parallel size for inference: launch N independent vLLM instances (each with
    # tensor_parallel_size GPUs) and load-balance requests across them.  Ideal for small
    # models where TP introduces unnecessary cross-GPU communication overhead.
    data_parallel_size: int = 1
    # GPU memory utilization for standalone/DP sampling vLLM instances (vLLM default
    # 0.9). Lower it when a sampling instance may share a GPU with another runtime
    # (e.g. optimal's fixed-sampling group coexisting with the flex backend) so the
    # instance still fits. 1.0 = use vLLM default.
    sampling_standalone_gpu_memory_utilization: float = 0.9

    # default sampling parameters for this model
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    logprobs: int = 0
    seed: int = 42
    min_response_tokens: int = 0
    # Default max_tokens for sampling when client does not specify one.
    # Applies to both tinker SDK sample() and vLLM OpenAI API (max_response_tokens).
    # If None, vLLM uses its own default (typically 16).
    default_max_tokens: int | None = None

    # default lora setting
    max_lora_rank: int = 16  # maximum rank for LoRA adapters
    max_loras: int = 1  # maximum number of LoRA adapters that can be applied simultaneously

    # default training setting
    micro_batch_size: int = 1  # micro-batch size for training
    # training backend: "hf" (HFTrainingBackend), "fsdp" (FSDPTrainingBackend),
    # "torchtp" (TorchTPTrainingBackend), or "flex" (FusedTorchTPVLLMFlexBackend).
    training_backend: str = "hf"
    # sampling backend: "vllm" (VLLMSamplingBackend), "fixed" (FixedSamplingBackend),
    # "router" (SamplingRuntimeRouter), or "torchtp" (TorchTPSamplingBackend).
    # If left as "auto", training_backend="torchtp" uses TorchTP sampling and all other
    # training backends use vLLM sampling.
    sampling_backend: str = "auto"
    # Optional quantization override for FixedSamplingBackend. This lets FlexBackend
    # keep its zero-copy sampling runtime unquantized while a fixed sampling runtime
    # independently loads a quantized base model.
    fixed_sampling_quantization: str | None = None
    # Number of data-parallel replicas for the FIXED sampling runtime used by the
    # SamplingRuntimeRouter in "fixed_plus_flex" mode (optimal deployment). This is
    # independent of the top-level data_parallel_size (which selects DPSamplingBackend
    # and would bypass the router). e.g. optimal uses fixed sampling DP3 + flex 1.
    fixed_sampling_data_parallel_size: int = 1
    # If True, VLLMSamplingBackend/FixedSamplingBackend pass enable_sleep_mode=True
    # to the underlying vLLM engine (via trinity's extra_engine_args passthrough) so
    # SamplingRuntimeRouter can sleep(level=1)/wake_up() this runtime instead of
    # keeping its full weights + KV cache resident while inactive. Default False to
    # avoid changing existing VLLMSamplingBackend deployments; FixedSamplingBackend
    # enables this automatically for itself.
    sampling_backend_enable_sleep_mode: bool = False
    # number of GPUs (Ray actors) for FSDP backend; default 1.
    # Multi-GPU (fsdp_num_gpus >= 2) uses contiguous batch sharding across
    # Ray actors; each actor runs FSDP-2 with micro-batch grad accumulation.
    fsdp_num_gpus: int = 1
    # TCP port for torch.distributed init (FSDP multi-GPU); default 29500
    fsdp_master_port: int = 29500
    # LoRA slot count per rank: rank -> slots for that rank (optional; code default if unset).
    # Example: fsdp_rank_slots: {8: 16, 16: 8}
    fsdp_rank_slots: dict[int, int] | None = None
    # optional override for FSDP backend HFModelConfig (e.g. attn_implementation)
    fsdp_override_config: dict[str, Any] | None = None
    # Attention implementation passed to AutoModelForCausalLM.from_pretrained for the
    # HF training backend (and used as default for FSDP backend if fsdp_override_config
    # does not specify one). Common values: "flash_attention_2", "sdpa", "eager".
    # If None, transformers picks its own default (usually "sdpa").
    attn_implementation: str | None = None
    # Quantization method for the sampling (vLLM) engine.
    # Supported values: "fp8", "awq", "gptq", "bitsandbytes", etc.
    # If None, no quantization is applied (model runs in dtype as-is).
    quantization: str | None = None

    # whether to colocate sampling and training on the same device
    # only for local testing purposes
    colocate: bool = False
    sampling_memory_fraction: float = 0.2  # fraction of GPU memory for sampling
    # Whether vLLM should force eager execution. The FlexBackend optimal path
    # uses enforce_eager=False together with sampling_pre_capture_alias=True so
    # CUDA Graph is captured over the IPC-aliased base weights, giving ~3x higher
    # sampling throughput than eager mode. Set True only for diagnostics or for
    # the conservative post-init-alias path (which auto-disables CUDA Graph).
    sampling_enforce_eager: bool = False
    # Disable vLLM CUDA Graph capture for sampling. This can also be used with
    # eager=True for diagnostics; FlexBackend also applies it automatically when
    # sampling_enforce_eager=False and sampling_pre_capture_alias=False.
    sampling_disable_cudagraph: bool = False
    # Enable vLLM sleep mode so KV cache physical pages can be unmapped/remapped
    # while preserving CUDA Graph virtual addresses. Required for the FlexBackend
    # keep-runtime sleep/wake switching path.
    sampling_enable_sleep_mode: bool = True
    # Alias FlexBackend IPC weights before vLLM CUDA Graph capture. This keeps
    # enforce_eager=False performance while ensuring graphs capture real storage.
    # This is the validated high-performance path and is enabled by default.
    sampling_pre_capture_alias: bool = True
    # Disable vLLM custom all-reduce. Required for the stable pre-capture alias +
    # CUDA Graph path (graph capture over IPC aliased weights fails otherwise).
    sampling_disable_custom_all_reduce: bool = True
    # Keep vLLM runtime/CUDA Graph while in training mode by sleeping CuMem
    # weights+kv_cache pages instead of destroying the sampling runtime. This
    # avoids per-switch vLLM rebuild/recapture and gives sub-100ms switches.
    sampling_keep_runtime_on_training: bool = True
    # Use Flex-specific sleep/wake: discard CuMem pages without CPU offload and
    # remap tags only, avoiding vLLM's dummy weight reload path.
    sampling_flex_kv_cache_only_sleep: bool = True
    sampling_sleep_wake_tags: list[str] = Field(default_factory=lambda: ["weights", "kv_cache"])
    # If True, FlexBackend releases/rebuilds the sampling runtime when LoRA
    # adapter snapshots are present. Default False keeps the high-performance
    # sleep/wake path; LoRA snapshots use unique internal vLLM lora_name values
    # to avoid stale adapter cache. Set True as a correctness fallback if a
    # future vLLM version regresses dynamic LoRA refresh behavior.
    sampling_rebuild_runtime_for_lora: bool = False
    # If >0, only this much KV cache (GiB per rank) is unmapped during Flex
    # sleep. Dummy weights are still discarded. This trades retained KV memory
    # for faster wake and is useful when training only needs part of KV memory.
    # Default 0.0 unmaps the full KV cache (safe for any model size); tune per
    # model (e.g. 12 for 32B TP=4) to keep wake latency under ~1s.
    sampling_partial_kv_cache_gb: float = 0.0
    # Reuse CUDA IPC descriptors across repeated training->sampling transforms.
    # Valid because base storage is kept alive and stable across mode switches
    # (validated by storage_stable checks); avoids per-switch descriptor rebuild.
    sampling_reuse_ipc_descriptors: bool = True
    # Warm up the sampling runtime (create vLLM + capture CUDA Graph + inject IPC
    # alias, then sleep it) during async_init so the first user-facing
    # transform_to_sampling hits the fast wake path (~100ms) instead of the
    # ~100s cold start. Only active for the optimal path (pre-capture alias +
    # keep-runtime sleep/wake). Set False for unit tests with fake engines.
    sampling_warmup_at_init: bool = True
    # Max context length for sampling (vLLM) only; if unset, max_model_len is used.
    # Can be set smaller (e.g. 2048) in testing to reduce GPU memory and startup time.
    sampling_max_model_len: int | None = None

    # OpenAI-compatible vLLM API: tool calling (required for qwenpaw ReAct agents).
    enable_auto_tool_choice: bool = False
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None

    # Evaluation deployment mode and scheduler component switches.
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def validate_colocate(self) -> "ModelConfig":
        if self.colocate and self.tensor_parallel_size != 1:
            raise ValueError("Colocate option is only supported for tensor_parallel_size=1.")
        if self.colocate and self.fsdp_num_gpus != self.data_parallel_size:
            raise ValueError(
                "Colocate requires fsdp_num_gpus == data_parallel_size so each "
                "training rank pairs with one colocated sampling replica per GPU."
            )
        if self.colocate and not (0.0 < self.sampling_memory_fraction < 1.0):
            raise ValueError("Colocate option requires 0 < sampling_memory_fraction < 1.")
        return self

    @model_validator(mode="after")
    def validate_fsdp_rank_slots(self) -> "ModelConfig":
        """Ensure fsdp_rank_slots keys are int (YAML/JSON may load them as str)."""
        if self.fsdp_rank_slots is not None and len(self.fsdp_rank_slots) > 0:
            self.fsdp_rank_slots = {int(k): v for k, v in self.fsdp_rank_slots.items()}
        return self

    @model_validator(mode="after")
    def validate_tool_calling(self) -> "ModelConfig":
        if self.enable_auto_tool_choice and not self.tool_call_parser:
            raise ValueError(
                "enable_auto_tool_choice requires tool_call_parser "
                "(e.g. hermes for Qwen3-Thinking models)."
            )
        return self


class AppConfig(BaseModel):
    """Runtime configuration for the TuFT server.

    This is a Pydantic model that can be serialized/deserialized for persistence.
    """

    model_config = {"arbitrary_types_allowed": True}

    worker_venv_path: str | None = None  # Ray worker venv; empty = no venv; required when using Ray
    checkpoint_dir: Path | None = Field(default_factory=_default_checkpoint_dir)
    supported_models: list[ModelConfig] = Field(default_factory=list)
    model_owner: str = "local-user"
    toy_backend_seed: int = 0
    # TODO: Temporary implementation for user authorization,
    # replace with proper auth system later
    authorized_users: dict[str, str] = Field(default_factory=dict)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    def ensure_directories(self) -> None:
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def check_validity(self) -> None:
        if not self.supported_models:
            raise ValueError("At least one supported model must be configured.")
        model_names = {model.model_name for model in self.supported_models}
        if len(model_names) != len(self.supported_models):
            raise ValueError("Model names in supported_models must be unique.")
        if len(model_names) > 1 and any(model.colocate for model in self.supported_models):
            raise ValueError(
                "Colocate option is only allowed when there is a single supported model."
            )

    def with_supported_models(self, models: Iterable[ModelConfig]) -> "AppConfig":
        updated = list(models)
        if updated:
            self.supported_models = updated
        return self

    def get_config_for_persistence(self) -> dict[str, Any]:
        """Get config fields for persistence signature.

        This is used to detect configuration drift across restarts.

        Security: exclude any secret material (e.g., API keys) from being
        serialized into persistence backends.
        """
        return self.model_dump(mode="json", exclude={"persistence", "authorized_users"})


def load_yaml_config(config_path: Path) -> AppConfig:
    """Loads an AppConfig from a YAML file."""
    from omegaconf import OmegaConf

    loaded = OmegaConf.load(config_path)
    try:
        # Convert OmegaConf to plain dict for Pydantic
        config_dict = OmegaConf.to_container(loaded, resolve=True)
        if not isinstance(config_dict, dict):
            raise ValueError("Config file must contain a dictionary at root level")
        return AppConfig.model_validate(config_dict)
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}") from e
