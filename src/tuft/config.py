"""Configuration helpers for the TuFT service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal

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


class ModelConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model_name: str  # name used in APIs
    model_path: Path  # path to model checkpoint
    max_model_len: int  # maximum context length supported by the model
    tensor_parallel_size: int = 1  # tensor parallel size

    # default sampling parameters for this model
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    logprobs: int = 0
    seed: int = 42
    min_response_tokens: int = 0

    # default lora setting
    max_lora_rank: int = 16  # maximum rank for LoRA adapters
    max_loras: int = 1  # maximum number of LoRA adapters that can be applied simultaneously

    # default training setting
    micro_batch_size: int = 1  # micro-batch size for training
    # training backend: "hf" (HFTrainingBackend) or "fsdp" (FSDPTrainingBackend)
    training_backend: str = "hf"
    # number of GPUs (Ray actors) for FSDP backend; default 1
    fsdp_num_gpus: int = 1
    # TCP port for torch.distributed init (FSDP multi-GPU); default 29500
    fsdp_master_port: int = 29500
    # LoRA slot count per rank: rank -> slots for that rank (optional; code default if unset).
    # Example: fsdp_rank_slots: {8: 16, 16: 8}
    fsdp_rank_slots: dict[int, int] | None = None
    # optional override for FSDP backend HFModelConfig (e.g. attn_implementation)
    fsdp_override_config: dict[str, Any] | None = None

    # Quantization method for the sampling (vLLM) engine.
    # Supported values: "fp8", "awq", "gptq", "bitsandbytes", etc.
    # If None, no quantization is applied (model runs in dtype as-is).
    quantization: str | None = None

    # whether to colocate sampling and training on the same device
    # only for local testing purposes
    colocate: bool = False
    sampling_memory_fraction: float = 0.2  # fraction of GPU memory for sampling
    # Max context length for sampling (vLLM) only; if unset, max_model_len is used.
    # Can be set smaller (e.g. 2048) in testing to reduce GPU memory and startup time.
    sampling_max_model_len: int | None = None

    # OpenAI-compatible vLLM API: tool calling (required for qwenpaw ReAct agents).
    enable_auto_tool_choice: bool = False
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None

    # -- Sampling Scheduling Configuration --
    # Whether to treat initial (untrained) adapters as the base model for scheduling
    # purposes. When True, requests targeting freshly-initialised adapters (lora_B == 0)
    # are coalesced into the base-model bucket, improving throughput.
    coalesce_initial_adapters: bool = True

    # Scheduling strategy for sampling requests:
    #   "none"       -- no scheduler; requests pass directly to the backend (no reordering).
    #   "batch"      -- naive adapter-based sort within the coalescing window. Requests are
    #                    grouped by effective_lora_id and sorted lexicographically. This
    #                    maximises adapter contiguity but may starve adapters whose IDs sort
    #                    later in every window.
    #   "batch_fcfs" -- fair FCFS-then-batch. Adapter groups are ordered by the arrival time
    #                    of their first request in the window (first-come-first-served between
    #                    groups), and requests within a group preserve arrival order.  This
    #                    combines FCFS fairness with batch contiguity -- eliminates starvation
    #                    while still keeping same-adapter requests contiguous for vLLM
    #                    throughput benefit.
    scheduling_strategy: Literal["none", "batch", "batch_fcfs"] = "batch_fcfs"

    # Scheduler tuning parameters (only effective when scheduling_strategy != "none")
    scheduler_coalesce_window_s: float = 0.005  # coalescing window in seconds
    scheduler_max_batch_size: int = 32  # max items per dispatch round
    scheduler_serialize_groups: bool = False  # serialize adapter groups for stronger batching

    @model_validator(mode="after")
    def validate_colocate(self) -> "ModelConfig":
        if self.colocate and self.tensor_parallel_size != 1:
            raise ValueError("Colocate option is only supported for tensor_parallel_size=1.")
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
