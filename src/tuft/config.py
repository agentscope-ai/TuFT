"""Configuration helpers for the TuFT service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, model_validator

from .persistence import PersistenceConfig


def _default_checkpoint_dir() -> Path | None:
    """Return None to let CLI set the default based on TUFT_HOME."""
    return None


# Roles a configured model can serve. Only the backends required by the
# declared capabilities are constructed, so a training-only deployment never
# initializes vLLM and a sampling-only deployment never loads a training model.
ModelCapability = Literal["training", "sampling"]
TRAINING_CAPABILITY: ModelCapability = "training"
SAMPLING_CAPABILITY: ModelCapability = "sampling"
ALL_MODEL_CAPABILITIES: tuple[ModelCapability, ...] = (TRAINING_CAPABILITY, SAMPLING_CAPABILITY)


def _default_capabilities() -> list[ModelCapability]:
    return list(ALL_MODEL_CAPABILITIES)


# Default multiplier from a LoRA adapter's rank to its ``lora_alpha``. The Tinker
# ``LoraConfig`` carries only a rank, so the alpha has to come from server config;
# 2 keeps the "hf" and "fsdp" training backends on the same update scaling.
DEFAULT_LORA_ALPHA_RATIO = 2
FSDP_QV_TARGET_MODULES = ("q_proj", "v_proj")


def compute_lora_alpha(rank: int, lora_alpha_ratio: int = DEFAULT_LORA_ALPHA_RATIO) -> int:
    """Effective peft ``lora_alpha`` for an adapter of the given rank.

    Single definition shared by every training backend, so selecting a backend
    can never change LoRA update scaling for the same rank and configuration.
    """
    return rank * lora_alpha_ratio


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
    # Which sides of the service this model provides. The default keeps the
    # historical behavior: every model both trains and samples. Declaring only
    # one capability skips constructing the other side's backend entirely, so
    # its GPU resources are never reserved; requests that need the missing
    # capability fail with a deterministic capability-disabled error. Changing
    # capabilities and restarting is supported: persisted runs and sessions of
    # a disabled capability are preserved and become usable again once the
    # capability is re-enabled.
    capabilities: list[ModelCapability] = Field(default_factory=_default_capabilities)
    tensor_parallel_size: int = 1  # tensor parallel size
    # Data parallel size for inference: launch N independent vLLM instances (each with
    # tensor_parallel_size GPUs) and load-balance requests across them.  Ideal for small
    # models where TP introduces unnecessary cross-GPU communication overhead.
    data_parallel_size: int = 1

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
    # Multiplier from a LoRA adapter's rank to its lora_alpha:
    # lora_alpha = rank * lora_alpha_ratio. Shared by the "hf" and "fsdp" training
    # backends so the same rank produces the same update scaling on both.
    #
    # Compatibility: before this setting existed the "hf" backend used ratio 1
    # (lora_alpha = rank) while "fsdp" used 2. The default is 2, so an existing "hf"
    # deployment that must keep its previous update scaling has to set
    # `lora_alpha_ratio: 1` explicitly. Checkpoints record their effective alpha and
    # a load that disagrees with this setting is rejected instead of silently
    # rescaling the adapter (see CheckpointRecord.validate_lora_alpha).
    lora_alpha_ratio: int = DEFAULT_LORA_ALPHA_RATIO

    # default training setting
    micro_batch_size: int = 1  # micro-batch size for training
    # training backend: "hf" (HFTrainingBackend) or "fsdp" (FSDPTrainingBackend)
    training_backend: Literal["hf", "fsdp"] = "hf"
    # number of GPUs (Ray actors) for FSDP backend; default 1.
    # Multi-GPU (fsdp_num_gpus >= 2) uses contiguous batch sharding across
    # Ray actors; each actor runs FSDP-2 with micro-batch grad accumulation.
    fsdp_num_gpus: int = 1
    # TCP port for torch.distributed init (FSDP multi-GPU); default 29500
    fsdp_master_port: int = 29500
    # LoRA slot count per rank: rank -> slots for that rank (optional; code default if unset).
    # Example: fsdp_rank_slots: {8: 8, 16: 2}
    fsdp_rank_slots: dict[int, int] | None = None
    # Explicit opt-in to Q/V-only FSDP geometry. This intentionally overrides client
    # modifiers so existing Q/V checkpoints remain loadable.
    fsdp_qv_only: bool = False
    # Homogeneous LoRA target geometry preallocated before fully_shard(). When omitted,
    # the modifiers match Tinker's public LoraConfig defaults and are resolved through
    # the same model-series map as the HF backend. Q/V-only geometry must use the
    # dedicated fsdp_qv_only setting above.
    fsdp_target_modules: list[str] | None = None
    # Explicit fused-parameter targets (peft target_parameters), e.g. the routed
    # experts of Qwen3.5-based MoE models. When omitted, they resolve from the
    # fsdp_train_* modifiers through the same model map as the HF backend; an
    # unknown model family resolves to none. fsdp_qv_only implies none.
    fsdp_target_parameters: list[str] | None = None
    fsdp_train_attn: bool = True
    fsdp_train_mlp: bool = True
    fsdp_train_unembed: bool = True
    # Keep Tinker's Qwen3.5 train_attn geometry by default. Operators can opt
    # into LoRA on the additional Gated DeltaNet A/B gate projections for both
    # HF and FSDP backends.
    qwen_gated_deltanet_full_lora: bool = False
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
    # Max context length for sampling (vLLM) only; if unset, max_model_len is used.
    # Can be set smaller (e.g. 2048) in testing to reduce GPU memory and startup time.
    sampling_max_model_len: int | None = None
    # Disable vLLM's TorchInductor/CUDA-graph path. It currently fails to finish
    # warmup reliably in TuFT's embedded and fractional-GPU configurations.
    sampling_enforce_eager: bool = True

    # OpenAI-compatible vLLM API: tool calling (required for qwenpaw ReAct agents).
    enable_auto_tool_choice: bool = False
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None

    @property
    def training_enabled(self) -> bool:
        return TRAINING_CAPABILITY in self.capabilities

    @property
    def sampling_enabled(self) -> bool:
        return SAMPLING_CAPABILITY in self.capabilities

    @model_validator(mode="after")
    def validate_capabilities(self) -> "ModelConfig":
        if not self.capabilities:
            raise ValueError(
                f"Model {self.model_name} declares no capabilities; at least one of "
                f"{list(ALL_MODEL_CAPABILITIES)} is required."
            )
        # Canonical order with duplicates removed, so equal capability sets
        # always serialize identically in API responses.
        self.capabilities = [c for c in ALL_MODEL_CAPABILITIES if c in self.capabilities]
        return self

    @model_validator(mode="after")
    def validate_colocate(self) -> "ModelConfig":
        if self.colocate and self.tensor_parallel_size != 1:
            raise ValueError("Colocate option is only supported for tensor_parallel_size=1.")
        if self.colocate and self.data_parallel_size > 1:
            raise ValueError("Colocate option is not supported with data_parallel_size > 1.")
        if self.colocate and not (self.training_enabled and self.sampling_enabled):
            raise ValueError(
                "Colocate places sampling and training on the same device, so it requires "
                f"both capabilities; model {self.model_name} declares {self.capabilities}."
            )
        return self

    @model_validator(mode="after")
    def validate_fsdp_rank_slots(self) -> "ModelConfig":
        """Normalize and validate the discrete ranks preallocated by FSDP."""
        if self.max_lora_rank < 1:
            raise ValueError("max_lora_rank must be at least 1")
        if self.fsdp_rank_slots is None:
            return self

        normalized = {int(rank): count for rank, count in self.fsdp_rank_slots.items()}
        if not normalized:
            raise ValueError("fsdp_rank_slots must contain at least one rank")
        if any(rank < 1 for rank in normalized):
            raise ValueError("fsdp_rank_slots ranks must be at least 1")
        if any(count < 1 for count in normalized.values()):
            raise ValueError("fsdp_rank_slots slot counts must be at least 1")
        if self.training_backend == "fsdp":
            if self.max_lora_rank not in normalized:
                raise ValueError(
                    "fsdp_rank_slots must define at least one slot for "
                    f"max_lora_rank={self.max_lora_rank}; configured ranks: "
                    f"{sorted(normalized)}"
                )
            ranks_above_max = sorted(rank for rank in normalized if rank > self.max_lora_rank)
            if ranks_above_max:
                raise ValueError(
                    f"fsdp_rank_slots contains ranks above max_lora_rank={self.max_lora_rank}: "
                    f"{ranks_above_max}"
                )
        self.fsdp_rank_slots = normalized
        return self

    @model_validator(mode="after")
    def validate_fsdp_target_modules(self) -> "ModelConfig":
        """Reject unusable explicit FSDP target-module geometries."""
        if self.fsdp_target_modules is not None:
            if not self.fsdp_target_modules:
                raise ValueError("fsdp_target_modules must contain at least one module")
            normalized = [module.strip() for module in self.fsdp_target_modules]
            if any(not module for module in normalized):
                raise ValueError("fsdp_target_modules cannot contain empty module names")
            if len(set(normalized)) != len(normalized):
                raise ValueError("fsdp_target_modules cannot contain duplicates")
            self.fsdp_target_modules = normalized
        if self.fsdp_target_parameters is not None:
            # An explicit empty list is meaningful: it pins the pool to "no
            # fused-parameter targets" instead of resolving them from flags.
            normalized_parameters = [param.strip() for param in self.fsdp_target_parameters]
            if any(not param for param in normalized_parameters):
                raise ValueError("fsdp_target_parameters cannot contain empty parameter names")
            if len(set(normalized_parameters)) != len(normalized_parameters):
                raise ValueError("fsdp_target_parameters cannot contain duplicates")
            self.fsdp_target_parameters = normalized_parameters
            if self.fsdp_qv_only and normalized_parameters:
                raise ValueError(
                    "fsdp_qv_only=true conflicts with fsdp_target_parameters="
                    f"{self.fsdp_target_parameters}; omit fsdp_target_parameters or set "
                    "it to []"
                )
        if self.fsdp_qv_only and self.training_backend != "fsdp":
            raise ValueError("fsdp_qv_only requires training_backend='fsdp'")
        if self.training_backend != "fsdp":
            return self

        qv_modules = set(FSDP_QV_TARGET_MODULES)
        explicit_modules = (
            set(self.fsdp_target_modules) if self.fsdp_target_modules is not None else None
        )
        if self.fsdp_qv_only:
            if explicit_modules is not None and explicit_modules != qv_modules:
                raise ValueError(
                    "fsdp_qv_only=true conflicts with fsdp_target_modules="
                    f"{self.fsdp_target_modules}; omit fsdp_target_modules or set it to "
                    "[q_proj, v_proj]"
                )
        elif explicit_modules == qv_modules:
            raise ValueError(
                "Q/V-only FSDP geometry requires fsdp_qv_only=true. Set "
                "fsdp_qv_only: true to explicitly accept client modifier overrides and "
                "load checkpoints targeting [q_proj, v_proj]."
            )
        return self

    @model_validator(mode="after")
    def validate_lora_alpha_ratio(self) -> "ModelConfig":
        if self.lora_alpha_ratio < 1:
            raise ValueError(
                f"lora_alpha_ratio must be >= 1, got {self.lora_alpha_ratio}. "
                "Use 1 to reproduce the previous 'hf' backend scaling (lora_alpha = rank)."
            )
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

    def get_model_config(self, model_name: str) -> ModelConfig | None:
        """Return the configuration for a supported model, or None if unknown."""
        return next(
            (model for model in self.supported_models if model.model_name == model_name),
            None,
        )

    def get_config_for_persistence(self) -> dict[str, Any]:
        """Get config fields for persistence signature.

        This is used to detect configuration drift across restarts.

        Security: exclude any secret material (e.g., API keys) from being
        serialized into persistence backends.

        Capabilities are excluded from the signature on purpose: switching a
        model between training/sampling/both profiles and restarting is a
        supported workflow, and must not be flagged as configuration drift.
        Records belonging to a disabled capability are preserved and validated
        against the rest of the model configuration instead.
        """
        return self.model_dump(
            mode="json",
            exclude={
                "persistence": True,
                "authorized_users": True,
                "supported_models": {"__all__": {"capabilities"}},
            },
        )


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
