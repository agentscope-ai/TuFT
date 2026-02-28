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

    # FSDP multi-GPU training settings
    training_backend: str = "hf"  # "hf" for single-GPU, "fsdp" for FSDP2 multi-GPU
    num_gpus_per_node: int = 1
    num_nodes: int = 1

    # whether to colocate sampling and training on the same device
    # only for local testing purposes
    colocate: bool = False
    sampling_memory_fraction: float = 0.2  # fraction of GPU memory for sampling

    @model_validator(mode="after")
    def validate_colocate(self) -> "ModelConfig":
        if self.colocate and self.tensor_parallel_size != 1:
            raise ValueError("Colocate option is only supported for tensor_parallel_size=1.")
        return self

    @model_validator(mode="after")
    def validate_fsdp(self) -> "ModelConfig":
        if self.training_backend == "fsdp":
            if self.colocate:
                raise ValueError("Colocate is not supported with FSDP training backend.")
            total_gpus = self.num_gpus_per_node * self.num_nodes
            if total_gpus < 1:
                raise ValueError("FSDP requires at least 1 GPU.")
        return self


class AppConfig(BaseModel):
    """Runtime configuration for the TuFT server.

    This is a Pydantic model that can be serialized/deserialized for persistence.
    """

    model_config = {"arbitrary_types_allowed": True}

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
