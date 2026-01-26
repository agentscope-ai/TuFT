"""Configuration helpers for the TuFT service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

from .persistence import PersistenceConfig


def _default_checkpoint_dir() -> Path:
    return Path.home() / ".cache" / "tuft" / "checkpoints"


def _default_persistence_config() -> PersistenceConfig:
    return PersistenceConfig()


@dataclass
class TelemetryConfig:
    """Configuration for OpenTelemetry integration.

    Attributes:
        enabled: Whether telemetry is enabled.
        service_name: Name of the service for tracing.
        otlp_endpoint: OTLP exporter endpoint. If None, uses OTEL_EXPORTER_OTLP_ENDPOINT env var.
        resource_attributes: Additional resource attributes as key-value pairs.
    """

    enabled: bool = False
    service_name: str = "tuft"
    otlp_endpoint: str | None = None
    resource_attributes: Dict[str, str] = field(default_factory=dict)


def _default_telemetry_config() -> TelemetryConfig:
    return TelemetryConfig()


@dataclass
class AppConfig:
    """Runtime configuration for the FastAPI service."""

    checkpoint_dir: Path = field(default_factory=_default_checkpoint_dir)
    supported_models: List[ModelConfig] = field(default_factory=list)
    model_owner: str = "local-user"
    toy_backend_seed: int = 0
    # TODO: Temporary implementation for user authorization,
    # replace with proper auth system later
    authorized_users: Dict[str, str] = field(default_factory=dict)
    persistence: PersistenceConfig = field(default_factory=_default_persistence_config)
    telemetry: TelemetryConfig = field(default_factory=_default_telemetry_config)

    def ensure_directories(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def check_validity(self) -> None:
        if not self.supported_models:
            raise ValueError("At least one supported model must be configured.")
        model_names = {model.model_name for model in self.supported_models}
        if len(model_names) != len(self.supported_models):
            raise ValueError("Model names in supported_models must be unique.")

    def with_supported_models(self, models: Iterable[ModelConfig]) -> "AppConfig":
        updated = list(models)
        if updated:
            self.supported_models = updated
        return self


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

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


def load_yaml_config(config_path: Path) -> AppConfig:
    """Loads an AppConfig from a YAML file."""
    from omegaconf import OmegaConf

    schema = OmegaConf.structured(AppConfig)
    loaded = OmegaConf.load(config_path)
    try:
        config = OmegaConf.merge(schema, loaded)
        app_config = OmegaConf.to_object(config)
        assert isinstance(
            app_config, AppConfig
        ), "Loaded config is not of type AppConfig, which should not happen."
        return app_config
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}") from e
