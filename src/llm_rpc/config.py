"""Configuration helpers for the LLM-RPC service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List


def _default_checkpoint_dir() -> Path:
    return Path.home() / ".cache" / "llm-rpc" / "checkpoints"


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
