"""Configuration helpers for the LLM-RPC service."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

_DEFAULT_SUPPORTED_MODELS = ["Qwen/Qwen3-4B-Instruct-2507"]


def _default_checkpoint_dir() -> Path:
    return Path.home() / ".cache" / "llm-rpc" / "checkpoints"


@dataclass
class AppConfig:
    """Runtime configuration for the FastAPI service."""

    checkpoint_dir: Path = field(default_factory=_default_checkpoint_dir)
    supported_models: List[str] = field(default_factory=lambda: list(_DEFAULT_SUPPORTED_MODELS))
    model_owner: str = "local-user"
    toy_backend_seed: int = 0

    def ensure_directories(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def with_supported_models(self, models: Iterable[str]) -> "AppConfig":
        updated = list(models)
        if updated:
            self.supported_models = updated
        return self
