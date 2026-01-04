"""Command line utilities for the local LLM-RPC server."""

from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

from .config import AppConfig, load_yaml_config
from .server import create_root_app

app = typer.Typer(help="Start the local LLM-RPC server.")

_HOST_OPTION = typer.Option("127.0.0.1", "--host", help="Interface to bind")
_PORT_OPTION = typer.Option(8000, "--port", "-p", help="Port to bind")
_LOG_LEVEL_OPTION = typer.Option("info", "--log-level", help="Uvicorn log level")
_RELOAD_OPTION = typer.Option(False, "--reload", help="Enable auto-reload (development only)")
_CHECKPOINT_DIR_OPTION = typer.Option(
    None,
    "--checkpoint-dir",
    help="Directory for storing checkpoints. Defaults to ~/.cache/llm-rpc/checkpoints.",
)
_MODEL_CONFIG_OPTION = typer.Option(
    None,
    "--model-config",
    help="Path to a model configuration file (YAML)",
)


def _build_config(
    model_config_path: Path | None,
    checkpoint_dir: Path | None,
) -> AppConfig:
    if model_config_path is None:
        raise typer.BadParameter("Model configuration file must be provided via --model-config")
    config = load_yaml_config(model_config_path)
    if checkpoint_dir is not None:
        config.checkpoint_dir = checkpoint_dir.expanduser()
    config.ensure_directories()
    return config


@app.callback(invoke_without_command=True)
def start(
    host: str = _HOST_OPTION,
    port: int = _PORT_OPTION,
    log_level: str = _LOG_LEVEL_OPTION,
    reload: bool = _RELOAD_OPTION,
    model_config: Path | None = _MODEL_CONFIG_OPTION,
    checkpoint_dir: Path | None = _CHECKPOINT_DIR_OPTION,
) -> None:
    """Start the FastAPI server using uvicorn."""
    config = _build_config(model_config, checkpoint_dir)
    uvicorn.run(
        create_root_app(config),
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
