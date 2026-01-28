"""Command line utilities for the local TuFT server."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
import uvicorn

from .config import AppConfig, load_yaml_config
from .server import create_root_app
from .telemetry import init_telemetry
from .telemetry.metrics import ResourceMetricsCollector


app = typer.Typer(help="TuFT - Tenant-unified Fine-Tuning Server.")

_HOST_OPTION = typer.Option("127.0.0.1", "--host", help="Interface to bind")
_PORT_OPTION = typer.Option(8000, "--port", "-p", help="Port to bind")
_LOG_LEVEL_OPTION = typer.Option("info", "--log-level", help="Uvicorn log level")
_RELOAD_OPTION = typer.Option(False, "--reload", help="Enable auto-reload (development only)")
_CONFIG_OPTION = typer.Option(
    None,
    "--config",
    "-c",
    help="Path to a TuFT configuration file (YAML)",
)
_CHECKPOINT_DIR_OPTION = typer.Option(
    None,
    "--checkpoint-dir",
    help="Override checkpoint_dir from config file. Defaults to ~/.cache/tuft/checkpoints.",
)


def _build_config(
    config_path: Path | None,
    checkpoint_dir: Path | None,
) -> AppConfig:
    if config_path is None:
        raise typer.BadParameter("Configuration file must be provided via --config")
    config = load_yaml_config(config_path)
    if checkpoint_dir is not None:
        config.checkpoint_dir = checkpoint_dir.expanduser()
    config.ensure_directories()
    return config


def _init_telemetry(config: AppConfig, log_level: str) -> None:
    """Initialize OpenTelemetry if enabled."""
    # Configure root logger level to ensure logs flow to OTel
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    if not config.telemetry.enabled:
        logging.basicConfig(level=numeric_level)
        return

    init_telemetry(config.telemetry)
    # Start resource metrics collection
    ResourceMetricsCollector.start(str(config.checkpoint_dir))


@app.command()
def launch(
    host: str = _HOST_OPTION,
    port: int = _PORT_OPTION,
    log_level: str = _LOG_LEVEL_OPTION,
    reload: bool = _RELOAD_OPTION,
    config_path: Path | None = _CONFIG_OPTION,
    checkpoint_dir: Path | None = _CHECKPOINT_DIR_OPTION,
) -> None:
    """Launch the TuFT server."""
    app_config = _build_config(config_path, checkpoint_dir)
    # Initialize telemetry before starting the server
    _init_telemetry(app_config, log_level)
    logging.getLogger("tuft").info("Server starting on %s:%s", host, port)
    uvicorn.run(
        create_root_app(app_config),
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
