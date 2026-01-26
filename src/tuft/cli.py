"""Command line utilities for the local TuFT server."""

from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

from .config import AppConfig, load_yaml_config
from .server import create_root_app

app = typer.Typer(help="Start the local TuFT server.")

_HOST_OPTION = typer.Option("127.0.0.1", "--host", help="Interface to bind")
_PORT_OPTION = typer.Option(8000, "--port", "-p", help="Port to bind")
_LOG_LEVEL_OPTION = typer.Option("info", "--log-level", help="Uvicorn log level")
_RELOAD_OPTION = typer.Option(False, "--reload", help="Enable auto-reload (development only)")
_CHECKPOINT_DIR_OPTION = typer.Option(
    None,
    "--checkpoint-dir",
    help="Directory for storing checkpoints. Defaults to ~/.cache/tuft/checkpoints.",
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


def _init_telemetry(config: AppConfig, log_level: str) -> None:
    """Initialize OpenTelemetry if enabled."""
    import logging

    # Configure root logger level to ensure logs flow to OTel
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level)

    if not config.telemetry.enabled:
        return
    try:
        from .telemetry import TelemetryConfig as OTelConfig
        from .telemetry import init_telemetry
        from .telemetry.metrics import ResourceMetricsCollector

        otel_config = OTelConfig(
            enabled=config.telemetry.enabled,
            service_name=config.telemetry.service_name,
            otlp_endpoint=config.telemetry.otlp_endpoint,
            resource_attributes=config.telemetry.resource_attributes,
        )
        init_telemetry(otel_config)
        # Start resource metrics collection
        ResourceMetricsCollector.start(str(config.checkpoint_dir))
    except ImportError:
        logging.getLogger(__name__).warning(
            "OpenTelemetry packages not installed. Install with: pip install tuft[otel]"
        )


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
    import logging as _logging

    config = _build_config(model_config, checkpoint_dir)
    # Initialize telemetry before starting the server
    _init_telemetry(config, log_level)
    _logging.getLogger("tuft").info("Server starting on %s:%s", host, port)
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
