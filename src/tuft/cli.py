"""Command line utilities for the local TuFT server."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import typer
import uvicorn

from .config import AppConfig, load_yaml_config
from .exceptions import ConfigMismatchError
from .persistence import (
    flush_all_data,
    get_current_namespace,
    get_redis_store,
    validate_config_signature,
)
from .server import create_root_app
from .telemetry import init_telemetry
from .telemetry.metrics import ResourceMetricsCollector


app = typer.Typer(help="TuFT - Tenant-unified Fine-Tuning Server.", no_args_is_help=True)


# Required for Typer to recognize subcommands when using no_args_is_help=True
@app.callback()
def callback() -> None:
    """TuFT - Tenant-unified Fine-Tuning Server."""


# Default paths based on TUFT_HOME
_TUFT_HOME = Path(os.environ.get("TUFT_HOME", Path.home() / ".tuft"))
_DEFAULT_CONFIG_PATH = _TUFT_HOME / "configs" / "tuft_config.yaml"
_DEFAULT_CHECKPOINT_DIR = _TUFT_HOME / "checkpoints"

_HOST_OPTION = typer.Option("127.0.0.1", "--host", help="Interface to bind", envvar="TUFT_HOST")
_PORT_OPTION = typer.Option(10610, "--port", "-p", help="Port to bind", envvar="TUFT_PORT")
_LOG_LEVEL_OPTION = typer.Option(
    "info", "--log-level", help="Uvicorn log level", envvar="TUFT_LOG_LEVEL"
)
_RELOAD_OPTION = typer.Option(False, "--reload", help="Enable auto-reload (development only)")
_CONFIG_OPTION = typer.Option(
    None,
    "--config",
    "-c",
    help=f"Path to a TuFT configuration file (YAML). Defaults to {_DEFAULT_CONFIG_PATH}",
    envvar="TUFT_CONFIG",
)
_CHECKPOINT_DIR_OPTION = typer.Option(
    None,
    "--checkpoint-dir",
    help=f"Override checkpoint_dir from config file. Defaults to {_DEFAULT_CHECKPOINT_DIR}",
    envvar="TUFT_CHECKPOINT_DIR",
)


def _resolve_config_path(config_path: Path | None) -> Path:
    """Resolve the config path, falling back to default if not provided."""
    if config_path is not None:
        return config_path
    if _DEFAULT_CONFIG_PATH.exists():
        return _DEFAULT_CONFIG_PATH
    raise typer.BadParameter(
        f"Configuration file must be provided via --config or TUFT_CONFIG, "
        f"or create a default config at {_DEFAULT_CONFIG_PATH}"
    )


_REFRESH_PERSISTENCE_OPTION = typer.Option(
    False,
    "--refresh-persistence",
    help=(
        "Clear all existing persistence data and start fresh. "
        "Use when config has changed and you want to discard old data."
    ),
)
_FORCE_REFRESH_PERSISTENCE_OPTION = typer.Option(
    False,
    "--force-refresh-persistence",
    help="Skip confirmation prompts when using --refresh-persistence.",
)


def _build_config(
    config_path: Path | None,
    checkpoint_dir: Path | None,
) -> AppConfig:
    resolved_config_path = _resolve_config_path(config_path)
    config = load_yaml_config(resolved_config_path)
    # Apply checkpoint_dir override, or use default if not in config
    if checkpoint_dir is not None:
        config.checkpoint_dir = checkpoint_dir.expanduser()
    elif config.checkpoint_dir is None:
        config.checkpoint_dir = _DEFAULT_CHECKPOINT_DIR
    # Guarantee checkpoint_dir is set after resolution
    assert config.checkpoint_dir is not None, "checkpoint_dir must be set after config resolution"
    config.ensure_directories()
    return config


def _handle_refresh_persistence(force_refresh: bool) -> None:
    """Handle the --refresh-persistence flag.

    Prompts for confirmation unless --force-refresh is provided,
    then clears all persistence data in the current namespace.
    """
    namespace = get_current_namespace()

    if not force_refresh:
        typer.secho(
            "\n🚨🚨🚨 CRITICAL WARNING 🚨🚨🚨\n",
            fg=typer.colors.RED,
            bold=True,
        )
        typer.secho(
            "--refresh-persistence will PERMANENTLY DELETE ALL persistence data!\n",
            fg=typer.colors.RED,
            bold=True,
        )
        typer.secho(
            f"📦 Target namespace: '{namespace}'\n",
            fg=typer.colors.YELLOW,
            bold=True,
        )
        typer.echo(
            f"This IRREVERSIBLE action will destroy ALL data in namespace '{namespace}':\n"
            "  ❌ All saved sessions\n"
            "  ❌ All training run records and checkpoint metadata (NOT local checkpoint files)\n"
            "  ❌ All future records\n"
            "  ❌ All sampling session records\n"
            "  ❌ Configuration signature\n"
            "\n"
            "⚠️  The server will start fresh with NO previous state.\n"
            "⚠️  This action CANNOT be undone!\n"
            "⚠️  Local checkpoint files on disk are NOT affected.\n"
            f"⚠️  Only data in namespace '{namespace}' will be affected.\n"
        )
        confirmed = typer.confirm(
            f"Do you REALLY want to delete all data in namespace '{namespace}'?",
            default=False,
        )
        if not confirmed:
            typer.echo("Aborted. No data was cleared.")
            raise typer.Exit(0)

    deleted_count, cleared_namespace = flush_all_data()
    typer.secho(
        f"✅ Cleared {deleted_count} keys from namespace '{cleared_namespace}'.",
        fg=typer.colors.GREEN,
    )
    typer.echo("Server will start with fresh state.\n")


def _validate_persistence_config(
    config: AppConfig, refresh_persistence: bool, force_refresh_persistence: bool
) -> None:
    """Validate that persistence config matches stored config.

    If refresh_persistence is True, clears existing data instead of validating.
    If config mismatch is detected, exits with an error message.
    """
    if not config.persistence.enabled:
        return

    # Configure the Redis store first
    store = get_redis_store()
    store.configure(config.persistence)

    if refresh_persistence:
        _handle_refresh_persistence(force_refresh_persistence)
        return

    try:
        validate_config_signature(config)
    except ConfigMismatchError as e:
        typer.secho(
            "\n 🚫 FATAL ERROR: Configuration Mismatch Detected 🚫",
            fg=typer.colors.RED,
            bold=True,
        )
        typer.echo(f"\n{e}\n")
        raise typer.Exit(1) from e


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
    refresh_persistence: bool = _REFRESH_PERSISTENCE_OPTION,
    force_refresh_persistence: bool = _FORCE_REFRESH_PERSISTENCE_OPTION,
) -> None:
    """Launch the TuFT server."""
    app_config = _build_config(config_path, checkpoint_dir)

    # Validate persistence configuration before starting
    _validate_persistence_config(app_config, refresh_persistence, force_refresh_persistence)

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
    app(prog_name="tuft")


if __name__ == "__main__":
    main()
