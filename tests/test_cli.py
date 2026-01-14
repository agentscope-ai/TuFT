from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llm_rpc import cli
from llm_rpc.config import AppConfig, ModelConfig, load_yaml_config


def test_start_passes_config(monkeypatch, tmp_path) -> None:
    recorded: dict[str, Any] = {}

    def fake_run(app, host, port, log_level, reload):  # type: ignore[no-untyped-def]
        recorded["app"] = app
        recorded["host"] = host
        recorded["port"] = port
        recorded["log_level"] = log_level
        recorded["reload"] = reload

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    runner = CliRunner()
    model_config_path = Path(__file__).parent / "data" / "models.yaml"
    result = runner.invoke(
        cli.app,
        [
            "--host",
            "0.0.0.0",
            "--port",
            "9999",
            "--log-level",
            "warning",
            "--model-config",
            str(model_config_path),
            "--checkpoint-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert recorded["host"] == "0.0.0.0"
    assert recorded["port"] == 9999
    assert recorded["log_level"] == "warning"
    assert recorded["reload"] is False

    # Test config loading and validation separately
    # (server_state is only initialized in lifespan, which doesn't run in this test)
    config = load_yaml_config(model_config_path)
    config.checkpoint_dir = tmp_path
    config.ensure_directories()

    assert config.checkpoint_dir == tmp_path
    defaults = AppConfig()
    assert config.model_owner == "tester"
    assert config.toy_backend_seed == defaults.toy_backend_seed
    assert len(config.supported_models) == 2
    assert config.supported_models[0].model_name == "Qwen/Qwen3-8B"
    assert config.supported_models[1].model_name == "Qwen/Qwen3-32B"
    config.check_validity()  # should not raise
    config.supported_models.append(
        ModelConfig(
            model_name="Qwen/Qwen3-8B",
            model_path=Path("/path/to/model"),
            max_model_len=8192,
        )
    )
    # should raise due to duplicate model names
    with pytest.raises(ValueError, match="Model names in supported_models must be unique."):
        config.check_validity()
    config.supported_models.clear()
    # should raise due to no supported models
    with pytest.raises(ValueError, match="At least one supported model must be configured."):
        config.check_validity()
