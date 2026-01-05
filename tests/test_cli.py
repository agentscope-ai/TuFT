from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llm_rpc import cli
from llm_rpc.config import AppConfig, ModelConfig


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
            str(Path(__file__).parent / "data" / "models.yaml"),
            "--checkpoint-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert recorded["host"] == "0.0.0.0"
    assert recorded["port"] == 9999
    assert recorded["log_level"] == "warning"
    assert recorded["reload"] is False
    server_state = recorded["app"].state.server_state
    assert server_state.config.checkpoint_dir == tmp_path
    defaults = AppConfig()
    assert server_state.config.model_owner == "tester"
    assert server_state.config.toy_backend_seed == defaults.toy_backend_seed
    assert len(server_state.config.supported_models) == 2
    assert server_state.config.supported_models[0].model_name == "Qwen/Qwen3-8B"
    assert server_state.config.supported_models[1].model_name == "Qwen/Qwen3-32B"
    server_state.config.check_validity()  # should not raise
    server_state.config.supported_models.append(
        ModelConfig(
            model_name="Qwen/Qwen3-8B",
            model_path=Path("/path/to/model"),
            max_model_len=8192,
        )
    )
    # should raise due to duplicate model names
    with pytest.raises(ValueError, match="Model names in supported_models must be unique."):
        server_state.config.check_validity()
    server_state.config.supported_models.clear()
    # should raise due to no supported models
    with pytest.raises(ValueError, match="At least one supported model must be configured."):
        server_state.config.check_validity()
