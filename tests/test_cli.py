from __future__ import annotations

from typing import Any

from typer.testing import CliRunner

from llm_rpc import cli
from llm_rpc.config import AppConfig


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
            "--checkpoint-dir",
            str(tmp_path),
            "--model-owner",
            "tester",
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
    assert server_state.config.supported_models == defaults.supported_models
    assert server_state.config.model_owner == "tester"
    assert server_state.config.toy_backend_seed == defaults.toy_backend_seed
