"""Tests for tuft.runtime._launcher module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tuft.runtime._launcher import EmbeddedServer


class TestEmbeddedServer:
    def test_init_attributes(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("supported_models: []")
        server = EmbeddedServer(config_path=config, host="127.0.0.1", port=9999)
        assert server.config_path == config
        assert server.host == "127.0.0.1"
        assert server.port == 9999
        assert server.process is None
        assert not server.is_running

    @patch("tuft.runtime._launcher.EmbeddedServer._wait_for_healthy", return_value=True)
    @patch("subprocess.Popen")
    def test_start_success(self, mock_popen, mock_wait, tmp_path, monkeypatch):
        monkeypatch.setenv("TUFT_HOME", str(tmp_path))
        config = tmp_path / "config.yaml"
        config.write_text("supported_models: []")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        server = EmbeddedServer(config_path=config)
        address = server.start()
        assert address == "http://127.0.0.1:10610"
        assert server.is_running

    @patch("tuft.runtime._launcher.EmbeddedServer._wait_for_healthy", return_value=False)
    @patch("subprocess.Popen")
    def test_start_failure_raises(self, mock_popen, mock_wait, tmp_path, monkeypatch):
        monkeypatch.setenv("TUFT_HOME", str(tmp_path))
        config = tmp_path / "config.yaml"
        config.write_text("supported_models: []")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b"some error"
        mock_popen.return_value = mock_proc

        server = EmbeddedServer(config_path=config, timeout=1)
        with pytest.raises(RuntimeError, match="failed to start"):
            server.start()

    @patch("tuft.runtime._launcher.EmbeddedServer._wait_for_healthy", return_value=True)
    @patch("subprocess.Popen")
    def test_shutdown_terminates_process(self, mock_popen, mock_wait, tmp_path, monkeypatch):
        monkeypatch.setenv("TUFT_HOME", str(tmp_path))
        config = tmp_path / "config.yaml"
        config.write_text("supported_models: []")

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        server = EmbeddedServer(config_path=config)
        server.start()
        server.shutdown()

        assert server.process is None
        assert server.address is None
