"""Tests for tuft.runtime init/shutdown/is_initialized flow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import tuft


@pytest.fixture(autouse=True)
def reset_runtime():
    """Ensure runtime state is clean before and after each test."""
    tuft.shutdown()
    yield
    tuft.shutdown()


class TestInit:
    @patch("tuft.runtime.discover", return_value="http://localhost:10610")
    @patch("tuft.runtime.tinker.ServiceClient")
    def test_init_connected_mode(self, mock_client_cls, mock_discover):
        mock_client_cls.return_value = MagicMock()
        tuft.init()
        assert tuft.is_initialized()
        # Should be in connected mode
        from tuft.runtime import _mode

        assert _mode == "connected"

    @patch("tuft.runtime.discover", return_value="http://localhost:10610")
    @patch("tuft.runtime.tinker.ServiceClient")
    def test_init_idempotent(self, mock_client_cls, mock_discover):
        mock_client_cls.return_value = MagicMock()
        tuft.init()
        tuft.init()  # should not raise
        assert tuft.is_initialized()

    @patch("tuft.runtime.discover", return_value="http://localhost:10610")
    @patch("tuft.runtime.tinker.ServiceClient")
    def test_init_raises_on_reinit_when_flag_false(self, mock_client_cls, mock_discover):
        mock_client_cls.return_value = MagicMock()
        tuft.init()
        with pytest.raises(RuntimeError, match="already initialized"):
            tuft.init(ignore_reinit_error=False)

    @patch("tuft.runtime.discover", return_value=None)
    def test_init_raises_when_no_service_and_no_config(self, mock_discover, tmp_path, monkeypatch):
        monkeypatch.setenv("TUFT_HOME", str(tmp_path))
        monkeypatch.delenv("TUFT_CONFIG", raising=False)
        monkeypatch.delenv("TUFT_MODEL_PATH", raising=False)
        with pytest.raises(RuntimeError, match="Cannot start TuFT"):
            tuft.init()


class TestShutdown:
    @patch("tuft.runtime.discover", return_value="http://localhost:10610")
    @patch("tuft.runtime.tinker.ServiceClient")
    def test_shutdown_resets_state(self, mock_client_cls, mock_discover):
        mock_client_cls.return_value = MagicMock()
        tuft.init()
        assert tuft.is_initialized()
        tuft.shutdown()
        assert not tuft.is_initialized()

    def test_shutdown_when_not_initialized(self):
        # Should not raise
        tuft.shutdown()


class TestGetServiceClient:
    @patch("tuft.runtime.discover", return_value="http://localhost:10610")
    @patch("tuft.runtime.tinker.ServiceClient")
    def test_returns_client_after_init(self, mock_client_cls, mock_discover):
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        tuft.init()
        client = tuft.get_service_client()
        assert client is mock_instance

    @patch("tuft.runtime.discover", return_value="http://localhost:10610")
    @patch("tuft.runtime.tinker.ServiceClient")
    def test_auto_init_on_get_service_client(self, mock_client_cls, mock_discover):
        mock_client_cls.return_value = MagicMock()
        # Should auto-init
        client = tuft.get_service_client()
        assert tuft.is_initialized()
        assert client is not None

    def test_raises_when_auto_connect_disabled(self, monkeypatch):
        monkeypatch.setenv("TUFT_ENABLE_AUTO_CONNECT", "0")
        with pytest.raises(RuntimeError, match="auto-connect is disabled"):
            tuft.get_service_client()


class TestIsInitialized:
    def test_false_initially(self):
        assert not tuft.is_initialized()

    @patch("tuft.runtime.discover", return_value="http://localhost:10610")
    @patch("tuft.runtime.tinker.ServiceClient")
    def test_true_after_init(self, mock_client_cls, mock_discover):
        mock_client_cls.return_value = MagicMock()
        tuft.init()
        assert tuft.is_initialized()
