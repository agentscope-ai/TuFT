"""Tests for tuft.runtime._discovery module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tuft.runtime._constants import DEFAULT_HOST, DEFAULT_PORT
from tuft.runtime._discovery import (
    _discover_from_address_file,
    _discover_from_default_port,
    _discover_from_env,
    discover,
)


@pytest.fixture
def mock_healthy():
    """Mock _check_health to return True for any address."""
    with patch("tuft.runtime._discovery._check_health", return_value=True) as m:
        yield m


@pytest.fixture
def mock_unhealthy():
    """Mock _check_health to return False for any address."""
    with patch("tuft.runtime._discovery._check_health", return_value=False) as m:
        yield m


class TestDiscoverFromEnv:
    def test_returns_address_when_set_and_healthy(self, mock_healthy, monkeypatch):
        monkeypatch.setenv("TUFT_ADDRESS", "http://myhost:9999")
        result = _discover_from_env()
        assert result == "http://myhost:9999"

    def test_returns_none_when_not_set(self, mock_healthy, monkeypatch):
        monkeypatch.delenv("TUFT_ADDRESS", raising=False)
        result = _discover_from_env()
        assert result is None

    def test_returns_none_when_unhealthy(self, mock_unhealthy, monkeypatch):
        monkeypatch.setenv("TUFT_ADDRESS", "http://dead:1234")
        result = _discover_from_env()
        assert result is None


class TestDiscoverFromAddressFile:
    def test_returns_address_when_file_exists_and_healthy(
        self, mock_healthy, tmp_path, monkeypatch
    ):
        # Override TUFT_HOME to use tmp dir
        monkeypatch.setenv("TUFT_HOME", str(tmp_path))
        addr_file = tmp_path / "tuft_current_server"
        addr_file.write_text("http://filehost:8080")
        result = _discover_from_address_file()
        assert result == "http://filehost:8080"

    def test_returns_none_when_no_file(self, mock_healthy, tmp_path, monkeypatch):
        monkeypatch.setenv("TUFT_HOME", str(tmp_path))
        result = _discover_from_address_file()
        assert result is None

    def test_returns_none_when_file_unhealthy(self, mock_unhealthy, tmp_path, monkeypatch):
        monkeypatch.setenv("TUFT_HOME", str(tmp_path))
        addr_file = tmp_path / "tuft_current_server"
        addr_file.write_text("http://dead:1111")
        result = _discover_from_address_file()
        assert result is None


class TestDiscoverFromDefaultPort:
    def test_returns_address_when_healthy(self, mock_healthy):
        result = _discover_from_default_port()
        assert result == f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"

    def test_returns_none_when_unhealthy(self, mock_unhealthy):
        result = _discover_from_default_port()
        assert result is None


class TestDiscover:
    def test_explicit_address_takes_priority(self, mock_healthy):
        result = discover(explicit_address="http://explicit:5555")
        assert result == "http://explicit:5555"

    def test_explicit_address_unhealthy_returns_none(self, mock_unhealthy):
        result = discover(explicit_address="http://dead:5555")
        assert result is None

    def test_env_takes_priority_over_file(self, tmp_path, monkeypatch):
        """When TUFT_ADDRESS is set and healthy, address file is not checked."""
        monkeypatch.setenv("TUFT_ADDRESS", "http://envhost:7777")
        monkeypatch.setenv("TUFT_HOME", str(tmp_path))
        addr_file = tmp_path / "tuft_current_server"
        addr_file.write_text("http://filehost:8888")

        with patch("tuft.runtime._discovery._check_health", return_value=True):
            result = discover()
        assert result == "http://envhost:7777"

    def test_returns_none_when_nothing_found(self, mock_unhealthy, tmp_path, monkeypatch):
        monkeypatch.delenv("TUFT_ADDRESS", raising=False)
        monkeypatch.setenv("TUFT_HOME", str(tmp_path))
        result = discover()
        assert result is None
