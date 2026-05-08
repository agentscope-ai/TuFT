"""Multi-level service discovery for TuFT.

Discovery order:
1. TUFT_ADDRESS environment variable
2. Address file (~/.tuft/tuft_current_server) + healthz validation
3. Process scan (psutil: look for tuft/uvicorn processes)
4. Default port probe (http://127.0.0.1:10610)

Each level validates via GET /api/v1/healthz before returning.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
import psutil

from ._constants import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    ENV_TUFT_ADDRESS,
    HEALTHZ_PATH,
    HEALTHZ_TIMEOUT,
    get_address_file,
)


logger = logging.getLogger(__name__)


def _check_health(address: str) -> bool:
    """Send a GET to /api/v1/healthz and return True if status 200."""
    url = address.rstrip("/") + HEALTHZ_PATH
    try:
        resp = httpx.get(url, timeout=HEALTHZ_TIMEOUT)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        return False


def _discover_from_env() -> Optional[str]:
    """Level 1: Check TUFT_ADDRESS environment variable."""
    address = os.environ.get(ENV_TUFT_ADDRESS)
    if address:
        address = address.strip()
        if _check_health(address):
            logger.info("Discovered TuFT service via TUFT_ADDRESS: %s", address)
            return address
        logger.debug("TUFT_ADDRESS=%s is set but not healthy", address)
    return None


def _discover_from_address_file() -> Optional[str]:
    """Level 2: Read address from ~/.tuft/tuft_current_server."""
    address_file = get_address_file()
    if not address_file.exists():
        return None
    try:
        address = address_file.read_text().strip()
    except OSError:
        return None
    if not address:
        return None
    if _check_health(address):
        logger.info("Discovered TuFT service via address file: %s", address)
        return address
    logger.debug("Address file points to %s but not healthy", address)
    return None


def _discover_from_process_scan() -> Optional[str]:
    """Level 3: Scan local processes for a running tuft server."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmdline_str = " ".join(cmdline)
            # Look for 'tuft launch' or 'uvicorn' serving tuft
            if "tuft" in cmdline_str and ("launch" in cmdline_str or "uvicorn" in cmdline_str):
                # Try to extract port from --port argument
                port = DEFAULT_PORT
                for i, arg in enumerate(cmdline):
                    if arg in ("--port", "-p") and i + 1 < len(cmdline):
                        try:
                            port = int(cmdline[i + 1])
                        except ValueError:
                            pass
                        break
                address = f"http://{DEFAULT_HOST}:{port}"
                if _check_health(address):
                    logger.info(
                        "Discovered TuFT service via process scan (pid=%s): %s",
                        proc.info["pid"],
                        address,
                    )
                    return address
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return None


def _discover_from_default_port() -> Optional[str]:
    """Level 4: Probe the default port."""
    address = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
    if _check_health(address):
        logger.info("Discovered TuFT service on default port: %s", address)
        return address
    return None


def discover(explicit_address: Optional[str] = None) -> Optional[str]:
    """Run multi-level service discovery.

    Args:
        explicit_address: If provided, check this address first (highest priority).

    Returns:
        A validated service address, or None if no healthy service found.
    """
    # Level 0: Explicit address passed to init()
    if explicit_address:
        explicit_address = explicit_address.strip()
        if _check_health(explicit_address):
            logger.info("Connected to explicitly provided address: %s", explicit_address)
            return explicit_address
        logger.warning("Explicit address %s is not healthy", explicit_address)
        return None

    # Level 1-4: progressive discovery
    for level_fn in (
        _discover_from_env,
        _discover_from_address_file,
        _discover_from_process_scan,
        _discover_from_default_port,
    ):
        result = level_fn()
        if result:
            return result

    return None
