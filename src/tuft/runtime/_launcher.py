"""Embedded mode launcher: start a TuFT server as a subprocess.

Responsibilities:
- Launch `tuft launch` in a subprocess
- Poll healthz until ready (timeout configurable)
- Write address file for discovery
- Register atexit cleanup (terminate subprocess + remove address file)
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

from ._constants import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    HEALTHZ_PATH,
    HEALTHZ_TIMEOUT,
    STARTUP_POLL_INTERVAL,
    STARTUP_TIMEOUT,
    get_address_file,
)


logger = logging.getLogger(__name__)


class EmbeddedServer:
    """Manages a TuFT server subprocess (embedded mode)."""

    def __init__(
        self,
        config_path: Path,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = STARTUP_TIMEOUT,
    ):
        self.config_path = config_path
        self.host = host
        self.port = port
        self.timeout = timeout
        self.process: Optional[subprocess.Popen] = None
        self._address: Optional[str] = None
        self._atexit_registered = False

    @property
    def address(self) -> Optional[str]:
        return self._address

    def start(self) -> str:
        """Start the server subprocess and wait for it to be healthy.

        Returns:
            The address of the running server.

        Raises:
            RuntimeError: If the server fails to start within the timeout.
        """
        if self.process is not None and self.process.poll() is None:
            # Already running
            if self._address:
                return self._address

        cmd = [
            sys.executable,
            "-m",
            "tuft.cli",
            "launch",
            "--config",
            str(self.config_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        env = os.environ.copy()
        # Ensure subprocess doesn't inherit auto-connect to avoid recursion
        env.pop("TUFT_ADDRESS", None)

        logger.info("Starting embedded TuFT server: %s", " ".join(cmd))

        # When log level is DEBUG or server fails, show subprocess output directly.
        # Otherwise pipe it for collection on failure.
        # For now, always inherit stderr so users can see server startup progress.
        inherit_io = True

        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=None if inherit_io else subprocess.PIPE,
            stderr=None if inherit_io else subprocess.PIPE,
            # Use a new process group so we can cleanly terminate
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )

        # Wait for healthz
        self._address = f"http://{self.host}:{self.port}"
        if not self._wait_for_healthy():
            # Collect stderr for debugging
            stderr_output = ""
            if self.process.stderr:
                try:
                    stderr_output = self.process.stderr.read().decode(errors="replace")[:2000]
                except Exception:
                    pass
            self._terminate()
            raise RuntimeError(
                f"TuFT embedded server failed to start within {self.timeout}s. "
                f"Config: {self.config_path}\n"
                f"Stderr: {stderr_output}"
            )

        # Write address file
        self._write_address_file()

        # Register cleanup
        if not self._atexit_registered:
            atexit.register(self.shutdown)
            self._atexit_registered = True

        logger.info("Embedded TuFT server ready at %s (pid=%d)", self._address, self.process.pid)
        return self._address

    def _wait_for_healthy(self) -> bool:
        """Poll healthz until healthy or timeout."""
        url = f"{self._address}{HEALTHZ_PATH}"
        deadline = time.monotonic() + self.timeout

        while time.monotonic() < deadline:
            # Check if process died
            if self.process and self.process.poll() is not None:
                return False
            try:
                resp = httpx.get(url, timeout=HEALTHZ_TIMEOUT)
                if resp.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.TimeoutException, OSError):
                pass
            time.sleep(STARTUP_POLL_INTERVAL)

        return False

    def _write_address_file(self) -> None:
        """Write the server address to the address file."""
        address_file = get_address_file()
        try:
            address_file.parent.mkdir(parents=True, exist_ok=True)
            address_file.write_text(self._address or "")
            logger.debug("Wrote address file: %s", address_file)
        except OSError as e:
            logger.warning("Failed to write address file %s: %s", address_file, e)

    def _remove_address_file(self) -> None:
        """Remove the address file if it points to our address."""
        address_file = get_address_file()
        try:
            if address_file.exists():
                content = address_file.read_text().strip()
                if content == self._address:
                    address_file.unlink()
                    logger.debug("Removed address file: %s", address_file)
        except OSError:
            pass

    def _terminate(self) -> None:
        """Terminate the subprocess."""
        if self.process is None:
            return
        if self.process.poll() is not None:
            return
        try:
            if sys.platform != "win32":
                # Kill the entire process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
            self.process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            try:
                self.process.kill()
                self.process.wait(timeout=5)
            except Exception:
                pass
        logger.info("Terminated embedded TuFT server (pid=%d)", self.process.pid)

    def shutdown(self) -> None:
        """Stop the embedded server and clean up."""
        self._terminate()
        self._remove_address_file()
        self.process = None
        self._address = None

    @property
    def is_running(self) -> bool:
        """Check if the subprocess is still running."""
        return self.process is not None and self.process.poll() is None
