"""Run the installer's shell test suite for torch backend selection.

The actual assertions live in tests/shell/test_torch_backend.sh, which sources
scripts/install.sh and exercises the shared tuft_* backend-resolution helpers
with stubbed nvidia-smi/uname binaries (deterministic on GPU and GPU-less
hosts alike). This wrapper makes the suite part of the regular pytest run.
"""

import shutil
import subprocess
from pathlib import Path

import pytest


_SHELL_SUITE = Path(__file__).resolve().parent / "shell" / "test_torch_backend.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_torch_backend_selection_shell_suite():
    result = subprocess.run(
        ["bash", str(_SHELL_SUITE)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"shell suite failed (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
