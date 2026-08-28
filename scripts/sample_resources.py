#!/usr/bin/env python3
"""Append a 5s resource sample line while a campaign arm runs.

Written for the t16/t24 arms of the saturated axis, which die with SIGKILL and
leave no error in the server log: the sampler records host memory, memory PSI
and the RSS of the eval driver / TuFT server / simulator so the state at the
moment of death is visible afterwards.

Usage: sample_resources.py OUT_CSV [INTERVAL_S]
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path


PATTERNS = {
    "eval": "run_eval_matrix.py",
    "server": "tuft launch",
    "sim": "simulator/run.py",
}


def _mem_available_mb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return -1.0


def _psi_full_avg10() -> float:
    try:
        text = Path("/proc/pressure/memory").read_text()
    except OSError:
        return -1.0
    m = re.search(r"full avg10=([0-9.]+)", text)
    return float(m.group(1)) if m else -1.0


def _rss_mb(pattern: str) -> float:
    """Summed RSS of processes whose cmdline contains pattern, in MiB."""
    total = 0.0
    for pid in Path("/proc").iterdir():
        if not pid.name.isdigit():
            continue
        try:
            cmdline = (pid / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if pattern not in cmdline:
                continue
            for line in (pid / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) / 1024
                    break
        except (OSError, ValueError):
            continue
    return total


def _gpu_mem_mb() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    return "|".join(line.strip() for line in out.splitlines() if line.strip())


def main(argv: list[str]) -> int:
    out = Path(argv[1])
    interval = float(argv[2]) if len(argv) > 2 else 5.0
    if not out.exists():
        out.write_text("time,mem_avail_mb,psi_full_avg10,eval_rss_mb,server_rss_mb,sim_rss_mb,gpu_mem_mb\n")
    while True:
        row = [
            time.strftime("%H:%M:%S"),
            f"{_mem_available_mb():.0f}",
            f"{_psi_full_avg10():.2f}",
            *(f"{_rss_mb(p):.0f}" for p in PATTERNS.values()),
            _gpu_mem_mb(),
        ]
        with out.open("a") as f:
            f.write(",".join(row) + "\n")
        time.sleep(interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
