#!/usr/bin/env python3
"""Verify that installed GPU runtime versions match TuFT's exact pins."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import tomllib
from pathlib import Path


PINNED_PACKAGES = ("torch", "vllm")


def exact_project_pins(pyproject_path: Path) -> dict[str, str]:
    """Return exact versions for the GPU packages guarded by release builds."""
    with pyproject_path.open("rb") as file:
        dependencies = tomllib.load(file)["project"]["dependencies"]

    pins: dict[str, str] = {}
    for dependency in dependencies:
        requirement = dependency.split(";", 1)[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*==\s*([^\s]+)", requirement)
        if match and match.group(1).lower() in PINNED_PACKAGES:
            pins[match.group(1).lower()] = match.group(2)

    missing = set(PINNED_PACKAGES) - pins.keys()
    if missing:
        raise RuntimeError("Expected exact project pins for: " + ", ".join(sorted(missing)))
    return pins


def verify_runtime_versions(pyproject_path: Path) -> None:
    """Raise when an installed version differs from its project pin."""
    for package, expected in exact_project_pins(pyproject_path).items():
        installed = importlib.metadata.version(package)
        if installed != expected:
            raise RuntimeError(
                f"{package} version mismatch: installed {installed}, expected {expected}"
            )
        print(f"Verified {package}=={installed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pyproject.toml",
    )
    args = parser.parse_args()
    verify_runtime_versions(args.pyproject)


if __name__ == "__main__":
    main()
