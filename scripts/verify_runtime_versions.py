#!/usr/bin/env python3
"""Verify that installed GPU runtime versions match TuFT's exact pins.

Comparison is PEP 440-aware: a pin of ``2.11.0`` accepts official local build
variants such as ``2.11.0+cu129`` or ``2.11.0+cpu`` (the behavior of a
``==2.11.0`` specifier), while a pin that itself carries a local segment
requires an exact match. Backend consistency -- did we get the CUDA/CPU
variant we asked for, and do the installed packages agree on a CUDA major
version -- is validated separately, optionally against an expected backend
(``--expected-backend cu130|cu129|cpu``, e.g. the value the installer records
in ``$TUFT_HOME/torch-backend``).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple


try:
    from packaging.markers import InvalidMarker, Marker
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, Version
except ImportError as exc:  # pragma: no cover - packaging ships with TuFT's deps
    raise SystemExit(
        "verify_runtime_versions.py requires the 'packaging' library, which is "
        "installed alongside TuFT's dependencies. Install it with: pip install packaging"
    ) from exc


PINNED_PACKAGES = ("torch", "vllm")


class Pin(NamedTuple):
    """An exact version pin plus its optional environment marker."""

    version: str
    marker: str | None


def exact_project_pins(pyproject_path: Path) -> dict[str, Pin]:
    """Return exact versions for the GPU packages guarded by release builds."""
    with pyproject_path.open("rb") as file:
        dependencies = tomllib.load(file)["project"]["dependencies"]

    pins: dict[str, Pin] = {}
    for dependency in dependencies:
        requirement, _, marker = dependency.partition(";")
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*==\s*([^\s]+)", requirement.strip())
        if match and match.group(1).lower() in PINNED_PACKAGES:
            pins[match.group(1).lower()] = Pin(match.group(2), marker.strip() or None)

    missing = set(PINNED_PACKAGES) - pins.keys()
    if missing:
        raise RuntimeError("Expected exact project pins for: " + ", ".join(sorted(missing)))
    return pins


def marker_applies(marker: str) -> bool:
    """Evaluate an environment marker; unknown markers count as applying."""
    try:
        return Marker(marker).evaluate()
    except InvalidMarker:
        return True


def version_satisfies_pin(installed: str, pinned: str) -> bool:
    """PEP 440 ``==`` check: local build suffixes satisfy a public pin."""
    try:
        specifier = SpecifierSet(f"=={pinned}")
        candidate = Version(installed)
    except (InvalidSpecifier, InvalidVersion):
        return installed == pinned
    return specifier.contains(candidate, prereleases=True)


def local_segment(version: str) -> str | None:
    """Return the PEP 440 local segment of a version (``cu129`` etc.), if any."""
    try:
        return Version(version).local
    except InvalidVersion:
        return version.split("+", 1)[1] if "+" in version else None


def backend_from_local(local: str | None) -> str | None:
    """Normalize a local version segment into a backend tag (cu129, cpu, ...)."""
    if not local:
        return None
    normalized = local.lower()
    if normalized.startswith("cpu"):
        return "cpu"
    match = re.match(r"cu(\d+)", normalized)
    if match:
        return f"cu{match.group(1)}"
    return normalized


def cuda_major(backend: str) -> str | None:
    """CUDA major version of a cuNNN backend tag: cu129 -> 12, cu130 -> 13."""
    match = re.fullmatch(r"cu(\d+)", backend)
    if match is None:
        return None
    digits = match.group(1)
    return digits[:-1] if len(digits) > 1 else digits


def check_backend_consistency(
    installed_versions: dict[str, str], expected_backend: str | None
) -> list[str]:
    """Return human-readable problems with the installed backend variants.

    ``expected_backend`` accepts the values the installer records: ``cpu``,
    ``cuNNN``, or ``default``/``None`` (no expectation about torch's variant).
    """
    problems: list[str] = []
    backends = {
        package: backend_from_local(local_segment(version))
        for package, version in installed_versions.items()
    }
    torch_backend = backends.get("torch")

    expected = (expected_backend or "").strip().lower()
    if expected and expected != "default":
        if expected == "cpu":
            if torch_backend != "cpu":
                problems.append(
                    "expected a CPU-only torch build (+cpu), but installed torch is "
                    + (f"a +{torch_backend} build" if torch_backend else "the default build")
                )
        elif re.fullmatch(r"cu\d+", expected):
            if torch_backend is None and "torch" in installed_versions:
                problems.append(
                    f"expected a +{expected} torch build (from the PyTorch {expected} index), "
                    "but installed torch has no local build tag (default PyPI build)"
                )
            elif torch_backend is not None and torch_backend != expected:
                problems.append(
                    f"expected a +{expected} torch build, but installed torch is +{torch_backend}"
                )
        else:
            problems.append(
                f"unknown expected backend '{expected_backend}' (expected cpu, cuNNN, or default)"
            )

    # Cross-package checks based on the local segments that are present.
    cuda_backends = {
        package: backend
        for package, backend in backends.items()
        if backend is not None and backend != "cpu" and backend.startswith("cu")
    }
    majors = {cuda_major(backend) for backend in cuda_backends.values()}
    majors.discard(None)
    if len(majors) > 1:
        details = ", ".join(f"{pkg} (+{backend})" for pkg, backend in sorted(cuda_backends.items()))
        problems.append(f"CUDA major version mismatch across packages: {details}")
    if torch_backend == "cpu" and cuda_backends:
        details = ", ".join(sorted(cuda_backends))
        problems.append(f"torch is a CPU-only build but CUDA builds are installed for: {details}")

    return problems


def verify_runtime_versions(
    pyproject_path: Path,
    expected_backend: str | None = None,
    get_version: Callable[[str], str] = importlib.metadata.version,
) -> None:
    """Raise when an installed version or backend variant differs from its pin."""
    installed_versions: dict[str, str] = {}
    for package, pin in exact_project_pins(pyproject_path).items():
        try:
            installed = get_version(package)
        except importlib.metadata.PackageNotFoundError:
            if pin.marker is not None and not marker_applies(pin.marker):
                print(f'Skipped {package}=={pin.version}: marker "{pin.marker}" does not apply')
                continue
            raise RuntimeError(
                f"{package} is pinned to {pin.version} but is not installed"
            ) from None
        if not version_satisfies_pin(installed, pin.version):
            raise RuntimeError(
                f"{package} version mismatch: installed {installed} does not satisfy "
                f"=={pin.version}"
            )
        installed_versions[package] = installed
        local = local_segment(installed)
        build = f" (build: +{local})" if local else ""
        print(f"Verified {package}=={installed} against pin =={pin.version}{build}")

    problems = check_backend_consistency(installed_versions, expected_backend)
    if problems:
        raise RuntimeError("Backend consistency check failed:\n- " + "\n- ".join(problems))
    if expected_backend:
        print(f"Backend consistency OK (expected: {expected_backend})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pyproject.toml",
    )
    parser.add_argument(
        "--expected-backend",
        default=None,
        help=(
            "Torch backend variant the environment was installed with "
            "(cpu, cuNNN such as cu130, or default). The installer records this "
            "value in $TUFT_HOME/torch-backend."
        ),
    )
    args = parser.parse_args()
    try:
        verify_runtime_versions(args.pyproject, args.expected_backend)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
