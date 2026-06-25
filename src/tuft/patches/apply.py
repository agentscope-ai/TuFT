"""Apply trinity-rft vLLM compatibility patches into site-packages.

Usage:
    python -m tuft.patches.apply          # apply patches
    python -m tuft.patches.apply --check  # dry-run check only

These patches are needed because trinity-rft's vllm_patch module
has hard-coded version upper bounds and import paths that break
with vLLM >= 0.20.0.

Patches applied:
  1. worker_patch.py  - Extends version check to 0.23.0, adds v23 variant
                        for new per-request logprobs API.
  2. api_patch_v13.py - Fixes moved import path for `log_non_default_args`.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path


def _get_trinity_patch_dir() -> Path:
    """Locate trinity's vllm_patch directory in site-packages."""
    spec = importlib.util.find_spec("trinity.common.models.vllm_patch")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(
            "Cannot find trinity.common.models.vllm_patch in the current environment. "
            "Is trinity-rft installed?"
        )
    return Path(list(spec.submodule_search_locations)[0])


def _get_source_dir() -> Path:
    """Get the directory containing our patch source files."""
    return Path(__file__).parent / "trinity_vllm"


def apply(*, dry_run: bool = False) -> list[str]:
    """Apply patches. Returns list of files patched."""
    target_dir = _get_trinity_patch_dir()
    source_dir = _get_source_dir()

    patched: list[str] = []
    for src_file in sorted(source_dir.glob("*.py")):
        if src_file.name.startswith("_"):
            continue
        dest = target_dir / src_file.name
        if dry_run:
            if dest.exists():
                # Compare contents
                if src_file.read_text() != dest.read_text():
                    patched.append(f"[NEEDS UPDATE] {dest}")
                else:
                    patched.append(f"[OK] {dest}")
            else:
                patched.append(f"[MISSING TARGET] {dest}")
        else:
            shutil.copy2(src_file, dest)
            patched.append(str(dest))

    return patched


def main() -> None:
    """CLI entry point."""
    dry_run = "--check" in sys.argv

    if dry_run:
        print("=== Dry-run: checking patch status ===")
    else:
        print("=== Applying trinity-rft vLLM patches ===")

    try:
        results = apply(dry_run=dry_run)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    for r in results:
        print(f"  {r}")

    if not dry_run:
        print(f"\nDone. {len(results)} file(s) patched.")
        # Invalidate any cached bytecode
        for pyc in _get_trinity_patch_dir().glob("__pycache__/*.pyc"):
            pyc.unlink()
        print("Cleared __pycache__ for trinity.common.models.vllm_patch")


if __name__ == "__main__":
    main()
