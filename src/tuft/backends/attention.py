"""Startup check for the attention implementation an FSDP model requests.

FSDP workers build the model lazily, on the first training request, so an
attention backend that cannot load used to fail long after the server
reported healthy. ``flash_attention_N`` uses the flash_attn package when it
is installed; otherwise transformers falls back to the matching Hugging Face
kernel (e.g. ``kernels-community/flash-attn2``) through the ``kernels``
package. This module resolves the same way when the backend is constructed,
so a missing backend stops the server at startup instead.
"""

from __future__ import annotations

import logging
import re
import sys


logger = logging.getLogger(__name__)

_FLASH_ATTENTION = re.compile(r"^flash_attention_(\d+)$")
# transformers' hub kernel pattern: "org/repo", optionally "@revision" and ":function".
_HUB_KERNEL = re.compile(r"^[^/:]+/[^/:]+(?:@[^/:]+)?(?::[^/:]+)?$")


def resolve_attn_kernel(attn_implementation: str | None) -> tuple[str, str | None] | None:
    """Return the (repo_id, revision) Hub kernel transformers will load.

    Returns None when the implementation needs no Hub kernel: sdpa, eager,
    flex, or a ``flash_attention_N`` whose flash_attn package is installed.
    """

    if attn_implementation is None:
        return None
    match = _FLASH_ATTENTION.match(attn_implementation)
    if match is not None:
        from transformers.modeling_flash_attention_utils import (
            FLASH_ATTENTION_COMPATIBILITY_MATRIX,
            FLASH_ATTN_KERNEL_FALLBACK,
        )

        entry = FLASH_ATTENTION_COMPATIBILITY_MATRIX.get(int(match.group(1)))
        if entry is not None and entry["general_availability_check"]():
            return None
        if attn_implementation not in FLASH_ATTN_KERNEL_FALLBACK:
            return None
        return FLASH_ATTN_KERNEL_FALLBACK[attn_implementation], None
    if _HUB_KERNEL.match(attn_implementation) is None:
        return None
    repo_id = attn_implementation.split(":", 1)[0]
    repo_id, _, revision = repo_id.partition("@")
    return repo_id.strip(), revision.strip() or None


def _environment() -> str:
    import torch

    cuda = getattr(torch.version, "cuda", None)
    return (
        f"torch {torch.__version__}, CUDA {cuda or 'none'}, "
        f"cp{sys.version_info.major}{sys.version_info.minor}, "
        f"cxx11abi={torch.compiled_with_cxx11_abi()}"
    )


def _unavailable(attn_implementation: str, reason: str) -> ValueError:
    return ValueError(
        f"attn_implementation={attn_implementation!r} cannot be loaded: {reason} "
        f"({_environment()}). Fetch the kernel with `python scripts/install_flash_attn.py` "
        "(add `--source modelscope` to download it from ModelScope), install a flash_attn "
        "wheel that matches this environment, or set attn_implementation: sdpa."
    )


def validate_attn_implementation(attn_implementation: str | None) -> None:
    """Raise ValueError if ``attn_implementation`` cannot be loaded here.

    The check only resolves a build variant; it neither downloads the kernel
    nor touches the GPU, so it is safe to run in the server process. Kernel
    overrides in ``LOCAL_KERNELS`` (e.g. a ModelScope copy) are honored.
    """

    if attn_implementation is None:
        return
    kernel = resolve_attn_kernel(attn_implementation)
    if kernel is None:
        return
    repo_id, revision = kernel

    from transformers.utils import is_kernels_available

    if not is_kernels_available():
        raise _unavailable(
            attn_implementation,
            f"the flash_attn package is not installed and the `kernels` package needed "
            f"for the {repo_id} fallback is missing or has an incompatible version",
        )

    from kernels import has_kernel
    from transformers.integrations.hub_kernels import get_attn_kernel_version

    version = None if revision is not None else get_attn_kernel_version(repo_id)
    try:
        available = has_kernel(repo_id, revision=revision, version=version, check_arch=False)
    except Exception as exc:
        raise _unavailable(attn_implementation, f"resolving {repo_id} failed: {exc}") from exc
    if not available:
        raise _unavailable(attn_implementation, f"{repo_id} has no build for this environment")
    logger.info(
        "attn_implementation=%s resolves to Hub kernel %s (%s)",
        attn_implementation,
        repo_id,
        f"revision {revision}" if revision is not None else f"version {version}",
    )
