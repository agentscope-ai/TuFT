"""Startup validation of the FSDP attention implementation (no GPU, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest
import transformers.modeling_flash_attention_utils as fa_utils
import transformers.utils
from transformers.utils.generic import is_flash_attention_requested

from tuft.backends.attention import (
    FLASH_ATTN2_KERNEL,
    FLASH_ATTN2_KERNEL_REVISION,
    resolve_attn_implementation,
    resolve_attn_kernel,
    validate_attn_implementation,
)
from tuft.config import ModelConfig


@pytest.fixture
def flash_attn_installed(monkeypatch):
    """Toggle whether transformers sees an installed flash_attn package."""

    def set_installed(installed: bool) -> None:
        entry = dict(fa_utils.FLASH_ATTENTION_COMPATIBILITY_MATRIX[2])
        entry["general_availability_check"] = lambda: installed
        monkeypatch.setitem(fa_utils.FLASH_ATTENTION_COMPATIBILITY_MATRIX, 2, entry)

    return set_installed


@pytest.fixture
def hub_kernel(monkeypatch):
    """Stub kernels.has_kernel and record the calls it receives."""

    kernels = pytest.importorskip("kernels")
    calls: list[dict] = []

    def set_available(available: bool | Exception) -> list[dict]:
        def fake_has_kernel(repo_id, **kwargs):
            calls.append({"repo_id": repo_id, **kwargs})
            if isinstance(available, Exception):
                raise available
            return available

        monkeypatch.setattr(transformers.utils, "is_kernels_available", lambda: True)
        monkeypatch.setattr(kernels, "has_kernel", fake_has_kernel)
        return calls

    return set_available


@pytest.mark.parametrize("attn", [None, "sdpa", "eager", "flex_attention"])
def test_non_kernel_implementations_need_no_kernel(attn):
    assert resolve_attn_kernel(attn) is None
    validate_attn_implementation(attn)


def test_installed_flash_attn_is_used_directly(flash_attn_installed):
    flash_attn_installed(True)
    assert resolve_attn_implementation("flash_attention_2") == "flash_attention_2"
    assert resolve_attn_kernel("flash_attention_2") is None


def test_missing_flash_attn_uses_pinned_hub_kernel(flash_attn_installed):
    flash_attn_installed(False)
    pinned = resolve_attn_implementation("flash_attention_2")
    assert pinned == f"{FLASH_ATTN2_KERNEL}@{FLASH_ATTN2_KERNEL_REVISION}"
    # transformers must still treat the pinned kernel as FlashAttention-2.
    assert is_flash_attention_requested(requested_attention_implementation=pinned, version=2)
    assert resolve_attn_kernel(pinned) == (FLASH_ATTN2_KERNEL, FLASH_ATTN2_KERNEL_REVISION)


def test_unpinned_flash_attention_falls_back_like_transformers(flash_attn_installed):
    flash_attn_installed(False)
    assert resolve_attn_kernel("flash_attention_2") == ("kernels-community/flash-attn2", None)


@pytest.mark.parametrize(
    ("attn", "expected"),
    [
        ("kernels-community/flash-attn2", ("kernels-community/flash-attn2", None)),
        ("kernels-community/flash-attn2@v2", ("kernels-community/flash-attn2", "v2")),
        (
            "kernels-community/vllm-flash-attn3:flash_attn_func",
            ("kernels-community/vllm-flash-attn3", None),
        ),
    ],
)
def test_hub_kernel_ids_are_parsed(attn, expected):
    assert resolve_attn_kernel(attn) == expected


def test_missing_kernels_package_fails_with_remedies(flash_attn_installed, monkeypatch):
    flash_attn_installed(False)
    monkeypatch.setattr(transformers.utils, "is_kernels_available", lambda: False)
    with pytest.raises(ValueError) as excinfo:
        validate_attn_implementation("flash_attention_2")
    message = str(excinfo.value)
    assert "kernels" in message
    assert "torch " in message and "CUDA" in message
    assert "attn_implementation: sdpa" in message


def test_available_hub_kernel_passes(flash_attn_installed, hub_kernel):
    flash_attn_installed(False)
    calls = hub_kernel(True)
    validate_attn_implementation("flash_attention_2")
    (call,) = calls
    assert call["repo_id"] == FLASH_ATTN2_KERNEL
    assert call["revision"] == FLASH_ATTN2_KERNEL_REVISION
    assert call["version"] is None
    # The server process must not touch the GPU.
    assert call["check_arch"] is False


def test_unpinned_hub_kernel_uses_transformers_kernel_version(hub_kernel):
    calls = hub_kernel(True)
    validate_attn_implementation("kernels-community/vllm-flash-attn3")
    assert calls[0]["revision"] is None
    assert isinstance(calls[0]["version"], int)


def test_hub_kernel_without_build_fails(flash_attn_installed, hub_kernel):
    flash_attn_installed(False)
    hub_kernel(False)
    with pytest.raises(ValueError, match="has no build for this environment"):
        validate_attn_implementation("flash_attention_2")


def test_hub_kernel_resolution_error_fails(flash_attn_installed, hub_kernel):
    flash_attn_installed(False)
    hub_kernel(OSError("offline"))
    with pytest.raises(ValueError, match=f"resolving {FLASH_ATTN2_KERNEL} failed: offline"):
        validate_attn_implementation("flash_attention_2")


@pytest.mark.parametrize("via_override", [False, True])
def test_fsdp_backend_rejects_unloadable_attention_at_startup(
    flash_attn_installed, hub_kernel, via_override
):
    from tuft.backends.fsdp_training_backend import FSDPTrainingBackend

    flash_attn_installed(False)
    hub_kernel(False)
    attn = {"attn_implementation": "flash_attention_2"}
    config = ModelConfig(
        model_name="qwen",
        model_path=Path("/tmp/Qwen3-0.6B"),
        max_model_len=1024,
        training_backend="fsdp",
        **({"fsdp_override_config": attn} if via_override else attn),
    )
    with pytest.raises(ValueError, match="flash_attention_2"):
        FSDPTrainingBackend(config)
