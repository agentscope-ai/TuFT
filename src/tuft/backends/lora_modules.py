"""Shared LoRA target-module resolution for training backends."""

from __future__ import annotations

from typing import Iterable

from tinker.types import LoraConfig

from tuft.backends.vllm_lora_compat import resolve_model_series_and_architecture


MODULE_MAP = {
    "llama": {
        "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "unembed": ["lm_head"],
    },
    "qwen": {
        "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "unembed": [],  # Qwen's unembed group is intentionally unsupported.
    },
}

# Qwen3.5 and Qwen3.8 share Transformers' ``qwen3_5`` architecture. Three out
# of every four text layers use Gated DeltaNet rather than full attention, and
# PEFT suffix-matches target names. Keeping the ``linear_attn.`` qualifier is
# important: it selects only the text DeltaNet projections and cannot match
# similarly named modules in the multimodal vision encoder.
#
# Adding these modules widened the geometry resolved for Qwen3.5/3.8 (issue
# #149). The strict set-equality validations downstream therefore reject LoRA
# state recorded before the widening; ``gated_deltanet_mismatch_hint`` gives
# those rejections a targeted explanation.
QWEN3_5_GATED_DELTANET_TARGET_MODULES = [
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.out_proj",
]

ARCHITECTURE_MODULE_MAP = {
    "qwen3_5": {
        "attn": QWEN3_5_GATED_DELTANET_TARGET_MODULES,
    },
}


def _assemble_target_modules(
    series: str,
    architecture: str | None,
    *,
    train_attn: bool,
    train_mlp: bool,
    train_unembed: bool,
) -> list[str]:
    target_modules: list[str] = []
    if train_attn:
        target_modules.extend(MODULE_MAP[series]["attn"])
        if architecture in ARCHITECTURE_MODULE_MAP:
            target_modules.extend(ARCHITECTURE_MODULE_MAP[architecture]["attn"])
    if train_mlp:
        target_modules.extend(MODULE_MAP[series]["mlp"])
    if train_unembed:
        target_modules.extend(MODULE_MAP[series]["unembed"])
    return target_modules


def resolve_target_modules(
    model_path: str,
    *,
    train_attn: bool,
    train_mlp: bool,
    train_unembed: bool,
) -> list[str]:
    """Resolve public LoRA modifier flags into concrete module names."""

    series, architecture = resolve_model_series_and_architecture(model_path)
    if series is None:
        raise ValueError(f"Unsupported model series: {model_path}")

    return _assemble_target_modules(
        series,
        architecture,
        train_attn=train_attn,
        train_mlp=train_mlp,
        train_unembed=train_unembed,
    )


def get_target_modules(model_path: str, lora_config: LoraConfig) -> list[str]:
    """Resolve a Tinker ``LoraConfig`` using the shared model-series map."""

    return resolve_target_modules(
        model_path,
        train_attn=lora_config.train_attn,
        train_mlp=lora_config.train_mlp,
        train_unembed=lora_config.train_unembed,
    )


def achievable_target_module_sets(model_path: str) -> list[list[str]] | None:
    """Every distinct module list client modifier flags can resolve to.

    Returns None when the model series is unknown, i.e. when an explicit
    server geometry is the only way to run the model at all. Used to fail
    fast at startup on configs whose explicit geometry no client request
    could ever match.
    """

    series, architecture = resolve_model_series_and_architecture(model_path)
    if series is None:
        return None
    achievable: list[list[str]] = []
    for train_attn in (True, False):
        for train_mlp in (True, False):
            for train_unembed in (True, False):
                modules = _assemble_target_modules(
                    series,
                    architecture,
                    train_attn=train_attn,
                    train_mlp=train_mlp,
                    train_unembed=train_unembed,
                )
                if modules and modules not in achievable:
                    achievable.append(modules)
    return achievable


def gated_deltanet_mismatch_hint(
    actual_modules: Iterable[str], expected_modules: Iterable[str]
) -> str | None:
    """Explain module-set mismatches caused by the Gated DeltaNet widening.

    Returns a notice when the two sets differ exactly by (a subset of) the
    Qwen3.5/3.8 Gated DeltaNet projections — the signature of LoRA state or
    server config recorded before issue #149 widened the resolved geometry —
    and None for every other mismatch.
    """

    difference = set(actual_modules) ^ set(expected_modules)
    if not difference or not difference <= set(QWEN3_5_GATED_DELTANET_TARGET_MODULES):
        return None
    return (
        "The sets differ only by the Qwen3.5/3.8 Gated DeltaNet projections "
        "(linear_attn.*): this TuFT release widened the LoRA geometry resolved for "
        "Gated DeltaNet models (issue #149), and LoRA state recorded before the "
        "widening is incompatible with geometry resolved after it. Start a new "
        "training run on this release to adopt the widened geometry."
    )
