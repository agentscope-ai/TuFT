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

# Qwen3.5-based models share Transformers' ``qwen3_5`` architecture (Qwen3.6
# and Qwen3.8 use it too). Three out of every four text layers use Gated
# DeltaNet rather than full attention, and PEFT suffix-matches target names.
# Keeping the ``linear_attn.`` qualifier is important: it selects only the
# text DeltaNet projections and cannot match similarly named modules in the
# multimodal vision encoder.
#
# Tinker's public ``train_attn`` geometry covers the Q/K/V, Z, and output
# projections. Its Qwen3.5 backend represents Q/K/V separately, while HF
# exposes one fused ``in_proj_qkv`` module. The recurrent A/B gate projections
# are supported as an operator opt-in but are intentionally not part of the
# default, so TuFT's public defaults match Tinker's API.
#
# Adding the default modules changed the resolved target list for Qwen3.5-based
# models (issue #149). Module lists must match exactly, so LoRA state saved
# before the change is rejected; ``gated_deltanet_mismatch_hint`` explains
# why in those errors.
#
# The MoE variant (model type ``qwen3_5_moe``, e.g. Qwen3.6-35B-A3B) resolves
# the same list. Its routed experts are fused parameters with no per-expert
# Linear modules, so the mlp names match only the shared expert: LoRA trains
# attention, these Gated DeltaNet projections, and the shared expert, while
# the routed experts stay frozen. vLLM applies LoRA to the same modules when
# serving, so trained and served modules agree.
QWEN3_5_GATED_DELTANET_TARGET_MODULES = [
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.out_proj",
]

QWEN3_5_DEFAULT_GATED_DELTANET_TARGET_MODULES = [
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
]

QWEN3_5_OPTIONAL_GATE_TARGET_MODULES = [
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
]

ARCHITECTURE_MODULE_MAP = {
    "qwen3_5": {
        "attn": QWEN3_5_DEFAULT_GATED_DELTANET_TARGET_MODULES,
        "attn_full": QWEN3_5_GATED_DELTANET_TARGET_MODULES,
    },
}


def _assemble_target_modules(
    series: str,
    architecture: str | None,
    *,
    train_attn: bool,
    train_mlp: bool,
    train_unembed: bool,
    qwen_gated_deltanet_full_lora: bool = False,
) -> list[str]:
    target_modules: list[str] = []
    if train_attn:
        target_modules.extend(MODULE_MAP[series]["attn"])
        if architecture in ARCHITECTURE_MODULE_MAP:
            architecture_modules = ARCHITECTURE_MODULE_MAP[architecture]
            key = "attn_full" if qwen_gated_deltanet_full_lora else "attn"
            target_modules.extend(architecture_modules[key])
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
    qwen_gated_deltanet_full_lora: bool = False,
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
        qwen_gated_deltanet_full_lora=qwen_gated_deltanet_full_lora,
    )


def get_target_modules(
    model_path: str,
    lora_config: LoraConfig,
    *,
    qwen_gated_deltanet_full_lora: bool = False,
) -> list[str]:
    """Resolve a Tinker ``LoraConfig`` using the shared model-series map."""

    return resolve_target_modules(
        model_path,
        train_attn=lora_config.train_attn,
        train_mlp=lora_config.train_mlp,
        train_unembed=lora_config.train_unembed,
        qwen_gated_deltanet_full_lora=qwen_gated_deltanet_full_lora,
    )


def achievable_target_module_sets(
    model_path: str,
    *,
    qwen_gated_deltanet_full_lora: bool = False,
) -> list[list[str]] | None:
    """Every distinct module list client LoRA flags can produce.

    Returns None when the model series is unknown — then an explicit server
    list is the only way to run the model at all. Used at startup to reject
    configs whose fsdp_target_modules no client request could ever match.
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
                    qwen_gated_deltanet_full_lora=qwen_gated_deltanet_full_lora,
                )
                if modules and modules not in achievable:
                    achievable.append(modules)
    return achievable


def find_unmatched_target_modules(
    module_names: Iterable[str], target_modules: Iterable[str]
) -> list[str]:
    """Target names that match no module, using PEFT's suffix rule.

    PEFT wraps a module when its name equals the target or ends with
    ``"." + target``. A target that matches nothing is silently skipped
    there, so the run would train fewer modules than it records. Callers
    reject the request instead when this returns a non-empty list.
    """

    names = list(module_names)
    unmatched: list[str] = []
    for target in dict.fromkeys(target_modules):
        suffix = "." + target
        if not any(name == target or name.endswith(suffix) for name in names):
            unmatched.append(target)
    return unmatched


def gated_deltanet_mismatch_hint(
    actual_modules: Iterable[str], expected_modules: Iterable[str]
) -> str | None:
    """Explain module-list mismatches caused by the Qwen3.5 change.

    Returns a notice when the two sets differ only by Qwen3.5 Gated DeltaNet
    modules, either because one side predates issue #149 or because the two
    sides disagree on the optional A/B gate coverage. Returns None for every
    other mismatch.
    """

    difference = set(actual_modules) ^ set(expected_modules)
    if not difference or not difference <= set(QWEN3_5_GATED_DELTANET_TARGET_MODULES):
        return None
    if difference <= set(QWEN3_5_OPTIONAL_GATE_TARGET_MODULES):
        return (
            "The two module sets differ only by the optional linear_attn.in_proj_a/b "
            "Gated DeltaNet gate modules. Configure qwen_gated_deltanet_full_lora "
            "consistently on the source and destination; changing it requires a new "
            "training run."
        )
    return (
        "The two module sets differ only by the linear_attn.* modules of the "
        "Qwen3.5 Gated DeltaNet layers. This TuFT release added the default "
        "Q/K/V, Z, and output modules to the LoRA target list for Qwen3.5-based "
        "models (issue #149), so "
        "checkpoints and training runs from before the change no longer match. "
        "Create a new training run to continue."
    )
