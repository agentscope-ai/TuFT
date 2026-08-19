"""Shared LoRA target-module resolution for training backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from tinker.types import LoraConfig

from tuft.backends.vllm_lora_compat import resolve_model_series_and_architecture


@dataclass(frozen=True)
class LoraTargets:
    """The complete LoRA geometry a request resolves to.

    ``modules`` feed peft ``target_modules`` (suffix-matched ``nn.Module``
    names); ``parameters`` feed peft ``target_parameters`` (suffix-matched
    ``nn.Parameter`` names), used for fused tensors that have no per-unit
    module to wrap, such as MoE routed experts. Both lists together define a
    checkpoint's geometry: two adapters are compatible only when both match.
    """

    modules: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)


MODULE_MAP = {
    "llama": {
        "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        "unembed": ["lm_head"],
    },
    "qwen": {
        "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
        # Qwen embedding-adapter serving is tracked in issue #153, so this
        # flag currently adds no target modules for the Qwen family.
        "unembed": [],
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
# the same module list; the mlp names match the shared expert. Its routed
# experts are fused 3D parameters with no per-expert Linear modules, so
# ``train_mlp`` additionally targets them through peft ``target_parameters``
# (see ``QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS``), matching Tinker's documented
# behavior of covering MoE layers. vLLM parses the peft-format expert adapter
# keys when serving, so trained and served targets agree.
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

# The fused routed-expert weights of Qwen3.5-based MoE models. These are 3D
# ``nn.Parameter`` tensors (num_experts stacked per-expert matrices) on the
# ``mlp.experts`` module, so they can only be targeted through peft
# ``target_parameters`` (peft>=0.20 for the multi-adapter support TuFT
# needs). peft trains per-expert LoRA slices stored as one fused pair per
# parameter, and saves them under the ``...mlp.experts.lora_*`` /
# ``...mlp.experts.base_layer.lora_*`` keys that vLLM's fused-MoE LoRA
# loader parses.
QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS = [
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
]

ARCHITECTURE_MODULE_MAP = {
    "qwen3_5": {
        "attn": QWEN3_5_DEFAULT_GATED_DELTANET_TARGET_MODULES,
        "attn_full": QWEN3_5_GATED_DELTANET_TARGET_MODULES,
    },
    # The MoE variant shares the dense Gated DeltaNet attention geometry and
    # additionally maps train_mlp to the fused routed-expert parameters.
    "qwen3_5_moe": {
        "attn": QWEN3_5_DEFAULT_GATED_DELTANET_TARGET_MODULES,
        "attn_full": QWEN3_5_GATED_DELTANET_TARGET_MODULES,
        "mlp_parameters": QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS,
    },
}


def _assemble_targets(
    series: str,
    architecture: str | None,
    *,
    train_attn: bool,
    train_mlp: bool,
    train_unembed: bool,
    qwen_gated_deltanet_full_lora: bool = False,
) -> LoraTargets:
    target_modules: list[str] = []
    target_parameters: list[str] = []
    architecture_map = ARCHITECTURE_MODULE_MAP.get(architecture or "")
    if train_attn:
        target_modules.extend(MODULE_MAP[series]["attn"])
        if architecture_map is not None:
            key = "attn_full" if qwen_gated_deltanet_full_lora else "attn"
            target_modules.extend(architecture_map[key])
    if train_mlp:
        target_modules.extend(MODULE_MAP[series]["mlp"])
        if architecture_map is not None:
            target_parameters.extend(architecture_map.get("mlp_parameters", []))
    if train_unembed:
        target_modules.extend(MODULE_MAP[series]["unembed"])
    return LoraTargets(modules=target_modules, parameters=target_parameters)


def resolve_lora_targets(
    model_path: str,
    *,
    train_attn: bool,
    train_mlp: bool,
    train_unembed: bool,
    qwen_gated_deltanet_full_lora: bool = False,
) -> LoraTargets:
    """Resolve public LoRA modifier flags into concrete module/parameter names."""

    series, architecture = resolve_model_series_and_architecture(model_path)
    if series is None:
        raise ValueError(f"Unsupported model series: {model_path}")

    return _assemble_targets(
        series,
        architecture,
        train_attn=train_attn,
        train_mlp=train_mlp,
        train_unembed=train_unembed,
        qwen_gated_deltanet_full_lora=qwen_gated_deltanet_full_lora,
    )


def resolve_target_modules(
    model_path: str,
    *,
    train_attn: bool,
    train_mlp: bool,
    train_unembed: bool,
    qwen_gated_deltanet_full_lora: bool = False,
) -> list[str]:
    """Resolve public LoRA modifier flags into concrete module names."""

    return resolve_lora_targets(
        model_path,
        train_attn=train_attn,
        train_mlp=train_mlp,
        train_unembed=train_unembed,
        qwen_gated_deltanet_full_lora=qwen_gated_deltanet_full_lora,
    ).modules


def get_lora_targets(
    model_path: str,
    lora_config: LoraConfig,
    *,
    qwen_gated_deltanet_full_lora: bool = False,
) -> LoraTargets:
    """Resolve a Tinker ``LoraConfig`` using the shared model-series map."""

    return resolve_lora_targets(
        model_path,
        train_attn=lora_config.train_attn,
        train_mlp=lora_config.train_mlp,
        train_unembed=lora_config.train_unembed,
        qwen_gated_deltanet_full_lora=qwen_gated_deltanet_full_lora,
    )


def get_target_modules(
    model_path: str,
    lora_config: LoraConfig,
    *,
    qwen_gated_deltanet_full_lora: bool = False,
) -> list[str]:
    """Resolve a Tinker ``LoraConfig`` into module names only."""

    return get_lora_targets(
        model_path,
        lora_config,
        qwen_gated_deltanet_full_lora=qwen_gated_deltanet_full_lora,
    ).modules


def achievable_lora_target_sets(
    model_path: str,
    *,
    qwen_gated_deltanet_full_lora: bool = False,
) -> list[LoraTargets] | None:
    """Every distinct LoRA geometry client LoRA flags can produce.

    Returns None when the model series is unknown — then an explicit server
    list is the only way to run the model at all. Used at startup to reject
    configs whose fsdp_target_modules/fsdp_target_parameters no client
    request could ever match.
    """

    series, architecture = resolve_model_series_and_architecture(model_path)
    if series is None:
        return None
    achievable: list[LoraTargets] = []
    for train_attn in (True, False):
        for train_mlp in (True, False):
            for train_unembed in (True, False):
                targets = _assemble_targets(
                    series,
                    architecture,
                    train_attn=train_attn,
                    train_mlp=train_mlp,
                    train_unembed=train_unembed,
                    qwen_gated_deltanet_full_lora=qwen_gated_deltanet_full_lora,
                )
                if (targets.modules or targets.parameters) and targets not in achievable:
                    achievable.append(targets)
    return achievable


def achievable_target_module_sets(
    model_path: str,
    *,
    qwen_gated_deltanet_full_lora: bool = False,
) -> list[list[str]] | None:
    """Every distinct module list client LoRA flags can produce."""

    achievable = achievable_lora_target_sets(
        model_path,
        qwen_gated_deltanet_full_lora=qwen_gated_deltanet_full_lora,
    )
    if achievable is None:
        return None
    module_sets: list[list[str]] = []
    for targets in achievable:
        if targets.modules and targets.modules not in module_sets:
            module_sets.append(targets.modules)
    return module_sets


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


def find_unmatched_target_parameters(
    parameter_names: Iterable[str], target_parameters: Iterable[str]
) -> list[str]:
    """Parameter targets that match no parameter, using PEFT's suffix rule.

    PEFT targets a parameter when its dotted name equals the target or ends
    with ``"." + target``, and silently skips targets that match nothing.
    Names are compared after dropping peft's own ``base_layer`` wrapper
    segments, because a parameter already wrapped for an earlier adapter is
    renamed to ``...experts.base_layer.base_layer.gate_up_proj`` while peft
    matching still sees the unwrapped name.
    """

    names = [
        ".".join(part for part in name.split(".") if part != "base_layer")
        for name in parameter_names
    ]
    unmatched: list[str] = []
    for target in dict.fromkeys(target_parameters):
        suffix = "." + target
        if not any(name == target or name.endswith(suffix) for name in names):
            unmatched.append(target)
    return unmatched


def routed_expert_mismatch_hint(
    actual_parameters: Iterable[str], expected_parameters: Iterable[str]
) -> str | None:
    """Explain parameter-target mismatches caused by the MoE expert change.

    Returns a notice when the two sets differ only by the Qwen3.5 MoE fused
    routed-expert parameters, which usually means one side predates issue
    #154 (before it, ``train_mlp`` trained only the shared expert and no
    parameter targets existed). Returns None for every other mismatch.
    """

    difference = set(actual_parameters) ^ set(expected_parameters)
    if not difference or not difference <= set(QWEN3_5_MOE_EXPERT_TARGET_PARAMETERS):
        return None
    return (
        "The two LoRA geometries differ only by the mlp.experts.* fused "
        "routed-expert parameters of Qwen3.5-based MoE models. This TuFT "
        "release extended train_mlp to cover the routed experts through peft "
        "target_parameters (issue #154), so checkpoints and training runs "
        "from before the change no longer match. Create a new training run "
        "to continue."
    )


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
