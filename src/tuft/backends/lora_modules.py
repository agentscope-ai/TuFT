"""Shared LoRA target-module resolution for training backends."""

from __future__ import annotations

from tinker.types import LoraConfig

from tuft.backends.vllm_lora_compat import resolve_model_architecture, resolve_model_series


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


def resolve_target_modules(
    model_path: str,
    *,
    train_attn: bool,
    train_mlp: bool,
    train_unembed: bool,
) -> list[str]:
    """Resolve public LoRA modifier flags into concrete module names."""

    series = resolve_model_series(model_path)
    if series is None:
        raise ValueError(f"Unsupported model series: {model_path}")

    target_modules: list[str] = []
    if train_attn:
        target_modules.extend(MODULE_MAP[series]["attn"])
        architecture = resolve_model_architecture(model_path)
        if architecture in ARCHITECTURE_MODULE_MAP:
            target_modules.extend(ARCHITECTURE_MODULE_MAP[architecture]["attn"])
    if train_mlp:
        target_modules.extend(MODULE_MAP[series]["mlp"])
    if train_unembed:
        target_modules.extend(MODULE_MAP[series]["unembed"])
    return target_modules


def get_target_modules(model_path: str, lora_config: LoraConfig) -> list[str]:
    """Resolve a Tinker ``LoraConfig`` using the shared model-series map."""

    return resolve_target_modules(
        model_path,
        train_attn=lora_config.train_attn,
        train_mlp=lora_config.train_mlp,
        train_unembed=lora_config.train_unembed,
    )
