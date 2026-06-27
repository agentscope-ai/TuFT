"""
Create mock LoRA adapters for Qwen3-8B for benchmarking purposes.
These adapters have random weights and are used purely for performance testing.
"""

import json
import os

import torch
from safetensors.torch import save_file


# Qwen3-8B architecture parameters
HIDDEN_SIZE = 4096
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE_SIZE = 12288
NUM_LAYERS = 36

# LoRA parameters
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Module dimensions: (input_dim, output_dim) for each target module
MODULE_DIMS = {
    "q_proj": (HIDDEN_SIZE, NUM_HEADS * HEAD_DIM),  # (4096, 4096)
    "k_proj": (HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM),  # (4096, 1024)
    "v_proj": (HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM),  # (4096, 1024)
    "o_proj": (NUM_HEADS * HEAD_DIM, HIDDEN_SIZE),  # (4096, 4096)
}


def create_adapter_config(save_dir: str, base_model_path: str):
    """Create adapter_config.json in PEFT format."""
    config = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": LORA_RANK,
        "rank_pattern": {},
        "revision": None,
        "target_modules": TARGET_MODULES,
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("  -> Saved adapter_config.json")


def create_adapter_weights(save_dir: str, seed: int):
    """Create random LoRA adapter weights in safetensors format."""
    torch.manual_seed(seed)
    state_dict = {}

    for layer_idx in range(NUM_LAYERS):
        for module_name in TARGET_MODULES:
            in_dim, out_dim = MODULE_DIMS[module_name]
            prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module_name}"

            # lora_A: (rank, in_dim) and lora_B: (out_dim, rank)
            state_dict[f"{prefix}.lora_A.weight"] = (
                torch.randn(LORA_RANK, in_dim, dtype=torch.float16) * 0.01
            )
            state_dict[f"{prefix}.lora_B.weight"] = torch.zeros(
                out_dim, LORA_RANK, dtype=torch.float16
            )

    save_file(state_dict, os.path.join(save_dir, "adapter_model.safetensors"))
    total_params = sum(t.numel() for t in state_dict.values())
    print(
        f"  -> Saved adapter_model.safetensors ({total_params:,} params, "
        f"{sum(t.nbytes for t in state_dict.values()) / 1024 / 1024:.1f} MB)"
    )


def main():
    base_model_path = "/mnt/cpfs/luyi/models/Qwen3-8B"
    output_base = "/mnt/nas/hanzhang.yhz/multilora_bench/mock_loras"
    num_adapters = 5

    print(f"Creating {num_adapters} mock LoRA adapters for Qwen3-8B")
    print(f"  LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print(f"  Target modules: {TARGET_MODULES}")
    print(f"  Layers: {NUM_LAYERS}")
    print()

    for i in range(num_adapters):
        adapter_name = f"adapter_{i}"
        save_dir = os.path.join(output_base, adapter_name)
        print(f"Creating {adapter_name} (seed={42 + i}):")
        create_adapter_config(save_dir, base_model_path)
        create_adapter_weights(save_dir, seed=42 + i)
        print()

    print(f"All {num_adapters} adapters created in {output_base}")


if __name__ == "__main__":
    main()
