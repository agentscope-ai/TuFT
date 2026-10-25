"""Conservative verification of immutable, mathematically zero LoRA snapshots."""

import json
import math
from pathlib import Path


def is_zero_lora_snapshot(path: Path) -> bool:
    """Verify every factor; metadata or an optimizer step count is insufficient.

    Unknown formats, extra trainable state, nonfinite factors, and incomplete
    A/B pairs all disable this optimization. Never deserialize pickle weights.
    """
    try:
        import torch
        from safetensors import safe_open

        config = json.loads((path / "adapter_config.json").read_text())
        if (
            config.get("peft_type") != "LORA"
            or config.get("bias", "none") != "none"
            or config.get("lora_bias", False)
            or config.get("use_dora", False)
            or config.get("modules_to_save")
            or config.get("target_parameters")
            or config.get("alpha_pattern")
            or config.get("rank_pattern")
            or config.get("layer_replication")
            or float(config.get("r", 0)) <= 0
            or not math.isfinite(float(config.get("r", 0)))
            or not math.isfinite(float(config.get("lora_alpha", float("nan"))))
        ):
            return False
        a_names, b_names = {}, {}
        with safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu") as weights:
            for name in weights.keys():
                if name.endswith(".lora_A.weight"):
                    a_names[name.removesuffix(".lora_A.weight")] = weights.get_slice(
                        name
                    ).get_shape()
                elif name.endswith(".lora_B.weight"):
                    b_names[name.removesuffix(".lora_B.weight")] = weights.get_slice(
                        name
                    ).get_shape()
                else:
                    return False
                tensor = weights.get_tensor(name)
                if tensor.ndim != 2 or tensor.numel() == 0 or not torch.isfinite(tensor).all():
                    return False
                if name.endswith(".lora_B.weight") and torch.count_nonzero(tensor):
                    return False
        return (
            bool(a_names)
            and a_names.keys() == b_names.keys()
            and all(a_names[k][0] == b_names[k][1] for k in a_names)
        )
    except Exception:
        # This is an optional fast path: corrupt/unknown snapshots still go
        # through the normal engine validation, never silently become base-only.
        return False
