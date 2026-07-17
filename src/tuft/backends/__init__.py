from .fsdp_training_backend import FSDPTrainingBackend
from .sampling_backend import BaseSamplingBackend, DPSamplingBackend, VLLMSamplingBackend
from .training_backend import BaseTrainingBackend, HFTrainingBackend


__all__ = [
    "BaseSamplingBackend",
    "DPSamplingBackend",
    "VLLMSamplingBackend",
    "BaseTrainingBackend",
    "HFTrainingBackend",
    "FSDPTrainingBackend",
    # Lazy-loaded (heavy transitive dependencies):
    "FlexBackend",  # pyright: ignore[reportUnsupportedDunderAll]
    "FlexBackendMode",  # pyright: ignore[reportUnsupportedDunderAll]
    "TransformDirection",  # pyright: ignore[reportUnsupportedDunderAll]
    "TransformResult",  # pyright: ignore[reportUnsupportedDunderAll]
    "FusedTorchTPVLLMFlexBackend",  # pyright: ignore[reportUnsupportedDunderAll]
]

_LAZY_IMPORTS = {
    "FlexBackend": (".flex", "FlexBackend"),
    "FlexBackendMode": (".flex", "FlexBackendMode"),
    "TransformDirection": (".flex", "TransformDirection"),
    "TransformResult": (".flex", "TransformResult"),
    "FusedTorchTPVLLMFlexBackend": (".flex.torchtp", "FusedTorchTPVLLMFlexBackend"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_path, attr = _LAZY_IMPORTS[name]
    value = getattr(importlib.import_module(module_path, __name__), attr)
    globals()[name] = value
    return value
