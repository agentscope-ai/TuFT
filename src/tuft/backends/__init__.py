# pyright: reportUnsupportedDunderAll=false

__all__ = [
    "BaseSamplingBackend",
    "DPSamplingBackend",
    "VLLMSamplingBackend",
    "FixedSamplingBackend",
    "SamplingRuntimeRouter",
    "BaseTrainingBackend",
    "HFTrainingBackend",
    "FSDPTrainingBackend",
    "TorchTPTrainingBackend",
    "TorchTPSamplingBackend",
    "FlexBackend",
    "FlexBackendMode",
    "TransformDirection",
    "TransformResult",
    "FusedTorchTPVLLMFlexBackend",
]

_LAZY_IMPORTS = {
    "BaseSamplingBackend": (".base_backend", "BaseSamplingBackend"),
    "BaseTrainingBackend": (".base_backend", "BaseTrainingBackend"),
    "DPSamplingBackend": (".sampling_backend", "DPSamplingBackend"),
    "VLLMSamplingBackend": (".sampling_backend", "VLLMSamplingBackend"),
    "FixedSamplingBackend": (".sampling_backend", "FixedSamplingBackend"),
    "SamplingRuntimeRouter": (".sampling_router", "SamplingRuntimeRouter"),
    "HFTrainingBackend": (".training_backend", "HFTrainingBackend"),
    "FSDPTrainingBackend": (".fsdp_training_backend", "FSDPTrainingBackend"),
    "TorchTPTrainingBackend": (".torchtp_backend", "TorchTPTrainingBackend"),
    "TorchTPSamplingBackend": (".torchtp_backend", "TorchTPSamplingBackend"),
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
