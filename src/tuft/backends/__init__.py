from .sampling_backend import BaseSamplingBackend, VLLMSamplingBackend
from .training_backend import BaseTrainingBackend, HFTrainingBackend


__all__ = [
    "BaseSamplingBackend",
    "VLLMSamplingBackend",
    "BaseTrainingBackend",
    "HFTrainingBackend",
]


def __getattr__(name: str):  # noqa: N807
    if name == "FSDPTrainingBackend":
        from .fsdp_training_backend import FSDPTrainingBackend

        return FSDPTrainingBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
