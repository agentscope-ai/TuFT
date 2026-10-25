"""vLLM 0.24 loader extension; imported only inside GPU workers.

Bind inside the model-loading allocation context. Unused initial allocations
can then be reclaimed by vLLM before KV allocation and graph capture, without
editing allocator pointer tables or retagging live allocations.
"""

import pickle

import torch
import vllm
from torch.multiprocessing.reductions import rebuild_cuda_tensor
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

from .shared_weight_layout import text_config
from .shared_weights import (
    bind_shared_weights,
    check_shared_weight_versions,
    resolve_shared_weight_aliases,
)


@register_model_loader("tuft_shared")
class SharedWeightLoader(BaseModelLoader):
    def download_model(self, model_config) -> None:
        pass

    def load_weights(self, model, model_config) -> None:
        check_shared_weight_versions(torch.__version__, vllm.__version__)
        text_config(model_config.hf_config)
        if model_config.quantization:
            raise ValueError("Shared loader requires unquantized Qwen weights")
        # This private startup file is created by our Ray actor, never a client.
        with open(self.load_config.model_loader_extra_config["manifest"], "rb") as handle:
            ranks = pickle.load(handle)
        if len(ranks) != get_tensor_model_parallel_world_size():
            raise ValueError("Shared manifest and vLLM TP group differ")
        descriptors = ranks[get_tensor_model_parallel_rank()]
        # Each producer sees only cuda:0. The TP consumer sees the whole group;
        # translate the storage device ordinal after verifying physical placement.
        weights = {}
        for name, descriptor in descriptors.items():
            args = list(descriptor)
            args[6] = torch.cuda.current_device()
            name = f"language_model.{name}" if hasattr(model, "language_model") else name
            weights[name] = rebuild_cuda_tensor(*args)
        weights = resolve_shared_weight_aliases(model, weights)
        model._tuft_shared_weight_report = bind_shared_weights(model, weights)
        # Keep imported tensors alive across LoRA wrapping, sleep and graph replay.
        model._tuft_shared_weights = weights
