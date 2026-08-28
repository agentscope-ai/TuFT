from __future__ import annotations

import os
import pickle
import sys
import traceback
from typing import Any

from vllm.v1.worker.gpu_worker import Worker as GPUWorker
from vllm.v1.worker.worker_base import CompilationTimes

from tuft.backends.flex.torchtp_zero_copy import inject_cuda_ipc_alias


class TuftFlexGPUWorker(GPUWorker):
    """vLLM GPU worker that aliases FlexBackend IPC weights before graph capture.

    The runner/backend writes all-rank CUDA IPC descriptors to the path specified
    by TUFT_FLEX_PRECAPTURE_IPC_PATH. vLLM calls compile_or_warm_up_model() after
    dummy model load and KV cache allocation, immediately before CUDA Graph
    capture. Injecting here ensures CUDA Graph captures the real aliased base
    storage rather than the dummy weight storage.
    """

    def _log_flex_hook(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)
        log_path = os.getenv("TUFT_FLEX_PRECAPTURE_LOG_PATH")
        if log_path:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    def _maybe_inject_flex_alias_before_capture(self) -> None:
        if bool(getattr(self, "_tuft_flex_precapture_alias_done", False)):
            return
        descriptor_path = os.getenv("TUFT_FLEX_PRECAPTURE_IPC_PATH")
        if not descriptor_path:
            return
        verify = os.getenv("TUFT_FLEX_PRECAPTURE_VERIFY", "0") in {"1", "true", "True"}
        with open(descriptor_path, "rb") as handle:
            descriptors: list[dict[str, dict[str, Any]]] = pickle.load(handle)
        result = inject_cuda_ipc_alias(self, descriptors, verify=verify)
        self._log_flex_hook(
            f"[TuftFlexGPUWorker] alias result rank={result.get('rank')} "
            f"injected={result.get('injected')} mismatched={result.get('mismatched')} "
            f"skipped={result.get('skipped')}"
        )
        if result.get("skipped"):
            self._log_flex_hook(
                f"[TuftFlexGPUWorker] skip examples: {result.get('examples')}"
            )
        self._tuft_flex_precapture_alias_result = result
        self._tuft_flex_precapture_alias_done = True

    def compile_or_warm_up_model(self) -> CompilationTimes:
        try:
            self._log_flex_hook("[TuftFlexGPUWorker] pre-capture alias hook start")
            self._maybe_inject_flex_alias_before_capture()
            self._log_flex_hook("[TuftFlexGPUWorker] pre-capture alias hook done")
            return super().compile_or_warm_up_model()
        except Exception:
            log_path = os.getenv("TUFT_FLEX_PRECAPTURE_LOG_PATH")
            if log_path:
                with open(log_path, "a", encoding="utf-8") as handle:
                    traceback.print_exc(file=handle)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise
