"""GPU placeholder actor used to reserve one GPU for non-Ray runtimes.

The flex backend runs in the server process (not a Ray actor), so Ray cannot
see its GPU usage and may schedule Ray-managed vLLM actors onto the same
device.  Creating a placeholder actor first lets Ray reserve one GPU; the
server then pins the flex runtime to exactly that GPU, and all Ray-managed
sampling replicas land on the remaining devices.
"""

from __future__ import annotations

import ray


@ray.remote
class GpuPlaceholder:
    """Hold one GPU and report which physical index Ray assigned."""

    def gpu_index(self) -> int:
        import os

        return int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])

    def ping(self) -> bool:
        return True
