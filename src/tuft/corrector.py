"""Log-probability corrector interface and mock implementation.

The Corrector sits on the sampling response path and restores every sampling
path (flex and fixed) to the trainer's reference so the two paths are
interchangeable from a tenant's point of view. The real Corrector is a small
network calibrated offline from synchronous rollout probes; until it is
trained, ``MockCorrector`` provides the same interface and only simulates the
correction overhead (a configurable per-sample latency and an optional
constant bias). With ``bias=0.0`` the returned log-probabilities are bit-for-bit
identical to the uncorrected ones.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

from tinker import types


class Corrector(Protocol):
    """Transforms a raw sampling response before it is returned to the tenant."""

    async def correct(self, response: types.SampleResponse) -> types.SampleResponse:
        ...

    def snapshot(self) -> dict[str, float]:
        ...


@dataclass
class MockCorrector:
    """Corrector stand-in that simulates correction overhead without fitting.

    Attributes:
        latency_s: simulated wall-clock cost applied once per sample response.
        bias: constant added to every returned response log-probability. The
            default 0.0 leaves all values unchanged.
    """

    latency_s: float = 0.0
    bias: float = 0.0
    corrected_sequences: int = field(default=0, init=False)
    corrected_tokens: int = field(default=0, init=False)
    total_overhead_s: float = field(default=0.0, init=False)

    async def correct(self, response: types.SampleResponse) -> types.SampleResponse:
        start = time.monotonic()
        if self.latency_s > 0:
            await asyncio.sleep(self.latency_s)

        sequences: list[types.SampledSequence] = []
        changed = self.bias != 0.0
        for seq in response.sequences:
            self.corrected_sequences += 1
            logprobs = seq.logprobs
            if changed and logprobs:
                corrected = [lp + self.bias for lp in logprobs]
                self.corrected_tokens += len(corrected)
                sequences.append(
                    types.SampledSequence(
                        stop_reason=seq.stop_reason,
                        _tokens_list=list(seq.tokens),
                        _logprobs_list=corrected,
                    )
                )
            else:
                if logprobs:
                    self.corrected_tokens += len(logprobs)
                sequences.append(seq)

        self.total_overhead_s += time.monotonic() - start
        if not changed:
            return response
        return types.SampleResponse(
            sequences=sequences,
            _prompt_logprobs_list=response._prompt_logprobs_list,
            _topk_prompt_logprobs_list=response._topk_prompt_logprobs_list,
        )

    def snapshot(self) -> dict[str, float]:
        return {
            "corrected_sequences": float(self.corrected_sequences),
            "corrected_tokens": float(self.corrected_tokens),
            "total_overhead_s": self.total_overhead_s,
            "latency_s": self.latency_s,
            "bias": self.bias,
        }


class NoopCorrector:
    """Pass-through corrector used when correction is disabled."""

    async def correct(self, response: types.SampleResponse) -> types.SampleResponse:
        return response

    def snapshot(self) -> dict[str, float]:
        return {"enabled": 0.0}


def build_corrector(
    *,
    enabled: bool,
    latency_ms: float = 0.0,
    bias: float = 0.0,
) -> Optional[Corrector]:
    """Create the corrector configured for this deployment, or None."""
    if not enabled:
        return None
    return MockCorrector(latency_s=max(0.0, latency_ms) / 1000.0, bias=bias)
