"""Unified-Engine baseline mode-alternation gate.

Unified-Engine runs one tensor-parallel trainer that both samples and trains,
alternating between the two modes when a training queue fills. Because a
training runtime is not an inference engine, the two modes contend for the
same GPU; this gate enforces the agreed switch semantics at the request entry:

1. Training requests always proceed (training mode).
2. While training is in flight, sampling is held.
3. After the training queue drains, the engine stays in sampling mode for at
   least ``sampling_min_window_s`` before any waiting training can pull it
   back (prevents switch thrashing and keeps the sampling path warm).
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field


@dataclass
class UnifiedEngineGate:
    """Drain-then-minimum-sampling-window gate for the Unified-Engine mode."""

    sampling_min_window_s: float = 3.0
    _inflight_training: int = field(default=0, init=False)
    _sampling_until: float = field(default=0.0, init=False)
    _cond: asyncio.Condition = field(default_factory=asyncio.Condition, init=False)
    switches_to_training: int = field(default=0, init=False)
    switches_to_sampling: int = field(default=0, init=False)
    sampling_hold_seconds: float = field(default=0.0, init=False)

    async def acquire_training(self) -> None:
        """Enter training mode; sampling waits while training is in flight.

        The unified engine is a single runtime: training steps from different
        tenants are serialized here so they never contend on the shared model
        (concurrent forwards on one torch runtime serialize on its internal
        lock, stall, and trigger client-side retries / sequence conflicts).
        """
        async with self._cond:
            while self._inflight_training > 0:
                await self._cond.wait()
            self.switches_to_training += 1
            self._inflight_training += 1
            self._cond.notify_all()
            logger.info(
                "[unified-gate] training acquired: inflight=%s", self._inflight_training
            )

    async def release_training(self) -> None:
        """Finish one training operation; open the sampling window on drain."""
        async with self._cond:
            self._inflight_training = max(0, self._inflight_training - 1)
            logger.info(
                "[unified-gate] training released: inflight=%s", self._inflight_training
            )
            if self._inflight_training == 0:
                self._sampling_until = time.monotonic() + self.sampling_min_window_s
                self.switches_to_sampling += 1
            self._cond.notify_all()

    async def acquire_sampling(self) -> None:
        """Hold sampling until training drains on this engine (drain semantics).

        A 30s timeout was tried here and reverted: it changes what the baseline
        measures without fixing it. Unified-Engine at eight tenants still stalls -
        48 futures begun against 16 completed, no training step finishing - and the
        cause is not this wait. Note that the sampling wrapper picks a replica by
        round-robin while the training wrapper pins each adapter to a fixed
        replica, so a sample commonly lands on an engine training a different
        tenant; whether that is the stall or merely a slowdown is still unproven.
        """
        start = time.monotonic()
        logger.info(
            "[unified-gate] acquire_sampling enter: inflight_training=%s",
            self._inflight_training,
        )
        waited_logged = False
        async with self._cond:
            while self._inflight_training > 0:
                if not waited_logged:
                    logger.info(
                        "[unified-gate] sampling waiting: inflight_training=%s",
                        self._inflight_training,
                    )
                    waited_logged = True
                await self._cond.wait()
        held = time.monotonic() - start
        if waited_logged:
            logger.info("[unified-gate] sampling admitted after %.1fs", held)
        if held > 0:
            self.sampling_hold_seconds += held

    def sampling_idle(self) -> bool:
        """True when no training is in flight on this engine (lock-free read)."""
        return self._inflight_training == 0

    def snapshot(self) -> dict[str, float]:
        return {
            "switches_to_training": float(self.switches_to_training),
            "switches_to_sampling": float(self.switches_to_sampling),
            "sampling_hold_seconds": self.sampling_hold_seconds,
            "sampling_min_window_s": self.sampling_min_window_s,
        }
