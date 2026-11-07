"""Reproducible client timing independent of tenant count and task interleaving."""

import hashlib
import math
import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class IndependentClients:
    seed: int
    arrival_window_s: float = 240
    observation_s: float = 600
    think_max_s: float = 20
    tool_delay_max_s: float = 3

    def __post_init__(self):
        for key in ("arrival_window_s", "observation_s", "think_max_s", "tool_delay_max_s"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.observation_s <= 0:
            raise ValueError("observation_s must be positive")

    def draw(self, kind, tenant, *coordinates):
        material = repr((self.seed, kind, tenant, coordinates)).encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))
        return rng.random()

    def start_delay(self, tenant):
        return self.arrival_window_s * self.draw("arrival", tenant)

    def think_delay(self, tenant, step):
        return self.think_max_s * self.draw("think", tenant, step) if step else 0

    def tool_delay(self, tenant, step, trajectory, turn):
        return self.tool_delay_max_s * self.draw("tool", tenant, step, trajectory, turn)

    @property
    def end_s(self):
        return self.arrival_window_s + self.observation_s

    def can_start_round(self, elapsed_s):
        return elapsed_s < self.end_s

    def metadata(self, tenants):
        return dict(
            **asdict(self),
            initial_delays_s=[self.start_delay(i) for i in range(tenants)],
            measurement_window_s=[self.arrival_window_s, self.end_s],
        )
