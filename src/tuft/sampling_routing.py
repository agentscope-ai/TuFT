"""Replica selection, independent of borrowing, training state, and cost estimates."""

from collections.abc import Sequence


class RoundRobinRouter:
    def __init__(self, replicas: int):
        if replicas < 1:
            raise ValueError("At least one replica is required")
        self.replicas = replicas
        self.cursor = 0

    def choose(self, eligible: Sequence[int]) -> int | None:
        if not eligible:
            return None
        selected = min(eligible, key=lambda index: (index - self.cursor) % self.replicas)
        self.cursor = (selected + 1) % self.replicas
        return selected
