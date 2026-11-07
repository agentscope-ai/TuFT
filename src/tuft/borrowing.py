"""Reactive borrowing admission; replica routing is a separate decision."""

from .config import BorrowingConfig


class BorrowingAdmission:
    def __init__(self, config: BorrowingConfig):
        self.config = config

    def unavailable_reason(
        self,
        *,
        failed: bool,
        eligible: bool,
        training: bool,
        quiet: bool,
        active: int,
        dedicated_full: bool,
    ) -> str | None:
        if not self.config.enabled or failed:
            return "unavailable"
        if not dedicated_full:
            return "dedicated_available"
        if not eligible:
            return "ineligible"
        if training:
            return "training"
        if quiet:
            return "quiet"
        if active >= self.config.max_inflight:
            return "capacity"
        return None
