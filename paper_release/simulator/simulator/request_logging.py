"""Request-arrival logging helpers for simulator experiments."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, Optional


RequestLogWriter = Callable[[Dict[str, Any]], None]


def make_request_arrival_record(
    *,
    tenant_id: str,
    request_kind: str,
    operation_type: str,
    task: str,
    request_rate: float,
    buffer_size: int,
    train_step: int,
    current_weight_version: int,
    staleness_limit: Optional[int],
    sync_mode: bool,
    async_sampling: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a compact JSON-serializable request-arrival event."""
    expected_period_s = 1.0 / request_rate if request_rate > 0 else None
    record: Dict[str, Any] = {
        "schema_version": 1,
        "event": "request_arrival",
        "timestamp_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "tenant_id": tenant_id,
        "request_kind": request_kind,
        "operation_type": operation_type,
        "task": task,
        "request_rate": request_rate,
        "expected_period_s": expected_period_s,
        "buffer_size": buffer_size,
        "train_step": train_step,
        "current_weight_version": current_weight_version,
        "staleness_limit": staleness_limit,
        "sync_mode": sync_mode,
        "async_sampling": async_sampling,
    }
    if extra:
        record.update(extra)
    return record


class JsonlRequestLog:
    """Single-threaded JSONL writer used by the asyncio simulator."""

    def __init__(self, output_path: str, flush_every: int = 1):
        self.output_path = output_path
        self.flush_every = max(1, flush_every)
        self._writes_since_flush = 0
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        self._fp = open(output_path, "w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._writes_since_flush += 1
        if self._writes_since_flush >= self.flush_every:
            self._fp.flush()
            self._writes_since_flush = 0

    def close(self) -> None:
        self._fp.flush()
        self._fp.close()
