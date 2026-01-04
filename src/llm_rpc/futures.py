"""Simple in-memory future registry for the synthetic Tinker API."""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from fastapi import HTTPException

from tinker import types
from tinker.types.try_again_response import TryAgainResponse

QueueState = Literal["active", "paused_capacity", "paused_rate_limit"]


@dataclass
class FutureRecord:
    payload: Any | None = None
    model_id: str | None = None
    queue_state: QueueState = "active"
    status: Literal["pending", "ready", "failed"] = "pending"
    error: types.RequestFailedResponse | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class FutureStore:
    """Runs controller work on a thread pool and tracks each request's lifecycle.

    Used for retrieve_future polling.
    """

    def __init__(self, *, max_workers: int = 4) -> None:
        self._records: dict[str, FutureRecord] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="llm-rpc-future"
        )

    def _store_record(self, record: FutureRecord) -> None:
        with self._lock:
            self._records[record.request_id] = record

    def enqueue(
        self,
        operation: Callable[[], Any],
        *,
        model_id: str | None = None,
        queue_state: QueueState = "active",
    ) -> types.UntypedAPIFuture:
        record = FutureRecord(model_id=model_id, queue_state=queue_state)
        self._store_record(record)

        def _runner() -> None:
            try:
                payload = operation()
            except HTTPException as exc:
                message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
                failure = types.RequestFailedResponse(
                    error=message,
                    category=types.RequestErrorCategory.User,
                )
                self._mark_failed(record.request_id, failure)
            except Exception as exc:  # pylint: disable=broad-except
                failure = types.RequestFailedResponse(
                    error=str(exc),
                    category=types.RequestErrorCategory.Server,
                )
                self._mark_failed(record.request_id, failure)
            else:
                self._mark_ready(record.request_id, payload)

        self._executor.submit(_runner)
        return types.UntypedAPIFuture(request_id=record.request_id, model_id=model_id)

    def create_ready_future(
        self,
        payload: Any,
        *,
        model_id: str | None = None,
    ) -> types.UntypedAPIFuture:
        record = FutureRecord(payload=payload, model_id=model_id, status="ready")
        self._store_record(record)
        return types.UntypedAPIFuture(request_id=record.request_id, model_id=model_id)

    def _mark_ready(self, request_id: str, payload: Any) -> None:
        with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return
            record.payload = payload
            record.status = "ready"
            record.error = None

    def _mark_failed(self, request_id: str, failure: types.RequestFailedResponse) -> None:
        with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return
            record.status = "failed"
            record.error = failure

    def retrieve(self, request_id: str) -> Any:
        with self._lock:
            record = self._records.get(request_id)
        if record is None:
            raise KeyError(request_id)
        if record.status == "pending":
            return TryAgainResponse(request_id=request_id, queue_state=record.queue_state)
        if record.status == "failed" and record.error is not None:
            return record.error
        return record.payload

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
