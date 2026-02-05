"""Simple in-memory future registry for the synthetic Tinker API."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from tinker import types
from tinker.types.try_again_response import TryAgainResponse

from .exceptions import (
    FutureCancelledException,
    FutureNotFoundException,
    ServerException,
    TuFTException,
    UserMismatchException,
)
from .persistence import (
    get_redis_store,
    is_persistence_enabled,
    load_record,
    save_record,
)
from .telemetry.metrics import get_metrics
from .telemetry.tracing import get_tracer


logger = logging.getLogger(__name__)


_get_tracer = lambda: get_tracer("tuft.futures")  # noqa: E731

QueueState = Literal["active", "paused_capacity", "paused_rate_limit"]

OperationType = Literal[
    "forward",
    "forward_backward",
    "optim_step",
    "save_weights",
    "save_weights_for_sampler",
    "load_weights",
    "sample",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FutureRecord(BaseModel):
    """Future record with persistence support.

    Fields:
        event: Not serialized (excluded) - created fresh on each instance.
               After restore, if status is ready/failed, event is auto-set.
        operation_type: Type of operation for recovery purposes.
        operation_args: Serializable arguments for the operation.
        future_id: Globally incrementing sequence number for ordering futures.
                   Used instead of timestamps to avoid timezone/clock issues.
        created_at: Timestamp when the future was created (for logging only).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    future_id: int = 0
    model_id: str | None = None
    user_id: str | None = None
    queue_state: QueueState = "active"
    status: Literal["pending", "ready", "failed"] = "pending"
    payload: Any | None = None
    error: TuFTException | None = None
    operation_type: OperationType | None = None
    operation_args: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=_now)
    # Runtime-only field, excluded from serialization
    event: asyncio.Event = Field(default_factory=asyncio.Event, exclude=True)

    @model_validator(mode="after")
    def _set_event_if_completed(self) -> "FutureRecord":
        """Set the event if the future is already completed."""
        if self.status in ("ready", "failed"):
            self.event.set()
        return self


class FutureStore:
    """Runs controller work asynchronously and tracks each request's lifecycle."""

    REDIS_KEY_PREFIX = "future"

    def __init__(self) -> None:
        self._records: dict[str, FutureRecord] = {}
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._next_future_id: int = 1
        self._restore_from_redis()

    def _build_key(self, request_id: str) -> str:
        return get_redis_store().build_key(self.REDIS_KEY_PREFIX, request_id)

    def _restore_from_redis(self) -> None:
        if not is_persistence_enabled():
            return
        store = get_redis_store()
        pattern = store.build_key(self.REDIS_KEY_PREFIX, "*")
        for key in store.keys(pattern):
            record = load_record(key, FutureRecord)
            if record is None:
                # Record may have expired (TTL) or failed to deserialize
                # This is expected for expired futures, just skip them
                continue
            if record.status != "pending":
                record.event.set()
            self._records[record.request_id] = record
            if record.future_id >= self._next_future_id:
                self._next_future_id = record.future_id + 1

    def _save_future(self, request_id: str) -> None:
        if not is_persistence_enabled():
            return
        record = self._records.get(request_id)
        if record is not None:
            # Use TTL for futures to prevent Redis from growing indefinitely
            # Futures are short-lived and can be safely expired
            ttl = get_redis_store().future_ttl
            save_record(self._build_key(request_id), record, ttl_seconds=ttl)

    def _allocate_future_id(self) -> int:
        """Allocate and return a new globally unique future_id."""
        future_id = self._next_future_id
        self._next_future_id += 1
        return future_id

    def get_current_future_id(self) -> int:
        """Get the current (latest allocated) future_id, or 0 if none allocated."""
        return self._next_future_id - 1 if self._next_future_id > 1 else 0

    def _delete_future(self, request_id: str) -> None:
        if not is_persistence_enabled():
            return
        get_redis_store().delete(self._build_key(request_id))

    def get_pending_futures_by_model(self) -> dict[str | None, list[FutureRecord]]:
        """Group all pending futures by model_id."""
        by_model: dict[str | None, list[FutureRecord]] = {}
        for record in self._records.values():
            if record.status == "pending":
                if record.model_id not in by_model:
                    by_model[record.model_id] = []
                by_model[record.model_id].append(record)

        for model_id in by_model:
            by_model[model_id].sort(key=lambda r: r.future_id)

        return by_model

    def mark_futures_failed_after_checkpoint(
        self,
        model_id: str | None,
        checkpoint_future_id: int | None,
        error_message: str = "Server restored from checkpoint. Please retry.",
    ) -> int:
        """Mark all futures for a model after a checkpoint as failed."""
        count = 0
        for record in self._records.values():
            if record.model_id != model_id:
                continue
            if checkpoint_future_id is None or record.future_id > checkpoint_future_id:
                record.status = "failed"
                record.error = FutureCancelledException(record.request_id, reason=error_message)
                record.event.set()
                self._save_future(record.request_id)
                count += 1
        return count

    def mark_all_pending_failed(
        self,
        error_message: str = "Server restarted while task was pending. Please retry.",
    ) -> int:
        """Mark all pending futures as failed."""
        count = 0
        for record in self._records.values():
            if record.status == "pending":
                record.status = "failed"
                record.error = FutureCancelledException(record.request_id, reason=error_message)
                record.event.set()
                self._save_future(record.request_id)
                count += 1
        return count

    def mark_pending_sample_futures_failed(
        self,
        error_message: str = "Server restarted while sample request was pending. Please retry.",
    ) -> int:
        """Mark all pending sample futures as failed."""
        count = 0
        for record in self._records.values():
            if record.status == "pending" and record.operation_type == "sample":
                record.status = "failed"
                record.error = FutureCancelledException(record.request_id, reason=error_message)
                record.event.set()
                self._save_future(record.request_id)
                count += 1
        return count

    def _store_record(self, record: FutureRecord) -> None:
        self._records[record.request_id] = record
        self._save_future(record.request_id)

    async def enqueue(
        self,
        operation: Callable[[], Any],
        user_id: str,
        *,
        model_id: str | None = None,
        queue_state: QueueState = "active",
        operation_type: OperationType | None = None,
        operation_args: dict[str, Any] | None = None,
    ) -> types.UntypedAPIFuture:
        """Enqueue a task (sync or async) and return a future immediately.

        Args:
            operation: The callable to execute.
            user_id: The user ID making the request.
            model_id: Optional model ID associated with this operation.
            queue_state: State of the queue.
            operation_type: Type of operation for recovery purposes.
            operation_args: Serializable arguments for recovery.
        """
        async with self._lock:
            future_id = self._allocate_future_id()
            record = FutureRecord(
                future_id=future_id,
                model_id=model_id,
                user_id=user_id,
                queue_state=queue_state,
                operation_type=operation_type,
                operation_args=operation_args,
            )
            self._store_record(record)

        # Update metrics
        metrics = get_metrics()
        metrics.futures_created.add(
            1,
            {"operation_type": operation_type or "unknown", "model_id": model_id or ""},
        )
        metrics.futures_queue_length.add(1, {"queue_state": queue_state})

        logger.info("Future enqueued: %s", record.request_id)
        enqueue_time = time.perf_counter()

        async def _runner() -> None:
            start_time = time.perf_counter()
            wait_time = start_time - enqueue_time

            with _get_tracer().start_as_current_span("future_store.execute_operation") as span:
                span.set_attribute("tuft.request_id", record.request_id)
                span.set_attribute("tuft.operation_type", operation_type or "unknown")
                if model_id:
                    span.set_attribute("tuft.model_id", model_id)

                logger.info("Future begin: %s", record.request_id)
                try:
                    if asyncio.iscoroutinefunction(operation):
                        payload = await operation()
                    else:
                        # Run sync operation in thread pool to avoid blocking
                        loop = asyncio.get_running_loop()
                        payload = await loop.run_in_executor(None, operation)
                except TuFTException as exc:
                    failure = exc
                    span.record_exception(exc)
                    logger.error("Future failed: %s", record.request_id)
                    await self._mark_failed(record.request_id, failure, operation_type)
                except Exception as exc:  # pylint: disable=broad-except
                    # Unknown exception
                    failure = ServerException(str(exc))
                    span.record_exception(exc)
                    logger.error("Future failed: %s", record.request_id)
                    await self._mark_failed(record.request_id, failure, operation_type)
                else:
                    logger.info("Future completed: %s", record.request_id)
                    await self._mark_ready(record.request_id, payload, operation_type)
                finally:
                    # Record execution time
                    execution_time = time.perf_counter() - start_time
                    metrics.futures_wait_time.record(
                        wait_time, {"operation_type": operation_type or "unknown"}
                    )
                    metrics.futures_execution_time.record(
                        execution_time, {"operation_type": operation_type or "unknown"}
                    )
                    metrics.futures_queue_length.add(-1, {"queue_state": queue_state})

                    # Clean up task reference
                    task = asyncio.current_task()
                    if task:
                        self._tasks.discard(task)

        # Create and track the task
        task = asyncio.create_task(_runner())
        self._tasks.add(task)
        return types.UntypedAPIFuture(request_id=record.request_id, model_id=model_id)

    async def create_ready_future(
        self,
        payload: Any,
        user_id: str,
        *,
        model_id: str | None = None,
    ) -> types.UntypedAPIFuture:
        """Create a future that's already completed."""
        async with self._lock:
            future_id = self._allocate_future_id()
            record = FutureRecord(
                future_id=future_id,
                payload=payload,
                model_id=model_id,
                user_id=user_id,
                status="ready",
            )
            record.event.set()
            self._store_record(record)

        return types.UntypedAPIFuture(request_id=record.request_id, model_id=model_id)

    async def _mark_ready(
        self, request_id: str, payload: Any, operation_type: str | None = None
    ) -> None:
        """Mark a future as ready with the given payload."""
        async with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return
            record.payload = payload
            record.status = "ready"
            record.error = None
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_future, request_id)
            record.event.set()

            # Update metrics
            get_metrics().futures_completed.add(
                1,
                {
                    "operation_type": operation_type or record.operation_type or "unknown",
                    "status": "ready",
                },
            )

    async def _mark_failed(
        self,
        request_id: str,
        failure: TuFTException,
        operation_type: str | None = None,
    ) -> None:
        """Mark a future as failed with the given error."""
        async with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return
            record.status = "failed"
            record.error = failure
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_future, request_id)
            record.event.set()

            # Update metrics
            get_metrics().futures_completed.add(
                1,
                {
                    "operation_type": operation_type or record.operation_type or "unknown",
                    "status": "failed",
                },
            )

    async def retrieve(
        self,
        request_id: str,
        user_id: str,
        *,
        timeout: float = 120,
    ) -> Any:
        """
        Retrieve the result of a future, waiting if it's still pending.

        Args:
            request_id: The ID of the request to retrieve
            user_id: The ID of the user making the request
            timeout: Maximum time to wait in seconds (None for no timeout)

        Returns:
            The payload if ready, or error response if failed

        Raises:
            FutureNotFoundException: If request_id not found (may have expired due to TTL)
            UserMismatchException: If user_id does not match the owner
            asyncio.TimeoutError: If timeout is exceeded
        """
        # Get the record
        async with self._lock:
            record = self._records.get(request_id)

        if record is None:
            # Record not found - may have expired due to TTL or never existed
            raise FutureNotFoundException(request_id)
        if record.user_id != user_id:
            record.status = "failed"
            raise UserMismatchException()
        # Wait for completion if still pending
        if record.status == "pending":
            try:
                await asyncio.wait_for(record.event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                # Return TryAgainResponse on timeout for backwards compatibility
                return TryAgainResponse(request_id=request_id, queue_state=record.queue_state)

        # Return result
        if record.status == "failed" and record.error is not None:
            raise record.error

        return record.payload

    async def cleanup(self, request_id: str) -> None:
        """Remove a completed request from the store to free memory."""
        async with self._lock:
            self._records.pop(request_id, None)
            self._delete_future(request_id)

    async def shutdown(self) -> None:
        """Cancel all pending tasks and clean up."""
        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete (with cancellation)
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()
        self._records.clear()
