"""Simple in-memory future registry for the synthetic Tinker API."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, Literal

from tinker import types
from tinker.types.try_again_response import TryAgainResponse

from .exceptions import FutureNotFoundException, LLMRPCException, UserMismatchException
from .persistence import (
    PersistedMarker,
    persistable,
    redis_persistent,
    unwrap_proxy,
)

QueueState = Literal["active", "paused_capacity", "paused_rate_limit"]

# Operation types for persistence and recovery
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


@persistable
@dataclass
class FutureRecord:
    """Future record with persistence support.

    Fields:
        event: Not serialized (init=False) - created fresh on each instance.
               After restore, if status is ready/failed, event is auto-set.
        operation_type: Type of operation for recovery purposes.
        operation_args: Serializable arguments for the operation.
        created_at: Timestamp when the future was created.
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str | None = None
    user_id: str | None = None
    queue_state: QueueState = "active"
    status: Literal["pending", "ready", "failed"] = "pending"
    payload: Any | None = None
    error: types.RequestFailedResponse | None = None
    # Operation info for recovery
    operation_type: OperationType | None = None
    operation_args: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=_now)
    # Not serialized - init=False fields are auto-excluded
    event: asyncio.Event = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # TODO: connect future event to checkpoint store
        self.event = asyncio.Event()
        # If already completed, set the event
        if self.status in ("ready", "failed"):
            self.event.set()


@redis_persistent(
    restore_callback="_on_restore",
)
class FutureStore:
    """Runs controller work asynchronously and tracks each request's lifecycle.

    Used for retrieve_future polling.
    The _records dict is automatically persisted to Redis.
    """

    # Persisted field - auto-synced to Redis
    _records: Annotated[dict[str, FutureRecord], PersistedMarker()]

    def __init__(self) -> None:
        # _records is auto-initialized by @redis_persistent decorator
        self._lock = asyncio.Lock()
        self._tasks = set()
        self._needs_checkpoint_recovery = False

    def _on_restore(self) -> None:
        """Restore callback: handle pending tasks after Redis restore.

        Called only when data is restored from Redis (server restart scenario).

        This method sets up state for checkpoint-based recovery:
        - For completed futures: ensure event is set
        - For pending futures: defer to async_restore_with_checkpoints()

        Note: Pending futures are NOT marked as failed here. They will be
        handled by async_restore_with_checkpoints() which has access to
        checkpoint information for proper recovery.
        """
        has_pending = False
        for request_id in list(self._records.keys()):
            record = self._records[request_id]
            actual_record = unwrap_proxy(record)

            if actual_record.status == "pending":
                # Don't mark as failed yet - defer to async checkpoint recovery
                has_pending = True
            else:
                # Already completed - ensure event is set
                actual_record.event.set()

        self._needs_checkpoint_recovery = has_pending

    def get_pending_futures_by_model(self) -> dict[str | None, list[FutureRecord]]:
        """Group all pending futures by model_id.

        Returns:
            Dict mapping model_id to list of pending FutureRecords, sorted by created_at.
        """
        by_model: dict[str | None, list[FutureRecord]] = {}
        for record in self._records.values():
            actual = unwrap_proxy(record)
            if actual.status == "pending":
                if actual.model_id not in by_model:
                    by_model[actual.model_id] = []
                by_model[actual.model_id].append(actual)

        # Sort each list by created_at
        for model_id in by_model:
            by_model[model_id].sort(key=lambda r: r.created_at)

        return by_model

    def mark_futures_failed_after_checkpoint(
        self,
        model_id: str | None,
        checkpoint_time: datetime | None,
        error_message: str = "Server restored from checkpoint. Please retry.",
    ) -> int:
        """Mark all pending futures for a model after a checkpoint time as failed.

        Args:
            model_id: The model ID to filter futures.
            checkpoint_time: The checkpoint creation time. Futures created after
                           this time will be marked as failed. If None, all
                           pending futures for this model are marked as failed.
            error_message: Error message to include in the failure response.

        Returns:
            Number of futures marked as failed.
        """
        count = 0
        for request_id in list(self._records.keys()):
            record = self._records[request_id]
            actual = unwrap_proxy(record)

            if actual.status != "pending":
                continue

            if actual.model_id != model_id:
                continue

            # If no checkpoint time, mark all as failed
            # If checkpoint time exists, only mark futures created after it
            if checkpoint_time is None or actual.created_at > checkpoint_time:
                actual.status = "failed"
                actual.error = types.RequestFailedResponse(
                    error=error_message,
                    category=types.RequestErrorCategory.Server,
                )
                actual.event.set()
                count += 1

        return count

    def mark_all_pending_failed(
        self,
        error_message: str = "Server restarted while task was pending. Please retry.",
    ) -> int:
        """Mark all pending futures as failed.

        This is a fallback for when checkpoint recovery is not possible.

        Returns:
            Number of futures marked as failed.
        """
        count = 0
        for request_id in list(self._records.keys()):
            record = self._records[request_id]
            actual = unwrap_proxy(record)

            if actual.status == "pending":
                actual.status = "failed"
                actual.error = types.RequestFailedResponse(
                    error=error_message,
                    category=types.RequestErrorCategory.Server,
                )
                actual.event.set()
                count += 1

        return count

    def _store_record(self, record: FutureRecord) -> None:
        """Synchronous method to store record (call within lock context)."""
        self._records[record.request_id] = record

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
            operation: The callable to execute asynchronously.
            user_id: User ID associated with this operation.
            model_id: Optional model ID associated with this operation.
            queue_state: Current queue state for retry handling.
            operation_type: Type of operation for persistence/recovery.
            operation_args: Serializable arguments for recovery purposes.
        """
        record = FutureRecord(
            model_id=model_id,
            user_id=user_id,
            queue_state=queue_state,
            operation_type=operation_type,
            operation_args=operation_args,
        )

        async with self._lock:
            self._store_record(record)

        async def _runner() -> None:
            try:
                if asyncio.iscoroutinefunction(operation):
                    payload = await operation()
                else:
                    # Run sync operation in thread pool to avoid blocking
                    loop = asyncio.get_running_loop()
                    payload = await loop.run_in_executor(None, operation)
            except LLMRPCException as exc:
                message = exc.detail
                failure = types.RequestFailedResponse(
                    error=message,
                    category=types.RequestErrorCategory.User,
                )
                await self._mark_failed(record.request_id, failure)
            except Exception as exc:  # pylint: disable=broad-except
                failure = types.RequestFailedResponse(
                    error=str(exc),
                    category=types.RequestErrorCategory.Server,
                )
                await self._mark_failed(record.request_id, failure)
            else:
                await self._mark_ready(record.request_id, payload)
            finally:
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
        record = FutureRecord(payload=payload, model_id=model_id, user_id=user_id, status="ready")
        record.event.set()

        async with self._lock:
            self._store_record(record)

        return types.UntypedAPIFuture(request_id=record.request_id, model_id=model_id)

    async def _mark_ready(self, request_id: str, payload: Any) -> None:
        """Mark a future as ready with the given payload."""
        async with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return
            # record.payload = payload
            # record.status = "ready"
            # record.error = None
            # record.event.set()

            # use unwrap_proxy to reduce redis sync times
            actual = unwrap_proxy(record)
            actual.payload = payload
            actual.status = "ready"
            actual.error = None
            actual.event.set()
            self._records[request_id] = actual

    async def _mark_failed(self, request_id: str, failure: types.RequestFailedResponse) -> None:
        """Mark a future as failed with the given error."""
        async with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return
            # record.status = "failed"
            # record.error = failure
            # record.event.set()

            # use unwrap_proxy to reduce redis sync times
            actual = unwrap_proxy(record)
            actual.status = "failed"
            actual.error = failure
            actual.event.set()
            self._records[request_id] = actual

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
            FutureNotFoundException: If request_id not found
            UserMismatchException: If user_id does not match the owner
            asyncio.TimeoutError: If timeout is exceeded
        """
        # Get the record
        async with self._lock:
            record = self._records.get(request_id)

        if record is None:
            raise FutureNotFoundException(request_id)
        if record.user_id != user_id:
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
            return record.error

        return record.payload

    async def cleanup(self, request_id: str) -> None:
        """Remove a completed request from the store to free memory."""
        async with self._lock:
            self._records.pop(request_id, None)

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
        # Note: We don't clear _records here as they are persisted
        # They will be handled on next server start via _on_restore

    @property
    def needs_checkpoint_recovery(self) -> bool:
        """Check if checkpoint-based recovery is needed."""
        return self._needs_checkpoint_recovery

    def clear_recovery_flag(self) -> None:
        """Clear the recovery flag after successful recovery."""
        self._needs_checkpoint_recovery = False
