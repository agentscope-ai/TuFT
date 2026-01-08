"""Simple in-memory future registry for the synthetic Tinker API."""

from __future__ import annotations

import asyncio
import uuid
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
    event: asyncio.Event = field(default_factory=asyncio.Event)


class FutureStore:
    """Runs controller work asynchronously and tracks each request's lifecycle.

    Used for retrieve_future polling.
    """

    def __init__(self) -> None:
        self._records: dict[str, FutureRecord] = {}
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    def _store_record(self, record: FutureRecord) -> None:
        """Synchronous method to store record (call within lock context)."""
        self._records[record.request_id] = record

    async def enqueue(
        self,
        operation: Callable[[], Any],
        *,
        model_id: str | None = None,
        queue_state: QueueState = "active",
    ) -> types.UntypedAPIFuture:
        """Enqueue a task (sync or async) and return a future immediately."""
        record = FutureRecord(model_id=model_id, queue_state=queue_state)

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
            except HTTPException as exc:
                message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
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
        *,
        model_id: str | None = None,
    ) -> types.UntypedAPIFuture:
        """Create a future that's already completed."""
        record = FutureRecord(payload=payload, model_id=model_id, status="ready")
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
            record.payload = payload
            record.status = "ready"
            record.error = None
            record.event.set()

    async def _mark_failed(self, request_id: str, failure: types.RequestFailedResponse) -> None:
        """Mark a future as failed with the given error."""
        async with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return
            record.status = "failed"
            record.error = failure
            record.event.set()

    async def retrieve(
        self,
        request_id: str,
        *,
        timeout: float = 120,
    ) -> Any:
        """
        Retrieve the result of a future, waiting if it's still pending.

        Args:
            request_id: The ID of the request to retrieve
            timeout: Maximum time to wait in seconds (None for no timeout)

        Returns:
            The payload if ready, or error response if failed

        Raises:
            KeyError: If request_id not found
            asyncio.TimeoutError: If timeout is exceeded
        """
        # Get the record
        async with self._lock:
            record = self._records.get(request_id)

        if record is None:
            raise KeyError(request_id)

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
        self._records.clear()
