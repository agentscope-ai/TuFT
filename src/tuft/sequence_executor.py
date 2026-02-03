import asyncio
import heapq
from typing import Any, Callable

from .exceptions import SequenceConflictException, SequenceTimeoutException


class SequenceExecutor:
    """An executor that processes tasks strictly in the order of their `sequence_id`.
    Out-of-order tasks are buffered until all previous sequence ids have been processed.
    """

    def __init__(self, timeout: float = 900, next_sequence_id: int = 0) -> None:
        self.pending_heap = []  # (sequence_id, func, kwargs, future)
        self.heap_lock = asyncio.Lock()
        self.next_sequence_id = next_sequence_id
        self._processing = False
        self.timeout = timeout

    async def submit(self, sequence_id: int, func: Callable, **kwargs) -> Any:
        if sequence_id < self.next_sequence_id:
            raise SequenceConflictException(expected=self.next_sequence_id, got=sequence_id)
        future = asyncio.Future()
        async with self.heap_lock:
            heapq.heappush(self.pending_heap, (sequence_id, func, kwargs, future))
            # Start processing if not already running
            if not self._processing:
                self._processing = True
                asyncio.create_task(self._process_tasks())
        try:
            result = await asyncio.wait_for(future, timeout=self.timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            raise SequenceTimeoutException(sequence_id) from e
        return result

    async def _process_tasks(self):
        while True:
            async with self.heap_lock:
                if not self.pending_heap:
                    self._processing = False
                    break
                # Peek at the smallest sequence_id
                sequence_id, func, kwargs, future = self.pending_heap[0]
                if sequence_id != self.next_sequence_id:
                    # wait next sequence_id
                    self._processing = False
                    break

                heapq.heappop(self.pending_heap)
                self.next_sequence_id += 1
            try:
                result = await func(**kwargs)
                if not future.done():
                    future.set_result(result)
            except Exception as e:
                if not future.done():
                    future.set_exception(e)
