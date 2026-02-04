import asyncio
import heapq
from typing import Any, Callable

from .exceptions import SequenceTimeoutException


class SequenceExecutor:
    """
    An executor that processes tasks according to their `sequence_id` order.

    Note: This executor only guarantees that tasks in the queue are executed in
    ascending order of `sequence_id`. It does not guarantee that earlier finished
    tasks always have smaller `sequence_id` than later finished ones. For example,
    if tasks with sequence_id=1, 2, 3, 4 are submitted concurrently, they will be
    executed in order. If a task with sequence_id=0 are submitted after these finish,
    it will be executed only after all previous tasks complete.
    """

    def __init__(self, timeout: float = 900) -> None:
        self.pending_heap = []  # (sequence_id, func, kwargs, future)
        self.heap_lock = asyncio.Lock()
        self._processing = False
        self.timeout = timeout

    async def submit(self, sequence_id: int, func: Callable, **kwargs) -> Any:
        """Submit a task with a specific sequence_id.

        Args:
            sequence_id (int): The sequence ID of the task.
            func (Callable): The async function to execute.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Any: The result of the function execution.

        Raises:
            SequenceTimeoutException: If the task times out waiting for its turn.
        """
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
                # get the next task to process
                _, func, kwargs, future = heapq.heappop(self.pending_heap)
            try:
                result = await func(**kwargs)
                if not future.done():
                    future.set_result(result)
            except Exception as e:
                if not future.done():
                    future.set_exception(e)
