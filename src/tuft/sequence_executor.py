import asyncio
import heapq
from typing import Any, Callable


class SequenceExecutor:
    """An executor that processes tasks in the order of their `sequence_id`.
    If tasks are added to the queue out of order, they will be buffered until
    all prior tasks have been processed. But if there are no pending tasks,
    new tasks will be processed immediately.
    """

    def __init__(self):
        self.task_queue = asyncio.Queue()
        self.pending_heap = []
        self.heap_lock = asyncio.Lock()
        self.processor_task = None

    async def submit(self, sequence_id: int, func: Callable, **kwargs) -> Any:
        """Submit a task with a specific sequence_id for ordered execution.

        Args:
            sequence_id: The sequence ID to determine execution order.
            func: The callable to execute. It should be an async function.
            **kwargs: Arguments to pass to the callable.
        """
        future = asyncio.Future()
        await self.task_queue.put((sequence_id, func, kwargs, future))
        if self.processor_task is None or self.processor_task.done():
            self.processor_task = asyncio.create_task(self._process_tasks())
        result = await future
        return result

    async def _process_tasks(self):
        while True:
            try:
                task_info = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
                async with self.heap_lock:
                    heapq.heappush(self.pending_heap, task_info)
            except asyncio.TimeoutError:
                async with self.heap_lock:
                    if not self.pending_heap and self.task_queue.empty():
                        break
            async with self.heap_lock:
                if self.pending_heap:
                    _, func, kwargs, future = heapq.heappop(self.pending_heap)
                else:
                    continue
            try:
                result = await func(**kwargs)
                if not future.done():
                    future.set_result(result)
            except Exception as e:
                if not future.done():
                    future.set_exception(e)
