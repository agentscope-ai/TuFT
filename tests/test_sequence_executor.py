import asyncio
import time

import pytest

from tuft.exceptions import SequenceConflictException, SequenceTimeoutException
from tuft.sequence_executor import SequenceExecutor


async def _task_function(timeout=0.1):
    await asyncio.sleep(timeout)
    return time.time()


async def test_sequence_executor_concurrent_submit():
    executor = SequenceExecutor()
    results = await asyncio.gather(
        executor.submit(2, _task_function, timeout=0.1),
        executor.submit(1, _task_function, timeout=0.1),
        executor.submit(3, _task_function, timeout=0.1),
        executor.submit(0, _task_function, timeout=0.1),
        executor.submit(4, _task_function, timeout=0.1),
    )

    assert len(results) == 5
    # Check that results are in the order of sequence IDs
    assert results[3] < results[1] < results[0] < results[2] < results[4]


async def test_sequence_executor_with_exceptions():
    executor = SequenceExecutor()

    async def task_with_exception(sid: int):
        if sid == 2:
            raise ValueError("Intentional error for testing")
        await asyncio.sleep(0.1)
        return sid

    results = []
    exceptions = []

    async def submit_task(seq_id):
        try:
            result = await executor.submit(seq_id, task_with_exception, sid=seq_id)
            results.append(result)
        except Exception as e:
            exceptions.append((seq_id, e))

    await asyncio.gather(
        submit_task(0),
        submit_task(1),
        submit_task(2),
        submit_task(3),
        submit_task(4),
    )

    assert results == [0, 1, 3, 4]
    assert len(exceptions) == 1
    assert exceptions[0][0] == 2
    assert isinstance(exceptions[0][1], ValueError)
    assert str(exceptions[0][1]) == "Intentional error for testing"


@pytest.mark.asyncio
async def test_sequence_executor_out_of_order():
    executor = SequenceExecutor(timeout=1, next_sequence_id=0)

    task = asyncio.create_task(executor.submit(1, _task_function, timeout=0.1))
    await asyncio.sleep(0.3)  # Ensure task with seq_id=1 is waiting
    await executor.submit(0, _task_function, timeout=0.1)
    await task  # This should complete now
    await executor.submit(2, _task_function, timeout=0.1)

    with pytest.raises(SequenceTimeoutException) as exc_info:
        await executor.submit(4, _task_function, timeout=0.1)  # seq_id=3 is missing
    assert exc_info.value.sequence_id == 4

    with pytest.raises(SequenceConflictException) as exc_info:
        await executor.submit(2, _task_function, timeout=0.1)  # seq_id=2 already processed
    assert exc_info.value.expected == 3
    assert exc_info.value.got == 2
