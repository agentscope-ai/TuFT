"""Ordering and failure recovery without invoking a training backend."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from tuft.exceptions import SequenceConflictException
from tuft.persistence import load_record
from tuft.training_controller import TrainingController, TrainingRunRecord


@pytest.fixture
def guard():
    controller = object.__new__(TrainingController)
    record = TrainingRunRecord(
        training_run_id="sequence-test",
        base_model="test-model",
        lora_rank=4,
        session_id="session",
        model_owner="tester",
    )
    controller.training_runs = {record.training_run_id: record}
    return controller, record


async def test_later_request_cannot_execute_before_earlier_request(guard):
    controller, record = guard
    forward = AsyncMock()
    optim = AsyncMock()
    with pytest.raises(SequenceConflictException):
        await controller._with_sequence_guard(record, 2, optim)
    optim.assert_not_awaited()
    assert record.next_seq_id == 1
    await controller._with_sequence_guard(record, 1, forward)
    await controller._with_sequence_guard(record, 2, optim)
    forward.assert_awaited_once()
    optim.assert_awaited_once()
    assert record.next_seq_id == 3


@pytest.mark.parametrize("retry", [False, True])
async def test_failed_slot_can_be_retried_or_abandoned_after_restore(guard, retry):
    controller, record = guard
    with pytest.raises(RuntimeError, match="out of memory"):
        await controller._with_sequence_guard(
            record, 1, AsyncMock(side_effect=RuntimeError("out of memory"))
        )
    restored = load_record(controller._build_key(record.training_run_id), TrainingRunRecord)
    assert restored is not None
    assert restored.next_seq_id == 1
    assert restored.failed_seq_id == 1
    controller.training_runs[restored.training_run_id] = restored
    await controller._with_sequence_guard(restored, 1 if retry else 2, AsyncMock())
    assert restored.next_seq_id == (2 if retry else 3)
    assert restored.failed_seq_id is None
    with pytest.raises(SequenceConflictException):
        await controller._with_sequence_guard(restored, 1, AsyncMock())


async def test_pending_operation_is_not_a_failed_slot(guard):
    controller, record = guard
    entered = asyncio.Event()
    release = asyncio.Event()
    optim = AsyncMock()

    async def forward():
        entered.set()
        await release.wait()

    first = asyncio.create_task(controller._with_sequence_guard(record, 1, forward))
    await entered.wait()
    later = asyncio.create_task(controller._with_sequence_guard(record, 3, optim))
    await asyncio.sleep(0)
    optim.assert_not_awaited()
    release.set()
    await first
    with pytest.raises(SequenceConflictException):
        await later
    optim.assert_not_awaited()
    assert record.next_seq_id == 2


@pytest.mark.parametrize("seq_id", [1, 2])
async def test_cancelled_retry_or_successor_clears_old_failure_evidence(guard, seq_id):
    controller, record = guard
    with pytest.raises(RuntimeError):
        await controller._with_sequence_guard(record, 1, AsyncMock(side_effect=RuntimeError()))
    with pytest.raises(asyncio.CancelledError):
        await controller._with_sequence_guard(
            record, seq_id, AsyncMock(side_effect=asyncio.CancelledError())
        )
    restored = load_record(controller._build_key(record.training_run_id), TrainingRunRecord)
    assert restored is not None
    assert restored.next_seq_id == seq_id
    assert restored.failed_seq_id is None
    later = AsyncMock()
    with pytest.raises(SequenceConflictException):
        await controller._with_sequence_guard(restored, seq_id + 1, later)
    later.assert_not_awaited()


async def test_consecutive_failures_only_allow_the_immediate_successor(guard):
    controller, record = guard
    for seq_id in (1, 2):
        with pytest.raises(RuntimeError):
            await controller._with_sequence_guard(
                record, seq_id, AsyncMock(side_effect=RuntimeError())
            )
    later = AsyncMock()
    with pytest.raises(SequenceConflictException):
        await controller._with_sequence_guard(record, 4, later)
    later.assert_not_awaited()
    assert record.next_seq_id == 2
    assert record.failed_seq_id == 2
    await controller._with_sequence_guard(record, 3, later)
    later.assert_awaited_once()
