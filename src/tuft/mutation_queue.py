"""Exclusive mutation queue with bounded overtaking and per-run FIFO order."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass(eq=False)
class _Waiter:
    tenant: str | None
    priority: bool
    ready: asyncio.Future[bool]


class MutationQueue:
    def __init__(self, max_overtakes: int) -> None:
        self.max_overtakes = max_overtakes
        self._overtakes = 0
        self._waiting: list[_Waiter] = []
        self._owner: _Waiter | None = None

    def _grant(self) -> None:
        if self._owner is not None or not self._waiting:
            return
        selected = 0
        seen = set()
        if self._overtakes < self.max_overtakes:
            for index, waiter in enumerate(self._waiting):
                # Unidentified maintenance may affect any adapter: it is a
                # global ordering barrier, not a candidate for overtaking.
                if waiter.tenant is None:
                    break
                if waiter.priority and waiter.tenant not in seen:
                    selected = index
                    break
                seen.add(waiter.tenant)
        self._overtakes = self._overtakes + 1 if selected else 0
        self._owner = self._waiting.pop(selected)
        self._owner.ready.set_result(selected > 0)

    @asynccontextmanager
    async def reserve(self, tenant: str | None, priority: bool) -> AsyncIterator[bool]:
        waiter = _Waiter(tenant, priority, asyncio.get_running_loop().create_future())
        self._waiting.append(waiter)
        self._grant()
        try:
            yield await asyncio.shield(waiter.ready)
        finally:
            if self._owner is waiter:
                self._owner = None
            else:
                self._waiting.remove(waiter)
            self._grant()
