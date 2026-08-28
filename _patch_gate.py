"""Temporary patch helper (deleted after use)."""

from pathlib import Path

p = Path("src/tuft/runtime/unified_engine_gate.py")
src = p.read_text()
old = '''    async def acquire_training(self) -> None:
        """Enter training mode; sampling waits while training is in flight."""
        async with self._cond:
            self._inflight_training += 1
            self._cond.notify_all()'''
new = '''    async def acquire_training(self) -> None:
        """Enter training mode; sampling waits while training is in flight.

        The unified engine is a single runtime: training steps from different
        tenants are serialized here so they never contend on the shared model
        (concurrent forwards on one torch runtime serialize on its internal
        lock, stall, and trigger client-side retries / sequence conflicts).
        """
        async with self._cond:
            while self._inflight_training > 0:
                await self._cond.wait()
            self._inflight_training += 1
            self._cond.notify_all()'''
count = src.count(old)
print(f"pattern count={count}")
if count == 1:
    p.write_text(src.replace(old, new))
    print("patched")
else:
    print("current snippet around acquire_training:")
    idx = src.find("async def acquire_training")
    print(src[idx: idx + 400])
