"""File-backed Redis-like store for small demos and tests.

This module implements a minimal subset of redis-py behaviors with a JSON
backing file. It is designed for low-volume usage where performance is not a
concern. All write operations flush the full in-memory state to disk.

Example:
    from pathlib import Path

    from tuft.persistence.file_redis import FileRedis

    store = FileRedis(Path("~/.cache/tuft/file_redis.json").expanduser())
    store.set("alpha", "1")
    store.setex("beta", 5, "2")
    assert store.get("alpha") == "1"
    for key in store.scan_iter(match="a*"):
        print(key)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class _FileRedisValue:
    value: str
    expires_at: float | None


class FileRedis:
    """Tiny file-backed Redis-like store for tests and demos.

    Args:
        file_path: Path to the JSON file used for persistence.

    Example:
        from pathlib import Path

        store = FileRedis(Path("/tmp/file_redis.json"))
        store.set("key", "value")
        assert store.get("key") == "value"
    """

    def __init__(self, file_path: Path) -> None:
        self._file_path = Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, _FileRedisValue] = {}
        self._load()

    def _load(self) -> None:
        """Load persisted data from disk into memory."""
        if not self._file_path.exists():
            return
        try:
            raw = json.loads(self._file_path.read_text(encoding="utf-8"))
            for key, payload in raw.items():
                if not isinstance(payload, dict):
                    continue
                self._data[key] = _FileRedisValue(
                    value=str(payload.get("value", "")),
                    expires_at=payload.get("expires_at"),
                )
        except (json.JSONDecodeError, OSError):
            logger.exception("Failed to load FileRedis data from %s", self._file_path)

    def _dump(self) -> None:
        """Write the in-memory store to disk as JSON."""
        payload = {
            key: {"value": entry.value, "expires_at": entry.expires_at}
            for key, entry in self._data.items()
        }
        tmp_path = self._file_path.with_suffix(self._file_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        tmp_path.replace(self._file_path)

    def _purge_expired(self) -> None:
        """Remove expired keys and persist the updated store."""
        now = time.time()
        expired = [
            key
            for key, entry in self._data.items()
            if entry.expires_at and entry.expires_at <= now
        ]
        if expired:
            for key in expired:
                self._data.pop(key, None)
            self._dump()

    def set(self, key: str, value: str) -> bool:
        """Set a key to a string value.

        Args:
            key: Key to set.
            value: String value to store.

        Returns:
            True on success.
        """
        with self._lock:
            self._data[key] = _FileRedisValue(value=value, expires_at=None)
            self._dump()
        return True

    def setex(self, key: str, ttl_seconds: int | float, value: str) -> bool:
        """Set a key with TTL in seconds.

        Args:
            key: Key to set.
            ttl_seconds: Time-to-live in seconds.
            value: String value to store.

        Returns:
            True on success.
        """
        with self._lock:
            expires_at = time.time() + float(ttl_seconds)
            self._data[key] = _FileRedisValue(value=value, expires_at=expires_at)
            self._dump()
        return True

    def get(self, key: str) -> str | None:
        """Get a string value by key.

        Args:
            key: Key to retrieve.

        Returns:
            The stored value, or None if missing/expired.
        """
        with self._lock:
            self._purge_expired()
            entry = self._data.get(key)
            return entry.value if entry else None

    def delete(self, *keys: str) -> int:
        """Delete one or more keys.

        Args:
            *keys: Keys to delete.

        Returns:
            Number of keys removed.
        """
        removed = 0
        with self._lock:
            for key in keys:
                if key in self._data:
                    self._data.pop(key, None)
                    removed += 1
            if removed:
                self._dump()
        return removed

    def exists(self, key: str) -> int:
        """Check if a key exists.

        Args:
            key: Key to check.

        Returns:
            1 if the key exists, otherwise 0.
        """
        with self._lock:
            self._purge_expired()
            return 1 if key in self._data else 0

    def scan_iter(self, match: str | None = None) -> Iterable[str]:
        """Iterate over keys matching a pattern.

        Args:
            match: Optional glob pattern (e.g., "prefix:*").

        Returns:
            An iterator of matching keys.
        """
        with self._lock:
            self._purge_expired()
            keys = list(self._data.keys())
        pattern = match or "*"
        for key in keys:
            if fnmatch(key, pattern):
                yield key

    def pipeline(self, transaction: bool = True) -> "FileRedisPipeline":
        """Create a pipeline for batched operations.

        Args:
            transaction: Ignored; kept for compatibility.

        Returns:
            A FileRedisPipeline instance.
        """
        _ = transaction  # kept for signature compatibility
        return FileRedisPipeline(self)

    def close(self) -> None:
        """No-op close for API compatibility."""
        return None


class FileRedisPipeline:
    """Minimal pipeline that writes once on exit.

    Example:
        with store.pipeline() as pipe:
            pipe.set("a", "1")
            pipe.setex("b", 10, "2")
            pipe.delete("c")
    """

    def __init__(self, store: FileRedis) -> None:
        self._store = store
        self._ops: list[tuple[str, tuple]] = []

    def __enter__(self) -> "FileRedisPipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self._execute()

    def set(self, key: str, value: str) -> "FileRedisPipeline":
        """Queue a SET operation."""
        self._ops.append(("set", (key, value)))
        return self

    def setex(self, key: str, ttl_seconds: int | float, value: str) -> "FileRedisPipeline":
        """Queue a SETEX operation."""
        self._ops.append(("setex", (key, ttl_seconds, value)))
        return self

    def delete(self, *keys: str) -> "FileRedisPipeline":
        """Queue a DELETE operation."""
        self._ops.append(("delete", keys))
        return self

    def execute(self) -> list[object]:
        """Execute queued operations (redis-py compatibility)."""
        self._execute()
        return []

    def _execute(self) -> None:
        """Apply queued operations and flush to disk."""
        with self._store._lock:
            for op, args in self._ops:
                if op == "set":
                    key, value = args
                    self._store._data[key] = _FileRedisValue(value=value, expires_at=None)
                elif op == "setex":
                    key, ttl_seconds, value = args
                    expires_at = time.time() + float(ttl_seconds)
                    self._store._data[key] = _FileRedisValue(value=value, expires_at=expires_at)
                elif op == "delete":
                    for key in args:
                        self._store._data.pop(key, None)
            if self._ops:
                self._store._dump()
            self._ops.clear()

