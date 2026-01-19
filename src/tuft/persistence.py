"""Simple Redis persistence module for TuFT.

This module provides direct Redis-based persistence using redis-py.
Each data record is stored as a separate Redis key with JSON serialization.

Key Design:
- Top-level records: {namespace}::{type}::{id}
- Nested records: {namespace}::{type}::{parent_id}::{nested_type}::{nested_id}

Persistence Modes:
- disabled: No persistence, all data is in-memory only
- redis_url: Use external Redis server via URL
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class PersistenceMode(str, Enum):
    """Persistence mode options."""

    DISABLED = "disabled"  # No persistence
    REDIS_URL = "redis_url"  # Use external Redis server


# Default TTL values in seconds
DEFAULT_FUTURE_TTL_SECONDS = 24 * 3600  # 1 day for future records (short-lived)


@dataclass
class PersistenceConfig:
    """Configuration for Redis persistence.

    Attributes:
        mode: Persistence mode - disabled or redis_url
        redis_url: Redis server URL (only used when mode=redis_url)
        namespace: Key namespace prefix
        future_ttl_seconds: TTL for future records in seconds. None means no expiry.
    """

    mode: PersistenceMode = PersistenceMode.DISABLED
    redis_url: str = "redis://localhost:6379/0"
    namespace: str = "tuft"
    future_ttl_seconds: int | None = DEFAULT_FUTURE_TTL_SECONDS  # Futures expire after 1 day

    @property
    def enabled(self) -> bool:
        """Check if persistence is enabled."""
        return self.mode != PersistenceMode.DISABLED

    @classmethod
    def disabled(cls, namespace: str = "tuft") -> "PersistenceConfig":
        """Create a disabled persistence config."""
        return cls(mode=PersistenceMode.DISABLED, namespace=namespace)

    @classmethod
    def from_redis_url(
        cls,
        redis_url: str,
        namespace: str = "tuft",
        future_ttl_seconds: int | None = DEFAULT_FUTURE_TTL_SECONDS,
    ) -> "PersistenceConfig":
        """Create a config using external Redis server."""
        return cls(
            mode=PersistenceMode.REDIS_URL,
            redis_url=redis_url,
            namespace=namespace,
            future_ttl_seconds=future_ttl_seconds,
        )


class RedisStore:
    """Global Redis connection and operation manager.

    Supports two modes:
    - External Redis server (via redis-py)
    - No persistence (disabled mode)
    """

    _instance: "RedisStore | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._redis: Any = None
        self._config: PersistenceConfig | None = None
        self._pid: int | None = None

    @classmethod
    def get_instance(cls) -> "RedisStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def configure(self, config: PersistenceConfig) -> None:
        self._config = config
        self._close_connections()
        self._pid = None

    def _close_connections(self) -> None:
        """Close all Redis connections."""
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                logger.exception("Failed to close Redis connection")
            self._redis = None

    def _get_redis(self) -> Any:
        if self._config is None or not self._config.enabled:
            return None

        current_pid = os.getpid()
        if self._redis is None or self._pid != current_pid:
            with self._lock:
                if self._redis is None or self._pid != current_pid:
                    self._close_connections()

                    if self._config.mode == PersistenceMode.REDIS_URL:
                        self._redis = self._create_redis_client()

                    if self._redis is not None:
                        self._pid = current_pid

        return self._redis

    def _create_redis_client(self) -> Any:
        """Create a redis-py client for external Redis server."""
        if self._config is None:
            return None
        try:
            import redis

            return redis.Redis.from_url(self._config.redis_url, decode_responses=True)
        except ImportError:
            logger.warning("redis package not installed, persistence will be disabled")
            return None

    @property
    def is_enabled(self) -> bool:
        return self._config is not None and self._config.enabled

    @property
    def namespace(self) -> str:
        return self._config.namespace if self._config else "tuft"

    @property
    def future_ttl(self) -> int | None:
        """Get the TTL for future records in seconds."""
        return self._config.future_ttl_seconds if self._config else DEFAULT_FUTURE_TTL_SECONDS

    def close(self) -> None:
        self._close_connections()
        self._pid = None

    def reset(self) -> None:
        self.close()
        self._config = None

    def build_key(self, *parts: str) -> str:
        """Build a Redis key from parts using :: as separator."""
        escaped = [p.replace("::", "__SEP__") for p in parts]
        return "::".join([self.namespace] + escaped)

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> bool:
        redis = self._get_redis()
        if redis is None:
            return False
        try:
            if ttl_seconds is not None:
                redis.setex(key, ttl_seconds, value)
            else:
                redis.set(key, value)
            return True
        except Exception:
            logger.exception("Failed to set key %s in Redis", key)
            return False

    def get(self, key: str) -> str | None:
        redis = self._get_redis()
        if redis is None:
            return None
        try:
            return redis.get(key)
        except Exception:
            logger.exception("Failed to get key %s from Redis", key)
            return None

    def delete(self, key: str) -> bool:
        redis = self._get_redis()
        if redis is None:
            return False
        try:
            redis.delete(key)
            return True
        except Exception:
            logger.exception("Failed to delete key %s from Redis", key)
            return False

    def keys(self, pattern: str) -> list[str]:
        """Get all keys matching the pattern using SCAN for better performance."""
        redis = self._get_redis()
        if redis is None:
            return []
        try:
            return list(redis.scan_iter(match=pattern))
        except Exception:
            logger.exception("Failed to scan keys with pattern %s from Redis", pattern)
            return []

    def delete_pattern(self, pattern: str) -> int:
        redis = self._get_redis()
        if redis is None:
            return 0
        try:
            keys = list(redis.scan_iter(match=pattern))
            if keys:
                return redis.delete(*keys)
            return 0
        except Exception:
            logger.exception("Failed to delete keys with pattern %s from Redis", pattern)
            return 0

    def exists(self, key: str) -> bool:
        redis = self._get_redis()
        if redis is None:
            return False
        try:
            return redis.exists(key) > 0
        except Exception:
            logger.exception("Failed to check existence of key %s in Redis", key)
            return False

    def pipeline(self) -> "RedisPipeline":
        """Create a pipeline for atomic batch operations.

        Usage:
            with store.pipeline() as pipe:
                pipe.set("key1", "value1")
                pipe.set("key2", "value2")
            # All operations are executed atomically on context exit
        """
        return RedisPipeline(self)


class RedisPipeline:
    """Pipeline for atomic batch Redis operations using MULTI/EXEC transactions."""

    def __init__(self, store: RedisStore) -> None:
        self._store = store
        self._redis = store._get_redis()
        self._pipe: Any = None
        if self._redis is not None:
            self._pipe = self._redis.pipeline(transaction=True)

    def __enter__(self) -> "RedisPipeline":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is None and self._pipe is not None:
            try:
                self._pipe.execute()
            except Exception:
                logger.exception("Failed to execute Redis pipeline")

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> "RedisPipeline":
        """Add a SET operation to the pipeline."""
        if self._pipe is not None:
            if ttl_seconds is not None:
                self._pipe.setex(key, ttl_seconds, value)
            else:
                self._pipe.set(key, value)
        return self

    def delete(self, key: str) -> "RedisPipeline":
        """Add a DELETE operation to the pipeline."""
        if self._pipe is not None:
            self._pipe.delete(key)
        return self


def save_record(key: str, record: BaseModel, ttl_seconds: int | None = None) -> bool:
    """Save a Pydantic model record to Redis.

    Args:
        key: Redis key to store the record under.
        record: Pydantic BaseModel instance to serialize and store.
        ttl_seconds: Optional TTL in seconds for the key. If None, no expiry is set.

    Returns:
        True if the record was saved successfully, False otherwise.
    """
    store = RedisStore.get_instance()
    if not store.is_enabled:
        return False
    try:
        # Use Pydantic's model_dump_json for serialization
        json_str = record.model_dump_json()
        return store.set(key, json_str, ttl_seconds=ttl_seconds)
    except Exception:
        logger.exception("Failed to save record with key %s to Redis", key)
        return False


def save_records_atomic(
    records: list[tuple[str, BaseModel]], ttl_seconds: int | None = None
) -> bool:
    """Save multiple Pydantic model records to Redis atomically using a transaction.

    Args:
        records: List of (key, record) tuples.
        ttl_seconds: Optional TTL in seconds for all keys. If None, no expiry is set.

    Returns:
        True if all records were saved successfully, False otherwise.
    """
    store = RedisStore.get_instance()
    if not store.is_enabled:
        return False
    try:
        with store.pipeline() as pipe:
            for key, record in records:
                json_str = record.model_dump_json()
                pipe.set(key, json_str, ttl_seconds=ttl_seconds)
        return True
    except Exception:
        logger.exception("Failed to save records atomically to Redis")
        return False


def load_record(key: str, target_class: type[T]) -> T | None:
    """Load a Pydantic model record from Redis.

    Args:
        key: Redis key to load from.
        target_class: The Pydantic model class to deserialize into.

    Returns:
        The deserialized record, or None if not found or on error.
    """
    store = RedisStore.get_instance()
    if not store.is_enabled:
        return None
    try:
        json_str = store.get(key)
        if json_str is None:
            return None
        return target_class.model_validate_json(json_str)
    except Exception:
        logger.exception("Failed to load record with key %s from Redis", key)
        return None


def delete_record(key: str) -> bool:
    """Delete a record from Redis."""
    store = RedisStore.get_instance()
    if not store.is_enabled:
        return False
    return store.delete(key)


def is_persistence_enabled() -> bool:
    """Check if persistence is enabled."""
    return RedisStore.get_instance().is_enabled


def get_redis_store() -> RedisStore:
    """Get the global Redis store instance."""
    return RedisStore.get_instance()
