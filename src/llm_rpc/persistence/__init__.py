"""
Redis persistence module for LLM-RPC.

This module provides transparent Redis-based persistence for class attributes
using a decorator-based approach. It supports both dataclass and Pydantic models.

## Key Features

- **Auto-sync**: ALL modifications to @persistable objects automatically sync to Redis
  - Container operations (add/remove keys) sync immediately
  - Attribute modifications on @persistable objects sync automatically
- **Efficient**: Each nested object gets its own Redis key
- **Type-safe**: Works with dataclass and Pydantic models
- **Complete**: Support for all Pottery container types
- **Optional**: Persistence is optional - when disabled, uses in-memory storage

## Configuration

Persistence can be configured via:
1. Environment variables (highest priority):
   - LLM_RPC_PERSISTENCE_ENABLED: "true" or "false"
   - LLM_RPC_REDIS_URL: Redis connection URL
   - LLM_RPC_PERSISTENCE_NAMESPACE: Key namespace prefix
   - LLM_RPC_PERSISTENCE_INSTANCE_ID: Instance identifier

2. CLI options:
   - --persistence/--no-persistence
   - --redis-url
   - --persistence-namespace
   - --persistence-instance-id

3. Configuration file (AppConfig.persistence)

## Usage Overview

### 1. Mark models as persistable

Use the @persistable decorator on dataclass/Pydantic models that will be stored.
Use PersistenceExclude to mark fields that should not be serialized.

```python
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict
from llm_rpc.persistence import persistable, PersistenceExclude


@persistable
@dataclass
class CheckpointRecord:
    checkpoint_id: str
    path: Path
    size_bytes: int


@persistable
@dataclass
class TrainingRunRecord:
    training_run_id: str
    base_model: str

    # Excluded from serialization - rebuilt via __post_deserialize__
    backend: Annotated[Any, PersistenceExclude()]

    # Excluded with a default factory - creates new lock on deserialize
    _execution_lock: Annotated[asyncio.Lock, PersistenceExclude(default_factory=asyncio.Lock)]

    # Excluded with a default value
    _cache: Annotated[dict, PersistenceExclude(default_factory=dict)]

    # Nested dict - automatically persisted with its own Redis key
    checkpoints: Dict[str, CheckpointRecord] = field(default_factory=dict)

    def __post_deserialize__(self):
        '''Called after restore - for complex initialization'''
        # Use this to rebuild non-serializable objects like backends
        # All other fields are already restored at this point
        pass
```

### 2. Mark controller fields for persistence

Use `Annotated[Type, PersistedMarker()]` on controller class fields.

```python
from typing import Annotated, Dict
from llm_rpc.persistence import (
    redis_persistent,
    PersistedMarker,
    PersistenceSettings,
)

# Configure at startup (usually done by CLI/server)
PersistenceSettings.configure(
    enabled=True,
    redis_url="redis://localhost:6379/0",
    namespace="llm_rpc",
    instance_id=None,
)


# Use without arguments - configuration is read from global settings
@redis_persistent
class TrainingController:
    config: AppConfig  # Not persisted

    # Persisted field - changes auto-sync to Redis
    training_runs: Annotated[Dict[str, TrainingRunRecord], PersistedMarker()]

    def __init__(self, config):
        self.config = config


# Or with explicit restore callback
@redis_persistent(restore_callback="_rebuild_backends")
class TrainingController:
    training_runs: Annotated[Dict[str, TrainingRunRecord], PersistedMarker()]

    def _rebuild_backends(self):
        '''Called when data is restored from Redis'''
        for record in self.training_runs.values():
            record.backend = self.backends.get(record.base_model)
```

### 3. Auto-sync behavior

ALL modifications to @persistable objects automatically sync to Redis:

```python
# Get a record - returns a proxy that auto-syncs
record = controller.training_runs["run-1"]

# Modify nested container - auto-syncs!
record.checkpoints["ckpt-1"] = CheckpointRecord(...)

# Modify scalar attribute - ALSO auto-syncs!
record.checkpoints["ckpt-1"].size_bytes = 2048  # This syncs too!

# The above modifications write directly to Redis keys
```

## Container Types

The module supports all Pottery container types:

| Type | Description |
|------|-------------|
| `PersistentDict` | Redis-backed dictionary |
| `PersistentList` | Redis-backed list |
| `PersistentSet` | Redis-backed set |
| `PersistentCounter` | Redis-backed counter (atomic increment/decrement) |
| `PersistentDeque` | Redis-backed double-ended queue |
| `PersistentNextId` | Distributed ID generator |
| `PersistentLock` | Distributed lock (Redlock algorithm) |
| `PersistentBloomFilter` | Probabilistic set membership |
| `PersistentHyperLogLog` | Cardinality estimation |
"""

from __future__ import annotations

from typing import Annotated, Any

# Import decorator (handles both Redis and in-memory modes)
from .decorator import redis_persistent

# Import configuration first (always available, no Redis dependency)
from .persistence_config import (
    REDIS_AVAILABLE,
    PersistenceConfig,
    PersistenceSettings,
    check_redis_available,
    get_instance_id,
    get_namespace,
    get_redis_url,
    is_persistence_enabled,
)

# Import proxy utilities (always available)
from .proxy import PersistentProxy, unwrap_proxy

# Import types (always available, no Redis dependency)
from .types import PersistedMarker, PersistenceExclude, persistable

# --- Stub classes for when Redis is not available ---


class _StubRedisConnection:
    """Stub RedisConnection when Redis is not available."""

    @classmethod
    def configure(cls, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    def get(cls) -> Any:
        raise RuntimeError(
            "Redis dependencies not installed. " "Install with: pip install llm-rpc[persistence]"
        )

    @classmethod
    def is_configured(cls) -> bool:
        return False

    @classmethod
    def close(cls) -> None:
        pass

    @classmethod
    def reset(cls) -> None:
        pass


class _StubPersistentDict(dict):  # type: ignore[type-arg]
    """In-memory dict when Redis is not available."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()


class _StubPersistentList(list):  # type: ignore[type-arg]
    """In-memory list when Redis is not available."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()


class _StubPersistentSet(set):  # type: ignore[type-arg]
    """In-memory set when Redis is not available."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()


class _StubPersistentCounter:
    """Stub counter when Redis is not available."""

    pass


class _StubPersistentDeque:
    """Stub deque when Redis is not available."""

    pass


class _StubPersistentNextId:
    """Stub NextId when Redis is not available."""

    pass


class _StubPersistentLock:
    """Stub Lock when Redis is not available."""

    pass


class _StubPersistentBloomFilter:
    """Stub BloomFilter when Redis is not available."""

    pass


class _StubPersistentHyperLogLog:
    """Stub HyperLogLog when Redis is not available."""

    pass


# --- Conditional exports ---


# Use a function to select the right implementation
def _get_redis_connection() -> type:
    if REDIS_AVAILABLE:
        from .connection import RedisConnection as _RedisConnection

        return _RedisConnection
    return _StubRedisConnection


def _get_persistent_dict() -> type:
    if REDIS_AVAILABLE:
        from .redis_containers import PersistentDict as _PersistentDict

        return _PersistentDict
    return _StubPersistentDict


def _get_persistent_list() -> type:
    if REDIS_AVAILABLE:
        from .redis_containers import PersistentList as _PersistentList

        return _PersistentList
    return _StubPersistentList


def _get_persistent_set() -> type:
    if REDIS_AVAILABLE:
        from .redis_containers import PersistentSet as _PersistentSet

        return _PersistentSet
    return _StubPersistentSet


def _get_persistent_counter() -> type:
    if REDIS_AVAILABLE:
        from .redis_containers import PersistentCounter as _PersistentCounter

        return _PersistentCounter
    return _StubPersistentCounter


def _get_persistent_deque() -> type:
    if REDIS_AVAILABLE:
        from .redis_containers import PersistentDeque as _PersistentDeque

        return _PersistentDeque
    return _StubPersistentDeque


def _get_persistent_next_id() -> type:
    if REDIS_AVAILABLE:
        from .redis_containers import PersistentNextId as _PersistentNextId

        return _PersistentNextId
    return _StubPersistentNextId


def _get_persistent_lock() -> type:
    if REDIS_AVAILABLE:
        from .redis_containers import PersistentLock as _PersistentLock

        return _PersistentLock
    return _StubPersistentLock


def _get_persistent_bloom_filter() -> type:
    if REDIS_AVAILABLE:
        from .redis_containers import PersistentBloomFilter as _PersistentBloomFilter

        return _PersistentBloomFilter
    return _StubPersistentBloomFilter


def _get_persistent_hyperloglog() -> type:
    if REDIS_AVAILABLE:
        from .redis_containers import PersistentHyperLogLog as _PersistentHyperLogLog

        return _PersistentHyperLogLog
    return _StubPersistentHyperLogLog


# Export the appropriate implementations
RedisConnection = _get_redis_connection()
PersistentDict = _get_persistent_dict()
PersistentList = _get_persistent_list()
PersistentSet = _get_persistent_set()
PersistentCounter = _get_persistent_counter()
PersistentDeque = _get_persistent_deque()
PersistentNextId = _get_persistent_next_id()
PersistentLock = _get_persistent_lock()
PersistentBloomFilter = _get_persistent_bloom_filter()
PersistentHyperLogLog = _get_persistent_hyperloglog()


__all__ = [
    # Decorator for controller classes
    "redis_persistent",
    # Type annotation marker for persisted fields
    "Annotated",  # Re-export for convenience
    "PersistedMarker",
    # Decorator and marker for model classes
    "persistable",
    "PersistenceExclude",
    # Configuration
    "PersistenceConfig",
    "PersistenceSettings",
    "is_persistence_enabled",
    "get_namespace",
    "get_instance_id",
    "get_redis_url",
    "check_redis_available",
    "REDIS_AVAILABLE",
    # Redis connection management
    "RedisConnection",
    # Container implementations
    "PersistentDict",
    "PersistentList",
    "PersistentSet",
    "PersistentCounter",
    "PersistentDeque",
    "PersistentNextId",
    "PersistentLock",
    "PersistentBloomFilter",
    "PersistentHyperLogLog",
    # Proxy utilities
    "PersistentProxy",
    "unwrap_proxy",
]
