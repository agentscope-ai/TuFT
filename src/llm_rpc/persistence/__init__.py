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
    RedisConnection,
)

# Configure Redis at startup
RedisConnection.configure("redis://localhost:6379/0")


# Can use without arguments - defaults apply
@redis_persistent
class TrainingController:
    config: AppConfig  # Not persisted

    # Persisted field - changes auto-sync to Redis
    training_runs: Annotated[Dict[str, TrainingRunRecord], PersistedMarker()]

    def __init__(self, config):
        self.config = config


# Or with explicit configuration
@redis_persistent(
    namespace="llm_rpc",
    instance_id=None,  # Single instance
    restore_callback="_rebuild_backends",
)
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

from typing import Annotated

from .connection import RedisConnection
from .decorator import redis_persistent
from .proxy import PersistentProxy, unwrap_proxy
from .redis_containers import (
    PersistentBloomFilter,
    PersistentCounter,
    PersistentDeque,
    PersistentDict,
    PersistentHyperLogLog,
    PersistentList,
    PersistentLock,
    PersistentNextId,
    PersistentSet,
)
from .types import PersistedMarker, PersistenceExclude, persistable

__all__ = [
    # Decorator for controller classes
    "redis_persistent",
    # Type annotation marker for persisted fields
    "Annotated",  # Re-export for convenience
    "PersistedMarker",
    # Decorator and marker for model classes
    "persistable",
    "PersistenceExclude",
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
