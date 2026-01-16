"""Simple Redis persistence module for TuFT.

This module provides direct Redis-based persistence using redis-py.
Each data record is stored as a separate Redis key with JSON serialization.

Key Design:
- Top-level records: {namespace}::{type}::{id}
- Nested records: {namespace}::{type}::{parent_id}::{nested_type}::{nested_id}

Persistence Modes:
- disabled: No persistence, all data is in-memory only
- redis_url: Use external Redis server via URL
- redislite: Use lightweight embedded Redis (redislite) with local file storage
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

T = TypeVar("T")


class PersistenceMode(str, Enum):
    """Persistence mode options."""

    DISABLED = "disabled"  # No persistence
    REDIS_URL = "redis_url"  # Use external Redis server
    REDISLITE = "redislite"  # Use lightweight embedded Redis


def _default_redislite_path() -> Path:
    """Return default path for redislite database file."""
    return Path.home() / ".cache" / "tuft" / "redis.db"


@dataclass
class PersistenceConfig:
    """Configuration for Redis persistence.

    Attributes:
        mode: Persistence mode - disabled, redis_url, or redislite
        redis_url: Redis server URL (only used when mode=redis_url)
        redislite_path: Path to redislite database file (only used when mode=redislite)
        namespace: Key namespace prefix
    """

    mode: PersistenceMode = PersistenceMode.DISABLED
    redis_url: str = "redis://localhost:6379/0"
    redislite_path: Path | None = None
    namespace: str = "tuft"

    @property
    def enabled(self) -> bool:
        """Check if persistence is enabled."""
        return self.mode != PersistenceMode.DISABLED

    @classmethod
    def disabled(cls, namespace: str = "tuft") -> "PersistenceConfig":
        """Create a disabled persistence config."""
        return cls(mode=PersistenceMode.DISABLED, namespace=namespace)

    @classmethod
    def from_redis_url(cls, redis_url: str, namespace: str = "tuft") -> "PersistenceConfig":
        """Create a config using external Redis server."""
        return cls(mode=PersistenceMode.REDIS_URL, redis_url=redis_url, namespace=namespace)

    @classmethod
    def from_redislite(
        cls, path: Path | str | None = None, namespace: str = "tuft"
    ) -> "PersistenceConfig":
        """Create a config using lightweight embedded Redis (redislite).

        Args:
            path: Path to store the redislite database file.
                  If None, uses default path (~/.cache/tuft/redis.db)
            namespace: Key namespace prefix
        """
        if path is None:
            path = _default_redislite_path()
        elif isinstance(path, str):
            path = Path(path)
        return cls(mode=PersistenceMode.REDISLITE, redislite_path=path, namespace=namespace)


class RedisStore:
    """Global Redis connection and operation manager.

    Supports three backends:
    - External Redis server (via redis-py)
    - Embedded Redis (via redislite)
    - No persistence (disabled mode)
    """

    _instance: "RedisStore | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._redis: Any = None
        self._redislite_instance: Any = None  # Keep reference to redislite.Redis
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
                pass
            self._redis = None

        if self._redislite_instance is not None:
            try:
                self._redislite_instance.close()
            except Exception:
                pass
            self._redislite_instance = None

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
                    elif self._config.mode == PersistenceMode.REDISLITE:
                        self._redis = self._create_redislite_client()

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
            return None

    def _create_redislite_client(self) -> Any:
        """Create a redislite client for embedded Redis."""
        try:
            import redislite

            if self._config is None or self._config.redislite_path is None:
                db_path = _default_redislite_path()
            else:
                db_path = self._config.redislite_path

            # Ensure parent directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create redislite instance
            self._redislite_instance = redislite.Redis(str(db_path), decode_responses=True)
            return self._redislite_instance
        except ImportError:
            return None

    @property
    def is_enabled(self) -> bool:
        return self._config is not None and self._config.enabled

    @property
    def namespace(self) -> str:
        return self._config.namespace if self._config else "tuft"

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

    def set(self, key: str, value: str) -> bool:
        redis = self._get_redis()
        if redis is None:
            return False
        try:
            redis.set(key, value)
            return True
        except Exception:
            return False

    def get(self, key: str) -> str | None:
        redis = self._get_redis()
        if redis is None:
            return None
        try:
            return redis.get(key)
        except Exception:
            return None

    def delete(self, key: str) -> bool:
        redis = self._get_redis()
        if redis is None:
            return False
        try:
            redis.delete(key)
            return True
        except Exception:
            return False

    def keys(self, pattern: str) -> list[str]:
        redis = self._get_redis()
        if redis is None:
            return []
        try:
            return list(redis.keys(pattern))
        except Exception:
            return []

    def delete_pattern(self, pattern: str) -> int:
        redis = self._get_redis()
        if redis is None:
            return 0
        try:
            keys = redis.keys(pattern)
            if keys:
                return redis.delete(*keys)
            return 0
        except Exception:
            return 0

    def exists(self, key: str) -> bool:
        redis = self._get_redis()
        if redis is None:
            return False
        try:
            return redis.exists(key) > 0
        except Exception:
            return False


# Fields that should not be serialized (runtime objects)
NON_SERIALIZABLE_FIELDS = {
    "_execution_lock",  # asyncio.Lock in TrainingRunRecord
    "event",  # asyncio.Event in FutureRecord
    "backend",  # BaseTrainingBackend reference
}


def _is_nested_dataclass_dict(field_type: Any) -> bool:
    """Check if field type is Dict[str, SomeDataclass]."""
    origin = get_origin(field_type)
    if origin is dict:
        args = get_args(field_type)
        if len(args) == 2:
            value_type = args[1]
            if isinstance(value_type, type) and is_dataclass(value_type):
                return True
    return False


def _is_nested_dataclass_list(field_type: Any) -> bool:
    """Check if field type is list[SomeDataclass]."""
    origin = get_origin(field_type)
    if origin is list:
        args = get_args(field_type)
        if len(args) == 1:
            item_type = args[0]
            if isinstance(item_type, type) and is_dataclass(item_type):
                return True
    return False


def serialize_value(value: Any) -> Any:
    """Serialize a single value to JSON-compatible format."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return {"__type__": "datetime", "value": value.isoformat()}
    if isinstance(value, Path):
        return {"__type__": "Path", "value": str(value)}
    if isinstance(value, asyncio.Lock):
        return None
    if isinstance(value, asyncio.Event):
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return serialize_dataclass(value)
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {"__type__": "set", "value": [serialize_value(item) for item in value]}
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")  # type: ignore[attr-defined]

    return str(value)


def serialize_dataclass(obj: Any, skip_nested_refs: bool = False) -> dict[str, Any]:
    """Serialize a dataclass to a dictionary.

    Args:
        obj: The dataclass instance to serialize.
        skip_nested_refs: If True, skip fields that contain nested dataclass
            dicts/lists (they should be stored separately).
    """
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"Expected dataclass instance, got {type(obj)}")

    result: dict[str, Any] = {}
    result["__class__"] = type(obj).__name__
    result["__module__"] = type(obj).__module__

    for field in fields(obj):
        name = field.name
        if name in NON_SERIALIZABLE_FIELDS:
            continue

        if skip_nested_refs:
            # Auto-detect nested dataclass fields
            if _is_nested_dataclass_dict(field.type):
                result[f"__{name}_keys__"] = list(getattr(obj, name, {}).keys())
                continue
            if _is_nested_dataclass_list(field.type):
                result[f"__{name}_count__"] = len(getattr(obj, name, []))
                continue

        value = getattr(obj, name)
        result[name] = serialize_value(value)

    return result


def deserialize_value(value: Any, expected_type: type | None = None) -> Any:
    """Deserialize a JSON value back to Python object."""
    if value is None:
        return None
    if isinstance(value, dict):
        if "__type__" in value:
            type_name = value["__type__"]
            if type_name == "datetime":
                return datetime.fromisoformat(value["value"])
            if type_name == "Path":
                return Path(value["value"])
            if type_name == "set":
                return set(deserialize_value(item) for item in value["value"])
        if "__class__" in value and "__module__" in value:
            return deserialize_dataclass(value)
        return {k: deserialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [deserialize_value(item) for item in value]
    return value


def deserialize_dataclass(data: dict[str, Any], target_class: type[T] | None = None) -> T:
    """Deserialize a dictionary to a dataclass instance."""
    if target_class is None:
        class_name = data.get("__class__")
        module_name = data.get("__module__")
        if class_name and module_name:
            import importlib

            try:
                module = importlib.import_module(module_name)
                target_class = getattr(module, class_name, None)
            except (ImportError, AttributeError):
                pass

    if target_class is None or not is_dataclass(target_class):
        return data  # type: ignore

    init_kwargs = {}
    field_info = {f.name: f for f in fields(target_class)}

    for name, field in field_info.items():
        if not field.init:
            continue
        if name in NON_SERIALIZABLE_FIELDS:
            if name == "_execution_lock":
                init_kwargs[name] = asyncio.Lock()
            elif name == "event":
                init_kwargs[name] = asyncio.Event()
            continue
        if name in data:
            init_kwargs[name] = deserialize_value(data[name])
        elif f"__{name}_keys__" in data:
            init_kwargs[name] = {}
        elif f"__{name}_count__" in data:
            init_kwargs[name] = []

    try:
        return target_class(**init_kwargs)
    except TypeError:
        filtered = {k: v for k, v in init_kwargs.items() if k in field_info and field_info[k].init}
        return target_class(**filtered)


def save_record(key: str, record: Any, skip_nested_refs: bool = False) -> bool:
    """Save a dataclass record to Redis."""
    store = RedisStore.get_instance()
    if not store.is_enabled:
        return False
    try:
        data = serialize_dataclass(record, skip_nested_refs=skip_nested_refs)
        json_str = json.dumps(data, ensure_ascii=False)
        return store.set(key, json_str)
    except Exception:
        return False


def load_record(key: str, target_class: type[T] | None = None) -> T | None:
    """Load a dataclass record from Redis."""
    store = RedisStore.get_instance()
    if not store.is_enabled:
        return None
    try:
        json_str = store.get(key)
        if json_str is None:
            return None
        data = json.loads(json_str)
        return deserialize_dataclass(data, target_class)
    except Exception:
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
