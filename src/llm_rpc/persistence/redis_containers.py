"""Redis-backed container implementations using Pottery with auto-sync support."""

from __future__ import annotations

import json
from collections.abc import MutableMapping, MutableSequence, MutableSet
from typing import Any, Callable, Generic, Iterator, TypeVar, overload

from pottery import (
    BloomFilter,
    HyperLogLog,
    NextID,
    RedisCounter,
    RedisDeque,
    RedisDict,
    RedisList,
    RedisSet,
    Redlock,
)

from .connection import RedisConnection
from .proxy import PersistentProxy, unwrap_proxy
from .serializers import ModelSerializer, is_model_instance, is_model_type
from .types import is_persistable
from .utils import escape_key, unescape_key

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


class PersistentDict(MutableMapping[K, V], Generic[K, V]):
    """
    A Redis-backed dictionary with automatic nested persistence.

    Key Features:
    - Each nested object gets its own Redis key for efficiency
    - Modifications to nested @persistable objects automatically sync to Redis
    - Returns proxy objects for @persistable values to enable auto-sync

    Redis Key Structure:
        {prefix}                           -> main dict metadata
        {prefix}::{escaped_key}            -> individual item
        {prefix}::{escaped_key}::{field}   -> nested container
    """

    def __init__(
        self,
        key_prefix: str,
        value_type: type[V] | None = None,
    ) -> None:
        self._redis = RedisConnection.get()
        self._key_prefix = key_prefix
        self._value_type = value_type

        # Main storage using pottery.RedisDict
        self._store = RedisDict(redis=self._redis, key=key_prefix)

        # Analyze nested container fields
        self._nested_fields = self._analyze_nested_fields()

        # Cache for loaded objects (to ensure same object reference)
        self._cache: dict[str, V] = {}

    def _analyze_nested_fields(self) -> dict[str, tuple[str, type | None]]:
        """Analyze value_type to find nested container fields."""
        if not self._value_type or not is_model_type(self._value_type):
            return {}

        return ModelSerializer.find_nested_container_fields(self._value_type)

    def _create_sync_callback(self, key: K, target_ref: list[Any]) -> Callable[[], None]:
        """
        Create a callback that re-saves the item when called.

        Args:
            key: The key for the item
            target_ref: A single-element list holding reference to the target object.
                        Using a list allows the callback to always use the current object state.
        """
        container = self
        escaped = escape_key(str(key))

        def sync():
            # Get the current object from the reference
            obj = target_ref[0] if target_ref else None
            if obj is None:
                return

            # Re-serialize and save to Redis (but keep the cache entry)
            if is_model_instance(obj):
                nested_field_names = set(container._nested_fields.keys())
                serialized = ModelSerializer.serialize(
                    obj,
                    nested_container_fields=nested_field_names,
                )
                container._store[escaped] = json.dumps(serialized)
            else:
                container._store[escaped] = json.dumps(obj)

            # Note: We don't clear the cache here because the object reference
            # is still valid and we want to preserve it for subsequent accesses

        return sync

    def __setitem__(self, key: K, value: V) -> None:
        """Store a value, automatically handling nested containers."""
        escaped = escape_key(str(key))

        # Unwrap proxy if necessary
        actual_value = unwrap_proxy(value)

        if is_model_instance(actual_value):
            nested_field_names = set(self._nested_fields.keys())

            # Serialize main object (excluding nested containers)
            serialized = ModelSerializer.serialize(
                actual_value,
                nested_container_fields=nested_field_names,
            )
            self._store[escaped] = json.dumps(serialized)

            # Store nested containers in separate Redis keys
            for field_name, (container_type, element_type) in self._nested_fields.items():
                nested_value = getattr(actual_value, field_name, None)

                # Unwrap nested value if it's a proxy or PersistentDict
                if isinstance(nested_value, PersistentDict):
                    nested_value = dict(nested_value.items())
                elif isinstance(nested_value, PersistentList):
                    nested_value = list(nested_value)
                elif isinstance(nested_value, PersistentSet):
                    nested_value = set(nested_value)

                if nested_value is None:
                    continue

                nested_prefix = f"{self._key_prefix}::{escaped}::{field_name}"

                if container_type == "dict" and isinstance(nested_value, dict):
                    nested_store = PersistentDict(
                        key_prefix=nested_prefix,
                        value_type=element_type,
                    )
                    nested_store.clear()
                    for nk, nv in nested_value.items():
                        nested_store[nk] = unwrap_proxy(nv)

                elif container_type == "list" and isinstance(nested_value, list):
                    nested_store = PersistentList(
                        key_prefix=nested_prefix,
                        element_type=element_type,
                    )
                    nested_store.clear()
                    # Only extend if there are items to add (rpush requires at least one value)
                    if nested_value:
                        nested_store.extend([unwrap_proxy(v) for v in nested_value])

                elif container_type == "set" and isinstance(nested_value, (set, frozenset)):
                    nested_store = PersistentSet(
                        key_prefix=nested_prefix,
                        element_type=element_type,
                    )
                    nested_store.clear()
                    nested_store.update({unwrap_proxy(v) for v in nested_value})
        else:
            self._store[escaped] = json.dumps(actual_value)

        # Invalidate cache entry (will be reloaded on next access)
        if escaped in self._cache:
            del self._cache[escaped]

    def __getitem__(self, key: K) -> V:
        """
        Retrieve a value, returning proxied objects for auto-sync.

        For @persistable model values:
        - Returns a PersistentProxy that auto-syncs attribute changes
        - Nested containers are PersistentDict/List/Set instances
        - Returns cached proxy if already loaded (same reference)

        Modifying the returned object's attributes will automatically
        persist the changes to Redis.
        """
        escaped = escape_key(str(key))

        if escaped not in self._store:
            raise KeyError(key)

        # Return cached proxy if available
        if escaped in self._cache:
            return self._cache[escaped]

        raw = self._store[escaped]
        if isinstance(raw, str):
            serialized = json.loads(raw)
        else:
            serialized = raw

        if self._value_type and is_model_type(self._value_type) and isinstance(serialized, dict):
            obj = ModelSerializer.deserialize(serialized, self._value_type)

            # Replace nested containers with PersistentDict/List/Set
            for field_name, (container_type, element_type) in self._nested_fields.items():
                nested_prefix = f"{self._key_prefix}::{escaped}::{field_name}"

                if container_type == "dict":
                    nested_store = PersistentDict(
                        key_prefix=nested_prefix,
                        value_type=element_type,
                    )
                    setattr(obj, field_name, nested_store)

                elif container_type == "list":
                    nested_store = PersistentList(
                        key_prefix=nested_prefix,
                        element_type=element_type,
                    )
                    setattr(obj, field_name, nested_store)

                elif container_type == "set":
                    nested_store = PersistentSet(
                        key_prefix=nested_prefix,
                        element_type=element_type,
                    )
                    setattr(obj, field_name, nested_store)

            # Wrap in proxy if @persistable for auto-sync of scalar attributes
            if is_persistable(self._value_type):
                # Create a reference holder so the callback always sees the current object
                target_ref: list[Any] = [obj]
                sync_callback = self._create_sync_callback(key, target_ref)
                proxy = PersistentProxy(obj, sync_callback, key, self)
                # Cache the proxy for subsequent accesses
                self._cache[escaped] = proxy  # type: ignore
                return proxy  # type: ignore

            return obj

        return serialized  # type: ignore[return-value]

    def __delitem__(self, key: K) -> None:
        """Delete a value and all its nested data."""
        escaped = escape_key(str(key))

        if escaped not in self._store:
            raise KeyError(key)

        # Delete nested data first
        for field_name in self._nested_fields.keys():
            nested_prefix = f"{self._key_prefix}::{escaped}::{field_name}"
            self._delete_by_prefix(nested_prefix)

        del self._store[escaped]

        # Clear from cache
        if escaped in self._cache:
            del self._cache[escaped]

    def _delete_by_prefix(self, prefix: str) -> None:
        """Delete all Redis keys matching a prefix pattern."""
        cursor: int = 0
        while True:
            result = self._redis.scan(cursor, match=f"{prefix}*")
            cursor, keys = int(result[0]), result[1]  # type: ignore[index]
            if keys:
                self._redis.delete(*keys)
            if cursor == 0:
                break

    def clear(self) -> None:
        """Clear all data including nested containers."""
        for escaped in list(self._store.keys()):
            for field_name in self._nested_fields.keys():
                nested_prefix = f"{self._key_prefix}::{escaped}::{field_name}"
                self._delete_by_prefix(nested_prefix)
        self._store.clear()
        self._cache.clear()

    def __iter__(self) -> Iterator[K]:
        for escaped_key in self._store.keys():
            yield unescape_key(escaped_key)  # type: ignore

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return escape_key(str(key)) in self._store

    def eager_load_all(self, collected_objects: list[Any] | None = None) -> dict[K, V]:
        """
        Eagerly load all values from Redis into memory.

        This is useful during recovery to ensure all data is fully loaded
        before calling __post_deserialize__ hooks.

        Args:
            collected_objects: If provided, all deserialized @persistable objects
                are appended to this list (for batch hook calling later).

        Returns:
            Dictionary of all loaded key-value pairs.
        """
        result: dict[K, V] = {}

        for escaped_key in self._store.keys():
            key = unescape_key(escaped_key)  # type: ignore

            raw = self._store[escaped_key]
            if isinstance(raw, str):
                serialized = json.loads(raw)
            else:
                serialized = raw

            if (
                self._value_type
                and is_model_type(self._value_type)
                and isinstance(serialized, dict)
            ):
                # Deserialize without calling hook (defer to batch call)
                obj = ModelSerializer.deserialize(
                    serialized,
                    self._value_type,
                    call_post_hook=False,
                    collected_objects=collected_objects,
                )

                # Replace nested containers with PersistentDict/List/Set
                for field_name, (container_type, element_type) in self._nested_fields.items():
                    nested_prefix = f"{self._key_prefix}::{escaped_key}::{field_name}"

                    if container_type == "dict":
                        nested_store: PersistentDict[Any, Any] = PersistentDict(
                            key_prefix=nested_prefix,
                            value_type=element_type,
                        )
                        # Recursively eager load nested dicts
                        if (
                            collected_objects is not None
                            and element_type
                            and is_model_type(element_type)
                        ):
                            nested_store.eager_load_all(collected_objects)
                        setattr(obj, field_name, nested_store)

                    elif container_type == "list":
                        nested_store_list: PersistentList[Any] = PersistentList(
                            key_prefix=nested_prefix,
                            element_type=element_type,
                        )
                        # Eagerly load list items
                        if (
                            collected_objects is not None
                            and element_type
                            and is_model_type(element_type)
                        ):
                            nested_store_list.eager_load_all(collected_objects)
                        setattr(obj, field_name, nested_store_list)

                    elif container_type == "set":
                        nested_store_set: PersistentSet[Any] = PersistentSet(
                            key_prefix=nested_prefix,
                            element_type=element_type,
                        )
                        setattr(obj, field_name, nested_store_set)

                # Wrap in proxy if @persistable for auto-sync
                if is_persistable(self._value_type):
                    target_ref: list[Any] = [obj]
                    sync_callback = self._create_sync_callback(key, target_ref)  # type: ignore[arg-type]
                    proxy = PersistentProxy(obj, sync_callback, key, self)
                    # Cache the proxy so __getitem__ returns same reference
                    self._cache[escaped_key] = proxy  # type: ignore
                    result[key] = proxy  # type: ignore[assignment]
                else:
                    result[key] = obj  # type: ignore[assignment]
            else:
                result[key] = serialized  # type: ignore[assignment]

        return result

    def keys(self) -> list[K]:  # type: ignore[override]
        return list(self)

    def values(self) -> list[V]:  # type: ignore[override]
        return [self[k] for k in self]

    def items(self) -> list[tuple[K, V]]:  # type: ignore[override]
        return [(k, self[k]) for k in self]

    def get(self, key: K, default: V | None = None) -> V | None:  # type: ignore[override]
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key: K, *args: Any) -> V:  # type: ignore[override]
        try:
            value = self[key]
            del self[key]
            return value
        except KeyError:
            if args:
                return args[0]
            raise

    def update(self, other: dict[K, V] | None = None, **kwargs: V) -> None:  # type: ignore[override]
        if other:
            for k, v in other.items():
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v  # type: ignore

    def setdefault(self, key: K, default: V | None = None) -> V | None:  # type: ignore[override]
        if key not in self:
            self[key] = default  # type: ignore
        return self.get(key)

    @property
    def key_prefix(self) -> str:
        return self._key_prefix

    def __repr__(self) -> str:
        return f"PersistentDict(key_prefix={self._key_prefix!r}, len={len(self)})"


class PersistentList(MutableSequence[T], Generic[T]):
    """
    A Redis-backed list with automatic synchronization.

    All modifications are immediately persisted to Redis.
    For @persistable elements, returns proxied objects that auto-sync.
    """

    def __init__(
        self,
        key_prefix: str,
        element_type: type[T] | None = None,
    ) -> None:
        self._redis = RedisConnection.get()
        self._key_prefix = key_prefix
        self._element_type = element_type
        self._store = RedisList(redis=self._redis, key=key_prefix)

    def _serialize_element(self, value: T) -> str:
        actual_value = unwrap_proxy(value)
        if is_model_instance(actual_value):
            serialized = ModelSerializer.serialize(actual_value)
            return json.dumps(serialized)
        return json.dumps(actual_value)

    def _deserialize_element(self, raw: Any, index: int) -> T:
        if isinstance(raw, str):
            data = json.loads(raw)
        else:
            data = raw

        if self._element_type and is_model_type(self._element_type) and isinstance(data, dict):
            obj = ModelSerializer.deserialize(data, self._element_type)

            # Wrap in proxy if @persistable
            if is_persistable(self._element_type):

                def sync():
                    self[index] = obj

                return PersistentProxy(obj, sync, index, self)  # type: ignore

            return obj
        return data  # type: ignore[return-value]

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index: int | slice) -> T | list[T]:
        if isinstance(index, slice):
            items = self._store[index]
            return [self._deserialize_element(item, i) for i, item in enumerate(items)]
        return self._deserialize_element(self._store[index], index)

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: list[T]) -> None: ...

    def __setitem__(self, index: int | slice, value: T | list[T]) -> None:  # type: ignore[override]
        if isinstance(index, slice):
            if not isinstance(value, list):
                raise TypeError("Can only assign a list to a slice")
            self._store[index] = [self._serialize_element(v) for v in value]  # type: ignore
        else:
            self._store[index] = self._serialize_element(value)  # type: ignore

    def __delitem__(self, index: int | slice) -> None:
        self._store.__delitem__(index)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self._store)

    def insert(self, index: int, value: T) -> None:
        self._store.insert(index, self._serialize_element(value))

    def append(self, value: T) -> None:
        self._store.append(self._serialize_element(value))

    def extend(self, values: list[T]) -> None:  # type: ignore[override]
        self._store.extend([self._serialize_element(v) for v in values])

    def clear(self) -> None:
        self._store.clear()

    def __iter__(self) -> Iterator[T]:
        for i, item in enumerate(self._store):
            yield self._deserialize_element(item, i)

    def __contains__(self, value: object) -> bool:
        try:
            serialized = self._serialize_element(value)  # type: ignore
            return serialized in self._store
        except (TypeError, ValueError):
            return False

    @property
    def key_prefix(self) -> str:
        return self._key_prefix

    def eager_load_all(self, collected_objects: list[Any] | None = None) -> list[T]:
        """
        Eagerly load all elements from Redis into memory.

        Args:
            collected_objects: If provided, all deserialized @persistable objects
                are appended to this list (for batch hook calling later).

        Returns:
            List of all loaded elements.
        """
        result: list[T] = []

        for i, raw in enumerate(self._store):
            if isinstance(raw, str):
                data = json.loads(raw)
            else:
                data = raw

            if self._element_type and is_model_type(self._element_type) and isinstance(data, dict):
                obj = ModelSerializer.deserialize(
                    data,
                    self._element_type,
                    call_post_hook=False,
                    collected_objects=collected_objects,
                )

                if is_persistable(self._element_type):

                    def sync(idx=i, value=obj):
                        self[idx] = value

                    result.append(PersistentProxy(obj, sync, i, self))  # type: ignore
                else:
                    result.append(obj)
            else:
                result.append(data)  # type: ignore[arg-type]

        return result

    def __repr__(self) -> str:
        return f"PersistentList(key_prefix={self._key_prefix!r}, len={len(self)})"


class PersistentSet(MutableSet[T], Generic[T]):
    """
    A Redis-backed set with automatic synchronization.

    All modifications are immediately persisted to Redis.
    """

    def __init__(
        self,
        key_prefix: str,
        element_type: type[T] | None = None,
    ) -> None:
        self._redis = RedisConnection.get()
        self._key_prefix = key_prefix
        self._element_type = element_type
        self._store = RedisSet(redis=self._redis, key=key_prefix)

    def _serialize_element(self, value: T) -> str:
        actual_value = unwrap_proxy(value)
        if is_model_instance(actual_value):
            serialized = ModelSerializer.serialize(actual_value)
            return json.dumps(serialized, sort_keys=True)
        return json.dumps(actual_value, sort_keys=True)

    def _deserialize_element(self, raw: Any) -> T:
        if isinstance(raw, str):
            data = json.loads(raw)
        else:
            data = raw

        if self._element_type and is_model_type(self._element_type) and isinstance(data, dict):
            return ModelSerializer.deserialize(data, self._element_type)
        return data  # type: ignore[return-value]

    def __contains__(self, value: object) -> bool:
        try:
            serialized = self._serialize_element(value)  # type: ignore
            return serialized in self._store
        except (TypeError, ValueError):
            return False

    def __iter__(self) -> Iterator[T]:
        for item in self._store:
            yield self._deserialize_element(item)

    def __len__(self) -> int:
        return len(self._store)

    def add(self, value: T) -> None:
        self._store.add(self._serialize_element(value))

    def discard(self, value: T) -> None:
        try:
            serialized = self._serialize_element(value)
            self._store.discard(serialized)
        except (TypeError, ValueError):
            pass

    def remove(self, value: T) -> None:
        serialized = self._serialize_element(value)
        self._store.remove(serialized)

    def update(self, *others: set[T]) -> None:
        for other in others:
            for value in other:
                self.add(value)

    def clear(self) -> None:
        self._store.clear()

    @property
    def key_prefix(self) -> str:
        return self._key_prefix

    def __repr__(self) -> str:
        return f"PersistentSet(key_prefix={self._key_prefix!r}, len={len(self)})"


class PersistentCounter(Generic[T]):
    """
    A Redis-backed counter (like collections.Counter).

    Uses pottery.RedisCounter for atomic increment/decrement operations.
    """

    def __init__(self, key_prefix: str) -> None:
        self._redis = RedisConnection.get()
        self._key_prefix = key_prefix
        self._store = RedisCounter(redis=self._redis, key=key_prefix)

    def __getitem__(self, key: T) -> int:
        return self._store[str(key)]

    def __setitem__(self, key: T, value: int) -> None:
        self._store[str(key)] = value

    def __delitem__(self, key: T) -> None:
        del self._store[str(key)]

    def increment(self, key: T, amount: int = 1) -> int:
        """Atomically increment a key and return the new value."""
        str_key = str(key)
        self._store[str_key] += amount
        return self._store[str_key]

    def decrement(self, key: T, amount: int = 1) -> int:
        """Atomically decrement a key and return the new value."""
        return self.increment(key, -amount)

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[T]:
        for key in self._store.keys():
            yield key  # type: ignore

    def clear(self) -> None:
        self._store.clear()

    @property
    def key_prefix(self) -> str:
        return self._key_prefix

    def __repr__(self) -> str:
        return f"PersistentCounter(key_prefix={self._key_prefix!r})"


class PersistentDeque(Generic[T]):
    """
    A Redis-backed deque (double-ended queue).

    Uses pottery.RedisDeque for efficient append/pop operations on both ends.
    """

    def __init__(
        self,
        key_prefix: str,
        element_type: type[T] | None = None,
        maxlen: int | None = None,
    ) -> None:
        self._redis = RedisConnection.get()
        self._key_prefix = key_prefix
        self._element_type = element_type
        self._store = RedisDeque(redis=self._redis, key=key_prefix, maxlen=maxlen)

    def _serialize_element(self, value: T) -> str:
        actual_value = unwrap_proxy(value)
        if is_model_instance(actual_value):
            serialized = ModelSerializer.serialize(actual_value)
            return json.dumps(serialized)
        return json.dumps(actual_value)

    def _deserialize_element(self, raw: Any) -> T:
        if isinstance(raw, str):
            data = json.loads(raw)
        else:
            data = raw

        if self._element_type and is_model_type(self._element_type) and isinstance(data, dict):
            return ModelSerializer.deserialize(data, self._element_type)
        return data  # type: ignore[return-value]

    def append(self, value: T) -> None:
        """Add to the right side."""
        self._store.append(self._serialize_element(value))

    def appendleft(self, value: T) -> None:
        """Add to the left side."""
        self._store.appendleft(self._serialize_element(value))

    def pop(self) -> T:
        """Remove and return from the right side."""
        raw = self._store.pop()
        return self._deserialize_element(raw)

    def popleft(self) -> T:
        """Remove and return from the left side."""
        raw = self._store.popleft()
        return self._deserialize_element(raw)

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[T]:
        for item in self._store:
            yield self._deserialize_element(item)

    def clear(self) -> None:
        self._store.clear()

    @property
    def key_prefix(self) -> str:
        return self._key_prefix

    def __repr__(self) -> str:
        return f"PersistentDeque(key_prefix={self._key_prefix!r}, len={len(self)})"


class PersistentNextId:
    """
    A Redis-backed distributed ID generator.

    Uses pottery.NextID for generating unique, sequential IDs across
    multiple processes/servers.
    """

    def __init__(self, key_prefix: str) -> None:
        """
        Initialize the ID generator.

        Args:
            key_prefix: Redis key prefix for the ID generator
        """
        self._redis = RedisConnection.get()
        self._key_prefix = key_prefix
        self._store = NextID(key=key_prefix, masters={self._redis})

    def __next__(self) -> int:
        """Get the next unique ID."""
        return next(self._store)

    def __iter__(self):
        return self

    @property
    def key_prefix(self) -> str:
        return self._key_prefix

    def __repr__(self) -> str:
        return f"PersistentNextId(key_prefix={self._key_prefix!r})"


class PersistentLock:
    """
    A Redis-backed distributed lock (Redlock algorithm).

    Uses pottery.Redlock for distributed locking across multiple
    processes/servers.

    Usage:
        lock = PersistentLock("my_resource")
        with lock:
            # Critical section
            ...
    """

    def __init__(
        self,
        key_prefix: str,
        auto_release_time: float = 10.0,
    ) -> None:
        """
        Initialize the distributed lock.

        Args:
            key_prefix: Redis key prefix for the lock
            auto_release_time: Auto-release time in seconds (default 10)
        """
        self._redis = RedisConnection.get()
        self._key_prefix = key_prefix
        self._store = Redlock(
            key=key_prefix,
            auto_release_time=auto_release_time,
            masters={self._redis},
        )

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """
        Acquire the lock.

        Args:
            blocking: Whether to block waiting for the lock
            timeout: Maximum time to wait (-1 for infinite)

        Returns:
            True if lock acquired, False otherwise
        """
        return self._store.acquire(blocking=blocking, timeout=timeout)

    def release(self) -> None:
        """Release the lock."""
        self._store.release()

    def locked(self) -> bool:
        """Check if the lock is currently held."""
        return bool(self._store.locked())

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    @property
    def key_prefix(self) -> str:
        return self._key_prefix

    def __repr__(self) -> str:
        return f"PersistentLock(key_prefix={self._key_prefix!r})"


class PersistentBloomFilter:
    """
    A Redis-backed Bloom filter for probabilistic set membership.

    Uses pottery.BloomFilter for space-efficient probabilistic data structure.
    """

    def __init__(
        self,
        key_prefix: str,
        num_elements: int = 1000,
        false_positives: float = 0.01,
    ) -> None:
        """
        Initialize the Bloom filter.

        Args:
            key_prefix: Redis key prefix
            num_elements: Expected number of elements
            false_positives: Acceptable false positive rate
        """
        self._redis = RedisConnection.get()
        self._key_prefix = key_prefix
        self._store = BloomFilter(
            num_elements=num_elements,
            false_positives=false_positives,
            redis=self._redis,
            key=key_prefix,
        )

    def add(self, element: str) -> None:
        """Add an element to the filter."""
        self._store.add(element)

    def __contains__(self, element: str) -> bool:
        """Check if an element might be in the filter."""
        return element in self._store

    def __len__(self) -> int:
        """Return the number of elements added."""
        return len(self._store)

    @property
    def key_prefix(self) -> str:
        return self._key_prefix

    def __repr__(self) -> str:
        return f"PersistentBloomFilter(key_prefix={self._key_prefix!r})"


class PersistentHyperLogLog:
    """
    A Redis-backed HyperLogLog for cardinality estimation.

    Uses pottery.HyperLogLog for approximate counting of unique elements.
    """

    def __init__(self, key_prefix: str) -> None:
        """
        Initialize the HyperLogLog.

        Args:
            key_prefix: Redis key prefix
        """
        self._redis = RedisConnection.get()
        self._key_prefix = key_prefix
        self._store = HyperLogLog(redis=self._redis, key=key_prefix)

    def add(self, *elements: str) -> None:
        """Add elements to the HyperLogLog."""
        self._store.add(*elements)

    def __len__(self) -> int:
        """Return the estimated cardinality."""
        return len(self._store)

    def update(self, *others: "PersistentHyperLogLog") -> None:
        """Merge other HyperLogLogs into this one."""
        for other in others:
            self._store.update(other._store)

    @property
    def key_prefix(self) -> str:
        return self._key_prefix

    def __repr__(self) -> str:
        return f"PersistentHyperLogLog(key_prefix={self._key_prefix!r})"
