"""Proxy objects for automatic persistence synchronization."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Generic, TypeVar

from .serializers import is_model_instance, is_model_type
from .types import get_excluded_fields, is_persistable

T = TypeVar("T")


class PersistentProxy(Generic[T]):
    """
    A proxy wrapper around a @persistable model instance.

    When attributes are modified on the proxy, the changes are
    automatically synchronized back to Redis.

    This enables:
        record = store["key"]
        record.field = new_value  # Automatically syncs to Redis!

    The proxy:
    - Intercepts __setattr__ to detect modifications
    - Calls the sync callback after any attribute change
    - Wraps nested @persistable objects recursively
    """

    __slots__ = ("_proxy_target", "_proxy_sync", "_proxy_key", "_proxy_container")

    def __init__(
        self,
        target: T,
        sync_callback: Callable[[], None],
        key: Any = None,
        container: Any = None,
    ):
        """
        Initialize the proxy.

        Args:
            target: The actual model instance being proxied
            sync_callback: Function to call when target needs to be synced
            key: The key in the parent container (for re-assignment)
            container: The parent container (for re-assignment)
        """
        object.__setattr__(self, "_proxy_target", target)
        object.__setattr__(self, "_proxy_sync", sync_callback)
        object.__setattr__(self, "_proxy_key", key)
        object.__setattr__(self, "_proxy_container", container)

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the proxied target."""
        target = object.__getattribute__(self, "_proxy_target")
        value = getattr(target, name)

        # If the value is a @persistable instance, wrap it in a proxy too
        if is_model_instance(value) and is_persistable(type(value)):
            sync = object.__getattribute__(self, "_proxy_sync")
            return PersistentProxy(value, sync)

        return value

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute and sync to Redis."""
        # Handle proxy internal attributes
        if name in ("_proxy_target", "_proxy_sync", "_proxy_key", "_proxy_container"):
            object.__setattr__(self, name, value)
            return

        target = object.__getattribute__(self, "_proxy_target")
        sync = object.__getattribute__(self, "_proxy_sync")

        # Set the attribute on the actual target
        setattr(target, name, value)

        # Trigger sync to Redis
        sync()

    def __delattr__(self, name: str) -> None:
        """Delete an attribute and sync to Redis."""
        target = object.__getattribute__(self, "_proxy_target")
        sync = object.__getattribute__(self, "_proxy_sync")

        delattr(target, name)
        sync()

    def __repr__(self) -> str:
        target = object.__getattribute__(self, "_proxy_target")
        return f"PersistentProxy({target!r})"

    def __str__(self) -> str:
        target = object.__getattribute__(self, "_proxy_target")
        return str(target)

    def __eq__(self, other: Any) -> bool:
        target = object.__getattribute__(self, "_proxy_target")
        if isinstance(other, PersistentProxy):
            other_target = object.__getattribute__(other, "_proxy_target")
            return target == other_target
        return target == other

    def __hash__(self) -> int:
        target = object.__getattribute__(self, "_proxy_target")
        return hash(target)

    def _get_target(self) -> T:
        """Get the underlying target object (for internal use)."""
        return object.__getattribute__(self, "_proxy_target")

    # Forward common dataclass/Pydantic methods
    def __iter__(self):
        target = object.__getattribute__(self, "_proxy_target")
        return iter(target)

    def __bool__(self) -> bool:
        """Return True if the proxy wraps a non-None target."""
        target = object.__getattribute__(self, "_proxy_target")
        return target is not None

    def __len__(self):
        target = object.__getattribute__(self, "_proxy_target")
        if hasattr(target, "__len__"):
            return len(target)
        # For objects without __len__, raise TypeError as expected
        raise TypeError(f"object of type '{type(target).__name__}' has no len()")

    def __contains__(self, item):
        target = object.__getattribute__(self, "_proxy_target")
        return item in target


def unwrap_proxy(obj: Any) -> Any:
    """
    Unwrap a PersistentProxy to get the underlying object.

    If the object is not a proxy, returns it unchanged.
    """
    if isinstance(obj, PersistentProxy):
        return object.__getattribute__(obj, "_proxy_target")
    return obj


def wrap_with_proxy(
    value: Any,
    sync_callback: Callable[[], None],
    key: Any = None,
    container: Any = None,
) -> Any:
    """
    Wrap a value with a PersistentProxy if appropriate.

    Only wraps @persistable model instances.
    """
    if is_model_instance(value) and is_persistable(type(value)):
        return PersistentProxy(value, sync_callback, key, container)
    return value
