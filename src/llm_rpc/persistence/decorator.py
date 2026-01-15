"""Redis persistence decorator for automatic state persistence."""

from __future__ import annotations

from functools import wraps
from typing import Annotated, Any, Callable, TypeVar, get_args, get_origin, get_type_hints, overload

from .persistence_config import REDIS_AVAILABLE, PersistenceSettings
from .types import PersistedMarker, call_post_deserialize

T = TypeVar("T")


@overload
def redis_persistent(cls: type[T]) -> type[T]: ...


@overload
def redis_persistent(
    *,
    restore_callback: str | None = None,
) -> Callable[[type[T]], type[T]]: ...


def redis_persistent(
    cls: type[T] | None = None,
    *,
    restore_callback: str | None = None,
) -> type[T] | Callable[[type[T]], type[T]]:
    """
    Decorator that enables Redis persistence for marked class attributes.

    Can be used with or without arguments:
        @redis_persistent
        class MyController: ...

        @redis_persistent()
        class MyController: ...

        @redis_persistent(restore_callback="_on_restore")
        class MyController: ...

    This decorator scans the class for attributes marked with
    `Annotated[Type, PersistedMarker()]` and replaces them with
    `PersistentDict` instances that automatically sync to Redis.

    When persistence is disabled (via PersistenceSettings), uses regular
    Python dicts instead, providing the same interface without Redis.

    Configuration is read from global PersistenceSettings:
    - namespace: from PersistenceSettings.get_namespace()
    - instance_id: from PersistenceSettings.get_effective_instance_id()
    - enabled: from PersistenceSettings.is_enabled()

    Args:
        restore_callback: Name of the instance method to call after data is
            restored from Redis. This callback is ONLY called when there is
            existing data in Redis (i.e., when recovering from a restart).
            Use this to rebuild non-serializable objects like backends.

    Example:
        @redis_persistent(restore_callback="_rebuild_backends")
        class TrainingController:
            # Not persisted - no PersistedMarker annotation
            config: AppConfig
            backends: Dict[str, Backend]

            # Persisted - marked with PersistedMarker
            training_runs: Annotated[Dict[str, TrainingRunRecord], PersistedMarker()]

            def __init__(self, config: AppConfig):
                self.config = config
                self.backends = self._create_backends()
                # training_runs is auto-initialized by decorator

            def _rebuild_backends(self):
                '''Called only when data is restored from Redis'''
                for record in self.training_runs.values():
                    record.backend = self.backends.get(record.base_model)

    Redis Key Format (when persistence is enabled):
        Without instance_id: {namespace}::{ClassName}::{field_name}
        With instance_id: {namespace}::{ClassName}::{instance_id}::{field_name}
    """

    def decorator(target_cls: type[T]) -> type[T]:
        original_init = target_cls.__init__

        @wraps(original_init)
        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            # Get persistence configuration from global settings
            persistence_enabled = PersistenceSettings.is_enabled()
            namespace = PersistenceSettings.get_namespace()
            effective_instance_id = PersistenceSettings.get_effective_instance_id()

            # Get type hints including Annotated metadata
            try:
                hints = get_type_hints(target_cls, include_extras=True)
            except Exception:
                hints = {}

            # Find persisted fields (those marked with PersistedMarker)
            persisted_fields: dict[str, tuple[type, PersistedMarker]] = {}

            for field_name, hint in hints.items():
                # Check if this is an Annotated type
                if get_origin(hint) is not Annotated:
                    continue

                type_args = get_args(hint)
                if len(type_args) < 2:
                    continue

                # Look for PersistedMarker in annotations
                for arg in type_args[1:]:
                    if isinstance(arg, PersistedMarker):
                        actual_type = type_args[0]
                        persisted_fields[field_name] = (actual_type, arg)
                        break

            # Build Redis key prefix
            class_name = target_cls.__name__
            if effective_instance_id:
                base_prefix = f"{namespace}::{class_name}::{effective_instance_id}"
            else:
                base_prefix = f"{namespace}::{class_name}"

            # Track if any data was restored
            has_restored_data = False

            # Collect all deserialized objects for batch hook calling
            collected_objects: list[Any] = []

            # Create PersistentDict or regular dict for each persisted field
            for field_name, (actual_type, _) in persisted_fields.items():
                field_prefix = f"{base_prefix}::{field_name}"

                # Extract value type from Dict[K, V]
                value_type = None
                if get_origin(actual_type) is dict:
                    dict_args = get_args(actual_type)
                    if len(dict_args) == 2:
                        value_type = dict_args[1]

                if persistence_enabled and REDIS_AVAILABLE:
                    # Use Redis-backed PersistentDict
                    from .redis_containers import PersistentDict

                    persistent_dict: PersistentDict[Any, Any] = PersistentDict(
                        key_prefix=field_prefix,
                        value_type=value_type,
                    )

                    # Check if there's existing data (recovery scenario)
                    if len(persistent_dict) > 0:
                        has_restored_data = True
                        # Eagerly load all data and collect objects for hook calling
                        persistent_dict.eager_load_all(collected_objects)

                    # Set the PersistentDict as the attribute
                    setattr(self, field_name, persistent_dict)
                else:
                    # Use regular dict when persistence is disabled
                    setattr(self, field_name, {})

            # Call original __init__
            original_init(self, *args, **kwargs)

            # Call __post_deserialize__ hooks on all collected objects
            # This happens AFTER all data is loaded, BEFORE restore_callback
            if collected_objects:
                for obj in collected_objects:
                    call_post_deserialize(obj)

            # Call restore callback if data was restored
            if has_restored_data and restore_callback:
                callback = getattr(self, restore_callback, None)
                if callback is not None and callable(callback):
                    try:
                        callback()
                    except Exception:
                        pass

        target_cls.__init__ = new_init  # type: ignore

        # Add metadata to class for introspection
        target_cls._redis_persistent_restore_callback = restore_callback  # type: ignore

        return target_cls

    # Handle both @redis_persistent and @redis_persistent() syntax
    if cls is not None:
        # Called as @redis_persistent without parentheses
        return decorator(cls)
    else:
        # Called as @redis_persistent() with parentheses
        return decorator
