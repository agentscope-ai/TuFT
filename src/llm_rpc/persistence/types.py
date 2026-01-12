"""Type definitions for Redis persistence."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Annotated, Any, Callable, get_args, get_origin, get_type_hints

logger = logging.getLogger(__name__)


class PersistedMarker:
    """
    Marker class for fields that should be persisted to Redis.

    Used with typing.Annotated on controller class fields to mark
    them for persistence.

    Usage:
        from typing import Annotated, Dict
        from llm_rpc.persistence import PersistedMarker

        class MyController:
            # This field will be persisted to Redis
            records: Annotated[Dict[str, Record], PersistedMarker()]
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "PersistedMarker()"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PersistedMarker):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash("PersistedMarker")


class PersistenceExclude:
    """
    Marker for fields that should be excluded from persistence/serialization.

    Use this on dataclass/Pydantic model fields that cannot be serialized
    (e.g., asyncio.Lock, backends, callbacks).

    Args:
        default: A default value to use when deserializing.
        default_factory: A callable that returns the default value when deserializing.
                        Use this for mutable defaults or objects that need to be created fresh.

    Usage:
        from dataclasses import dataclass, field
        from typing import Annotated
        from llm_rpc.persistence import PersistenceExclude

        @dataclass
        class TrainingRunRecord:
            training_run_id: str

            # Excluded with None default
            backend: Annotated[Any, PersistenceExclude()]

            # Excluded with a factory function
            _execution_lock: Annotated[asyncio.Lock, PersistenceExclude(
                default_factory=asyncio.Lock
            )]

            # Excluded with a default value
            _cache: Annotated[dict, PersistenceExclude(default_factory=dict)]

            # Excluded with a custom factory
            _connection: Annotated[Any, PersistenceExclude(
                default_factory=lambda: create_connection()
            )]
    """

    __slots__ = ("default", "default_factory")

    def __init__(
        self,
        default: Any = None,
        default_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.default = default
        self.default_factory = default_factory

    def get_default_value(self) -> Any:
        """Get the default value for this excluded field."""
        if self.default_factory is not None:
            return self.default_factory()
        return self.default

    def __repr__(self) -> str:
        if self.default_factory:
            return f"PersistenceExclude(default_factory={self.default_factory!r})"
        if self.default is not None:
            return f"PersistenceExclude(default={self.default!r})"
        return "PersistenceExclude()"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PersistenceExclude):
            return NotImplemented
        return self.default == other.default and self.default_factory == other.default_factory

    def __hash__(self) -> int:
        return hash(("PersistenceExclude", self.default))


# Types that are known to be non-serializable
NON_SERIALIZABLE_TYPES = (
    asyncio.Lock,
    asyncio.Event,
    asyncio.Task,
    asyncio.Semaphore,
    asyncio.Condition,
    asyncio.Queue,
)


# Registry for persistable model classes
_PERSISTABLE_MODELS: dict[str, type] = {}


def register_persistable(cls: type) -> None:
    """Register a class as persistable for deserialization."""
    key = f"{cls.__module__}.{cls.__name__}"
    _PERSISTABLE_MODELS[key] = cls


def get_persistable_class(module: str, name: str) -> type | None:
    """Get a registered persistable class by module and name."""
    key = f"{module}.{name}"
    return _PERSISTABLE_MODELS.get(key)


def get_all_persistable_classes() -> dict[str, type]:
    """Get all registered persistable classes."""
    return _PERSISTABLE_MODELS.copy()


def persistable(cls: type) -> type:
    """
    Class decorator to mark a dataclass/Pydantic model as persistable.

    Persistable classes:
    - Are automatically registered for deserialization
    - Have their PersistenceExclude-marked fields skipped during serialization
    - Can define a __post_deserialize__ method for custom initialization
    - Have their nested container fields automatically persisted

    The __post_deserialize__ hook:
    - Called after the object is deserialized from Redis
    - Use this for complex initialization that can't be done with default_factory
    - Can access all deserialized fields

    Usage:
        from dataclasses import dataclass, field
        from llm_rpc.persistence import persistable, PersistenceExclude
        from typing import Annotated

        @persistable
        @dataclass
        class TrainingRunRecord:
            training_run_id: str
            base_model: str

            # Simple exclusion with factory
            _execution_lock: Annotated[asyncio.Lock, PersistenceExclude(
                default_factory=asyncio.Lock
            )]

            # Excluded, will be set in __post_deserialize__
            backend: Annotated[Any, PersistenceExclude()]

            def __post_deserialize__(self):
                '''Called after deserialization, for complex initialization'''
                # This is called after all fields are restored
                # self.base_model is available here
                pass
    """
    # Register the class
    register_persistable(cls)

    # Mark as persistable
    cls._is_persistable = True  # type: ignore

    # Validate fields and warn about potential issues
    _validate_persistable_fields(cls)

    return cls


def _validate_persistable_fields(cls: type) -> None:
    """Validate fields of a persistable class and warn about issues."""
    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        return

    excluded = get_excluded_fields(cls)

    for field_name, hint in hints.items():
        if field_name in excluded:
            continue

        # Get the actual type (strip Annotated wrapper)
        actual_type = hint
        if get_origin(hint) is Annotated:
            actual_type = get_args(hint)[0]

        # Check if type is known to be non-serializable
        try:
            if isinstance(actual_type, type):
                for non_serial in NON_SERIALIZABLE_TYPES:
                    if issubclass(actual_type, non_serial):
                        logger.warning(
                            f"Field '{field_name}' in {cls.__name__} has non-serializable type "
                            f"{actual_type.__name__}. Consider marking it with PersistenceExclude() "
                            f"or using init=False in dataclass field."
                        )
                        break
        except TypeError:
            pass


def is_persistable(cls: type | None) -> bool:
    """Check if a class is marked as persistable."""
    if cls is None:
        return False
    return getattr(cls, "_is_persistable", False)


def get_excluded_fields(cls: type) -> dict[str, PersistenceExclude]:
    """
    Get the excluded fields and their PersistenceExclude configs for a class.

    Returns a dict mapping field names to their PersistenceExclude instances.
    """
    excluded: dict[str, PersistenceExclude] = {}

    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        return excluded

    for field_name, hint in hints.items():
        if get_origin(hint) is Annotated:
            args = get_args(hint)
            for arg in args[1:]:
                if isinstance(arg, PersistenceExclude):
                    excluded[field_name] = arg
                    break

    # Auto-exclude non-init dataclass fields
    if dataclasses.is_dataclass(cls):
        for field in dataclasses.fields(cls):
            if not field.init and field.name not in excluded:
                # Create a default PersistenceExclude for non-init fields
                excluded[field.name] = PersistenceExclude()

    return excluded


def get_excluded_field_names(cls: type) -> set[str]:
    """Get just the names of excluded fields."""
    return set(get_excluded_fields(cls).keys())


def call_post_deserialize(obj: Any) -> None:
    """Call __post_deserialize__ hook if it exists on the object."""
    hook = getattr(obj, "__post_deserialize__", None)
    if hook is not None and callable(hook):
        try:
            hook()
        except Exception as e:
            logger.warning(
                f"__post_deserialize__ hook failed for {type(obj).__name__}: {e}"
            )
