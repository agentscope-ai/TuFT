"""Unified serializer supporting dataclass and Pydantic models."""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from .types import (
    call_post_deserialize,
    get_excluded_fields,
    get_persistable_class,
    is_persistable,
)

# Try to import Pydantic
try:
    from pydantic import BaseModel

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = None  # type: ignore


def is_dataclass_instance(obj: Any) -> bool:
    """Check if obj is a dataclass instance (not a class)."""
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def is_dataclass_type(t: type | None) -> bool:
    """Check if t is a dataclass type."""
    return t is not None and dataclasses.is_dataclass(t) and isinstance(t, type)


def is_pydantic_instance(obj: Any) -> bool:
    """Check if obj is a Pydantic model instance."""
    if not HAS_PYDANTIC or BaseModel is None:
        return False
    return isinstance(obj, BaseModel)


def is_pydantic_type(t: type | None) -> bool:
    """Check if t is a Pydantic model type."""
    if not HAS_PYDANTIC or BaseModel is None:
        return False
    try:
        return t is not None and isinstance(t, type) and issubclass(t, BaseModel)
    except TypeError:
        return False


def is_model_instance(obj: Any) -> bool:
    """Check if obj is either a dataclass or Pydantic instance."""
    return is_dataclass_instance(obj) or is_pydantic_instance(obj)


def is_model_type(t: type | None) -> bool:
    """Check if t is either a dataclass or Pydantic type."""
    return is_dataclass_type(t) or is_pydantic_type(t)


class ModelSerializer:
    """
    Unified serializer for dataclass and Pydantic models.

    Features:
    - Automatic handling of common types (datetime, Path)
    - Skip non-serializable types (asyncio.Lock, asyncio.Event, etc.)
    - Support for nested models marked with @persistable
    - Field exclusion via Exclude marker
    - Support for dict, list, set containers
    """

    # Types that should always be skipped during serialization
    SKIP_TYPES = (asyncio.Lock, asyncio.Event, asyncio.Task, asyncio.Semaphore)

    # Additional types to skip - set dynamically to avoid circular imports
    _extra_skip_types: tuple = ()

    @classmethod
    def add_skip_types(cls, *types: type) -> None:
        """Add additional types to skip during serialization."""
        cls._extra_skip_types = cls._extra_skip_types + types

    @classmethod
    def should_skip(cls, value: Any) -> bool:
        """Check if a value should be skipped during serialization."""
        if isinstance(value, cls.SKIP_TYPES):
            return True
        if cls._extra_skip_types and isinstance(value, cls._extra_skip_types):
            return True
        return False

    @classmethod
    def get_model_fields(cls, model_type: type) -> dict[str, type | None]:
        """
        Get field names and their types from a model type.

        Args:
            model_type: A dataclass or Pydantic model type

        Returns:
            Dict mapping field names to their types
        """
        if is_dataclass_type(model_type):
            result = {}
            hints = get_type_hints(model_type)
            for field in dataclasses.fields(model_type):
                result[field.name] = hints.get(field.name)
            return result

        if is_pydantic_type(model_type):
            return {
                name: info.annotation
                for name, info in model_type.model_fields.items()  # type: ignore
            }

        return {}

    @classmethod
    def serialize(
        cls,
        obj: Any,
        nested_container_fields: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Serialize a dataclass or Pydantic model to a dictionary.

        Uses the @persistable decorator and Exclude markers to determine
        which fields to serialize.

        Args:
            obj: The model instance to serialize
            nested_container_fields: Fields containing nested containers
                                    that are stored separately (skipped here)

        Returns:
            Serialized dictionary representation

        Raises:
            TypeError: If obj is not a supported model type
        """
        if nested_container_fields is None:
            nested_container_fields = set()

        # Get excluded fields from class definition
        excluded = get_excluded_fields(type(obj))

        result: dict[str, Any] = {}

        # Handle Pydantic models
        if is_pydantic_instance(obj):
            exclude_set = set(excluded.keys()) | nested_container_fields
            raw = obj.model_dump(exclude=exclude_set, mode="python")  # type: ignore
            for key, value in raw.items():
                result[key] = cls._serialize_value(value)
            return result

        # Handle dataclass
        if is_dataclass_instance(obj):
            for field in dataclasses.fields(obj):
                name = field.name

                # Skip excluded fields
                if name in excluded:
                    continue

                # Skip nested container fields (stored separately)
                if name in nested_container_fields:
                    continue

                value = getattr(obj, name)

                # Skip non-serializable types
                if cls.should_skip(value):
                    continue

                result[name] = cls._serialize_value(value)
            return result

        raise TypeError(f"Cannot serialize {type(obj)}, must be dataclass or Pydantic model")

    @classmethod
    def _serialize_value(cls, value: Any) -> Any:
        """Serialize a single value with type-specific handling."""
        if value is None:
            return None

        # Handle datetime
        if isinstance(value, datetime):
            return {"__type__": "datetime", "value": value.isoformat()}

        # Handle Path
        if isinstance(value, Path):
            return {"__type__": "Path", "value": str(value)}

        # Handle nested models
        if is_model_instance(value):
            # For @persistable models, serialize with type info for reconstruction
            if is_persistable(type(value)):
                type_name = type(value).__name__
                module = type(value).__module__
                return {
                    "__type__": "model",
                    "class": type_name,
                    "module": module,
                    "value": cls.serialize(value),
                }
            # For other Pydantic models (e.g., tinker types), use model_dump
            # This preserves them as plain dicts that FastAPI can serialize
            if is_pydantic_instance(value):
                return value.model_dump(mode="python")
            # For other dataclasses, serialize as dict
            if is_dataclass_instance(value):
                return cls.serialize(value)

        # Handle lists
        if isinstance(value, list):
            return {"__type__": "list", "value": [cls._serialize_value(item) for item in value]}

        # Handle sets
        if isinstance(value, (set, frozenset)):
            return {"__type__": "set", "value": [cls._serialize_value(item) for item in value]}

        # Handle dicts (non-container dicts that are stored inline)
        if isinstance(value, dict):
            return {
                "__type__": "dict",
                "value": {str(k): cls._serialize_value(v) for k, v in value.items()},
            }

        # Handle basic types
        if isinstance(value, (str, int, float, bool)):
            return value

        # Handle enums
        if hasattr(value, "value") and hasattr(type(value), "__members__"):
            return {
                "__type__": "enum",
                "class": type(value).__name__,
                "module": type(value).__module__,
                "value": value.value,
            }

        # Other types - try to return as-is (may fail during JSON serialization)
        return value

    @classmethod
    def deserialize(
        cls,
        data: dict[str, Any],
        target_class: type,
        call_post_hook: bool = True,
        collected_objects: list[Any] | None = None,
    ) -> Any:
        """
        Deserialize a dictionary to a dataclass or Pydantic model.

        Handles PersistenceExclude fields by using their default/default_factory.
        After deserialization, calls __post_deserialize__ hook if present
        (unless collected_objects is provided, in which case objects are collected
        for batch hook calling later).

        Args:
            data: The serialized dictionary
            target_class: The target class to deserialize into
            call_post_hook: Whether to call __post_deserialize__ after creation
            collected_objects: If provided, deserialized objects are added to this
                list instead of having their hooks called immediately. This allows
                batch calling of hooks after all objects are loaded.

        Returns:
            Deserialized model instance

        Raises:
            TypeError: If target_class is not a supported model type
        """
        excluded = get_excluded_fields(target_class)

        # Handle Pydantic models
        if is_pydantic_type(target_class):
            processed = {}

            # First, add deserialized data fields
            for key, value in data.items():
                if key in excluded:
                    continue
                processed[key] = cls._deserialize_value(value, None)

            # Then, add excluded fields with their defaults
            for field_name, exclude_marker in excluded.items():
                if field_name not in processed:
                    processed[field_name] = exclude_marker.get_default_value()

            instance = target_class.model_validate(processed)  # type: ignore

            # Collect object for batch hook calling, or call hook immediately
            if collected_objects is not None:
                collected_objects.append(instance)
            elif call_post_hook:
                call_post_deserialize(instance)

            return instance

        # Handle dataclass
        if is_dataclass_type(target_class):
            init_kwargs = {}
            fields_info = {f.name: f for f in dataclasses.fields(target_class)}
            type_hints = get_type_hints(target_class)

            for name, field_info in fields_info.items():
                # Handle excluded fields with their defaults
                if name in excluded:
                    exclude_marker = excluded[name]
                    if field_info.init:
                        init_kwargs[name] = exclude_marker.get_default_value()
                    continue

                if not field_info.init:
                    continue

                if name in data:
                    value = data[name]
                    field_type = type_hints.get(name)
                    init_kwargs[name] = cls._deserialize_value(value, field_type)

            instance = target_class(**init_kwargs)

            # For non-init excluded fields with explicit default/default_factory,
            # set them after construction. Skip if no explicit default is provided
            # to allow __post_init__ to handle initialization.
            for name, exclude_marker in excluded.items():
                if name in fields_info and not fields_info[name].init:
                    # Only set if there's an explicit default or default_factory
                    if (
                        exclude_marker.default is not None
                        or exclude_marker.default_factory is not None
                    ):
                        try:
                            object.__setattr__(instance, name, exclude_marker.get_default_value())
                        except Exception:
                            pass

            # Collect object for batch hook calling, or call hook immediately
            if collected_objects is not None:
                collected_objects.append(instance)
            elif call_post_hook:
                call_post_deserialize(instance)

            return instance

        raise TypeError(
            f"Cannot deserialize to {target_class}, must be dataclass or Pydantic model"
        )

    @classmethod
    def _deserialize_value(
        cls,
        value: Any,
        expected_type: type | None,
    ) -> Any:
        """Deserialize a single value with type-specific handling."""
        if value is None:
            return None

        # Handle special type markers
        if isinstance(value, dict) and "__type__" in value:
            type_name = value["__type__"]

            if type_name == "datetime":
                return datetime.fromisoformat(value["value"])

            if type_name == "Path":
                return Path(value["value"])

            if type_name == "model":
                nested_class = get_persistable_class(value["module"], value["class"])
                if nested_class is not None:
                    return cls.deserialize(value["value"], nested_class)
                # If class not registered, return raw dict
                return value["value"]

            if type_name == "list":
                element_type = None
                if expected_type:
                    origin = get_origin(expected_type)
                    if origin is list:
                        args = get_args(expected_type)
                        if args:
                            element_type = args[0]
                return [cls._deserialize_value(item, element_type) for item in value["value"]]

            if type_name == "set":
                element_type = None
                if expected_type:
                    origin = get_origin(expected_type)
                    if origin in (set, frozenset):
                        args = get_args(expected_type)
                        if args:
                            element_type = args[0]
                return set(cls._deserialize_value(item, element_type) for item in value["value"])

            if type_name == "dict":
                value_type = None
                if expected_type:
                    origin = get_origin(expected_type)
                    if origin is dict:
                        args = get_args(expected_type)
                        if len(args) == 2:
                            value_type = args[1]
                return {k: cls._deserialize_value(v, value_type) for k, v in value["value"].items()}

            if type_name == "enum":
                enum_class = get_persistable_class(value["module"], value["class"])
                if enum_class is not None:
                    return enum_class(value["value"])
                return value["value"]

        # Handle plain dicts (old format or simple dicts)
        if isinstance(value, dict):
            value_type = None
            if expected_type:
                origin = get_origin(expected_type)
                if origin is dict:
                    args = get_args(expected_type)
                    if len(args) == 2:
                        value_type = args[1]
            return {k: cls._deserialize_value(v, value_type) for k, v in value.items()}

        # Handle plain lists
        if isinstance(value, list):
            element_type = None
            if expected_type:
                origin = get_origin(expected_type)
                if origin is list:
                    args = get_args(expected_type)
                    if args:
                        element_type = args[0]
            return [cls._deserialize_value(item, element_type) for item in value]

        return value

    @classmethod
    def find_nested_container_fields(
        cls,
        model_type: type,
    ) -> dict[str, tuple[str, type | None]]:
        """
        Find fields in a model that are container types (dict, list, set).

        These fields need special handling for nested persistence.

        Args:
            model_type: The model type to analyze

        Returns:
            Dict mapping field names to (container_type, element_type)
            container_type is one of: "dict", "list", "set"
        """
        excluded = get_excluded_fields(model_type)
        nested: dict[str, tuple[str, type | None]] = {}
        fields = cls.get_model_fields(model_type)

        for name, field_type in fields.items():
            if name in excluded:
                continue
            if field_type is None:
                continue

            origin = get_origin(field_type)
            args = get_args(field_type)

            if origin is dict and len(args) == 2:
                value_type = args[1]
                # Only persist dicts with persistable value types
                if is_persistable(value_type) or is_model_type(value_type):
                    nested[name] = ("dict", value_type)

            elif origin is list and len(args) == 1:
                element_type = args[0]
                if is_persistable(element_type) or is_model_type(element_type):
                    nested[name] = ("list", element_type)

            elif origin in (set, frozenset) and len(args) == 1:
                element_type = args[0]
                if is_persistable(element_type) or is_model_type(element_type):
                    nested[name] = ("set", element_type)

        return nested
