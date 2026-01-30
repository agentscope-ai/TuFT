"""Simple Redis persistence module for TuFT.

This module provides direct Redis-based persistence using redis-py.
Each data record is stored as a separate Redis key with JSON serialization.

Key Design:
- Top-level records: {namespace}::{type}::{id}
- Nested records: {namespace}::{type}::{parent_id}::{nested_type}::{nested_id}

Persistence Modes:
- DISABLE: No persistence, all data is in-memory only
- REDIS_URL: Use external Redis server via URL
- FILE_REDIS: Use file-backed storage for tests and demos

Config Validation:
- On startup, the current config signature is compared with the stored signature
- If mismatch is detected, server stops with an error message
- Use --refresh-persistence to override and clear existing data
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


def _get_tracer():
    """Lazy import tracer to avoid circular imports."""
    from tuft.telemetry.tracing import get_tracer

    return get_tracer("tuft.redis_store")


def _get_metrics():
    """Lazy import metrics to avoid circular imports."""
    from tuft.telemetry.metrics import get_metrics

    return get_metrics()


T = TypeVar("T", bound=BaseModel)


class PersistenceMode(str, Enum):
    """Persistence mode options."""

    DISABLE = "disabled"  # No persistence
    REDIS_URL = "redis_url"  # Use external Redis server
    FILE_REDIS = "file_redis"  # Use file-backed storage for tests/demos


# Default TTL values in seconds
DEFAULT_FUTURE_TTL_SECONDS = 24 * 3600  # 1 day for future records (short-lived)


class ConfigCheckField:
    """Available fields that can be checked for configuration validation.

    Field names correspond directly to AppConfig attribute names.
    SUPPORTED_MODELS is always required (mandatory) for restore safety.
    """

    SUPPORTED_MODELS = "supported_models"
    CHECKPOINT_DIR = "checkpoint_dir"
    MODEL_OWNER = "model_owner"
    TOY_BACKEND_SEED = "toy_backend_seed"
    AUTHORIZED_USERS = "authorized_users"
    TELEMETRY = "telemetry"


# Default fields to check (supported_models is mandatory)
DEFAULT_CHECK_FIELDS: list[str] = [ConfigCheckField.SUPPORTED_MODELS]


class PersistenceConfig(BaseModel):
    """Configuration for Redis persistence.

    Attributes:
        mode: Persistence mode - DISABLE, REDIS_URL, or FILE_REDIS
        redis_url: Redis server URL (only used when mode=REDIS_URL)
        file_path: JSON file path (only used when mode=FILE_REDIS)
        namespace: Key namespace prefix for Redis keys. Defaults to "tuft".
        future_ttl_seconds: TTL for future records in seconds. None means no expiry.
        check_fields: List of AppConfig fields to validate on restart.
                     Defaults to ["SUPPORTED_MODELS"]. SUPPORTED_MODELS is always
                     checked regardless of this setting for restore safety.
                     Available fields: SUPPORTED_MODELS, CHECKPOINT_DIR, MODEL_OWNER,
                     TOY_BACKEND_SEED, AUTHORIZED_USERS, TELEMETRY.
    """

    # Allow Path type
    model_config = {"arbitrary_types_allowed": True}

    mode: PersistenceMode = PersistenceMode.DISABLE
    redis_url: str = "redis://localhost:6379/0"
    file_path: Path | None = None
    namespace: str = "tuft"  # Default namespace for Redis keys
    future_ttl_seconds: int | None = DEFAULT_FUTURE_TTL_SECONDS
    check_fields: list[str] = Field(default_factory=lambda: DEFAULT_CHECK_FIELDS.copy())

    @property
    def enabled(self) -> bool:
        """Check if persistence is enabled."""
        return self.mode != PersistenceMode.DISABLE

    def get_check_fields(self) -> list[str]:
        """Get the fields to check, ensuring SUPPORTED_MODELS is always included."""
        fields = list(self.check_fields)
        if ConfigCheckField.SUPPORTED_MODELS not in fields:
            fields.insert(0, ConfigCheckField.SUPPORTED_MODELS)
        return fields

    @classmethod
    def from_redis_url(
        cls,
        redis_url: str,
        namespace: str = "tuft",
        future_ttl_seconds: int | None = DEFAULT_FUTURE_TTL_SECONDS,
        check_fields: list[str] | None = None,
    ) -> "PersistenceConfig":
        """Create a config using external Redis server."""
        return cls(
            mode=PersistenceMode.REDIS_URL,
            redis_url=redis_url,
            namespace=namespace,
            future_ttl_seconds=future_ttl_seconds,
            check_fields=check_fields or DEFAULT_CHECK_FIELDS.copy(),
        )

    @classmethod
    def from_file_redis(
        cls,
        file_path: Path | None = None,
        namespace: str = "tuft",
        future_ttl_seconds: int | None = DEFAULT_FUTURE_TTL_SECONDS,
        check_fields: list[str] | None = None,
    ) -> "PersistenceConfig":
        """Create a config using file-backed storage."""
        return cls(
            mode=PersistenceMode.FILE_REDIS,
            file_path=file_path,
            namespace=namespace,
            future_ttl_seconds=future_ttl_seconds,
            check_fields=check_fields or DEFAULT_CHECK_FIELDS.copy(),
        )


class RedisStore:
    """Global Redis connection and operation manager.

    Supports two modes:
    - External Redis server (via redis-py)
    - No persistence (DISABLE mode)
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

                    if self._config.mode in (PersistenceMode.REDIS_URL, PersistenceMode.FILE_REDIS):
                        logger.info("Redis connection begin")
                        self._redis = self._create_redis_client()

                    if self._redis is not None:
                        self._pid = current_pid
                        logger.info("Redis connection established")

        return self._redis

    def _create_redis_client(self) -> Any:
        """Create a client for the configured persistence backend."""
        if self._config is None:
            return None
        try:
            if self._config.mode == PersistenceMode.FILE_REDIS:
                from .file_redis import FileRedis

                file_path = self._config.file_path or (
                    Path.home() / ".cache" / "tuft" / "file_redis.json"
                )
                return FileRedis(file_path=file_path)
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

        start_time = time.perf_counter()
        tracer = _get_tracer()

        try:
            with tracer.start_as_current_span("redis.SET") as span:
                span.set_attribute("db.system", "redis")
                span.set_attribute("db.operation", "SET")
                if ttl_seconds is not None:
                    redis.setex(key, ttl_seconds, value)
                else:
                    redis.set(key, value)

            # Record metrics
            duration = time.perf_counter() - start_time
            metrics = _get_metrics()
            metrics.redis_operation_duration.record(duration, {"operation": "SET"})
            if duration > 0.1:
                logger.warning("Redis operation slow: SET (%.3fs)", duration)

            return True
        except Exception:
            logger.exception("Failed to set key %s in Redis", key)
            logger.error("Redis connection failed")
            return False

    def get(self, key: str) -> str | None:
        redis = self._get_redis()
        if redis is None:
            return None

        start_time = time.perf_counter()
        tracer = _get_tracer()

        try:
            with tracer.start_as_current_span("redis.GET") as span:
                span.set_attribute("db.system", "redis")
                span.set_attribute("db.operation", "GET")
                result = redis.get(key)

            # Record metrics
            duration = time.perf_counter() - start_time
            metrics = _get_metrics()
            metrics.redis_operation_duration.record(duration, {"operation": "GET"})
            if duration > 0.1:
                logger.warning("Redis operation slow: GET (%.3fs)", duration)

            return result
        except Exception:
            logger.exception("Failed to get key %s from Redis", key)
            logger.error("Redis connection failed")
            return None

    def delete(self, key: str) -> bool:
        redis = self._get_redis()
        if redis is None:
            return False

        start_time = time.perf_counter()
        tracer = _get_tracer()

        try:
            with tracer.start_as_current_span("redis.DEL") as span:
                span.set_attribute("db.system", "redis")
                span.set_attribute("db.operation", "DEL")
                redis.delete(key)

            # Record metrics
            duration = time.perf_counter() - start_time
            metrics = _get_metrics()
            metrics.redis_operation_duration.record(duration, {"operation": "DEL"})

            return True
        except Exception:
            logger.exception("Failed to delete key %s from Redis", key)
            return False

    def keys(self, pattern: str) -> list[str]:
        """Get all keys matching the pattern using SCAN for better performance."""
        redis = self._get_redis()
        if redis is None:
            return []

        start_time = time.perf_counter()
        tracer = _get_tracer()

        try:
            with tracer.start_as_current_span("redis.SCAN") as span:
                span.set_attribute("db.system", "redis")
                span.set_attribute("db.operation", "SCAN")
                result = list(redis.scan_iter(match=pattern))

            # Record metrics
            duration = time.perf_counter() - start_time
            metrics = _get_metrics()
            metrics.redis_operation_duration.record(duration, {"operation": "SCAN"})

            return result
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


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ConfigSignature(BaseModel):
    """Stores a complete snapshot of AppConfig for validation on restart.

    Since AppConfig is now a Pydantic model, we directly store its serialized
    form (excluding the persistence field which is runtime-only).
    """

    # Serialized AppConfig data (excludes persistence)
    config_data: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    created_at: datetime = Field(default_factory=_now)
    namespace: str = "tuft"

    @classmethod
    def from_app_config(cls, config: Any) -> "ConfigSignature":
        """Create a signature by serializing the AppConfig."""
        # Use the method on AppConfig to get persistence-safe data
        config_data = config.get_config_for_persistence()
        namespace = config.persistence.namespace if config.persistence else "tuft"
        return cls(config_data=config_data, namespace=namespace)

    def _get_field_value(self, field_name: str) -> Any:
        """Get the value of a field by name."""
        return self.config_data.get(field_name)

    def _normalize_for_comparison(self, value: Any) -> Any:
        if isinstance(value, list):
            normalized_items = []
            for item in value:
                if isinstance(item, dict):
                    normalized_items.append(tuple(sorted(item.items())))
                else:
                    normalized_items.append(item)
            # Sort for order-independent comparison
            return sorted(normalized_items, key=lambda x: str(x))
        return value

    def _compare_field(self, other: "ConfigSignature", field_name: str) -> bool:
        """Compare a single field between two signatures."""
        current_value = self._get_field_value(field_name)
        other_value = other._get_field_value(field_name)
        current_normalized = self._normalize_for_comparison(current_value)
        other_normalized = self._normalize_for_comparison(other_value)

        return current_normalized == other_normalized

    def _get_field_diff(self, other: "ConfigSignature", field_name: str) -> dict[str, Any] | None:
        """Get the difference for a single field.

        Returns:
            {"current": value, "stored": value} if different, None otherwise.
        """
        current_value = self._get_field_value(field_name)
        other_value = other._get_field_value(field_name)

        current_normalized = self._normalize_for_comparison(current_value)
        other_normalized = self._normalize_for_comparison(other_value)

        if current_normalized != other_normalized:
            return {"current": current_value, "stored": other_value}
        return None

    def matches(
        self,
        other: "ConfigSignature",
        check_fields: list[str] | None = None,
    ) -> bool:
        """Check if this signature matches another signature.

        Args:
            other: The other signature to compare against.
            check_fields: List of field names to check. If None, uses DEFAULT_CHECK_FIELDS.
                         SUPPORTED_MODELS is always included (mandatory).

        Returns:
            True if all specified fields match, False otherwise.
        """
        fields_to_check = self._get_fields_to_check(check_fields)

        for field_name in fields_to_check:
            if not self._compare_field(other, field_name):
                return False
        return True

    def get_diff(
        self,
        other: "ConfigSignature",
        check_fields: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Get the differences between this signature and another.

        Args:
            other: The other signature to compare against.
            check_fields: List of field names to check. If None, uses DEFAULT_CHECK_FIELDS.
                         SUPPORTED_MODELS is always included (mandatory).

        Returns:
            Dict mapping field names to their differences.
        """
        fields_to_check = self._get_fields_to_check(check_fields)
        diff: dict[str, dict[str, Any]] = {}

        for field_name in fields_to_check:
            field_diff = self._get_field_diff(other, field_name)
            if field_diff is not None:
                diff[field_name] = field_diff

        return diff

    def _get_fields_to_check(self, check_fields: list[str] | None) -> list[str]:
        """Get the list of fields to check, ensuring mandatory fields are included."""
        if check_fields is None:
            return DEFAULT_CHECK_FIELDS.copy()

        # Ensure SUPPORTED_MODELS is always included (mandatory)
        fields = list(check_fields)
        if ConfigCheckField.SUPPORTED_MODELS not in fields:
            fields.insert(0, ConfigCheckField.SUPPORTED_MODELS)
        return fields


CONFIG_SIGNATURE_KEY = "config_signature"


def save_config_signature(config: Any) -> bool:
    """Save the config signature to Redis.

    Args:
        config: The AppConfig to create a signature from.

    Returns:
        True if saved successfully, False otherwise.
    """
    store = RedisStore.get_instance()
    if not store.is_enabled:
        return False

    signature = ConfigSignature.from_app_config(config)
    key = store.build_key(CONFIG_SIGNATURE_KEY)

    try:
        json_str = signature.model_dump_json()
        return store.set(key, json_str)
    except Exception:
        logger.exception("Failed to save config signature to Redis")
        return False


def load_config_signature() -> ConfigSignature | None:
    """Load the config signature from Redis.

    Returns:
        The stored ConfigSignature, or None if not found.
    """
    store = RedisStore.get_instance()
    if not store.is_enabled:
        return None

    key = store.build_key(CONFIG_SIGNATURE_KEY)

    try:
        json_str = store.get(key)
        if json_str is None:
            return None
        return ConfigSignature.model_validate_json(json_str)
    except Exception:
        logger.exception("Failed to load config signature from Redis")
        return None


def has_existing_data() -> bool:
    """Check if there is any existing data in the current namespace.

    Returns:
        True if any keys exist in the namespace, False otherwise.
    """
    store = RedisStore.get_instance()
    if not store.is_enabled:
        return False

    pattern = f"{store.namespace}::*"
    keys = store.keys(pattern)
    return len(keys) > 0


def validate_config_signature(config: Any) -> bool:
    """Validate that the current config matches the stored config signature.

    This function ONLY reads from Redis, it does NOT write.
    The signature should be saved after successful restore using
    save_config_signature().

    The fields to check are read from config.persistence.check_fields.
    SUPPORTED_MODELS is always checked regardless of this setting.

    This function handles several cases:
    1. No signature AND no other data in namespace -> fresh start (return True)
    2. No signature BUT other data exists -> corrupted/incompatible state, raise error
    3. Signature exists and matches -> OK (return False, not fresh)
    4. Signature exists but doesn't match -> raise error

    Args:
        config: The current AppConfig to validate.

    Returns:
        True if this is a fresh start (no existing data), False otherwise.

    Raises:
        ConfigMismatchError: If the configs don't match or state is corrupted.
    """
    from tuft.exceptions import ConfigMismatchError

    stored = load_config_signature()

    if stored is None:
        # Check if there's any other data in the namespace
        if has_existing_data():
            # Data exists but no signature -> corrupted/incompatible state
            logger.warning(
                "Redis namespace has data but no config signature. "
                "This indicates a corrupted or incompatible persistence state."
            )
            raise ConfigMismatchError(
                diff={
                    "_state": {
                        "current": "valid configuration",
                        "stored": "missing signature (corrupted or legacy data)",
                    }
                }
            )
        else:
            # No data at all -> fresh start
            logger.info("No stored config signature found - fresh start")
            return True

    # Get check_fields from persistence config
    check_fields = config.persistence.get_check_fields() if config.persistence else None

    current = ConfigSignature.from_app_config(config)
    if not current.matches(stored, check_fields=check_fields):
        diff = current.get_diff(stored, check_fields=check_fields)
        raise ConfigMismatchError(diff)

    logger.debug("Config signature validated successfully")
    return False


def get_current_namespace() -> str:
    """Get the current Redis namespace.

    Returns:
        The namespace string, or 'tuft' if not configured.
    """
    store = RedisStore.get_instance()
    return store.namespace


def flush_all_data() -> tuple[int, str]:
    """Clear all data from the current Redis namespace.

    This removes all keys with the current namespace prefix.
    Use with caution - this is destructive!

    Returns:
        A tuple of (number of keys deleted, namespace that was cleared).
    """
    store = RedisStore.get_instance()
    if not store.is_enabled:
        return 0, store.namespace

    pattern = f"{store.namespace}::*"
    deleted_count = store.delete_pattern(pattern)
    return deleted_count, store.namespace
