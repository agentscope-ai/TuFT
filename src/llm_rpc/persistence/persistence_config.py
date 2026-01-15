"""Global persistence configuration management.

This module provides a centralized configuration for Redis persistence.
All persistence-related settings are managed through a global singleton
that can be configured via:
1. Environment variables (highest priority)
2. Configuration objects (AppConfig.persistence in YAML)
3. Default values

Environment variables:
- LLM_RPC_PERSISTENCE_ENABLED: "true" or "false" (default: "false")
- LLM_RPC_REDIS_URL: Redis connection URL (default: "redis://localhost:6379/0")
- LLM_RPC_PERSISTENCE_NAMESPACE: Key namespace prefix (default: "llm_rpc")
- LLM_RPC_PERSISTENCE_INSTANCE_ID: Instance identifier (default: None)
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass


@dataclass
class PersistenceConfig:
    """Configuration for Redis persistence.

    This is a data class that holds persistence configuration settings.
    It can be used to configure the global PersistenceSettings singleton.

    Attributes:
        enabled: Whether persistence is enabled
        redis_url: Redis connection URL
        namespace: Key namespace prefix for all Redis keys
        instance_id: Instance identifier for multi-instance deployment
            - None: Single instance mode (default). All instances share data.
            - str: Specified ID (e.g., "node-1"). Instances with same ID share data.
            - "auto": Generate a random UUID. Each instance gets isolated data.
    """

    enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    namespace: str = "llm_rpc"
    instance_id: str | None = None


class PersistenceSettings:
    """
    Global persistence settings singleton.

    This class manages the global state for persistence configuration.
    It should be configured once at application startup.

    Usage:
        # Configure at startup (usually in CLI or server initialization)
        PersistenceSettings.configure(
            enabled=True,
            redis_url="redis://localhost:6379/0",
            namespace="llm_rpc",
            instance_id=None,
        )

        # Or from a config object
        PersistenceSettings.configure_from_config(persistence_config)

        # Access settings anywhere
        if PersistenceSettings.is_enabled():
            redis = RedisConnection.get()
            ...

        # Get namespace/instance_id for building keys
        ns = PersistenceSettings.get_namespace()
        iid = PersistenceSettings.get_instance_id()
    """

    _lock = threading.Lock()
    _enabled: bool = False
    _redis_url: str = "redis://localhost:6379/0"
    _namespace: str = "llm_rpc"
    _instance_id: str | None = None
    _configured: bool = False

    @classmethod
    def configure(
        cls,
        enabled: bool | None = None,
        redis_url: str | None = None,
        namespace: str | None = None,
        instance_id: str | None = None,
    ) -> None:
        """
        Configure persistence settings.

        Environment variables take precedence over passed arguments.

        Args:
            enabled: Whether persistence is enabled
            redis_url: Redis connection URL
            namespace: Key namespace prefix
            instance_id: Instance identifier
        """
        with cls._lock:
            # Read from environment variables (highest priority)
            env_enabled = os.getenv("LLM_RPC_PERSISTENCE_ENABLED", "").lower()
            if env_enabled in ("true", "1", "yes"):
                cls._enabled = True
            elif env_enabled in ("false", "0", "no"):
                cls._enabled = False
            elif enabled is not None:
                cls._enabled = enabled

            env_redis_url = os.getenv("LLM_RPC_REDIS_URL")
            if env_redis_url:
                cls._redis_url = env_redis_url
            elif redis_url is not None:
                cls._redis_url = redis_url

            env_namespace = os.getenv("LLM_RPC_PERSISTENCE_NAMESPACE")
            if env_namespace:
                cls._namespace = env_namespace
            elif namespace is not None:
                cls._namespace = namespace

            env_instance_id = os.getenv("LLM_RPC_PERSISTENCE_INSTANCE_ID")
            if env_instance_id:
                cls._instance_id = env_instance_id
            elif instance_id is not None:
                cls._instance_id = instance_id

            cls._configured = True

    @classmethod
    def configure_from_config(cls, config: PersistenceConfig) -> None:
        """Configure persistence settings from a PersistenceConfig object."""
        cls.configure(
            enabled=config.enabled,
            redis_url=config.redis_url,
            namespace=config.namespace,
            instance_id=config.instance_id,
        )

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if persistence is enabled."""
        # Auto-configure from environment if not yet configured
        if not cls._configured:
            cls.configure()
        return cls._enabled

    @classmethod
    def get_redis_url(cls) -> str:
        """Get the Redis connection URL."""
        if not cls._configured:
            cls.configure()
        return cls._redis_url

    @classmethod
    def get_namespace(cls) -> str:
        """Get the namespace for Redis keys."""
        if not cls._configured:
            cls.configure()
        return cls._namespace

    @classmethod
    def get_instance_id(cls) -> str | None:
        """Get the instance ID for multi-instance deployment."""
        if not cls._configured:
            cls.configure()
        return cls._instance_id

    @classmethod
    def get_effective_instance_id(cls) -> str | None:
        """Get the effective instance ID, resolving 'auto' to a UUID.

        Returns:
            The instance ID, or a new UUID if instance_id is "auto".
        """
        import uuid

        instance_id = cls.get_instance_id()
        if instance_id == "auto":
            return str(uuid.uuid4())
        return instance_id

    @classmethod
    def is_configured(cls) -> bool:
        """Check if settings have been explicitly configured."""
        return cls._configured

    @classmethod
    def reset(cls) -> None:
        """Reset all settings to defaults. Useful for testing."""
        with cls._lock:
            cls._enabled = False
            cls._redis_url = "redis://localhost:6379/0"
            cls._namespace = "llm_rpc"
            cls._instance_id = None
            cls._configured = False

    @classmethod
    def get_config(cls) -> PersistenceConfig:
        """Get the current configuration as a PersistenceConfig object."""
        return PersistenceConfig(
            enabled=cls.is_enabled(),
            redis_url=cls.get_redis_url(),
            namespace=cls.get_namespace(),
            instance_id=cls.get_instance_id(),
        )


# Convenience functions for accessing global settings
def is_persistence_enabled() -> bool:
    """Check if persistence is enabled."""
    return PersistenceSettings.is_enabled()


def get_namespace() -> str:
    """Get the namespace for Redis keys."""
    return PersistenceSettings.get_namespace()


def get_instance_id() -> str | None:
    """Get the instance ID for multi-instance deployment."""
    return PersistenceSettings.get_instance_id()


def get_redis_url() -> str:
    """Get the Redis connection URL."""
    return PersistenceSettings.get_redis_url()


# Check if Redis dependencies are available
def check_redis_available() -> bool:
    """Check if Redis dependencies (redis, pottery) are available."""
    try:
        import pottery  # noqa: F401
        import redis  # noqa: F401

        return True
    except ImportError:
        return False


REDIS_AVAILABLE = check_redis_available()
