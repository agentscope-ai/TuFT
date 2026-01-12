"""Global Redis connection management."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis import Redis


class RedisConnection:
    """
    Global Redis connection singleton.

    This class manages a single Redis connection that is shared across
    all persistence operations. Configure it once at application startup.

    Usage:
        # At application startup
        RedisConnection.configure("redis://localhost:6379/0")

        # Get connection anywhere in the application
        redis = RedisConnection.get()

        # At shutdown
        RedisConnection.close()
    """

    _instance: "Redis | None" = None
    _lock = threading.Lock()

    @classmethod
    def configure(cls, url: str = "redis://localhost:6379", **kwargs) -> None:
        """
        Configure the Redis connection.

        Should be called once at application startup before any
        persistence operations.

        Args:
            url: Redis connection URL (e.g., "redis://localhost:6379/0")
            **kwargs: Additional arguments passed to Redis.from_url()
        """
        from redis import Redis

        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
            # decode_responses=True ensures all values are returned as strings
            cls._instance = Redis.from_url(url, decode_responses=True, **kwargs)

    @classmethod
    def get(cls) -> "Redis":
        """
        Get the Redis connection instance.

        Returns:
            The configured Redis connection

        Raises:
            RuntimeError: If configure() has not been called
        """
        if cls._instance is None:
            raise RuntimeError(
                "Redis connection not configured. "
                "Call RedisConnection.configure(url) first."
            )
        return cls._instance

    @classmethod
    def is_configured(cls) -> bool:
        """Check if Redis connection is configured."""
        return cls._instance is not None

    @classmethod
    def close(cls) -> None:
        """Close the Redis connection and reset the singleton."""
        with cls._lock:
            if cls._instance:
                cls._instance.close()
                cls._instance = None

    @classmethod
    def reset(cls) -> None:
        """Alias for close(), useful for testing."""
        cls.close()
