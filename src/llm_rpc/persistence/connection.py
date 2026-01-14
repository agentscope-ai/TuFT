"""Global Redis connection management."""

from __future__ import annotations

import os
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
    _url: str | None = None
    _kwargs: dict = {}
    _pid: int | None = None  # Track the PID to detect fork

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
        with cls._lock:
            cls._url = url
            cls._kwargs = kwargs
            # Close existing connection if any
            if cls._instance is not None:
                cls._instance.close()
                cls._instance = None
            cls._pid = None

    @classmethod
    def _create_connection(cls) -> "Redis":
        """Create a new Redis connection instance."""
        from redis import Redis

        return Redis.from_url(cls._url, decode_responses=True, **cls._kwargs)  # type: ignore[arg-type]

    @classmethod
    def get(cls) -> "Redis":
        """
        Get the Redis connection instance.

        This method is fork-safe: if the current process is a fork of the
        original process that created the connection, a new connection will
        be created for this process.

        Returns:
            The configured Redis connection

        Raises:
            RuntimeError: If configure() has not been called
        """
        if cls._url is None:
            raise RuntimeError(
                "Redis connection not configured. " "Call RedisConnection.configure(url) first."
            )

        current_pid = os.getpid()

        # Check if we're in a forked process or need to create connection
        if cls._instance is None or cls._pid != current_pid:
            with cls._lock:
                # Double-check after acquiring lock
                if cls._instance is None or cls._pid != current_pid:
                    if cls._instance is not None:
                        # Close stale connection from parent process
                        try:
                            cls._instance.close()
                        except Exception:
                            pass
                    cls._instance = cls._create_connection()
                    cls._pid = current_pid

        return cls._instance

    @classmethod
    def is_configured(cls) -> bool:
        """Check if Redis connection is configured."""
        return cls._url is not None

    @classmethod
    def close(cls) -> None:
        """Close the Redis connection and reset the singleton."""
        with cls._lock:
            if cls._instance:
                try:
                    cls._instance.close()
                except Exception:
                    pass
                cls._instance = None
            cls._pid = None

    @classmethod
    def reset(cls) -> None:
        """Close connection and clear configuration, useful for testing."""
        with cls._lock:
            if cls._instance:
                try:
                    cls._instance.close()
                except Exception:
                    pass
                cls._instance = None
            cls._url = None
            cls._kwargs = {}
            cls._pid = None
