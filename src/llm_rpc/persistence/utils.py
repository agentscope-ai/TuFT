"""Utility functions for Redis persistence."""

from __future__ import annotations

import urllib.parse


def escape_key(key: str) -> str:
    """
    Escape dict key to prevent conflicts with Redis key delimiter '::'.

    Uses URL encoding to escape special characters.
    - ':' becomes '%3A'
    - '::' becomes '%3A%3A'
    - Other special characters are also escaped

    Args:
        key: The original dict key

    Returns:
        URL-encoded key safe for use in Redis key paths
    """
    return urllib.parse.quote(str(key), safe="")


def unescape_key(escaped: str) -> str:
    """
    Unescape a previously escaped key.

    Args:
        escaped: The URL-encoded key

    Returns:
        The original unescaped key
    """
    return urllib.parse.unquote(escaped)


def build_redis_key(
    namespace: str,
    class_name: str,
    instance_id: str | None,
    *parts: str,
) -> str:
    """
    Build a Redis key with proper formatting.

    Key format:
    - Without instance_id: {namespace}::{class_name}::{part1}::{part2}::...
    - With instance_id: {namespace}::{class_name}::{instance_id}::{part1}::...

    Args:
        namespace: The key namespace prefix
        class_name: The name of the class
        instance_id: Optional instance identifier for multi-instance deployment
        *parts: Additional key parts (field name, escaped keys, etc.)

    Returns:
        Properly formatted Redis key
    """
    components = [namespace, class_name]
    if instance_id:
        components.append(instance_id)
    components.extend(parts)
    return "::".join(components)
