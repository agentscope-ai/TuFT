"""LLM-RPC service package."""

from .server import create_root_app

__all__ = ["create_root_app"]
