"""
UI adapters for pydantic-ai integration.

This module provides UIAdapter implementations for different UI protocols.
"""

from .tanstack import TanStackAIAdapter, TanStackEventStream

__all__ = [
    "TanStackAIAdapter",
    "TanStackEventStream",
]
