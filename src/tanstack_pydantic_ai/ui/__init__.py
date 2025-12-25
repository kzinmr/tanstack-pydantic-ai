"""
UI adapters for various frontend protocols.

Currently supports:
- tanstack: TanStack AI protocol adapter
"""

from .tanstack import (
    RequestData,
    TanStackAIAdapter,
    TanStackEventStream,
    ToolCallFunction,
    ToolCallPart,
    UIMessage,
)

__all__ = [
    "RequestData",
    "TanStackAIAdapter",
    "TanStackEventStream",
    "ToolCallFunction",
    "ToolCallPart",
    "UIMessage",
]
