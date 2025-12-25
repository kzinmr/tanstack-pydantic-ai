"""
Request types for TanStack AI protocol.

Defines the input format expected from TanStack AI frontends.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class UIMessage(BaseModel):
    """A message in the TanStack AI format."""

    role: Literal["user", "assistant", "system", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    toolCalls: Optional[List[ToolCallPart]] = None
    toolCallId: Optional[str] = None


class ToolCallFunction(BaseModel):
    """Function details in a tool call."""

    name: str
    arguments: str  # JSON string


class ToolCallPart(BaseModel):
    """A tool call in the TanStack AI format."""

    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class RequestData(BaseModel):
    """
    Request data from TanStack AI frontend.

    This is the protocol-specific input format that the adapter
    transforms into pydantic-ai arguments.
    """

    messages: List[UIMessage]
    model: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    # For continuation after deferred tools
    run_id: Optional[str] = None
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    approvals: Dict[str, Union[bool, Dict[str, Any]]] = Field(default_factory=dict)
