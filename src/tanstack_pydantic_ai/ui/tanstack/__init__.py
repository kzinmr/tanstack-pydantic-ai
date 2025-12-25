"""
TanStack AI protocol adapter for pydantic-ai.

This module provides UIAdapter and UIEventStream implementations
for the TanStack AI protocol, following the pydantic-ai UI adapter pattern.

Usage:
    ```python
    from pydantic_ai import Agent
    from tanstack_pydantic_ai.ui.tanstack import TanStackAIAdapter

    agent = Agent(...)

    # From FastAPI request
    @app.post("/api/chat")
    async def chat(request: Request):
        adapter = TanStackAIAdapter.from_request(agent, await request.body())
        return StreamingResponse(
            adapter.streaming_response(),
            headers=adapter.response_headers,
        )

    # Manual usage
    adapter = TanStackAIAdapter(agent=agent, run_input=request_data)
    async for chunk in adapter.run_stream():
        print(chunk)
    ```
"""

from ._adapter import TanStackAIAdapter
from ._event_stream import TanStackEventStream
from .request_types import RequestData, ToolCallFunction, ToolCallPart, UIMessage

__all__ = [
    "TanStackAIAdapter",
    "TanStackEventStream",
    "RequestData",
    "UIMessage",
    "ToolCallPart",
    "ToolCallFunction",
]
