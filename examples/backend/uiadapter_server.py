"""
Example FastAPI server using TanStackAIAdapter with stateful continuation.

This demonstrates the UIAdapter-based approach for TanStack AI integration,
following the pydantic-ai UIAdapter pattern with stateful HITL support.

Stateful Continuation:
    - /api/chat saves message history to store after each run
    - /api/chat/continue loads history from store using run_id
    - Clients only need to send: {run_id, tool_results, approvals}

Usage:
    uvicorn examples.backend.uiadapter_server:app --reload
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent

from tanstack_pydantic_ai import InMemoryRunStore, TanStackAIAdapter

# Create your agent
# In production, configure with your actual model
agent: Agent = Agent("openai:gpt-4o-mini")

# Stateful store for HITL continuation
# In production, use Redis or database-backed store
store = InMemoryRunStore()


app = FastAPI(title="TanStack AI + pydantic-ai (UIAdapter)")


@app.post("/api/chat")
async def chat(request: Request):
    """
    Main chat endpoint using UIAdapter pattern.

    The TanStackAIAdapter handles:
    - Request parsing (auto-generates run_id if not provided)
    - Message format conversion
    - Agent execution
    - Event stream transformation
    - SSE encoding
    - Saves message history to store for continuation

    Response chunks include `id` field which is the run_id.
    Use this id for /api/chat/continue requests.
    """
    adapter = TanStackAIAdapter.from_request(
        agent=agent,
        body=await request.body(),
        accept=request.headers.get("accept"),
        store=store,  # Enable stateful continuation
    )

    return StreamingResponse(
        adapter.streaming_response(),
        headers=dict(adapter.response_headers),
    )


@app.post("/api/chat/continue")
async def chat_continue(request: Request):
    """
    Continue a chat after deferred tool execution.

    Request body:
        {
            "run_id": "<id from previous response chunks>",
            "tool_results": {"<toolCallId>": <result>},
            "approvals": {"<toolCallId>": true/false},
            "messages": []  // Optional, ignored in stateful mode
        }

    The adapter:
    - Loads message history from store using run_id
    - Applies tool_results and approvals
    - Continues the agent run
    - Saves updated history to store
    """
    try:
        adapter = TanStackAIAdapter.from_request(
            agent=agent,
            body=await request.body(),
            accept=request.headers.get("accept"),
            store=store,
        )

        if not adapter.is_continuation:
            raise HTTPException(
                status_code=400,
                detail="Not a continuation request. Use /api/chat for new conversations.",
            )

        return StreamingResponse(
            adapter.streaming_response(),
            headers=dict(adapter.response_headers),
        )
    except ValueError as e:
        # Unknown run_id
        raise HTTPException(status_code=404, detail=str(e))


# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}
