"""
FastAPI server for TanStack AI integration with pydantic-ai.

This module provides HTTP endpoints for chat streaming using pydantic-ai agents.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults

from tanstack_pydantic_ai import (
    InMemoryRunStore,
    ToolResultStreamChunk,
    dump_chunk,
    encode_done,
    now_ms,
    sse_data,
)
from tanstack_pydantic_ai.streaming import stream_chat, stream_continue


class ChatMessage(BaseModel):
    """Single chat message with role and content."""

    role: Literal["user", "assistant"]
    content: str


class ChatStartRequest(BaseModel):
    """Request to start a new chat stream."""

    model: Optional[str] = None
    messages: List[ChatMessage]


class ContinueRequest(BaseModel):
    """Request to continue a chat after deferred tool execution."""

    run_id: str
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    approvals: Dict[str, Union[bool, Dict[str, Any]]] = Field(default_factory=dict)


def create_app(
    agent: Agent,
    *,
    store: Optional[InMemoryRunStore] = None,
    model_default: Optional[str] = None,
) -> FastAPI:
    """
    Create a FastAPI app with chat streaming endpoints.

    Args:
        agent: The pydantic-ai Agent instance
        store: Optional run store for managing conversation state
        model_default: Optional default model name

    Returns:
        FastAPI application with /chat/stream and /chat/continue endpoints
    """
    run_store = store or InMemoryRunStore()
    app = FastAPI()

    @app.post("/chat/stream")
    @app.post("/chat")
    async def chat_stream(req: ChatStartRequest):
        """Stream a chat response from the agent."""
        if not req.messages or req.messages[-1].role != "user":
            raise HTTPException(
                status_code=400, detail="messages must end with a user message"
            )

        run_id = uuid.uuid4().hex
        model_override = req.model or model_default
        model_name = model_override or "unknown"

        # Convert ChatMessage list to tuple format for build_message_history
        history_tuples = [
            (msg.role, msg.content) for msg in req.messages[:-1]
        ]
        user_prompt = req.messages[-1].content

        async def gen():
            from tanstack_pydantic_ai.streaming import build_message_history

            history = build_message_history(history_tuples)

            stream = stream_chat(
                agent,
                user_prompt,
                message_history=history,
                model=model_override,
                run_id=run_id,
            )

            async for chunk in stream:
                yield sse_data(dump_chunk(chunk))

            # Get result after streaming completes
            try:
                result = await stream.result()
                run_store.set_messages(run_id, result.all_messages(), model_override)

                if isinstance(result.output, DeferredToolRequests):
                    run_store.set_pending(run_id, result.output, model_override)
                else:
                    run_store.set_pending(run_id, None, model_override)
            except Exception:
                # Error was already yielded in stream
                pass

            yield encode_done()

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/chat/continue")
    async def chat_continue(req: ContinueRequest):
        """Continue a chat after deferred tool execution."""
        run_state = run_store.get(req.run_id)
        if run_state is None or run_state.pending is None:
            raise HTTPException(
                status_code=404, detail="unknown run_id or nothing pending"
            )

        model_override = run_state.model
        model_name = model_override or "unknown"
        history = run_state.messages

        deferred = DeferredToolResults(
            approvals=req.approvals,
            calls=req.tool_results,
        )

        async def gen():
            # Emit tool results from client before resuming
            for tool_call_id, value in req.tool_results.items():
                yield sse_data(
                    dump_chunk(
                        ToolResultStreamChunk(
                            id=req.run_id,
                            model=model_name,
                            timestamp=now_ms(),
                            toolCallId=tool_call_id,
                            content=json.dumps(value, ensure_ascii=False),
                        )
                    )
                )

            stream = stream_continue(
                agent,
                deferred,
                message_history=history,
                model=model_override,
                run_id=req.run_id,
            )

            async for chunk in stream:
                yield sse_data(dump_chunk(chunk))

            # Get result after streaming completes
            try:
                result = await stream.result()
                run_store.set_messages(
                    req.run_id, result.all_messages(), model_override
                )

                if isinstance(result.output, DeferredToolRequests):
                    run_store.set_pending(req.run_id, result.output, model_override)
                else:
                    run_store.set_pending(req.run_id, None, model_override)
            except Exception:
                # Error was already yielded in stream
                pass

            yield encode_done()

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
