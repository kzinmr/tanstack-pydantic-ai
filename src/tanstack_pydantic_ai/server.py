from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    DeferredToolRequests,
    DeferredToolResults,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)

from .chunks import (
    ApprovalObj,
    ApprovalRequestedStreamChunk,
    ContentStreamChunk,
    DoneStreamChunk,
    ErrorObj,
    ErrorStreamChunk,
    StreamChunk,
    ThinkingStreamChunk,
    ToolCall,
    ToolCallFunction,
    ToolCallStreamChunk,
    ToolInputAvailableStreamChunk,
    ToolResultStreamChunk,
)
from .sse import dump_chunk, encode_done, now_ms, sse_data
from .store import InMemoryRunStore


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatStartRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]


class ContinueRequest(BaseModel):
    run_id: str
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    approvals: Dict[str, Union[bool, Dict[str, Any]]] = Field(default_factory=dict)


def build_message_history(msgs: List[ChatMessage]) -> List[ModelMessage]:
    history: List[ModelMessage] = []
    for msg in msgs:
        if msg.role == "user":
            history.append(ModelRequest([UserPromptPart(content=msg.content)]))
        else:
            history.append(ModelResponse([TextPart(content=msg.content)]))
    return history


class StreamState:
    """Mutable state for tracking stream accumulation."""

    def __init__(self, run_id: str, model_name: str):
        self.run_id = run_id
        self.model_name = model_name
        self.text_accumulator: Dict[int, str] = {}
        self.thinking_accumulator: Dict[int, str] = {}
        self.tool_call_index = 0


def handle_stream_event(
    event: Any,
    state: StreamState,
) -> List[StreamChunk]:
    """
    Map pydantic-ai stream events to TanStack StreamChunks.

    Handles:
    - PartStartEvent / PartDeltaEvent for text and thinking content
    - FunctionToolCallEvent for tool calls
    - FunctionToolResultEvent for tool results
    """
    chunks: List[StreamChunk] = []
    ts = now_ms()
    run_id = state.run_id
    model_name = state.model_name

    # Handle PartStartEvent - initialize accumulators with initial content
    if isinstance(event, PartStartEvent):
        if isinstance(event.part, TextPart):
            state.text_accumulator[event.index] = event.part.content
            if event.part.content:
                chunks.append(
                    ContentStreamChunk(
                        id=run_id,
                        model=model_name,
                        timestamp=ts,
                        content=event.part.content,
                        delta=event.part.content,
                        role="assistant",
                    )
                )
        elif isinstance(event.part, ThinkingPart):
            state.thinking_accumulator[event.index] = event.part.content
            if event.part.content:
                chunks.append(
                    ThinkingStreamChunk(
                        id=run_id,
                        model=model_name,
                        timestamp=ts,
                        content=event.part.content,
                        delta=event.part.content,
                    )
                )

    # Handle PartDeltaEvent - accumulate text/thinking deltas
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta):
            delta_text = event.delta.content_delta
            idx = event.index
            prev = state.text_accumulator.get(idx, "")
            new_content = prev + delta_text
            state.text_accumulator[idx] = new_content
            chunks.append(
                ContentStreamChunk(
                    id=run_id,
                    model=model_name,
                    timestamp=ts,
                    content=new_content,
                    delta=delta_text,
                    role="assistant",
                )
            )
        elif isinstance(event.delta, ThinkingPartDelta):
            # ThinkingPartDelta may have None content_delta (signature-only)
            if event.delta.content_delta:
                delta_text = event.delta.content_delta
                idx = event.index
                prev = state.thinking_accumulator.get(idx, "")
                new_content = prev + delta_text
                state.thinking_accumulator[idx] = new_content
                chunks.append(
                    ThinkingStreamChunk(
                        id=run_id,
                        model=model_name,
                        timestamp=ts,
                        content=new_content,
                        delta=delta_text,
                    )
                )

    # Handle FunctionToolCallEvent - emit tool call chunk
    elif isinstance(event, FunctionToolCallEvent):
        part: ToolCallPart = event.part
        chunks.append(
            ToolCallStreamChunk(
                id=run_id,
                model=model_name,
                timestamp=ts,
                index=state.tool_call_index,
                toolCall=ToolCall(
                    id=part.tool_call_id,
                    function=ToolCallFunction(
                        name=part.tool_name,
                        arguments=json.dumps(
                            part.args if isinstance(part.args, dict) else part.args,
                            ensure_ascii=False,
                        ),
                    ),
                ),
            )
        )
        state.tool_call_index += 1

    # Handle FunctionToolResultEvent - emit tool result chunk
    elif isinstance(event, FunctionToolResultEvent):
        if isinstance(event.result, ToolReturnPart):
            content = event.result.content
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            chunks.append(
                ToolResultStreamChunk(
                    id=run_id,
                    model=model_name,
                    timestamp=ts,
                    toolCallId=event.result.tool_call_id,
                    content=content,
                )
            )

    return chunks


def create_app(
    agent: Agent,
    *,
    store: Optional[InMemoryRunStore] = None,
    model_default: Optional[str] = None,
) -> FastAPI:
    run_store = store or InMemoryRunStore()
    app = FastAPI()

    def emit_deferred_chunks(
        run_id: str,
        model_name: str,
        pending: DeferredToolRequests,
    ):
        """
        Emit approval/input-available chunks for deferred tool calls.

        Note: ToolCallStreamChunk is NOT emitted here because it was already
        emitted in real-time via FunctionToolCallEvent during streaming.
        """
        for part in pending.approvals:
            yield sse_data(
                dump_chunk(
                    ApprovalRequestedStreamChunk(
                        id=run_id,
                        model=model_name,
                        timestamp=now_ms(),
                        toolCallId=part.tool_call_id,
                        toolName=part.tool_name,
                        input=part.args,
                        approval=ApprovalObj(id=uuid.uuid4().hex),
                    )
                )
            )

        for part in pending.calls:
            yield sse_data(
                dump_chunk(
                    ToolInputAvailableStreamChunk(
                        id=run_id,
                        model=model_name,
                        timestamp=now_ms(),
                        toolCallId=part.tool_call_id,
                        toolName=part.tool_name,
                        input=part.args,
                    )
                )
            )

    def run_stream_kwargs(message_history: List[ModelMessage], model_override: Optional[str]):
        kwargs: Dict[str, Any] = {
            "message_history": message_history,
            "output_type": [str, DeferredToolRequests],
        }
        if model_override is not None:
            kwargs["model"] = model_override
        return kwargs

    @app.post("/chat/stream")
    @app.post("/chat")
    async def chat_stream(req: ChatStartRequest):
        if not req.messages or req.messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="messages must end with a user message")

        run_id = uuid.uuid4().hex
        model_override = req.model or model_default
        model_name = model_override or "unknown"

        history = build_message_history(req.messages[:-1])
        user_prompt = req.messages[-1].content

        async def gen():
            state = StreamState(run_id, model_name)
            result = None

            try:
                async for event in agent.run_stream_events(
                    user_prompt,
                    **run_stream_kwargs(history, model_override),
                ):
                    # Handle stream events and emit chunks
                    for chunk in handle_stream_event(event, state):
                        yield sse_data(dump_chunk(chunk))

                    # Capture the final result
                    if isinstance(event, AgentRunResultEvent):
                        result = event.result

                # After stream completes, process final result
                if result is not None:
                    run_store.set_messages(run_id, result.all_messages(), model_override)
                    output = result.output

                    if isinstance(output, DeferredToolRequests):
                        run_store.set_pending(run_id, output, model_override)
                        for payload in emit_deferred_chunks(run_id, model_name, output):
                            yield payload
                        yield sse_data(
                            dump_chunk(
                                DoneStreamChunk(
                                    id=run_id,
                                    model=model_name,
                                    timestamp=now_ms(),
                                    finishReason="tool_calls",
                                )
                            )
                        )
                    else:
                        run_store.set_pending(run_id, None, model_override)
                        yield sse_data(
                            dump_chunk(
                                DoneStreamChunk(
                                    id=run_id,
                                    model=model_name,
                                    timestamp=now_ms(),
                                    finishReason="stop",
                                )
                            )
                        )

            except Exception as exc:
                yield sse_data(
                    dump_chunk(
                        ErrorStreamChunk(
                            id=run_id,
                            model=model_name,
                            timestamp=now_ms(),
                            error=ErrorObj(message=str(exc)),
                        )
                    )
                )
                yield sse_data(
                    dump_chunk(
                        DoneStreamChunk(
                            id=run_id,
                            model=model_name,
                            timestamp=now_ms(),
                            finishReason="stop",
                        )
                    )
                )
            finally:
                yield encode_done()

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/chat/continue")
    async def chat_continue(req: ContinueRequest):
        run_state = run_store.get(req.run_id)
        if run_state is None or run_state.pending is None:
            raise HTTPException(status_code=404, detail="unknown run_id or nothing pending")

        model_override = run_state.model
        model_name = model_override or "unknown"
        history = run_state.messages

        deferred = DeferredToolResults(
            approvals=req.approvals,
            calls=req.tool_results,
        )

        async def gen():
            state = StreamState(req.run_id, model_name)
            result = None

            try:
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

                async for event in agent.run_stream_events(
                    "",
                    deferred_tool_results=deferred,
                    **run_stream_kwargs(history, model_override),
                ):
                    # Handle stream events and emit chunks
                    for chunk in handle_stream_event(event, state):
                        yield sse_data(dump_chunk(chunk))

                    # Capture the final result
                    if isinstance(event, AgentRunResultEvent):
                        result = event.result

                # After stream completes, process final result
                if result is not None:
                    run_store.set_messages(req.run_id, result.all_messages(), model_override)
                    output = result.output

                    if isinstance(output, DeferredToolRequests):
                        run_store.set_pending(req.run_id, output, model_override)
                        for payload in emit_deferred_chunks(req.run_id, model_name, output):
                            yield payload
                        yield sse_data(
                            dump_chunk(
                                DoneStreamChunk(
                                    id=req.run_id,
                                    model=model_name,
                                    timestamp=now_ms(),
                                    finishReason="tool_calls",
                                )
                            )
                        )
                    else:
                        run_store.set_pending(req.run_id, None, model_override)
                        yield sse_data(
                            dump_chunk(
                                DoneStreamChunk(
                                    id=req.run_id,
                                    model=model_name,
                                    timestamp=now_ms(),
                                    finishReason="stop",
                                )
                            )
                        )

            except Exception as exc:
                yield sse_data(
                    dump_chunk(
                        ErrorStreamChunk(
                            id=req.run_id,
                            model=model_name,
                            timestamp=now_ms(),
                            error=ErrorObj(message=str(exc)),
                        )
                    )
                )
                yield sse_data(
                    dump_chunk(
                        DoneStreamChunk(
                            id=req.run_id,
                            model=model_name,
                            timestamp=now_ms(),
                            finishReason="stop",
                        )
                    )
                )
            finally:
                yield encode_done()

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
