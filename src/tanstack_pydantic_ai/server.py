from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from .chunks import (
    ApprovalObj,
    ApprovalRequestedStreamChunk,
    ContentStreamChunk,
    DoneStreamChunk,
    ErrorObj,
    ErrorStreamChunk,
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
        index = 0
        for part in pending.approvals:
            tool_call_id = part.tool_call_id
            tool_name = part.tool_name
            args = part.args
            yield sse_data(
                dump_chunk(
                    ToolCallStreamChunk(
                        id=run_id,
                        model=model_name,
                        timestamp=now_ms(),
                        index=index,
                        toolCall=ToolCall(
                            id=tool_call_id,
                            function=ToolCallFunction(
                                name=tool_name,
                                arguments=json.dumps(args, ensure_ascii=False),
                            ),
                        ),
                    )
                )
            )
            index += 1
            yield sse_data(
                dump_chunk(
                    ApprovalRequestedStreamChunk(
                        id=run_id,
                        model=model_name,
                        timestamp=now_ms(),
                        toolCallId=tool_call_id,
                        toolName=tool_name,
                        input=args,
                        approval=ApprovalObj(id=uuid.uuid4().hex),
                    )
                )
            )

        for part in pending.calls:
            tool_call_id = part.tool_call_id
            tool_name = part.tool_name
            args = part.args
            yield sse_data(
                dump_chunk(
                    ToolCallStreamChunk(
                        id=run_id,
                        model=model_name,
                        timestamp=now_ms(),
                        index=index,
                        toolCall=ToolCall(
                            id=tool_call_id,
                            function=ToolCallFunction(
                                name=tool_name,
                                arguments=json.dumps(args, ensure_ascii=False),
                            ),
                        ),
                    )
                )
            )
            index += 1
            yield sse_data(
                dump_chunk(
                    ToolInputAvailableStreamChunk(
                        id=run_id,
                        model=model_name,
                        timestamp=now_ms(),
                        toolCallId=tool_call_id,
                        toolName=tool_name,
                        input=args,
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
            prev_text = ""
            try:
                async with agent.run_stream(
                    user_prompt,
                    **run_stream_kwargs(history, model_override),
                ) as response:
                    async for text in response.stream_text(delta=False):
                        delta = text[len(prev_text):] if text.startswith(prev_text) else text
                        prev_text = text
                        chunk = ContentStreamChunk(
                            id=run_id,
                            model=model_name,
                            timestamp=now_ms(),
                            content=text,
                            delta=delta,
                            role="assistant",
                        )
                        yield sse_data(dump_chunk(chunk))

                    output = await response.get_output()
                    run_store.set_messages(run_id, response.all_messages(), model_override)

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
        state = run_store.get(req.run_id)
        if state is None or state.pending is None:
            raise HTTPException(status_code=404, detail="unknown run_id or nothing pending")

        model_override = state.model
        model_name = model_override or "unknown"
        history = state.messages

        deferred = DeferredToolResults(
            approvals=req.approvals,
            calls=req.tool_results,
        )

        async def gen():
            prev_text = ""
            try:
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

                async with agent.run_stream(
                    "",
                    deferred_tool_results=deferred,
                    **run_stream_kwargs(history, model_override),
                ) as response:
                    async for text in response.stream_text(delta=False):
                        delta = text[len(prev_text):] if text.startswith(prev_text) else text
                        prev_text = text
                        yield sse_data(
                            dump_chunk(
                                ContentStreamChunk(
                                    id=req.run_id,
                                    model=model_name,
                                    timestamp=now_ms(),
                                    content=text,
                                    delta=delta,
                                    role="assistant",
                                )
                            )
                        )

                    output = await response.get_output()
                    run_store.set_messages(req.run_id, response.all_messages(), model_override)

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
