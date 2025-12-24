# tanstack-pydantic-ai

TanStack AI-compatible SSE backend scaffolding built on pydantic-ai and FastAPI.

## Quick start

```python
from fastapi import FastAPI
from pydantic_ai import Agent, CallDeferred

from tanstack_pydantic_ai import create_app

agent = Agent("openai:gpt-5-mini", output_type=[str])

# Optional: example deferred tool for client-side execution
@agent.tool
async def client_side_search(query: str) -> str:
    raise CallDeferred(metadata={"kind": "client_side_search"})

app: FastAPI = create_app(agent)
```

Run:

```sh
uvicorn your_module:app --reload
```

## Demo + verification

To run the bundled demo agent and confirm SSE parsing with TanStack AI v0.2.0:

1. Start the FastAPI demo server (uses the fake `DemoAgent` that emits deterministic text).

```sh
uv run uvicorn examples.backend.demo_server:app --host 127.0.0.1 --port 8000 &
```

2. In another shell, run the frontend verifier (uses `StreamProcessor` to consume SSE).

```sh
cd examples/frontend
npm run verify
```

3. After verification, stop the server (`kill %1` if backgrounded from step 1).

## Endpoints

- `POST /chat/stream` (alias: `POST /chat`)
  - Request: `{ "model"?: string, "messages": [{ "role": "user" | "assistant", "content": string }] }`
  - Response: `text/event-stream` of StreamChunk JSON frames plus `data: [DONE]`
- `POST /chat/continue`
  - Request: `{ "run_id": string, "tool_results"?: Record<string, any>, "approvals"?: Record<string, boolean | object> }`

## Notes

- [StreamChunk](https://tanstack.com/ai/latest/docs/reference/type-aliases/StreamChunk)
- Tool approvals and external execution use pydantic-ai [Deferred Tools](https://ai.pydantic.dev/deferred-tools/).
