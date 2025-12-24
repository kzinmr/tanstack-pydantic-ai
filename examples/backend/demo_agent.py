from __future__ import annotations

import asyncio
from typing import AsyncGenerator, List, Optional, Sequence

from pydantic_ai import DeferredToolRequests, ModelMessage, ModelResponse, TextPart


class DemoStream:
    def __init__(self, prompt: str, texts: Sequence[str]) -> None:
        self._prompt = prompt
        self._texts = texts
        self._messages: List[ModelMessage] = [
            ModelResponse([TextPart(content=texts[-1])])
        ]

    async def __aenter__(self) -> "DemoStream":
        return self

    async def __aexit__(self, *_) -> None:  # pragma: no cover
        return None

    async def stream_text(self, delta: bool = False) -> AsyncGenerator[str, None]:
        for text in self._texts:
            yield text
            await asyncio.sleep(0)

    async def get_output(self) -> Optional[DeferredToolRequests]:
        return None

    def all_messages(self) -> List[ModelMessage]:
        return self._messages


class DemoAgent:
    def __init__(self, *, model_name: str = "demo:assistant") -> None:
        self.model_name = model_name

    def run_stream(self, prompt: str, **kwargs: object) -> DemoStream:
        texts = [
            "Hello from fake AI agent.",
            f"Prompt received: {prompt}",
        ]
        return DemoStream(prompt, texts)
