from __future__ import annotations

import json
import time

from .chunks import StreamChunk


def now_ms() -> int:
    return int(time.time() * 1000)


def sse_data(payload: str) -> str:
    return f"data: {payload}\n\n"


def dump_chunk(chunk: StreamChunk) -> str:
    return json.dumps(chunk.model_dump(exclude_none=True), ensure_ascii=False)


def encode_chunk(chunk: StreamChunk) -> str:
    return sse_data(dump_chunk(chunk))


def encode_done() -> str:
    return sse_data("[DONE]")
