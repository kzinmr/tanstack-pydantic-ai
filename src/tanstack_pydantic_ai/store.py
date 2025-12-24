from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pydantic_ai import DeferredToolRequests, ModelMessage


@dataclass
class RunState:
    messages: List[ModelMessage] = field(default_factory=list)
    pending: Optional[DeferredToolRequests] = None
    model: Optional[str] = None


class InMemoryRunStore:
    def __init__(self) -> None:
        self._runs: Dict[str, RunState] = {}

    def get(self, run_id: str) -> Optional[RunState]:
        return self._runs.get(run_id)

    def set(self, run_id: str, state: RunState) -> None:
        self._runs[run_id] = state

    def upsert(self, run_id: str, model: Optional[str]) -> RunState:
        state = self._runs.get(run_id)
        if state is None:
            state = RunState(model=model)
            self._runs[run_id] = state
        elif model is not None:
            state.model = model
        return state

    def set_messages(
        self,
        run_id: str,
        messages: List[ModelMessage],
        model: Optional[str],
    ) -> RunState:
        state = self.upsert(run_id, model)
        state.messages = messages
        return state

    def set_pending(
        self,
        run_id: str,
        pending: Optional[DeferredToolRequests],
        model: Optional[str],
    ) -> RunState:
        state = self.upsert(run_id, model)
        state.pending = pending
        return state

    def clear(self, run_id: str) -> None:
        self._runs.pop(run_id, None)
