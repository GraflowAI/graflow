from __future__ import annotations

from typing import Any, List, TypedDict

from typing_extensions import NotRequired


class AgentStep(TypedDict):
    """Single execution step emitted by an LLM agent."""

    type: str
    is_final: bool
    is_partial: bool
    content: NotRequired[List[str]]


class AgentResult(TypedDict):
    """Result of a synchronous LLM agent run."""

    output: Any  # str | BaseModel
    steps: List[AgentStep]
    metadata: dict[str, Any]
