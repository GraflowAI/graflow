"""Data types for coordination."""

import json
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class SerializedTaskRecord:
    """Redis persistent TaskRecord (Content-Addressable version)."""

    # Task identification
    task_id: str
    session_id: str  # Workflow instance ID
    graph_hash: str  # Content-Addressable key
    trace_id: str

    # Metadata
    created_at: float

    # Distributed tracing
    group_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "SerializedTaskRecord":
        """Deserialize from JSON string."""
        return cls(**json.loads(data))
