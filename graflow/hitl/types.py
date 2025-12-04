"""Core types for Human-in-the-Loop functionality."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class FeedbackType(Enum):
    """Types of feedback that can be requested."""

    APPROVAL = "approval"  # Boolean approval (approved/rejected)
    TEXT = "text"  # Free-form text input
    SELECTION = "selection"  # Single selection from options
    MULTI_SELECTION = "multi_selection"  # Multiple selections
    CUSTOM = "custom"  # Custom feedback structure


@dataclass
class FeedbackRequest:
    """Represents a feedback request."""

    feedback_id: str  # Unique request ID
    task_id: str  # Task requesting feedback
    session_id: str  # Workflow session ID
    feedback_type: FeedbackType  # Type of feedback
    prompt: str  # Prompt for human
    options: Optional[list[str]] = None  # Options for selection types
    metadata: dict[str, Any] = field(default_factory=dict)  # Custom metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    timeout: float = 180.0  # Polling timeout in seconds (default: 3 minutes)
    status: str = "pending"  # pending, completed, timeout, cancelled

    # Channel integration
    channel_key: Optional[str] = None  # Channel key to write response to
    write_to_channel: bool = False  # Whether to auto-write to channel

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "feedback_id": self.feedback_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "feedback_type": self.feedback_type.value,
            "prompt": self.prompt,
            "options": self.options,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "timeout": self.timeout,
            "status": self.status,
            "channel_key": self.channel_key,
            "write_to_channel": self.write_to_channel,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackRequest:
        """Restore from dictionary."""
        data = data.copy()
        data["feedback_type"] = FeedbackType(data["feedback_type"])
        return cls(**data)


@dataclass
class FeedbackResponse:
    """Represents a feedback response."""

    feedback_id: str  # Request ID
    response_type: FeedbackType  # Type of response

    # Approval responses
    approved: Optional[bool] = None
    reason: Optional[str] = None

    # Text responses
    text: Optional[str] = None

    # Selection responses
    selected: Optional[str] = None
    selected_multiple: Optional[list[str]] = None

    # Custom responses
    custom_data: Optional[dict[str, Any]] = None

    # Metadata
    responded_at: str = field(default_factory=lambda: datetime.now().isoformat())
    responded_by: Optional[str] = None  # User ID or system identifier

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "feedback_id": self.feedback_id,
            "response_type": self.response_type.value,
            "approved": self.approved,
            "reason": self.reason,
            "text": self.text,
            "selected": self.selected,
            "selected_multiple": self.selected_multiple,
            "custom_data": self.custom_data,
            "responded_at": self.responded_at,
            "responded_by": self.responded_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackResponse:
        """Restore from dictionary."""
        data = data.copy()
        data["response_type"] = FeedbackType(data["response_type"])
        return cls(**data)


class FeedbackTimeoutError(Exception):
    """Raised when feedback request times out."""

    def __init__(self, feedback_id: str, timeout: float):
        self.feedback_id = feedback_id
        self.timeout = timeout
        super().__init__(
            f"Feedback request {feedback_id} timed out after {timeout} seconds"
        )

    def __reduce__(self):
        """Allow pickling of this exception."""
        return (self.__class__, (self.feedback_id, self.timeout))
