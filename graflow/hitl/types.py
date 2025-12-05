"""Core types for Human-in-the-Loop functionality."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class FeedbackType(Enum):
    """Types of feedback that can be requested."""

    APPROVAL = "approval"  # Boolean approval (approved/rejected)
    TEXT = "text"  # Free-form text input
    SELECTION = "selection"  # Single selection from options
    MULTI_SELECTION = "multi_selection"  # Multiple selections
    CUSTOM = "custom"  # Custom feedback structure


class FeedbackRequest(BaseModel):
    """Represents a feedback request."""

    feedback_id: str  # Unique request ID
    task_id: str  # Task requesting feedback
    session_id: str  # Workflow session ID
    feedback_type: FeedbackType  # Type of feedback
    prompt: str  # Prompt for human
    options: Optional[list[str]] = None  # Options for selection types
    metadata: dict[str, Any] = Field(default_factory=dict)  # Custom metadata
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    timeout: float = 180.0  # Polling timeout in seconds (default: 3 minutes)
    status: str = "pending"  # pending, completed, timeout, cancelled

    # Channel integration
    channel_key: Optional[str] = None  # Channel key to write response to
    write_to_channel: bool = False  # Whether to auto-write to channel

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        data = self.model_dump()
        data["feedback_type"] = self.feedback_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackRequest:
        """Restore from dictionary."""
        data = data.copy()
        if isinstance(data.get("feedback_type"), str):
            data["feedback_type"] = FeedbackType(data["feedback_type"])
        return cls(**data)


class FeedbackResponse(BaseModel):
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
    responded_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    responded_by: Optional[str] = None  # User ID or system identifier

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        data = self.model_dump()
        data["response_type"] = self.response_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackResponse:
        """Restore from dictionary."""
        data = data.copy()
        if isinstance(data.get("response_type"), str):
            data["response_type"] = FeedbackType(data["response_type"])
        return cls(**data)


class FeedbackTimeoutError(Exception):
    """Raised when feedback request times out."""

    def __init__(self, feedback_id: str, timeout: float):
        self.feedback_id = feedback_id
        self.timeout = timeout
        super().__init__(f"Feedback request {feedback_id} timed out after {timeout} seconds")

    def __reduce__(self) -> tuple[type[FeedbackTimeoutError], tuple[str, float]]:
        """Allow pickling of this exception."""
        return (self.__class__, (self.feedback_id, self.timeout))
