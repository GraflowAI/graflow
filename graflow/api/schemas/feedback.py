"""Pydantic schemas for Feedback API endpoints."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class FeedbackResponseRequest(BaseModel):
    """Request body for providing feedback response."""

    # Approval
    approved: Optional[bool] = Field(None, description="Approval decision (true/false)")
    reason: Optional[str] = Field(None, description="Reason for approval/rejection")

    # Text
    text: Optional[str] = Field(None, description="Free-form text input")

    # Selection
    selected: Optional[str] = Field(None, description="Selected option (single selection)")
    selected_multiple: Optional[list[str]] = Field(None, description="Selected options (multiple selection)")

    # Custom
    custom_data: Optional[dict[str, Any]] = Field(None, description="Custom feedback data")

    # Metadata
    responded_by: Optional[str] = Field(None, description="User ID or identifier of respondent")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "approved": True,
                    "reason": "Approved by manager - deployment looks good",
                    "responded_by": "alice@example.com",
                },
                {"text": "Please fix the typos in section 3 before proceeding", "responded_by": "bob@example.com"},
                {"selected": "option_b", "responded_by": "charlie@example.com"},
            ]
        }
    }


class FeedbackRequestResponse(BaseModel):
    """Response model for feedback request details."""

    feedback_id: str = Field(..., description="Unique feedback request ID")
    task_id: str = Field(..., description="Task ID that requested feedback")
    session_id: str = Field(..., description="Workflow session ID")
    feedback_type: str = Field(..., description="Type of feedback (approval, text, selection, etc.)")
    prompt: str = Field(..., description="Prompt displayed to human")
    options: Optional[list[str]] = Field(None, description="Available options for selection types")
    status: str = Field(..., description="Request status (pending, completed, timeout, cancelled)")
    created_at: str = Field(..., description="ISO 8601 timestamp of request creation")
    timeout: float = Field(..., description="Polling timeout in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
    channel_key: Optional[str] = Field(None, description="Channel key for response (if using channel integration)")
    write_to_channel: bool = Field(False, description="Whether response will be written to channel")

    model_config = {
        "json_schema_extra": {
            "example": {
                "feedback_id": "deploy_task_abc12345",
                "task_id": "deploy_task",
                "session_id": "session_67890",
                "feedback_type": "approval",
                "prompt": "Approve deployment to production?",
                "options": None,
                "status": "pending",
                "created_at": "2025-01-28T10:00:00Z",
                "timeout": 180.0,
                "metadata": {"environment": "production"},
                "channel_key": None,
                "write_to_channel": False,
            }
        }
    }


class FeedbackResponseDetails(BaseModel):
    """Response model for feedback response details."""

    feedback_id: str = Field(..., description="Unique feedback request ID")
    response_type: str = Field(..., description="Type of response")
    approved: Optional[bool] = Field(None, description="Approval decision")
    reason: Optional[str] = Field(None, description="Reason for decision")
    text: Optional[str] = Field(None, description="Text response")
    selected: Optional[str] = Field(None, description="Selected option")
    selected_multiple: Optional[list[str]] = Field(None, description="Multiple selections")
    custom_data: Optional[dict[str, Any]] = Field(None, description="Custom response data")
    responded_at: str = Field(..., description="ISO 8601 timestamp of response")
    responded_by: Optional[str] = Field(None, description="User who responded")

    model_config = {
        "json_schema_extra": {
            "example": {
                "feedback_id": "deploy_task_abc12345",
                "response_type": "approval",
                "approved": True,
                "reason": "Approved by manager",
                "text": None,
                "selected": None,
                "selected_multiple": None,
                "custom_data": None,
                "responded_at": "2025-01-28T10:02:30Z",
                "responded_by": "alice@example.com",
            }
        }
    }


class FeedbackDetailResponse(BaseModel):
    """Response model combining request and response details."""

    request: FeedbackRequestResponse = Field(..., description="Feedback request details")
    response: Optional[FeedbackResponseDetails] = Field(None, description="Feedback response (if provided)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "request": {
                    "feedback_id": "deploy_task_abc12345",
                    "task_id": "deploy_task",
                    "session_id": "session_67890",
                    "feedback_type": "approval",
                    "prompt": "Approve deployment to production?",
                    "options": None,
                    "status": "completed",
                    "created_at": "2025-01-28T10:00:00Z",
                    "timeout": 180.0,
                    "metadata": {},
                    "channel_key": None,
                    "write_to_channel": False,
                },
                "response": {
                    "feedback_id": "deploy_task_abc12345",
                    "response_type": "approval",
                    "approved": True,
                    "reason": "Approved by manager",
                    "text": None,
                    "selected": None,
                    "selected_multiple": None,
                    "custom_data": None,
                    "responded_at": "2025-01-28T10:02:30Z",
                    "responded_by": "alice@example.com",
                },
            }
        }
    }


class MessageResponse(BaseModel):
    """Generic message response for success/error messages."""

    message: str = Field(..., description="Response message")
    feedback_id: Optional[str] = Field(None, description="Feedback ID (if applicable)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "Feedback provided successfully", "feedback_id": "deploy_task_abc12345"},
                {"message": "Feedback request cancelled", "feedback_id": "deploy_task_abc12345"},
            ]
        }
    }


class PendingFeedbackListResponse(BaseModel):
    """Response model for listing pending feedback requests."""

    count: int = Field(..., description="Number of pending requests")
    requests: list[FeedbackRequestResponse] = Field(..., description="List of pending feedback requests")

    model_config = {
        "json_schema_extra": {
            "example": {
                "count": 2,
                "requests": [
                    {
                        "feedback_id": "deploy_task_abc12345",
                        "task_id": "deploy_task",
                        "session_id": "session_67890",
                        "feedback_type": "approval",
                        "prompt": "Approve deployment?",
                        "options": None,
                        "status": "pending",
                        "created_at": "2025-01-28T10:00:00Z",
                        "timeout": 180.0,
                        "metadata": {},
                        "channel_key": None,
                        "write_to_channel": False,
                    }
                ],
            }
        }
    }
