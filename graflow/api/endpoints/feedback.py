"""Feedback API endpoints for Human-in-the-Loop functionality."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from graflow.api.schemas.feedback import (
    FeedbackDetailResponse,
    FeedbackRequestResponse,
    FeedbackResponseDetails,
    FeedbackResponseRequest,
    MessageResponse,
    PendingFeedbackListResponse,
)
from graflow.hitl.manager import FeedbackManager
from graflow.hitl.types import FeedbackResponse

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


def get_feedback_manager(request: Request) -> FeedbackManager:
    """Get FeedbackManager from app state.

    Args:
        request: FastAPI request object

    Returns:
        FeedbackManager instance

    Raises:
        HTTPException: If FeedbackManager not found in app state
    """
    feedback_manager = getattr(request.app.state, "feedback_manager", None)
    if not feedback_manager:
        raise HTTPException(
            status_code=500, detail="FeedbackManager not initialized. Did you use create_feedback_api()?"
        )
    return feedback_manager


@router.get(
    "",
    response_model=PendingFeedbackListResponse,
    summary="List pending feedback requests",
    description="""
List all pending feedback requests, optionally filtered by session ID.

Returns a list of feedback requests with status='pending' that are awaiting human response.
This endpoint is useful for building feedback dashboards or notification systems.
    """,
    responses={
        200: {
            "description": "List of pending feedback requests",
            "content": {
                "application/json": {
                    "example": {
                        "count": 2,
                        "requests": [
                            {
                                "feedback_id": "deploy_task_abc12345",
                                "task_id": "deploy_task",
                                "session_id": "session_67890",
                                "feedback_type": "approval",
                                "prompt": "Approve deployment to production?",
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
            },
        }
    },
)
async def list_pending_feedback(request: Request, session_id: Optional[str] = None) -> PendingFeedbackListResponse:
    """List pending feedback requests.

    Args:
        request: FastAPI request object
        session_id: Optional filter by session ID

    Returns:
        PendingFeedbackListResponse with count and list of requests
    """
    feedback_manager = get_feedback_manager(request)
    requests = feedback_manager.list_pending_requests(session_id)

    # Convert to response models
    response_requests = [FeedbackRequestResponse(**req.to_dict()) for req in requests]

    return PendingFeedbackListResponse(count=len(response_requests), requests=response_requests)


@router.get(
    "/{feedback_id}",
    response_model=FeedbackDetailResponse,
    summary="Get feedback details",
    description="""
Get detailed information about a specific feedback request, including the response if provided.

This endpoint returns both the original request and the response (if the feedback has been provided).
Use this to check the status of a specific feedback request.
    """,
    responses={
        200: {
            "description": "Feedback request and response details",
        },
        404: {
            "description": "Feedback request not found",
            "content": {"application/json": {"example": {"detail": "Feedback request not found"}}},
        },
    },
)
async def get_feedback(request: Request, feedback_id: str) -> FeedbackDetailResponse:
    """Get feedback request details.

    Args:
        request: FastAPI request object
        feedback_id: Feedback request ID

    Returns:
        FeedbackDetailResponse with request and optional response

    Raises:
        HTTPException: 404 if feedback request not found
    """
    feedback_manager = get_feedback_manager(request)

    # Get request
    feedback_request = feedback_manager.get_request(feedback_id)
    if not feedback_request:
        raise HTTPException(status_code=404, detail=f"Feedback request '{feedback_id}' not found")

    # Get response (if exists)
    feedback_response = feedback_manager.get_response(feedback_id)

    # Build response
    request_data = FeedbackRequestResponse(**feedback_request.to_dict())
    response_data = None
    if feedback_response:
        response_data = FeedbackResponseDetails(**feedback_response.to_dict())

    return FeedbackDetailResponse(request=request_data, response=response_data)


@router.post(
    "/{feedback_id}/respond",
    response_model=FeedbackResponseDetails,
    summary="Provide feedback response",
    description="""
Provide a response to a pending feedback request.

This endpoint allows humans to respond to feedback requests from workflows.
The response type should match the original request type (approval, text, selection, etc.).

After providing feedback:
1. The response is stored in the backend
2. The request status is updated to 'completed'
3. A notification is published (for Redis backend)
4. If channel integration is enabled, the response is written to the workflow channel

Returns the complete feedback response with all details for confirmation.
    """,
    responses={
        200: {
            "description": "Feedback provided successfully",
            "content": {
                "application/json": {
                    "example": {
                        "feedback_id": "deploy_task_abc12345",
                        "response_type": "approval",
                        "approved": True,
                        "reason": "Approved via API",
                        "text": None,
                        "selected": None,
                        "selected_multiple": None,
                        "custom_data": None,
                        "responded_at": "2025-12-27T10:30:00.123456",
                        "responded_by": "user@example.com",
                    }
                }
            },
        },
        400: {
            "description": "Failed to provide feedback",
            "content": {"application/json": {"example": {"detail": "Failed to provide feedback"}}},
        },
        404: {
            "description": "Feedback request not found",
            "content": {"application/json": {"example": {"detail": "Feedback request not found"}}},
        },
    },
)
async def respond_to_feedback(
    request: Request, feedback_id: str, body: FeedbackResponseRequest
) -> FeedbackResponseDetails:
    """Provide feedback response.

    Args:
        request: FastAPI request object
        feedback_id: Feedback request ID
        body: Response data (approved, text, selected, etc.)

    Returns:
        FeedbackResponseDetails with complete response information

    Raises:
        HTTPException: 404 if request not found, 400 if failed to provide feedback
    """
    feedback_manager = get_feedback_manager(request)

    # Get original request to determine type
    feedback_request = feedback_manager.get_request(feedback_id)
    if not feedback_request:
        raise HTTPException(status_code=404, detail=f"Feedback request '{feedback_id}' not found")

    # Create response based on request type
    response = FeedbackResponse(
        feedback_id=feedback_id,
        response_type=feedback_request.feedback_type,
        approved=body.approved,
        reason=body.reason,
        text=body.text,
        selected=body.selected,
        selected_multiple=body.selected_multiple,
        custom_data=body.custom_data,
        responded_by=body.responded_by,
    )

    # Provide feedback
    success = feedback_manager.provide_feedback(feedback_id, response)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to provide feedback")

    # Return the complete response details for confirmation
    return FeedbackResponseDetails(**response.to_dict())


@router.delete(
    "/{feedback_id}",
    response_model=MessageResponse,
    summary="Cancel feedback request",
    description="""
Cancel a pending feedback request.

This endpoint marks a feedback request as 'cancelled'. The workflow will not wait for
this feedback anymore. Use this when a feedback request is no longer needed or was created by mistake.
    """,
    responses={
        200: {
            "description": "Feedback request cancelled successfully",
            "content": {
                "application/json": {
                    "example": {"message": "Feedback request cancelled", "feedback_id": "deploy_task_abc12345"}
                }
            },
        },
        404: {
            "description": "Feedback request not found",
            "content": {"application/json": {"example": {"detail": "Feedback request not found"}}},
        },
    },
)
async def cancel_feedback(request: Request, feedback_id: str) -> MessageResponse:
    """Cancel pending feedback request.

    Args:
        request: FastAPI request object
        feedback_id: Feedback request ID

    Returns:
        MessageResponse with success message

    Raises:
        HTTPException: 404 if feedback request not found
    """
    feedback_manager = get_feedback_manager(request)

    # Get request
    feedback_request = feedback_manager.get_request(feedback_id)
    if not feedback_request:
        raise HTTPException(status_code=404, detail=f"Feedback request '{feedback_id}' not found")

    # Update status to cancelled
    feedback_request.status = "cancelled"
    feedback_manager.store_request(feedback_request)

    return MessageResponse(message="Feedback request cancelled", feedback_id=feedback_id)
