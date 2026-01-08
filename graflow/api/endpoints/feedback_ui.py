"""Web UI endpoints for HITL feedback."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from graflow.hitl.types import FeedbackType

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feedback-ui"])


@router.get("/ui/feedback/", response_class=HTMLResponse)
async def list_pending_feedback(
    request: Request,
    session_id: Optional[str] = None,
    n_recent: int = 100,
):
    """Display list of recent feedback requests (all statuses).

    Args:
        request: FastAPI request object
        session_id: Optional session_id filter
        n_recent: Maximum number of recent requests to show (default: 100)

    Returns:
        HTML response with feedback list
    """
    # Get feedback manager from app state
    feedback_manager = request.app.state.feedback_manager

    # Get recent requests (all statuses, already sorted by created_at desc)
    sorted_requests = feedback_manager.list_requests(session_id=session_id, n_recent=n_recent)

    # Mask feedback_ids for security (API-side masking)
    # Convert to dict and replace feedback_id with masked version
    masked_requests = []
    for req in sorted_requests:
        req_dict = req.to_dict()  # Use to_dict() to properly convert Enum to string
        # Mask feedback_id: show first character + 14 asterisks + last character
        original_id = req_dict["feedback_id"]
        req_dict["feedback_id"] = (
            f"{original_id[:1]}**************{original_id[-1:]}" if original_id else "****************"
        )
        masked_requests.append(req_dict)

    # Render list
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "feedback/list.html",
        {
            "request": request,
            "count": len(masked_requests),
            "requests": masked_requests,
        },
    )


@router.get("/ui/feedback/{feedback_id}", response_class=HTMLResponse)
async def show_feedback_form(
    request: Request,
    feedback_id: str,
):
    """Display feedback form for the given feedback_id.

    Args:
        request: FastAPI request object
        feedback_id: Feedback request ID (acts as authentication token)

    Returns:
        HTML response with feedback form
    """
    # Get feedback manager from app state
    feedback_manager = request.app.state.feedback_manager

    # Get feedback request
    feedback_request = feedback_manager.get_request(feedback_id)

    if not feedback_request:
        raise HTTPException(status_code=404, detail="Feedback request not found")

    # Show expired page for completed or cancelled requests
    # For pending and timeout states, show the form (user can still provide feedback)
    if feedback_request.status in ("completed", "cancelled"):
        return RedirectResponse(url=f"/ui/feedback/{feedback_id}/expired")

    # Render form (for pending or timeout status)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "feedback/form.html",
        {
            "request": request,
            "feedback_request": feedback_request,
        },
    )


@router.post("/ui/feedback/{feedback_id}/submit")
async def submit_feedback(
    request: Request,
    feedback_id: str,
    approved: Optional[str] = Form(None),
    reason: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    selected: Optional[str] = Form(None),
    selected_multiple: Optional[list[str]] = Form(None),
    custom_data: Optional[str] = Form(None),
    responded_by: Optional[str] = Form(None),
):
    """Process submitted feedback form.

    This is the web UI endpoint for form submissions. It differs from the REST API:
    - Accepts form data (not JSON)
    - Returns redirect to success page (not JSON response)
    - Success page displays the complete feedback response

    For REST API feedback submission, use POST /api/feedback/{feedback_id}/respond

    Args:
        request: FastAPI request object
        feedback_id: Feedback request ID
        approved: Approval decision (for approval type)
        reason: Reason for decision
        text: Text input (for text type)
        selected: Selected option (for selection type)
        selected_multiple: Selected options (for multi_selection type)
        custom_data: Custom JSON data (for custom type)
        responded_by: User identifier

    Returns:
        Redirect to success page
    """
    # Get feedback manager from app state
    feedback_manager = request.app.state.feedback_manager

    # Get feedback request
    feedback_request = feedback_manager.get_request(feedback_id)

    if not feedback_request:
        raise HTTPException(status_code=404, detail="Feedback request not found")

    # Reject submission for completed or cancelled requests
    # Allow submission for pending or timeout states
    if feedback_request.status in ("completed", "cancelled"):
        return RedirectResponse(url=f"/ui/feedback/{feedback_id}/expired")

    # Import here to avoid circular imports
    from graflow.hitl.types import FeedbackResponse

    # Build response based on feedback type
    response_data = {
        "feedback_id": feedback_id,
        "response_type": feedback_request.feedback_type,
        "responded_at": datetime.now().isoformat(),
        "responded_by": responded_by,
    }

    # Add type-specific fields
    if feedback_request.feedback_type == FeedbackType.APPROVAL:
        response_data["approved"] = approved == "true" if approved else None
        response_data["reason"] = reason

    elif feedback_request.feedback_type == FeedbackType.TEXT:
        response_data["text"] = text

    elif feedback_request.feedback_type == FeedbackType.SELECTION:
        response_data["selected"] = selected

    elif feedback_request.feedback_type == FeedbackType.MULTI_SELECTION:
        response_data["selected_multiple"] = selected_multiple or []

    elif feedback_request.feedback_type == FeedbackType.CUSTOM:
        # Parse JSON
        try:
            response_data["custom_data"] = json.loads(custom_data) if custom_data else {}
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in custom_data: {e}") from e

    # Create response object
    try:
        feedback_response = FeedbackResponse(**response_data)
    except Exception as e:
        logger.error("Failed to create FeedbackResponse: %s", e, extra={"error": str(e)})
        raise HTTPException(status_code=400, detail=f"Invalid feedback response: {e}") from e

    # Submit response via FeedbackManager
    try:
        success = feedback_manager.provide_feedback(feedback_id, feedback_response)
    except Exception as e:
        logger.error("Failed to provide feedback: %s", e, extra={"feedback_id": feedback_id, "error": str(e)})
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {e}") from e

    if not success:
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

    # Redirect to success page
    return RedirectResponse(
        url=f"/ui/feedback/{feedback_id}/success",
        status_code=303,  # See Other (POST -> GET redirect)
    )


@router.get("/ui/feedback/{feedback_id}/success", response_class=HTMLResponse)
async def show_success_page(
    request: Request,
    feedback_id: str,
):
    """Display success page after feedback submission.

    Args:
        request: FastAPI request object
        feedback_id: Feedback request ID

    Returns:
        HTML response with success message
    """
    # Get feedback manager from app state
    feedback_manager = request.app.state.feedback_manager

    # Get response
    feedback_response = feedback_manager.get_response(feedback_id)

    # Render success page
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "feedback/success.html",
        {
            "request": request,
            "response": feedback_response,
        },
    )


@router.get("/ui/feedback/{feedback_id}/expired", response_class=HTMLResponse)
async def show_expired_page(
    request: Request,
    feedback_id: str,
):
    """Display expired/already responded page.

    Args:
        request: FastAPI request object
        feedback_id: Feedback request ID

    Returns:
        HTML response with expired message
    """
    # Get feedback manager from app state
    feedback_manager = request.app.state.feedback_manager

    # Get request (may be None)
    feedback_request = feedback_manager.get_request(feedback_id)

    # Render expired page
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "feedback/expired.html",
        {
            "request": request,
            "feedback_request": feedback_request,
        },
    )
