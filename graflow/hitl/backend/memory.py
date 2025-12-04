"""In-memory feedback storage backend."""

from __future__ import annotations

import threading
from typing import Optional

from graflow.hitl.backend.base import FeedbackBackend
from graflow.hitl.types import FeedbackRequest, FeedbackResponse


class MemoryFeedbackBackend(FeedbackBackend):
    """In-memory feedback storage backend.

    This backend stores feedback requests and responses in memory using dictionaries.
    Suitable for single-process workflows and testing.
    """

    def __init__(self) -> None:
        """Initialize memory backend."""
        self._requests: dict[str, FeedbackRequest] = {}
        self._responses: dict[str, FeedbackResponse] = {}
        self._lock = threading.Lock()

    def store_request(self, request: FeedbackRequest) -> None:
        """Store feedback request in memory."""
        with self._lock:
            self._requests[request.feedback_id] = request

    def get_request(self, feedback_id: str) -> Optional[FeedbackRequest]:
        """Get feedback request from memory."""
        return self._requests.get(feedback_id)

    def store_response(self, response: FeedbackResponse) -> None:
        """Store feedback response in memory."""
        with self._lock:
            self._responses[response.feedback_id] = response

    def get_response(self, feedback_id: str) -> Optional[FeedbackResponse]:
        """Get feedback response from memory."""
        return self._responses.get(feedback_id)

    def list_pending_requests(
        self, session_id: Optional[str] = None
    ) -> list[FeedbackRequest]:
        """List pending feedback requests from memory."""
        requests = [
            req for req in self._requests.values()
            if req.status == "pending"
        ]

        # Filter by session_id if provided
        if session_id:
            requests = [req for req in requests if req.session_id == session_id]

        return requests
