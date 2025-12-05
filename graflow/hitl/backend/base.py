"""Abstract base class for feedback storage backends."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Optional

from graflow.hitl.types import FeedbackRequest, FeedbackResponse


class FeedbackBackend(ABC):
    """Abstract interface for feedback storage backends."""

    @abstractmethod
    def store_request(self, request: FeedbackRequest) -> None:
        """Persist a feedback request.

        Args:
            request: FeedbackRequest to store
        """

    @abstractmethod
    def get_request(self, feedback_id: str) -> Optional[FeedbackRequest]:
        """Retrieve a feedback request by ID.

        Args:
            feedback_id: Unique feedback request ID

        Returns:
            FeedbackRequest if found, None otherwise
        """

    @abstractmethod
    def store_response(self, response: FeedbackResponse) -> None:
        """Persist a feedback response.

        Args:
            response: FeedbackResponse to store
        """

    @abstractmethod
    def get_response(self, feedback_id: str) -> Optional[FeedbackResponse]:
        """Retrieve a feedback response by ID.

        Args:
            feedback_id: Unique feedback request ID

        Returns:
            FeedbackResponse if found, None otherwise
        """

    @abstractmethod
    def list_pending_requests(self, session_id: Optional[str] = None) -> list[FeedbackRequest]:
        """List pending requests, optionally scoped by session.

        Args:
            session_id: Optional session ID to filter by

        Returns:
            List of pending FeedbackRequest objects
        """

    @abstractmethod
    def list_requests(self, session_id: Optional[str] = None, n_recent: int = 100) -> list[FeedbackRequest]:
        """List recent requests (all statuses), optionally scoped by session.

        Args:
            session_id: Optional session ID to filter by
            n_recent: Maximum number of recent requests to return (default: 100)

        Returns:
            List of FeedbackRequest objects sorted by created_at descending (newest first)
        """

    def publish(self, feedback_id: str) -> None:
        """Publish a notification that feedback has been provided (optional).

        Args:
            feedback_id: Feedback request ID

        Note:
            Default implementation is a no-op for backends without notifications.
        """
        # Default: no-op for backends without notifications

    def start_listener(self, feedback_id: str, notification_event: threading.Event) -> Optional[threading.Thread]:
        """Start background listener to set notification_event when response arrives.

        Args:
            feedback_id: Feedback request ID to listen for
            notification_event: Event to set when response arrives

        Returns:
            Thread object if listener was started, None otherwise

        Note:
            Default implementation returns None (no listener support).
        """
        return None

    def close(self) -> None:
        """Clean up backend resources.

        Note:
            Default implementation does nothing.
        """
        # Default: nothing to clean up
