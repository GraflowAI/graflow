"""Feedback manager for Human-in-the-Loop functionality."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Optional

from graflow.hitl.backend.base import FeedbackBackend
from graflow.hitl.backend.redis import RedisFeedbackBackend
from graflow.hitl.types import (
    FeedbackRequest,
    FeedbackResponse,
    FeedbackTimeoutError,
    FeedbackType,
)

if TYPE_CHECKING:
    from graflow.channels.base import Channel

logger = logging.getLogger(__name__)


class FeedbackManager:
    """Manages feedback requests and responses with backend persistence."""

    def __init__(
        self,
        backend: str | FeedbackBackend = "filesystem",
        backend_config: Optional[dict] = None,
        channel_manager: Optional[Channel] = None,
    ):
        """Initialize feedback manager.

        Args:
            backend: Backend instance or "filesystem"/"redis" string
            backend_config: Backend-specific configuration (used only if backend is a string)
                - filesystem: {"data_dir": "feedback_data"}
                - redis: {"redis_client": redis.Redis, "host": "localhost", "port": 6379, "db": 0, "expiration_days": 7}
            channel_manager: Optional Channel instance for writing feedback to channels
        """
        self.channel_manager = channel_manager

        # Initialize backend
        if isinstance(backend, FeedbackBackend):
            self._backend = backend
        elif backend == "filesystem":
            from graflow.hitl.backend.filesystem import FilesystemFeedbackBackend
            backend_config = backend_config or {}
            self._backend = FilesystemFeedbackBackend(
                data_dir=backend_config.get("data_dir", "feedback_data")
            )
        elif backend == "redis":
            backend_config = backend_config or {}
            self._backend = RedisFeedbackBackend(
                redis_client=backend_config.get("redis_client"),
                host=backend_config.get("host", "localhost"),
                port=backend_config.get("port", 6379),
                db=backend_config.get("db", 0),
                expiration_days=backend_config.get("expiration_days", 7),
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}. Use 'filesystem' or 'redis'.")

    def request_feedback(
        self,
        task_id: str,
        session_id: str,
        feedback_type: FeedbackType,
        prompt: str,
        options: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        timeout: float = 180.0,  # Default: 3 minutes
        channel_key: Optional[str] = None,
        write_to_channel: bool = False,
    ) -> FeedbackResponse:
        """Request feedback and wait for response.

        Args:
            task_id: Task requesting feedback
            session_id: Workflow session ID
            feedback_type: Type of feedback
            prompt: Prompt for human
            options: Options for selection types
            metadata: Custom metadata
            timeout: Polling timeout in seconds (default: 180s / 3 minutes)
            channel_key: Optional channel key to write response to
            write_to_channel: Whether to auto-write response to channel

        Returns:
            FeedbackResponse

        Raises:
            FeedbackTimeoutException: If timeout exceeded without response
        """
        # Generate feedback ID
        feedback_id = f"{task_id}_{uuid.uuid4().hex[:8]}"

        # Check if response already exists (resume case)
        existing_response = self._backend.get_response(feedback_id)
        if existing_response:
            logger.info(
                "Found existing response for %s",
                feedback_id,
                extra={"feedback_id": feedback_id, "session_id": session_id}
            )
            return existing_response

        # Create and store request
        request = FeedbackRequest(
            feedback_id=feedback_id,
            task_id=task_id,
            session_id=session_id,
            feedback_type=feedback_type,
            prompt=prompt,
            options=options,
            metadata=metadata or {},
            timeout=timeout,
            status="pending",
            channel_key=channel_key,
            write_to_channel=write_to_channel,
        )
        self._backend.store_request(request)

        logger.info(
            "Created feedback request: %s - %s",
            feedback_id,
            prompt,
            extra={
                "feedback_id": feedback_id,
                "task_id": task_id,
                "session_id": session_id,
                "timeout": timeout
            }
        )

        # Poll for response
        response = self._poll_for_response(feedback_id, timeout)

        if response:
            # Update request status
            request.status = "completed"
            self._backend.store_request(request)
            return response
        else:
            # Timeout - update status and raise exception
            request.status = "timeout"
            self._backend.store_request(request)
            raise FeedbackTimeoutError(feedback_id, timeout)

    def provide_feedback(
        self,
        feedback_id: str,
        response: FeedbackResponse,
    ) -> bool:
        """Provide feedback response (called by external API).

        Args:
            feedback_id: Feedback request ID
            response: Feedback response

        Returns:
            True if successful, False if request not found
        """
        # Get request
        request = self._backend.get_request(feedback_id)
        if not request:
            logger.warning(
                "Request %s not found",
                feedback_id,
                extra={"feedback_id": feedback_id}
            )
            return False

        # Store response
        self._backend.store_response(response)

        # Update request status
        request.status = "completed"
        self._backend.store_request(request)

        # Write to channel if requested
        self._write_to_channel_if_needed(request, response)

        # Publish notification
        self._backend.publish(feedback_id)

        logger.info(
            "Feedback provided for %s",
            feedback_id,
            extra={"feedback_id": feedback_id}
        )
        return True

    def _write_to_channel_if_needed(
        self,
        request: FeedbackRequest,
        response: FeedbackResponse,
    ) -> None:
        """Write feedback response to channel if requested.

        Args:
            request: Original feedback request
            response: Feedback response to write
        """
        if not (request.write_to_channel and request.channel_key and self.channel_manager):
            return

        try:
            # Write response to channel based on feedback type
            if request.feedback_type == FeedbackType.APPROVAL:
                self.channel_manager.set(request.channel_key, response.approved)
            elif request.feedback_type == FeedbackType.TEXT:
                self.channel_manager.set(request.channel_key, response.text)
            elif request.feedback_type == FeedbackType.SELECTION:
                self.channel_manager.set(request.channel_key, response.selected)
            elif request.feedback_type == FeedbackType.MULTI_SELECTION:
                self.channel_manager.set(request.channel_key, response.selected_multiple)
            elif request.feedback_type == FeedbackType.CUSTOM:
                self.channel_manager.set(request.channel_key, response.custom_data)

            # Also write full response object to {channel_key}.__feedback_response__
            self.channel_manager.set(
                f"{request.channel_key}.__feedback_response__",
                response.to_dict()
            )

            logger.info(
                "Wrote feedback to channel key: %s",
                request.channel_key,
                extra={
                    "feedback_id": request.feedback_id,
                    "channel_key": request.channel_key
                }
            )
        except Exception as e:
            logger.error(
                "Failed to write to channel: %s",
                str(e),
                extra={"feedback_id": request.feedback_id, "error": str(e)}
            )
            # Don't fail the entire operation if channel write fails

    def list_pending_requests(
        self,
        session_id: Optional[str] = None
    ) -> list[FeedbackRequest]:
        """List pending feedback requests.

        Args:
            session_id: Optional filter by session ID

        Returns:
            List of pending FeedbackRequest objects
        """
        return self._backend.list_pending_requests(session_id)

    def get_request(self, feedback_id: str) -> Optional[FeedbackRequest]:
        """Get feedback request by ID.

        Args:
            feedback_id: Feedback request ID

        Returns:
            FeedbackRequest if found, None otherwise
        """
        return self._backend.get_request(feedback_id)

    def get_response(self, feedback_id: str) -> Optional[FeedbackResponse]:
        """Get feedback response by ID.

        Args:
            feedback_id: Feedback request ID

        Returns:
            FeedbackResponse if found, None otherwise
        """
        return self._backend.get_response(feedback_id)

    def update_request(self, request: FeedbackRequest) -> None:
        """Update feedback request in backend.

        Args:
            request: FeedbackRequest to update
        """
        self._backend.store_request(request)

    def _poll_for_response(
        self,
        feedback_id: str,
        timeout: float,
    ) -> Optional[FeedbackResponse]:
        """Poll for response with timeout.

        Args:
            feedback_id: Feedback request ID
            timeout: Timeout in seconds

        Returns:
            FeedbackResponse if received, None if timeout
        """
        poll_interval = 0.5  # Poll every 500ms
        elapsed = 0.0

        # Start background listener if backend supports it
        notification_event = threading.Event()
        listener_thread = self._backend.start_listener(feedback_id, notification_event)

        try:
            while elapsed < timeout:
                # Check for response
                response = self._backend.get_response(feedback_id)
                if response:
                    logger.info(
                        "Received feedback response for %s after %.1f seconds",
                        feedback_id,
                        elapsed,
                        extra={"feedback_id": feedback_id, "elapsed": elapsed}
                    )
                    return response

                # Wait for notification or poll interval
                if listener_thread:
                    # Wait for event with timeout
                    notification_event.wait(timeout=poll_interval)
                    if notification_event.is_set():
                        # Notification received, fetch response
                        response = self._backend.get_response(feedback_id)
                        if response:
                            logger.info(
                                "Received feedback response for %s via notification after %.1f seconds",
                                feedback_id,
                                elapsed,
                                extra={"feedback_id": feedback_id, "elapsed": elapsed, "via": "notification"}
                            )
                            return response
                        # Clear event and continue waiting
                        notification_event.clear()
                else:
                    # No listener support, simple sleep
                    time.sleep(poll_interval)

                elapsed += poll_interval
        finally:
            # Cleanup is handled by backend.close() if needed
            pass

        return None
