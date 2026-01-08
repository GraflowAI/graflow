"""Filesystem-based feedback storage backend."""

from __future__ import annotations

import fcntl
import json
import logging
from pathlib import Path
from typing import Optional

from graflow.hitl.backend.base import FeedbackBackend
from graflow.hitl.types import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)


class FilesystemFeedbackBackend(FeedbackBackend):
    """Filesystem-based feedback storage backend.

    This backend stores feedback requests and responses as JSON files on disk.
    Uses file locking (fcntl) for basic concurrency control.
    Suitable for local development and single-node deployments.

    Directory structure:
        data_dir/
            requests/
                <feedback_id>.json
            responses/
                <feedback_id>.json
    """

    def __init__(self, data_dir: str = "feedback_data") -> None:
        """Initialize filesystem backend.

        Args:
            data_dir: Directory to store feedback data (default: "feedback_data")
        """
        self.data_dir = Path(data_dir)
        self.requests_dir = self.data_dir / "requests"
        self.responses_dir = self.data_dir / "responses"

        # Create directories if they don't exist
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initialized FilesystemFeedbackBackend with data_dir=%s",
            self.data_dir,
            extra={"data_dir": str(self.data_dir)},
        )

    def store_request(self, request: FeedbackRequest) -> None:
        """Store feedback request to file with locking.

        Args:
            request: FeedbackRequest to store
        """
        file_path = self.requests_dir / f"{request.feedback_id}.json"

        try:
            with open(file_path, "w") as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(request.to_dict(), f, indent=2, ensure_ascii=False)
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            logger.debug(
                "Stored feedback request %s to %s",
                request.feedback_id,
                file_path,
                extra={"feedback_id": request.feedback_id, "file_path": str(file_path)},
            )
        except Exception as e:
            logger.error(
                "Failed to store request %s: %s",
                request.feedback_id,
                str(e),
                extra={"feedback_id": request.feedback_id, "error": str(e)},
            )
            raise

    def get_request(self, feedback_id: str) -> Optional[FeedbackRequest]:
        """Get feedback request from file.

        Args:
            feedback_id: Feedback request ID

        Returns:
            FeedbackRequest if found, None otherwise
        """
        file_path = self.requests_dir / f"{feedback_id}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path) as f:
                # Acquire shared lock for reading
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            return FeedbackRequest.from_dict(data)
        except Exception as e:
            logger.error(
                "Failed to load request %s: %s",
                feedback_id,
                str(e),
                extra={"feedback_id": feedback_id, "error": str(e)},
            )
            return None

    def store_response(self, response: FeedbackResponse) -> None:
        """Store feedback response to file with locking.

        Args:
            response: FeedbackResponse to store
        """
        file_path = self.responses_dir / f"{response.feedback_id}.json"

        try:
            with open(file_path, "w") as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(response.to_dict(), f, indent=2, ensure_ascii=False)
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            logger.debug(
                "Stored feedback response %s to %s",
                response.feedback_id,
                file_path,
                extra={"feedback_id": response.feedback_id, "file_path": str(file_path)},
            )
        except Exception as e:
            logger.error(
                "Failed to store response %s: %s",
                response.feedback_id,
                str(e),
                extra={"feedback_id": response.feedback_id, "error": str(e)},
            )
            raise

    def get_response(self, feedback_id: str) -> Optional[FeedbackResponse]:
        """Get feedback response from file.

        Args:
            feedback_id: Feedback request ID

        Returns:
            FeedbackResponse if found, None otherwise
        """
        file_path = self.responses_dir / f"{feedback_id}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path) as f:
                # Acquire shared lock for reading
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            return FeedbackResponse.from_dict(data)
        except Exception as e:
            logger.error(
                "Failed to load response %s: %s",
                feedback_id,
                str(e),
                extra={"feedback_id": feedback_id, "error": str(e)},
            )
            return None

    def list_pending_requests(self, session_id: Optional[str] = None) -> list[FeedbackRequest]:
        """List pending feedback requests from filesystem.

        Args:
            session_id: Optional filter by session ID

        Returns:
            List of pending FeedbackRequest objects
        """
        pending_requests: list[FeedbackRequest] = []

        # Iterate through all request files
        try:
            for file_path in self.requests_dir.glob("*.json"):
                try:
                    with open(file_path) as f:
                        # Acquire shared lock for reading
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                        try:
                            data = json.load(f)
                        finally:
                            # Release lock
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                    request = FeedbackRequest.from_dict(data)

                    # Filter by status
                    if request.status != "pending":
                        continue

                    # Filter by session_id if provided
                    if session_id and request.session_id != session_id:
                        continue

                    pending_requests.append(request)

                except Exception as e:
                    logger.warning(
                        "Failed to load request from %s: %s",
                        file_path,
                        str(e),
                        extra={"file_path": str(file_path), "error": str(e)},
                    )
                    # Continue with other files

        except Exception as e:
            logger.error("Failed to list pending requests: %s", str(e), extra={"error": str(e)})

        return pending_requests

    def list_requests(self, session_id: Optional[str] = None, n_recent: int = 100) -> list[FeedbackRequest]:
        """List recent feedback requests from filesystem (all statuses).

        Args:
            session_id: Optional filter by session ID
            n_recent: Maximum number of recent requests to return (default: 100)

        Returns:
            List of FeedbackRequest objects sorted by created_at descending (newest first)
        """
        requests: list[FeedbackRequest] = []

        # Iterate through all request files
        try:
            for file_path in self.requests_dir.glob("*.json"):
                try:
                    with open(file_path) as f:
                        # Acquire shared lock for reading
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                        try:
                            data = json.load(f)
                        finally:
                            # Release lock
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                    request = FeedbackRequest.from_dict(data)

                    # Filter by session_id if provided
                    if session_id and request.session_id != session_id:
                        continue

                    requests.append(request)

                except Exception as e:
                    logger.warning(
                        "Failed to load request from %s: %s",
                        file_path,
                        str(e),
                        extra={"file_path": str(file_path), "error": str(e)},
                    )
                    # Continue with other files

        except Exception as e:
            logger.error("Failed to list requests: %s", str(e), extra={"error": str(e)})

        # Sort by created_at descending (newest first)
        requests.sort(key=lambda r: r.created_at, reverse=True)

        # Limit to n_recent
        return requests[:n_recent]

    # Optional methods - use default implementations (no-op)
    # publish() - inherited no-op
    # start_listener() - inherited returns None
    # close() - inherited no-op
