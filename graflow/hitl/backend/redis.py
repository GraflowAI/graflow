"""Redis-based feedback storage backend."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

from graflow.hitl.backend.base import FeedbackBackend
from graflow.hitl.types import FeedbackRequest, FeedbackResponse

if TYPE_CHECKING:
    import redis

logger = logging.getLogger(__name__)


class RedisFeedbackBackend(FeedbackBackend):
    """Redis-based feedback storage backend.

    This backend stores feedback requests and responses in Redis as JSON strings.
    Suitable for distributed workflows with multiple workers.

    Features:
    - Persistent storage across process restarts
    - Pub/Sub notifications for real-time feedback delivery
    - Automatic expiration (7 days default)
    - JSON serialization for simple storage and retrieval
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        expiration_days: int = 7,
    ) -> None:
        """Initialize Redis backend.

        Args:
            redis_client: Optional existing Redis client
            host: Redis host (default: localhost)
            port: Redis port (default: 6379)
            db: Redis database number (default: 0)
            expiration_days: Days until keys expire (default: 7)
        """
        try:
            import redis as redis_module
        except ImportError as e:
            raise ImportError("Redis backend requires redis package. Install with: pip install redis") from e

        self._redis_client: redis.Redis = redis_client or redis_module.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
        )
        self._expiration_seconds = expiration_days * 24 * 60 * 60
        self._pubsub = self._redis_client.pubsub()

    def store_request(self, request: FeedbackRequest) -> None:
        """Store feedback request in Redis as JSON.

        Args:
            request: FeedbackRequest to store
        """
        key = f"feedback:request:{request.feedback_id}"
        data_json = request.model_dump_json()
        self._redis_client.set(key, data_json)
        self._redis_client.expire(key, self._expiration_seconds)

    def get_request(self, feedback_id: str) -> Optional[FeedbackRequest]:
        """Get feedback request from Redis.

        Args:
            feedback_id: Feedback request ID

        Returns:
            FeedbackRequest if found, None otherwise
        """
        key = f"feedback:request:{feedback_id}"
        data_json = self._redis_client.get(key)
        if not data_json:
            return None

        return FeedbackRequest.model_validate_json(str(data_json))

    def store_response(self, response: FeedbackResponse) -> None:
        """Store feedback response in Redis as JSON.

        Args:
            response: FeedbackResponse to store
        """
        key = f"feedback:response:{response.feedback_id}"
        data_json = response.model_dump_json()
        self._redis_client.set(key, data_json)
        self._redis_client.expire(key, self._expiration_seconds)

    def get_response(self, feedback_id: str) -> Optional[FeedbackResponse]:
        """Get feedback response from Redis.

        Args:
            feedback_id: Feedback request ID

        Returns:
            FeedbackResponse if found, None otherwise
        """
        key = f"feedback:response:{feedback_id}"
        data_json = self._redis_client.get(key)
        if not data_json:
            return None

        return FeedbackResponse.model_validate_json(str(data_json))

    def list_pending_requests(self, session_id: Optional[str] = None) -> list[FeedbackRequest]:
        """List pending feedback requests from Redis.

        Args:
            session_id: Optional filter by session ID

        Returns:
            List of pending FeedbackRequest objects
        """
        keys = self._redis_client.keys("feedback:request:*")
        if not keys:
            return []

        requests: list[FeedbackRequest] = []
        for key in keys:  # type: ignore
            try:
                data_json = self._redis_client.get(key)
                if not data_json:
                    continue

                request = FeedbackRequest.model_validate_json(str(data_json))

                # Filter by status
                if request.status != "pending":
                    continue

                # Filter by session_id if provided
                if session_id and request.session_id != session_id:
                    continue

                requests.append(request)
            except Exception as e:
                # Skip keys that can't be read (wrong type, corrupted, etc.)
                logger.warning(
                    "Failed to read feedback request from key %s: %s. Skipping.",
                    key,
                    str(e),
                    extra={"key": key, "error": str(e)},
                )
                continue

        return requests

    def list_requests(self, session_id: Optional[str] = None, n_recent: int = 100) -> list[FeedbackRequest]:
        """List recent feedback requests from Redis (all statuses).

        Args:
            session_id: Optional filter by session ID
            n_recent: Maximum number of recent requests to return (default: 100)

        Returns:
            List of FeedbackRequest objects sorted by created_at descending (newest first)
        """
        keys = self._redis_client.keys("feedback:request:*")
        if not keys:
            return []

        requests: list[FeedbackRequest] = []
        for key in keys:  # type: ignore
            try:
                data_json = self._redis_client.get(key)
                if not data_json:
                    continue

                request = FeedbackRequest.model_validate_json(str(data_json))

                # Filter by session_id if provided
                if session_id and request.session_id != session_id:
                    continue

                requests.append(request)
            except Exception as e:
                # Skip keys that can't be read (wrong type, corrupted, etc.)
                logger.warning(
                    "Failed to read feedback request from key %s: %s. Skipping.",
                    key,
                    str(e),
                    extra={"key": key, "error": str(e)},
                )
                continue

        # Sort by created_at descending (newest first)
        requests.sort(key=lambda r: r.created_at, reverse=True)

        # Limit to n_recent
        return requests[:n_recent]

    def publish(self, feedback_id: str) -> None:
        """Publish notification via Redis Pub/Sub.

        Args:
            feedback_id: Feedback request ID to publish notification for
        """
        channel = f"feedback:{feedback_id}"
        self._redis_client.publish(channel, "completed")

    def start_listener(self, feedback_id: str, notification_event: threading.Event) -> Optional[threading.Thread]:
        """Start Redis Pub/Sub listener for feedback notifications.

        Args:
            feedback_id: Feedback request ID to listen for
            notification_event: Event to set when notification arrives

        Returns:
            Thread object that is listening for notifications, or None if failed
        """
        channel = f"feedback:{feedback_id}"

        try:
            self._pubsub.subscribe(channel)
        except Exception as e:
            logger.warning(
                "Failed to subscribe to Redis channel %s: %s",
                channel,
                str(e),
                extra={"feedback_id": feedback_id, "channel": channel, "error": str(e)},
            )
            return None

        def listener_thread() -> None:
            """Background thread listening for Redis Pub/Sub messages."""
            try:
                for message in self._pubsub.listen():
                    if message and message.get("type") == "message":
                        notification_event.set()
                        break
            except Exception as e:
                logger.warning(
                    "Error in Redis Pub/Sub listener for %s: %s",
                    feedback_id,
                    str(e),
                    extra={"feedback_id": feedback_id, "channel": channel, "error": str(e)},
                )
            finally:
                try:
                    self._pubsub.unsubscribe(channel)
                except Exception as e:
                    logger.debug(
                        "Failed to unsubscribe from channel %s: %s",
                        channel,
                        str(e),
                        extra={"feedback_id": feedback_id, "channel": channel, "error": str(e)},
                    )

        thread = threading.Thread(target=listener_thread, daemon=True)
        thread.start()
        return thread

    def close(self) -> None:
        """Close Redis connections and cleanup resources."""
        try:
            self._pubsub.close()
        except Exception as e:
            logger.debug(
                "Error closing Redis Pub/Sub connection: %s",
                str(e),
                extra={"error": str(e)},
            )
