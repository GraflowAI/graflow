"""Redis-based feedback storage backend."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

from graflow.hitl.backend.base import FeedbackBackend
from graflow.hitl.types import FeedbackRequest, FeedbackResponse

if TYPE_CHECKING:
    import redis


class RedisFeedbackBackend(FeedbackBackend):
    """Redis-based feedback storage backend.

    This backend stores feedback requests and responses in Redis.
    Suitable for distributed workflows with multiple workers.

    Features:
    - Persistent storage across process restarts
    - Pub/Sub notifications for real-time feedback delivery
    - Automatic expiration (7 days default)
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
            raise ImportError(
                "Redis backend requires redis package. "
                "Install with: pip install redis"
            ) from e

        self._redis_client: redis.Redis = redis_client or redis_module.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
        )
        self._expiration_seconds = expiration_days * 24 * 60 * 60
        self._pubsub = self._redis_client.pubsub()

    def store_request(self, request: FeedbackRequest) -> None:
        """Store feedback request in Redis."""
        key = f"feedback:request:{request.feedback_id}"
        self._redis_client.hset(key, mapping=request.to_dict())
        self._redis_client.expire(key, self._expiration_seconds)

    def get_request(self, feedback_id: str) -> Optional[FeedbackRequest]:
        """Get feedback request from Redis."""
        key = f"feedback:request:{feedback_id}"
        data = self._redis_client.hgetall(key)
        if data:
            return FeedbackRequest.from_dict(data)  # type: ignore
        return None

    def store_response(self, response: FeedbackResponse) -> None:
        """Store feedback response in Redis."""
        key = f"feedback:response:{response.feedback_id}"
        self._redis_client.hset(key, mapping=response.to_dict())
        self._redis_client.expire(key, self._expiration_seconds)

    def get_response(self, feedback_id: str) -> Optional[FeedbackResponse]:
        """Get feedback response from Redis."""
        key = f"feedback:response:{feedback_id}"
        data = self._redis_client.hgetall(key)
        if data:
            return FeedbackResponse.from_dict(data)  # type: ignore
        return None

    def list_pending_requests(
        self, session_id: Optional[str] = None
    ) -> list[FeedbackRequest]:
        """List pending feedback requests from Redis."""
        keys = self._redis_client.keys("feedback:request:*")
        if keys is None:
            return []

        requests = []
        for key in keys:  # type: ignore
            data = self._redis_client.hgetall(key)
            if data and data.get("status") == "pending":  # type: ignore
                requests.append(FeedbackRequest.from_dict(data))  # type: ignore

        # Filter by session_id if provided
        if session_id:
            requests = [req for req in requests if req.session_id == session_id]

        return requests

    def publish(self, feedback_id: str) -> None:
        """Publish notification via Redis Pub/Sub."""
        self._redis_client.publish(f"feedback:{feedback_id}", "completed")

    def start_listener(
        self, feedback_id: str, notification_event: threading.Event
    ) -> Optional[threading.Thread]:
        """Start Redis Pub/Sub listener for feedback notifications.

        Args:
            feedback_id: Feedback request ID to listen for
            notification_event: Event to set when notification arrives

        Returns:
            Thread object that is listening for notifications
        """
        channel = f"feedback:{feedback_id}"
        self._pubsub.subscribe(channel)

        def listener_thread():
            """Background thread listening for Redis Pub/Sub messages."""
            try:
                for message in self._pubsub.listen():
                    if message and message["type"] == "message":
                        notification_event.set()
                        break
            finally:
                self._pubsub.unsubscribe(channel)

        thread = threading.Thread(target=listener_thread, daemon=True)
        thread.start()
        return thread

    def close(self) -> None:
        """Close Redis connections."""
        try:
            self._pubsub.close()
        except Exception:
            pass
