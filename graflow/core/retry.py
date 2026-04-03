# graflow/core/retry.py
# This file is part of Graflow, a graph-based workflow management system.
# It implements retry control for failed task re-execution.

from typing import Dict, Optional

from graflow.exceptions import RetryLimitExceededError


class RetryController:
    """Controls task retry on failure.

    This is the single source of truth for retry state.
    TaskExecutionContext delegates all retry operations here.

    Lifecycle:
        1. Engine catches an exception from task execution
        2. Engine calls accept_retry() to check if retry is allowed
        3. If allowed, engine calls increment() and re-enqueues the task
        4. If not allowed, engine calls check_retry_limit() which raises RetryLimitExceededError
    """

    def __init__(self, default_max_retries: int = 0):
        self.default_max_retries: int = default_max_retries
        self.retry_counts: Dict[str, int] = {}
        self.node_max_retries: Dict[str, int] = {}
        self.last_errors: Dict[str, Exception] = {}

    def set_node_max_retries(self, node_id: str, max_retries: int) -> None:
        """Set maximum retry count for a specific node."""
        self.node_max_retries[node_id] = max_retries

    def get_max_retries_for_node(self, node_id: str) -> int:
        """Get maximum retry count for a node (node-specific or default)."""
        return self.node_max_retries.get(node_id, self.default_max_retries)

    def accept_retry(self, node_id: str) -> bool:
        """Return True if the node has remaining retry attempts."""
        return self.retry_counts.get(node_id, 0) < self.get_max_retries_for_node(node_id)

    def increment(self, node_id: str, error: Optional[Exception] = None) -> int:
        """Increment retry count after a failed attempt.

        Args:
            node_id: The node identifier
            error: The exception that caused the retry

        Returns:
            The new retry count (1-based).
        """
        count = self.retry_counts.get(node_id, 0) + 1
        self.retry_counts[node_id] = count
        if error is not None:
            self.last_errors[node_id] = error
        return count

    def check_retry_limit(self, node_id: str) -> None:
        """Raise RetryLimitExceededError if the retry limit has been reached.

        Called by the engine when a task fails and cannot be retried.
        """
        count = self.retry_counts.get(node_id, 0)
        max_retries = self.get_max_retries_for_node(node_id)
        if count >= max_retries:
            raise RetryLimitExceededError(
                task_id=node_id,
                retry_count=count,
                max_retries=max_retries,
                last_error=self.last_errors.get(node_id),
            )

    def get_retry_count(self, node_id: str) -> int:
        """Return how many times the given node has been retried (0 if never)."""
        return self.retry_counts.get(node_id, 0)

    def get_last_error(self, node_id: str) -> Optional[Exception]:
        """Return the last error for the given node, or None."""
        return self.last_errors.get(node_id)
