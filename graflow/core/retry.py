# graflow/core/retry.py
# This file is part of Graflow, a graph-based workflow management system.
# It implements retry control for failed task re-execution.

import random
from dataclasses import dataclass
from typing import Dict, Optional

from graflow.exceptions import RetryLimitExceededError


@dataclass
class RetryPolicy:
    """Retry policy with exponential backoff support.

    Args:
        max_retries: Maximum number of retry attempts after initial failure (0 = no retries).
        initial_interval: Wait time in seconds before the first retry.
        backoff_factor: Multiplier applied to the interval after each retry
                       (e.g., 2.0 means 1s → 2s → 4s).
        max_interval: Upper bound on the wait time in seconds.
        jitter: If True, randomize the delay by ±50% to avoid thundering herd.

    Example:
        ```python
        policy = RetryPolicy(
            max_retries=3,
            initial_interval=1.0,
            backoff_factor=2.0,
        )
        # Delays: 1.0s, 2.0s, 4.0s
        ```
    """

    max_retries: int = 0
    initial_interval: float = 1.0
    backoff_factor: float = 2.0
    max_interval: float = 60.0
    jitter: bool = False

    def get_delay(self, retry_count: int) -> float:
        """Calculate the delay before the given retry attempt.

        Args:
            retry_count: The upcoming retry number (0-based: 0 = first retry).

        Returns:
            Delay in seconds.
        """
        delay = self.initial_interval * (self.backoff_factor**retry_count)
        delay = min(delay, self.max_interval)
        if self.jitter:
            delay *= random.uniform(0.5, 1.5)
        return delay


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
        self.node_policies: Dict[str, RetryPolicy] = {}
        self.last_errors: Dict[str, Exception] = {}

    def set_node_max_retries(self, node_id: str, max_retries: int) -> None:
        """Set maximum retry count for a specific node."""
        self.node_max_retries[node_id] = max_retries

    def set_node_policy(self, node_id: str, policy: RetryPolicy) -> None:
        """Set retry policy for a specific node.

        This also sets max_retries from the policy for consistency.
        """
        self.node_policies[node_id] = policy
        self.node_max_retries[node_id] = policy.max_retries

    def get_policy_for_node(self, node_id: str) -> Optional[RetryPolicy]:
        """Get retry policy for a node, or None if not set."""
        return self.node_policies.get(node_id)

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

    def get_delay(self, node_id: str) -> float:
        """Return the backoff delay for the current retry attempt.

        If a RetryPolicy is set for the node, uses its get_delay().
        Otherwise returns 0.0 (no delay, backward compatible).

        Args:
            node_id: The node identifier

        Returns:
            Delay in seconds before the next retry.
        """
        policy = self.node_policies.get(node_id)
        if policy is None:
            return 0.0
        # retry_count is already incremented before this call,
        # so use (count - 1) as the 0-based retry index
        count = self.retry_counts.get(node_id, 0)
        return policy.get_delay(max(0, count - 1))

    def get_last_error(self, node_id: str) -> Optional[Exception]:
        """Return the last error for the given node, or None."""
        return self.last_errors.get(node_id)
