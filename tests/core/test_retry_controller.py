"""Unit tests for RetryController and RetryPolicy.

Tests cover:
  - increment() counting
  - accept_retry() with default and per-node limits
  - check_retry_limit() raising RetryLimitExceededError
  - get_retry_count() for unknown nodes
  - Per-node max_retries override via set_node_max_retries()
  - Last error tracking
  - Independent retry tracking across multiple nodes
  - default_max_retries=0 means no retries by default
  - RetryPolicy exponential backoff delay calculation
  - RetryController.set_node_policy() / get_delay() integration
"""

import pytest

from graflow.core.retry import RetryController, RetryPolicy
from graflow.exceptions import RetryLimitExceededError


class TestRetryControllerDefaults:
    """Tests for default behavior (no retries)."""

    def test_default_max_retries_is_zero(self):
        """By default, no retries are allowed."""
        ctrl = RetryController()
        assert ctrl.default_max_retries == 0

    def test_accept_retry_false_by_default(self):
        """With default_max_retries=0, accept_retry() is always False."""
        ctrl = RetryController()
        assert ctrl.accept_retry("node_a") is False

    def test_custom_default_max_retries(self):
        """Custom default_max_retries applies to all nodes."""
        ctrl = RetryController(default_max_retries=3)
        assert ctrl.get_max_retries_for_node("any_node") == 3
        assert ctrl.accept_retry("any_node") is True


class TestRetryControllerIncrement:
    """Tests for increment() method."""

    def test_first_increment_returns_one(self):
        """First retry increment returns 1."""
        ctrl = RetryController(default_max_retries=3)
        assert ctrl.increment("node_a") == 1

    def test_subsequent_increments(self):
        """Subsequent increments return 2, 3, ..."""
        ctrl = RetryController(default_max_retries=5)
        results = [ctrl.increment("node_a") for _ in range(4)]
        assert results == [1, 2, 3, 4]

    def test_increment_independent_per_node(self):
        """Each node has its own independent retry counter."""
        ctrl = RetryController(default_max_retries=5)
        ctrl.increment("node_a")
        ctrl.increment("node_a")
        ctrl.increment("node_b")
        assert ctrl.get_retry_count("node_a") == 2
        assert ctrl.get_retry_count("node_b") == 1

    def test_increment_stores_error(self):
        """increment() stores the error that caused the retry."""
        ctrl = RetryController(default_max_retries=3)
        err = ValueError("connection failed")
        ctrl.increment("node_a", error=err)
        assert ctrl.get_last_error("node_a") is err

    def test_increment_updates_last_error(self):
        """Each increment can update the last error."""
        ctrl = RetryController(default_max_retries=3)
        err1 = ValueError("first failure")
        err2 = TimeoutError("second failure")
        ctrl.increment("node_a", error=err1)
        ctrl.increment("node_a", error=err2)
        assert ctrl.get_last_error("node_a") is err2

    def test_increment_without_error_preserves_previous(self):
        """increment() without error does not clear the previous error."""
        ctrl = RetryController(default_max_retries=3)
        err = ValueError("failure")
        ctrl.increment("node_a", error=err)
        ctrl.increment("node_a")  # no error
        assert ctrl.get_last_error("node_a") is err


class TestRetryControllerGetRetryCount:
    """Tests for get_retry_count() method."""

    def test_unknown_node_returns_zero(self):
        """get_retry_count() returns 0 for nodes that have never been retried."""
        ctrl = RetryController()
        assert ctrl.get_retry_count("nonexistent") == 0

    def test_returns_current_count(self):
        """get_retry_count() returns the number of retries so far."""
        ctrl = RetryController(default_max_retries=5)
        ctrl.increment("node_a")
        ctrl.increment("node_a")
        assert ctrl.get_retry_count("node_a") == 2


class TestRetryControllerAcceptRetry:
    """Tests for accept_retry() method."""

    def test_accept_retry_under_limit(self):
        """accept_retry() returns True when retry_count < max_retries."""
        ctrl = RetryController(default_max_retries=3)
        ctrl.increment("node_a")  # count=1
        assert ctrl.accept_retry("node_a") is True

    def test_reject_retry_at_limit(self):
        """accept_retry() returns False when retry_count == max_retries."""
        ctrl = RetryController(default_max_retries=2)
        ctrl.increment("node_a")  # count=1
        ctrl.increment("node_a")  # count=2
        assert ctrl.accept_retry("node_a") is False

    def test_reject_retry_above_limit(self):
        """accept_retry() returns False when retry_count > max_retries."""
        ctrl = RetryController(default_max_retries=1)
        ctrl.increment("node_a")
        ctrl.increment("node_a")
        assert ctrl.accept_retry("node_a") is False

    def test_accept_retry_uses_per_node_limit(self):
        """accept_retry() respects per-node max_retries over default."""
        ctrl = RetryController(default_max_retries=0)
        ctrl.set_node_max_retries("node_a", 2)
        assert ctrl.accept_retry("node_a") is True
        ctrl.increment("node_a")
        assert ctrl.accept_retry("node_a") is True
        ctrl.increment("node_a")
        assert ctrl.accept_retry("node_a") is False


class TestRetryControllerCheckRetryLimit:
    """Tests for check_retry_limit() method."""

    def test_no_error_under_limit(self):
        """check_retry_limit() does not raise when under limit."""
        ctrl = RetryController(default_max_retries=3)
        ctrl.increment("node_a")
        ctrl.check_retry_limit("node_a")  # Should not raise

    def test_raises_at_limit(self):
        """check_retry_limit() raises RetryLimitExceededError at limit."""
        ctrl = RetryController(default_max_retries=2)
        ctrl.increment("node_a")
        ctrl.increment("node_a")
        with pytest.raises(RetryLimitExceededError) as exc_info:
            ctrl.check_retry_limit("node_a")
        assert exc_info.value.task_id == "node_a"
        assert exc_info.value.retry_count == 2
        assert exc_info.value.max_retries == 2

    def test_raises_with_last_error(self):
        """check_retry_limit() includes last_error in the exception."""
        ctrl = RetryController(default_max_retries=1)
        err = ConnectionError("server down")
        ctrl.increment("node_a", error=err)
        with pytest.raises(RetryLimitExceededError) as exc_info:
            ctrl.check_retry_limit("node_a")
        assert exc_info.value.last_error is err

    def test_no_error_for_unknown_node(self):
        """check_retry_limit() does not raise for nodes with no retries (count=0)."""
        ctrl = RetryController(default_max_retries=3)
        ctrl.check_retry_limit("nonexistent")  # count=0 < 3, no error

    def test_raises_at_zero_limit(self):
        """check_retry_limit() raises immediately at default_max_retries=0 after first failure."""
        ctrl = RetryController(default_max_retries=0)
        # No retries allowed — but check_retry_limit is only called after increment
        # so count=0 < 0 is False... actually 0 >= 0 is True
        # This means: with max_retries=0, no retries are allowed at all
        # Even before any increment, check should pass (count=0 == max=0)
        with pytest.raises(RetryLimitExceededError):
            ctrl.check_retry_limit("node_a")


class TestRetryControllerNodeMaxRetries:
    """Tests for per-node max_retries configuration."""

    def test_get_max_retries_default(self):
        """get_max_retries_for_node() returns default when no per-node value set."""
        ctrl = RetryController(default_max_retries=3)
        assert ctrl.get_max_retries_for_node("any_node") == 3

    def test_set_and_get_node_max_retries(self):
        """set_node_max_retries() overrides default for that node."""
        ctrl = RetryController(default_max_retries=0)
        ctrl.set_node_max_retries("node_a", 5)
        assert ctrl.get_max_retries_for_node("node_a") == 5
        assert ctrl.get_max_retries_for_node("node_b") == 0  # unchanged

    def test_override_node_max_retries(self):
        """set_node_max_retries() can be called multiple times to update."""
        ctrl = RetryController(default_max_retries=0)
        ctrl.set_node_max_retries("node_a", 3)
        ctrl.set_node_max_retries("node_a", 5)
        assert ctrl.get_max_retries_for_node("node_a") == 5


class TestRetryControllerLastError:
    """Tests for last error tracking."""

    def test_no_last_error_initially(self):
        """get_last_error() returns None for nodes with no errors."""
        ctrl = RetryController()
        assert ctrl.get_last_error("node_a") is None

    def test_last_error_tracked_per_node(self):
        """Each node tracks its own last error independently."""
        ctrl = RetryController(default_max_retries=5)
        err_a = ValueError("error A")
        err_b = TypeError("error B")
        ctrl.increment("node_a", error=err_a)
        ctrl.increment("node_b", error=err_b)
        assert ctrl.get_last_error("node_a") is err_a
        assert ctrl.get_last_error("node_b") is err_b


class TestRetryControllerMultipleNodes:
    """Tests for retry tracking across multiple independent nodes."""

    def test_independent_retry_counts(self):
        """Retrying one node does not affect another."""
        ctrl = RetryController(default_max_retries=5)
        ctrl.increment("node_a")
        ctrl.increment("node_a")
        ctrl.increment("node_b")
        assert ctrl.get_retry_count("node_a") == 2
        assert ctrl.get_retry_count("node_b") == 1

    def test_one_node_exhausted_others_fine(self):
        """One node exhausting retries does not block other nodes."""
        ctrl = RetryController(default_max_retries=2)
        ctrl.increment("node_a")
        ctrl.increment("node_a")  # exhausted
        ctrl.increment("node_b")  # still has budget

        assert ctrl.accept_retry("node_a") is False
        assert ctrl.accept_retry("node_b") is True

    def test_per_node_limits_independent(self):
        """Per-node limits are independent."""
        ctrl = RetryController(default_max_retries=0)
        ctrl.set_node_max_retries("node_a", 1)
        ctrl.set_node_max_retries("node_b", 3)
        ctrl.increment("node_a")

        assert ctrl.accept_retry("node_a") is False
        assert ctrl.accept_retry("node_b") is True


class TestRetryPolicyDefaults:
    """Tests for RetryPolicy default values."""

    def test_default_values(self):
        policy = RetryPolicy()
        assert policy.max_retries == 0
        assert policy.initial_interval == 1.0
        assert policy.backoff_factor == 2.0
        assert policy.max_interval == 60.0
        assert policy.jitter is False

    def test_custom_values(self):
        policy = RetryPolicy(max_retries=5, initial_interval=0.5, backoff_factor=3.0, max_interval=30.0, jitter=True)
        assert policy.max_retries == 5
        assert policy.initial_interval == 0.5
        assert policy.backoff_factor == 3.0
        assert policy.max_interval == 30.0
        assert policy.jitter is True


class TestRetryPolicyGetDelay:
    """Tests for RetryPolicy.get_delay() calculation."""

    def test_first_retry_delay(self):
        """First retry (retry_count=0) returns initial_interval."""
        policy = RetryPolicy(max_retries=3, initial_interval=1.0, backoff_factor=2.0)
        assert policy.get_delay(0) == 1.0

    def test_exponential_backoff(self):
        """Delays follow exponential backoff: initial * factor^retry_count."""
        policy = RetryPolicy(max_retries=5, initial_interval=1.0, backoff_factor=2.0)
        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 4.0
        assert policy.get_delay(3) == 8.0

    def test_custom_backoff_factor(self):
        """Custom backoff factor (e.g., 3.0) is applied correctly."""
        policy = RetryPolicy(max_retries=3, initial_interval=0.5, backoff_factor=3.0)
        assert policy.get_delay(0) == 0.5
        assert policy.get_delay(1) == 1.5
        assert policy.get_delay(2) == 4.5

    def test_max_interval_cap(self):
        """Delay is capped at max_interval."""
        policy = RetryPolicy(max_retries=10, initial_interval=1.0, backoff_factor=10.0, max_interval=50.0)
        # 1.0 * 10^3 = 1000, capped at 50
        assert policy.get_delay(3) == 50.0

    def test_jitter_within_range(self):
        """With jitter=True, delay is within [0.5 * base, 1.5 * base]."""
        policy = RetryPolicy(max_retries=3, initial_interval=10.0, backoff_factor=1.0, jitter=True)
        # With factor=1.0, base delay is always 10.0
        # Jitter range: [5.0, 15.0]
        for _ in range(50):
            delay = policy.get_delay(0)
            assert 5.0 <= delay <= 15.0

    def test_no_backoff_factor_one(self):
        """backoff_factor=1.0 means constant delay."""
        policy = RetryPolicy(max_retries=5, initial_interval=2.0, backoff_factor=1.0)
        for i in range(5):
            assert policy.get_delay(i) == 2.0


class TestRetryControllerPolicy:
    """Tests for RetryController with RetryPolicy."""

    def test_set_node_policy_sets_max_retries(self):
        """set_node_policy() also sets max_retries from the policy."""
        ctrl = RetryController()
        policy = RetryPolicy(max_retries=3)
        ctrl.set_node_policy("node_a", policy)
        assert ctrl.get_max_retries_for_node("node_a") == 3

    def test_get_policy_for_node_returns_none_by_default(self):
        """get_policy_for_node() returns None when no policy is set."""
        ctrl = RetryController()
        assert ctrl.get_policy_for_node("node_a") is None

    def test_get_policy_for_node_returns_set_policy(self):
        """get_policy_for_node() returns the policy that was set."""
        ctrl = RetryController()
        policy = RetryPolicy(max_retries=2, initial_interval=0.5)
        ctrl.set_node_policy("node_a", policy)
        assert ctrl.get_policy_for_node("node_a") is policy

    def test_accept_retry_with_policy(self):
        """accept_retry() works correctly with a policy-based max_retries."""
        ctrl = RetryController()
        ctrl.set_node_policy("node_a", RetryPolicy(max_retries=2))
        assert ctrl.accept_retry("node_a") is True
        ctrl.increment("node_a")
        assert ctrl.accept_retry("node_a") is True
        ctrl.increment("node_a")
        assert ctrl.accept_retry("node_a") is False


class TestRetryControllerGetDelay:
    """Tests for RetryController.get_delay()."""

    def test_no_policy_returns_zero(self):
        """Without a policy, get_delay() returns 0.0 (backward compatible)."""
        ctrl = RetryController(default_max_retries=3)
        ctrl.increment("node_a")
        assert ctrl.get_delay("node_a") == 0.0

    def test_delay_after_first_increment(self):
        """After first increment (count=1), delay uses retry_count=0."""
        ctrl = RetryController()
        ctrl.set_node_policy("node_a", RetryPolicy(max_retries=3, initial_interval=1.0, backoff_factor=2.0))
        ctrl.increment("node_a")  # count becomes 1
        assert ctrl.get_delay("node_a") == 1.0  # policy.get_delay(0) = 1.0

    def test_delay_increases_with_retries(self):
        """Delay increases exponentially with each increment."""
        ctrl = RetryController()
        ctrl.set_node_policy("node_a", RetryPolicy(max_retries=5, initial_interval=1.0, backoff_factor=2.0))
        ctrl.increment("node_a")  # count=1 -> delay(0)=1.0
        assert ctrl.get_delay("node_a") == 1.0
        ctrl.increment("node_a")  # count=2 -> delay(1)=2.0
        assert ctrl.get_delay("node_a") == 2.0
        ctrl.increment("node_a")  # count=3 -> delay(2)=4.0
        assert ctrl.get_delay("node_a") == 4.0

    def test_delay_capped_by_max_interval(self):
        """Delay is capped at max_interval from the policy."""
        ctrl = RetryController()
        ctrl.set_node_policy(
            "node_a", RetryPolicy(max_retries=10, initial_interval=1.0, backoff_factor=10.0, max_interval=5.0)
        )
        ctrl.increment("node_a")
        ctrl.increment("node_a")  # count=2 -> delay(1) = 1.0*10^1 = 10.0, capped at 5.0
        assert ctrl.get_delay("node_a") == 5.0

    def test_delay_zero_before_any_increment(self):
        """Before any increment (count=0), get_delay uses max(0, -1) = 0."""
        ctrl = RetryController()
        ctrl.set_node_policy("node_a", RetryPolicy(max_retries=3, initial_interval=1.0, backoff_factor=2.0))
        # count=0, delay(max(0, -1)) = delay(0) = 1.0
        # Actually this edge case: count=0, so get_delay(max(0, 0-1)) = get_delay(0) = 1.0
        assert ctrl.get_delay("node_a") == 1.0
