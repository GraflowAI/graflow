"""Unit tests for CycleController.

Tests cover:
  - increment() 1-based counting
  - accept_next_cycle() with default and per-node limits
  - check_cycle_limit() raising CycleLimitExceededError
  - get_cycle_count() for unknown nodes
  - Per-node max_cycles override via set_node_max_cycles()
  - Independent cycle tracking across multiple nodes
"""

import pytest

from graflow.core.cycle import CycleController
from graflow.exceptions import CycleLimitExceededError


class TestCycleControllerIncrement:
    """Tests for increment() method."""

    def test_first_increment_returns_one(self):
        """First increment for a node returns 1 (1-based)."""
        ctrl = CycleController(default_max_cycles=10)
        assert ctrl.increment("node_a") == 1

    def test_subsequent_increments(self):
        """Subsequent increments return 2, 3, 4, ..."""
        ctrl = CycleController(default_max_cycles=10)
        results = [ctrl.increment("node_a") for _ in range(5)]
        assert results == [1, 2, 3, 4, 5]

    def test_increment_independent_per_node(self):
        """Each node has its own independent counter."""
        ctrl = CycleController(default_max_cycles=10)
        ctrl.increment("node_a")
        ctrl.increment("node_a")
        ctrl.increment("node_b")
        assert ctrl.get_cycle_count("node_a") == 2
        assert ctrl.get_cycle_count("node_b") == 1

    def test_increment_does_not_check_limit(self):
        """increment() never raises even when exceeding max_cycles."""
        ctrl = CycleController(default_max_cycles=2)
        # Should not raise even though count exceeds max_cycles
        assert ctrl.increment("node_a") == 1
        assert ctrl.increment("node_a") == 2
        assert ctrl.increment("node_a") == 3  # Exceeds limit, but no error


class TestCycleControllerGetCycleCount:
    """Tests for get_cycle_count() method."""

    def test_unknown_node_returns_zero(self):
        """get_cycle_count() returns 0 for nodes that have never executed."""
        ctrl = CycleController()
        assert ctrl.get_cycle_count("nonexistent") == 0

    def test_returns_current_count(self):
        """get_cycle_count() returns the number of times increment() was called."""
        ctrl = CycleController()
        ctrl.increment("node_a")
        ctrl.increment("node_a")
        assert ctrl.get_cycle_count("node_a") == 2


class TestCycleControllerCanExecute:
    """Tests for accept_next_cycle() method."""

    def test_accept_next_cycle_before_any_increment(self):
        """accept_next_cycle() returns True when no increments have occurred."""
        ctrl = CycleController(default_max_cycles=3)
        assert ctrl.accept_next_cycle("node_a") is True

    def test_accept_next_cycle_under_limit(self):
        """accept_next_cycle() returns True when count < max_cycles."""
        ctrl = CycleController(default_max_cycles=3)
        ctrl.increment("node_a")  # count=1
        ctrl.increment("node_a")  # count=2
        assert ctrl.accept_next_cycle("node_a") is True

    def test_cannot_execute_at_limit(self):
        """accept_next_cycle() returns False when count == max_cycles."""
        ctrl = CycleController(default_max_cycles=3)
        ctrl.increment("node_a")  # count=1
        ctrl.increment("node_a")  # count=2
        ctrl.increment("node_a")  # count=3
        assert ctrl.accept_next_cycle("node_a") is False

    def test_cannot_execute_above_limit(self):
        """accept_next_cycle() returns False when count > max_cycles."""
        ctrl = CycleController(default_max_cycles=2)
        ctrl.increment("node_a")
        ctrl.increment("node_a")
        ctrl.increment("node_a")  # count=3 > max=2
        assert ctrl.accept_next_cycle("node_a") is False

    def test_accept_next_cycle_reflects_current_count(self):
        """accept_next_cycle() always uses the current stored cycle count."""
        ctrl = CycleController(default_max_cycles=3)
        assert ctrl.accept_next_cycle("node_a") is True   # count=0
        ctrl.increment("node_a")                      # count=1
        assert ctrl.accept_next_cycle("node_a") is True
        ctrl.increment("node_a")                      # count=2
        assert ctrl.accept_next_cycle("node_a") is True
        ctrl.increment("node_a")                      # count=3 == max
        assert ctrl.accept_next_cycle("node_a") is False

    def test_accept_next_cycle_uses_per_node_limit(self):
        """accept_next_cycle() respects per-node max_cycles over default."""
        ctrl = CycleController(default_max_cycles=100)
        ctrl.set_node_max_cycles("node_a", 2)
        ctrl.increment("node_a")  # count=1
        ctrl.increment("node_a")  # count=2
        assert ctrl.accept_next_cycle("node_a") is False
        # Other nodes still use default
        ctrl.increment("node_b")
        assert ctrl.accept_next_cycle("node_b") is True


class TestCycleControllerCheckCycleLimit:
    """Tests for check_cycle_limit() method."""

    def test_no_error_under_limit(self):
        """check_cycle_limit() does not raise when under limit."""
        ctrl = CycleController(default_max_cycles=3)
        ctrl.increment("node_a")  # count=1
        ctrl.check_cycle_limit("node_a")  # Should not raise

    def test_raises_at_limit(self):
        """check_cycle_limit() raises CycleLimitExceededError when count == max_cycles."""
        ctrl = CycleController(default_max_cycles=3)
        ctrl.increment("node_a")
        ctrl.increment("node_a")
        ctrl.increment("node_a")  # count=3 == max_cycles
        with pytest.raises(CycleLimitExceededError) as exc_info:
            ctrl.check_cycle_limit("node_a")
        assert exc_info.value.task_id == "node_a"
        assert exc_info.value.cycle_count == 3
        assert exc_info.value.max_cycles == 3

    def test_raises_above_limit(self):
        """check_cycle_limit() raises when count > max_cycles."""
        ctrl = CycleController(default_max_cycles=2)
        for _ in range(5):
            ctrl.increment("node_a")
        with pytest.raises(CycleLimitExceededError) as exc_info:
            ctrl.check_cycle_limit("node_a")
        assert exc_info.value.cycle_count == 5
        assert exc_info.value.max_cycles == 2

    def test_no_error_for_unknown_node(self):
        """check_cycle_limit() does not raise for nodes with no increments (count=0)."""
        ctrl = CycleController(default_max_cycles=3)
        ctrl.check_cycle_limit("nonexistent")  # count=0 < 3, no error

    def test_uses_per_node_limit(self):
        """check_cycle_limit() uses per-node max_cycles if set."""
        ctrl = CycleController(default_max_cycles=100)
        ctrl.set_node_max_cycles("node_a", 2)
        ctrl.increment("node_a")
        ctrl.increment("node_a")  # count=2 == per-node max
        with pytest.raises(CycleLimitExceededError) as exc_info:
            ctrl.check_cycle_limit("node_a")
        assert exc_info.value.max_cycles == 2


class TestCycleControllerNodeMaxCycles:
    """Tests for per-node max_cycles configuration."""

    def test_get_max_cycles_default(self):
        """get_max_cycles_for_node() returns default when no per-node value set."""
        ctrl = CycleController(default_max_cycles=42)
        assert ctrl.get_max_cycles_for_node("any_node") == 42

    def test_set_and_get_node_max_cycles(self):
        """set_node_max_cycles() overrides default for that node."""
        ctrl = CycleController(default_max_cycles=100)
        ctrl.set_node_max_cycles("node_a", 5)
        assert ctrl.get_max_cycles_for_node("node_a") == 5
        assert ctrl.get_max_cycles_for_node("node_b") == 100  # unchanged

    def test_override_node_max_cycles(self):
        """set_node_max_cycles() can be called multiple times to update the limit."""
        ctrl = CycleController(default_max_cycles=100)
        ctrl.set_node_max_cycles("node_a", 5)
        ctrl.set_node_max_cycles("node_a", 10)
        assert ctrl.get_max_cycles_for_node("node_a") == 10

    def test_multiple_nodes_independent_limits(self):
        """Different nodes can have different per-node limits."""
        ctrl = CycleController(default_max_cycles=100)
        ctrl.set_node_max_cycles("node_a", 3)
        ctrl.set_node_max_cycles("node_b", 7)
        assert ctrl.get_max_cycles_for_node("node_a") == 3
        assert ctrl.get_max_cycles_for_node("node_b") == 7
        assert ctrl.get_max_cycles_for_node("node_c") == 100


class TestCycleControllerMultipleNodes:
    """Tests for cycle tracking across multiple independent nodes."""

    def test_independent_cycle_counts(self):
        """Incrementing one node does not affect another."""
        ctrl = CycleController(default_max_cycles=10)
        ctrl.increment("node_a")
        ctrl.increment("node_a")
        ctrl.increment("node_b")
        ctrl.increment("node_b")
        ctrl.increment("node_b")
        assert ctrl.get_cycle_count("node_a") == 2
        assert ctrl.get_cycle_count("node_b") == 3

    def test_one_node_at_limit_others_fine(self):
        """One node exceeding its limit does not block other nodes."""
        ctrl = CycleController(default_max_cycles=2)
        ctrl.increment("node_a")
        ctrl.increment("node_a")  # at limit
        ctrl.increment("node_b")  # under limit

        assert ctrl.accept_next_cycle("node_a") is False
        assert ctrl.accept_next_cycle("node_b") is True

    def test_per_node_limits_independent(self):
        """Per-node limits are independent; node_a's limit doesn't affect node_b."""
        ctrl = CycleController(default_max_cycles=100)
        ctrl.set_node_max_cycles("node_a", 2)
        ctrl.increment("node_a")
        ctrl.increment("node_a")
        ctrl.increment("node_b")
        ctrl.increment("node_b")

        # node_a at its per-node limit of 2
        with pytest.raises(CycleLimitExceededError):
            ctrl.check_cycle_limit("node_a")
        # node_b still under default limit of 100
        ctrl.check_cycle_limit("node_b")  # no error
