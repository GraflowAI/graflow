# graflow/core/cycle.py
# This file is part of Graflow, a graph-based workflow management system.
# It implements cycle control to prevent infinite loops in task execution.

from typing import Dict, Optional

from graflow.exceptions import CycleLimitExceededError


class CycleController:
    """Controls cycle execution and prevents infinite loops.

    This is the single source of truth for cycle state.
    TaskExecutionContext delegates all cycle operations here.
    """

    def __init__(self, default_max_cycles: int = 100):
        self.default_max_cycles: int = default_max_cycles
        self.cycle_counts: Dict[str, int] = {}
        self.node_max_cycles: Dict[str, int] = {}

    def set_node_max_cycles(self, node_id: str, max_cycles: int) -> None:
        """Set maximum cycle count for a specific node."""
        self.node_max_cycles[node_id] = max_cycles

    def get_max_cycles_for_node(self, node_id: str) -> int:
        """Get maximum cycle count for a node (node-specific or default)."""
        return self.node_max_cycles.get(node_id, self.default_max_cycles)

    def can_execute(self, node_id: str, iteration: Optional[int] = None) -> bool:
        """Check if node can execute another cycle.

        Args:
            node_id: The node identifier
            iteration: Optional explicit iteration count to check.
                      If None, uses the current cycle count for the node.
        """
        if iteration is None:
            iteration = self.cycle_counts.get(node_id, 0)
        return iteration < self.get_max_cycles_for_node(node_id)

    def register_cycle(self, node_id: str) -> int:
        """Register a cycle execution and return new count.

        Raises:
            CycleLimitExceededError: If the cycle limit has been reached.
        """
        count = self.cycle_counts.get(node_id, 0)
        max_cycles = self.get_max_cycles_for_node(node_id)
        if count >= max_cycles:
            raise CycleLimitExceededError(task_id=node_id, cycle_count=count, max_cycles=max_cycles)
        count += 1
        self.cycle_counts[node_id] = count
        return count

    def get_cycle_count(self, node_id: str) -> int:
        """Return how many times the given node has executed (0 if never)."""
        return self.cycle_counts.get(node_id, 0)
