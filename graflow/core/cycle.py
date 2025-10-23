# graflow/core/cycle.py
# This file is part of Graflow, a graph-based workflow management system.
# It implements cycle control to prevent infinite loops in task execution.

from typing import Dict, Optional


class CycleController:
    """Controls cycle execution and prevents infinite loops."""

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
        """Check if node can be executed based on iteration count."""
        if iteration is None:
            iteration = self.cycle_counts.get(node_id, 0)
        max_cycles = self.get_max_cycles_for_node(node_id)
        return iteration < max_cycles

    def register_cycle(self, node_id: str) -> int:
        """Register a cycle execution and return current count."""
        self.cycle_counts[node_id] = self.cycle_counts.get(node_id, 0) + 1
        return self.cycle_counts[node_id]

    def get_cycle_count(self, node_id: str) -> int:
        """Return how many times the given node has executed (0 if never)."""
        return self.cycle_counts.get(node_id, 0)
