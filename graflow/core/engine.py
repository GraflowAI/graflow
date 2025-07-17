"""Workflow execution engine for graflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from .context import ExecutionContext


class WorkflowEngine:
    """Workflow execution engine with pluggable strategies."""

    def __init__(self, strategy: str = "sequential"):
        """Initialize the workflow engine.

        Args:
            strategy: Execution strategy (currently only "sequential" is supported)
        """
        self.strategy = strategy

    def execute(self, context: ExecutionContext) -> None:
        """Execute workflow using the provided context.

        Args:
            context: ExecutionContext containing the execution state and graph
        """
        assert context.graph is not None, "Graph must be set before execution"

        print(f"Starting execution from: {context.start_node}")

        while not context.is_completed():
            node = context.get_next_node()
            if node is None:
                break

            # Check if node exists in graph
            if node not in context.graph.nodes:
                print(f"Warning: Node {node} not found in graph")
                continue

            # Execute the task
            task = context.graph.nodes[node]["task"]
            print(f"Running task: {node}")

            # Execute task and store result
            try:
                result = task.run()
                context.set_result(node, result)
                context.mark_executed(node)
            except Exception as e:
                print(f"Error executing task {node}: {e}")
                context.set_result(node, e)

            context.increment_step()

            # Add successor nodes to queue
            for succ in context.graph.successors(node):
                context.add_to_queue(succ)

        print(f"Execution completed after {context.steps} steps")

    def execute_with_cycles(self, graph: nx.DiGraph, start_node: str, max_steps: int = 10) -> None:
        """Execute tasks allowing cycles from global graph.

        Args:
            graph: The workflow graph to execute
            start_node: Starting node for execution
            max_steps: Maximum number of execution steps
        """
        from .context import ExecutionContext  # noqa: PLC0415

        # Create ExecutionContext and delegate to it
        exec_context = ExecutionContext.create(graph, start_node, max_steps=max_steps)
        self.execute(exec_context)
