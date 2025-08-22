"""Workflow execution engine for graflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from graflow import exceptions
from graflow.core.graph import TaskGraph

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
        Raises:
            exceptions.GraflowRuntimeError: If execution fails due to a runtime error
        """
        assert context.graph is not None, "Graph must be set before execution"

        print(f"Starting execution from: {context.start_node}")

        while not context.is_completed():
            task_id = context.get_next_task()
            if task_id is None:
                break

            # Check if task exists in graph
            graph = context.graph.nx_graph()
            if task_id not in graph.nodes:
                print(f"Warning: Node {task_id} not found in graph")
                continue

            # Execute the task
            task = graph.nodes[task_id]["task"]

            # Execute task with proper context management
            try:
                with context.executing_task(task) as _ctx:
                    result = task.run()
                    context.set_result(task_id, result)
                    context.mark_executed(task_id)
            except Exception as e:
                context.set_result(task_id, e)
                raise exceptions.as_runtime_error(e) from e

            context.increment_step()

            # Handle task completion and successor scheduling
            if context.goto_called:
                # If goto was called, skip successors entirely
                print(f"🚫 Goto called in {task_id}, skipping successors")
            else:
                # Add successor nodes to queue
                for succ in graph.successors(task_id):
                    succ_task = graph.nodes[succ]["task"]
                    context.add_to_queue(succ_task)

        print(f"Execution completed after {context.steps} steps")

    def execute_with_cycles(self, graph: TaskGraph, start_node: str, max_steps: int = 10) -> None:
        """Execute tasks allowing cycles from global graph.

        Args:
            graph: The workflow graph to execute
            start_node: Starting node for execution
            max_steps: Maximum number of execution steps
        """
        from .context import ExecutionContext

        # Create ExecutionContext and delegate to it
        exec_context = ExecutionContext.create(graph, start_node, max_steps=max_steps)
        self.execute(exec_context)
