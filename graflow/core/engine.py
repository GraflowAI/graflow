"""Workflow execution engine for graflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from graflow import exceptions
from graflow.core.graph import TaskGraph

if TYPE_CHECKING:
    from .context import ExecutionContext


class WorkflowEngine:
    """Workflow execution engine for unified task execution."""

    def __init__(self) -> None:
        """Initialize the workflow engine."""
        pass

    def execute(self, context: ExecutionContext, start_task_id: Optional[str] = None) -> None:
        """Execute workflow or single task using the provided context.

        Args:
            context: ExecutionContext containing the execution state and graph
            start_task_id: Optional task ID to start execution from. If None, uses context.get_next_task()
        Raises:
            exceptions.GraflowRuntimeError: If execution fails due to a runtime error
        """
        assert context.graph is not None, "Graph must be set before execution"

        print(f"Starting execution from: {start_task_id or context.start_node}")

        # Initialize first task
        if start_task_id is not None:
            task_id = start_task_id
        else:
            task_id = context.get_next_task()

        while task_id is not None and not context.is_completed():
            # Reset goto flag for each task
            context.reset_goto_flag()

            # Check if task exists in graph
            graph = context.graph
            if task_id not in graph.nodes:
                print(f"Error: Node {task_id} not found in graph")
                break  # Terminate execution

            # Execute the task
            task = graph.get_node(task_id)

            # Execute task with proper context management
            try:
                with context.executing_task(task):
                    result = task.run()
                    context.set_result(task_id, result)
            except Exception as e:
                context.set_result(task_id, e)
                raise exceptions.as_runtime_error(e) from e

            # Step increment for all executed tasks
            context.increment_step()

            # Handle successor scheduling
            if context.goto_called:
                print(f"🚫 Goto called in {task_id}, skipping successors")
            else:
                # Add successor nodes to queue
                for succ in graph.successors(task_id):
                    succ_task = graph.get_node(succ)
                    context.add_to_queue(succ_task)

            task_id = context.get_next_task()

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
