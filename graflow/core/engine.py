"""Workflow execution engine for graflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from graflow import exceptions
from graflow.core.graph import TaskGraph

if TYPE_CHECKING:
    from .context import ExecutionContext
    from .handler import TaskHandler
    from .task import Executable


class WorkflowEngine:
    """Workflow execution engine for unified task execution."""

    def __init__(self) -> None:
        """Initialize the workflow engine."""
        self._handlers: dict[str, TaskHandler] = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default handlers."""
        from graflow.core.handlers.direct import DirectTaskHandler
        self._handlers['direct'] = DirectTaskHandler()

    def _get_handler(self, task: Executable) -> TaskHandler:
        """Get handler for the given task based on its handler_type.

        Args:
            task: Executable task with handler_type attribute

        Returns:
            TaskHandler instance

        Raises:
            ValueError: If handler_type is unknown
        """
        handler_type = getattr(task, 'handler_type', 'direct')

        if handler_type not in self._handlers:
            raise ValueError(
                f"Unknown handler type: {handler_type}. "
                f"Supported: {', '.join(self._handlers.keys())}"
            )

        return self._handlers[handler_type]

    def register_handler(self, handler_type: str, handler: TaskHandler) -> None:
        """Register a custom handler.

        Args:
            handler_type: Handler type identifier
            handler: TaskHandler instance
        """
        self._handlers[handler_type] = handler

    def _execute_task(self, task: Executable, context: ExecutionContext) -> Any:
        """Execute task using appropriate handler.

        Args:
            task: Executable task with handler_type attribute
            context: Execution context

        Note:
            Handler is responsible for calling context.set_result()
        """
        handler = self._get_handler(task)
        return handler.execute_task(task, context)

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

        # Execute tasks while we have tasks and haven't exceeded max steps
        # Note: Don't check is_completed() here as it would return True immediately
        # after dequeuing the first task (queue becomes empty)
        while task_id is not None and context.steps < context.max_steps:
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
                    # Execute task using handler
                    # Handler is responsible for setting result
                    self._execute_task(task, context)
            except Exception as e:
                # Exception already stored by handler, just re-raise
                raise exceptions.as_runtime_error(e)

            # Handle successor scheduling
            if context.goto_called:
                print(f"🚫 Goto called in {task_id}, skipping successors")
            else:
                # Add successor nodes to queue
                successors = list(graph.successors(task_id))

                # Prevent re-queuing internal members of a ParallelGroup
                from graflow.core.task import ParallelGroup  # Local import to avoid cycle
                if isinstance(task, ParallelGroup):
                    member_ids = {member.task_id for member in task.tasks}
                    successors = [succ for succ in successors if succ not in member_ids]

                for succ in successors:
                    succ_task = graph.get_node(succ)
                    context.add_to_queue(succ_task)

            # Track completion and step count after scheduling
            context.mark_task_completed(task_id)
            context.increment_step()

            # Handle deferred checkpoint requests after queue updates
            if context.checkpoint_requested:
                from graflow.core.checkpoint import CheckpointManager

                checkpoint_path, checkpoint_metadata = CheckpointManager.create_checkpoint(
                    context,
                    path=context.checkpoint_request_path,
                    metadata=context.checkpoint_request_metadata,
                )
                print(f"Checkpoint created: {checkpoint_path}")
                context.checkpoint_metadata = checkpoint_metadata.to_dict()
                context.last_checkpoint_path = checkpoint_path
                context.clear_checkpoint_request()

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
