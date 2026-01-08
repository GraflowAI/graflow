"""Workflow execution engine for graflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from graflow import exceptions
from graflow.core.graph import TaskGraph

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.handler import TaskHandler
    from graflow.core.task import Executable
    from graflow.hitl.types import FeedbackTimeoutError

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """Workflow execution engine for unified task execution."""

    def __init__(self) -> None:
        """Initialize the workflow engine."""
        self._handlers: dict[str, TaskHandler] = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default handlers."""
        from graflow.core.handlers.direct import DirectTaskHandler

        self._handlers["direct"] = DirectTaskHandler()

    def _get_handler(self, task: Executable) -> TaskHandler:
        """Get handler for the given task based on its handler_type.

        Args:
            task: Executable task with handler_type attribute

        Returns:
            TaskHandler instance

        Raises:
            ValueError: If handler_type is unknown
        """
        handler_type = getattr(task, "handler_type", "direct")

        if handler_type not in self._handlers:
            raise ValueError(f"Unknown handler type: {handler_type}. Supported: {', '.join(self._handlers.keys())}")

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

    def _handle_feedback_timeout(
        self,
        error: FeedbackTimeoutError,  # FeedbackTimeoutError (imported dynamically to avoid circular import)
        task_id: str,
        task: Executable,
        context: ExecutionContext,
    ) -> None:
        """Handle feedback timeout by creating checkpoint and exiting.

        Args:
            error: FeedbackTimeoutError exception with feedback_id and timeout attributes
            task_id: ID of task that timed out waiting for feedback
            context: Execution context

        Note:
            This method is called only when error is an instance of FeedbackTimeoutError.
            Type is Any to avoid circular import issues.
        """
        logger.info(
            "Feedback timeout for %s, creating checkpoint",
            error.feedback_id,
            extra={
                "feedback_id": error.feedback_id,
                "task_id": task_id,
                "session_id": context.session_id,
                "timeout": error.timeout,
            },
        )

        # Create checkpoint with resuming task_id (task will be resolved from graph on resume)
        from graflow.core.checkpoint import CheckpointManager

        checkpoint_path, checkpoint_metadata = CheckpointManager.create_checkpoint(
            context,
            metadata={
                "reason": "feedback_timeout",
                "feedback_id": error.feedback_id,
                "task_id": task_id,
                "timeout": error.timeout,
            },
            resuming_task_id=task_id,
        )

        logger.info(
            "Checkpoint created at: %s - Waiting for feedback: %s",
            checkpoint_path,
            error.feedback_id,
            extra={
                "checkpoint_path": checkpoint_path,
                "checkpoint_id": checkpoint_metadata.checkpoint_id,
                "feedback_id": error.feedback_id,
                "session_id": context.session_id,
            },
        )

        # Store checkpoint info in context
        context.checkpoint_metadata = checkpoint_metadata.to_dict()
        context.last_checkpoint_path = checkpoint_path

        # Update FeedbackRequest with checkpoint information
        feedback_manager = context.feedback_manager
        feedback_request = feedback_manager.get_request(error.feedback_id)

        if feedback_request:
            # Add checkpoint info to metadata
            feedback_request.metadata.update(
                {
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_id": checkpoint_metadata.checkpoint_id,
                    "checkpoint_created_at": checkpoint_metadata.created_at,
                    "checkpoint_steps": checkpoint_metadata.steps,
                    "checkpoint_reason": "feedback_timeout",
                }
            )

            # Update request in backend
            feedback_manager.store_request(feedback_request)

            logger.info(
                "Updated feedback request with checkpoint info",
                extra={
                    "feedback_id": error.feedback_id,
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_id": checkpoint_metadata.checkpoint_id,
                },
            )

    def _handle_deferred_checkpoint(self, context: ExecutionContext) -> None:
        """Handle deferred checkpoint requests.

        Args:
            context: Execution context with checkpoint request
        """
        from graflow.core.checkpoint import CheckpointManager

        checkpoint_path, checkpoint_metadata = CheckpointManager.create_checkpoint(
            context,
            path=context.checkpoint_request_path,
            metadata=context.checkpoint_request_metadata,
        )
        logger.info(
            "Checkpoint created at: %s",
            checkpoint_path,
            extra={
                "session_id": context.session_id,
                "checkpoint_id": checkpoint_metadata.checkpoint_id,
                "checkpoint_steps": checkpoint_metadata.steps,
            },
        )
        context.checkpoint_metadata = checkpoint_metadata.to_dict()
        context.last_checkpoint_path = checkpoint_path
        context.clear_checkpoint_request()

    def execute(  # noqa: PLR0912
        self, context: ExecutionContext, start_task_id: Optional[str] = None
    ) -> Any:
        """Execute workflow or single task using the provided context.

        Args:
            context: ExecutionContext containing the execution state and graph
            start_task_id: Optional task ID to start execution from. If None, uses context.get_next_task()
        Raises:
            exceptions.GraflowRuntimeError: If execution fails due to a runtime error
        Returns:
            The result returned by the last executed handler (which may be ``None``).
        """
        assert context.graph is not None, "Graph must be set before execution"

        # Determine workflow name for tracing
        workflow_name = getattr(context.graph, "name", None) or f"workflow_{context.session_id[:8]}"

        # Call tracer hook: workflow start (skip for nested contexts to avoid duplicate traces)
        # Nested contexts (branches/parallel groups) are already tracked via parent context's tracer
        # Only the root context initiates workflow-level tracing
        if context.parent_context is None:
            context.tracer.on_workflow_start(workflow_name, context)

        logger.info(
            "Starting execution from task: %s",
            start_task_id or context.start_node,
            extra={"session_id": context.session_id, "workflow": workflow_name},
        )

        # Initialize first task
        if start_task_id is not None:
            task_id = start_task_id
        else:
            task_id = context.get_next_task()

        last_result: Any = None

        # Execute tasks while we have tasks and haven't exceeded max steps
        # Note: Don't check is_completed() here as it would return True immediately
        # after dequeuing the first task (queue becomes empty)
        graph = context.graph
        while task_id is not None and context.steps < context.max_steps:
            # Reset control flow flags for each task
            context.reset_control_flags()

            # Check if task exists in graph
            if task_id not in graph.nodes:
                logger.error(
                    "Node not found in graph: %s",
                    task_id,
                    extra={"session_id": context.session_id, "available_nodes": list(graph.nodes.keys())[:10]},
                )
                raise exceptions.GraflowRuntimeError(
                    f"Node '{task_id}' not found in graph. "
                    f"Available nodes: {', '.join(list(graph.nodes.keys())[:5])}..."
                )

            # Execute the task
            task = graph.get_node(task_id)

            # Execute task with proper context management
            try:
                with context.executing_task(task):
                    # Execute task using handler
                    # Handler is responsible for setting result
                    last_result = self._execute_task(task, context)
            except Exception as e:
                # Check if this is a FeedbackTimeoutError (HITL)
                from graflow.hitl.types import FeedbackTimeoutError

                if isinstance(e, FeedbackTimeoutError):
                    # Handle feedback timeout by creating checkpoint
                    self._handle_feedback_timeout(e, task_id, task, context)
                    # Exit workflow (feedback pending)
                    # Return early to allow external process to provide feedback
                    return None

                # Exception already stored by handler, just re-raise
                raise exceptions.as_runtime_error(e)

            # Handle control flow and successor scheduling
            if context.cancel_called:
                # Workflow cancellation requested (abnormal exit)
                # Do NOT mark task as completed - this is a failure
                logger.warning(
                    "Workflow cancellation requested by task '%s': %s",
                    task_id,
                    context.ctrl_message,
                    extra={"task_id": task_id, "session_id": context.session_id, "ctrl_message": context.ctrl_message},
                )
                # Raise cancellation error immediately
                from graflow.exceptions import GraflowWorkflowCanceledError

                raise GraflowWorkflowCanceledError(context.ctrl_message or "Workflow canceled", task_id=task_id)

            # Common post-task processing (for successful tasks)
            context.mark_task_completed(task_id)
            context.increment_step()

            # Handle deferred checkpoint requests
            if context.checkpoint_requested:
                self._handle_deferred_checkpoint(context)

            if context.terminate_called:
                # Workflow termination requested (normal exit)
                logger.info(
                    "Workflow termination requested by task '%s': %s",
                    task_id,
                    context.ctrl_message,
                    extra={"task_id": task_id, "session_id": context.session_id, "ctrl_message": context.ctrl_message},
                )
                # Exit workflow execution loop
                break

            elif context.goto_called:
                # Goto called: skip normal successors
                logger.debug(
                    "Goto called in task, skipping normal successors",
                    extra={"task_id": task_id, "session_id": context.session_id},
                )

            else:
                # Normal successor processing: add successor nodes to queue
                successors = list(graph.successors(task_id))
                for succ in successors:
                    succ_task = graph.get_node(succ)
                    context.add_to_queue(succ_task)

            # Get next task from queue
            task_id = context.get_next_task()

        logger.info(
            "Execution completed",
            extra={"session_id": context.session_id, "steps": context.steps, "workflow": workflow_name},
        )

        # Call tracer hook: workflow end (skip for nested contexts to avoid duplicate traces)
        # Nested contexts (branches/parallel groups) are already tracked via parent context's tracer
        # Only the root context closes workflow-level tracing
        if context.parent_context is None:
            context.tracer.on_workflow_end(workflow_name, context, result=last_result)

        return last_result

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
