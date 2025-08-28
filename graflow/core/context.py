"""Execution engine for graflow with cycle support and global graph integration."""

from __future__ import annotations

import pickle
import time
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar, Union

from graflow.channels.base import Channel
from graflow.channels.memory import MemoryChannel
from graflow.channels.typed import TypedChannel
from graflow.coordination.executor import GroupExecutor
from graflow.core.cycle import CycleController
from graflow.core.engine import WorkflowEngine
from graflow.core.function_registry import TaskFunctionManager
from graflow.core.graph import TaskGraph
from graflow.exceptions import CycleLimitExceededError
from graflow.queue.base import TaskQueue, TaskSpec
from graflow.queue.factory import QueueBackend, TaskQueueFactory

if TYPE_CHECKING:
    from .task import Executable

T = TypeVar('T')


class TaskExecutionContext:
    """Per-task execution context managing task-specific state and cycles."""

    def __init__(self, task_id: str, execution_context: ExecutionContext):
        """Initialize task execution context.

        Args:
            task_id: The ID of the task being executed
            execution_context: Reference to the main ExecutionContext
        """
        self.task_id = task_id
        self.execution_context = execution_context
        self.start_time = time.time()
        self.cycle_count = 0
        self.max_cycles = execution_context.cycle_controller.get_max_cycles_for_node(task_id)
        self.retries = 0
        self.max_retries = execution_context.default_max_retries
        self.local_data: dict[str, Any] = {}

    @property
    def session_id(self) -> str:
        """Get session ID from execution context."""
        return self.execution_context.session_id

    def can_iterate(self) -> bool:
        """Check if this task can execute another cycle."""
        return self.cycle_count < self.max_cycles

    def register_cycle(self) -> int:
        """Register a cycle execution and return new count."""
        if not self.can_iterate():
            raise ValueError(
                f"Cycle limit exceeded for task {self.task_id}: "
                f"{self.cycle_count}/{self.max_cycles} cycles"
            )
        self.cycle_count += 1
        # Also register with the global cycle controller for consistency
        self.execution_context.cycle_controller.cycle_counts[self.task_id] = self.cycle_count
        return self.cycle_count

    def next_iteration(self, data: Any = None) -> str:
        """Create iteration task using this task's context."""
        return self.execution_context.next_iteration(data, self.task_id)

    def next_task(self, executable: Executable, goto: bool = False) -> str:
        """Create new dynamic task or jump to existing task."""
        return self.execution_context.next_task(executable, goto=goto)

    def get_channel(self) -> Channel:
        """Get communication channel."""
        return self.execution_context.get_channel()

    def get_typed_channel(self, message_type: Type[T]) -> TypedChannel[T]:
        """Get a type-safe communication channel.

        Args:
            message_type: TypedDict class defining message structure

        Returns:
            TypedChannel wrapper for type-safe communication
        """
        channel = self.execution_context.get_channel()
        return TypedChannel(channel, message_type)

    @property
    def function_manager(self) -> TaskFunctionManager:
        """Get the function manager instance from execution context."""
        return self.execution_context.function_manager

    def get_result(self, node: str, default: Any = None) -> Any:
        """Get execution result for a node from channel."""
        return self.execution_context.get_result(node, default)

    def set_local_data(self, key: str, value: Any) -> None:
        """Set task-local data."""
        self.local_data[key] = value

    def get_local_data(self, key: str, default: Any = None) -> Any:
        """Get task-local data."""
        return self.local_data.get(key, default)

    def elapsed_time(self) -> float:
        """Get elapsed time since task started."""
        return time.time() - self.start_time

    def __str__(self) -> str:
        return f"TaskExecutionContext(task_id={self.task_id}, cycle_count={self.cycle_count})"


class ExecutionContext:
    """
    Encapsulates execution state and provides execution methods.
    This class manages the execution queue, task results, and provides methods
    to execute tasks in a workflow graph. It also supports cycle detection
    and inter-task communication via channels.
    Different execution context can be created for different workflow runs.
    """

    def __init__(
        self,
        graph: TaskGraph,
        start_node: Optional[str] = None,
        max_steps: int = 10,
        default_max_cycles: int = 10,
        default_max_retries: int = 3,
        steps: int = 0,
        # Phase 1: Optional parameters for TaskQueue integration
        queue_backend: Union[QueueBackend, str] = QueueBackend.IN_MEMORY,
        queue_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize ExecutionContext with configurable queue backend."""
        session_id = str(uuid.uuid4().int)
        self.session_id = session_id
        self.graph = graph
        self.start_node = start_node
        self.max_steps = max_steps
        self.default_max_retries = default_max_retries
        self.steps = steps
        self.executed = []

        queue_config = queue_config or {}
        if start_node:
            queue_config['start_node'] = start_node

        # Phase 1: Abstract TaskQueue implementation
        if isinstance(queue_backend, str):
            queue_backend = QueueBackend(queue_backend)

        self.task_queue: TaskQueue = TaskQueueFactory.create(
            queue_backend, self, **queue_config
        )

        self.cycle_controller = CycleController(default_max_cycles)
        self.channel = MemoryChannel(session_id) # Use session_id for unique channel name
        self._function_manager = TaskFunctionManager()

        # Task execution context management
        self._task_execution_stack: list[TaskExecutionContext] = []
        self._task_contexts: dict[str, TaskExecutionContext] = {}

        # Group execution
        self.group_executor: Optional[GroupExecutor] = None

        # Track if goto (jump to existing task) was called in current task execution
        self._goto_called_in_current_task: bool = False

    @classmethod
    def create(cls, graph: TaskGraph, start_node: str, max_steps: int = 10,
               default_max_cycles: int = 10, default_max_retries: int = 3,
               queue_backend: Union[QueueBackend, str] = QueueBackend.IN_MEMORY,
               queue_config: Optional[Dict[str, Any]] = None) -> ExecutionContext:
        """Create a new execution context."""
        return cls(
            graph=graph,
            start_node=start_node,
            max_steps=max_steps,
            default_max_cycles=default_max_cycles,
            default_max_retries=default_max_retries,
            queue_backend=queue_backend,
            queue_config=queue_config,
        )

    @property
    def queue(self) -> TaskQueue:
        """Get the task queue instance."""
        return self.task_queue

    @property
    def function_manager(self) -> TaskFunctionManager:
        """Get the function manager instance."""
        return self._function_manager

    def add_to_queue(self, executable: Executable) -> None:
        """Add executable to execution queue."""
        task_spec = TaskSpec(
            executable=executable,
            execution_context=self
        )
        self.task_queue.enqueue(task_spec)

    def mark_executed(self, task_id: str) -> None:
        """Mark a node as executed."""
        if task_id not in self.executed:
            self.executed.append(task_id)

    def is_completed(self) -> bool:
        """Check if execution is completed (complete compatibility)."""
        return self.task_queue.is_empty() or self.steps >= self.max_steps

    def get_next_task(self) -> Optional[str]:
        """Get the next node to execute (complete compatibility)."""
        return self.task_queue.get_next_task()

    def increment_step(self) -> None:
        """Increment the step counter."""
        self.steps += 1

    def set_result(self, task_id: str, result: Any) -> None:
        """Store execution result for a node using channel."""
        channel_key = f"{task_id}.__result__"
        self.channel.set(channel_key, result)

    def get_result(self, task_id: str, default: Any = None) -> Any:
        """Get execution result for a node from channel."""
        channel_key = f"{task_id}.__result__"
        return self.channel.get(channel_key, default)

    def get_channel(self) -> Channel:
        """Get the channel for inter-task communication."""
        return self.channel

    def create_task_context(self, task_id: str) -> TaskExecutionContext:
        """Create and manage a task execution context."""
        task_ctx = TaskExecutionContext(task_id, self)
        self._task_contexts[task_id] = task_ctx
        return task_ctx

    def push_task_context(self, task_ctx: TaskExecutionContext) -> None:
        """Push task context onto execution stack."""
        self._task_execution_stack.append(task_ctx)

    def pop_task_context(self) -> Optional[TaskExecutionContext]:
        """Pop task context from execution stack."""
        return self._task_execution_stack.pop() if self._task_execution_stack else None

    @property
    def current_task_context(self) -> Optional[TaskExecutionContext]:
        """Get current task execution context."""
        return self._task_execution_stack[-1] if self._task_execution_stack else None

    @property
    def current_task_id(self) -> Optional[str]:
        """Get the currently executing task ID (backward compatibility)."""
        ctx = self.current_task_context
        return ctx.task_id if ctx else None

    @property
    def goto_called(self) -> bool:
        """Check if goto was called in current task execution."""
        return self._goto_called_in_current_task

    def reset_goto_flag(self) -> None:
        """Reset goto flag for next task execution."""
        self._goto_called_in_current_task = False

    def next_task(self, executable: Executable, goto: bool = False) -> str:
        """Generate a new task or jump to existing task node.

        Args:
            executable: Executable object to execute as the new task
            goto: If True, skip successors of current task (works for both existing and new tasks)

        Returns:
            The task ID from the executable
        """
        task_id = executable.task_id

        if goto:
            # Explicit goto: Skip successors regardless of whether task is new or existing
            if task_id in self.graph.nodes:
                # Existing task: Jump to it
                print(f"🔄 Goto: Jumping to existing task: {task_id}")
                self.add_to_queue(executable)
            else:
                # New task: Create it but still skip successors
                print(f"✨ Goto: Creating new task (skip successors): {task_id}")
                self.graph.add_node(executable, task_id)
                self.add_to_queue(executable)
            self._goto_called_in_current_task = True
        # Auto-detect behavior (no goto specified)
        elif task_id in self.graph.nodes:
            # Existing task: Jump to it (auto-detected, skip successors)
            print(f"🔄 Jumping to existing task: {task_id}")
            self.add_to_queue(executable)
            self._goto_called_in_current_task = True
        else:
            # New task: Create dynamic task (normal successor processing)
            print(f"✨ Creating new dynamic task: {task_id}")
            self.graph.add_node(executable, task_id)
            self.add_to_queue(executable)
            # Note: _goto_called_in_current_task remains False for normal processing

        return task_id

    def next_iteration(self, data: Any = None, task_id: Optional[str] = None) -> str:
        """Generate an iteration task for the current task (for cycles).

        Args:
            data: Data to pass to the next iteration
            task_id: Optional task ID (uses current task if None)

        Returns:
            The generated iteration task ID

        Raises:
            ValueError: If no current task is available or cycle limit exceeded
        """
        if task_id is None:
            current_ctx = self.current_task_context
            if not current_ctx:
                raise ValueError("No current task available for iteration")
            task_id = current_ctx.task_id

        if not task_id:
            raise ValueError("No current task available for iteration")

        if task_id not in self.graph.nodes:
            raise ValueError(f"Task {task_id} not found in graph")

        # Get or create task context
        task_ctx = self._task_contexts.get(task_id)
        if not task_ctx:
            task_ctx = self.create_task_context(task_id)

        # Use task context for cycle management
        if not task_ctx.can_iterate():
            raise CycleLimitExceededError(
                task_id=task_id,
                cycle_count=task_ctx.cycle_count,
                max_cycles=task_ctx.max_cycles
            )

        # Register this cycle execution
        cycle_count = task_ctx.register_cycle()

        # Get the current task function
        current_task = self.graph.get_node(task_id)

        # Generate iteration task ID with cycle count
        iteration_id = f"{task_id}_cycle_{cycle_count}_{uuid.uuid4().hex[:8]}"

        # Create iteration function with data
        def iteration_func():
            if data is not None:
                return current_task(task_ctx, data)
            else:
                return current_task(task_ctx)

        from .task import TaskWrapper
        iteration_task = TaskWrapper(iteration_id, iteration_func, inject_context=False)
        return self.next_task(iteration_task)

    @contextmanager
    def executing_task(self, task: Executable):
        """Context manager for task execution with proper cleanup.

        Args:
            task: The task being executed

        Yields:
            TaskExecutionContext: The task execution context
        """
        task_ctx = self.create_task_context(task.task_id)
        self.push_task_context(task_ctx)
        try:
            task.set_execution_context(self)
            yield task_ctx
        finally:
            self.pop_task_context()

    def execute(self) -> None:
        """Execute tasks using this context."""
        engine = WorkflowEngine()
        engine.execute(self)

    def save(self, path: str = "execution_context.pkl") -> None:
        """Save execution context to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = "execution_context.pkl") -> ExecutionContext:
        """Load execution context from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)

def create_execution_context(start_node: str = "ROOT", max_steps: int = 10) -> ExecutionContext:
    """Create an initial execution context with a single root node."""
    graph = TaskGraph()
    return ExecutionContext.create(graph, start_node, max_steps=max_steps)

def execute_with_cycles(graph: TaskGraph, start_node: str, max_steps: int = 10) -> None:
    """Execute tasks allowing cycles from global graph."""
    engine = WorkflowEngine()
    engine.execute_with_cycles(graph, start_node, max_steps)

