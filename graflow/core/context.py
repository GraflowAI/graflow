"""Execution engine for graflow with cycle support and global graph integration."""

from __future__ import annotations

import pickle
import time
import uuid
from collections import deque
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar

from graflow.channels.base import Channel
from graflow.channels.memory import MemoryChannel
from graflow.channels.typed import TypedChannel
from graflow.core.cycle import CycleController
from graflow.core.engine import WorkflowEngine
from graflow.core.graph import TaskGraph
from graflow.exceptions import CycleLimitExceededError

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

    def next_task(self, executable: Executable) -> str:
        """Create new dynamic task."""
        return self.execution_context.next_task(executable)

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
    ):
        """Initialize ExecutionContext."""
        session_id = str(uuid.uuid4().int)
        self.session_id = session_id
        self.graph = graph
        self.queue = deque([start_node])
        self.start_node = start_node
        self.max_steps = max_steps
        self.default_max_retries = default_max_retries
        self.steps = steps
        self.executed = []

        self.cycle_controller = CycleController(default_max_cycles)
        self.channel = MemoryChannel(session_id) # Use session_id for unique channel name

        # Task execution context management
        self._task_execution_stack: list[TaskExecutionContext] = []
        self._task_contexts: dict[str, TaskExecutionContext] = {}

        if self.start_node and not self.queue:
            self.queue.append(self.start_node)

    @classmethod
    def create(cls, graph: TaskGraph, start_node: str, max_steps: int = 10,
               default_max_cycles: int = 10, default_max_retries: int = 3) -> ExecutionContext:
        """Create a new execution context."""
        return cls(
            graph=graph,
            start_node=start_node,
            max_steps=max_steps,
            default_max_cycles=default_max_cycles,
            default_max_retries=default_max_retries,
        )

    def add_to_queue(self, node: str) -> None:
        """Add a node to the execution queue."""
        self.queue.append(node)

    def mark_executed(self, node: str) -> None:
        """Mark a node as executed."""
        if node not in self.executed:
            self.executed.append(node)

    def is_completed(self) -> bool:
        """Check if execution is completed."""
        return not self.queue or self.steps >= self.max_steps

    def get_next_node(self) -> Optional[str]:
        """Get the next node to execute."""
        return self.queue.popleft() if self.queue else None

    def increment_step(self) -> None:
        """Increment the step counter."""
        self.steps += 1

    def set_result(self, node: str, result: Any) -> None:
        """Store execution result for a node using channel."""
        channel_key = f"{node}.__result__"
        self.channel.set(channel_key, result)

    def get_result(self, node: str, default: Any = None) -> Any:
        """Get execution result for a node from channel."""
        channel_key = f"{node}.__result__"
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

    def next_task(self, executable: Executable) -> str:
        """Generate a new task dynamically during execution.

        Args:
            executable: Executable object to execute as the new task

        Returns:
            The task ID from the executable
        """
        task_id = executable.task_id

        # Add to execution queue
        self.add_to_queue(task_id)

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
        current_task = self.graph.nx_graph().nodes[task_id]['task']

        # Generate iteration task ID with cycle count
        iteration_id = f"{task_id}_cycle_{cycle_count}_{uuid.uuid4().hex[:8]}"

        # Create iteration function with data
        def iteration_func():
            if data is not None:
                return current_task.func(task_ctx, data)
            else:
                return current_task.func(task_ctx)

        from .task import TaskWrapper  # noqa: PLC0415
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

