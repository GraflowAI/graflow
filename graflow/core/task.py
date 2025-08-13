"""Core classes for graflow - Executable, Task, ParallelGroup, TaskWrapper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from graflow.coordination.executor import GroupExecutor
from graflow.coordination.task_spec import TaskSpec
from graflow.core.context import ExecutionContext
from graflow.exceptions import GraflowRuntimeError


class Executable(ABC):
    """Abstract base class for all executable entities in graflow."""

    @property
    @abstractmethod
    def task_id(self) -> str:
        """Return the task_id of this executable."""
        pass

    def set_execution_context(self, context: ExecutionContext) -> None:
        """Set the execution context for this executable."""
        self._execution_context = context

    def get_execution_context(self) -> ExecutionContext:
        """Get the execution context for this executable."""

        if not hasattr(self, '_execution_context'):
            raise GraflowRuntimeError("Execution context not set. Call set_execution_context() first.")

        return self._execution_context

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute this executable."""
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Allow direct function call on the executable."""
        pass

    def __rshift__(self, other: Executable) -> Executable:
        """Create dependency self >> other using >> operator."""
        self._add_dependency_edge(self.task_id, other.task_id)
        return other

    def __lshift__(self, other: Executable) -> Executable:
        """Create dependency other >> self using << operator."""
        self._add_dependency_edge(other.task_id, self.task_id)
        return self

    def __or__(self, other: Executable) -> ParallelGroup:
        """Create a parallel group using | operator."""
        if isinstance(other, Task | TaskWrapper):
            return ParallelGroup([self, other])
        elif isinstance(other, ParallelGroup):
            # Add this task to the existing ParallelGroup instead of creating a new one
            other.tasks.insert(0, self)  # Insert at the beginning to maintain order
            other._add_dependency_edge(other.task_id, self.task_id)
            return other
        else:
            raise TypeError(f"Can only combine with Task, TaskWrapper, or ParallelGroup: {type(other)}")

    def _add_dependency_edge(self, from_task: str, to_task: str) -> None:
        """Add dependency edge in current context."""
        from .workflow import current_workflow_context
        current_context = current_workflow_context()
        current_context.add_edge(from_task, to_task)

    def _register_to_context(self) -> None:
        """Register this root task to current workflow context."""
        from .workflow import current_workflow_context
        current_context = current_workflow_context()
        current_context.add_node(self.task_id, self)

class Task(Executable):
    """A pseudo root task that serves as the starting point for workflow execution."""

    def __init__(self, task_id: str) -> None:
        """Initialize a task."""
        self._task_id = task_id
        # Register to current workflow context
        self._register_to_context()

    @property
    def task_id(self) -> str:
        """Return the task_id of this root task."""
        return self._task_id

    def __call__(self, *args, **kwargs) -> Any:
        """Allow direct function call on the root task."""
        return self.run()

    def run(self) -> Any:
        """Execute this task (typically a no-op)."""
        print(f"Starting workflow from root: {self._task_id}")
        pass

    def __repr__(self) -> str:
        """Return string representation of this root task."""
        return f"Task({self._task_id})"


class ParallelGroup(Executable):
    """A parallel group of executable entities."""

    _group_counter = 0

    def __init__(self, tasks: list[Executable]) -> None:
        """Initialize a parallel group with a list of tasks."""
        # Get name from current context or use global counter
        self._task_id = self._get_group_name()
        self.tasks = list(tasks)

        # Register this parallel group to current workflow context
        self._register_to_context()
        for task in self.tasks:
            self._add_dependency_edge(self._task_id, task.task_id)

    @property
    def task_id(self) -> str:
        """Return the task_id of this parallel group."""
        return self._task_id

    def __call__(self, *args, **kwargs) -> Any:
        """Allow direct function call on the parallel group."""
        return self.run()

    def run(self) -> Any:
        """Execute all tasks in this parallel group."""
        context = self.get_execution_context()
        executor = context.group_executor or GroupExecutor()

        task_specs = []

        for task in self.tasks:
            # Set execution context for each task
            task.set_execution_context(context)

            # Create appropriate execution function based on task type
            if isinstance(task, TaskWrapper) and task.inject_context:
                def create_context_func(t=task):
                    def context_func():
                        with context.executing_task(t) as task_ctx:
                            return t.func(task_ctx)
                    return context_func
                task_func = create_context_func()
            else:
                task_func = task.run

            task_specs.append(TaskSpec(task.task_id, context, task_func))

        executor.execute_parallel_group(self.task_id, task_specs)

    def __rshift__(self, other: Executable) -> Executable:
        """Create dependency from all tasks in parallel group to other."""
        for task in self.tasks:
            self._add_dependency_edge(task.task_id, other.task_id)
        return other

    def __lshift__(self, other: Executable) -> Executable:
        """Create dependency from other to all tasks in parallel group."""
        for task in self.tasks:
            self._add_dependency_edge(other.task_id, task.task_id)
        return other

    def __or__(self, other: Executable) -> ParallelGroup:
        """Extend parallel group with | operator."""
        if isinstance(other, Task | TaskWrapper):
            # Add the new task to current group instead of creating a new one
            self.tasks.append(other)
            # Add edge from this group to the new task
            self._add_dependency_edge(self.task_id, other.task_id)
            return self
        elif isinstance(other, ParallelGroup):
            # Merge the other group into this one
            self.tasks.extend(other.tasks)
            # Add edges from this group to all tasks in the other group
            for task in other.tasks:
                self._add_dependency_edge(self.task_id, task.task_id)
            # Remove the other group from the workflow graph
            other._remove_from_context()
            return self
        else:
            raise TypeError(f"Can only combine with Task, TaskWrapper, or ParallelGroup: {type(other)}")

    def __repr__(self) -> str:
        """Return string representation of this parallel group."""
        task_ids = [task.task_id for task in self.tasks]
        return f"ParallelGroup({task_ids})"

    def _get_group_name(self) -> str:
        """Get group name from current context or global counter."""
        from .workflow import current_workflow_context
        current_context = current_workflow_context()
        return current_context.get_next_group_name()

    def _remove_from_context(self) -> None:
        """Remove this parallel group from the current workflow context."""
        from .workflow import current_workflow_context
        current_context = current_workflow_context()
        graph = current_context.graph._graph
        if graph.has_node(self.task_id):
            # Remove all edges connected to this node
            edges_to_remove = list(graph.edges(self.task_id))
            for edge in edges_to_remove:
                graph.remove_edge(*edge)
            # Remove the node itself
            graph.remove_node(self.task_id)


class TaskWrapper(Executable):
    """Wrapper class for function-based tasks created with @task decorator."""

    def __init__(self, task_id: str, func, inject_context: bool = False) -> None:
        """Initialize a task wrapper with task_id and function."""
        self._task_id = task_id
        self.func = func
        self.inject_context = inject_context
        # Register to current workflow context or global graph
        self._register_to_context()

    @property
    def task_id(self) -> str:
        """Return the task_id of this task wrapper."""
        return self._task_id

    def __call__(self, *args, **kwargs) -> Any:
        """Allow direct function call."""
        if self.inject_context:
            exec_context = self.get_execution_context()
            task_context = exec_context.current_task_context
            if task_context:
                return self.func(task_context, *args, **kwargs)
            else:
                # Fallback: create temporary task context
                with exec_context.executing_task(self) as task_ctx:
                    return self.func(task_ctx, *args, **kwargs)
        return self.func(*args, **kwargs)

    def run(self) -> Any:
        """Execute the wrapped function."""
        if self.inject_context:
            exec_context = self.get_execution_context()
            task_context = exec_context.current_task_context
            if task_context:
                return self.func(task_context)
            else:
                # Fallback: create temporary task context
                with exec_context.executing_task(self) as task_ctx:
                    return self.func(task_ctx)
        return self.func()

    def __repr__(self) -> str:
        """Return string representation of this task wrapper."""
        return f"TaskWrapper({self._task_id})"
