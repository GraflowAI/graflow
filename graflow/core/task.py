"""Core classes for graflow - Executable, Task, ParallelGroup, TaskWrapper."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Executable(ABC):
    """Abstract base class for all executable entities in graflow."""

    @abstractmethod
    def run(self):
        """Execute this executable."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this executable."""
        pass

    def __rshift__(self, other: Executable) -> Executable:
        """Create dependency self >> other using >> operator."""
        self._add_dependency_edge(self.name, other.name)
        return other

    def __lshift__(self, other: Executable) -> Executable:
        """Create dependency other >> self using << operator."""
        self._add_dependency_edge(other.name, self.name)
        return self

    def __or__(self, other: Executable) -> ParallelGroup:
        """Create a parallel group using | operator."""
        if isinstance(other, Task | TaskWrapper):
            return ParallelGroup([self, other])
        elif isinstance(other, ParallelGroup):
            # Add this task to the existing ParallelGroup instead of creating a new one
            other.tasks.insert(0, self)  # Insert at the beginning to maintain order
            other._add_dependency_edge(other.name, self.name)
            return other
        else:
            raise TypeError(f"Can only combine with Task, TaskWrapper, or ParallelGroup: {type(other)}")

    def _add_dependency_edge(self, from_task: str, to_task: str) -> None:
        """Add dependency edge in current context."""
        from .workflow import get_current_workflow_context  # noqa: PLC0415 avoid circular import
        current_context = get_current_workflow_context()
        current_context.add_edge(from_task, to_task)

    def _register_to_context(self) -> None:
        """Register this root task to current workflow context."""
        from .workflow import get_current_workflow_context  # noqa: PLC0415 avoid circular import
        current_context = get_current_workflow_context()
        current_context.add_node(self.name, self)

class Task(Executable):
    """A pseudo root task that serves as the starting point for workflow execution."""

    def __init__(self, name: str) -> None:
        """Initialize a task."""
        self._name = name
        # Register to current workflow context
        self._register_to_context()

    @property
    def name(self) -> str:
        """Return the name of this root task."""
        return self._name

    def run(self) -> None:
        """Execute this task (typically a no-op)."""
        print(f"Starting workflow from root: {self._name}")
        pass

    def __repr__(self) -> str:
        """Return string representation of this root task."""
        return f"Task({self._name})"


class ParallelGroup(Executable):
    """A parallel group of executable entities."""

    _group_counter = 0

    def __init__(self, tasks: list[Executable]) -> None:
        """Initialize a parallel group with a list of tasks."""
        # Get name from current context or use global counter
        self._name = self._get_group_name()
        self.tasks = list(tasks)

        # Register this parallel group to current workflow context
        self._register_to_context()
        for task in self.tasks:
            self._add_dependency_edge(self._name, task.name)

    @property
    def name(self) -> str:
        """Return the name of this parallel group."""
        return self._name

    def run(self) -> None:
        """Execute all tasks in this parallel group."""
        print(f"Running parallel group: {self._name}")
        print(f"  Parallel tasks: {[task.name for task in self.tasks]}")

        # Execute all tasks (sequentially for now, but conceptually parallel)
        for task in self.tasks:
            print(f"  - Executing in parallel: {task.name}")
            task.run()

        print(f"  Parallel group {self._name} completed")

    def __rshift__(self, other: Executable) -> Executable:
        """Create dependency from all tasks in parallel group to other."""
        for task in self.tasks:
            self._add_dependency_edge(task.name, other.name)
        return other

    def __lshift__(self, other: Executable) -> Executable:
        """Create dependency from other to all tasks in parallel group."""
        for task in self.tasks:
            self._add_dependency_edge(other.name, task.name)
        return other

    def __or__(self, other: Executable) -> ParallelGroup:
        """Extend parallel group with | operator."""
        if isinstance(other, Task | TaskWrapper):
            # Add the new task to current group instead of creating a new one
            self.tasks.append(other)
            # Add edge from this group to the new task
            self._add_dependency_edge(self.name, other.name)
            return self
        elif isinstance(other, ParallelGroup):
            # Merge the other group into this one
            self.tasks.extend(other.tasks)
            # Add edges from this group to all tasks in the other group
            for task in other.tasks:
                self._add_dependency_edge(self.name, task.name)
            # Remove the other group from the workflow graph
            other._remove_from_context()
            return self
        else:
            raise TypeError(f"Can only combine with Task, TaskWrapper, or ParallelGroup: {type(other)}")

    def __repr__(self) -> str:
        """Return string representation of this parallel group."""
        task_names = [task.name for task in self.tasks]
        return f"ParallelGroup({task_names})"

    def _get_group_name(self) -> str:
        """Get group name from current context or global counter."""
        from .workflow import get_current_workflow_context  # noqa: PLC0415 avoid circular import
        current_context = get_current_workflow_context()
        return current_context.get_next_group_name()

    def _remove_from_context(self) -> None:
        """Remove this parallel group from the current workflow context."""
        from .workflow import get_current_workflow_context  # noqa: PLC0415 avoid circular import
        current_context = get_current_workflow_context()
        graph = current_context.graph
        if graph.has_node(self.name):
            # Remove all edges connected to this node
            edges_to_remove = list(graph.edges(self.name))
            for edge in edges_to_remove:
                graph.remove_edge(*edge)
            # Remove the node itself
            graph.remove_node(self.name)


class TaskWrapper(Executable):
    """Wrapper class for function-based tasks created with @task decorator."""

    def __init__(self, name: str, func) -> None:
        """Initialize a task wrapper with name and function."""
        self._name = name
        self.func = func
        # Register to current workflow context or global graph
        self._register_to_context()

    @property
    def name(self) -> str:
        """Return the name of this task wrapper."""
        return self._name

    def run(self) -> None:
        """Execute the wrapped function."""
        print(f"Running task: {self._name}")
        return self.func()

    def __call__(self, *args, **kwargs):
        """Allow direct function call."""
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        """Return string representation of this task wrapper."""
        return f"TaskWrapper({self._name})"
