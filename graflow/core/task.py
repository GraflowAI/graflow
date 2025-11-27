"""Core classes for graflow - Executable, Task, ParallelGroup, TaskWrapper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

from graflow.coordination.coordinator import CoordinationBackend
from graflow.coordination.executor import GroupExecutor
from graflow.core.context import ExecutionContext
from graflow.exceptions import GraflowRuntimeError

if TYPE_CHECKING:
    from graflow.core.handlers.group_policy import GroupExecutionPolicy


class Executable(ABC):
    """Abstract base class for all executable entities in graflow."""

    def __init__(self) -> None:
        """Initialize executable with default handler type."""
        self.handler_type: str = "direct"

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

    def __getstate__(self):
        """Custom serialization to exclude execution context."""
        state = self.__dict__.copy()
        # Remove execution context as it contains runtime state (connections, etc.)
        # and should be re-injected on the worker side.
        if '_execution_context' in state:
            del state['_execution_context']
        return state

    def __setstate__(self, state):
        """Restore state."""
        self.__dict__.update(state)

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute this executable."""
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Allow direct function call on the executable."""
        pass

    def __rshift__(self, other: Executable) -> SequentialTask:
        """Create dependency self >> other using >> operator.

        Returns SequentialTask to track the chain.

        Args:
            other: Task or chain to depend on

        Returns:
            SequentialTask with combined tasks
        """
        if isinstance(other, SequentialTask):
            # self >> (a >> b >> c) → SequentialTask([self, a, b, c])
            # Edge: self -> a (leftmost)
            self._add_dependency_edge(self.task_id, other.leftmost.task_id)
            return SequentialTask([self, *other.tasks])
        else:
            # self >> other → SequentialTask([self, other])
            # Edge: self -> other
            self._add_dependency_edge(self.task_id, other.task_id)
            return SequentialTask([self, other])

    def __lshift__(self, other: Executable) -> SequentialTask:
        """Create dependency other >> self using << operator.

        Returns SequentialTask to track the chain.

        Args:
            other: Task or chain to prepend

        Returns:
            SequentialTask with combined tasks
        """
        if isinstance(other, SequentialTask):
            # self << (a >> b >> c) → SequentialTask([a, b, c, self])
            # Edge: c (rightmost) -> self
            other.rightmost._add_dependency_edge(other.rightmost.task_id, self.task_id)
            return SequentialTask([*other.tasks, self])
        else:
            # self << other → SequentialTask([other, self])
            # Edge: other -> self
            other._add_dependency_edge(other.task_id, self.task_id)
            return SequentialTask([other, self])

    def __or__(self, other: Executable) -> ParallelGroup:
        """Create a parallel group using | operator.

        If combining with SequentialTask, uses its leftmost task.

        Args:
            other: Task, chain, or group to combine in parallel

        Returns:
            ParallelGroup with appropriate tasks
        """
        if isinstance(other, SequentialTask):
            # a | (b >> c) → ParallelGroup([a, b])
            return ParallelGroup([self, other.leftmost])
        elif isinstance(other, Task | TaskWrapper):
            return ParallelGroup([self, other])
        elif isinstance(other, ParallelGroup):
            # Add this task to the existing ParallelGroup instead of creating a new one
            other.tasks.insert(0, self)  # Insert at the beginning to maintain order
            other._add_dependency_edge(other.task_id, self.task_id)
            return other
        else:
            raise TypeError(
                f"Can only combine with Task, TaskWrapper, SequentialTask, or ParallelGroup: {type(other)}"
            )

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


class SequentialTask(Executable):
    """Sequential task chain wrapper (not registered in graph).

    This wrapper represents a chain of tasks connected by >> operators.
    It maintains a list of all tasks in the chain in execution order.
    The individual tasks are still registered in the graph, but SequentialTask
    itself is not registered as a graph node.

    SequentialTask inherits from Executable to maintain interface compatibility,
    but does not register itself to the workflow context.

    Example:
        a >> b >> c creates SequentialTask(tasks=[a, b, c])
        with edges: a->b, b->c in the graph

    Attributes:
        tasks: List of all tasks in execution order (leftmost to rightmost)
    """

    def __init__(self, tasks: list[Executable]):
        """Initialize a sequential task chain.

        Args:
            tasks: List of tasks in execution order (leftmost to rightmost)

        Raises:
            ValueError: If tasks list is empty

        Note:
            Does not call _register_to_context(), so this is not added to the graph.
        """
        if not tasks:
            raise ValueError("SequentialTask requires at least one task")
        super().__init__()
        self.tasks = list(tasks)

    @property
    def leftmost(self) -> Executable:
        """Return the first task in the chain."""
        return self.tasks[0]

    @property
    def rightmost(self) -> Executable:
        """Return the last task in the chain."""
        return self.tasks[-1]

    @property
    def task_id(self) -> str:
        """Return the leftmost task_id (entry point of the chain)."""
        return self.leftmost.task_id

    def run(self, *args, **kwargs) -> Any:
        """Execute the chain starting from the leftmost task.

        Note: This executes only the leftmost task directly.
        The full chain execution should be managed by WorkflowEngine
        following the graph dependencies.
        """
        return self.leftmost.run(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        """Allow direct function call on the chain."""
        return self.run(*args, **kwargs)

    def __rshift__(self, other: Executable) -> SequentialTask:
        """Extend the chain: (a >> b) >> c.

        Args:
            other: Task or chain to append

        Returns:
            Extended SequentialTask with combined tasks
        """
        if isinstance(other, SequentialTask):
            # Chain to another chain: (a >> b) >> (c >> d)
            # → SequentialTask([a, b, c, d])
            # Edge: b -> c (rightmost -> other.leftmost)
            self.rightmost._add_dependency_edge(
                self.rightmost.task_id,
                other.leftmost.task_id
            )
            return SequentialTask(self.tasks + other.tasks)
        else:
            # Chain to a single task: (a >> b) >> c
            # → SequentialTask([a, b, c])
            # Edge: b -> c (rightmost -> other)
            self.rightmost._add_dependency_edge(
                self.rightmost.task_id,
                other.task_id
            )
            return SequentialTask([*self.tasks, other])

    def __lshift__(self, other: Executable) -> SequentialTask:
        """Prepend to the chain: c << (a >> b).

        Args:
            other: Task or chain to prepend

        Returns:
            Extended SequentialTask with combined tasks
        """
        if isinstance(other, SequentialTask):
            # Chain from another chain: (c >> d) << (a >> b)
            # → SequentialTask([a, b, c, d])
            # Edge: b -> c (other.rightmost -> leftmost)
            other.rightmost._add_dependency_edge(
                other.rightmost.task_id,
                self.leftmost.task_id
            )
            return SequentialTask(other.tasks + self.tasks)
        else:
            # Chain from a single task: c << (a >> b)
            # → SequentialTask([other, a, b])
            # Edge: other -> a
            other._add_dependency_edge(
                other.task_id,
                self.leftmost.task_id
            )
            return SequentialTask([other, *self.tasks])

    def __or__(self, other: Executable) -> ParallelGroup:
        """Create parallel group: (a >> b) | c.

        Uses the leftmost task (a) for the parallel group, not the rightmost.
        This ensures that the entry point of the chain is used for parallelization.

        Args:
            other: Task, chain, or group to combine in parallel

        Returns:
            ParallelGroup with leftmost tasks
        """
        if isinstance(other, SequentialTask):
            # (a >> b) | (c >> d) → ParallelGroup([a, c])
            return ParallelGroup([self.leftmost, other.leftmost])
        elif isinstance(other, Task | TaskWrapper):
            # (a >> b) | c → ParallelGroup([a, c])
            return ParallelGroup([self.leftmost, other])
        elif isinstance(other, ParallelGroup):
            # (a >> b) | existing_group → add a to group
            other.tasks.insert(0, self.leftmost)
            return other
        else:
            raise TypeError(
                f"Can only combine with Task, TaskWrapper, SequentialTask, or ParallelGroup: {type(other)}"
            )

    def __len__(self) -> int:
        """Return the number of tasks in the chain."""
        return len(self.tasks)

    def __getitem__(self, index: int) -> Executable:
        """Get task at index."""
        return self.tasks[index]

    def __iter__(self):
        """Iterate over tasks."""
        return iter(self.tasks)

    def __repr__(self) -> str:
        """String representation."""
        task_ids = " >> ".join(t.task_id for t in self.tasks)
        return f"SequentialTask([{task_ids}])"


class Task(Executable):
    """A pseudo root task that serves as the starting point for workflow execution."""

    def __init__(self, task_id: str, register_to_context: bool = True) -> None:
        """Initialize a task."""
        super().__init__()
        self._task_id = task_id

        # Add serialization attributes required for checkpoint/resume
        # These attributes allow Task objects to be serialized using the reference strategy
        self.__name__ = task_id
        self.__module__ = self.__class__.__module__
        self.__qualname__ = f"{self.__class__.__name__}.{task_id}"

        # Register to current workflow context
        if register_to_context:
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
        super().__init__()
        # Get name from current context or use global counter
        self._task_id = self._get_group_name()
        self.tasks = list(tasks)

        # Execution configuration for with_execution()
        self._execution_config = {
            "backend": None,  # None = use default backend
            "backend_config": {},
            "policy": "strict",
        }

        # Register this parallel group to current workflow context
        self._register_to_context()

    @property
    def task_id(self) -> str:
        """Return the task_id of this parallel group."""
        return self._task_id

    def set_group_name(self, name: str) -> ParallelGroup:
        """Set the group name (task_id) for this parallel group."""
        old_task_id = self._task_id

        # Rename the node in the workflow context
        from .workflow import current_workflow_context
        current_context = current_workflow_context()
        current_context.rename_node(old_task_id, name)

        # Update local task_id
        self._task_id = name
        return self

    def with_execution(
        self,
        backend: Optional[CoordinationBackend] = None,
        backend_config: Optional[dict] = None,
        policy: Union[str, GroupExecutionPolicy] = "strict",
    ) -> ParallelGroup:
        """Configure execution backend and group policy for this parallel group.

        Args:
            backend: Coordinator backend (DIRECT, THREADING, REDIS)
            backend_config: Backend-specific configuration
                - THREADING: {"thread_count": int}
                - REDIS: Future extension
            policy: Group execution policy name or instance

        Returns:
            Self (for method chaining)

        Examples:
            # Default: All tasks must succeed (strict mode)
            (task_a | task_b | task_c).with_execution()

            # Best-effort: Continue even if tasks fail
            (task_a | task_b | task_c).with_execution(policy="best_effort")

            # At least 3 out of 4 tasks must succeed
            (task_a | task_b | task_c | task_d).with_execution(
                policy=AtLeastNGroupPolicy(min_success=3)
            )

            # Critical tasks must succeed
            (task_a | task_b).with_execution(
                policy=CriticalGroupPolicy(critical_task_ids=["task_a"])
            )

        Note:
            Individual task execution handlers should be set at task level using @task(handler="...").
            The ``policy`` parameter controls parallel group success/failure criteria.
        """
        if backend is not None:
            self._execution_config["backend"] = backend

        if backend_config is not None:
            self._execution_config["backend_config"].update(backend_config)

        from graflow.core.handlers.group_policy import canonicalize_group_policy

        self._execution_config["policy"] = canonicalize_group_policy(policy)

        return self

    def __call__(self, *args, **kwargs) -> Any:
        """Allow direct function call on the parallel group."""
        return self.run()

    def run(self) -> Any:
        """Execute all tasks in this parallel group."""
        context = self.get_execution_context()

        for task in self.tasks:
            # Set execution context for each task
            task.set_execution_context(context)

        # Extract policy configuration
        policy = self._execution_config.get("policy", "strict")
        backend = self._execution_config.get("backend")
        backend_config = self._execution_config.get("backend_config", {})

        # GroupExecutor is stateless - call static method directly
        GroupExecutor.execute_parallel_group(
            self.task_id,
            self.tasks,
            context,
            backend=backend,
            backend_config=backend_config,
            policy=policy,
        )

    def __rshift__(self, other: Executable) -> SequentialTask:
        """Create dependency from parallel group to other.

        Note: Creates edge from the GROUP itself, not from individual tasks.
        This ensures successors are handled at the group level after all
        parallel tasks complete, preventing premature or duplicate execution.

        Returns SequentialTask to track chains.

        Args:
            other: Task or chain to depend on

        Returns:
            SequentialTask starting from this group
        """
        if isinstance(other, SequentialTask):
            # group >> (a >> b >> c) → SequentialTask([group, a, b, c])
            # Edge: group -> a (leftmost)
            self._add_dependency_edge(self.task_id, other.leftmost.task_id)
            return SequentialTask([self, *other.tasks])
        else:
            # group >> other → SequentialTask([group, other])
            # Edge: group -> other
            self._add_dependency_edge(self.task_id, other.task_id)
            return SequentialTask([self, other])

    def __lshift__(self, other: Executable) -> SequentialTask:
        """Create dependency from other to parallel group.

        Note: Creates edge to the GROUP itself, not to individual tasks.
        This ensures the predecessor relationship is at the group level.

        Returns SequentialTask to track chains.

        Args:
            other: Task or chain to prepend

        Returns:
            SequentialTask ending with this group
        """
        if isinstance(other, SequentialTask):
            # group << (a >> b >> c) → SequentialTask([a, b, c, group])
            # Edge: c (rightmost) -> group
            other.rightmost._add_dependency_edge(other.rightmost.task_id, self.task_id)
            return SequentialTask([*other.tasks, self])
        else:
            # group << other → SequentialTask([other, group])
            # Edge: other -> group
            other._add_dependency_edge(other.task_id, self.task_id)
            return SequentialTask([other, self])

    def __or__(self, other: Executable) -> ParallelGroup:
        """Extend parallel group with | operator.

        If combining with SequentialTask, uses its leftmost task.

        Strategy for ParallelGroup | ParallelGroup:
        - If either group has external dependencies (successors), create a new ParallelGroup
        - Otherwise, merge into this group for simplicity

        Args:
            other: Task, chain, or group to add to this parallel group

        Returns:
            Extended ParallelGroup or new ParallelGroup containing both groups
        """
        if isinstance(other, SequentialTask):
            # group | (a >> b) → add a (leftmost) to group
            self.tasks.append(other.leftmost)
            return self
        elif isinstance(other, Task | TaskWrapper):
            # Add the new task to current group instead of creating a new one
            self.tasks.append(other)
            return self
        elif isinstance(other, ParallelGroup):
            # Check if either group has external dependencies (successors)
            self_has_deps = self._has_successors()
            other_has_deps = other._has_successors()

            if self_has_deps or other_has_deps:
                # Cannot merge - create new ParallelGroup containing both groups
                return ParallelGroup([self, other])
            else:
                # No external dependencies - safe to merge
                self.tasks.extend(other.tasks)
                # Remove the other group from the workflow graph
                other._remove_from_context()
                return self
        else:
            raise TypeError(
                f"Can only combine with Task, TaskWrapper, SequentialTask, or ParallelGroup: {type(other)}"
            )

    def __repr__(self) -> str:
        """Return string representation of this parallel group."""
        task_ids = [task.task_id for task in self.tasks]
        return f"ParallelGroup({task_ids})"

    def _has_successors(self) -> bool:
        """Check if this parallel group has any successor tasks in the graph.

        Returns:
            True if this group has successors (dependencies), False otherwise
        """
        from .workflow import current_workflow_context
        try:
            current_context = current_workflow_context()
            graph = current_context.graph
            successors = graph.successors(self.task_id)
            member_ids = set(graph.get_parallel_group_members(self.task_id))
            external_successors = [s for s in successors if s not in member_ids]
            return len(external_successors) > 0
        except Exception:
            # If we can't access the graph, assume no successors
            return False

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

    def __init__(
        self,
        task_id: str,
        func,
        inject_context: bool = False,
        inject_llm_client: bool = False,
        inject_llm_agent: Optional[str] = None,
        register_to_context: bool = True,
        handler_type: Optional[str] = None
    ) -> None:
        """Initialize a task wrapper with task_id and function.

        Args:
            task_id: Task identifier
            func: Function to wrap
            inject_context: Whether to inject TaskExecutionContext as first argument
            inject_llm_client: Whether to inject shared LLMClient instance as first argument
            inject_llm_agent: Agent name string to inject LLMAgent as first argument
                            (agent must be registered in ExecutionContext)
            register_to_context: Whether to register to workflow context
            handler_type: Execution handler type ("direct", "docker", or custom)
        """
        super().__init__()
        self._task_id = task_id
        self.func = func
        self.inject_context = inject_context
        self.inject_llm_client = inject_llm_client
        self.inject_llm_agent = inject_llm_agent

        # Add serialization attributes required for checkpoint/resume
        # These may be overwritten by @task decorator if used
        self.__name__ = getattr(func, '__name__', task_id)
        self.__module__ = getattr(func, '__module__', self.__class__.__module__)
        self.__qualname__ = getattr(func, '__qualname__', task_id)
        self.__doc__ = getattr(func, '__doc__', None)

        # Set handler_type attribute (from Executable base class)
        if handler_type is not None:
            self.handler_type = handler_type
        # Register to current workflow context or global graph
        if register_to_context:
            self._register_to_context()

    @property
    def task_id(self) -> str:
        """Return the task_id of this task wrapper."""
        return self._task_id

    def __call__(self, *args, **kwargs) -> Any:
        """Allow direct function call with automatic dependency injection.

        Multiple injection types can co-exist:
        - inject_context: Injects TaskExecutionContext as positional argument
        - inject_llm_client: Injects LLMClient as named argument 'llm_client'
        - inject_llm_agent: Injects LLMAgent as named argument 'llm_agent'
        """
        # Prepare injection kwargs
        injection_kwargs = {}

        exec_context: ExecutionContext = self.get_execution_context()

        # LLMClient injection (named argument)
        if self.inject_llm_client:
            llm_client = exec_context.llm_client
            injection_kwargs['llm_client'] = llm_client

        # LLMAgent injection (named argument)
        if self.inject_llm_agent:
            try:
                agent = exec_context.get_llm_agent(self.inject_llm_agent)
            except KeyError:
                raise RuntimeError(
                    f"Task '{self._task_id}' requires LLMAgent '{self.inject_llm_agent}', but it's not registered. "
                    f"Register it with: context.register_llm_agent('{self.inject_llm_agent}', agent)"
                )
            injection_kwargs['llm_agent'] = agent

        # TaskExecutionContext injection (positional argument)
        if self.inject_context:
            task_context = exec_context.current_task_context
            if not task_context:
                # Fallback: create temporary task context
                with exec_context.executing_task(self) as task_ctx:
                    return self.func(task_ctx, *args, **{**kwargs, **injection_kwargs})
            return self.func(task_context, *args, **{**kwargs, **injection_kwargs})

        # No injection or only llm_client/llm_agent - pass injection kwargs
        return self.func(*args, **{**kwargs, **injection_kwargs})

    def run(self) -> Any:
        """Execute the wrapped function with dependency injection.

        Multiple injection types can co-exist:
        - inject_context: Injects TaskExecutionContext as positional argument
        - inject_llm_client: Injects LLMClient as named argument 'llm_client'
        - inject_llm_agent: Injects LLMAgent as named argument 'llm_agent'
        """
        # Prepare injection kwargs
        injection_kwargs = {}
        exec_context: ExecutionContext = self.get_execution_context()

        # LLMClient injection (named argument)
        if self.inject_llm_client:
            llm_client = exec_context.llm_client
            injection_kwargs['llm_client'] = llm_client

        # LLMAgent injection (named argument)
        if self.inject_llm_agent:
            try:
                agent = exec_context.get_llm_agent(self.inject_llm_agent)
            except KeyError:
                raise RuntimeError(
                    f"Task '{self._task_id}' requires LLMAgent '{self.inject_llm_agent}', but it's not registered. "
                    f"Register it with: context.register_llm_agent('{self.inject_llm_agent}', agent)"
                )
            injection_kwargs['llm_agent'] = agent

        # TaskExecutionContext injection (positional argument)
        if self.inject_context:
            task_context = exec_context.current_task_context
            if not task_context:
                # Fallback: create temporary task context
                with exec_context.executing_task(self) as task_ctx:
                    return self.func(task_ctx, **injection_kwargs)
            return self.func(task_context, **injection_kwargs)

        # No context injection - just pass injection kwargs
        return self.func(**injection_kwargs)

    def __repr__(self) -> str:
        """Return string representation of this task wrapper."""
        return f"TaskWrapper({self._task_id})"
