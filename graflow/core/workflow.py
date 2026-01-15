"""Workflow context manager for graflow."""

from __future__ import annotations

import contextvars
import uuid
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from graflow.core.graph import TaskGraph
from graflow.exceptions import GraflowRuntimeError, GraphCompilationError
from graflow.trace.base import Tracer

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.task import Executable
    from graflow.llm.agents.base import LLMAgent
    from graflow.prompts.base import PromptManager

LLMAgentFactory = Callable[["ExecutionContext"], "LLMAgent"]
LLMAgentProvider = Union["LLMAgent", LLMAgentFactory]

# Context variable for current workflow context
_current_context: contextvars.ContextVar[Optional[WorkflowContext]] = contextvars.ContextVar(
    "current_workflow", default=None
)


class WorkflowContext:
    """
    Context for Workflow definition and scoped task registration.
    This class manages the workflow graph and provides methods to add tasks,
    edges, and execute the workflow."""

    def __init__(
        self,
        name: str,
        tracer: Optional[Tracer] = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        """Initialize a new workflow context.

        Args:
            name: Name for this workflow
            tracer: Optional tracer for workflow execution tracking
            prompt_manager: Optional prompt manager for prompt template management
        """
        self.name = name
        self.graph = TaskGraph()
        self._task_counter = 0
        self._group_counter = 0
        self._redis_client: Optional[Any] = None
        self._tracer = tracer
        self._prompt_manager: Optional[PromptManager] = prompt_manager
        self._llm_agent_providers: dict[str, LLMAgentProvider] = {}
        self._token: Optional[contextvars.Token] = None

    def __enter__(self):
        """Enter the workflow context."""
        # Store previous context if any
        self._token = _current_context.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the workflow context."""
        # Restore previous context
        if self._token is not None:
            _current_context.reset(self._token)
            self._token = None

    def add_node(self, name: str, task: Executable, skip_if_exists: bool = False) -> None:
        """Add a task node to this workflow's graph."""
        self.graph.add_node(task, name, skip_if_exists=skip_if_exists)

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge between tasks in this workflow's graph."""
        self.graph.add_edge(from_node, to_node)

    def rename_node(self, old_task_id: str, new_task_id: str) -> None:
        """Rename a node in this workflow's graph.

        Args:
            old_task_id: Current task ID to rename
            new_task_id: New task ID to assign
        """
        self.graph.rename_node(old_task_id, new_task_id)

    def set_redis_client(self, redis_client: Any) -> None:
        """Set the Redis client for this workflow context.

        Args:
            redis_client: Redis client instance to be shared across the workflow
        """
        self._redis_client = redis_client

    def get_redis_client(self) -> Optional[Any]:
        """Get the Redis client for this workflow context.

        Returns:
            Redis client instance if set, None otherwise
        """
        return self._redis_client

    def register_llm_agent(self, name: str, agent_or_factory: LLMAgentProvider) -> None:
        """Register an LLMAgent to be attached when the workflow executes.

        Args:
            name: Identifier used when injecting the agent via ``@task(inject_llm_agent=...)``
            agent_or_factory: Either an ``LLMAgent`` instance or a callable that receives
                the ``ExecutionContext`` and returns an ``LLMAgent``. Factories are useful
                when the agent needs the workflow session_id or other runtime data.
        """
        self._llm_agent_providers[name] = agent_or_factory

    def execute(
        self,
        start_node: Optional[str] = None,
        max_steps: int = 10000,
        ret_context: bool = False,
        initial_channel: Optional[dict[str, Any]] = None,
    ) -> Any | tuple[Any, ExecutionContext]:
        """Execute the workflow starting from the specified node.

        Args:
            start_node: Optional starting node (auto-detected if None)
            max_steps: Maximum execution steps
            ret_context: If True, return (result, ExecutionContext) tuple
            initial_channel: Optional dict of initial channel values to set before execution

        Returns:
            Result from the last executed task handler (may be ``None``).
            If ret_context=True, returns tuple of (result, ExecutionContext).
        """

        if start_node is None:
            # Find start nodes (nodes with no predecessors)
            candidate_nodes = self.graph.get_start_nodes()
            if not candidate_nodes:
                raise GraphCompilationError("No start node specified and no nodes with no predecessors found.")
            elif len(candidate_nodes) > 1:
                raise GraphCompilationError("Multiple start nodes found, please specify one.")
            start_node = candidate_nodes[0]
            if start_node is None:
                raise GraphCompilationError("No valid start node found in the workflow graph.")

        from graflow.core.context import ExecutionContext
        from graflow.core.engine import WorkflowEngine

        exec_context = ExecutionContext.create(
            self.graph,
            start_node,
            max_steps=max_steps,
            tracer=self._tracer,
            prompt_manager=self._prompt_manager,
        )

        # Set initial channel values if provided
        if initial_channel:
            ch = exec_context.get_channel()
            for key, value in initial_channel.items():
                ch.set(key, value)

        self._attach_llm_agents(exec_context)

        engine = WorkflowEngine()
        result = engine.execute(exec_context)

        return (result, exec_context) if ret_context else result

    def show_info(self) -> None:
        """Display information about this workflow's graph."""
        print(f"=== Workflow '{self.name}' Information ===")
        print(self.graph)

        # Cycle detection
        cycles = self.graph.detect_cycles()
        if cycles:
            print(f"Cycles detected: {cycles}")
        else:
            print("No cycles detected")

    def visualize_dependencies(self) -> None:
        """Visualize task dependencies in this workflow."""
        print(f"=== Workflow '{self.name}' Dependencies ===")
        for node in self.graph.nodes:
            successors = self.graph.successors(node)
            if successors:
                print(f"{node} >> {' >> '.join(successors)}")
            else:
                print(f"{node} (no dependencies)")

    def clear(self) -> None:
        """Clear all tasks from this workflow."""
        self.graph.clear()
        self._task_counter = 0
        self._group_counter = 0

    def get_next_group_name(self) -> str:
        """Get the next group name for this workflow."""
        self._group_counter += 1
        return f"ParallelGroup_{self._group_counter}"

    def _attach_llm_agents(self, exec_context: ExecutionContext) -> None:
        """Attach workflow-level LLMAgents to the newly created ExecutionContext."""
        if not self._llm_agent_providers:
            return

        from graflow.llm.agents.base import LLMAgent

        for name, provider in self._llm_agent_providers.items():
            if isinstance(provider, LLMAgent):
                agent = provider
            elif callable(provider):
                agent = provider(exec_context)
            else:
                raise TypeError(f"LLMAgent provider for '{name}' must be LLMAgent or callable")
            exec_context.register_llm_agent(name, agent)


def get_current_workflow_context(create_if_not_exist: bool = False) -> Optional[WorkflowContext]:
    """Get the current workflow context if it exists.

    Args:
        create_if_not_exist: If True, create a new context if none exists.
                             If False, return None if no context exists.

    Returns:
        Current WorkflowContext or None if no context exists and create_if_not_exist is False.
    """
    ctx = _current_context.get()
    if ctx is None and create_if_not_exist:
        name = uuid.uuid4().hex
        ctx = WorkflowContext(name)
        _current_context.set(ctx)
    return ctx


def current_workflow_context() -> WorkflowContext:
    """Get the current workflow context if any.

    Args:
        create_if_not_exist: If True, create a new context if none exists.
                           If False, return None if no context exists.

    Returns:
        Current WorkflowContext or None if no context exists and create_if_not_exist is False.
    """
    ctx = _current_context.get()
    if ctx is None:
        name = uuid.uuid4().hex
        ctx = WorkflowContext(name)
        _current_context.set(ctx)
    return ctx


def require_workflow_context() -> WorkflowContext:
    """Get the current workflow context, raising an error if none exists.

    Returns:
        Current WorkflowContext
    Raises:
        GraflowRuntimeError: If no current workflow context exists.
    """
    ctx = _current_context.get()
    if ctx is None:
        raise GraflowRuntimeError("No current workflow context exists")
    return ctx


def set_current_workflow_context(context: WorkflowContext) -> None:
    """Set the current workflow context."""
    _current_context.set(context)


def clear_workflow_context() -> None:
    """Clear the current workflow context."""
    _current_context.set(None)


def workflow(
    name: str,
    tracer: Optional[Tracer] = None,
    prompt_manager: Optional[PromptManager] = None,
) -> WorkflowContext:
    """Context manager for creating a workflow.

    Args:
        name: Name of the workflow
        tracer: Optional tracer for workflow execution tracking
        prompt_manager: Optional prompt manager for prompt template management

    Returns:
        WorkflowContext instance

    Example:
        ```python
        from graflow.prompts import YAMLPromptManager

        # Create prompt manager
        prompt_manager = YAMLPromptManager(prompts_dir="./prompts")

        # Use in workflow
        with workflow("customer_engagement", prompt_manager=prompt_manager) as wf:
            task_a >> task_b
            wf.execute()
        ```
    """
    return WorkflowContext(name, tracer=tracer, prompt_manager=prompt_manager)
