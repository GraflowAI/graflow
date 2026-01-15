"""Execution engine for graflow with cycle support and global graph integration."""

from __future__ import annotations

import os
import re
import time
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar

from graflow.channels.base import Channel
from graflow.channels.factory import ChannelFactory
from graflow.channels.typed import TypedChannel
from graflow.core.cycle import CycleController
from graflow.core.engine import WorkflowEngine
from graflow.core.graph import TaskGraph
from graflow.exceptions import CycleLimitExceededError
from graflow.queue.base import TaskSpec
from graflow.queue.local import LocalTaskQueue
from graflow.trace.noop import NoopTracer

if TYPE_CHECKING:
    from graflow.core.task import Executable
    from graflow.hitl.types import FeedbackResponse, FeedbackType
    from graflow.llm.agents.base import LLMAgent
    from graflow.llm.client import LLMClient
    from graflow.prompts.base import PromptManager
    from graflow.trace.base import Tracer

T = TypeVar("T")

# Compiled regex pattern for iteration task detection (performance optimization)
_ITERATION_PATTERN: re.Pattern[str] = re.compile(r"(_cycle_\d+_[0-9a-f]+)+$")


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
    def _logger(self):
        """Get logger instance (lazy initialization to avoid module-level import)."""
        if not hasattr(self, "_logger_instance"):
            import logging

            self._logger_instance = logging.getLogger(__name__)
        return self._logger_instance

    @property
    def trace_id(self) -> str:
        """Get trace ID from execution context."""
        return self.execution_context.trace_id

    @property
    def session_id(self) -> str:
        """Get session ID from execution context."""
        return self.execution_context.session_id

    @property
    def graph(self) -> TaskGraph:
        """Get the task graph."""
        return self.execution_context.graph

    def can_iterate(self) -> bool:
        """Check if this task can execute another cycle."""
        return self.cycle_count < self.max_cycles

    def register_cycle(self) -> int:
        """Register a cycle execution and return new count."""
        if not self.can_iterate():
            raise ValueError(
                f"Cycle limit exceeded for task {self.task_id}: {self.cycle_count}/{self.max_cycles} cycles"
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

    def terminate_workflow(self, message: str) -> None:
        """Request workflow termination (normal exit).

        Args:
            message: Termination reason/message
        """
        self.execution_context.terminate_workflow(message)

    def cancel_workflow(self, message: str) -> None:
        """Request workflow cancellation (abnormal exit).

        Args:
            message: Cancellation reason/message
        """
        self.execution_context.cancel_workflow(message)

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

    # === LLM integration accessors ===

    @property
    def llm_client(self) -> LLMClient:
        """Get shared LLMClient instance from execution context.

        Returns:
            LLMClient instance (auto-created with default model if not set)

        Example:
            ```python
            @task(inject_context=True)
            def my_task(context: TaskExecutionContext):
                # Access LLMClient through context
                llm = context.llm_client
                response = llm.completion(
                    messages=[{"role": "user", "content": "Hello"}]
                )
                return response.choices[0].message.content
            ```
        """
        return self.execution_context.llm_client

    @property
    def prompt_manager(self) -> PromptManager:
        """Get prompt manager from execution context.

        Returns:
            PromptManager instance (auto-created with default YAML backend if not set)

        Example:
            ```python
            @task(inject_context=True)
            def my_task(context: TaskExecutionContext):
                # Access prompt manager through context
                pm = context.prompt_manager
                prompt = pm.get_prompt("greeting", label="production")
                rendered = prompt.render(name="Alice")
                return rendered
            ```
        """
        return self.execution_context.prompt_manager

    def get_llm_agent(self, name: str) -> LLMAgent:
        """Get registered LLMAgent by name from execution context.

        Args:
            name: Agent identifier

        Returns:
            LLMAgent instance

        Raises:
            KeyError: If agent not registered

        Example:
            ```python
            @task(inject_context=True)
            def my_task(context: TaskExecutionContext):
                # Access registered agent through context
                agent = context.get_llm_agent("researcher")
                result = agent.run("Research topic X")
                return result["output"]
            ```
        """
        return self.execution_context.get_llm_agent(name)

    def register_llm_agent(self, name: str, agent: LLMAgent) -> None:
        """Register an LLMAgent via the underlying execution context."""
        self.execution_context.register_llm_agent(name, agent)

    def set_local_data(self, key: str, value: Any) -> None:
        """Set task-local data."""
        self.local_data[key] = value

    def get_local_data(self, key: str, default: Any = None) -> Any:
        """Get task-local data."""
        return self.local_data.get(key, default)

    def elapsed_time(self) -> float:
        """Get elapsed time since task started."""
        return time.time() - self.start_time

    # === HITL (Human-in-the-Loop) integration ===

    def request_feedback(
        self,
        feedback_type: str | FeedbackType,  # str or FeedbackType enum
        prompt: str,
        options: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        timeout: float = 180.0,  # Default: 3 minutes
        channel_key: Optional[str] = None,
        write_to_channel: bool = False,
        handler: Optional[Any] = None,  # FeedbackHandler (avoid circular import)
    ) -> FeedbackResponse:
        """Request human feedback with optional callback handler.

        Args:
            feedback_type: Type of feedback ("approval", "text", "selection", etc.) or FeedbackType enum
            prompt: Prompt for human
            options: Options for selection types
            metadata: Custom metadata
            timeout: Polling timeout in seconds (default: 180 / 3 minutes)
            channel_key: Optional channel key to write response to
            write_to_channel: Whether to auto-write response to channel
            handler: Optional FeedbackHandler for callbacks (on_request_created, on_response_received, on_request_timeout)

        Returns:
            FeedbackResponse

        Raises:
            FeedbackTimeoutError: If timeout exceeded

        Example:
            ```python
            @task(inject_context=True)
            def my_task(context):
                # Approval feedback
                response = context.request_feedback(
                    feedback_type="approval",
                    prompt="Approve this deployment?",
                    timeout=180
                )
                if response.approved:
                    deploy()

                # Text input
                response = context.request_feedback(
                    feedback_type="text",
                    prompt="Enter your comment:",
                    timeout=180
                )
                comment = response.text

                # Selection
                response = context.request_feedback(
                    feedback_type="selection",
                    prompt="Choose mode:",
                    options=["fast", "balanced", "thorough"],
                    timeout=180
                )
                mode = response.selected

                # With channel integration
                response = context.request_feedback(
                    feedback_type="approval",
                    prompt="Approve deployment?",
                    channel_key="deployment_approved",
                    write_to_channel=True,
                    timeout=180
                )
                # Response automatically written to channel["deployment_approved"]
            ```
        """
        # Convert string to enum
        if isinstance(feedback_type, str):
            from graflow.hitl.types import FeedbackType

            feedback_type = FeedbackType(feedback_type)

        # Get feedback manager from execution context
        feedback_manager = self.execution_context.feedback_manager

        # Check if we're resuming from a checkpoint with an existing feedback_id
        # This allows reusing the same feedback_id after checkpoint resume
        existing_feedback_id = None
        if hasattr(self.execution_context, "checkpoint_metadata") and self.execution_context.checkpoint_metadata:
            # Check if this task has a pending feedback_id in checkpoint metadata
            user_metadata = self.execution_context.checkpoint_metadata.get("user_metadata", {})
            checkpoint_task_id = user_metadata.get("task_id")
            if checkpoint_task_id == self.task_id:
                existing_feedback_id = user_metadata.get("feedback_id")
                if existing_feedback_id:
                    self._logger.info(
                        "Found feedback_id in checkpoint metadata for task %s: %s",
                        self.task_id,
                        existing_feedback_id,
                        extra={"task_id": self.task_id, "feedback_id": existing_feedback_id},
                    )

        # Request feedback (with optional existing feedback_id for resume case)
        return feedback_manager.request_feedback(
            task_id=self.task_id,
            session_id=self.execution_context.session_id,
            feedback_type=feedback_type,
            prompt=prompt,
            options=options,
            metadata=metadata,
            timeout=timeout,
            channel_key=channel_key,
            write_to_channel=write_to_channel,
            feedback_id=existing_feedback_id,
            handler=handler,
        )

    def request_approval(
        self,
        prompt: str,
        metadata: Optional[dict] = None,
        timeout: float = 180.0,  # Default: 3 minutes
        channel_key: Optional[str] = None,
        write_to_channel: bool = False,
    ) -> bool:
        """Request approval (convenience method).

        Args:
            prompt: Approval prompt
            metadata: Custom metadata
            timeout: Polling timeout
            channel_key: Optional channel key to write response to
            write_to_channel: Whether to auto-write to channel

        Returns:
            True if approved, False if rejected
        """
        from graflow.hitl.types import FeedbackType

        response = self.request_feedback(
            feedback_type=FeedbackType.APPROVAL,
            prompt=prompt,
            metadata=metadata,
            timeout=timeout,
            channel_key=channel_key,
            write_to_channel=write_to_channel,
        )
        return bool(response.approved)

    def request_text_input(
        self,
        prompt: str,
        metadata: Optional[dict[str, Any]] = None,
        timeout: float = 180.0,
        channel_key: Optional[str] = None,
        write_to_channel: bool = False,
    ) -> str:
        """Request free-form text input."""
        from graflow.hitl.types import FeedbackType

        response = self.request_feedback(
            feedback_type=FeedbackType.TEXT,
            prompt=prompt,
            metadata=metadata,
            timeout=timeout,
            channel_key=channel_key,
            write_to_channel=write_to_channel,
        )
        return response.text or ""

    def request_selection(
        self,
        prompt: str,
        options: list[str],
        metadata: Optional[dict[str, Any]] = None,
        timeout: float = 180.0,
        channel_key: Optional[str] = None,
        write_to_channel: bool = False,
    ) -> str:
        """Request selection from provided options."""
        from graflow.hitl.types import FeedbackType

        response = self.request_feedback(
            feedback_type=FeedbackType.SELECTION,
            prompt=prompt,
            options=options,
            metadata=metadata,
            timeout=timeout,
            channel_key=channel_key,
            write_to_channel=write_to_channel,
        )
        return response.selected or ""

    # === Checkpointing ===

    def checkpoint(
        self, metadata: Optional[dict[str, Any]] = None, *, path: Optional[str] = None, deferred: bool = True
    ) -> Optional[tuple[str, Any]]:
        """Create or request a checkpoint from within the task.

        Args:
            metadata: User-supplied metadata stored alongside the checkpoint.
            path: Optional explicit checkpoint path. When omitted a path is generated.
            deferred: If True (default), the checkpoint is created after the
                current task completes and before successors execute. When False,
                the checkpoint is written immediately and the workflow will
                resume from this task when restored.

        Returns:
            (path, metadata) tuple for immediate checkpoints; None when deferred.
        """
        enriched_metadata: dict[str, Any] = dict(metadata) if metadata else {}
        enriched_metadata.setdefault("task_id", self.task_id)
        enriched_metadata.setdefault("cycle_count", self.cycle_count)
        enriched_metadata.setdefault("elapsed_time", self.elapsed_time())
        enriched_metadata.setdefault("mode", "deferred" if deferred else "immediate")

        if deferred:
            self.execution_context.request_checkpoint(
                enriched_metadata,
                path=path,
            )
            return None

        # Immediate checkpoint: capture current task_id so resume
        # continues from the same task.
        from graflow.core.checkpoint import CheckpointManager

        checkpoint_path, checkpoint_metadata = CheckpointManager.create_checkpoint(
            self.execution_context,
            path=path,
            metadata=enriched_metadata,
            resuming_task_id=self.task_id,
        )
        self.execution_context.checkpoint_metadata = checkpoint_metadata.to_dict()
        self.execution_context.last_checkpoint_path = checkpoint_path
        return checkpoint_path, checkpoint_metadata

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
        max_steps: int = 10000,
        default_max_cycles: int = 10,
        default_max_retries: int = 3,
        steps: int = 0,
        channel_backend: str = "memory",
        config: Optional[Dict[str, Any]] = None,
        parent_context: Optional[ExecutionContext] = None,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        tracer: Optional[Tracer] = None,
        llm_client: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        """Initialize ExecutionContext with configurable queue backend."""
        self.parent_context = parent_context

        # session_id: Unique identifier for this execution context (used for channels, isolation)
        # trace_id: Shared identifier for distributed tracing (W3C TraceContext-compliant 32-digit hex)
        # For root contexts: trace_id = session_id (both are unique)
        # For branch contexts: trace_id is inherited, session_id is unique
        self.session_id = session_id or uuid.uuid4().hex
        self.trace_id = trace_id or (parent_context.trace_id if parent_context else self.session_id)

        # Graph hash for distributed execution (Content-Addressable)
        self.graph_hash: Optional[str] = None

        # Each context gets its own tracer instance to avoid shared mutable state
        self.tracer = tracer if tracer is not None else NoopTracer()

        self.graph = graph
        self.start_node = start_node
        self.max_steps = max_steps
        self.default_max_retries = default_max_retries
        self.steps = steps

        # Preserve original config (without runtime additions)
        base_config: Dict[str, Any] = dict(config) if config else {}

        # Ensure redis_client has connection parameters for serialization
        from graflow.utils.redis_utils import ensure_redis_connection_params

        ensure_redis_connection_params(base_config)

        # Create channel using factory with same config (before removing redis_client)
        # ChannelFactory.create_channel() may use redis_client if present
        self.channel = ChannelFactory.create_channel(backend=channel_backend, name=self.session_id, **base_config)

        # Remove redis_client from base_config after channel creation
        # (contains unpicklable locks and is no longer needed after extraction)
        base_config.pop("redis_client", None)

        if parent_context is not None:
            # Prefill branch context channel with parent state
            for key in parent_context.channel.keys():
                self.channel.set(key, parent_context.channel.get(key))

        self.task_queue: LocalTaskQueue = LocalTaskQueue(self, start_node)
        self.cycle_controller = CycleController(default_max_cycles)

        # Task execution context management
        self._task_execution_stack: list[TaskExecutionContext] = []
        self._task_contexts: dict[str, TaskExecutionContext] = {}

        # Track if goto (jump to existing task) was called in current task execution
        self._goto_called_in_current_task: bool = False

        # Track workflow termination/cancellation requests
        self._terminate_called_in_current_task: bool = False
        self._cancel_called_in_current_task: bool = False
        self.ctrl_message: Optional[str] = None

        # Save backend configuration for serialization
        self._channel_backend_type = channel_backend
        self._original_config = base_config

        # Checkpoint-related state
        self.completed_tasks: list[str] = []  # Preserves execution order
        self.checkpoint_metadata: dict[str, Any] = {}
        self.last_checkpoint_path: Optional[str] = None
        self.checkpoint_requested: bool = False
        self.checkpoint_request_metadata: Optional[dict[str, Any]] = None
        self.checkpoint_request_path: Optional[str] = None

        # LLM integration
        self._llm_client: Optional[LLMClient] = llm_client
        self._llm_agents: Dict[str, Any] = {}  # LLMAgent registry
        self._llm_agents_yaml: Dict[str, str] = {}  # YAML serialization cache

        # Prompt management
        self._prompt_manager: Optional[PromptManager] = prompt_manager

        # HITL (Human-in-the-Loop) integration
        from graflow.hitl.manager import FeedbackManager

        # Pass channel to FeedbackManager for write_to_channel support
        # Use same backend as channel for consistency (redis for distributed, filesystem for local)
        feedback_backend = "redis" if channel_backend == "redis" else "filesystem"
        self.feedback_manager = FeedbackManager(
            backend=feedback_backend, backend_config=base_config, channel_manager=self.channel
        )

    @property
    def _logger(self):
        """Get logger instance (lazy initialization to avoid module-level import)."""
        if not hasattr(self, "_logger_instance"):
            import logging

            self._logger_instance = logging.getLogger(__name__)
        return self._logger_instance

    def create_branch_context(self, branch_id: str) -> ExecutionContext:
        """Create a child execution context for parallel execution.

        Branch contexts have:
        - Unique session_id for isolation (channels, identification)
          Pattern: {parent_session_id}_{branch_id} for traceability
        - Shared trace_id for distributed tracing (all branches in same trace)
        - Cloned tracer with parent span context inherited from tracer state

        This ensures proper W3C TraceContext compliance while maintaining execution isolation.
        The session_id pattern allows inferring the parent-child relationship from the ID.

        Args:
            branch_id: Identifier for this branch (used for session_id uniqueness)

        Returns:
            New ExecutionContext with shared trace context but isolated execution state
        """
        # Use deterministic pattern for session_id to maintain parent-child traceability
        branch_session_id = f"{self.session_id}_{branch_id}"

        # Clone tracer (parent span context inherited from tracer state)
        cloned_tracer = None
        if self.tracer:
            cloned_tracer = self.tracer.clone(self.trace_id)

        branch_context = ExecutionContext(
            graph=self.graph,
            start_node=None,
            max_steps=self.max_steps,
            default_max_cycles=self.cycle_controller.default_max_cycles,
            default_max_retries=self.default_max_retries,
            channel_backend=self._channel_backend_type,
            config=self._original_config,
            parent_context=self,
            session_id=branch_session_id,  # Hierarchical session ID
            trace_id=self.trace_id,  # Shared trace ID for distributed tracing
            tracer=cloned_tracer,
            prompt_manager=self._prompt_manager,  # Share prompt manager with branch
        )
        return branch_context

    def merge_results(self, sub_context: ExecutionContext) -> None:
        """Merge channel data and metrics from a completed branch context."""
        if self.channel is not sub_context.channel:
            for key in sub_context.channel.keys():
                self.channel.set(key, sub_context.channel.get(key))

        self.steps += sub_context.steps
        self.cycle_controller.cycle_counts.update(sub_context.cycle_controller.cycle_counts)

    def mark_branch_completed(self, branch_id: str) -> None:
        """Hook for branch completion bookkeeping (currently a no-op)."""
        # Placeholder for barrier synchronization or logging extensions.
        return

    @classmethod
    def create(
        cls,
        graph: TaskGraph,
        start_node: Optional[str] = None,
        max_steps: int = 10000,
        default_max_cycles: int = 10,
        default_max_retries: int = 3,
        channel_backend: str = "memory",
        config: Optional[Dict[str, Any]] = None,
        tracer: Optional[Tracer] = None,
        llm_client: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,
    ) -> ExecutionContext:
        """Create a new execution context with configurable channel backend.

        Args:
            graph: Task graph defining the workflow
            start_node: Optional starting task node (can be None for checkpoint scenarios)
            max_steps: Maximum execution steps
            default_max_cycles: Default maximum cycles for tasks
            default_max_retries: Default maximum retry attempts
            channel_backend: Backend for inter-task communication (default: memory)
            config: Configuration applied to both queue and channel (e.g., redis_client, key_prefix)
            tracer: Optional tracer for workflow execution tracking (default: creates new NoopTracer)
            llm_client: Optional LLMClient instance for LLM integration
            prompt_manager: Optional PromptManager instance for prompt template management

        Example:
            ```python
            from graflow.llm import LLMClient
            from graflow.prompts import YAMLPromptManager

            # Create LLMClient
            llm_client = LLMClient(model="gpt-5-mini", temperature=0.7)

            # Create context with LLMClient
            context = ExecutionContext.create(
                graph, start_node,
                llm_client=llm_client
            )

            # Create context with prompt manager
            prompt_manager = YAMLPromptManager(prompts_dir="./prompts")
            context = ExecutionContext.create(
                graph, start_node,
                prompt_manager=prompt_manager
            )

            # Create context with Redis backend for HITL
            import redis
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            context = ExecutionContext.create(
                graph, start_node,
                feedback_backend="redis",
                feedback_config={"redis_client": redis_client}
            )
            ```
        """
        if start_node is None:
            # Find start nodes (nodes with no predecessors)
            candidate_nodes = graph.get_start_nodes()
            if candidate_nodes:
                start_node = candidate_nodes[0]  # REVIEWME: Choose first start node if multiple exist

        return cls(
            graph=graph,
            start_node=start_node,
            max_steps=max_steps,
            default_max_cycles=default_max_cycles,
            default_max_retries=default_max_retries,
            channel_backend=channel_backend,
            config=config,
            tracer=tracer,
            llm_client=llm_client,
            prompt_manager=prompt_manager,
        )

    @property
    def queue(self) -> LocalTaskQueue:
        """Get the task queue instance."""
        return self.task_queue

    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration dictionary."""
        return self._original_config

    def add_to_queue(self, executable: Executable) -> None:
        """Add executable to execution queue with trace context."""
        # Always set trace context for distributed tracing
        # Trace ID (workflow-wide ID, shared across all branches)
        trace_id = self.trace_id

        # Parent span ID (currently executing task ID)
        parent_span_id = self.current_task_id if hasattr(self, "current_task_id") else None

        task_spec = TaskSpec(
            executable=executable,
            execution_context=self,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )
        self.task_queue.enqueue(task_spec)

        # Tracer: Task queued
        self.tracer.on_task_queued(executable, self)

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
        if self._task_execution_stack:
            return self._task_execution_stack[-1]
        if self.parent_context is not None:
            return self.parent_context.current_task_context
        return None

    @property
    def current_task_id(self) -> Optional[str]:
        """Get the currently executing task ID (backward compatibility)."""
        ctx = self.current_task_context
        return ctx.task_id if ctx else None

    # === LLM integration ===

    @property
    def llm_client(self) -> LLMClient:
        """Get LLM client instance.

        Lazily creates a default LLMClient if not explicitly set.
        Default model is resolved from GRAFLOW_LLM_MODEL environment variable,
        falling back to "gpt-5-mini".

        Returns:
            LLMClient instance

        Example:
            ```python
            # .env file:
            # GRAFLOW_LLM_MODEL=gpt-4o

            # Access LLM client (auto-created with default model)
            client = context.llm_client
            response = client.completion([{"role": "user", "content": "Hello"}])

            # Or inject explicitly
            from graflow.llm import LLMClient
            llm_client = LLMClient(model="gpt-5-mini", temperature=0.7)
            context = ExecutionContext.create(
                graph, start_node,
                llm_client=llm_client
            )
            ```
        """
        if self._llm_client is None:
            # Lazy initialization: create default LLMClient
            from graflow.llm.client import LLMClient
            from graflow.utils.dotenv import load_env

            # Load environment variables from .env file
            load_env()

            # Get default model from environment or use fallback
            default_model = os.getenv("GRAFLOW_LLM_MODEL", "gpt-5-mini")

            self._llm_client = LLMClient(model=default_model)

        return self._llm_client

    @property
    def prompt_manager(self) -> PromptManager:
        """Get prompt manager instance.

        Lazily creates a default YAMLPromptManager if not explicitly set.
        Default prompts directory is resolved from GRAFLOW_PROMPTS_DIR environment variable,
        falling back to "./prompts" in the current working directory.

        Returns:
            PromptManager instance

        Example:
            ```python
            # .env file:
            # GRAFLOW_PROMPTS_DIR=./my_prompts

            # Access prompt manager (auto-created with default YAML backend)
            pm = context.prompt_manager
            prompt = pm.get_prompt("greeting", label="production")
            rendered = prompt.render(name="Alice")

            # Or inject explicitly
            from graflow.prompts import YAMLPromptManager
            prompt_manager = YAMLPromptManager(prompts_dir="./prompts")
            context = ExecutionContext.create(
                graph, start_node,
                prompt_manager=prompt_manager
            )
            ```
        """
        if self._prompt_manager is None:
            # Lazy initialization: create default YAMLPromptManager
            # Constructor handles GRAFLOW_PROMPTS_DIR env var and default resolution
            from graflow.prompts.yaml_manager import YAMLPromptManager

            self._prompt_manager = YAMLPromptManager()

        return self._prompt_manager

    def register_llm_agent(self, name: str, agent: LLMAgent) -> None:
        """Register LLM Agent for use in tasks.

        For distributed execution, agents are serialized to YAML using Google ADK's
        built-in serialization and stored in _llm_agents_yaml for worker processes.

        Args:
            name: Agent identifier (used in @task(inject_llm_agent=True, agent_name=...))
            agent: LLMAgent instance (e.g., AdkLLMAgent)

        Example:
            ```python
            from google.adk.agents import BaseAgent
            from graflow.llm import AdkLLMAgent

            # Create ADK agent
            adk_agent = BaseAgent(
                name="supervisor",
                model="gemini-2.5-flash",
                tools=[search_tool]
            )
            agent = AdkLLMAgent(adk_agent)

            # Register for use in tasks
            context.register_llm_agent("supervisor", agent)
            ```
        """
        self._llm_agents[name] = agent

        # Serialize agent to YAML for distributed execution (if AdkLLMAgent)
        try:
            from graflow.llm.agents.adk_agent import AdkLLMAgent
            from graflow.llm.serialization import agent_to_yaml

            if isinstance(agent, AdkLLMAgent):
                yaml_str = agent_to_yaml(agent._adk_agent)
                self._llm_agents_yaml[name] = yaml_str
        except (ImportError, AttributeError):
            # ADK not available or agent is not AdkLLMAgent, skip serialization
            pass

    def get_llm_agent(self, name: str) -> LLMAgent:
        """Get registered LLM Agent by name.

        For distributed execution, agents are lazily restored from YAML in worker
        processes on first access.

        Args:
            name: Agent identifier

        Returns:
            LLMAgent instance

        Raises:
            KeyError: If agent not found in registry

        Example:
            ```python
            @task(inject_llm_agent=True, agent_name="supervisor")
            def run_analysis(agent: LLMAgent, query: str) -> str:
                # Agent is automatically injected by decorator
                result = agent.run(query)
                return result["output"]
            ```
        """
        # Check if agent is already in memory
        if name in self._llm_agents:
            return self._llm_agents[name]

        # Try to restore from YAML (distributed execution scenario)
        if name in self._llm_agents_yaml:
            try:
                from graflow.llm.agents.adk_agent import AdkLLMAgent
                from graflow.llm.serialization import yaml_to_agent

                # Restore ADK agent from YAML
                adk_agent = yaml_to_agent(self._llm_agents_yaml[name])
                agent = AdkLLMAgent._from_adk_agent(adk_agent, self.session_id)

                # Cache for future access
                self._llm_agents[name] = agent
                return agent
            except (ImportError, Exception):
                # ADK not available or deserialization failed
                pass

        raise KeyError(f"LLMAgent '{name}' not found in registry")

    # === Checkpoint helpers ===

    def mark_task_completed(self, task_id: str) -> None:
        """Track task completion for checkpoint state."""
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)

    def request_checkpoint(
        self,
        metadata: Optional[dict[str, Any]] = None,
        *,
        path: Optional[str] = None,
    ) -> None:
        """Request a checkpoint after current task completes."""
        self.checkpoint_requested = True
        # Store a shallow copy to avoid accidental mutation by caller
        self.checkpoint_request_metadata = dict(metadata) if metadata else {}
        self.checkpoint_request_path = path

    def clear_checkpoint_request(self) -> None:
        """Reset checkpoint request flags."""
        self.checkpoint_requested = False
        self.checkpoint_request_metadata = None
        self.checkpoint_request_path = None

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Collect minimal checkpoint state for serialization."""
        return {
            "schema_version": "1.0",
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "start_node": self.start_node,
            "steps": self.steps,
            "completed_tasks": self.completed_tasks,
            "cycle_counts": dict(self.cycle_controller.cycle_counts),
            "backend": {
                "channel": self._channel_backend_type,
            },
        }

    @property
    def goto_called(self) -> bool:
        """Check if goto was called in current task execution."""
        return self._goto_called_in_current_task

    def reset_goto_flag(self) -> None:
        """Reset goto flag for next task execution."""
        self._goto_called_in_current_task = False

    @property
    def terminate_called(self) -> bool:
        """Check if workflow termination was requested in current task execution."""
        return self._terminate_called_in_current_task

    @property
    def cancel_called(self) -> bool:
        """Check if workflow cancellation was requested in current task execution."""
        return self._cancel_called_in_current_task

    def reset_terminate_flag(self) -> None:
        """Reset terminate flag for next task execution."""
        self._terminate_called_in_current_task = False
        self.ctrl_message = None

    def reset_cancel_flag(self) -> None:
        """Reset cancel flag for next task execution."""
        self._cancel_called_in_current_task = False
        self.ctrl_message = None

    def reset_control_flags(self) -> None:
        """Reset all control flow flags for next task execution."""
        self._goto_called_in_current_task = False
        self._terminate_called_in_current_task = False
        self._cancel_called_in_current_task = False
        self.ctrl_message = None

    def terminate_workflow(self, message: str) -> None:
        """Request workflow termination (normal exit).

        This method allows a task to request graceful workflow termination.
        When called, the current task completes normally but no subsequent
        tasks are executed. The workflow exits with a successful status.

        Use this when:
        - A condition is met that makes further processing unnecessary
        - Early exit is desired without indicating an error

        Args:
            message: Reason for termination (for logging/debugging)

        Example:
            @task(inject_context=True)
            def check_cache(context):
                if cache_hit:
                    context.terminate_workflow("Data found in cache")
                return result
        """
        self._terminate_called_in_current_task = True
        self.ctrl_message = message
        self._logger.info(
            "Workflow termination requested: %s", message, extra={"session_id": self.session_id, "ctrl_msg": message}
        )

    def cancel_workflow(self, message: str) -> None:
        """Request workflow cancellation (abnormal exit).

        This method allows a task to cancel the entire workflow execution
        due to an error or invalid state. When called, the workflow immediately
        raises a GraflowWorkflowCanceledError, and the current task is NOT
        marked as completed.

        Use this when:
        - Invalid data is detected that prevents continuation
        - A critical error occurs that invalidates the workflow
        - User explicitly requests cancellation

        Args:
            message: Reason for cancellation (for logging/debugging)

        Example:
            @task(inject_context=True)
            def validate_data(context, data):
                if not data.is_valid():
                    context.cancel_workflow("Invalid data format")
                return processed_data
        """
        self._cancel_called_in_current_task = True
        self.ctrl_message = message
        self._logger.warning(
            "Workflow cancellation requested: %s", message, extra={"session_id": self.session_id, "ctrl_msg": message}
        )

    def next_task(self, executable: Executable, goto: bool = False, _is_iteration: bool = False) -> str:
        """Generate a new task or jump to existing task node.

        For iteration/cycle tasks, use next_iteration() instead.

        Args:
            executable: Executable object to execute as the new task
            goto: If True, skip successors of current task (works for both existing and new tasks)
            _is_iteration: Internal flag to mark iteration tasks (set by next_iteration)

        Returns:
            The task ID from the executable
        """
        task_id = executable.task_id
        is_new_task = task_id not in self.graph.nodes

        if goto:
            # Explicit goto: Skip successors regardless of whether task is new or existing
            if is_new_task:
                # New task: Create it but still skip successors
                self._logger.debug(
                    "Goto: Creating new task (skip successors): %s",
                    task_id,
                    extra={"session_id": self.session_id, "goto": True, "is_new": True},
                )
                self.graph.add_node(executable, task_id)
            else:
                # Existing task: Jump to it
                self._logger.debug(
                    "Goto: Jumping to existing task: %s",
                    task_id,
                    extra={"session_id": self.session_id, "goto": True, "is_new": False},
                )
            self.add_to_queue(executable)
            self._goto_called_in_current_task = True
        # Auto-detect behavior (no goto specified)
        elif is_new_task:
            # New task: Create dynamic task (normal successor processing)
            self._logger.debug(
                "Creating new dynamic task: %s", task_id, extra={"session_id": self.session_id, "is_dynamic": True}
            )
            self.graph.add_node(executable, task_id)
            self.add_to_queue(executable)
            # Note: _goto_called_in_current_task remains False for normal processing
        else:
            # Existing task: Jump to it (auto-detected, skip successors)
            self._logger.debug(
                "Jumping to existing task: %s", task_id, extra={"session_id": self.session_id, "auto_goto": True}
            )
            self.add_to_queue(executable)
            self._goto_called_in_current_task = True

        # Tracer: Dynamic task added
        current_task_id = self.current_task_id if hasattr(self, "current_task_id") else None
        self.tracer.on_dynamic_task_added(
            task_id=task_id,
            parent_task_id=current_task_id,
            is_iteration=_is_iteration,
            metadata={"task_type": type(executable).__name__, "is_new_task": is_new_task, "goto": goto},
        )

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

        # Extract base task ID (strip _cycle_* suffix if present)
        # This handles nested iterations where task_id might be "task_cycle_1_abc_cycle_2_def"
        base_task_id = _ITERATION_PATTERN.sub("", task_id)
        if base_task_id and base_task_id in self.graph.nodes:
            task_id = base_task_id

        if task_id not in self.graph.nodes:
            raise ValueError(f"Task {task_id} not found in graph")

        # Get or create task context
        task_ctx = self._task_contexts.get(task_id)
        if not task_ctx:
            task_ctx = self.create_task_context(task_id)

        # Use task context for cycle management
        if not task_ctx.can_iterate():
            raise CycleLimitExceededError(
                task_id=task_id, cycle_count=task_ctx.cycle_count, max_cycles=task_ctx.max_cycles
            )

        # Register this cycle execution
        cycle_count = task_ctx.register_cycle()

        # Get the current task function
        current_task = self.graph.get_node(task_id)

        # Generate iteration task ID with cycle count
        iteration_id = f"{task_id}_cycle_{cycle_count}_{uuid.uuid4().hex[:8]}"

        # Create iteration function with data
        def iteration_func():
            # Set execution context on current_task before calling it
            current_task.set_execution_context(task_ctx.execution_context)

            # Check if current_task has inject_context
            inject_context = bool(getattr(current_task, "inject_context", False))
            if inject_context:
                # Don't pass task_ctx, only pass data (TaskWrapper will inject context)
                if data is not None:
                    return current_task(data)
                else:
                    return current_task()
            # Pass task_ctx for tasks without inject_context
            elif data is not None:
                return current_task(task_ctx, data)
            else:
                return current_task(task_ctx)

        from .task import TaskWrapper

        iteration_task = TaskWrapper(iteration_id, iteration_func, inject_context=False)

        # Add iteration task via next_task with _is_iteration=True
        return self.next_task(iteration_task, _is_iteration=True)

    @contextmanager
    def executing_task(self, task: Executable):
        """Context manager for task execution with proper cleanup.

        Calls tracer hooks for task start/end automatically.

        Args:
            task: The task being executed

        Yields:
            TaskExecutionContext: The task execution context
        """
        task_ctx = self.create_task_context(task.task_id)

        # Call tracer hook: task start (before pushing to stack)
        # This ensures current_task_id points to parent task
        self.tracer.on_task_start(task, self)

        # Push task context to stack after tracer hook
        self.push_task_context(task_ctx)

        error: Optional[Exception] = None

        try:
            task.set_execution_context(self)
            yield task_ctx
            # Result will be set by handler, we don't need to retrieve it here
        except Exception as e:
            error = e
            raise
        finally:
            # Call tracer hook: task end (always called, even on error)
            # Note: result is passed as None since it's stored in context by handler
            self.tracer.on_task_end(task, self, result=None, error=error)
            self.pop_task_context()

    def execute(self) -> Any:
        """Execute tasks using this context.

        Returns:
            Result from the last executed task handler (may be ``None``).
        """
        engine = WorkflowEngine()
        return engine.execute(self)

    def __getstate__(self) -> dict:
        """Pickle serialization: exclude un-serializable objects.

        Removes Redis clients, Locks, and other un-picklable objects,
        saving only the configuration needed to reconstruct them.

        Returns:
            Dictionary of serializable state
        """
        state = self.__dict__.copy()

        # Save channel data before removing channel object
        # For MemoryChannel: save the data dict
        # For RedisChannel: data persists in Redis, no need to save
        channel_data = {}
        channel_backend_type = state.get("_channel_backend_type", "memory")

        # Only save channel data for memory backend
        if channel_backend_type == "memory" and self.channel is not None:
            try:
                # Get all keys from channel and save their values
                for key in self.channel.keys():
                    channel_data[key] = self.channel.get(key)
            except Exception:
                # If getting keys fails, skip channel data preservation
                pass
        state["_channel_data"] = channel_data

        # Remove un-serializable objects (will be reconstructed in __setstate__)
        state.pop("task_queue", None)
        state.pop("channel", None)
        state.pop("feedback_manager", None)  # Will be reconstructed in __setstate__

        # LLM: Exclude agent instances (only keep YAML for distributed execution)
        state["_llm_agents"] = {}  # Agent instances not serialized

        # Ensure backend config is saved (should already be set in __init__)
        if "_channel_backend_type" not in state:
            state["_channel_backend_type"] = "memory"
        if "_original_config" not in state:
            state["_original_config"] = {}

        # Remove redis_client from _original_config (connection params were already extracted in __init__)
        # The redis_client object contains unpicklable thread locks (RLock)
        original_config = state.get("_original_config", None)
        if original_config:
            cleaned_config = original_config.copy()
            if "redis_client" in cleaned_config:
                del cleaned_config["redis_client"]
            state["_original_config"] = cleaned_config

        return state

    def __setstate__(self, state: dict) -> None:
        """Pickle deserialization: reconstruct excluded objects.

        Reconstructs TaskQueue, Channel, etc. from saved configuration.

        Args:
            state: Serialized state dictionary
        """
        self.__dict__.update(state)

        # Get backend configuration
        channel_backend_type = state.get("_channel_backend_type", "memory")
        config = state.get("_original_config", {})

        # Add start_node to config for queue
        if self.start_node:
            config = {**config, "start_node": self.start_node}

        # Reconstruct TaskQueue (always in-memory after simplification)
        from graflow.channels.factory import ChannelFactory

        queue_start_node = config.get("start_node")
        self.task_queue = LocalTaskQueue(self, queue_start_node)

        # Reconstruct Channel
        self.channel = ChannelFactory.create_channel(backend=channel_backend_type, name=self.session_id, **config)

        # Restore channel data (for MemoryChannel)
        # For RedisChannel, data already exists in Redis
        channel_data = state.get("_channel_data", {})
        if channel_data:
            for key, value in channel_data.items():
                self.channel.set(key, value)

        # Ensure checkpoint attributes exist for older checkpoints
        if not hasattr(self, "completed_tasks") or self.completed_tasks is None:
            self.completed_tasks = []
        if not hasattr(self, "checkpoint_metadata") or self.checkpoint_metadata is None:
            self.checkpoint_metadata = {}
        if not hasattr(self, "last_checkpoint_path"):
            self.last_checkpoint_path = None
        self.checkpoint_requested = False
        self.checkpoint_request_metadata = None
        self.checkpoint_request_path = None

        # Ensure LLM attributes exist for older checkpoints
        if not hasattr(self, "_llm_client"):
            self._llm_client = None
        if not hasattr(self, "_llm_agents"):
            self._llm_agents = {}
        if not hasattr(self, "_llm_agents_yaml"):
            self._llm_agents_yaml = {}

        # Ensure prompt manager attribute exists for older checkpoints
        if not hasattr(self, "_prompt_manager"):
            self._prompt_manager = None

        # Reconstruct FeedbackManager
        # Use same backend as channel for consistency (redis for distributed, filesystem for local)
        feedback_backend = "redis" if channel_backend_type == "redis" else "filesystem"
        from graflow.hitl.manager import FeedbackManager

        self.feedback_manager = FeedbackManager(
            backend=feedback_backend, backend_config=config, channel_manager=self.channel
        )

    def save(self, path: str = "execution_context.pkl") -> None:
        """Save execution context to a pickle file using cloudpickle.

        Args:
            path: Path to save the context

        Note:
            Uses cloudpickle for better support of lambdas and closures.
        """
        from graflow.core.serialization import dump

        with open(path, "wb") as f:
            dump(self, f)

    @classmethod
    def load(cls, path: str = "execution_context.pkl") -> ExecutionContext:
        """Load execution context from a pickle file using cloudpickle.

        Args:
            path: Path to load the context from

        Returns:
            Loaded ExecutionContext instance
        """
        from graflow.core.serialization import load

        with open(path, "rb") as f:
            return load(f)


def create_execution_context(
    start_node: str = "ROOT", max_steps: int = 10, tracer: Optional[Tracer] = None
) -> ExecutionContext:
    """Create an initial execution context with a single root node.

    Args:
        start_node: Starting task node
        max_steps: Maximum execution steps
        tracer: Optional tracer for workflow execution tracking
    """
    from graflow.core.task import Task
    from graflow.trace.noop import NoopTracer

    if tracer is None:
        tracer = NoopTracer()
    graph = TaskGraph()
    # Add dummy task for start_node to satisfy validation
    graph.add_node(Task(start_node), start_node)
    return ExecutionContext.create(graph, start_node, max_steps=max_steps, tracer=tracer)


def execute_with_cycles(graph: TaskGraph, start_node: str, max_steps: int = 10) -> None:
    """Execute tasks allowing cycles from global graph."""
    engine = WorkflowEngine()
    engine.execute_with_cycles(graph, start_node, max_steps)
