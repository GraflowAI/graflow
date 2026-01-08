"""Abstract base class for workflow tracing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import networkx as nx

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.task import Executable


class Tracer(ABC):
    """Abstract base class for workflow execution tracing.

    Provides:
    - Automatic runtime graph tracking (when enabled)
    - Execution order tracking
    - Template methods for trace/span/event lifecycle
    - Abstract output methods for subclasses to implement

    Subclasses must implement _output_* methods to provide
    concrete tracing behavior (e.g., console output, LangFuse integration).

    Design Pattern:
        Uses Template Method pattern - base class handles graph tracking,
        subclasses handle output formatting.
    """

    def __init__(self, enable_runtime_graph: bool = True):
        """Initialize tracer.

        Args:
            enable_runtime_graph: If True, track runtime execution graph
        """
        self.enable_runtime_graph = enable_runtime_graph
        self._runtime_graph: Optional[nx.DiGraph] = nx.DiGraph() if enable_runtime_graph else None
        self._execution_order: List[str] = []
        self._current_trace_id: Optional[str] = None
        self._span_stack: List[str] = []  # Track nested spans

    # === Concrete lifecycle methods with automatic graph tracking ===

    def trace_start(self, name: str, trace_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start a new trace (root span).

        Automatically tracks in runtime graph if enabled.
        Calls _output_trace_start for subclass-specific output.

        Args:
            name: Trace name (typically workflow name)
            trace_id: Optional trace ID (W3C TraceContext format: 32-digit hex)
            metadata: Optional metadata dictionary
        """
        self._current_trace_id = trace_id

        # Automatic runtime graph tracking
        if self.enable_runtime_graph:
            self._add_node_to_runtime_graph(
                name, status="running", metadata={"type": "trace", "trace_id": trace_id, **(metadata or {})}
            )

        # Call subclass output method
        self._output_trace_start(name, trace_id, metadata)

    def trace_end(self, name: str, output: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """End the current trace.

        Automatically updates runtime graph if enabled.
        Calls _output_trace_end for subclass-specific output.

        Args:
            name: Trace name (must match trace_start)
            output: Optional output/result
            metadata: Optional metadata dictionary
        """
        # Automatic runtime graph tracking
        if self.enable_runtime_graph:
            self._update_node_in_runtime_graph(
                name,
                status="completed",
                end_time=datetime.now(),
                metadata={"output": str(output) if output else None, **(metadata or {})},
            )

        # Call subclass output method
        self._output_trace_end(name, output, metadata)

    def span_start(
        self, name: str, parent_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Start a new span (child of current trace or parent span).

        Automatically tracks in runtime graph if enabled.
        Calls _output_span_start for subclass-specific output.

        Args:
            name: Span name (typically task_id)
            parent_name: Optional parent span name
            metadata: Optional metadata dictionary (e.g., task_type, handler_type)
        """
        # Automatic runtime graph tracking
        if self.enable_runtime_graph:
            self._add_node_to_runtime_graph(name, status="running", metadata={"type": "span", **(metadata or {})})
            if parent_name:
                self._add_edge_to_runtime_graph(parent_name, name, relation="parent-child")

        # Call subclass output method
        self._output_span_start(name, parent_name, metadata)

    def span_end(self, name: str, output: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """End a span.

        Automatically updates runtime graph if enabled.
        Calls _output_span_end for subclass-specific output.

        Args:
            name: Span name (must match span_start)
            output: Optional output/result
            metadata: Optional metadata dictionary (e.g., status, error)
        """
        error = metadata.get("error") if metadata else None
        status = "failed" if error else "completed"

        # Automatic runtime graph tracking
        if self.enable_runtime_graph:
            self._update_node_in_runtime_graph(
                name,
                status=status,
                end_time=datetime.now(),
                metadata={"output": str(output) if output else None, **(metadata or {})},
            )

        # Call subclass output method
        self._output_span_end(name, output, metadata)

    def event(self, name: str, parent_span: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record an event (point-in-time observation).

        Events are not tracked in runtime graph by default.
        Calls _output_event for subclass-specific output.

        Args:
            name: Event name (e.g., "task_queued", "checkpoint_created")
            parent_span: Optional parent span name
            metadata: Optional metadata dictionary
        """
        # Call subclass output method
        self._output_event(name, parent_span, metadata)

    # === Abstract output methods for subclasses ===

    @abstractmethod
    def _output_trace_start(self, name: str, trace_id: Optional[str], metadata: Optional[Dict[str, Any]]) -> None:
        """Output trace start (subclass implements output logic)."""
        pass

    @abstractmethod
    def _output_trace_end(self, name: str, output: Optional[Any], metadata: Optional[Dict[str, Any]]) -> None:
        """Output trace end (subclass implements output logic)."""
        pass

    @abstractmethod
    def _output_span_start(self, name: str, parent_name: Optional[str], metadata: Optional[Dict[str, Any]]) -> None:
        """Output span start (subclass implements output logic)."""
        pass

    @abstractmethod
    def _output_span_end(self, name: str, output: Optional[Any], metadata: Optional[Dict[str, Any]]) -> None:
        """Output span end (subclass implements output logic)."""
        pass

    @abstractmethod
    def _output_event(self, name: str, parent_span: Optional[str], metadata: Optional[Dict[str, Any]]) -> None:
        """Output event (subclass implements output logic)."""
        pass

    # === Workflow-level hooks (concrete implementations) ===

    def on_workflow_start(self, workflow_name: str, context: ExecutionContext) -> None:
        """Called when workflow execution starts.

        Default implementation: starts a trace with trace_id from context.

        Args:
            workflow_name: Workflow name
            context: ExecutionContext
        """
        self.trace_start(
            workflow_name,
            trace_id=context.trace_id,
            metadata={
                "start_node": context.start_node,
                "max_steps": context.max_steps,
                "session_id": context.session_id,
            },
        )

    def on_workflow_end(self, workflow_name: str, context: ExecutionContext, result: Optional[Any] = None) -> None:
        """Called when workflow execution ends.

        Default implementation: ends the trace.

        Args:
            workflow_name: Workflow name
            context: ExecutionContext
            result: Optional workflow result
        """
        self.trace_end(workflow_name, output=result, metadata={"steps": context.steps})

    # === Task-level hooks (concrete implementations) ===

    def on_task_queued(self, task: Executable, context: ExecutionContext) -> None:
        """Called when task is added to execution queue.

        Default implementation: no-op (can be overridden).

        Args:
            task: Executable task
            context: ExecutionContext
        """
        pass  # Default: no-op (intentional, not abstract)

    def on_task_start(self, task: Executable, context: ExecutionContext) -> None:
        """Called when task execution starts.

        Default implementation: starts a span for the task.

        Args:
            task: Executable task
            context: ExecutionContext
        """
        parent_task_id = None
        if hasattr(context, "current_task_id") and context.current_task_id:
            parent_task_id = context.current_task_id

        self.span_start(
            task.task_id,
            parent_name=parent_task_id,
            metadata={"task_type": type(task).__name__, "handler_type": getattr(task, "handler_type", "direct")},
        )

    def on_task_end(
        self,
        task: Executable,
        context: ExecutionContext,
        result: Optional[Any] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Called when task execution ends.

        Default implementation: ends the span for the task.

        Args:
            task: Executable task
            context: ExecutionContext
            result: Optional task result
            error: Optional exception if task failed
        """
        metadata = {}
        if error:
            metadata["error"] = str(error)
            metadata["error_type"] = type(error).__name__

        self.span_end(task.task_id, output=result, metadata=metadata)

    # === Parallel group hooks (concrete implementations) ===

    def on_parallel_group_start(self, group_id: str, member_ids: List[str], context: ExecutionContext) -> None:
        """Called when ParallelGroup execution starts.

        Default implementation: adds parallel-member edges to runtime graph
        if both parent and child nodes exist.

        Note: In actual execution, member task nodes may not exist yet when
        this hook is called. Edges will be created only if both nodes exist.

        Args:
            group_id: ParallelGroup task ID
            member_ids: List of member task IDs
            context: ExecutionContext
        """
        # Add parallel relationships in runtime graph
        # Note: Member nodes may not exist yet, so check before adding edge
        if self.enable_runtime_graph and self._runtime_graph is not None:
            for member_id in member_ids:
                # Only add edge if both nodes exist
                if group_id in self._runtime_graph and member_id in self._runtime_graph:
                    self._add_edge_to_runtime_graph(group_id, member_id, relation="parallel-member")

    def on_parallel_group_end(
        self, group_id: str, member_ids: List[str], context: ExecutionContext, results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called when ParallelGroup execution ends.

        Default implementation: no-op (can be overridden).

        Args:
            group_id: ParallelGroup task ID
            member_ids: List of member task IDs
            context: ExecutionContext
            results: Optional dict of member results
        """
        pass  # Default: no-op (intentional, not abstract)

    def on_dynamic_task_added(
        self,
        task_id: str,
        parent_task_id: Optional[str] = None,
        is_iteration: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called when a dynamic task is added to the workflow.

        Default implementation: no-op (can be overridden).

        Args:
            task_id: Task ID of the dynamically added task
            parent_task_id: Task ID that triggered the dynamic task creation
            is_iteration: True if this is an iteration task (from next_iteration)
            metadata: Optional task metadata
        """
        pass  # Default: no-op (intentional, not abstract)

    # === Distributed execution support (concrete implementation) ===

    def attach_to_trace(self, trace_id: str, parent_span_id: Optional[str] = None) -> None:
        """Attach this tracer to an existing trace (for distributed execution).

        Default implementation: sets trace context and calls output method.

        Args:
            trace_id: Existing trace ID (from TaskSpec)
            parent_span_id: Optional parent span ID
        """
        self._current_trace_id = trace_id
        # Call subclass output method
        self._output_attach_to_trace(trace_id, parent_span_id)

    @abstractmethod
    def _output_attach_to_trace(self, trace_id: str, parent_span_id: Optional[str]) -> None:
        """Output attach to trace (subclass implements output logic)."""
        pass

    @abstractmethod
    def clone(self, trace_id: str) -> Tracer:
        """Clone this tracer for branch/parallel execution.

        Creates a new tracer instance with:
        - Cloned configuration (enable_runtime_graph=False for branches)
        - Shared thread-safe resources (clients, connections)
        - Isolated mutable state (_span_stack, _root_span, etc.)
        - Parent span context inherited from current tracer state

        This prevents race conditions when multiple threads execute parallel tasks.

        Args:
            trace_id: Trace ID to attach to (shared across all branches)

        Returns:
            New tracer instance with copied state
        """
        pass

    def flush(self) -> None:
        """Flush any buffered trace data to external services.

        Default implementation: calls shutdown() for backward compatibility.

        Subclasses can override this method to:
        - Flush buffered trace data without closing connections
        - Ensure all pending spans are sent

        Note:
            This is called after ParallelGroup execution to ensure
            all branch tracer data is sent immediately.
        """
        pass  # Default: no-op (intentional, not abstract)

    def shutdown(self) -> None:
        """Flush remaining traces and cleanup resources.

        Default implementation: no-op (can be overridden by subclasses).

        Subclasses should override this method to:
        - Flush any buffered trace data
        - Close connections to external tracing services
        - Release any held resources
        """
        pass  # Default: no-op (intentional, not abstract)

    # === Public utility methods ===

    def get_runtime_graph(self) -> Optional[nx.DiGraph]:
        """Get the runtime execution graph.

        Returns:
            networkx DiGraph if enabled, None otherwise
        """
        return self._runtime_graph

    def get_execution_order(self) -> List[str]:
        """Get the execution order of tasks.

        Returns:
            List of task IDs in execution order
        """
        return self._execution_order.copy()

    def get_current_span_id(self) -> Optional[str]:
        """Return identifier for the currently active span if available."""
        return None

    def export_runtime_graph(self, format: str = "dict") -> Optional[Dict[str, Any]]:
        """Export runtime graph in specified format.

        Args:
            format: Export format ("dict", "json", "graphml")

        Returns:
            Graph data in specified format, or None if graph disabled
        """
        if self._runtime_graph is None:
            return None

        if format == "dict":
            return {
                "nodes": [{"id": node, **self._runtime_graph.nodes[node]} for node in self._runtime_graph.nodes()],
                "edges": [
                    {"source": u, "target": v, **self._runtime_graph.edges[u, v]}
                    for u, v in self._runtime_graph.edges()
                ],
                "execution_order": self._execution_order,
            }
        elif format == "json":
            import json
            from datetime import datetime

            def json_serializer(obj: Any) -> str:
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            dict_data = self.export_runtime_graph("dict")
            if dict_data is None:
                return None
            return dict(json.loads(json.dumps(dict_data, default=json_serializer)))
        elif format == "graphml":
            import io

            buffer = io.StringIO()
            nx.write_graphml(self._runtime_graph, buffer)
            return {"graphml": buffer.getvalue()}
        else:
            raise ValueError(f"Unsupported export format: {format}")

    # === Protected helper methods for graph tracking ===

    def _add_node_to_runtime_graph(
        self, node_id: str, status: str = "running", metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add node to runtime graph (internal helper).

        Args:
            node_id: Node identifier (task_id)
            status: Initial status ("running", "completed", "failed")
            metadata: Optional metadata
        """
        if self._runtime_graph is not None:
            self._runtime_graph.add_node(node_id, status=status, start_time=datetime.now(), metadata=metadata or {})
            self._execution_order.append(node_id)

    def _update_node_in_runtime_graph(
        self,
        node_id: str,
        status: Optional[str] = None,
        end_time: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update node in runtime graph (internal helper).

        Args:
            node_id: Node identifier
            status: Optional new status
            end_time: Optional end time
            metadata: Optional metadata to merge
        """
        if self._runtime_graph is not None and node_id in self._runtime_graph:
            node_data = self._runtime_graph.nodes[node_id]
            if status is not None:
                node_data["status"] = status
            if end_time is not None:
                node_data["end_time"] = end_time
            if metadata is not None:
                node_data["metadata"].update(metadata)

    def _add_edge_to_runtime_graph(self, parent_id: str, child_id: str, relation: str = "parent-child") -> None:
        """Add edge to runtime graph (internal helper).

        Args:
            parent_id: Parent node ID
            child_id: Child node ID
            relation: Edge relation type ("parent-child", "successor", etc.)
        """
        if self._runtime_graph is not None:
            if parent_id in self._runtime_graph and child_id in self._runtime_graph:
                self._runtime_graph.add_edge(parent_id, child_id, relation=relation)
