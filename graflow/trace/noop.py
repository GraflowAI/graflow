"""No-op tracer implementation that only tracks runtime graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from graflow.trace.base import Tracer

if TYPE_CHECKING:
    pass


class NoopTracer(Tracer):
    """No-op tracer that tracks runtime graph but produces no output.

    This is the default tracer for ExecutionContext. It maintains
    the runtime execution graph and execution order for debugging
    and visualization purposes (inherited from base Tracer class),
    but does not produce any console output or send data to external services.

    Use Cases:
    - Default tracer when no explicit tracing is needed
    - Lightweight runtime graph tracking for visualization
    - Base for testing trace integration

    Note:
        Runtime graph tracking and hook implementations are handled
        automatically by the base Tracer class.
        NoopTracer only implements no-op output methods.
    """

    def __init__(self, enable_runtime_graph: bool = True):
        """Initialize NoopTracer.

        Args:
            enable_runtime_graph: If True, track runtime execution graph
        """
        super().__init__(enable_runtime_graph=enable_runtime_graph)

    # === Output methods (all no-ops) ===

    def _output_trace_start(self, name: str, trace_id: Optional[str], metadata: Optional[Dict[str, Any]]) -> None:
        """Trace start output (no-op)."""
        pass

    def _output_trace_end(self, name: str, output: Optional[Any], metadata: Optional[Dict[str, Any]]) -> None:
        """Trace end output (no-op)."""
        pass

    def _output_span_start(self, name: str, parent_name: Optional[str], metadata: Optional[Dict[str, Any]]) -> None:
        """Span start output (no-op)."""
        pass

    def _output_span_end(self, name: str, output: Optional[Any], metadata: Optional[Dict[str, Any]]) -> None:
        """Span end output (no-op)."""
        pass

    def _output_event(self, name: str, parent_span: Optional[str], metadata: Optional[Dict[str, Any]]) -> None:
        """Event output (no-op)."""
        pass

    def _output_attach_to_trace(self, trace_id: str, parent_span_id: Optional[str]) -> None:
        """Attach to trace output (no-op)."""
        pass

    def clone(self, trace_id: str) -> NoopTracer:
        """Clone this tracer for branch/parallel execution.

        Creates a new NoopTracer instance with disabled runtime_graph tracking
        (parent tracer tracks it).

        Args:
            trace_id: Trace ID to attach to (shared across all branches)

        Returns:
            New NoopTracer instance for branch execution
        """
        # Create new instance without runtime graph tracking
        branch_tracer = NoopTracer(enable_runtime_graph=False)

        # Set trace context
        branch_tracer._current_trace_id = trace_id

        return branch_tracer
