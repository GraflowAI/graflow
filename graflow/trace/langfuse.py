"""LangFuse tracer implementation for LLM workflow observability."""

from __future__ import annotations

import copy
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from graflow.trace.base import Tracer
from graflow.utils.dotenv import load_env

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langfuse import Langfuse, LangfuseSpan
    from langfuse.types import TraceContext
    from opentelemetry import context as otel_context
    from opentelemetry import trace
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    from graflow.core.context import ExecutionContext

# Optional imports
try:
    from langfuse import Langfuse  # type: ignore[import-not-found]

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

# OpenTelemetry imports for context propagation
try:
    from opentelemetry import context as otel_context
    from opentelemetry import trace
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class LangFuseTracer(Tracer):
    """LangFuse tracer for LLM workflow observability.

    Sends workflow execution traces to LangFuse platform using manual observations API.
    Requires langfuse and python-dotenv packages to be installed.

    Configuration:
        Loads credentials from environment variables (via .env file):
        - LANGFUSE_PUBLIC_KEY: LangFuse public API key
        - LANGFUSE_SECRET_KEY: LangFuse secret API key
        - LANGFUSE_HOST: LangFuse API host (optional, defaults to https://cloud.langfuse.com)

    Important:
        - A LangFuse server must be running and accessible at the configured host
        - Connection errors are logged but don't interrupt workflow execution
        - For local development, run: docker run -p 3000:3000 langfuse/langfuse

    Example:
        # .env file:
        # LANGFUSE_PUBLIC_KEY=pk-xxx
        # LANGFUSE_SECRET_KEY=sk-xxx
        # LANGFUSE_HOST=http://localhost:3000  # For local server

        from graflow.trace import LangFuseTracer

        tracer = LangFuseTracer()
        context = ExecutionContext.create(graph, start_node, tracer=tracer)

    Note:
        Runtime graph tracking is still available if enable_runtime_graph=True.
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        enable_runtime_graph: bool = True,
        enabled: bool = True,
    ):
        """Initialize LangFuseTracer.

        Args:
            public_key: LangFuse public key (overrides env var)
            secret_key: LangFuse secret key (overrides env var)
            host: LangFuse host URL (overrides env var)
            enable_runtime_graph: If True, track runtime execution graph
            enabled: If False, acts as no-op (useful for testing)

        Raises:
            ImportError: If langfuse is not installed
            ValueError: If credentials are not provided
        """
        super().__init__(enable_runtime_graph=enable_runtime_graph)

        if not LANGFUSE_AVAILABLE:
            raise ImportError("langfuse package is required for LangFuseTracer. Install with: pip install langfuse")

        self.enabled = enabled

        # Initialize attributes with type hints
        self.client: Optional[Langfuse] = None  # type: ignore
        self._root_span: Optional[LangfuseSpan] = None  # type: ignore
        self._span_stack: List[LangfuseSpan] = []  # type: ignore
        self._otel_context_tokens: List[Any] = []  # OpenTelemetry context tokens for cleanup

        if not enabled:
            # No-op mode for testing
            return

        # Load environment variables from .env file
        load_env()

        # Get credentials (parameters override env vars)
        final_public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        final_secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        final_host = host or os.getenv("LANGFUSE_HOST")

        if not final_public_key or not final_secret_key:
            raise ValueError(
                "LangFuse credentials not found. "
                "Provide public_key and secret_key parameters, "
                "or set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables."
            )

        # Initialize LangFuse client
        self.client = Langfuse(
            public_key=final_public_key,
            secret_key=final_secret_key,
            host=final_host,
        )  # type: ignore

    # === Output methods (implement LangFuse output) ===

    def _output_trace_start(self, name: str, trace_id: Optional[str], metadata: Optional[Dict[str, Any]]) -> None:
        """Output trace start to LangFuse."""
        if not self.enabled or not self.client:
            return

        try:
            # In v3, create a root span to represent the trace
            # Use plain dict for trace_context as per v3 API
            trace_context: Optional[TraceContext] = None
            if trace_id:
                trace_context = {"trace_id": trace_id}

            # Create root span (represents the trace)
            self._root_span = self.client.start_span(trace_context=trace_context, name=name, metadata=metadata or {})
        except Exception as e:
            logger.warning(f"Failed to start LangFuse trace '{name}': {e}")
            self._root_span = None

    def _output_trace_end(self, name: str, output: Optional[Any], metadata: Optional[Dict[str, Any]]) -> None:
        """Output trace end to LangFuse."""
        if not self.enabled or not self._root_span:
            return

        try:
            # Update root span with output and metadata, then end it
            self._root_span.update(output=output, metadata=metadata or {})
            self._root_span.end()

            # Flush to ensure data is sent
            if self.client:
                self.client.flush()
        except Exception as e:
            logger.warning(f"Failed to end LangFuse trace '{name}': {e}")
        finally:
            self._root_span = None

    def _output_span_start(self, name: str, parent_name: Optional[str], metadata: Optional[Dict[str, Any]]) -> None:
        """Output span start to LangFuse and set OpenTelemetry context.

        This method creates a Langfuse span and sets OpenTelemetry context
        with the span's trace_id and span_id. This allows LiteLLM and Google ADK
        to automatically detect the parent span and create nested traces.
        """
        if not self.enabled or not self._root_span:
            return

        try:
            # Create span under current span or root span
            if self._span_stack:
                # Nested span - create child from current span
                parent_span = self._span_stack[-1]
                span = parent_span.start_span(name=name, metadata=metadata or {})
            else:
                # Top-level span under root (or first span in worker process)
                # If parent_span_id is set (from attach_to_trace), we need to
                # connect this span to the parent task in distributed tracing
                # Note: In v3, this is handled via trace_context in attach_to_trace
                span = self._root_span.start_span(name=name, metadata=metadata or {})

            # Push to stack
            self._span_stack.append(span)

            # Set OpenTelemetry context for LiteLLM/ADK automatic propagation
            if OTEL_AVAILABLE and hasattr(span, "trace_id") and hasattr(span, "id"):
                try:
                    # Convert Langfuse hex IDs to OpenTelemetry int IDs
                    # Langfuse v3: trace_id is 32-char hex, span.id is 16-char hex
                    trace_id_int = int(span.trace_id, 16)
                    span_id_int = int(span.id, 16)

                    # Create OpenTelemetry SpanContext
                    span_context = SpanContext(
                        trace_id=trace_id_int,
                        span_id=span_id_int,
                        is_remote=False,
                        trace_flags=TraceFlags(0x01),  # Sampled
                    )

                    # Set as current context (for LiteLLM/ADK to detect)
                    ctx = trace.set_span_in_context(NonRecordingSpan(span_context))
                    token = otel_context.attach(ctx)
                    self._otel_context_tokens.append(token)

                    logger.debug(
                        f"Set OpenTelemetry context for span '{name}': "
                        f"trace_id={span.trace_id[:8]}..., span_id={span.id[:8]}..."
                    )
                except Exception as e:
                    logger.warning(f"Failed to set OpenTelemetry context for span '{name}': {e}")

        except Exception as e:
            logger.warning(f"Failed to start LangFuse span '{name}': {e}")

    def _output_span_end(self, name: str, output: Optional[Any], metadata: Optional[Dict[str, Any]]) -> None:
        """Output span end to LangFuse and clear OpenTelemetry context."""
        if not self.enabled or not self._span_stack:
            return

        # Pop current span
        span = self._span_stack.pop()

        try:
            # Determine status from metadata
            error = metadata.get("error") if metadata else None

            # Update span with output and metadata
            span.update(
                output=output,
                metadata=metadata or {},
                level="ERROR" if error else None,
                status_message=str(error) if error else None,
            )

            # End the span
            span.end()

            # Clear OpenTelemetry context
            if OTEL_AVAILABLE and self._otel_context_tokens:
                try:
                    token = self._otel_context_tokens.pop()
                    otel_context.detach(token)
                    logger.debug(f"Cleared OpenTelemetry context for span '{name}'")
                except Exception as e:
                    logger.warning(f"Failed to clear OpenTelemetry context for span '{name}': {e}")

        except Exception as e:
            logger.warning(f"Failed to end LangFuse span '{name}': {e}")

    def _output_event(self, name: str, parent_span: Optional[str], metadata: Optional[Dict[str, Any]]) -> None:
        """Output event to LangFuse."""
        if not self.enabled or not self._root_span:
            return

        try:
            # Create event under current span or root span
            if self._span_stack:
                # Event under current span
                parent = self._span_stack[-1]
                parent.create_event(name=name, metadata=metadata or {})
            else:
                # Event under root span
                self._root_span.create_event(name=name, metadata=metadata or {})
        except Exception as e:
            logger.warning(f"Failed to create LangFuse event '{name}': {e}")

    def _output_attach_to_trace(self, trace_id: str, parent_span_id: Optional[str]) -> None:
        """Output attach to trace to LangFuse.

        Args:
            trace_id: Trace ID to attach to (session_id from parent)
            parent_span_id: Parent span ID for distributed tracing.
                           The first span created will be a child of this span.
        """
        if not self.enabled or not self.client:
            return

        try:
            # In v3, use trace_context dict to attach to existing trace
            # Create root span attached to the existing trace with parent span if provided
            trace_context: TraceContext = {"trace_id": trace_id}
            if parent_span_id:
                trace_context["parent_span_id"] = parent_span_id

            self._root_span = self.client.start_span(trace_context=trace_context, name=f"worker_trace_{trace_id[:8]}")
        except Exception as e:
            logger.warning(f"Failed to attach to LangFuse trace '{trace_id}': {e}")
            self._root_span = None

    # === Overridden hooks for LangFuse-specific behavior ===

    def on_dynamic_task_added(
        self,
        task_id: str,
        parent_task_id: Optional[str] = None,
        is_iteration: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Dynamic task added hook (override to add event output)."""
        event_metadata = {
            **(metadata or {}),
            "task_id": task_id,
            "parent_task_id": parent_task_id,
            "is_iteration": is_iteration,
        }

        event_name = "task_iteration_added" if is_iteration else "dynamic_task_added"
        self.event(event_name, metadata=event_metadata)

    def on_parallel_group_start(self, group_id: str, member_ids: List[str], context: ExecutionContext) -> None:
        """Parallel group start hook (override to add event output)."""
        # Log event
        self.event(
            "parallel_group_start",
            metadata={"group_id": group_id, "member_count": len(member_ids), "member_ids": member_ids},
        )

        # Call parent implementation for graph tracking
        super().on_parallel_group_start(group_id, member_ids, context)

    def on_parallel_group_end(
        self, group_id: str, member_ids: List[str], context: ExecutionContext, results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Parallel group end hook (override to add event output)."""
        success_count = len(results) if results else 0
        self.event(
            "parallel_group_end",
            metadata={"group_id": group_id, "member_count": len(member_ids), "success_count": success_count},
        )

    def clone(self, trace_id: str) -> LangFuseTracer:
        """Clone this tracer for branch/parallel execution.

        Creates an isolated tracer with its own span stack to avoid race conditions
        in parallel execution, while sharing the Langfuse client.

        Args:
            trace_id: Trace ID to attach to (shared across all branches)

        Returns:
            New LangFuseTracer instance with copied state
        """
        if not self.enabled or not self.client:
            # Return a disabled tracer if parent is disabled
            return LangFuseTracer(enabled=False)

        # Create a new tracer instance with its own state
        # We pass enabled=False initially to skip client initialization
        branch_tracer = LangFuseTracer(enabled=False)

        # Share the client (thread-safe in v3) but create new state
        branch_tracer.enabled = True
        branch_tracer.client = self.client  # Share the Langfuse client
        branch_tracer._span_stack = []  # New stack for this branch
        branch_tracer._otel_context_tokens = []  # New OTel context tokens for this branch
        branch_tracer._current_trace_id = trace_id

        # Set root span to parent's current span (from span stack)
        # This ensures branch spans are nested under the parent span (e.g., ParallelGroup)
        # Use shallow copy to avoid sharing the same span reference
        if self._span_stack:
            branch_tracer._root_span = copy.copy(self._span_stack[-1])
        else:
            # Fallback to parent's root span if no current span
            branch_tracer._root_span = copy.copy(self._root_span) if self._root_span else None

        return branch_tracer

    def get_current_span_id(self) -> Optional[str]:
        """Return the active Langfuse span id, if any."""
        if not self.enabled:
            return None

        if self._span_stack:
            current = self._span_stack[-1]
            return getattr(current, "id", None)

        if self._root_span:
            return getattr(self._root_span, "id", None)

        return None

    def flush(self) -> None:
        """Flush traces to LangFuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.warning(f"Failed to flush LangFuse traces: {e}")

    def shutdown(self) -> None:
        """
        Shutdown the tracer and cleanup resources.
        """
        self.flush()
        self._root_span = None
