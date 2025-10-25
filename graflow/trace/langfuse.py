"""LangFuse tracer implementation for LLM workflow observability."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from graflow.trace.base import Tracer

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.task import Executable

# Optional imports
try:
    from langfuse import Langfuse
    from langfuse.client import StatefulSpanClient, StatefulTraceClient
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None  # type: ignore
    StatefulTraceClient = None  # type: ignore
    StatefulSpanClient = None  # type: ignore

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None  # type: ignore


class LangFuseTracer(Tracer):
    """LangFuse tracer for LLM workflow observability.

    Sends workflow execution traces to LangFuse platform using manual observations API.
    Requires langfuse and python-dotenv packages to be installed.

    Configuration:
        Loads credentials from environment variables (via .env file):
        - LANGFUSE_PUBLIC_KEY: LangFuse public API key
        - LANGFUSE_SECRET_KEY: LangFuse secret API key
        - LANGFUSE_HOST: LangFuse API host (optional, defaults to cloud)

    Example:
        # .env file:
        # LANGFUSE_PUBLIC_KEY=pk-xxx
        # LANGFUSE_SECRET_KEY=sk-xxx
        # LANGFUSE_HOST=https://cloud.langfuse.com

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
        enabled: bool = True
    ):
        """Initialize LangFuseTracer.

        Args:
            public_key: LangFuse public key (overrides env var)
            secret_key: LangFuse secret key (overrides env var)
            host: LangFuse host URL (overrides env var)
            enable_runtime_graph: If True, track runtime execution graph
            enabled: If False, acts as no-op (useful for testing)

        Raises:
            ImportError: If langfuse or python-dotenv is not installed
            ValueError: If credentials are not provided
        """
        super().__init__(enable_runtime_graph=enable_runtime_graph)

        if not LANGFUSE_AVAILABLE:
            raise ImportError(
                "langfuse package is required for LangFuseTracer. "
                "Install with: pip install langfuse"
            )

        if not DOTENV_AVAILABLE:
            raise ImportError(
                "python-dotenv package is required for LangFuseTracer. "
                "Install with: pip install python-dotenv"
            )

        self.enabled = enabled

        if not enabled:
            # No-op mode for testing
            self.client = None
            self._trace_client = None
            self._span_stack: List[StatefulSpanClient] = []
            return

        # Load environment variables from .env file
        load_dotenv()

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
        )

        # Track current trace and span stack
        self._trace_client: Optional[StatefulTraceClient] = None
        self._span_stack: List[StatefulSpanClient] = []

    # === Output methods (implement LangFuse output) ===

    def _output_trace_start(
        self,
        name: str,
        trace_id: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Output trace start to LangFuse."""
        if not self.enabled or not self.client:
            return

        # Create LangFuse trace
        self._trace_client = self.client.trace(
            id=trace_id,  # Use session_id as trace_id
            name=name,
            metadata=metadata or {}
        )

    def _output_trace_end(
        self,
        name: str,
        output: Optional[Any],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Output trace end to LangFuse."""
        if not self.enabled or not self._trace_client:
            return

        # Update trace with output
        self._trace_client.update(
            output=output,
            metadata=metadata or {}
        )

        # Flush to ensure data is sent
        if self.client:
            self.client.flush()

        self._trace_client = None

    def _output_span_start(
        self,
        name: str,
        parent_name: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Output span start to LangFuse."""
        if not self.enabled or not self._trace_client:
            return

        # Create span under current trace or parent span
        if self._span_stack:
            # Nested span
            parent_span = self._span_stack[-1]
            span = parent_span.span(
                name=name,
                metadata=metadata or {}
            )
        else:
            # Top-level span
            span = self._trace_client.span(
                name=name,
                metadata=metadata or {}
            )

        # Push to stack
        self._span_stack.append(span)

    def _output_span_end(
        self,
        name: str,
        output: Optional[Any],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Output span end to LangFuse."""
        if not self.enabled or not self._span_stack:
            return

        # Pop current span
        span = self._span_stack.pop()

        # Determine status from metadata
        error = metadata.get("error") if metadata else None
        level = "ERROR" if error else "DEFAULT"

        # Update span with output
        span.end(
            output=output,
            metadata=metadata or {},
            level=level
        )

    def _output_event(
        self,
        name: str,
        parent_span: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Output event to LangFuse."""
        if not self.enabled or not self._trace_client:
            return

        # Create event under current span or trace
        if self._span_stack:
            # Event under current span
            parent = self._span_stack[-1]
            parent.event(
                name=name,
                metadata=metadata or {}
            )
        else:
            # Event under trace
            self._trace_client.event(
                name=name,
                metadata=metadata or {}
            )

    def _output_attach_to_trace(
        self,
        trace_id: str,
        parent_span_id: Optional[str]
    ) -> None:
        """Output attach to trace to LangFuse."""
        if not self.enabled or not self.client:
            return

        # Get existing trace by ID
        # Note: LangFuse SDK doesn't have direct "attach to trace" API
        # We need to create a new trace client with the existing trace_id
        self._trace_client = self.client.trace(
            id=trace_id,
            name=f"worker_trace_{trace_id[:8]}"
        )

    # === Overridden hooks for LangFuse-specific behavior ===

    def on_task_queued(
        self,
        task: Executable,
        context: ExecutionContext
    ) -> None:
        """Task queued hook (override to add event output)."""
        self.event(
            "task_queued",
            metadata={
                "task_id": task.task_id,
                "task_type": type(task).__name__
            }
        )

    def on_parallel_group_start(
        self,
        group_id: str,
        member_ids: List[str],
        context: ExecutionContext
    ) -> None:
        """Parallel group start hook (override to add event output)."""
        # Log event
        self.event(
            "parallel_group_start",
            metadata={
                "group_id": group_id,
                "member_count": len(member_ids),
                "member_ids": member_ids
            }
        )

        # Call parent implementation for graph tracking
        super().on_parallel_group_start(group_id, member_ids, context)

    def on_parallel_group_end(
        self,
        group_id: str,
        member_ids: List[str],
        context: ExecutionContext,
        results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Parallel group end hook (override to add event output)."""
        success_count = len(results) if results else 0
        self.event(
            "parallel_group_end",
            metadata={
                "group_id": group_id,
                "member_count": len(member_ids),
                "success_count": success_count
            }
        )

    def shutdown(self) -> None:
        """Flush remaining traces to LangFuse.

        Should be called at the end of the program to ensure all data is sent.
        """
        if self.enabled and self.client:
            self.client.flush()
