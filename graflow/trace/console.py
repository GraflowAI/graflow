"""Console tracer implementation with formatted output."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from graflow.trace.base import Tracer

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext


class ConsoleTracer(Tracer):
    """Console tracer that outputs formatted execution traces to stdout.

    Provides human-readable output with:
    - Hierarchical indentation for nested spans
    - Timestamps for each event
    - Status indicators (✓, ✗, →, ⚡)
    - Duration tracking
    - Optional color support (via ANSI codes)

    Example output:
        → [12:34:56.123] TRACE START: my_workflow (trace_id: abc123)
          → [12:34:56.124] SPAN START: task_1 (type: computation)
          ✓ [12:34:56.234] SPAN END: task_1 (110ms)
          → [12:34:56.235] SPAN START: task_2 (type: computation)
          ✓ [12:34:56.345] SPAN END: task_2 (110ms)
        ✓ [12:34:56.346] TRACE END: my_workflow (223ms)

    Note:
        Runtime graph tracking and base hook implementations are handled
        automatically by the base Tracer class.
        ConsoleTracer implements console output methods and overrides
        parallel group hooks for event logging.
    """

    def __init__(
        self,
        enable_runtime_graph: bool = True,
        enable_colors: bool = True,
        show_metadata: bool = True,
        indent_size: int = 2,
    ):
        """Initialize ConsoleTracer.

        Args:
            enable_runtime_graph: If True, track runtime execution graph
            enable_colors: If True, use ANSI color codes for output
            show_metadata: If True, display metadata in output
            indent_size: Number of spaces per indentation level
        """
        super().__init__(enable_runtime_graph=enable_runtime_graph)
        self.enable_colors = enable_colors
        self.show_metadata = show_metadata
        self.indent_size = indent_size
        self._indent_level = 0
        self._span_start_times: Dict[str, datetime] = {}

    # === Color codes ===

    def _colorize(self, text: str, color: str) -> str:
        """Apply ANSI color to text if colors enabled.

        Args:
            text: Text to colorize
            color: Color name (gray, green, red, yellow, blue, cyan)

        Returns:
            Colorized text or plain text if colors disabled
        """
        if not self.enable_colors:
            return text

        colors = {
            "gray": "\033[90m",
            "green": "\033[92m",
            "red": "\033[91m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "cyan": "\033[96m",
            "reset": "\033[0m",
        }

        color_code = colors.get(color, "")
        reset_code = colors["reset"]
        return f"{color_code}{text}{reset_code}"

    # === Formatting helpers ===

    def _format_timestamp(self) -> str:
        """Format current timestamp for display."""
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm

    def _format_duration(self, start_time: datetime, end_time: datetime) -> str:
        """Format duration between timestamps."""
        duration = (end_time - start_time).total_seconds() * 1000  # ms
        if duration < 1000:
            return f"{duration:.0f}ms"
        else:
            return f"{duration / 1000:.2f}s"

    def _format_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        """Format metadata dict for display."""
        if not metadata or not self.show_metadata:
            return ""

        items = []
        for key, value in metadata.items():
            if key in ["trace_id", "task_type", "handler_type", "type"]:
                items.append(f"{key}: {value}")

        return f" ({', '.join(items)})" if items else ""

    def _print(self, message: str, icon: str = "→", color: str = "gray") -> None:
        """Print formatted message with indentation.

        Args:
            message: Message to print
            icon: Icon/prefix (→, ✓, ✗, ⚡)
            color: Text color
        """
        indent = " " * (self._indent_level * self.indent_size)
        timestamp = self._colorize(f"[{self._format_timestamp()}]", "gray")
        icon_colored = self._colorize(icon, color)
        print(f"{indent}{icon_colored} {timestamp} {message}")

    # === Output methods (implement console output) ===

    def _output_trace_start(self, name: str, trace_id: Optional[str], metadata: Optional[Dict[str, Any]]) -> None:
        """Output trace start to console."""
        self._span_start_times[name] = datetime.now()
        meta_str = self._format_metadata({"trace_id": trace_id, **(metadata or {})})
        message = self._colorize(f"TRACE START: {name}{meta_str}", "cyan")
        self._print(message, icon="→", color="cyan")
        self._indent_level += 1

    def _output_trace_end(self, name: str, output: Optional[Any], metadata: Optional[Dict[str, Any]]) -> None:
        """Output trace end to console."""
        self._indent_level -= 1
        duration_str = ""
        if name in self._span_start_times:
            duration = self._format_duration(self._span_start_times[name], datetime.now())
            duration_str = f" ({duration})"
            del self._span_start_times[name]

        message = self._colorize(f"TRACE END: {name}{duration_str}", "cyan")
        self._print(message, icon="✓", color="green")

    def _output_span_start(self, name: str, parent_name: Optional[str], metadata: Optional[Dict[str, Any]]) -> None:
        """Output span start to console."""
        self._span_start_times[name] = datetime.now()
        meta_str = self._format_metadata(metadata)
        message = self._colorize(f"SPAN START: {name}{meta_str}", "blue")
        self._print(message, icon="→", color="blue")
        self._indent_level += 1

    def _output_span_end(self, name: str, output: Optional[Any], metadata: Optional[Dict[str, Any]]) -> None:
        """Output span end to console."""
        self._indent_level -= 1
        error = metadata.get("error") if metadata else None
        duration_str = ""
        if name in self._span_start_times:
            duration = self._format_duration(self._span_start_times[name], datetime.now())
            duration_str = f" ({duration})"
            del self._span_start_times[name]

        if error:
            message = self._colorize(f"SPAN END: {name} - ERROR: {error}{duration_str}", "red")
            self._print(message, icon="✗", color="red")
        else:
            message = self._colorize(f"SPAN END: {name}{duration_str}", "blue")
            self._print(message, icon="✓", color="green")

    def _output_event(self, name: str, parent_span: Optional[str], metadata: Optional[Dict[str, Any]]) -> None:
        """Output event to console."""
        meta_str = self._format_metadata(metadata)
        message = self._colorize(f"EVENT: {name}{meta_str}", "yellow")
        self._print(message, icon="⚡", color="yellow")

    def _output_attach_to_trace(self, trace_id: str, parent_span_id: Optional[str]) -> None:
        """Output attach to trace to console."""
        message = self._colorize(f"ATTACHED TO TRACE: {trace_id[:16]}... (parent: {parent_span_id or 'none'})", "cyan")
        self._print(message, icon="🔗", color="cyan")

    def clone(self, trace_id: str) -> ConsoleTracer:
        """Clone this tracer for branch/parallel execution.

        Creates a new ConsoleTracer instance with:
        - Cloned configuration (colors, metadata, indent settings)
        - Isolated mutable state (_indent_level, _span_start_times)
        - Disabled runtime_graph tracking (parent tracks it)

        Args:
            trace_id: Trace ID to attach to (shared across all branches)

        Returns:
            New ConsoleTracer instance for branch execution
        """
        # Create new instance with same configuration but no runtime graph
        branch_tracer = ConsoleTracer(
            enable_runtime_graph=False,  # Branch doesn't track runtime graph
            enable_colors=self.enable_colors,
            show_metadata=self.show_metadata,
            indent_size=self.indent_size,
        )

        # Inherit indentation so branch spans appear nested under their parent.
        branch_tracer._indent_level = max(self._indent_level, 0)

        # Set trace context
        branch_tracer._current_trace_id = trace_id

        return branch_tracer

    # === Overridden hooks for event logging ===

    def on_parallel_group_start(self, group_id: str, member_ids: List[str], context: ExecutionContext) -> None:
        """Parallel group start hook (override to add event output)."""
        # Log event
        self.event(f"parallel_group_start: {group_id}", metadata={"members": len(member_ids)})

        # Call parent implementation for graph tracking
        super().on_parallel_group_start(group_id, member_ids, context)

    def on_parallel_group_end(
        self, group_id: str, member_ids: List[str], context: ExecutionContext, results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Parallel group end hook (override to add event output)."""
        success_count = len(results) if results else 0
        self.event(f"parallel_group_end: {group_id}", metadata={"completed": success_count, "total": len(member_ids)})

    def on_dynamic_task_added(
        self,
        task_id: str,
        parent_task_id: Optional[str] = None,
        is_iteration: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Override to print dynamic task addition event."""
        task_type = "iteration" if is_iteration else "dynamic"
        parent_info = f" (parent: {parent_task_id})" if parent_task_id else ""

        message = self._colorize(f"+ {task_type.upper()} TASK: {task_id}{parent_info}", "yellow")
        self._print(message, icon="⚡", color="yellow")

        if self.show_metadata and metadata:
            self._print("", f"  {self._format_metadata(metadata)}")
