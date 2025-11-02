"""Unit tests for ConsoleTracer."""

from graflow.trace.console import ConsoleTracer


class TestConsoleTracerBasics:
    """Test basic ConsoleTracer functionality."""

    def test_initialization(self):
        """Test ConsoleTracer initialization."""
        tracer = ConsoleTracer()
        assert tracer is not None
        assert tracer.enable_runtime_graph is True
        assert tracer.show_metadata is True
        assert tracer.enable_colors is True

    def test_initialization_show_metadata(self):
        """Test ConsoleTracer with show_metadata mode."""
        tracer = ConsoleTracer(show_metadata=True)
        assert tracer.show_metadata is True

    def test_initialization_no_colors(self):
        """Test ConsoleTracer without colors."""
        tracer = ConsoleTracer(enable_colors=False)
        assert tracer.enable_colors is False

    def test_initialization_without_runtime_graph(self):
        """Test ConsoleTracer with runtime graph disabled."""
        tracer = ConsoleTracer(enable_runtime_graph=False)
        assert tracer.enable_runtime_graph is False
        assert tracer.get_runtime_graph() is None


class TestConsoleTracerOutput:
    """Test ConsoleTracer output formatting."""

    def test_trace_start_output(self, capsys):
        """Test trace start outputs to console."""
        tracer = ConsoleTracer(enable_colors=False)
        tracer.trace_start("test_workflow")

        captured = capsys.readouterr()
        assert "TRACE START" in captured.out
        assert "test_workflow" in captured.out

    def test_trace_end_output(self, capsys):
        """Test trace end outputs to console."""
        tracer = ConsoleTracer(enable_colors=False)
        tracer.trace_start("test_workflow")
        tracer.trace_end("test_workflow")

        captured = capsys.readouterr()
        assert "TRACE END" in captured.out
        assert "test_workflow" in captured.out

    def test_span_start_output(self, capsys):
        """Test span start outputs to console."""
        tracer = ConsoleTracer(enable_colors=False)
        tracer.trace_start("workflow")
        tracer.span_start("task_1", metadata={"task_type": "Task"})

        captured = capsys.readouterr()
        assert "task_1" in captured.out

    def test_span_end_output(self, capsys):
        """Test span end outputs to console."""
        tracer = ConsoleTracer(enable_colors=False)
        tracer.trace_start("workflow")
        tracer.span_start("task_1")
        tracer.span_end("task_1")

        captured = capsys.readouterr()
        assert "task_1" in captured.out
        assert "completed" in captured.out.lower() or "✓" in captured.out

    def test_event_output(self, capsys):
        """Test event outputs to console."""
        tracer = ConsoleTracer(enable_colors=False)
        tracer.trace_start("workflow")
        tracer.event("task_queued", metadata={"task_id": "task_1"})

        captured = capsys.readouterr()
        assert "task_queued" in captured.out

    def test_indentation(self, capsys):
        """Test output indentation for nested spans."""
        tracer = ConsoleTracer(enable_colors=False)
        tracer.trace_start("workflow")
        tracer.span_start("parent")
        tracer.span_start("child", parent_name="parent")

        captured = capsys.readouterr()
        lines = captured.out.split("\n")
        # Child should have more indentation than parent
        parent_line = [l for l in lines if "parent" in l and "child" not in l]
        child_line = [l for l in lines if "child" in l]
        if parent_line and child_line:
            # Simple check: child line should have more leading spaces
            parent_indent = len(parent_line[0]) - len(parent_line[0].lstrip())
            child_indent = len(child_line[0]) - len(child_line[0].lstrip())
            assert child_indent > parent_indent

    def test_show_metadata_output(self, capsys):
        """Test show_metadata mode displays metadata."""
        tracer = ConsoleTracer(enable_colors=False, show_metadata=True)
        tracer.trace_start("workflow")
        tracer.span_start("task", metadata={"task_type": "CustomTask", "param": "value"})

        captured = capsys.readouterr()
        # When show_metadata=True, metadata should be visible
        assert "task_type" in captured.out or "CustomTask" in captured.out


class TestConsoleTracerRuntimeGraph:
    """Test ConsoleTracer runtime graph tracking."""

    def test_runtime_graph_tracks_execution(self):
        """Test runtime graph tracks execution alongside console output."""
        tracer = ConsoleTracer()
        tracer.trace_start("workflow")
        tracer.span_start("task_1")
        tracer.span_end("task_1")
        tracer.trace_end("workflow")

        graph = tracer.get_runtime_graph()
        assert graph is not None
        assert "task_1" in graph.nodes
        assert graph.nodes["task_1"]["status"] == "completed"

    def test_execution_order(self):
        """Test execution order is tracked."""
        tracer = ConsoleTracer()
        tracer.trace_start("workflow")
        tracer.span_start("task_1")
        tracer.span_end("task_1")
        tracer.span_start("task_2")
        tracer.span_end("task_2")
        tracer.trace_end("workflow")

        order = tracer.get_execution_order()
        # trace_start also records workflow in execution order
        assert order == ["workflow", "task_1", "task_2"]


class TestConsoleTracerCloning:
    """Test ConsoleTracer cloning for parallel execution."""

    def test_clone_creates_new_instance(self):
        """Test clone creates a new tracer instance."""
        tracer = ConsoleTracer(show_metadata=True)
        cloned = tracer.clone("trace_123")
        assert cloned is not tracer
        assert isinstance(cloned, ConsoleTracer)

    def test_clone_preserves_config(self):
        """Test clone preserves configuration."""
        tracer = ConsoleTracer(show_metadata=True, enable_colors=False)
        cloned = tracer.clone("trace_123")
        assert cloned.show_metadata is True
        assert cloned.enable_colors is False

    def test_clone_has_disabled_runtime_graph(self):
        """Test cloned tracer has runtime graph disabled (branch tracers don't track)."""
        tracer = ConsoleTracer(enable_colors=False)
        tracer.trace_start("workflow")
        tracer.span_start("task_1")

        cloned = tracer.clone("trace_123")

        # Original tracer should have task_1
        assert "task_1" in tracer.get_runtime_graph().nodes # type: ignore

        # Cloned tracer should have runtime graph disabled
        # (branch tracers don't track runtime graph, parent does)
        assert cloned.get_runtime_graph() is None


class TestConsoleTracerHooks:
    """Test ConsoleTracer convenience hook methods."""

    def test_on_workflow_lifecycle(self, capsys):
        """Test workflow lifecycle hooks."""
        from graflow.core.context import ExecutionContext
        from graflow.core.graph import TaskGraph

        tracer = ConsoleTracer(enable_colors=False)
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        tracer.on_workflow_start("test_workflow", context)
        tracer.on_workflow_end("test_workflow", context, result="success")

        captured = capsys.readouterr()
        assert "test_workflow" in captured.out

    def test_on_task_lifecycle(self, capsys):
        """Test task lifecycle hooks."""
        from graflow.core.context import ExecutionContext
        from graflow.core.graph import TaskGraph
        from graflow.core.task import Task

        tracer = ConsoleTracer(enable_colors=False)
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)
        task = Task(task_id="test_task", register_to_context=False)

        tracer.on_task_start(task, context)
        tracer.on_task_end(task, context, result="result")

        captured = capsys.readouterr()
        assert "test_task" in captured.out

    def test_on_task_error(self, capsys):
        """Test task error is displayed."""
        from graflow.core.context import ExecutionContext
        from graflow.core.graph import TaskGraph
        from graflow.core.task import Task

        tracer = ConsoleTracer(enable_colors=False)
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)
        task = Task(task_id="failing_task", register_to_context=False)

        tracer.on_task_start(task, context)
        tracer.on_task_end(task, context, error=Exception("Test error"))

        captured = capsys.readouterr()
        assert "failing_task" in captured.out
        assert ("error" in captured.out.lower() or "✗" in captured.out or "failed" in captured.out.lower())

    def test_on_dynamic_task_added(self, capsys):
        """Test dynamic task added event."""
        from graflow.core.context import ExecutionContext
        from graflow.core.graph import TaskGraph

        tracer = ConsoleTracer(enable_colors=False)
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        tracer.on_dynamic_task_added(
            task_id="dynamic_task",
            parent_task_id="parent",
            is_iteration=False,
            metadata={"task_type": "Task"}
        )

        captured = capsys.readouterr()
        assert "dynamic_task" in captured.out

    def test_on_parallel_group(self, capsys):
        """Test parallel group events."""
        from graflow.core.context import ExecutionContext
        from graflow.core.graph import TaskGraph

        tracer = ConsoleTracer(enable_colors=False)
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        tracer.on_parallel_group_start("pg_1", ["task_a", "task_b"], context)
        tracer.on_parallel_group_end("pg_1", ["task_a", "task_b"], context)

        captured = capsys.readouterr()
        assert "pg_1" in captured.out or "parallel" in captured.out.lower()
