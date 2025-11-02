"""Unit tests for NoopTracer."""

from graflow.trace.noop import NoopTracer


class TestNoopTracerBasics:
    """Test basic NoopTracer functionality."""

    def test_initialization(self):
        """Test NoopTracer initialization."""
        tracer = NoopTracer()
        assert tracer is not None
        assert tracer.enable_runtime_graph is True

    def test_initialization_without_runtime_graph(self):
        """Test NoopTracer initialization with runtime graph disabled."""
        tracer = NoopTracer(enable_runtime_graph=False)
        assert tracer.enable_runtime_graph is False
        assert tracer.get_runtime_graph() is None

    def test_trace_lifecycle(self):
        """Test trace start/end does not raise errors."""
        tracer = NoopTracer()
        # Should not raise any errors
        tracer.trace_start("test_trace")
        tracer.trace_end("test_trace")

    def test_span_lifecycle(self):
        """Test span start/end does not raise errors."""
        tracer = NoopTracer()
        tracer.trace_start("test_trace")
        tracer.span_start("test_span")
        tracer.span_end("test_span")
        tracer.trace_end("test_trace")

    def test_event_recording(self):
        """Test event recording does not raise errors."""
        tracer = NoopTracer()
        tracer.trace_start("test_trace")
        tracer.event("test_event")
        tracer.trace_end("test_trace")

    def test_flush(self):
        """Test flush does not raise errors."""
        tracer = NoopTracer()
        tracer.flush()


class TestNoopTracerRuntimeGraph:
    """Test NoopTracer runtime graph tracking."""

    def test_runtime_graph_enabled_by_default(self):
        """Test runtime graph is enabled by default."""
        tracer = NoopTracer()
        graph = tracer.get_runtime_graph()
        assert graph is not None
        assert len(graph.nodes) == 0

    def test_runtime_graph_tracks_spans(self):
        """Test runtime graph tracks span start/end."""
        tracer = NoopTracer()
        tracer.trace_start("workflow")
        tracer.span_start("task_1")
        tracer.span_end("task_1")
        tracer.trace_end("workflow")

        graph = tracer.get_runtime_graph()
        assert graph is not None
        assert "task_1" in graph.nodes
        assert graph.nodes["task_1"]["status"] == "completed"

    def test_runtime_graph_tracks_execution_order(self):
        """Test runtime graph tracks execution order."""
        tracer = NoopTracer()
        tracer.trace_start("workflow")
        tracer.span_start("task_1")
        tracer.span_end("task_1")
        tracer.span_start("task_2")
        tracer.span_end("task_2")
        tracer.trace_end("workflow")

        execution_order = tracer.get_execution_order()
        # trace_start also records workflow in execution order
        assert execution_order == ["workflow", "task_1", "task_2"]

    def test_runtime_graph_tracks_parent_child(self):
        """Test runtime graph tracks parent-child relationships."""
        tracer = NoopTracer()
        tracer.trace_start("workflow")
        tracer.span_start("parent_task")
        tracer.span_start("child_task", parent_name="parent_task")
        tracer.span_end("child_task")
        tracer.span_end("parent_task")
        tracer.trace_end("workflow")

        graph = tracer.get_runtime_graph()
        assert graph is not None
        assert "parent_task" in graph.nodes
        assert "child_task" in graph.nodes
        # Check parent-child edge
        assert graph.has_edge("parent_task", "child_task")

    def test_export_runtime_graph_dict(self):
        """Test exporting runtime graph as dict."""
        tracer = NoopTracer()
        tracer.trace_start("workflow")
        tracer.span_start("task_1")
        tracer.span_end("task_1")
        tracer.trace_end("workflow")

        export = tracer.export_runtime_graph("dict")
        assert export is not None
        assert "nodes" in export
        assert "edges" in export
        # Both workflow and task_1 are recorded
        assert len(export["nodes"]) == 2
        node_ids = [n["id"] for n in export["nodes"]]
        assert "workflow" in node_ids
        assert "task_1" in node_ids


class TestNoopTracerCloning:
    """Test NoopTracer cloning for parallel execution."""

    def test_clone_creates_new_instance(self):
        """Test clone creates a new tracer instance."""
        tracer = NoopTracer()
        cloned = tracer.clone("trace_123")
        assert cloned is not tracer
        assert isinstance(cloned, NoopTracer)

    def test_clone_has_independent_runtime_graph(self):
        """Test cloned tracer has disabled runtime graph (branch tracers don't track)."""
        tracer = NoopTracer()
        tracer.trace_start("workflow")
        tracer.span_start("task_1")

        cloned = tracer.clone("trace_123")

        # Original tracer should have task_1
        original_graph = tracer.get_runtime_graph()
        assert original_graph is not None
        assert "task_1" in original_graph.nodes

        # Cloned tracer should have runtime graph disabled
        # (branch tracers don't track runtime graph, parent does)
        cloned_graph = cloned.get_runtime_graph()
        assert cloned_graph is None

    def test_clone_with_runtime_graph_disabled(self):
        """Test cloning when runtime graph is disabled."""
        tracer = NoopTracer(enable_runtime_graph=False)
        cloned = tracer.clone("trace_123")
        assert cloned.get_runtime_graph() is None


class TestNoopTracerHooks:
    """Test NoopTracer convenience hook methods."""

    def test_on_workflow_start_end(self):
        """Test workflow start/end hooks."""
        from graflow.core.context import ExecutionContext
        from graflow.core.graph import TaskGraph

        tracer = NoopTracer()
        # Create minimal graph and context
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Should not raise errors
        tracer.on_workflow_start("test_workflow", context)
        tracer.on_workflow_end("test_workflow", context)

        # Verify workflow was tracked in runtime graph
        runtime_graph = tracer.get_runtime_graph()
        assert runtime_graph is not None
        assert "test_workflow" in runtime_graph.nodes

    def test_on_task_start_end(self):
        """Test task start/end hooks."""
        from graflow.core.context import ExecutionContext
        from graflow.core.graph import TaskGraph
        from graflow.core.task import Task

        tracer = NoopTracer()
        # Create minimal graph and context
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)
        # Create task with task_id only (Task doesn't take callable directly)
        task = Task(task_id="test_task", register_to_context=False)

        # Should not raise errors
        tracer.on_task_start(task, context)
        tracer.on_task_end(task, context)

        # Verify runtime graph tracked the task
        runtime_graph = tracer.get_runtime_graph()
        assert runtime_graph is not None
        assert "test_task" in runtime_graph.nodes
