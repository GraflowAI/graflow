"""Unit tests for Tracer base class and runtime graph functionality."""

from datetime import datetime
from typing import Optional

import networkx as nx

from graflow.trace.base import Tracer


class TestTracerImplementation(Tracer):
    """Test implementation of Tracer for testing base class functionality."""

    def __init__(self, enable_runtime_graph: bool = True):
        super().__init__(enable_runtime_graph=enable_runtime_graph)
        self.trace_start_calls = []
        self.trace_end_calls = []
        self.span_start_calls = []
        self.span_end_calls = []
        self.event_calls = []
        self.flush_calls = []
        self.attach_calls = []

    def _output_trace_start(self, name: str, trace_id: Optional[str], metadata: Optional[dict]) -> None:
        self.trace_start_calls.append({"name": name, "trace_id": trace_id, "metadata": metadata})

    def _output_trace_end(self, name: str, output: Optional[object], metadata: Optional[dict]) -> None:
        self.trace_end_calls.append({"name": name, "output": output, "metadata": metadata})

    def _output_span_start(self, name: str, parent_name: Optional[str], metadata: Optional[dict]) -> None:
        self.span_start_calls.append({"name": name, "parent_name": parent_name, "metadata": metadata})

    def _output_span_end(self, name: str, output: Optional[object], metadata: Optional[dict]) -> None:
        self.span_end_calls.append({"name": name, "output": output, "metadata": metadata})

    def _output_event(self, name: str, parent_span: Optional[str], metadata: Optional[dict]) -> None:
        self.event_calls.append({"name": name, "parent_span": parent_span, "metadata": metadata})

    def _output_attach_to_trace(self, trace_id: str, parent_span_id: Optional[str]) -> None:
        self.attach_calls.append({"trace_id": trace_id, "parent_span_id": parent_span_id})

    def flush(self) -> None:
        self.flush_calls.append(True)

    def clone(self, trace_id: str) -> "TestTracerImplementation":
        cloned = TestTracerImplementation(enable_runtime_graph=False)
        return cloned


class TestTracerTemplateMethod:
    """Test Tracer Template Method pattern."""

    def test_trace_start_calls_output_hook(self):
        """Test trace_start calls _output_trace_start hook."""
        tracer = TestTracerImplementation()
        tracer.trace_start("test_workflow", trace_id="trace_123", metadata={"key": "value"})

        assert len(tracer.trace_start_calls) == 1
        call = tracer.trace_start_calls[0]
        assert call["name"] == "test_workflow"
        assert call["trace_id"] == "trace_123"
        assert call["metadata"] == {"key": "value"}

    def test_trace_end_calls_output_hook(self):
        """Test trace_end calls _output_trace_end hook."""
        tracer = TestTracerImplementation()
        tracer.trace_start("workflow")
        tracer.trace_end("workflow", output={"result": "success"}, metadata={"duration": 1.5})

        assert len(tracer.trace_end_calls) == 1
        call = tracer.trace_end_calls[0]
        assert call["name"] == "workflow"
        assert call["output"] == {"result": "success"}
        assert call["metadata"] == {"duration": 1.5}

    def test_span_start_calls_output_hook(self):
        """Test span_start calls _output_span_start hook."""
        tracer = TestTracerImplementation()
        tracer.trace_start("workflow")
        tracer.span_start("task_1", parent_name="parent", metadata={"task_type": "Task"})

        assert len(tracer.span_start_calls) == 1
        call = tracer.span_start_calls[0]
        assert call["name"] == "task_1"
        assert call["parent_name"] == "parent"
        assert call["metadata"] == {"task_type": "Task"}

    def test_span_end_calls_output_hook(self):
        """Test span_end calls _output_span_end hook."""
        tracer = TestTracerImplementation()
        tracer.trace_start("workflow")
        tracer.span_start("task_1")
        tracer.span_end("task_1", output=42, metadata={"status": "success"})

        assert len(tracer.span_end_calls) == 1
        call = tracer.span_end_calls[0]
        assert call["name"] == "task_1"
        assert call["output"] == 42
        assert call["metadata"] == {"status": "success"}

    def test_event_calls_output_hook(self):
        """Test event calls _output_event hook."""
        tracer = TestTracerImplementation()
        tracer.trace_start("workflow")
        tracer.event("task_queued", parent_span="task_1", metadata={"next_task": "task_2"})

        assert len(tracer.event_calls) == 1
        call = tracer.event_calls[0]
        assert call["name"] == "task_queued"
        assert call["parent_span"] == "task_1"
        assert call["metadata"] == {"next_task": "task_2"}

    def test_attach_to_trace_calls_output_hook(self):
        """Test attach_to_trace calls _output_attach_to_trace hook."""
        tracer = TestTracerImplementation()
        tracer.attach_to_trace("trace_123", parent_span_id="span_456")

        assert len(tracer.attach_calls) == 1
        call = tracer.attach_calls[0]
        assert call["trace_id"] == "trace_123"
        assert call["parent_span_id"] == "span_456"


class TestTracerRuntimeGraph:
    """Test Tracer runtime graph tracking."""

    def test_runtime_graph_enabled_by_default(self):
        """Test runtime graph is enabled by default."""
        tracer = TestTracerImplementation()
        graph = tracer.get_runtime_graph()
        assert graph is not None
        assert isinstance(graph, nx.DiGraph)

    def test_runtime_graph_can_be_disabled(self):
        """Test runtime graph can be disabled."""
        tracer = TestTracerImplementation(enable_runtime_graph=False)
        graph = tracer.get_runtime_graph()
        assert graph is None

    def test_runtime_graph_tracks_span_start(self):
        """Test runtime graph tracks span start."""
        tracer = TestTracerImplementation()
        tracer.trace_start("workflow")
        tracer.span_start("task_1", metadata={"task_type": "Task"})

        graph = tracer.get_runtime_graph()
        assert graph is not None
        assert "task_1" in graph.nodes
        node_data = graph.nodes["task_1"]
        assert node_data["status"] == "running"
        assert "start_time" in node_data
        assert isinstance(node_data["start_time"], datetime)
        # Runtime graph adds 'type': 'span' to metadata
        assert node_data["metadata"]["task_type"] == "Task"
        assert node_data["metadata"]["type"] == "span"

    def test_runtime_graph_tracks_span_end(self):
        """Test runtime graph tracks span end."""
        tracer = TestTracerImplementation()
        tracer.trace_start("workflow")
        tracer.span_start("task_1")
        tracer.span_end("task_1", output={"result": 42}, metadata={"duration": 1.5})

        graph = tracer.get_runtime_graph()
        assert graph is not None
        node_data = graph.nodes["task_1"]
        assert node_data["status"] == "completed"
        assert "end_time" in node_data
        assert isinstance(node_data["end_time"], datetime)
        # Output is stored in metadata as a string
        assert "result" in node_data["metadata"]["output"]
        assert node_data["metadata"]["duration"] == 1.5

    def test_runtime_graph_tracks_parent_child_relationship(self):
        """Test runtime graph tracks parent-child relationships."""
        tracer = TestTracerImplementation()
        tracer.trace_start("workflow")
        tracer.span_start("parent_task")
        tracer.span_start("child_task", parent_name="parent_task")

        graph = tracer.get_runtime_graph()
        assert graph is not None
        assert graph.has_edge("parent_task", "child_task")
        edge_data = graph.edges["parent_task", "child_task"]
        assert edge_data["relation"] == "parent-child"

    def test_runtime_graph_tracks_execution_order(self):
        """Test runtime graph tracks execution order."""
        tracer = TestTracerImplementation()
        tracer.trace_start("workflow")
        tracer.span_start("task_1")
        tracer.span_end("task_1")
        tracer.span_start("task_2")
        tracer.span_end("task_2")
        tracer.span_start("task_3")
        tracer.span_end("task_3")

        execution_order = tracer.get_execution_order()
        # trace_start also records workflow in execution order
        assert execution_order == ["workflow", "task_1", "task_2", "task_3"]

    def test_export_runtime_graph_dict(self):
        """Test exporting runtime graph as dict."""
        tracer = TestTracerImplementation()
        tracer.trace_start("workflow")
        tracer.span_start("task_1", metadata={"task_type": "Task"})
        tracer.span_start("task_2", parent_name="task_1")
        tracer.span_end("task_2")
        tracer.span_end("task_1")

        export = tracer.export_runtime_graph("dict")
        assert export is not None
        assert "nodes" in export
        assert "edges" in export
        # Includes workflow node
        assert len(export["nodes"]) == 3
        assert len(export["edges"]) == 1

        # Check node structure
        node_ids = [n["id"] for n in export["nodes"]]
        assert "workflow" in node_ids
        assert "task_1" in node_ids
        assert "task_2" in node_ids

        # Check edge structure
        edge = export["edges"][0]
        assert edge["source"] == "task_1"
        assert edge["target"] == "task_2"

    def test_export_runtime_graph_when_disabled(self):
        """Test exporting runtime graph when disabled returns None."""
        tracer = TestTracerImplementation(enable_runtime_graph=False)
        export = tracer.export_runtime_graph("dict")
        assert export is None


class TestTracerConvenienceHooks:
    """Test Tracer convenience hook methods."""

    def test_on_workflow_start_creates_trace(self):
        """Test on_workflow_start creates a trace."""
        from graflow.core.context import ExecutionContext
        from graflow.core.graph import TaskGraph

        tracer = TestTracerImplementation()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)
        tracer.on_workflow_start("test_workflow", context)

        # Should call trace_start
        assert len(tracer.trace_start_calls) == 1
        call = tracer.trace_start_calls[0]
        assert call["name"] == "test_workflow"
        assert call["trace_id"] == context.trace_id

    def test_on_workflow_end_ends_trace(self):
        """Test on_workflow_end ends the trace."""
        from graflow.core.context import ExecutionContext
        from graflow.core.graph import TaskGraph

        tracer = TestTracerImplementation()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)
        tracer.on_workflow_start("workflow", context)
        tracer.on_workflow_end("workflow", context, result="success")

        # Should call trace_end
        assert len(tracer.trace_end_calls) == 1
        call = tracer.trace_end_calls[0]
        assert call["name"] == "workflow"
        assert call["output"] == "success"

    def test_on_task_start_creates_span(self):
        """Test on_task_start creates a span."""
        from graflow.core.context import ExecutionContext
        from graflow.core.graph import TaskGraph
        from graflow.core.task import TaskWrapper

        tracer = TestTracerImplementation()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)
        task = TaskWrapper(task_id="test_task", func=lambda: None, register_to_context=False)

        tracer.on_task_start(task, context)

        # Should call span_start
        assert len(tracer.span_start_calls) == 1
        call = tracer.span_start_calls[0]
        assert call["name"] == "test_task"
        assert call["metadata"]["task_type"] == "TaskWrapper"

    def test_on_task_end_ends_span(self):
        """Test on_task_end ends the span."""
        from graflow.core.context import ExecutionContext
        from graflow.core.graph import TaskGraph
        from graflow.core.task import TaskWrapper

        tracer = TestTracerImplementation()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)
        task = TaskWrapper(task_id="test_task", func=lambda: "result", register_to_context=False)

        tracer.on_task_start(task, context)
        tracer.on_task_end(task, context, result="result", error=None)

        # Should call span_end
        assert len(tracer.span_end_calls) == 1
        call = tracer.span_end_calls[0]
        assert call["name"] == "test_task"
        assert call["output"] == "result"

    def test_on_task_end_with_error(self):
        """Test on_task_end with error records error in runtime graph."""
        from graflow.core.context import ExecutionContext
        from graflow.core.graph import TaskGraph
        from graflow.core.task import TaskWrapper

        tracer = TestTracerImplementation()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)
        task = TaskWrapper(task_id="failing_task", func=lambda: None, register_to_context=False)

        tracer.on_task_start(task, context)
        error = Exception("Test error")
        tracer.on_task_end(task, context, result=None, error=error)

        # Runtime graph should record error
        graph = tracer.get_runtime_graph()
        assert graph is not None
        node_data = graph.nodes["failing_task"]
        assert node_data["status"] == "failed"
        assert "Test error" in node_data["metadata"]["error"]

    def test_on_parallel_group_start(self):
        """Test on_parallel_group_start hook can be called."""
        from graflow.core.context import ExecutionContext
        from graflow.core.graph import TaskGraph

        tracer = TestTracerImplementation()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # on_parallel_group_start doesn't create spans automatically in base class
        # It's a hook for tracers to override. Just verify it doesn't crash.
        tracer.on_parallel_group_start("pg_1", ["task_a", "task_b"], context)
        # No assertion needed - just checking it doesn't raise

    def test_on_parallel_group_end(self):
        """Test on_parallel_group_end hook can be called."""
        from graflow.core.context import ExecutionContext
        from graflow.core.graph import TaskGraph

        tracer = TestTracerImplementation()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        tracer.on_parallel_group_start("pg_1", ["task_a", "task_b"], context)
        # on_parallel_group_end is a no-op in base class
        # Just verify it doesn't crash
        tracer.on_parallel_group_end("pg_1", ["task_a", "task_b"], context, results={"task_a": 1, "task_b": 2})
        # No assertion needed - just checking it doesn't raise


class TestTracerRuntimeGraphEdgeCases:
    """Test runtime graph edge cases and error handling."""

    def test_span_end_without_start(self):
        """Test span_end without corresponding start is handled gracefully."""
        tracer = TestTracerImplementation()
        tracer.trace_start("workflow")
        # End a span that was never started - should not raise
        tracer.span_end("nonexistent_task")

        graph = tracer.get_runtime_graph()
        assert graph is not None
        # Should not have created a node
        assert "nonexistent_task" not in graph.nodes

    def test_multiple_traces(self):
        """Test multiple trace sessions."""
        tracer = TestTracerImplementation()

        # First trace
        tracer.trace_start("workflow_1")
        tracer.span_start("task_1a")
        tracer.span_end("task_1a")
        tracer.trace_end("workflow_1")

        # Second trace
        tracer.trace_start("workflow_2")
        tracer.span_start("task_2a")
        tracer.span_end("task_2a")
        tracer.trace_end("workflow_2")

        # Both tasks should be in runtime graph
        graph = tracer.get_runtime_graph()
        assert graph is not None
        assert "task_1a" in graph.nodes
        assert "task_2a" in graph.nodes

        # Execution order should include both
        order = tracer.get_execution_order()
        assert "task_1a" in order
        assert "task_2a" in order

    def test_deep_nesting(self):
        """Test deeply nested spans."""
        tracer = TestTracerImplementation()
        tracer.trace_start("workflow")

        # Create deep nesting
        tracer.span_start("level_1")
        tracer.span_start("level_2", parent_name="level_1")
        tracer.span_start("level_3", parent_name="level_2")
        tracer.span_start("level_4", parent_name="level_3")

        graph = tracer.get_runtime_graph()
        assert graph is not None
        # Should have all nodes
        assert all(f"level_{i}" in graph.nodes for i in range(1, 5))
        # Should have edges forming a chain
        assert graph.has_edge("level_1", "level_2")
        assert graph.has_edge("level_2", "level_3")
        assert graph.has_edge("level_3", "level_4")
