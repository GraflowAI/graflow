"""Integration tests for tracing with actual workflow execution."""

from unittest.mock import MagicMock, patch

from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.core.task import TaskWrapper
from graflow.trace.console import ConsoleTracer
from graflow.trace.noop import NoopTracer


class TestSimpleWorkflowTracing:
    """Test tracing with simple sequential workflows."""

    def test_noop_tracer_with_simple_workflow(self):
        """Test NoopTracer with simple sequential workflow."""
        # Setup tracer and context
        tracer = NoopTracer()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Create simple tasks
        task1 = TaskWrapper(task_id="task_1", func=lambda: "result_1", register_to_context=False)
        task2 = TaskWrapper(task_id="task_2", func=lambda: "result_2", register_to_context=False)

        # Simulate workflow execution
        tracer.on_workflow_start("simple_workflow", context)

        tracer.on_task_start(task1, context)
        result1 = task1.func()
        tracer.on_task_end(task1, context, result=result1)

        tracer.on_task_start(task2, context)
        result2 = task2.func()
        tracer.on_task_end(task2, context, result=result2)

        tracer.on_workflow_end("simple_workflow", context, result={"task_1": result1, "task_2": result2})

        # Verify runtime graph tracking
        runtime_graph = tracer.get_runtime_graph()
        assert runtime_graph is not None
        assert "simple_workflow" in runtime_graph.nodes
        assert "task_1" in runtime_graph.nodes
        assert "task_2" in runtime_graph.nodes

        # Verify execution order
        execution_order = tracer.get_execution_order()
        assert "simple_workflow" in execution_order
        assert "task_1" in execution_order
        assert "task_2" in execution_order
        assert execution_order.index("task_1") < execution_order.index("task_2")

        # Verify all tasks completed
        assert runtime_graph.nodes["task_1"]["status"] == "completed"
        assert runtime_graph.nodes["task_2"]["status"] == "completed"

    def test_console_tracer_with_simple_workflow(self, capsys):
        """Test ConsoleTracer with simple sequential workflow."""
        # Setup tracer and context
        tracer = ConsoleTracer(enable_colors=False)
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Create simple tasks
        task1 = TaskWrapper(task_id="task_1", func=lambda: "result_1", register_to_context=False)
        task2 = TaskWrapper(task_id="task_2", func=lambda: "result_2", register_to_context=False)

        # Simulate workflow execution
        tracer.on_workflow_start("simple_workflow", context)
        tracer.on_task_start(task1, context)
        tracer.on_task_end(task1, context, result="result_1")
        tracer.on_task_start(task2, context)
        tracer.on_task_end(task2, context, result="result_2")
        tracer.on_workflow_end("simple_workflow", context)

        # Verify console output
        captured = capsys.readouterr()
        assert "simple_workflow" in captured.out
        assert "task_1" in captured.out
        assert "task_2" in captured.out

        # Verify runtime graph tracking
        runtime_graph = tracer.get_runtime_graph()
        assert runtime_graph is not None
        assert "task_1" in runtime_graph.nodes
        assert "task_2" in runtime_graph.nodes


class TestParallelGroupTracing:
    """Test tracing with ParallelGroup execution."""

    def test_noop_tracer_with_parallel_group(self):
        """Test NoopTracer with ParallelGroup execution."""
        # Setup tracer and context
        tracer = NoopTracer()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Create parallel tasks
        task_a = TaskWrapper(task_id="task_a", func=lambda: "result_a", register_to_context=False)
        task_b = TaskWrapper(task_id="task_b", func=lambda: "result_b", register_to_context=False)
        task_c = TaskWrapper(task_id="task_c", func=lambda: "result_c", register_to_context=False)

        # Simulate workflow execution with parallel group
        tracer.on_workflow_start("parallel_workflow", context)

        # Parallel group start
        group_id = "parallel_group_1"
        member_ids = ["task_a", "task_b", "task_c"]
        tracer.on_parallel_group_start(group_id, member_ids, context)

        # Execute parallel tasks
        for task in [task_a, task_b, task_c]:
            tracer.on_task_start(task, context)
            result = task.func()
            tracer.on_task_end(task, context, result=result)

        # Parallel group end
        results = {"task_a": "result_a", "task_b": "result_b", "task_c": "result_c"}
        tracer.on_parallel_group_end(group_id, member_ids, context, results=results)

        tracer.on_workflow_end("parallel_workflow", context, result=results)

        # Verify runtime graph tracking
        runtime_graph = tracer.get_runtime_graph()
        assert runtime_graph is not None
        assert "task_a" in runtime_graph.nodes
        assert "task_b" in runtime_graph.nodes
        assert "task_c" in runtime_graph.nodes

        # Verify all parallel tasks completed
        assert runtime_graph.nodes["task_a"]["status"] == "completed"
        assert runtime_graph.nodes["task_b"]["status"] == "completed"
        assert runtime_graph.nodes["task_c"]["status"] == "completed"

    def test_console_tracer_with_parallel_group(self, capsys):
        """Test ConsoleTracer with ParallelGroup execution."""
        # Setup tracer and context
        tracer = ConsoleTracer(enable_colors=False)
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Create parallel tasks
        task_a = TaskWrapper(task_id="task_a", func=lambda: "result_a", register_to_context=False)
        task_b = TaskWrapper(task_id="task_b", func=lambda: "result_b", register_to_context=False)

        # Simulate workflow execution
        tracer.on_workflow_start("parallel_workflow", context)

        group_id = "parallel_group_1"
        member_ids = ["task_a", "task_b"]
        tracer.on_parallel_group_start(group_id, member_ids, context)

        tracer.on_task_start(task_a, context)
        tracer.on_task_end(task_a, context, result="result_a")
        tracer.on_task_start(task_b, context)
        tracer.on_task_end(task_b, context, result="result_b")

        tracer.on_parallel_group_end(group_id, member_ids, context, results={"task_a": "result_a", "task_b": "result_b"})
        tracer.on_workflow_end("parallel_workflow", context)

        # Verify console output
        captured = capsys.readouterr()
        assert "parallel_workflow" in captured.out
        assert "task_a" in captured.out
        assert "task_b" in captured.out


class TestDynamicTaskTracing:
    """Test tracing with dynamic task generation."""

    def test_noop_tracer_with_dynamic_tasks(self):
        """Test NoopTracer with dynamically generated tasks."""
        # Setup tracer and context
        tracer = NoopTracer()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Simulate workflow execution
        tracer.on_workflow_start("dynamic_workflow", context)

        # Parent task starts
        parent_task = TaskWrapper(task_id="parent_task", func=lambda: "parent_result", register_to_context=False)
        tracer.on_task_start(parent_task, context)

        # Dynamically add child tasks
        for i in range(3):
            child_id = f"dynamic_child_{i}"
            tracer.on_dynamic_task_added(
                task_id=child_id,
                parent_task_id="parent_task",
                is_iteration=False,
                metadata={"task_type": "TaskWrapper", "index": i}
            )

            # Execute dynamic child
            child_task = TaskWrapper(task_id=child_id, func=lambda i=i: f"result_{i}", register_to_context=False)
            tracer.on_task_start(child_task, context)
            tracer.on_task_end(child_task, context, result=f"result_{i}")

        # Parent task ends
        tracer.on_task_end(parent_task, context, result="parent_result")
        tracer.on_workflow_end("dynamic_workflow", context)

        # Verify runtime graph tracking
        runtime_graph = tracer.get_runtime_graph()
        assert runtime_graph is not None
        assert "parent_task" in runtime_graph.nodes
        assert "dynamic_child_0" in runtime_graph.nodes
        assert "dynamic_child_1" in runtime_graph.nodes
        assert "dynamic_child_2" in runtime_graph.nodes

        # Verify all tasks completed
        assert runtime_graph.nodes["parent_task"]["status"] == "completed"
        assert runtime_graph.nodes["dynamic_child_0"]["status"] == "completed"
        assert runtime_graph.nodes["dynamic_child_1"]["status"] == "completed"
        assert runtime_graph.nodes["dynamic_child_2"]["status"] == "completed"

    def test_console_tracer_with_dynamic_tasks(self, capsys):
        """Test ConsoleTracer with dynamically generated tasks."""
        # Setup tracer and context
        tracer = ConsoleTracer(enable_colors=False)
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Simulate workflow execution
        tracer.on_workflow_start("dynamic_workflow", context)

        parent_task = TaskWrapper(task_id="parent_task", func=lambda: "parent_result", register_to_context=False)
        tracer.on_task_start(parent_task, context)

        # Dynamically add and execute child task
        tracer.on_dynamic_task_added(
            task_id="dynamic_child",
            parent_task_id="parent_task",
            is_iteration=False,
            metadata={"task_type": "TaskWrapper"}
        )

        child_task = TaskWrapper(task_id="dynamic_child", func=lambda: "child_result", register_to_context=False)
        tracer.on_task_start(child_task, context)
        tracer.on_task_end(child_task, context, result="child_result")

        tracer.on_task_end(parent_task, context, result="parent_result")
        tracer.on_workflow_end("dynamic_workflow", context)

        # Verify console output
        captured = capsys.readouterr()
        assert "dynamic_workflow" in captured.out
        assert "parent_task" in captured.out
        assert "dynamic_child" in captured.out


class TestRuntimeGraphAnalysis:
    """Test runtime graph analysis and export."""

    def test_runtime_graph_export_dict(self):
        """Test exporting runtime graph as dictionary."""
        # Setup tracer and context
        tracer = NoopTracer()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Create workflow with dependencies
        tracer.on_workflow_start("analysis_workflow", context)

        task1 = TaskWrapper(task_id="task_1", func=lambda: "result_1", register_to_context=False)
        task2 = TaskWrapper(task_id="task_2", func=lambda: "result_2", register_to_context=False)
        task3 = TaskWrapper(task_id="task_3", func=lambda: "result_3", register_to_context=False)

        # Task 1 executes
        tracer.on_task_start(task1, context)
        tracer.on_task_end(task1, context, result="result_1")

        # Task 2 and 3 execute in parallel (children of task1)
        tracer.on_task_start(task2, context)
        tracer.on_task_start(task3, context)
        tracer.on_task_end(task2, context, result="result_2")
        tracer.on_task_end(task3, context, result="result_3")

        tracer.on_workflow_end("analysis_workflow", context)

        # Export as dictionary
        export_dict = tracer.export_runtime_graph("dict")
        assert export_dict is not None
        assert "nodes" in export_dict
        assert "edges" in export_dict

        # Verify nodes
        node_ids = [node["id"] for node in export_dict["nodes"]]
        assert "analysis_workflow" in node_ids
        assert "task_1" in node_ids
        assert "task_2" in node_ids
        assert "task_3" in node_ids

        # Verify node statuses
        for node in export_dict["nodes"]:
            if node["id"] != "analysis_workflow":
                assert node["status"] == "completed"
                assert "start_time" in node

    def test_runtime_graph_execution_order(self):
        """Test execution order tracking in runtime graph."""
        # Setup tracer and context
        tracer = NoopTracer()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Create workflow
        tracer.on_workflow_start("order_workflow", context)

        tasks = []
        for i in range(5):
            task = TaskWrapper(task_id=f"task_{i}", func=lambda i=i: f"result_{i}", register_to_context=False)
            tasks.append(task)
            tracer.on_task_start(task, context)
            tracer.on_task_end(task, context, result=f"result_{i}")

        tracer.on_workflow_end("order_workflow", context)

        # Verify execution order
        execution_order = tracer.get_execution_order()
        assert "order_workflow" in execution_order

        # Verify tasks executed in order
        task_indices = [execution_order.index(f"task_{i}") for i in range(5)]
        assert task_indices == sorted(task_indices)

    def test_runtime_graph_parent_child_relationships(self):
        """Test parent-child relationship tracking in runtime graph."""
        # Setup tracer and context
        tracer = NoopTracer()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Create nested workflow
        tracer.on_workflow_start("nested_workflow", context)

        # Parent task
        parent_task = TaskWrapper(task_id="parent", func=lambda: "parent_result", register_to_context=False)
        tracer.on_task_start(parent_task, context)

        # Start child with explicit parent
        tracer.span_start("child", parent_name="parent")
        tracer.span_end("child")

        tracer.on_task_end(parent_task, context, result="parent_result")
        tracer.on_workflow_end("nested_workflow", context)

        # Verify parent-child relationship in graph
        runtime_graph = tracer.get_runtime_graph()
        assert runtime_graph is not None
        assert runtime_graph.has_edge("parent", "child")

        # Verify edge type
        edge_data = runtime_graph.edges["parent", "child"]
        assert edge_data["relation"] == "parent-child"


class TestTracerWithErrorHandling:
    """Test tracing with error handling scenarios."""

    def test_noop_tracer_with_task_error(self):
        """Test NoopTracer with task execution error."""
        # Setup tracer and context
        tracer = NoopTracer()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Simulate workflow execution
        tracer.on_workflow_start("error_workflow", context)

        # Task that fails
        failing_task = TaskWrapper(task_id="failing_task", func=lambda: 1 / 0, register_to_context=False)
        tracer.on_task_start(failing_task, context)

        # Simulate error
        error = Exception("Division by zero error")
        tracer.on_task_end(failing_task, context, result=None, error=error)

        tracer.on_workflow_end("error_workflow", context)

        # Verify error is tracked in runtime graph
        runtime_graph = tracer.get_runtime_graph()
        assert runtime_graph is not None
        assert "failing_task" in runtime_graph.nodes
        assert runtime_graph.nodes["failing_task"]["status"] == "failed"
        assert "Division by zero error" in runtime_graph.nodes["failing_task"]["metadata"]["error"]

    def test_console_tracer_with_task_error(self, capsys):
        """Test ConsoleTracer with task execution error."""
        # Setup tracer and context
        tracer = ConsoleTracer(enable_colors=False)
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Simulate workflow execution
        tracer.on_workflow_start("error_workflow", context)

        failing_task = TaskWrapper(task_id="failing_task", func=lambda: 1 / 0, register_to_context=False)
        tracer.on_task_start(failing_task, context)

        error = Exception("Test error")
        tracer.on_task_end(failing_task, context, result=None, error=error)

        tracer.on_workflow_end("error_workflow", context)

        # Verify error is displayed in console output
        captured = capsys.readouterr()
        assert "failing_task" in captured.out
        assert ("error" in captured.out.lower() or "failed" in captured.out.lower() or "✗" in captured.out)


class TestLangFuseTracerIntegration:
    """Test LangFuseTracer with actual workflow execution."""

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse", create=True)
    def test_langfuse_tracer_with_simple_workflow(self, mock_langfuse_class):
        """Test LangFuseTracer with simple workflow execution."""
        from graflow.trace.langfuse import LangFuseTracer

        # Setup mock
        mock_client = MagicMock()
        mock_root_span = MagicMock()
        mock_task_span = MagicMock()
        mock_root_span.start_span.return_value = mock_task_span
        mock_client.start_span.return_value = mock_root_span
        mock_langfuse_class.return_value = mock_client

        # Setup tracer and context
        tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Simulate workflow execution
        tracer.on_workflow_start("langfuse_workflow", context)

        task1 = TaskWrapper(task_id="task_1", func=lambda: "result_1", register_to_context=False)
        tracer.on_task_start(task1, context)
        tracer.on_task_end(task1, context, result="result_1")

        tracer.on_workflow_end("langfuse_workflow", context, result={"task_1": "result_1"})

        # Verify LangFuse client was called
        mock_client.start_span.assert_called()  # Trace start
        mock_root_span.start_span.assert_called()  # Task span
        mock_task_span.update.assert_called()  # Task end
        mock_task_span.end.assert_called()  # Task end

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse", create=True)
    def test_langfuse_tracer_flush_after_workflow(self, mock_langfuse_class):
        """Test LangFuseTracer flush after workflow execution."""
        from graflow.trace.langfuse import LangFuseTracer

        # Setup mock
        mock_client = MagicMock()
        mock_root_span = MagicMock()
        mock_client.start_span.return_value = mock_root_span
        mock_langfuse_class.return_value = mock_client

        # Setup tracer and context
        tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Simulate workflow execution
        tracer.on_workflow_start("workflow", context)
        tracer.on_workflow_end("workflow", context)

        # Flush tracer
        tracer.flush()

        # Verify flush was called (may be called multiple times - by workflow_end and explicit flush)
        assert mock_client.flush.call_count >= 1


class TestDistributedExecutionTracing:
    """Test tracing with distributed execution (TaskWorker simulation)."""

    def test_noop_tracer_with_worker_simulation(self):
        """Test NoopTracer with simulated distributed task execution."""
        # Setup tracer and context
        tracer = NoopTracer()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Simulate main process workflow
        tracer.on_workflow_start("distributed_workflow", context)

        # Task queued for worker
        task1 = TaskWrapper(task_id="worker_task_1", func=lambda: "result_1", register_to_context=False)
        tracer.on_task_queued(task1, context)

        # Simulate worker picking up task
        tracer.on_task_start(task1, context)
        result1 = task1.func()
        tracer.on_task_end(task1, context, result=result1)

        # Another worker task
        task2 = TaskWrapper(task_id="worker_task_2", func=lambda: "result_2", register_to_context=False)
        tracer.on_task_queued(task2, context)
        tracer.on_task_start(task2, context)
        result2 = task2.func()
        tracer.on_task_end(task2, context, result=result2)

        tracer.on_workflow_end("distributed_workflow", context)

        # Verify runtime graph tracking
        runtime_graph = tracer.get_runtime_graph()
        assert runtime_graph is not None
        assert "worker_task_1" in runtime_graph.nodes
        assert "worker_task_2" in runtime_graph.nodes

        # Verify tasks completed
        assert runtime_graph.nodes["worker_task_1"]["status"] == "completed"
        assert runtime_graph.nodes["worker_task_2"]["status"] == "completed"

    def test_console_tracer_with_worker_simulation(self, capsys):
        """Test ConsoleTracer with simulated distributed task execution."""
        # Setup tracer and context
        tracer = ConsoleTracer(enable_colors=False)
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Simulate distributed workflow
        tracer.on_workflow_start("distributed_workflow", context)

        task = TaskWrapper(task_id="worker_task", func=lambda: "result", register_to_context=False)
        tracer.on_task_queued(task, context)
        tracer.on_task_start(task, context)
        tracer.on_task_end(task, context, result="result")

        tracer.on_workflow_end("distributed_workflow", context)

        # Verify console output
        captured = capsys.readouterr()
        assert "distributed_workflow" in captured.out
        assert "worker_task" in captured.out

    def test_tracer_cloning_for_parallel_workers(self):
        """Test tracer cloning for parallel worker execution."""
        # Setup main tracer
        main_tracer = NoopTracer()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=main_tracer)

        # Start workflow on main tracer
        main_tracer.on_workflow_start("parallel_workers_workflow", context)

        # Clone tracer for worker 1
        worker1_tracer = main_tracer.clone("trace_worker_1")
        assert isinstance(worker1_tracer, NoopTracer)
        # Branch tracers should have runtime graph disabled
        assert worker1_tracer.get_runtime_graph() is None

        # Clone tracer for worker 2
        worker2_tracer = main_tracer.clone("trace_worker_2")
        assert isinstance(worker2_tracer, NoopTracer)
        assert worker2_tracer.get_runtime_graph() is None

        # Simulate worker 1 executing task
        task1 = TaskWrapper(task_id="worker1_task", func=lambda: "result1", register_to_context=False)
        worker1_tracer.on_task_start(task1, context)
        worker1_tracer.on_task_end(task1, context, result="result1")

        # Simulate worker 2 executing task
        task2 = TaskWrapper(task_id="worker2_task", func=lambda: "result2", register_to_context=False)
        worker2_tracer.on_task_start(task2, context)
        worker2_tracer.on_task_end(task2, context, result="result2")

        # Main tracer still tracks the workflow
        main_tracer.on_workflow_end("parallel_workers_workflow", context)

        # Verify main tracer has runtime graph
        main_graph = main_tracer.get_runtime_graph()
        assert main_graph is not None
        assert "parallel_workers_workflow" in main_graph.nodes

    @patch("graflow.trace.langfuse.LANGFUSE_AVAILABLE", True)
    @patch("graflow.trace.langfuse.Langfuse", create=True)
    def test_langfuse_tracer_with_distributed_execution(self, mock_langfuse_class):
        """Test LangFuseTracer with distributed execution simulation."""
        from graflow.trace.langfuse import LangFuseTracer

        # Setup mock
        mock_client = MagicMock()
        mock_root_span = MagicMock()
        mock_task_span = MagicMock()
        mock_root_span.start_span.return_value = mock_task_span
        mock_client.start_span.return_value = mock_root_span
        mock_langfuse_class.return_value = mock_client

        # Setup main tracer
        main_tracer = LangFuseTracer(public_key="pk", secret_key="sk", enabled=True)
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=main_tracer)

        # Start workflow
        main_tracer.on_workflow_start("distributed_langfuse_workflow", context)

        # Clone tracer for worker (simulating distributed execution)
        worker_tracer = main_tracer.clone("worker_trace_123")
        assert worker_tracer.client is main_tracer.client  # Shares same client

        # Worker executes task
        task = TaskWrapper(task_id="distributed_task", func=lambda: "result", register_to_context=False)
        worker_tracer.on_task_start(task, context)
        worker_tracer.on_task_end(task, context, result="result")

        # End workflow on main tracer
        main_tracer.on_workflow_end("distributed_langfuse_workflow", context)

        # Verify LangFuse client was called
        assert mock_client.start_span.called

    def test_task_queued_tracking(self):
        """Test tracking of queued tasks in distributed execution."""
        # Setup tracer and context
        tracer = NoopTracer()
        graph = TaskGraph()
        context = ExecutionContext(graph=graph, tracer=tracer)

        # Start workflow
        tracer.on_workflow_start("queued_tasks_workflow", context)

        # Queue multiple tasks
        tasks = []
        for i in range(3):
            task = TaskWrapper(task_id=f"queued_task_{i}", func=lambda i=i: f"result_{i}", register_to_context=False)
            tasks.append(task)
            tracer.on_task_queued(task, context)

        # Execute tasks in order
        for task in tasks:
            tracer.on_task_start(task, context)
            tracer.on_task_end(task, context, result="result")

        tracer.on_workflow_end("queued_tasks_workflow", context)

        # Verify all tasks are tracked
        runtime_graph = tracer.get_runtime_graph()
        assert runtime_graph is not None
        for i in range(3):
            task_id = f"queued_task_{i}"
            assert task_id in runtime_graph.nodes
            assert runtime_graph.nodes[task_id]["status"] == "completed"
