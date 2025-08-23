"""
Dynamic task generation scenario tests.

Tests various dynamic task creation scenarios including next_task(),
next_iteration(), conditional creation, and task jumping.
"""

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph
from graflow.core.task import TaskWrapper


def test_next_task_functionality():
    """Test basic next_task() functionality."""
    # Create execution context with a dummy task
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "start", max_steps=10)

    # Create a dummy task context to simulate being in a task
    dummy_task_ctx = context.create_task_context("start")
    context.push_task_context(dummy_task_ctx)

    # Create and add a dynamic task
    def dynamic_work():
        return {"result": "dynamic_complete"}

    dynamic_task = TaskWrapper("dynamic_processor", dynamic_work)
    task_id = context.next_task(dynamic_task)

    # Verify task creation
    assert task_id == "dynamic_processor"
    assert task_id in context.graph.nodes
    assert task_id in context.queue.to_list()


def test_next_iteration_functionality():
    """Test basic next_iteration() functionality."""
    # Create execution context with a base task
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "start", max_steps=10)

    # Add a base task to the graph first
    def base_task(ctx, data):
        count = data.get("count", 0)
        return {"processed": True, "count": count}

    base_wrapper = TaskWrapper("counter", base_task)
    graph.add_node(base_wrapper, "counter")

    # Set current task context and create iteration
    counter_task_ctx = context.create_task_context("counter")
    context.push_task_context(counter_task_ctx)

    iteration_id = context.next_iteration({"count": 1, "message": "hello"})

    # Verify iteration task creation
    assert iteration_id.startswith("counter_cycle_")
    assert iteration_id in context.graph.nodes
    assert iteration_id in context.queue.to_list()


def test_conditional_task_creation():
    """Test conditional task creation based on values."""
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "start", max_steps=10)

    # Create a processor task context
    processor_task_ctx = context.create_task_context("processor")
    context.push_task_context(processor_task_ctx)

    created_tasks = []

    def process_data(value):
        if value > 50:
            # High value processing
            high_task = TaskWrapper(
                f"high_value_handler_{value}",
                lambda: f"high_value_processing_{value}"
            )
            task_id = context.next_task(high_task)
            created_tasks.append(task_id)

        elif value > 10:
            # Medium value processing
            med_task = TaskWrapper(
                f"medium_value_handler_{value}",
                lambda: f"medium_value_processing_{value}"
            )
            task_id = context.next_task(med_task)
            created_tasks.append(task_id)

        return {"processed": value}

    # Test with different values
    process_data(75)  # Should create high value task
    process_data(25)  # Should create medium value task
    process_data(5)   # Should not create any task

    # Verify correct tasks were created
    assert len(created_tasks) == 2
    assert "high_value_handler_75" in created_tasks
    assert "medium_value_handler_25" in created_tasks

    # Verify tasks are in graph and queue
    for task_id in created_tasks:
        assert task_id in context.graph.nodes
        assert task_id in context.queue.to_list()


def test_engine_execution_with_dynamic_tasks():
    """Test engine execution with dynamic task creation."""
    # Create graph with connected tasks
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "start_task", max_steps=20)

    @task("start_task", inject_context=True)
    def start_task_func(task_ctx):
        # Create a dynamic task that should execute before successors
        @task("dynamic_urgent")
        def dynamic_work():
            return "dynamic_complete"

        task_ctx.next_task(dynamic_work)
        return "start_complete"

    @task("successor_task", inject_context=True)
    def successor_task_func(task_ctx):
        return "successor_complete"

    @task("final_task", inject_context=True)
    def final_task_func(task_ctx):
        return "final_complete"

    # Build graph with dependencies
    graph.add_node(start_task_func, "start_task")
    graph.add_node(successor_task_func, "successor_task")
    graph.add_node(final_task_func, "final_task")

    # Add edges: start -> successor -> final
    graph.add_edge("start_task", "successor_task")
    graph.add_edge("successor_task", "final_task")

    # Execute the workflow
    context.execute()

    # Verify all tasks executed
    assert "start_task" in context.executed
    assert "dynamic_urgent" in context.executed
    assert "successor_task" in context.executed
    assert "final_task" in context.executed

    # Verify execution order - dynamic task should execute before successors
    executed_order = list(context.executed)
    start_idx = executed_order.index("start_task")
    dynamic_idx = executed_order.index("dynamic_urgent")
    successor_idx = executed_order.index("successor_task")

    assert dynamic_idx > start_idx
    assert successor_idx > dynamic_idx


def test_multiple_dynamic_tasks():
    """Test multiple dynamic tasks created in sequence."""
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "coordinator", max_steps=15)

    @task("coordinator", inject_context=True)
    def coordinator_func(task_ctx):
        # Create multiple dynamic tasks
        created_workers = []
        for i in range(3):
            def make_worker_func(worker_id):
                @task(f"worker_{worker_id}")
                def worker_func():
                    return f"worker_{worker_id}_done"
                return worker_func

            worker_task = make_worker_func(i)
            worker_id = task_ctx.next_task(worker_task)
            created_workers.append(worker_id)

        return {"workers_created": created_workers}

    @task("cleanup", inject_context=True)
    def cleanup_func(task_ctx):
        return "cleanup_complete"

    # Build graph
    graph.add_node(coordinator_func, "coordinator")
    graph.add_node(cleanup_func, "cleanup")
    graph.add_edge("coordinator", "cleanup")

    # Execute workflow
    context.execute()

    # Verify all tasks executed
    assert "coordinator" in context.executed
    assert "worker_0" in context.executed
    assert "worker_1" in context.executed
    assert "worker_2" in context.executed
    assert "cleanup" in context.executed

    # Verify workers executed before cleanup
    executed_order = list(context.executed)
    cleanup_idx = executed_order.index("cleanup")

    for i in range(3):
        worker_idx = executed_order.index(f"worker_{i}")
        assert worker_idx < cleanup_idx


def test_dynamic_tasks_with_processing():
    """Test dynamic tasks combined with processing."""
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "processor", max_steps=20)

    @task("processor", inject_context=True)
    def processor_func(task_ctx):
        # Create multiple dynamic analysis tasks
        created_analysis = []
        for i in range(3):
            def analysis_func_factory(current_count):
                @task(f"analysis_{current_count}")
                def analysis_func():
                    return f"analysis_{current_count}_complete"
                return analysis_func

            analysis_task = analysis_func_factory(i)
            analysis_id = task_ctx.next_task(analysis_task)
            created_analysis.append(analysis_id)

        return {"analysis_created": created_analysis}

    @task("final_report", inject_context=True)
    def final_report_func(task_ctx):
        return "report_complete"

    # Build graph
    graph.add_node(processor_func, "processor")
    graph.add_node(final_report_func, "final_report")
    graph.add_edge("processor", "final_report")

    # Execute workflow
    context.execute()

    # Verify all tasks executed
    assert "processor" in context.executed
    assert "analysis_0" in context.executed
    assert "analysis_1" in context.executed
    assert "analysis_2" in context.executed
    assert "final_report" in context.executed

    # Verify analysis tasks executed before final report
    executed_order = list(context.executed)
    report_idx = executed_order.index("final_report")

    for i in range(3):
        analysis_idx = executed_order.index(f"analysis_{i}")
        assert analysis_idx < report_idx


def test_jump_to_existing_task_nodes():
    """Test jumping to existing task nodes."""
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "start_node", max_steps=15)

    @task("start_node", inject_context=True)
    def start_node_func(task_ctx):
        # Jump to an existing task node that's already in the graph
        jump_to_task = graph.get_node("existing_task")
        task_ctx.next_task(jump_to_task)
        return "start_complete"

    @task("existing_task", inject_context=True)
    def existing_task_func(task_ctx):
        # This existing task can also jump to another existing task
        other_task = graph.get_node("other_existing")
        task_ctx.next_task(other_task)
        return "existing_complete"

    @task("other_existing", inject_context=True)
    def other_existing_func(task_ctx):
        return "other_complete"

    @task("successor_node", inject_context=True)
    def successor_node_func(task_ctx):
        return "successor_complete"

    @task("final_node", inject_context=True)
    def final_node_func(task_ctx):
        return "final_complete"

    # Build graph with all nodes first
    graph.add_node(start_node_func, "start_node")
    graph.add_node(existing_task_func, "existing_task")
    graph.add_node(other_existing_func, "other_existing")
    graph.add_node(successor_node_func, "successor_node")
    graph.add_node(final_node_func, "final_node")

    # Add edges: start -> successor -> final
    # Note: existing_task and other_existing are NOT connected by edges
    graph.add_edge("start_node", "successor_node")
    graph.add_edge("successor_node", "final_node")

    # Execute workflow
    context.execute()

    # Verify jumped tasks executed
    assert "start_node" in context.executed
    assert "existing_task" in context.executed
    assert "other_existing" in context.executed

    # Verify successors were skipped due to goto behavior
    assert "successor_node" not in context.executed
    assert "final_node" not in context.executed


def test_complex_task_jumping():
    """Test complex task jumping scenarios."""
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "controller", max_steps=25)

    @task("controller", inject_context=True)
    def controller_func(task_ctx):
        # Jump to multiple existing tasks based on conditions
        tasks_to_execute = ["validator", "transformer", "analyzer"]

        for task_name in tasks_to_execute:
            existing_task = graph.get_node(task_name)
            task_ctx.next_task(existing_task)

        return "controller_complete"

    @task("validator", inject_context=True)
    def validator_func(task_ctx):
        return "validation_complete"

    @task("transformer", inject_context=True)
    def transformer_func(task_ctx):
        return "transform_complete"

    @task("analyzer", inject_context=True)
    def analyzer_func(task_ctx):
        # Analyzer decides to jump to processor if needed
        processor_task = graph.get_node("data_processor")
        task_ctx.next_task(processor_task)
        return "analysis_complete"

    @task("data_processor", inject_context=True)
    def processor_func(task_ctx):
        return "processing_complete"

    @task("successor", inject_context=True)
    def successor_func(task_ctx):
        return "successor_complete"

    # Build complex graph
    graph.add_node(controller_func, "controller")
    graph.add_node(validator_func, "validator")
    graph.add_node(transformer_func, "transformer")
    graph.add_node(analyzer_func, "analyzer")
    graph.add_node(processor_func, "data_processor")
    graph.add_node(successor_func, "successor")

    # Only connect controller -> successor (all other tasks are jumped to)
    graph.add_edge("controller", "successor")

    # Execute workflow
    context.execute()

    # Verify all jumped tasks executed
    expected_executed = {"controller", "validator", "transformer", "analyzer", "data_processor"}
    actually_executed = set(context.executed)

    assert expected_executed.issubset(actually_executed)

    # Verify successor was skipped due to goto behavior
    assert "successor" not in actually_executed
