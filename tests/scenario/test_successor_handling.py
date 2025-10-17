"""
Test successor handling for dynamic tasks
"""

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph


def test_dynamic_task_with_successors():
    """Test that dynamic tasks get their successors processed."""
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "main_task", max_steps=10)

    @task("main_task", inject_context=True)
    def main_task(task_ctx):
        # Create dynamic task
        @task("dynamic_processor")
        def dynamic_processor():
            return "processed_data"

        # Add dynamic task to queue
        task_ctx.next_task(dynamic_processor)
        return "main_complete"

    @task("dynamic_cleanup")
    def dynamic_cleanup():
        return "cleanup_complete"

    @task("final_task")
    def final_task():
        return "final_complete"

    # Build graph with edges
    graph.add_node(main_task, "main_task")
    graph.add_node(dynamic_cleanup, "dynamic_cleanup")
    graph.add_node(final_task, "final_task")

    # Add edges: main_task -> final_task
    graph.add_edge("main_task", "final_task")

    # Execute workflow
    context.execute()

    # Verify execution via stored results in the context channel
    assert context.get_result("main_task") == "main_complete"
    assert context.get_result("dynamic_processor") == "processed_data"
    assert context.get_result("final_task") == "final_complete"


def test_dynamic_task_with_manual_successors():
    """Test dynamic task with manually added successors."""
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "controller", max_steps=15)

    @task("controller", inject_context=True)
    def controller(task_ctx):
        # Create dynamic task
        @task("data_processor")
        def data_processor():
            return "data_processed"

        # Add dynamic task
        task_ctx.next_task(data_processor)

        # Manually add edge for dynamic task (simulating runtime graph modification)
        graph.add_edge("data_processor", "cleanup_task2")

        return "controller_complete"

    @task("cleanup_task2")
    def cleanup_task2():
        return "cleanup_complete"

    @task("final_task2")
    def final_task2():
        return "final_complete"

    # Build initial graph
    graph.add_node(controller, "controller")
    graph.add_node(cleanup_task2, "cleanup_task2")
    graph.add_node(final_task2, "final_task2")

    # Add edges
    graph.add_edge("controller", "final_task2")

    # Execute workflow
    context.execute()

    # Verify execution via stored results in the context channel
    assert context.get_result("controller") == "controller_complete"
    assert context.get_result("data_processor") == "data_processed"
    assert context.get_result("cleanup_task2") == "cleanup_complete"
    assert context.get_result("final_task2") == "final_complete"
