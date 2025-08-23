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
    graph.add_node("main_task", task=main_task)
    graph.add_node("dynamic_cleanup", task=dynamic_cleanup)
    graph.add_node("final_task", task=final_task)

    # Add edges: main_task -> final_task
    graph.add_edge("main_task", "final_task")

    # Execute workflow
    context.execute()

    # Verify execution
    assert "main_task" in context.executed, "main_task should be executed"
    assert "dynamic_processor" in context.executed, "dynamic_processor should be created and executed"
    assert "final_task" in context.executed, "final_task should be executed as successor of main_task"


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
    graph.add_node("controller", task=controller)
    graph.add_node("cleanup_task2", task=cleanup_task2)
    graph.add_node("final_task2", task=final_task2)

    # Add edges
    graph.add_edge("controller", "final_task2")

    # Execute workflow
    context.execute()

    # Verify execution
    assert "controller" in context.executed, "controller should be executed"
    assert "data_processor" in context.executed, "data_processor should be created and executed"
    assert "cleanup_task2" in context.executed, "cleanup_task2 should be executed as successor of data_processor"
    assert "final_task2" in context.executed, "final_task2 should be executed as successor of controller"
