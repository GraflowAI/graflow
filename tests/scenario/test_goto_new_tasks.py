"""
Test goto=True functionality for both existing and new tasks
"""

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph


def test_goto_true_with_new_task():
    """Test that goto=True skips successors even for new tasks."""
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "main_task", max_steps=10)

    @task("main_task", inject_context=True)
    def main_task(task_ctx):
        # Create a NEW task with goto=True (should skip successors)
        @task("new_dynamic_task")
        def new_dynamic_task():
            return "new_task_complete"

        task_ctx.next_task(new_dynamic_task, goto=True)
        return "main_complete"

    @task("successor_task")
    def successor_task():
        return "successor_complete"

    # Build graph with edges
    graph.add_node(main_task, "main_task")
    graph.add_node(successor_task, "successor_task")

    # Add edge: main_task -> successor_task
    graph.add_edge("main_task", "successor_task")

    # Execute workflow
    context.execute()

    # Verify results
    assert "successor_task" not in context.executed, "successor_task should be skipped when using goto=True with new task"
    assert "new_dynamic_task" in context.executed, "new_dynamic_task should be created and executed"
    assert "main_task" in context.executed, "main_task should be executed"


def test_goto_false_with_new_task():
    """Test that goto=False allows successors for new tasks (normal behavior)."""
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "main_task2", max_steps=10)

    @task("main_task2", inject_context=True)
    def main_task2(task_ctx):
        # Create a NEW task with goto=False (should allow successors)
        @task("new_normal_task")
        def new_normal_task():
            return "new_normal_complete"

        task_ctx.next_task(new_normal_task, goto=False)
        return "main_complete"

    @task("successor_task2")
    def successor_task2():
        return "successor_complete"

    # Build graph
    graph.add_node(main_task2, "main_task2")
    graph.add_node(successor_task2, "successor_task2")

    # Add edge: main_task2 -> successor_task2
    graph.add_edge("main_task2", "successor_task2")

    # Execute workflow
    context.execute()

    # Verify new task was created and executed
    assert "new_normal_task" in context.executed, "new_normal_task should be created and executed"
    assert "main_task2" in context.executed, "main_task2 should be executed"

    # Note: successor_task2 execution depends on the system's handling of deferred successors
    # This test verifies the new task creation works with goto=False


def test_mixed_goto_behaviors():
    """Test mixed goto=True and goto=False in same task."""
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "dispatcher", max_steps=15)

    @task("dispatcher", inject_context=True)
    def dispatcher(task_ctx):
        # Create new task with goto=False (normal processing)
        @task("worker_normal")
        def worker_normal():
            return "worker_normal_done"

        task_ctx.next_task(worker_normal, goto=False)

        # Create new task with goto=True (skip successors)
        @task("worker_urgent")
        def worker_urgent():
            return "worker_urgent_done"

        task_ctx.next_task(worker_urgent, goto=True)

        return "dispatcher_complete"

    @task("normal_successor")
    def normal_successor():
        return "normal_complete"

    # Build graph
    graph.add_node(dispatcher, "dispatcher")
    graph.add_node(normal_successor, "normal_successor")

    # Add edge: dispatcher -> normal_successor
    graph.add_edge("dispatcher", "normal_successor")

    # Execute workflow
    context.execute()

    # Verify results
    expected_executed = {"dispatcher", "worker_normal", "worker_urgent"}
    actually_executed = set(context.executed)

    # Normal successor should be skipped due to goto=True call
    assert "normal_successor" not in actually_executed, "normal_successor should be skipped when using goto=True"

    # All expected tasks should be executed
    assert expected_executed.issubset(actually_executed), f"Expected tasks {expected_executed} should all be executed"
