"""
Test goto functionality for next_task
"""


from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph


def test_goto_skips_successors():
    """Test that goto=True skips successors of current task."""
    graph = TaskGraph()
    execution_log: list[str] = []

    @task("main_task", inject_context=True)
    def main_task(task_ctx):
        execution_log.append("main_task")
        # Jump to existing task with goto=True
        existing_task = graph.get_node("target_task")
        task_ctx.next_task(existing_task, goto=True)
        return "main_complete"

    @task("target_task")
    def target_task():
        execution_log.append("target_task")
        return "target_complete"

    @task("successor_task")
    def successor_task():
        execution_log.append("successor_task")
        return "successor_complete"

    # Build graph with edges
    graph.add_node(main_task, "main_task")
    graph.add_node(target_task, "target_task")
    graph.add_node(successor_task, "successor_task")

    # Add edge: main_task -> successor_task
    graph.add_edge("main_task", "successor_task")

    context = ExecutionContext.create(graph, "main_task", max_steps=10)

    # Execute workflow
    context.execute()

    # Verify successor_task was NOT executed due to goto
    assert "successor_task" not in execution_log, "successor_task should be skipped when using goto=True"
    assert "main_task" in execution_log, "main_task should be executed"
    assert "target_task" in execution_log, "target_task should be executed via goto"


def test_normal_next_task_runs_successors():
    """Test that normal next_task still runs successors."""
    graph = TaskGraph()
    execution_log: list[str] = []

    @task("main_task2", inject_context=True)
    def main_task2(task_ctx):
        execution_log.append("main_task2")
        # Create new dynamic task (not goto)
        @task("dynamic_task")
        def dynamic_task():
            execution_log.append("dynamic_task")
            return "dynamic_complete"

        task_ctx.next_task(dynamic_task)
        return "main_complete"

    @task("successor_task2")
    def successor_task2():
        execution_log.append("successor_task2")
        return "successor_complete"

    # Build graph
    graph.add_node(main_task2, "main_task2")
    graph.add_node(successor_task2, "successor_task2")

    # Add edge: main_task2 -> successor_task2
    graph.add_edge("main_task2", "successor_task2")

    context = ExecutionContext.create(graph, "main_task2", max_steps=10)

    # Execute workflow
    context.execute()

    # Verify successor_task2 WAS executed (normal behavior)
    assert "successor_task2" in execution_log, "successor_task2 should be executed with normal next_task"
    assert "main_task2" in execution_log, "main_task2 should be executed"
    assert "dynamic_task" in execution_log, "dynamic_task should be executed"


def test_mixed_goto_and_normal():
    """Test mixed usage of goto and normal next_task."""
    graph = TaskGraph()
    execution_log: list[str] = []

    @task("controller", inject_context=True)
    def controller(task_ctx):
        execution_log.append("controller")
        # Normal dynamic task creation
        @task("worker")
        def worker():
            execution_log.append("worker")
            return "worker_done"

        task_ctx.next_task(worker)

        # Goto jump (should skip controller's successors)
        target = graph.get_node("special_task")
        task_ctx.next_task(target, goto=True)

        return "controller_complete"

    @task("special_task")
    def special_task():
        execution_log.append("special_task")
        return "special_complete"

    @task("normal_successor")
    def normal_successor():
        execution_log.append("normal_successor")
        return "normal_complete"

    # Build graph
    graph.add_node(controller, "controller")
    graph.add_node(special_task, "special_task")
    graph.add_node(normal_successor, "normal_successor")

    # Add edge: controller -> normal_successor
    graph.add_edge("controller", "normal_successor")

    context = ExecutionContext.create(graph, "controller", max_steps=15)

    # Execute workflow
    context.execute()

    # Verify results
    executed = set(execution_log)

    # Normal successor should be skipped due to goto
    assert "normal_successor" not in executed, "normal_successor should be skipped when using goto=True"

    # All expected tasks should be executed
    assert {"controller", "worker", "special_task"}.issubset(executed)
