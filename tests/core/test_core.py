"""Tests for core graflow functionality."""

from graflow.core.decorators import task
from graflow.core.task import ParallelGroup, SequentialTask, Task, TaskWrapper
from graflow.core.workflow import workflow


def test_task_creation():
    """Test Task creation and basic functionality."""
    task_obj = Task("A")
    assert task_obj.task_id == "A"
    assert str(task_obj) == "Task(A)"


def test_task_run():
    """Test Task execution."""
    with workflow("test_run"):
        task_obj = Task("A")
        task_obj.run()  # Should print "Running task: A"


def test_decorator():
    """Test @task decorator functionality."""
    @task
    def test_func():
        return "test_result"

    assert isinstance(test_func, TaskWrapper)
    assert test_func.task_id == "test_func"


def test_decorator_with_name():
    """Test @task decorator with custom name."""
    @task(id="custom_name")
    def test_func():
        return "test_result"

    assert isinstance(test_func, TaskWrapper)
    assert test_func.task_id == "custom_name"


def test_parallel_group_creation():
    """Test ParallelGroup creation and functionality."""
    with workflow("test_parallel_group"):
        task_a = Task("A")
        task_b = Task("B")
        group = ParallelGroup([task_a, task_b])

        assert group.task_id.startswith("ParallelGroup_")
        assert len(group.tasks) == 2


def test_sequential_operator():
    """Test >> operator for sequential composition."""
    with workflow("test_sequential"):
        task_a = Task("A")
        task_b = Task("B")
        result = task_a >> task_b

        # The >> operator returns SequentialTask
        assert isinstance(result, SequentialTask)
        assert result.leftmost == task_a
        assert result.rightmost == task_b


def test_parallel_operator():
    """Test | operator for parallel composition."""
    with workflow("test_parallel"):
        task_a = Task("A")
        task_b = Task("B")
        group = task_a | task_b

        assert isinstance(group, ParallelGroup)
        assert len(group.tasks) == 2


def test_decorator_chaining():
    """Test chaining decorated functions."""
    @task
    def func1():
        return "result1"

    @task
    def func2():
        return "result2"

    result = func1 >> func2
    assert isinstance(result, SequentialTask)
    assert result.leftmost == func1
    assert result.rightmost == func2


def test_workflow_context():
    """Test workflow context functionality."""
    with workflow("test_workflow") as ctx:
        @task
        def task1():
            return "task1_result"

        @task
        def task2():
            return "task2_result"

        task1 >> task2 # type: ignore

        assert "task1" in ctx.graph.nodes
        assert "task2" in ctx.graph.nodes
        assert list(ctx.graph.edges) == [("task1", "task2")]
