"""Tests for revised graflow core functionality."""

from graflow.core.decorators import task
from graflow.core.task import ParallelGroup, Task, TaskWrapper, clear_workflow_context


def test_task_creation():
    """Test Task creation and basic functionality."""
    clear_workflow_context()
    task_obj = Task("A")
    assert task_obj.name == "A"
    assert str(task_obj) == "Task(A)"


def test_task_run():
    """Test Task execution."""
    clear_workflow_context()
    task_obj = Task("A")
    task_obj.run()  # Should print and not raise


def test_task_decorator():
    """Test @task decorator functionality."""
    clear_workflow_context()

    @task
    def test_func():
        return "test result"

    assert test_func.name == "test_func"
    assert isinstance(test_func, TaskWrapper)
    assert test_func() == "test result"  # Can call directly


def test_task_decorator_with_name():
    """Test @task decorator with custom name."""
    clear_workflow_context()

    @task(name="custom_name")
    def test_func():
        return "test result"

    assert test_func.name == "custom_name"


def test_parallel_group_creation():
    """Test ParallelGroup creation and functionality."""
    clear_workflow_context()
    task_a = Task("A")
    task_b = Task("B")
    group = ParallelGroup([task_a, task_b])

    assert len(group.tasks) == 2
    assert group.tasks[0].name == "A"
    assert group.tasks[1].name == "B"
    assert "ParallelGroup_" in group.name


def test_sequential_operator():
    """Test >> operator for sequential composition."""
    clear_workflow_context()
    task_a = Task("A")
    task_b = Task("B")
    result = task_a >> task_b

    assert result == task_b  # >> returns the right operand


def test_reverse_operator():
    """Test << operator for reverse dependencies."""
    clear_workflow_context()
    task_a = Task("A")
    task_b = Task("B")
    result = task_b << task_a  # a >> b

    assert result == task_b  # << returns self


def test_parallel_operator():
    """Test | operator for parallel composition."""
    clear_workflow_context()
    task_a = Task("A")
    task_b = Task("B")
    group = task_a | task_b

    assert isinstance(group, ParallelGroup)
    assert len(group.tasks) == 2


def test_complex_composition():
    """Test complex composition with multiple operators."""
    clear_workflow_context()

    @task
    def start():
        pass

    @task
    def middle1():
        pass

    @task
    def middle2():
        pass

    @task
    def end():
        pass

    # Create complex flow
    start >> (middle1 | middle2) >> end

    # Should not raise and should create proper dependencies


def test_parallel_group_extension():
    """Test extending parallel groups with | operator."""
    clear_workflow_context()
    task_a = Task("A")
    task_b = Task("B")
    task_c = Task("C")

    group1 = task_a | task_b
    group2 = group1 | task_c

    assert len(group2.tasks) == 3
