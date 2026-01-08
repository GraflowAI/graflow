"""Test various task decorator syntax options."""

import pytest

from graflow.core.decorators import task
from graflow.core.task import TaskWrapper


@pytest.fixture(autouse=True)
def setup_workflow_context():
    """Setup workflow context for each test."""
    from contextvars import ContextVar

    import graflow.core.workflow as workflow_module
    from graflow.core.workflow import WorkflowContext

    # Create a new context var and workflow context
    context_var = ContextVar("test_workflow_context")
    workflow_context = WorkflowContext("test_workflow")
    context_var.set(workflow_context)
    original_context_var = workflow_module._current_context
    workflow_module._current_context = context_var

    yield

    # Cleanup
    workflow_module._current_context = original_context_var


def test_task_decorator_without_parentheses():
    """Test @task syntax without parentheses."""

    @task
    def my_function():
        return "test"

    assert isinstance(my_function, TaskWrapper)
    assert my_function.task_id == "my_function"
    assert my_function() == "test"


def test_task_decorator_with_empty_parentheses():
    """Test @task() syntax with empty parentheses."""

    @task()
    def my_function():
        return "test"

    assert isinstance(my_function, TaskWrapper)
    assert my_function.task_id == "my_function"
    assert my_function() == "test"


def test_task_decorator_with_string_id():
    """Test @task('task_id') syntax."""

    @task("custom_task_id")
    def my_function():
        return "test"

    assert isinstance(my_function, TaskWrapper)
    assert my_function.task_id == "custom_task_id"
    assert my_function() == "test"


def test_task_decorator_with_keyword_id():
    """Test @task(id='task_id') syntax."""

    @task(id="custom_task_id")
    def my_function():
        return "test"

    assert isinstance(my_function, TaskWrapper)
    assert my_function.task_id == "custom_task_id"
    assert my_function() == "test"


def test_task_decorator_with_inject_context():
    """Test @task(inject_context=True) syntax."""

    @task(inject_context=True)
    def my_function():
        return "test"

    assert isinstance(my_function, TaskWrapper)
    assert my_function.task_id == "my_function"
    assert my_function.inject_context is True


def test_task_decorator_with_string_and_inject_context():
    """Test @task('task_id') with inject_context."""

    @task("custom_id")
    def my_function():
        return "test"

    # The inject_context should default to False when using string syntax
    assert isinstance(my_function, TaskWrapper)
    assert my_function.task_id == "custom_id"
    assert my_function.inject_context is False


def test_task_decorator_mixed_syntax_equivalence():
    """Test that @task('id') and @task(id='id') produce equivalent results."""

    @task("test_id_1")
    def func1():
        return "test1"

    @task(id="test_id_2")
    def func2():
        return "test2"

    # Both should be TaskWrapper instances with correct IDs
    assert func1.task_id == "test_id_1"
    assert func2.task_id == "test_id_2"
    assert isinstance(func1, TaskWrapper)
    assert isinstance(func2, TaskWrapper)
    assert func1.inject_context == func2.inject_context is False


def test_task_decorator_with_string_and_inject_context_keyword():
    """Test @task('task_id', inject_context=True) syntax."""

    @task("custom_id_with_context", inject_context=True)
    def my_function():
        return "test"

    assert isinstance(my_function, TaskWrapper)
    assert my_function.task_id == "custom_id_with_context"
    assert my_function.inject_context is True


def test_task_decorator_all_syntax_variations():
    """Test all supported syntax variations work correctly."""

    @task
    def func1():
        return "test1"

    @task()
    def func2():
        return "test2"

    @task("custom_id_3")
    def func3():
        return "test3"

    @task("custom_id_4", inject_context=True)
    def func4():
        return "test4"

    @task(id="custom_id_5")
    def func5():
        return "test5"

    @task(inject_context=True)
    def func6():
        return "test6"

    # Verify all are TaskWrapper instances
    functions = [func1, func2, func3, func4, func5, func6]
    for func in functions:
        assert isinstance(func, TaskWrapper)

    # Verify task IDs
    assert func1.task_id == "func1"
    assert func2.task_id == "func2"
    assert func3.task_id == "custom_id_3"
    assert func4.task_id == "custom_id_4"
    assert func5.task_id == "custom_id_5"
    assert func6.task_id == "func6"

    # Verify inject_context settings
    assert func1.inject_context is False
    assert func2.inject_context is False
    assert func3.inject_context is False
    assert func4.inject_context is True
    assert func5.inject_context is False
    assert func6.inject_context is True
