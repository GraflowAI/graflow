"""Pytest-style tests for ParallelGroup functionality."""

import pytest

from graflow.coordination.coordinator import CoordinationBackend
from graflow.coordination.executor import GroupExecutor
from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.core.task import ParallelGroup, Task, TaskWrapper
from graflow.core.workflow import WorkflowContext


@pytest.fixture
def execution_context():
    """Create execution context for tests."""
    graph = TaskGraph()
    return ExecutionContext.create(graph, "test", max_steps=10)


class TestParallelGroup:
    """Test cases for ParallelGroup class."""

    def test_parallel_group_creation(self):
        """Test ParallelGroup creation and basic properties."""
        with WorkflowContext("test_workflow"):
            task1 = Task("task1")
            task2 = Task("task2")
            parallel_group = ParallelGroup([task1, task2])

            assert len(parallel_group.tasks) == 2
            assert task1 in parallel_group.tasks
            assert task2 in parallel_group.tasks
            assert parallel_group.task_id.startswith("ParallelGroup_")

    def test_parallel_group_or_operator(self):
        """Test | operator for creating parallel groups."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            task3 = Task("task3")
            parallel_group = task1 | task2 | task3

            assert isinstance(parallel_group, ParallelGroup)
            assert len(parallel_group.tasks) == 3

    def test_parallel_group_run_with_default_executor(self, execution_context, mocker):
        """Test ParallelGroup.run() with default GroupExecutor."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            parallel_group = ParallelGroup([task1, task2])
            parallel_group.set_execution_context(execution_context)

            mock_execute = mocker.patch.object(GroupExecutor, 'execute_parallel_group')
            parallel_group.run()

            mock_execute.assert_called_once()
            args, kwargs = mock_execute.call_args
            group_id, task_specs = args

            assert group_id == parallel_group.task_id
            assert len(task_specs) == 2
            assert task_specs[0].task_id == "task1"
            assert task_specs[1].task_id == "task2"

    def test_parallel_group_run_with_custom_executor(self, execution_context, mocker):
        """Test ParallelGroup.run() with custom GroupExecutor."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            custom_executor = GroupExecutor(CoordinationBackend.DIRECT)
            execution_context.group_executor = custom_executor

            parallel_group = ParallelGroup([task1, task2])
            parallel_group.set_execution_context(execution_context)

            mock_execute = mocker.patch.object(custom_executor, 'execute_parallel_group')
            parallel_group.run()

            mock_execute.assert_called_once()

    def test_parallel_group_with_task_wrapper(self, execution_context, mocker):
        """Test ParallelGroup with TaskWrapper that requires context injection."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            def test_func(ctx):
                return f"result_{ctx.task_id}"

            task_wrapper = TaskWrapper("wrapper_task", test_func, inject_context=True)
            parallel_group = ParallelGroup([task1, task_wrapper])
            parallel_group.set_execution_context(execution_context)

            mock_execute = mocker.patch.object(GroupExecutor, 'execute_parallel_group')
            parallel_group.run()

            mock_execute.assert_called_once()
            args, kwargs = mock_execute.call_args
            group_id, task_specs = args

            # Check that wrapper task has context function
            wrapper_spec = task_specs[1]
            assert wrapper_spec.task_id == "wrapper_task"
            assert wrapper_spec.func != task_wrapper.run

    def test_parallel_group_dependency_operators(self):
        """Test >> and << operators with ParallelGroup."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            parallel_group = ParallelGroup([task1, task2])
            result_task = Task("result")

            # Test >> operator
            chained = parallel_group >> result_task
            assert chained == result_task

            # Test << operator
            source_task = Task("source")
            chained = parallel_group << source_task
            assert chained == source_task

    def test_parallel_group_repr(self):
        """Test string representation of ParallelGroup."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            parallel_group = ParallelGroup([task1, task2])

            repr_str = repr(parallel_group)
            assert "ParallelGroup" in repr_str
            assert "task1" in repr_str
            assert "task2" in repr_str


class TestParallelGroupIntegration:
    """Integration tests for ParallelGroup with real execution."""

    def test_direct_execution_integration(self, mocker):
        """Test ParallelGroup with direct execution backend."""
        results = []

        with WorkflowContext("test"):
            def task_func_1():
                results.append("task1_executed")
                return "result1"

            def task_func_2():
                results.append("task2_executed")
                return "result2"

            task1 = TaskWrapper("task1", task_func_1)
            task2 = TaskWrapper("task2", task_func_2)

            parallel_group = ParallelGroup([task1, task2])

            graph = TaskGraph()
            context = ExecutionContext.create(graph, "test")
            context.group_executor = GroupExecutor(CoordinationBackend.DIRECT)

            parallel_group.set_execution_context(context)

            # Capture print output to verify execution
            mock_print = mocker.patch('builtins.print')
            parallel_group.run()

            # Verify print calls show direct execution
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("Running parallel group:" in call for call in print_calls)
            assert any("Direct tasks:" in call for call in print_calls)

            # Verify both tasks were executed
            assert "task1_executed" in results
            assert "task2_executed" in results