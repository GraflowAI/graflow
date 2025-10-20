"""Scenario tests for parallel group error handling with real execution."""

import pytest

from graflow.coordination.coordinator import CoordinationBackend
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.handlers.group_policy import (
    AtLeastNGroupPolicy,
    CriticalGroupPolicy,
    GroupExecutionPolicy,
)
from graflow.core.workflow import workflow
from graflow.exceptions import ParallelGroupError


class TestDefaultStrictMode:
    """Test default strict mode behavior - all tasks must succeed."""

    def test_all_tasks_succeed_with_threading(self):
        """Test that all tasks succeeding works with threading backend."""
        with workflow("test") as wf:
            @task
            def task_a():
                return "result_a"

            @task
            def task_b():
                return "result_b"

            @task
            def task_c():
                return "result_c"

            # Create parallel group with threading backend
            parallel = (task_a | task_b | task_c).with_execution(
                backend=CoordinationBackend.THREADING
            )

            @task
            def final_task():
                return "final"

            parallel >> final_task

            engine = WorkflowEngine()
            context = ExecutionContext.create(wf.graph, parallel.task_id)

            # Should complete successfully
            engine.execute(context)

            # Verify results
            assert context.get_result("task_a") == "result_a"
            assert context.get_result("task_b") == "result_b"
            assert context.get_result("task_c") == "result_c"

    def test_one_task_fails_raises_error_with_threading(self):
        """Test that one task failure raises ParallelGroupError with threading."""
        with workflow("test") as wf:
            @task
            def task_a():
                return "result_a"

            @task
            def task_b():
                raise ValueError("Task B intentionally failed!")

            @task
            def task_c():
                return "result_c"

            # Create parallel group with threading backend
            parallel = (task_a | task_b | task_c).with_execution(
                backend=CoordinationBackend.THREADING
            )

            engine = WorkflowEngine()
            context = ExecutionContext.create(wf.graph, parallel.task_id)

            # Should raise ParallelGroupError
            with pytest.raises(ParallelGroupError) as exc_info:
                engine.execute(context)

            error = exc_info.value
            assert error.group_id == parallel.task_id
            assert len(error.failed_tasks) == 1
            assert error.failed_tasks[0][0] == "task_b"
            assert error.failed_tasks[0][1] is not None and "Task B intentionally failed" in error.failed_tasks[0][1]
            assert set(error.successful_tasks) == {"task_a", "task_c"}

    def test_all_tasks_fail_with_threading(self):
        """Test that all tasks failing raises ParallelGroupError."""
        with workflow("test") as wf:
            @task
            def task_a():
                raise ValueError("Task A failed!")

            @task
            def task_b():
                raise RuntimeError("Task B failed!")

            @task
            def task_c():
                raise TypeError("Task C failed!")

            parallel = (task_a | task_b | task_c).with_execution(
                backend=CoordinationBackend.THREADING
            )

            engine = WorkflowEngine()
            context = ExecutionContext.create(wf.graph, parallel.task_id)

            with pytest.raises(ParallelGroupError) as exc_info:
                engine.execute(context)

            error = exc_info.value
            assert len(error.failed_tasks) == 3
            assert len(error.successful_tasks) == 0

    def test_strict_mode_with_direct_backend(self):
        """Test strict mode with direct execution backend."""
        with workflow("test") as wf:
            @task
            def task_a():
                return "a"

            @task
            def task_b():
                raise Exception("Failed!")

            parallel = (task_a | task_b).with_execution(
                backend=CoordinationBackend.DIRECT
            )

            engine = WorkflowEngine()
            context = ExecutionContext.create(wf.graph, parallel.task_id)

            with pytest.raises(ParallelGroupError) as exc_info:
                engine.execute(context)

            error = exc_info.value
            assert len(error.failed_tasks) == 1
            assert error.successful_tasks == ["task_a"]


class TestCustomHandlers:
    """Test custom handler implementations in real workflows."""

    def test_best_effort_handler_continues_on_failure(self):
        """Test that best-effort handler continues even when tasks fail."""

        with workflow("test") as wf:
            @task
            def task_a():
                return "success"

            @task
            def task_b():
                raise ValueError("Task B failed!")

            @task
            def task_c():
                return "success"

            @task
            def final_task():
                return "completed"

            # Use best-effort handler
            parallel = (task_a | task_b | task_c).with_execution(
                backend=CoordinationBackend.THREADING,
                policy="best_effort"
            )

            parallel >> final_task

            engine = WorkflowEngine()
            context = ExecutionContext.create(wf.graph, parallel.task_id)

            # Should NOT raise exception and continue to final_task
            engine.execute(context)

            # Verify final task executed
            assert context.get_result("final_task") == "completed"
            # Verify successful tasks completed
            assert context.get_result("task_a") == "success"
            assert context.get_result("task_c") == "success"

    def test_at_least_n_success_handler(self):
        """Test at-least-N-success handler with various scenarios."""

        # Test 1: 3 out of 4 succeed (min=2) - should pass
        with workflow("test1") as wf1:
            @task
            def task_a():
                return "a"

            @task
            def task_b():
                return "b"

            @task
            def task_c():
                return "c"

            @task
            def task_d():
                raise Exception("Failed!")

            parallel = (task_a | task_b | task_c | task_d).with_execution(
                backend=CoordinationBackend.THREADING,
                policy=AtLeastNGroupPolicy(min_success=2)
            )

            engine = WorkflowEngine()
            context = ExecutionContext.create(wf1.graph, parallel.task_id)

            # Should succeed (3 >= 2)
            engine.execute(context)

        # Test 2: 1 out of 4 succeed (min=2) - should fail
        with workflow("test2") as wf2:
            @task
            def task_e():
                return "e"

            @task
            def task_f():
                raise Exception("Failed!")

            @task
            def task_g():
                raise Exception("Failed!")

            @task
            def task_h():
                raise Exception("Failed!")

            parallel2 = (task_e | task_f | task_g | task_h).with_execution(
                backend=CoordinationBackend.THREADING,
                policy=AtLeastNGroupPolicy(min_success=2)
            )

            engine2 = WorkflowEngine()
            context2 = ExecutionContext.create(wf2.graph, parallel2.task_id)

            # Should fail (1 < 2)
            with pytest.raises(ParallelGroupError) as exc_info:
                engine2.execute(context2)

            error = exc_info.value
            assert "1/2" in str(error)

    def test_critical_tasks_handler(self):
        """Test critical tasks handler that only fails on critical task failures."""

        # Test 1: Critical task succeeds, optional task fails - should pass
        with workflow("test1") as wf1:
            @task
            def critical_task():
                return "critical success"

            @task
            def optional_task():
                raise Exception("Optional task failed!")

            parallel = (critical_task | optional_task).with_execution(
                backend=CoordinationBackend.THREADING,
                policy=CriticalGroupPolicy(critical_task_ids=["critical_task"])
            )

            engine = WorkflowEngine()
            context = ExecutionContext.create(wf1.graph, parallel.task_id)

            # Should succeed (critical task passed)
            engine.execute(context)
            assert context.get_result("critical_task") == "critical success"

        # Test 2: Critical task fails - should fail
        with workflow("test2") as wf2:
            @task
            def critical_task2():
                raise Exception("Critical task failed!")

            @task
            def optional_task2():
                return "optional success"

            parallel2 = (critical_task2 | optional_task2).with_execution(
                backend=CoordinationBackend.THREADING,
                policy=CriticalGroupPolicy(critical_task_ids=["critical_task2"])
            )

            engine2 = WorkflowEngine()
            context2 = ExecutionContext.create(wf2.graph, parallel2.task_id)

            # Should fail (critical task failed)
            with pytest.raises(ParallelGroupError) as exc_info:
                engine2.execute(context2)

            error = exc_info.value
            assert "critical_task2" in str(error)


class TestErrorPropagation:
    """Test error propagation and exception details."""

    def test_error_details_preserved(self):
        """Test that error details are preserved in ParallelGroupError."""
        with workflow("test") as wf:
            @task
            def task_a():
                raise ValueError("Specific error message from task_a")

            @task
            def task_b():
                raise RuntimeError("Different error from task_b")

            @task
            def task_c():
                return "success"

            parallel = (task_a | task_b | task_c).with_execution(
                backend=CoordinationBackend.THREADING
            )

            engine = WorkflowEngine()
            context = ExecutionContext.create(wf.graph, parallel.task_id)

            with pytest.raises(ParallelGroupError) as exc_info:
                engine.execute(context)

            error = exc_info.value
            # Check that specific error messages are preserved
            failed_task_dict = dict(error.failed_tasks)
            assert failed_task_dict["task_a"] is not None and "Specific error message from task_a" in failed_task_dict["task_a"]
            assert failed_task_dict["task_b"] is not None and "Different error from task_b" in failed_task_dict["task_b"]

    def test_successor_not_executed_on_failure(self):
        """Test that successor tasks are not executed when parallel group fails."""
        with workflow("test") as wf:
            @task
            def task_a():
                return "a"

            @task
            def task_b():
                raise Exception("Failed!")

            @task
            def successor_task():
                return "should not execute"

            parallel = (task_a | task_b).with_execution(
                backend=CoordinationBackend.THREADING
            )

            parallel >> successor_task

            engine = WorkflowEngine()
            context = ExecutionContext.create(wf.graph, parallel.task_id)

            with pytest.raises(ParallelGroupError):
                engine.execute(context)

            # Successor should not have executed
            assert context.get_result("successor_task") is None


class TestHandlerWithDifferentBackends:
    """Test handlers work with different execution backends."""

    def test_custom_handler_with_direct_backend(self):
        """Test custom handler with DIRECT backend."""

        class TestPolicy(GroupExecutionPolicy):
            def get_name(self):
                return "test"

            def on_group_finished(self, group_id, tasks, results, context):
                # Custom logic: fail if more than 1 task fails
                failed = [tid for tid, r in results.items() if not r.success]
                if len(failed) > 1:
                    raise ParallelGroupError(
                        f"Too many failures: {len(failed)}",
                        group_id=group_id,
                        failed_tasks=[(tid, results[tid].error_message) for tid in failed],
                        successful_tasks=[tid for tid, r in results.items() if r.success],
                    )

        with workflow("test") as wf:
            @task
            def task_a():
                return "a"

            @task
            def task_b():
                raise Exception("Failed!")

            parallel = (task_a | task_b).with_execution(
                backend=CoordinationBackend.DIRECT,
                policy=TestPolicy(),
            )

            engine = WorkflowEngine()
            context = ExecutionContext.create(wf.graph, parallel.task_id)

            # Should succeed (only 1 failure, threshold is > 1)
            engine.execute(context)

    def test_custom_handler_with_threading_backend(self):
        """Test custom handler with THREADING backend."""

        class PercentagePolicy(GroupExecutionPolicy):
            def __init__(self, min_percentage: float):
                self.min_percentage = min_percentage

            def get_name(self):
                return f"percentage_{int(self.min_percentage * 100)}"

            def on_group_finished(self, group_id, tasks, results, context):
                success_count = sum(1 for r in results.values() if r.success)
                success_rate = success_count / len(results)
                if success_rate < self.min_percentage:
                    failed = [
                        (tid, r.error_message) for tid, r in results.items() if not r.success
                    ]
                    raise ParallelGroupError(
                        f"Success rate {success_rate:.1%} < {self.min_percentage:.1%}",
                        group_id=group_id,
                        failed_tasks=failed,
                        successful_tasks=[tid for tid, r in results.items() if r.success],
                    )

        with workflow("test") as wf:
            @task
            def task_a():
                return "a"

            @task
            def task_b():
                return "b"

            @task
            def task_c():
                return "c"

            @task
            def task_d():
                raise Exception("Failed!")

            # 75% success rate (3/4), need 70%
            parallel = (task_a | task_b | task_c | task_d).with_execution(
                backend=CoordinationBackend.THREADING,
                policy=PercentagePolicy(min_percentage=0.70),
            )

            engine = WorkflowEngine()
            context = ExecutionContext.create(wf.graph, parallel.task_id)

            # Should succeed (75% >= 70%)
            engine.execute(context)
