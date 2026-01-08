"""
Scenario tests for workflow control features (terminate_workflow and cancel_workflow).

These tests verify real-world scenarios for early workflow termination and cancellation.
"""

import pytest

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph
from graflow.exceptions import GraflowWorkflowCanceledError


def test_cache_hit_terminates_workflow():
    """Scenario: Cache hit allows early termination, skipping expensive operations."""
    graph = TaskGraph()
    execution_log: list[str] = []
    cache_data = {"result": "cached_value"}

    @task("check_cache", inject_context=True)
    def check_cache(task_ctx):
        execution_log.append("check_cache")
        # Simulate cache hit
        if cache_data:
            task_ctx.terminate_workflow("Cache hit - skipping fetch and transform")
            return cache_data
        return None

    @task("fetch_data")
    def fetch_data():
        execution_log.append("fetch_data")
        return {"result": "fetched_value"}

    @task("transform_data")
    def transform_data():
        execution_log.append("transform_data")
        return {"result": "transformed_value"}

    @task("load_data")
    def load_data():
        execution_log.append("load_data")
        return "loaded"

    # Build pipeline: check_cache -> fetch_data -> transform_data -> load_data
    graph.add_node(check_cache, "check_cache")
    graph.add_node(fetch_data, "fetch_data")
    graph.add_node(transform_data, "transform_data")
    graph.add_node(load_data, "load_data")

    graph.add_edge("check_cache", "fetch_data")
    graph.add_edge("fetch_data", "transform_data")
    graph.add_edge("transform_data", "load_data")

    context = ExecutionContext.create(graph, "check_cache", max_steps=10)
    context.execute()

    # Verify only check_cache was executed (cache hit)
    assert execution_log == ["check_cache"], f"Expected only check_cache, got {execution_log}"
    assert "check_cache" in context.completed_tasks, "check_cache should be completed"
    assert "fetch_data" not in execution_log, "fetch_data should not execute on cache hit"
    assert "transform_data" not in execution_log, "transform_data should not execute on cache hit"
    assert "load_data" not in execution_log, "load_data should not execute on cache hit"


def test_cache_miss_executes_full_pipeline():
    """Scenario: Cache miss leads to full pipeline execution."""
    graph = TaskGraph()
    execution_log: list[str] = []
    cache_data = None  # Cache miss

    @task("check_cache", inject_context=True)
    def check_cache(task_ctx):
        execution_log.append("check_cache")
        if cache_data:
            task_ctx.terminate_workflow("Cache hit")
            return cache_data
        return None

    @task("fetch_data")
    def fetch_data():
        execution_log.append("fetch_data")
        return {"result": "fetched_value"}

    @task("transform_data")
    def transform_data():
        execution_log.append("transform_data")
        return {"result": "transformed_value"}

    # Build pipeline
    graph.add_node(check_cache, "check_cache")
    graph.add_node(fetch_data, "fetch_data")
    graph.add_node(transform_data, "transform_data")

    graph.add_edge("check_cache", "fetch_data")
    graph.add_edge("fetch_data", "transform_data")

    context = ExecutionContext.create(graph, "check_cache", max_steps=10)
    context.execute()

    # Verify full pipeline executed (cache miss)
    assert execution_log == ["check_cache", "fetch_data", "transform_data"], (
        f"Expected full pipeline, got {execution_log}"
    )
    assert "check_cache" in context.completed_tasks
    assert "fetch_data" in context.completed_tasks
    assert "transform_data" in context.completed_tasks


def test_data_validation_failure_cancels_workflow():
    """Scenario: Invalid data causes workflow cancellation."""
    graph = TaskGraph()
    execution_log: list[str] = []
    invalid_data = {"value": "invalid"}

    @task("validate_input", inject_context=True)
    def validate_input(task_ctx):
        execution_log.append("validate_input")
        # Simulate validation failure
        if invalid_data.get("value") == "invalid":
            task_ctx.cancel_workflow("Data validation failed: invalid format")
        return invalid_data

    @task("process_data")
    def process_data():
        execution_log.append("process_data")
        return {"processed": True}

    @task("store_result")
    def store_result():
        execution_log.append("store_result")
        return "stored"

    # Build pipeline
    graph.add_node(validate_input, "validate_input")
    graph.add_node(process_data, "process_data")
    graph.add_node(store_result, "store_result")

    graph.add_edge("validate_input", "process_data")
    graph.add_edge("process_data", "store_result")

    context = ExecutionContext.create(graph, "validate_input", max_steps=10)

    # Verify cancellation exception is raised
    with pytest.raises(GraflowWorkflowCanceledError) as exc_info:
        context.execute()

    assert exc_info.value.task_id == "validate_input"
    assert "validation failed" in str(exc_info.value).lower()

    # Verify validate_input was NOT marked as completed (cancellation)
    assert "validate_input" not in context.completed_tasks, "validate_input should NOT be completed on cancellation"

    # Verify downstream tasks were not executed
    assert execution_log == ["validate_input"], f"Expected only validate_input, got {execution_log}"
    assert "process_data" not in execution_log, "process_data should not execute after cancellation"
    assert "store_result" not in execution_log, "store_result should not execute after cancellation"


def test_data_validation_success_continues_workflow():
    """Scenario: Valid data allows workflow to continue."""
    graph = TaskGraph()
    execution_log: list[str] = []
    valid_data = {"value": "valid"}

    @task("validate_input", inject_context=True)
    def validate_input(task_ctx):
        execution_log.append("validate_input")
        if valid_data.get("value") == "invalid":
            task_ctx.cancel_workflow("Data validation failed")
        return valid_data

    @task("process_data")
    def process_data():
        execution_log.append("process_data")
        return {"processed": True}

    @task("store_result")
    def store_result():
        execution_log.append("store_result")
        return "stored"

    # Build pipeline
    graph.add_node(validate_input, "validate_input")
    graph.add_node(process_data, "process_data")
    graph.add_node(store_result, "store_result")

    graph.add_edge("validate_input", "process_data")
    graph.add_edge("process_data", "store_result")

    context = ExecutionContext.create(graph, "validate_input", max_steps=10)
    context.execute()

    # Verify full pipeline executed (validation passed)
    assert execution_log == ["validate_input", "process_data", "store_result"], (
        f"Expected full pipeline, got {execution_log}"
    )
    assert "validate_input" in context.completed_tasks
    assert "process_data" in context.completed_tasks
    assert "store_result" in context.completed_tasks


def test_conditional_workflow_early_completion():
    """Scenario: Workflow completes early when condition is met."""
    graph = TaskGraph()
    execution_log: list[str] = []

    @task("check_status", inject_context=True)
    def check_status(task_ctx):
        execution_log.append("check_status")
        channel = task_ctx.get_channel()
        status = channel.get("workflow_status", "PENDING")

        if status == "COMPLETED":
            task_ctx.terminate_workflow("Workflow already completed")
            return "already_done"

        return "needs_processing"

    @task("process_step1")
    def process_step1():
        execution_log.append("process_step1")
        return "step1_done"

    @task("process_step2")
    def process_step2():
        execution_log.append("process_step2")
        return "step2_done"

    # Build pipeline
    graph.add_node(check_status, "check_status")
    graph.add_node(process_step1, "process_step1")
    graph.add_node(process_step2, "process_step2")

    graph.add_edge("check_status", "process_step1")
    graph.add_edge("process_step1", "process_step2")

    # Test 1: Status is COMPLETED (early termination)
    context = ExecutionContext.create(graph, "check_status", max_steps=10)
    context.get_channel().set("workflow_status", "COMPLETED")
    context.execute()

    assert execution_log == ["check_status"], f"Expected early termination, got {execution_log}"
    assert "check_status" in context.completed_tasks

    # Test 2: Status is PENDING (normal execution)
    execution_log.clear()
    context2 = ExecutionContext.create(graph, "check_status", max_steps=10)
    context2.get_channel().set("workflow_status", "PENDING")
    context2.execute()

    assert execution_log == ["check_status", "process_step1", "process_step2"], (
        f"Expected full pipeline, got {execution_log}"
    )


def test_parallel_tasks_with_termination():
    """Scenario: One branch terminates while others would continue."""
    graph = TaskGraph()
    execution_log: list[str] = []

    @task("start_task")
    def start_task():
        execution_log.append("start_task")
        return "started"

    @task("branch_a", inject_context=True)
    def branch_a(task_ctx):
        execution_log.append("branch_a")
        # Terminate early
        task_ctx.terminate_workflow("Branch A completes early")
        return "branch_a_done"

    @task("branch_b")
    def branch_b():
        execution_log.append("branch_b")
        return "branch_b_done"

    @task("final_task")
    def final_task():
        execution_log.append("final_task")
        return "final_done"

    # Build graph: start -> (branch_a | branch_b) -> final
    graph.add_node(start_task, "start_task")
    graph.add_node(branch_a, "branch_a")
    graph.add_node(branch_b, "branch_b")
    graph.add_node(final_task, "final_task")

    graph.add_edge("start_task", "branch_a")
    graph.add_edge("start_task", "branch_b")
    graph.add_edge("branch_a", "final_task")
    graph.add_edge("branch_b", "final_task")

    context = ExecutionContext.create(graph, "start_task", max_steps=10)
    context.execute()

    # Verify that execution stopped after branch_a or branch_b terminated
    assert "start_task" in execution_log
    # Note: Exact execution depends on queue order, but workflow should terminate
    # after one of the branches calls terminate_workflow
    assert "final_task" not in execution_log, "final_task should not execute after termination"


def test_error_detection_cancels_workflow():
    """Scenario: Critical error detected mid-workflow causes cancellation."""
    graph = TaskGraph()
    execution_log: list[str] = []

    @task("prepare_data")
    def prepare_data():
        execution_log.append("prepare_data")
        return {"data": "prepared"}

    @task("check_integrity", inject_context=True)
    def check_integrity(task_ctx):
        execution_log.append("check_integrity")
        # Simulate integrity check failure
        integrity_ok = False
        if not integrity_ok:
            task_ctx.cancel_workflow("Data integrity check failed - aborting workflow")
        return "integrity_ok"

    @task("process_data")
    def process_data():
        execution_log.append("process_data")
        return {"processed": True}

    @task("finalize")
    def finalize():
        execution_log.append("finalize")
        return "finalized"

    # Build pipeline
    graph.add_node(prepare_data, "prepare_data")
    graph.add_node(check_integrity, "check_integrity")
    graph.add_node(process_data, "process_data")
    graph.add_node(finalize, "finalize")

    graph.add_edge("prepare_data", "check_integrity")
    graph.add_edge("check_integrity", "process_data")
    graph.add_edge("process_data", "finalize")

    context = ExecutionContext.create(graph, "prepare_data", max_steps=10)

    with pytest.raises(GraflowWorkflowCanceledError) as exc_info:
        context.execute()

    assert "integrity" in str(exc_info.value).lower()

    # Verify prepare_data completed but check_integrity did not
    assert "prepare_data" in context.completed_tasks
    assert "check_integrity" not in context.completed_tasks

    # Verify execution stopped at check_integrity
    assert execution_log == ["prepare_data", "check_integrity"], (
        f"Expected execution up to check_integrity, got {execution_log}"
    )
    assert "process_data" not in execution_log
    assert "finalize" not in execution_log


def test_mixed_control_flow_priority():
    """Scenario: Test control flow priority (cancel > terminate > goto > normal)."""
    graph = TaskGraph()
    execution_log: list[str] = []

    # Test 1: Cancel has highest priority
    @task("task_cancel", inject_context=True)
    def task_cancel(task_ctx):
        execution_log.append("task_cancel")
        task_ctx.cancel_workflow("Canceling")
        task_ctx.terminate_workflow("This should not take effect")
        return "done"

    @task("successor_cancel")
    def successor_cancel():
        execution_log.append("successor_cancel")
        return "done"

    graph.add_node(task_cancel, "task_cancel")
    graph.add_node(successor_cancel, "successor_cancel")
    graph.add_edge("task_cancel", "successor_cancel")

    context = ExecutionContext.create(graph, "task_cancel", max_steps=10)

    with pytest.raises(GraflowWorkflowCanceledError):
        context.execute()

    assert execution_log == ["task_cancel"]
    assert "task_cancel" not in context.completed_tasks, "Cancel should prevent completion"

    # Test 2: Terminate takes effect when no cancel
    execution_log.clear()
    graph2 = TaskGraph()

    @task("task_terminate", inject_context=True)
    def task_terminate(task_ctx):
        execution_log.append("task_terminate")
        task_ctx.terminate_workflow("Terminating normally")
        return "done"

    @task("successor_terminate")
    def successor_terminate():
        execution_log.append("successor_terminate")
        return "done"

    graph2.add_node(task_terminate, "task_terminate")
    graph2.add_node(successor_terminate, "successor_terminate")
    graph2.add_edge("task_terminate", "successor_terminate")

    context2 = ExecutionContext.create(graph2, "task_terminate", max_steps=10)
    context2.execute()

    assert execution_log == ["task_terminate"]
    assert "task_terminate" in context2.completed_tasks, "Terminate should mark task as completed"
    assert "successor_terminate" not in execution_log, "Successors should not execute after terminate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
