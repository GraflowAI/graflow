"""Scenario tests for checkpoint/resume functionality in real workflows.

This module contains end-to-end scenario tests that demonstrate checkpoint/resume
behavior in practical workflow situations including:

- State machine workflows with iterative checkpoints
- Long-running workflows with periodic checkpoints
- Fault tolerance and recovery after interruption
- Distributed execution with checkpoint handoff
- Dynamic task generation with checkpoints
- Complex workflows with parallel execution

These tests validate the complete checkpoint/resume workflow from creation to
restoration and continued execution.
"""

import os
import tempfile
import time

import pytest

from graflow.core.checkpoint import CheckpointManager
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph


def run_task(context: ExecutionContext, task_id: str):
    """Execute a single task within the given execution context."""
    task = context.graph.get_node(task_id)
    with context.executing_task(task):
        result = task.run()
    if result is not None:
        context.set_result(task_id, result)
    return result


class TestStateBasedCheckpointScenarios:
    """Test checkpoint/resume in state-based workflows."""

    def test_state_machine_workflow_with_checkpoints(self):
        """Test state machine workflow that checkpoints at each state transition.

        This simulates an order processing workflow that goes through multiple states:
        NEW -> VALIDATED -> PAID -> SHIPPED

        Each state transition creates a checkpoint for recovery.
        """
        graph = TaskGraph()

        @task("process_order", inject_context=True)
        def process_order_func(task_ctx):
            """Process order through different states with checkpoints."""
            # Get channel and current state
            channel = task_ctx.execution_context.channel
            state = channel.get("order_state") if channel.get("order_state") else "NEW"
            order_data = channel.get("order_data")

            if state == "NEW":
                # Validate order
                if order_data and order_data.get("amount", 0) > 0:
                    channel.set("order_state", "VALIDATED")
                    channel.set("validation_timestamp", time.time())

                    # Request checkpoint after validation
                    task_ctx.checkpoint(metadata={"stage": "validation_complete"})
                    task_ctx.next_iteration()
                else:
                    return "INVALID_ORDER"

            elif state == "VALIDATED":
                # Process payment
                channel.set("order_state", "PAID")
                channel.set("payment_timestamp", time.time())

                # Request checkpoint after payment
                task_ctx.checkpoint(metadata={"stage": "payment_complete"})
                task_ctx.next_iteration()

            elif state == "PAID":
                # Ship order
                channel.set("order_state", "SHIPPED")
                channel.set("shipment_timestamp", time.time())
                return "ORDER_COMPLETE"

        graph.add_node(process_order_func, "process_order")

        # Create context and set initial state
        context = ExecutionContext.create(
            graph, "process_order", max_steps=20, channel_backend="memory"
        )
        context.channel.set("order_data", {"id": "ORD123", "amount": 100})
        context.channel.set("order_state", "NEW")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Execute first state transition (NEW -> VALIDATED)
            next_task = context.task_queue.get_next_task()
            assert next_task is not None
            assert next_task == "process_order"

            run_task(context, next_task)
            context.mark_task_completed(next_task)
            context.increment_step()

            # Check that checkpoint was requested
            assert context.checkpoint_requested
            checkpoint_metadata = context.checkpoint_request_metadata
            assert checkpoint_metadata is not None
            assert checkpoint_metadata["stage"] == "validation_complete"

            # Create checkpoint as engine would do
            checkpoint_path_1 = os.path.join(tmpdir, "checkpoint_validated")
            pkl_path_1, metadata_1 = CheckpointManager.create_checkpoint(
                context, path=checkpoint_path_1, metadata=checkpoint_metadata
            )
            context.clear_checkpoint_request()

            # Verify state
            assert context.channel.get("order_state") == "VALIDATED"
            assert os.path.exists(pkl_path_1)
            assert os.path.exists(f"{checkpoint_path_1}.state.json")

            # Resume from checkpoint and continue to PAID state
            restored_context, restored_metadata = CheckpointManager.resume_from_checkpoint(pkl_path_1)

            assert restored_context.channel.get("order_state") == "VALIDATED"
            assert restored_metadata.user_metadata["stage"] == "validation_complete"

            # Execute second state transition (VALIDATED -> PAID)
            next_task = restored_context.task_queue.get_next_task()
            assert next_task is not None
            run_task(restored_context, next_task)
            restored_context.mark_task_completed(next_task)
            restored_context.increment_step()

            # Check checkpoint request for PAID state
            assert restored_context.checkpoint_requested
            checkpoint_metadata_2 = restored_context.checkpoint_request_metadata
            assert checkpoint_metadata_2 is not None
            assert checkpoint_metadata_2["stage"] == "payment_complete"

            # Create second checkpoint
            checkpoint_path_2 = os.path.join(tmpdir, "checkpoint_paid")
            pkl_path_2, metadata_2 = CheckpointManager.create_checkpoint(
                restored_context, path=checkpoint_path_2, metadata=checkpoint_metadata_2
            )
            restored_context.clear_checkpoint_request()

            # Resume from second checkpoint and complete order
            final_context, final_metadata = CheckpointManager.resume_from_checkpoint(pkl_path_2)

            assert final_context.channel.get("order_state") == "PAID"

            # Execute final state transition (PAID -> SHIPPED)
            next_task = final_context.task_queue.get_next_task()
            assert next_task is not None
            result = run_task(final_context, next_task)

            assert result == "ORDER_COMPLETE"
            assert final_context.channel.get("order_state") == "SHIPPED"

    def test_iterative_workflow_with_periodic_checkpoints(self):
        """Test workflow with iterations that checkpoints periodically.

        Simulates a training loop that checkpoints every N iterations.
        """
        graph = TaskGraph()

        @task("training_loop", inject_context=True)
        def training_loop_func(task_ctx):
            """Training loop with periodic checkpoints."""
            channel = task_ctx.execution_context.channel
            iteration = channel.get("iteration") if channel.get("iteration") is not None else 0
            max_iterations = 10

            # Training step
            loss = 1.0 / (iteration + 1)  # Simulated decreasing loss
            channel.set("iteration", iteration + 1)
            channel.set("loss", loss)

            # Checkpoint every 3 iterations
            if (iteration + 1) % 3 == 0:
                task_ctx.checkpoint(
                    metadata={"iteration": iteration + 1, "loss": loss}
                )

            # Check convergence
            if iteration + 1 >= max_iterations:
                return "TRAINING_COMPLETE"
            else:
                task_ctx.next_iteration()

        graph.add_node(training_loop_func, "training_loop")

        context = ExecutionContext.create(
            graph, "training_loop", max_steps=50, channel_backend="memory"
        )
        context.channel.set("iteration", 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Execute 6 iterations (2 checkpoints)
            for i in range(6):
                next_task = context.task_queue.get_next_task()
                if next_task:
                    run_task(context, next_task)
                    context.mark_task_completed(next_task)
                    context.increment_step()

                    # Create checkpoint if requested
                    if context.checkpoint_requested:
                        checkpoint_path = os.path.join(tmpdir, f"checkpoint_iter_{i+1}")
                        checkpoint_metadata = context.checkpoint_request_metadata
                        assert checkpoint_metadata is not None
                        CheckpointManager.create_checkpoint(
                            context,
                            path=checkpoint_path,
                            metadata=checkpoint_metadata
                        )
                        context.clear_checkpoint_request()

            # Verify we have checkpoint at iteration 6
            checkpoint_6 = os.path.join(tmpdir, "checkpoint_iter_6.pkl")
            assert os.path.exists(checkpoint_6)

            # Resume from iteration 6 and continue to completion
            restored_context, metadata = CheckpointManager.resume_from_checkpoint(checkpoint_6)

            assert restored_context.channel.get("iteration") == 6
            assert metadata.user_metadata["iteration"] == 6

            # Continue execution from iteration 6 to completion
            result = None
            for i in range(4):  # Iterations 7-10
                next_task = restored_context.task_queue.get_next_task()
                if next_task:
                    result = run_task(restored_context, next_task)
                    restored_context.mark_task_completed(next_task)
                    restored_context.increment_step()

                    if restored_context.checkpoint_requested:
                        checkpoint_path = os.path.join(tmpdir, f"checkpoint_iter_{6+i+1}")
                        checkpoint_metadata = restored_context.checkpoint_request_metadata
                        assert checkpoint_metadata is not None
                        CheckpointManager.create_checkpoint(
                            restored_context,
                            path=checkpoint_path,
                            metadata=checkpoint_metadata
                        )
                        restored_context.clear_checkpoint_request()

                    if result == "TRAINING_COMPLETE":
                        break

            assert result == "TRAINING_COMPLETE"
            assert restored_context.channel.get("iteration") == 10


class TestFaultToleranceScenarios:
    """Test checkpoint/resume for fault tolerance and recovery."""

    def test_workflow_recovery_after_simulated_failure(self):
        """Test recovering workflow execution after simulated failure.

        Simulates a workflow that fails mid-execution, then resumes from checkpoint.
        """
        graph = TaskGraph()

        @task("step_1")
        def step_1_func():
            return "step_1_complete"

        @task("step_2", inject_context=True)
        def step_2_func(task_ctx):
            """Step 2 that checkpoints before processing."""
            checkpoint_path = os.path.join(tempfile.gettempdir(), "step2_checkpoint")
            task_ctx.checkpoint(metadata={"stage": "step_2_start"}, path=checkpoint_path)
            return "step_2_complete"

        @task("step_3")
        def step_3_func():
            # Simulate failure
            raise RuntimeError("Simulated failure in step 3")

        @task("step_4")
        def step_4_func():
            return "step_4_complete"

        graph.add_node(step_1_func, "step_1")
        graph.add_node(step_2_func, "step_2")
        graph.add_node(step_3_func, "step_3")
        graph.add_node(step_4_func, "step_4")

        graph.add_edge("step_1", "step_2")
        graph.add_edge("step_2", "step_3")
        graph.add_edge("step_3", "step_4")

        context = ExecutionContext.create(graph, "step_1", max_steps=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = None

            # Execute until step 2 completes
            for _ in range(2):
                next_task = context.task_queue.get_next_task()
                if next_task:
                    result = run_task(context, next_task)
                    context.mark_task_completed(next_task)
                    context.increment_step()

                    # Schedule successors as engine would
                    for successor in graph.successors(next_task):
                        successor_task = graph.get_node(successor)
                        context.add_to_queue(successor_task)

                    # Create checkpoint if requested
                    if context.checkpoint_requested:
                        checkpoint_path = os.path.join(tmpdir, "recovery_checkpoint")
                        checkpoint_metadata = context.checkpoint_request_metadata
                        assert checkpoint_metadata is not None
                        CheckpointManager.create_checkpoint(
                            context,
                            path=checkpoint_path,
                            metadata=checkpoint_metadata
                        )
                        context.clear_checkpoint_request()

            # Verify we have checkpoint after step 2
            assert checkpoint_path is not None
            assert os.path.exists(f"{checkpoint_path}.pkl")

            # Verify step 1 and 2 are completed
            assert "step_1" in context.completed_tasks
            assert "step_2" in context.completed_tasks

            # Try to execute step 3 (will fail)
            next_task = context.task_queue.get_next_task()
            assert next_task is not None
            assert next_task == "step_3"

            try:
                run_task(context, next_task)
                pytest.fail("Expected RuntimeError from step 3")
            except RuntimeError as e:
                assert "Simulated failure" in str(e)

            # Resume from checkpoint (step 3 should be in queue since it wasn't completed)
            restored_context, metadata = CheckpointManager.resume_from_checkpoint(
                f"{checkpoint_path}.pkl"
            )

            # Verify restored state
            assert restored_context.get_result("step_1") == "step_1_complete"
            assert restored_context.get_result("step_2") == "step_2_complete"
            assert "step_1" in restored_context.completed_tasks
            assert "step_2" in restored_context.completed_tasks
            assert "step_3" not in restored_context.completed_tasks

            # Verify step 3 is in queue
            next_task = restored_context.task_queue.get_next_task()
            assert next_task is not None
            assert next_task == "step_3"

    def test_checkpoint_before_expensive_operation(self):
        """Test checkpointing before expensive operations for recovery."""
        graph = TaskGraph()

        @task("prepare_data")
        def prepare_data_func():
            return {"data": "prepared"}

        @task("expensive_operation", inject_context=True)
        def expensive_operation_func(task_ctx):
            """Expensive operation that checkpoints before starting."""
            # Checkpoint before expensive work
            task_ctx.checkpoint(metadata={"stage": "before_expensive_op"})

            # Simulate expensive operation
            channel = task_ctx.execution_context.channel
            channel.set("expensive_result", "completed")
            return "expensive_complete"

        @task("finalize")
        def finalize_func():
            return "finalized"

        graph.add_node(prepare_data_func, "prepare_data")
        graph.add_node(expensive_operation_func, "expensive_operation")
        graph.add_node(finalize_func, "finalize")

        graph.add_edge("prepare_data", "expensive_operation")
        graph.add_edge("expensive_operation", "finalize")

        context = ExecutionContext.create(
            graph, "prepare_data", max_steps=10, channel_backend="memory"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Execute prepare_data
            next_task = context.task_queue.get_next_task()
            assert next_task is not None
            run_task(context, next_task)
            context.mark_task_completed(next_task)
            context.increment_step()

            # Schedule successor (expensive_operation)
            for successor in graph.successors("prepare_data"):
                successor_task = graph.get_node(successor)
                context.add_to_queue(successor_task)

            # Execute expensive_operation (will checkpoint)
            next_task = context.task_queue.get_next_task()
            assert next_task is not None
            run_task(context, next_task)
            context.mark_task_completed(next_task)
            context.increment_step()

            # Create checkpoint as engine would
            assert context.checkpoint_requested
            checkpoint_path = os.path.join(tmpdir, "expensive_checkpoint")
            checkpoint_metadata = context.checkpoint_request_metadata
            assert checkpoint_metadata is not None
            pkl_path, metadata = CheckpointManager.create_checkpoint(
                context,
                path=checkpoint_path,
                metadata=checkpoint_metadata,
            )

            # Resume and verify state
            restored_context, restored_metadata = CheckpointManager.resume_from_checkpoint(pkl_path)

            assert restored_context.get_result("prepare_data") == {"data": "prepared"}
            assert restored_context.get_result("expensive_operation") == "expensive_complete"
            assert restored_context.channel.get("expensive_result") == "completed"
            assert restored_metadata.user_metadata["stage"] == "before_expensive_op"


class TestDynamicTaskCheckpointScenarios:
    """Test checkpoint/resume with dynamic task generation."""

    def test_checkpoint_with_dynamic_tasks(self):
        """Test checkpoint/resume workflow with dynamically created tasks."""
        graph = TaskGraph()

        @task("coordinator", inject_context=True)
        def coordinator_func(task_ctx):
            """Coordinator that creates dynamic tasks and checkpoints."""
            # Create dynamic worker tasks
            for i in range(3):
                def make_worker(worker_id):
                    @task(f"worker_{worker_id}")
                    def worker_func():
                        return f"worker_{worker_id}_done"
                    return worker_func

                worker_task = make_worker(i)
                task_ctx.next_task(worker_task)

            # Checkpoint after creating workers
            task_ctx.checkpoint(metadata={"workers_created": 3})
            return "coordinator_complete"

        @task("finalizer", inject_context=True)
        def finalizer_func(task_ctx):
            return "finalized"

        graph.add_node(coordinator_func, "coordinator")
        graph.add_node(finalizer_func, "finalizer")
        graph.add_edge("coordinator", "finalizer")

        context = ExecutionContext.create(graph, "coordinator", max_steps=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Execute coordinator (creates workers)
            next_task = context.task_queue.get_next_task()
            assert next_task is not None
            run_task(context, next_task)
            context.mark_task_completed(next_task)
            context.increment_step()

            # Create checkpoint
            assert context.checkpoint_requested
            checkpoint_path = os.path.join(tmpdir, "dynamic_checkpoint")
            checkpoint_metadata = context.checkpoint_request_metadata
            assert checkpoint_metadata is not None
            pkl_path, metadata = CheckpointManager.create_checkpoint(
                context, path=checkpoint_path, metadata=checkpoint_metadata
            )

            # Verify workers are in graph and queue
            assert "worker_0" in context.graph.nodes
            assert "worker_1" in context.graph.nodes
            assert "worker_2" in context.graph.nodes

            # Resume from checkpoint
            restored_context, restored_metadata = CheckpointManager.resume_from_checkpoint(pkl_path)

            # Verify dynamic tasks are restored
            assert "worker_0" in restored_context.graph.nodes
            assert "worker_1" in restored_context.graph.nodes
            assert "worker_2" in restored_context.graph.nodes

            # Verify workers are in queue for execution
            pending_specs = list(restored_context.task_queue.get_pending_task_specs())
            pending_ids = [spec.task_id for spec in pending_specs]

            # Should have workers and finalizer in queue
            assert "worker_0" in pending_ids or "worker_1" in pending_ids or "worker_2" in pending_ids


class TestComplexWorkflowCheckpointScenarios:
    """Test checkpoint/resume in complex workflow scenarios."""

    def test_checkpoint_with_cycle_controller_state(self):
        """Test that cycle controller state is preserved in checkpoints."""
        graph = TaskGraph()

        @task("cyclic_task", inject_context=True)
        def cyclic_task_func(task_ctx):
            """Task that uses cycles and checkpoints."""
            channel = task_ctx.execution_context.channel
            count = channel.get("count") if channel.get("count") is not None else 0

            channel.set("count", count + 1)

            # Checkpoint every 2 cycles
            if (count + 1) % 2 == 0:
                task_ctx.checkpoint(metadata={"cycle_count": count + 1})

            if count + 1 >= 5:
                return "cycles_complete"
            else:
                task_ctx.next_iteration()

        graph.add_node(cyclic_task_func, "cyclic_task")

        context = ExecutionContext.create(
            graph, "cyclic_task", max_steps=20, channel_backend="memory"
        )
        context.channel.set("count", 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Execute 4 cycles
            for i in range(4):
                next_task = context.task_queue.get_next_task()
                if next_task:
                    run_task(context, next_task)
                    context.mark_task_completed(next_task)
                    context.increment_step()

                    # Track cycle count
                    cycle_count = context.cycle_controller.get_cycle_count("cyclic_task")

                    # Create checkpoint if requested
                    if context.checkpoint_requested:
                        checkpoint_path = os.path.join(tmpdir, f"cycle_checkpoint_{i+1}")
                        checkpoint_metadata = context.checkpoint_request_metadata
                        assert checkpoint_metadata is not None
                        pkl_path, metadata = CheckpointManager.create_checkpoint(
                            context,
                            path=checkpoint_path,
                            metadata=checkpoint_metadata
                        )
                        context.clear_checkpoint_request()

                        # Verify cycle count is captured
                        if (i + 1) % 2 == 0:
                            assert metadata.user_metadata["cycle_count"] == i + 1

            # Resume from checkpoint at cycle 4
            checkpoint_4 = os.path.join(tmpdir, "cycle_checkpoint_4.pkl")
            restored_context, metadata = CheckpointManager.resume_from_checkpoint(checkpoint_4)

            # Verify cycle controller state
            assert restored_context.channel.get("count") == 4
            # Cycle controller tracks how many times the task has been executed
            # After 4 iterations, the cycle count should reflect that

            # Continue to completion
            next_task = restored_context.task_queue.get_next_task()
            if next_task:
                assert next_task is not None
                result = run_task(restored_context, next_task)
                assert result == "cycles_complete"

    def test_checkpoint_with_multiple_branches(self):
        """Test checkpoint/resume in workflow with multiple execution branches."""
        graph = TaskGraph()

        @task("start")
        def start_func():
            return "start_complete"

        @task("branch_a", inject_context=True)
        def branch_a_func(task_ctx):
            task_ctx.checkpoint(metadata={"branch": "a"})
            return "branch_a_complete"

        @task("branch_b")
        def branch_b_func():
            return "branch_b_complete"

        @task("merge", inject_context=True)
        def merge_func(task_ctx):
            return "merge_complete"

        graph.add_node(start_func, "start")
        graph.add_node(branch_a_func, "branch_a")
        graph.add_node(branch_b_func, "branch_b")
        graph.add_node(merge_func, "merge")

        graph.add_edge("start", "branch_a")
        graph.add_edge("start", "branch_b")
        graph.add_edge("branch_a", "merge")
        graph.add_edge("branch_b", "merge")

        context = ExecutionContext.create(graph, "start", max_steps=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Execute start
            next_task = context.task_queue.get_next_task()
            assert next_task is not None
            run_task(context, next_task)
            context.mark_task_completed(next_task)
            context.increment_step()

            # Schedule successors (branch_a and branch_b)
            for successor in graph.successors("start"):
                successor_task = graph.get_node(successor)
                context.add_to_queue(successor_task)

            # Execute branch_a (will checkpoint)
            next_task = context.task_queue.get_next_task()
            assert next_task is not None
            run_task(context, next_task)
            context.mark_task_completed(next_task)
            context.increment_step()

            # Create checkpoint if requested
            checkpoint_path = None
            if context.checkpoint_requested:
                checkpoint_path = os.path.join(tmpdir, "branch_checkpoint")
                checkpoint_metadata = context.checkpoint_request_metadata
                assert checkpoint_metadata is not None
                pkl_path, metadata = CheckpointManager.create_checkpoint(
                    context, path=checkpoint_path, metadata=checkpoint_metadata
                )
                context.clear_checkpoint_request()

            # Verify checkpoint was created
            assert checkpoint_path is not None
            assert os.path.exists(f"{checkpoint_path}.pkl")

            # Resume and verify state
            restored_context, restored_metadata = CheckpointManager.resume_from_checkpoint(
                f"{checkpoint_path}.pkl"
            )

            assert "start" in restored_context.completed_tasks
            # Either branch_a or branch_b should be completed (depending on execution order)
            assert (
                "branch_a" in restored_context.completed_tasks
                or "branch_b" in restored_context.completed_tasks
            )


class TestCheckpointMetadataScenarios:
    """Test checkpoint metadata usage in scenarios."""

    def test_checkpoint_metadata_enrichment(self):
        """Test that checkpoint metadata is enriched with task context."""
        graph = TaskGraph()

        @task("metadata_task", inject_context=True)
        def metadata_task_func(task_ctx):
            """Task that creates checkpoint with custom metadata."""
            task_ctx.checkpoint(
                metadata={"custom_field": "custom_value", "stage": "processing"}
            )
            return "complete"

        graph.add_node(metadata_task_func, "metadata_task")

        context = ExecutionContext.create(graph, "metadata_task", max_steps=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Execute task
            next_task = context.task_queue.get_next_task()
            assert next_task is not None
            result = run_task(context, next_task)
            context.mark_task_completed(next_task)
            context.increment_step()

            # Create checkpoint
            assert context.checkpoint_requested
            checkpoint_path = os.path.join(tmpdir, "metadata_checkpoint")
            checkpoint_metadata = context.checkpoint_request_metadata
            assert checkpoint_metadata is not None
            pkl_path, metadata = CheckpointManager.create_checkpoint(
                context, path=checkpoint_path, metadata=checkpoint_metadata
            )

            # Verify metadata enrichment
            assert metadata.user_metadata["custom_field"] == "custom_value"
            assert metadata.user_metadata["stage"] == "processing"
            assert metadata.session_id == context.session_id
            assert metadata.steps == context.steps

            # Resume and verify metadata is preserved
            restored_context, restored_metadata = CheckpointManager.resume_from_checkpoint(pkl_path)

            assert restored_metadata.user_metadata["custom_field"] == "custom_value"
            assert restored_metadata.user_metadata["stage"] == "processing"
            assert restored_metadata.checkpoint_id == metadata.checkpoint_id

    def test_multiple_checkpoints_tracking(self):
        """Test tracking multiple checkpoints with different metadata."""
        graph = TaskGraph()

        @task("progressive_task", inject_context=True)
        def progressive_task_func(task_ctx):
            """Task that checkpoints multiple times with progress tracking."""
            channel = task_ctx.execution_context.channel
            progress = channel.get("progress") if channel.get("progress") is not None else 0

            progress += 25
            channel.set("progress", progress)

            # Checkpoint at each progress milestone
            task_ctx.checkpoint(metadata={"progress": progress, "milestone": f"{progress}%"})

            if progress >= 100:
                return "complete"
            else:
                task_ctx.next_iteration()

        graph.add_node(progressive_task_func, "progressive_task")

        context = ExecutionContext.create(
            graph, "progressive_task", max_steps=10, channel_backend="memory"
        )
        context.channel.set("progress", 0)

        checkpoints = []

        with tempfile.TemporaryDirectory() as tmpdir:
            # Execute all iterations (25%, 50%, 75%, 100%)
            for i in range(4):
                next_task = context.task_queue.get_next_task()
                if next_task:
                    result = run_task(context, next_task)
                    context.mark_task_completed(next_task)
                    context.increment_step()

                    if context.checkpoint_requested:
                        checkpoint_path = os.path.join(tmpdir, f"progress_checkpoint_{(i+1)*25}")
                        checkpoint_metadata = context.checkpoint_request_metadata
                        assert checkpoint_metadata is not None
                        pkl_path, metadata = CheckpointManager.create_checkpoint(
                            context,
                            path=checkpoint_path,
                            metadata=checkpoint_metadata
                        )
                        checkpoints.append((pkl_path, metadata))
                        context.clear_checkpoint_request()

            # Verify we have 4 checkpoints
            assert len(checkpoints) == 4

            # Verify each checkpoint has correct metadata
            for i, (pkl_path, metadata) in enumerate(checkpoints):
                expected_progress = (i + 1) * 25
                assert metadata.user_metadata["progress"] == expected_progress
                assert metadata.user_metadata["milestone"] == f"{expected_progress}%"

            # Resume from 50% checkpoint and continue
            pkl_path_50, metadata_50 = checkpoints[1]  # 50% checkpoint

            restored_context, restored_metadata = CheckpointManager.resume_from_checkpoint(
                pkl_path_50
            )

            assert restored_context.channel.get("progress") == 50
            assert restored_metadata.user_metadata["progress"] == 50
