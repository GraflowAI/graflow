"""Test for deferred checkpoint resume bug.

Bug: When a task requests a deferred checkpoint via task_ctx.checkpoint(),
the checkpoint is created BEFORE successor tasks are added to the queue.
This means resuming from the checkpoint results in an empty queue,
and successor tasks are never executed.

Expected: After resuming from a deferred checkpoint, the engine should
execute remaining tasks (successors of the checkpointed task).
"""

import os
import tempfile

from graflow.core.checkpoint import CheckpointManager
from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.workflow import workflow


class TestDeferredCheckpointResume:
    """Test that deferred checkpoint includes successor tasks."""

    def test_deferred_checkpoint_resumes_with_successors(self):
        """Resuming from a deferred checkpoint should execute remaining tasks.

        Scenario: step_1 >> step_2 >> step_3
        - step_2 requests a deferred checkpoint
        - Checkpoint should include step_3 in the queue
        - Resuming should execute step_3 and return its result
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint")

            # === Part 1: Execute workflow with checkpoint ===
            with workflow("checkpoint_test") as wf:

                @task
                def step_1() -> str:
                    return "data_prepared"

                @task(inject_context=True)
                def step_2(task_ctx: TaskExecutionContext) -> str:
                    task_ctx.checkpoint(path=checkpoint_path, metadata={"stage": "after_step_2"})
                    return "data_processed"

                @task
                def step_3() -> str:
                    return "completed"

                _ = step_1 >> step_2 >> step_3
                _result, context = wf.execute("step_1", ret_context=True)

            # Verify initial execution completed all steps
            assert context.get_result("step_1") == "data_prepared"
            assert context.get_result("step_2") == "data_processed"
            assert context.get_result("step_3") == "completed"
            assert context.last_checkpoint_path is not None

            # === Part 2: Resume from checkpoint ===
            restored_ctx, _metadata = CheckpointManager.resume_from_checkpoint(context.last_checkpoint_path)

            # Verify checkpoint state: step_1 and step_2 completed
            assert "step_1" in restored_ctx.completed_tasks
            assert "step_2" in restored_ctx.completed_tasks
            assert "step_3" not in restored_ctx.completed_tasks
            assert restored_ctx.get_result("step_1") == "data_prepared"
            assert restored_ctx.get_result("step_2") == "data_processed"

            # Resume execution - step_3 should be executed
            engine = WorkflowEngine()
            final_result = engine.execute(restored_ctx)

            # BUG: step_3 result is None because it was never executed
            # EXPECTED: step_3 should execute and return "completed"
            assert restored_ctx.get_result("step_3") == "completed", (
                "step_3 should have been executed after resuming from checkpoint. "
                f"Got: {restored_ctx.get_result('step_3')}"
            )
            assert final_result == "completed"

    def test_deferred_checkpoint_queue_contains_successor(self):
        """Verify the checkpoint's queue contains successor tasks.

        This is the root cause test: check that the saved checkpoint
        has the successor task in its pending queue.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint")

            with workflow("queue_test") as wf:

                @task
                def task_a() -> str:
                    return "a_done"

                @task(inject_context=True)
                def task_b(task_ctx: TaskExecutionContext) -> str:
                    task_ctx.checkpoint(path=checkpoint_path)
                    return "b_done"

                @task
                def task_c() -> str:
                    return "c_done"

                _ = task_a >> task_b >> task_c
                _result, context = wf.execute("task_a", ret_context=True)

            # Resume and check queue state
            restored_ctx, _ = CheckpointManager.resume_from_checkpoint(context.last_checkpoint_path)  # type: ignore

            # The queue should contain task_c as a pending task
            pending = list(restored_ctx.task_queue.get_pending_task_specs())
            pending_ids = [spec.task_id for spec in pending]

            assert "task_c" in pending_ids, (
                f"task_c should be in pending queue after checkpoint resume. Pending tasks: {pending_ids}"
            )

            # === Resume execution from restored context ===
            engine = WorkflowEngine()
            final_result = engine.execute(restored_ctx)

            # Verify prior results are still intact
            assert restored_ctx.get_result("task_a") == "a_done"
            assert restored_ctx.get_result("task_b") == "b_done"

            # Verify task_c was executed from the restored context
            assert restored_ctx.get_result("task_c") == "c_done", (
                f"task_c should have been executed after resume. Got: {restored_ctx.get_result('task_c')}"
            )
            assert final_result == "c_done"
            assert "task_c" in restored_ctx.completed_tasks
