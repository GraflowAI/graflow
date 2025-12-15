"""Basic Checkpoint/Resume Example.

This example demonstrates the fundamental checkpoint/resume workflow:
1. Execute tasks and create a checkpoint
2. Resume from the checkpoint
3. Continue execution from saved state

Key Concepts:
- Checkpoint creation: task_ctx.checkpoint() sets flag, engine creates checkpoint automatically
- Resume: CheckpointManager.resume_from_checkpoint(path)
- Results preservation: Task results are saved in channel
- High-level API: Use workflow() context and wf.execute()
"""

import os
import tempfile

from graflow.core.checkpoint import CheckpointManager
from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow


def main():
    """Run basic checkpoint/resume demonstration."""
    print("=" * 60)
    print("Basic Checkpoint/Resume Example")
    print("=" * 60)

    # Create temporary directory for checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "basic_checkpoint")

        # ========================================
        # Part 1: Create workflow and checkpoint
        # ========================================
        print("\n[Part 1] Creating workflow and executing with checkpoint...")
        print("-" * 60)

        with workflow("basic_checkpoint") as wf:

            @task
            def step_1() -> str:
                """First step: data preparation."""
                print("Step 1: Preparing data...")
                return "data_prepared"

            @task(inject_context=True)
            def step_2(task_ctx: TaskExecutionContext) -> str:
                """Second step: processing with checkpoint."""
                print("Step 2: Processing data...")

                # Request checkpoint after processing
                # Note: Checkpoint created by engine AFTER this task completes
                task_ctx.checkpoint(path=checkpoint_path, metadata={"stage": "processing_complete"})

                return "data_processed"

            @task
            def step_3() -> str:
                """Third step: finalization."""
                print("Step 3: Finalizing...")
                return "completed"

            # Define sequential pipeline: step_1 -> step_2 -> step_3
            step_1 >> step_2 >> step_3  # type: ignore

            print("\nWorkflow Information:")
            print(f"Name: {wf.name}")
            print(f"Tasks: {list(wf.graph.nodes.keys())}")

            # Execute the workflow and get context
            print("\nExecuting workflow...")
            _result, context = wf.execute("step_1", ret_context=True)

            # Checkpoint was automatically created by engine
            print("\n📸 Checkpoint created automatically by engine")
            assert context.last_checkpoint_path is not None, "Expected checkpoint path but got None"
            print(f"  - Checkpoint path: {context.last_checkpoint_path}")
            print(f"  - Steps: {context.steps}")

            # Verify checkpoint files exist
            base_path = context.last_checkpoint_path.replace('.pkl', '')
            print("\n📁 Checkpoint files:")
            print(f"  - {base_path}.pkl: {os.path.exists(f'{base_path}.pkl')}")
            print(f"  - {base_path}.state.json: {os.path.exists(f'{base_path}.state.json')}")
            print(f"  - {base_path}.meta.json: {os.path.exists(f'{base_path}.meta.json')}")

        # ========================================
        # Part 2: Resume from checkpoint
        # ========================================
        print("\n" + "=" * 60)
        print("[Part 2] Resuming from checkpoint...")
        print("-" * 60)

        # Resume from checkpoint
        restored_context, restored_metadata = CheckpointManager.resume_from_checkpoint(
            context.last_checkpoint_path
        )

        print("\n✓ Checkpoint restored")
        print(f"  - Session ID: {restored_metadata.session_id}")
        print(f"  - Steps at checkpoint: {restored_metadata.steps}")
        print(f"  - Completed tasks: {list(restored_context.completed_tasks)}")

        # Verify results are preserved
        print("\n📦 Preserved results:")
        print(f"  - step_1 result: {restored_context.get_result('step_1')}")
        print(f"  - step_2 result: {restored_context.get_result('step_2')}")

        # Continue execution with restored context
        print("\n▶ Continuing execution from checkpoint...")
        from graflow.core.engine import WorkflowEngine
        engine = WorkflowEngine()
        final_result = engine.execute(restored_context)

        # ========================================
        # Summary
        # ========================================
        print("\n" + "=" * 60)
        print("Summary")
        print("-" * 60)
        print("Original execution: step_1 → step_2 → [checkpoint]")
        print("Resumed execution: [restore] → step_3")
        print("\nFinal results:")
        print(f"  - step_1: {restored_context.get_result('step_1')}")
        print(f"  - step_2: {restored_context.get_result('step_2')}")
        print(f"  - step_3: {restored_context.get_result('step_3')}")
        print(f"\n✓ Final result: {final_result}")
        print("\n✅ Checkpoint/resume workflow completed successfully!")


if __name__ == "__main__":
    main()
