"""Fault Recovery with Checkpoints Example.

This example demonstrates using checkpoints for fault tolerance and recovery.
Simulates a workflow with potential failure points and shows how to recover.

Key Concepts:
- Checkpoint before expensive operations
- Recovery from failures
- Idempotent task execution
- Production fault tolerance patterns

Use Cases:
- Workflows with unreliable dependencies
- Expensive operations that shouldn't be repeated
- Production workflows requiring fault tolerance
- API calls with rate limits or quotas
"""

import os
import random
import tempfile
import time

from graflow.core.checkpoint import CheckpointManager
from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.workflow import workflow

# Simulate failure probability
FAILURE_MODE = True  # Set to True to simulate failures


def main():
    """Run fault recovery demonstration."""
    print("=" * 70)
    print("Fault Recovery with Checkpoints Example")
    print("Demonstrates checkpoint-based fault tolerance")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Track retry attempts
        max_retries = 3
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                if retry_count > 0:
                    print(f"\n{'='*70}")
                    print(f"Retry Attempt {retry_count}/{max_retries - 1}")
                    print(f"{'='*70}")

                # ========================================
                # Build workflow
                # ========================================
                if retry_count == 0:
                    print("\n[Setup] Building workflow pipeline...")
                    print("-" * 70)
                    print("✓ Pipeline: fetch → validate → process → finalize")
                    print("  - Checkpoint before validation")
                    print("  - Checkpoint after expensive processing")
                    print("  - Automatic retry on failure")

                # ========================================
                # Execute with fault tolerance
                # ========================================
                if retry_count == 0:
                    print("\n" + "=" * 70)
                    print("[Execution] Running workflow with fault tolerance...")
                    print("-" * 70)

                # Check if we have a checkpoint to resume from
                checkpoint_path = os.path.join(tmpdir, "fault_recovery_checkpoint.pkl")
                if retry_count > 0 and os.path.exists(checkpoint_path):
                    # Resume from checkpoint
                    print(f"\n🔄 Resuming from checkpoint (attempt {retry_count})...")
                    restored_context, restored_metadata = CheckpointManager.resume_from_checkpoint(checkpoint_path)
                    print(f"   Restored from stage: {restored_metadata.user_metadata.get('stage')}")

                    # Continue execution
                    engine = WorkflowEngine()
                    final_result = engine.execute(restored_context)
                    final_context = restored_context
                else:
                    # Execute from beginning
                    with workflow("fault_recovery") as wf:

                        @task
                        def fetch_data() -> dict:
                            """Fetch data from external source (always succeeds)."""
                            print("📥 Fetching data from external source...")
                            time.sleep(0.1)
                            data = {
                                "records": 1000,
                                "source": "database",
                                "timestamp": time.time()
                            }
                            print(f"   ✓ Fetched {data['records']} records")
                            return data

                        @task(inject_context=True)
                        def validate_data(task_ctx: TaskExecutionContext) -> dict:
                            """Validate data (checkpoint before expensive processing)."""
                            channel = task_ctx.get_channel()
                            data = channel.get("fetch_data.__result__")

                            print("\n🔍 Validating data...")

                            # Checkpoint before expensive validation
                            checkpoint_path = os.path.join(tmpdir, "fault_recovery_checkpoint")
                            task_ctx.checkpoint(path=checkpoint_path, metadata={
                                "stage": "before_validation",
                                "records": data["records"]
                            })
                            print("   📸 Checkpoint created before validation")

                            # Simulate validation
                            time.sleep(0.1)
                            validated_data = {
                                **data,
                                "validated": True,
                                "validation_time": time.time()
                            }

                            print(f"   ✓ Validated {data['records']} records")
                            return validated_data

                        @task(inject_context=True)
                        def expensive_processing(task_ctx: TaskExecutionContext) -> dict:
                            """Expensive processing with potential failure."""
                            channel = task_ctx.get_channel()
                            data = channel.get("validate_data.__result__")

                            # Check if we've already completed this step (idempotency)
                            processing_status = channel.get("processing_status", "pending")

                            if processing_status == "completed":
                                print("\n⏭  Processing already completed (idempotent check)")
                                return channel.get("processing_result")

                            print("\n⚙️  Starting expensive processing...")
                            print(f"   Processing {data['records']} records...")

                            # Simulate expensive operation
                            time.sleep(0.2)

                            # Simulate potential failure (50% chance on first try)
                            if FAILURE_MODE and random.random() < 0.5:
                                print("   ✗ ERROR: Infrastructure failure during processing!")
                                raise RuntimeError("Simulated processing failure")

                            # Processing succeeded
                            result = {
                                **data,
                                "processed": True,
                                "processing_time": time.time()
                            }

                            # Mark as completed for idempotency
                            channel.set("processing_status", "completed")
                            channel.set("processing_result", result)

                            print(f"   ✓ Successfully processed {data['records']} records")

                            # Checkpoint after expensive processing
                            checkpoint_path = os.path.join(tmpdir, "fault_recovery_checkpoint")
                            task_ctx.checkpoint(path=checkpoint_path, metadata={
                                "stage": "processing_complete",
                                "records": data["records"]
                            })
                            print("   📸 Checkpoint created after processing")

                            return result

                        @task
                        def finalize() -> str:
                            """Finalize the workflow."""
                            print("\n✅ Finalizing workflow...")
                            time.sleep(0.1)
                            print("   ✓ Workflow finalized")
                            return "SUCCESS"

                        # Define pipeline
                        fetch_data >> validate_data >> expensive_processing >> finalize  # type: ignore

                        # Execute workflow
                        final_result, final_context = wf.execute("fetch_data", ret_context=True)

                # Workflow completed successfully
                success = True

                # ========================================
                # Summary
                # ========================================
                print("\n" + "=" * 70)
                print("Summary")
                print("-" * 70)

                print("\nWorkflow Status: SUCCESS")
                print(f"Total Steps: {final_context.steps}")
                print(f"Retry Attempts: {retry_count}")
                print(f"Completed Tasks: {list(final_context.completed_tasks)}")

                print("\n📊 Results:")
                for task_id in ["fetch_data", "validate_data", "expensive_processing", "finalize"]:
                    result = final_context.get_result(task_id)
                    if result:
                        if isinstance(result, dict):
                            print(f"  - {task_id}: {result.get('records', 'N/A')} records")
                        else:
                            print(f"  - {task_id}: {result}")

                print("\n💡 Fault Tolerance Benefits:")
                print("  ✓ Automatic checkpoint before expensive operations")
                print("  ✓ Resume from checkpoint on failure (no re-fetch)")
                print("  ✓ Idempotent task execution (processing_status check)")
                print("  ✓ Automatic retry with checkpoint recovery")

                print("\n🎯 Key Pattern:")
                print("  1. Checkpoint before expensive/risky operations")
                print("  2. Use channel to track completion status")
                print("  3. Implement idempotency checks")
                print("  4. Resume from last successful checkpoint on failure")

                print("\n✅ Workflow completed successfully with fault tolerance!")

            except Exception as e:
                retry_count += 1
                print(f"\n❌ Workflow failed: {e}")
                if retry_count < max_retries:
                    print(f"   Will retry ({retry_count}/{max_retries})...")
                else:
                    print(f"\n❌ Max retries ({max_retries}) reached")
                    raise


if __name__ == "__main__":
    main()
