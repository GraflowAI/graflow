"""Periodic Checkpoints Example.

This example demonstrates periodic checkpointing in long-running workflows.
Simulates ML training with checkpoints every N iterations.

Key Concepts:
- Periodic checkpointing (e.g., every 5 epochs)
- Progress tracking via channel state
- Resume from latest checkpoint
- Convergence-based termination

Use Cases:
- ML model training
- Batch processing jobs
- Data pipelines
- Long-running computations
"""

import os
import tempfile
import time

from graflow.core.checkpoint import CheckpointManager
from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.workflow import workflow


def main():
    """Run periodic checkpoint demonstration."""
    print("=" * 70)
    print("Periodic Checkpoints Example")
    print("ML Training Simulation with Checkpoints Every 3 Epochs")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # ========================================
        # Part 1: Train with periodic checkpoints
        # ========================================
        print("\n[Part 1] Starting training with periodic checkpoints...")
        print("-" * 70)

        with workflow("ml_training") as wf:

            @task(inject_context=True)
            def training_loop(task_ctx: TaskExecutionContext):
                """Simulate ML training with periodic checkpoints.

                Training loop:
                - Run training epoch
                - Update metrics
                - Checkpoint every 3 epochs
                - Check convergence
                """
                channel = task_ctx.get_channel()
                epoch = channel.get("epoch", 0)
                max_epochs = 20
                checkpoint_frequency = 3  # Checkpoint every 3 epochs

                print(f"\n📈 Epoch {epoch + 1}/{max_epochs}", flush=True)

                # Simulate training step
                time.sleep(0.05)  # Simulate computation

                # Calculate simulated metrics (loss decreases over time)
                loss = 1.0 / (epoch + 1)
                accuracy = min(0.95, 0.5 + (epoch * 0.03))

                # Update state
                epoch += 1
                channel.set("epoch", epoch)
                channel.set("loss", loss)
                channel.set("accuracy", accuracy)

                print(f"   Loss: {loss:.4f}")
                print(f"   Accuracy: {accuracy:.4f}")

                # Periodic checkpoint
                if epoch % checkpoint_frequency == 0:
                    print(f"   📸 Checkpoint milestone (epoch {epoch})")
                    checkpoint_path = os.path.join(tmpdir, f"checkpoint_epoch_{epoch}")
                    task_ctx.checkpoint(
                        path=checkpoint_path,
                        metadata={"epoch": epoch, "loss": loss, "accuracy": accuracy, "milestone": f"epoch_{epoch}"},
                    )

                # Check convergence or cycle limit
                # Note: max_cycles defaults to 10, so stop at epoch 10 to stay within limit
                if accuracy >= 0.95:
                    print("   ✓ Target accuracy reached!")
                    return "TRAINING_COMPLETE"
                elif epoch >= 10:  # Stop at 10 epochs to stay within cycle limit
                    print(f"   ⏸ Pausing at epoch {epoch}")
                    return None  # Don't request next_iteration
                elif epoch >= max_epochs:
                    print("   ⚠ Max epochs reached")
                    return "MAX_EPOCHS"
                else:
                    # Continue training
                    task_ctx.next_iteration()
                    return None

            # Execute workflow - will run until convergence or max_cycles (10)
            # With max_cycles=10, workflow will run up to 10 iterations
            print("\nExecuting training workflow...")
            _result, context = wf.execute(training_loop.task_id, max_steps=50, ret_context=True)

            current_epoch = context.channel.get("epoch")
            print(f"\n✓ Training paused at epoch {current_epoch}")
            print(f"  Last checkpoint: {context.last_checkpoint_path}")

        # ========================================
        # Part 2: Resume from latest checkpoint
        # ========================================
        print("\n" + "=" * 70)
        print("[Part 2] Resuming from latest checkpoint...")
        print("-" * 70)

        # Get latest checkpoint
        # Find the latest checkpoint file
        checkpoint_files = sorted(
            [f for f in os.listdir(tmpdir) if f.startswith("checkpoint_epoch_") and f.endswith(".pkl")]
        )
        if checkpoint_files:
            latest_checkpoint_file = checkpoint_files[-1]
            checkpoint_path = os.path.join(tmpdir, latest_checkpoint_file)
        else:
            checkpoint_path = os.path.join(tmpdir, "checkpoint_epoch_10.pkl")

        print("\n📂 Resuming from checkpoint:")
        restored_context, restored_metadata = CheckpointManager.resume_from_checkpoint(checkpoint_path)
        print(f"  - Epoch: {restored_metadata.user_metadata['epoch']}")
        print(f"  - Loss: {restored_metadata.user_metadata['loss']:.4f}")
        print(f"  - Accuracy: {restored_metadata.user_metadata['accuracy']:.4f}")

        resumed_epoch = restored_context.channel.get("epoch")
        print(f"\n✓ Training restored at epoch {resumed_epoch}")

        # ========================================
        # Part 3: Continue training to completion
        # ========================================
        print("\n" + "=" * 70)
        print("[Part 3] Continuing training to completion...")
        print("-" * 70)

        # Continue training
        print("\n▶ Resuming training...")
        engine = WorkflowEngine()
        final_result = engine.execute(restored_context)

        # ========================================
        # Summary
        # ========================================
        print("\n" + "=" * 70)
        print("Summary")
        print("-" * 70)

        final_epoch = restored_context.channel.get("epoch")
        final_loss = restored_context.channel.get("loss")
        final_accuracy = restored_context.channel.get("accuracy")

        print("\n📊 Training Results:")
        print(f"  - Total epochs: {final_epoch}")
        print(f"  - Final loss: {final_loss:.4f}")
        print(f"  - Final accuracy: {final_accuracy:.4f}")
        print(f"  - Result: {final_result}")

        print("\n💾 Checkpoints Created:")
        # List all checkpoint files
        checkpoint_files = sorted(
            [f for f in os.listdir(tmpdir) if f.startswith("checkpoint_epoch_") and f.endswith(".pkl")]
        )
        for i, checkpoint_file in enumerate(checkpoint_files, 1):
            # Extract epoch number from filename
            epoch_num = int(checkpoint_file.split("_")[-1].replace(".pkl", ""))
            print(f"  {i}. Epoch {epoch_num:2d}")

        print("\n💡 Benefits of Periodic Checkpointing:")
        print("  ✓ Resume from any checkpoint if training is interrupted")
        print("  ✓ No need to retrain from scratch")
        print("  ✓ Can analyze model performance at different epochs")
        print("  ✓ Fault tolerance for long-running jobs")

        print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
