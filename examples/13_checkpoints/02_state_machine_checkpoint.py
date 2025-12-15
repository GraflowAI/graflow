"""State Machine with Checkpoints Example.

This example demonstrates checkpoint/resume in a state-based workflow.
The workflow processes an order through multiple states:
NEW → VALIDATED → PAID → SHIPPED

Each state transition creates a checkpoint, enabling recovery at any stage.

Key Concepts:
- Channel-based state machine for idempotent task execution
- Checkpoint at each state transition
- Resume continues from saved state
- Production-ready pattern for approval workflows

Use Cases:
- Order processing workflows
- Multi-stage approval processes
- Pipeline workflows with validation gates
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
    """Run state machine checkpoint demonstration."""
    print("=" * 70)
    print("State Machine with Checkpoints Example")
    print("Order Processing: NEW → VALIDATED → PAID → SHIPPED")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # ========================================
        # Part 1: Execute until first checkpoint
        # ========================================
        print("\n[Part 1] Starting order processing workflow...")
        print("-" * 70)

        with workflow("order_processing") as wf:

            @task(inject_context=True)
            def process_order(task_ctx: TaskExecutionContext):
                """Process order through state machine with checkpoints.

                State transitions:
                1. NEW → VALIDATED (validate order)
                2. VALIDATED → PAID (process payment)
                3. PAID → SHIPPED (ship order)

                Each transition creates a checkpoint for recovery.
                """
                # Access channel
                channel = task_ctx.get_channel()
                state = channel.get("order_state", "NEW")
                order_data = channel.get("order_data")

                # Initialize order data if not set (first execution)
                if not order_data:
                    order_data = {
                        "id": "ORD-2024-001",
                        "amount": 99.99,
                        "customer": "alice@example.com"
                    }
                    channel.set("order_data", order_data)
                    channel.set("order_state", "NEW")

                print(f"\n📦 Processing order in state: {state}")
                print(f"   Order ID: {order_data.get('id')}")
                print(f"   Amount: ${order_data.get('amount')}")

                if state == "NEW":
                    # State 1: Validate order
                    print("   🔍 Validating order...")
                    time.sleep(0.1)  # Simulate validation

                    if order_data and order_data.get("amount", 0) > 0:
                        channel.set("order_state", "VALIDATED")
                        channel.set("validated_at", time.time())
                        print("   ✓ Order validated successfully")

                        # Checkpoint after validation
                        checkpoint_path = os.path.join(tmpdir, "checkpoint_validated")
                        task_ctx.checkpoint(path=checkpoint_path, metadata={
                            "stage": "validation_complete",
                            "order_id": order_data["id"]
                        })
                        print("   📸 Checkpoint requested: validation_complete")

                        # Re-queue self for next state
                        task_ctx.next_iteration()
                    else:
                        print("   ✗ Invalid order")
                        return "ORDER_INVALID"

                elif state == "VALIDATED":
                    # State 2: Process payment
                    print("   💳 Processing payment...")
                    time.sleep(0.1)  # Simulate payment processing

                    channel.set("order_state", "PAID")
                    channel.set("paid_at", time.time())
                    print("   ✓ Payment processed successfully")

                    # Checkpoint after payment
                    checkpoint_path = os.path.join(tmpdir, "checkpoint_paid")
                    task_ctx.checkpoint(path=checkpoint_path, metadata={
                        "stage": "payment_complete",
                        "order_id": order_data["id"],
                        "amount": order_data["amount"]
                    })
                    print("   📸 Checkpoint requested: payment_complete")

                    # Re-queue self for next state
                    task_ctx.next_iteration()

                elif state == "PAID":
                    # State 3: Ship order
                    print("   📦 Shipping order...")
                    time.sleep(0.1)  # Simulate shipping

                    channel.set("order_state", "SHIPPED")
                    channel.set("shipped_at", time.time())
                    print("   ✓ Order shipped successfully")

                    return "ORDER_COMPLETE"

                return None

            # Execute workflow and get context (initial data set in task)
            print("\nExecuting workflow...")
            _result, context = wf.execute("process_order", ret_context=True)

            print("\n✓ First execution completed")
            print(f"  State: {context.channel.get('order_state')}")
            print(f"  Last checkpoint: {context.last_checkpoint_path}")

        # ========================================
        # Part 2: Resume from first checkpoint
        # ========================================
        print("\n" + "=" * 70)
        print("[Part 2] Resuming from checkpoint 1...")
        print("-" * 70)

        # Resume from first checkpoint
        checkpoint_1_path = os.path.join(tmpdir, "checkpoint_validated.pkl")
        restored_context, restored_metadata = CheckpointManager.resume_from_checkpoint(checkpoint_1_path)

        print("\n✓ Restored from checkpoint")
        print(f"  Stage: {restored_metadata.user_metadata['stage']}")
        print(f"  State: {restored_context.channel.get('order_state')}")

        # Continue execution (VALIDATED → PAID)
        print("\n▶ Continuing execution...")
        engine = WorkflowEngine()
        _result = engine.execute(restored_context)

        print("\n✓ Second execution completed")
        print(f"  State: {restored_context.channel.get('order_state')}")
        print(f"  Last checkpoint: {restored_context.last_checkpoint_path}")

        # ========================================
        # Part 3: Resume from second checkpoint and complete
        # ========================================
        print("\n" + "=" * 70)
        print("[Part 3] Resuming from checkpoint 2 and completing order...")
        print("-" * 70)

        # Resume from second checkpoint
        checkpoint_2_path = os.path.join(tmpdir, "checkpoint_paid.pkl")
        final_context, final_metadata = CheckpointManager.resume_from_checkpoint(checkpoint_2_path)

        print("\n✓ Restored from checkpoint")
        print(f"  Stage: {final_metadata.user_metadata['stage']}")
        print(f"  State: {final_context.channel.get('order_state')}")

        # Complete workflow (PAID → SHIPPED)
        print("\n▶ Completing workflow...")
        final_result = engine.execute(final_context)

        # ========================================
        # Summary
        # ========================================
        print("\n" + "=" * 70)
        print("Summary")
        print("-" * 70)

        order_data = final_context.channel.get("order_data")
        final_state = final_context.channel.get("order_state")

        print(f"\nOrder: {order_data['id']}")
        print(f"Amount: ${order_data['amount']}")
        print(f"Final State: {final_state}")
        print(f"Result: {final_result}")

        print("\n📊 Execution Timeline:")
        print("  1. NEW → VALIDATED [checkpoint 1]")
        print(f"     └─ Metadata: {restored_metadata.user_metadata}")
        print("  2. VALIDATED → PAID [checkpoint 2]")
        print(f"     └─ Metadata: {final_metadata.user_metadata}")
        print("  3. PAID → SHIPPED [complete]")

        print("\n💡 Key Benefit:")
        print("   If workflow crashes at any point, resume from last checkpoint")
        print("   and continue from that state - no need to restart from beginning!")

        print("\n✅ State machine workflow completed successfully!")


if __name__ == "__main__":
    main()
