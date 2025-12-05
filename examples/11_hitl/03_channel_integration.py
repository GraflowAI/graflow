"""HITL with channel integration for inter-task communication.

This example demonstrates:
1. Writing feedback responses to channels
2. Reading feedback from channels in subsequent tasks
3. Parallel task coordination using feedback channels
"""

from __future__ import annotations

import logging
import threading
import time

from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.hitl.types import FeedbackResponse, FeedbackType

# Configure logging to show INFO level messages from graflow.hitl
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def main():
    """Demonstrate HITL with channel integration."""

    print("=" * 80)
    print("HITL Example 3: Channel Integration")
    print("=" * 80)

    with workflow("hitl_channels") as wf:

        @task(inject_context=True)
        def request_config(context):
            """Request configuration from human."""
            print("\n[Task 1] Requesting configuration...")

            # Provide feedback in background thread
            def provide_feedback():
                time.sleep(0.5)
                manager = context.execution_context.feedback_manager
                pending = manager.list_pending_requests()
                if pending:
                    feedback_id = pending[0].feedback_id
                    response = FeedbackResponse(
                        feedback_id=feedback_id,
                        response_type=FeedbackType.SELECTION,
                        selected="high_performance",
                    )
                    manager.provide_feedback(feedback_id, response)
                    print("[Human] Configuration selected: high_performance")

            thread = threading.Thread(target=provide_feedback, daemon=True)
            thread.start()

            # Request selection with channel integration
            try:
                response = context.request_feedback(
                    feedback_type="selection",
                    prompt="Select processing mode:",
                    options=["low_cost", "balanced", "high_performance"],
                    channel_key="processing_mode",  # Write to channel
                    write_to_channel=True,
                    timeout=3.0
                )

                print(f"[Task 1] Configuration received: {response.selected}")
                # Note: response.selected is also written to channel["processing_mode"]
                return response.selected

            except Exception as e:
                print(f"[Task 1] Error: {e}")
                return None

        @task(inject_context=True)
        def process_with_config(context):
            """Process using configuration from channel."""
            # Read configuration from channel (written by previous task)
            channel = context.get_channel()
            config = channel.get("processing_mode")

            print(f"\n[Task 2] Processing with mode: {config}")

            # Simulate processing based on configuration
            if config == "high_performance":
                print("[Task 2] Using parallel processing...")
                processing_time = 0.5
            elif config == "balanced":
                print("[Task 2] Using balanced resources...")
                processing_time = 1.0
            else:
                print("[Task 2] Using minimal resources...")
                processing_time = 1.5

            time.sleep(processing_time)
            print("[Task 2] ✓ Processing complete!")
            return {"mode": config, "duration": processing_time}

        @task(inject_context=True)
        def generate_report(context):
            """Generate final report."""
            channel = context.get_channel()
            config = channel.get("processing_mode")
            result = channel.get("process_with_config.__result__")

            print("\n[Task 3] Generating report...")
            print(f"[Task 3] Mode: {config}")
            print(f"[Task 3] Duration: {result['duration']:.2f}s")

            return {
                "configuration": config,
                "result": result,
                "timestamp": time.time()
            }

        # Define workflow
        request_config >> process_with_config >> generate_report

        # Execute
        print("\n" + "=" * 80)
        print("Starting workflow execution...")
        print("=" * 80)

        result = wf.execute()

        print("\n" + "=" * 80)
        print("Workflow completed!")
        print(f"Result: {result}")
        print("=" * 80)


if __name__ == "__main__":
    main()
