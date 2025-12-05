"""HITL with timeout and checkpoint creation.

This example demonstrates:
1. Feedback timeout triggering automatic checkpoint creation
2. Checkpoint metadata containing feedback_id
3. How to resume workflow after feedback is provided externally

Note: This is a simplified example. In production, you would:
- Run the workflow in a separate process
- Provide feedback via REST API
- Resume from checkpoint when feedback is received
"""

from __future__ import annotations

import logging
import time

from graflow.core.decorators import task
from graflow.core.workflow import workflow

# Configure logging to show INFO level messages from graflow.hitl
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def main():
    """Demonstrate HITL timeout and checkpoint creation."""

    print("=" * 80)
    print("HITL Example 2: Timeout and Checkpoint")
    print("=" * 80)

    # Create workflow
    with workflow("hitl_checkpoint") as wf:

        @task(inject_context=True)
        def analyze_data(context):
            """Analyze data and prepare report."""
            print("\n[Task 1] Analyzing data...")
            time.sleep(0.5)
            report = {
                "anomalies_detected": 3,
                "confidence": 0.85,
                "recommendation": "manual_review"
            }
            print(f"[Task 1] Analysis complete: {report}")
            return report

        @task(inject_context=True)
        def request_review(context):
            """Request human review (will timeout for demo)."""
            print("\n[Task 2] Requesting human review...")

            # Get report from channel
            channel = context.get_channel()
            report = channel.get("analyze_data.__result__")
            print(f"[Task 2] Report: {report}")

            try:
                # This will timeout after 2 seconds (no feedback provided)
                response = context.request_feedback(
                    feedback_type="text",
                    prompt=f"Review findings: {report['anomalies_detected']} anomalies. Provide your assessment:",
                    timeout=2.0  # Short timeout for demo
                )

                print(f"[Task 2] Review received: {response.text}")
                return response.text

            except Exception as e:
                # This will be caught by WorkflowEngine and checkpoint created
                print(f"[Task 2] Timeout: {e}")
                raise

        @task(inject_context=True)
        def finalize_report(context):
            """Finalize report with human review."""
            # Get review from channel
            channel = context.get_channel()
            review = channel.get("request_review.__result__")
            print(f"\n[Task 3] Finalizing report with review: {review}")
            return {"status": "completed", "review": review}

        # Define workflow
        _ = analyze_data >> request_review >> finalize_report

        # Execute
        print("\n" + "=" * 80)
        print("Starting workflow execution...")
        print("=" * 80)

        try:
            result = wf.execute()
            print(f"\nWorkflow completed: {result}")
        except Exception as e:
            print(f"\n[Workflow] Execution interrupted: {type(e).__name__}")
            print("[Workflow] Checkpoint should have been created")
            print("[Workflow] In production:")
            print("  1. Checkpoint saves workflow state")
            print("  2. Human provides feedback via API")
            print("  3. Workflow resumes from checkpoint")
            print("  4. Task continues with feedback")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
