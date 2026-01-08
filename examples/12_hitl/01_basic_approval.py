"""Basic HITL (Human-in-the-Loop) approval workflow.

This example demonstrates:
1. Requesting approval from a human during workflow execution
2. Handling immediate approval (within timeout)
3. Using FeedbackManager to provide feedback programmatically
"""

from __future__ import annotations

import logging
import threading
import time

from graflow.core.decorators import task
from graflow.core.workflow import workflow

# Configure logging to show INFO level messages from graflow.hitl
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def main():
    """Demonstrate basic HITL approval workflow."""

    print("=" * 80)
    print("HITL Example 1: Basic Approval Workflow")
    print("=" * 80)

    # Create workflow
    with workflow("hitl_approval") as wf:

        @task(inject_context=True)
        def prepare_deployment(context):
            """Prepare deployment package."""
            print("\n[Task 1] Preparing deployment package...")
            time.sleep(0.5)
            deployment_info = {"version": "v1.2.3", "environment": "production", "services": ["api", "web", "worker"]}
            print(f"[Task 1] Deployment ready: {deployment_info}")
            return deployment_info

        @task(inject_context=True)
        def request_approval(context):
            """Request human approval for deployment."""
            print("\n[Task 2] Requesting approval for deployment...")

            # Get deployment info from channel (written by previous task)
            channel = context.get_channel()
            deployment_info = channel.get("prepare_deployment.__result__")

            # In a real scenario, we'd simulate human feedback in another thread
            # For this example, we'll provide feedback programmatically
            def provide_feedback_after_delay():
                time.sleep(1.0)  # Simulate human thinking time
                manager = context.execution_context.feedback_manager
                pending = manager.list_pending_requests()
                if pending:
                    from graflow.hitl.types import FeedbackResponse, FeedbackType

                    feedback_id = pending[0].feedback_id
                    response = FeedbackResponse(
                        feedback_id=feedback_id,
                        response_type=FeedbackType.APPROVAL,
                        approved=True,
                        reason="Deployment looks good, approved!",
                        responded_by="human_operator",
                    )
                    manager.provide_feedback(feedback_id, response)
                    print("[Human] Approval provided!")

            # Start feedback provider thread
            thread = threading.Thread(target=provide_feedback_after_delay, daemon=True)
            thread.start()

            # Request approval (will block until feedback received or timeout)
            try:
                response = context.request_feedback(
                    feedback_type="approval",
                    prompt=f"Approve deployment {deployment_info['version']} to {deployment_info['environment']}?",
                    timeout=5.0,  # 5 second timeout for demo
                )

                if response.approved:
                    print("[Task 2] ✓ Deployment approved!")
                    return True
                else:
                    print("[Task 2] ✗ Deployment rejected!")
                    return False

            except Exception as e:
                print(f"[Task 2] Error: {e}")
                return False

        @task(inject_context=True)
        def execute_deployment(context):
            """Execute deployment if approved."""
            # Get approval result from channel
            channel = context.get_channel()
            approved = channel.get("request_approval.__result__")

            if approved:
                print("\n[Task 3] Executing deployment...")
                time.sleep(0.5)
                print("[Task 3] ✓ Deployment completed successfully!")
                return {"status": "success", "deployed_at": time.time()}
            else:
                print("\n[Task 3] Deployment cancelled (not approved)")
                return {"status": "cancelled"}

        # Define workflow
        prepare_deployment >> request_approval >> execute_deployment

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
