"""HITL Feedback API Example.

This example demonstrates workflow execution with feedback requests
that can be answered via REST API from another process.

Requirements:
    uv sync --all-extras

Usage:
    1. Start Redis:
       docker run -p 6379:6379 redis:7.2

    2. Start the API server (in another terminal):
       uv run python -m graflow.api --backend redis --redis-host localhost --redis-port 6379

    3. Run this workflow:
       uv run python examples/12_hitl/04_api_feedback.py [--redis-host HOST] [--redis-port PORT]

       Example with custom port:
       uv run python examples/12_hitl/04_api_feedback.py --redis-port 16379

    4. The workflow will wait for feedback. Provide it via API:
       # List pending requests
       curl http://localhost:8000/api/feedback

       # Provide feedback
       curl -X POST http://localhost:8000/api/feedback/{feedback_id}/respond \\
         -H "Content-Type: application/json" \\
         -d '{"approved": true, "reason": "Approved via API"}'

    5. The workflow will continue after receiving feedback
"""

from __future__ import annotations

import argparse
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
    """Demonstrate distributed feedback via REST API."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="HITL Feedback API Example")
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="Redis host (default: localhost)"
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (default: 6379)"
    )
    args = parser.parse_args()

    try:
        import redis
    except ImportError:
        print("Error: redis package not installed")
        print("Install with: pip install redis")
        return

    print("=" * 80)
    print("HITL Example 5: Distributed Feedback via REST API")
    print("=" * 80)

    # Connect to Redis
    print(f"\n[Setup] Connecting to Redis at {args.redis_host}:{args.redis_port}...")
    try:
        redis_client = redis.Redis(
            host=args.redis_host, port=args.redis_port, db=0, decode_responses=True
        )
        redis_client.ping()
        print("        ✓ Connected to Redis")
    except Exception as e:
        print(f"        ✗ Failed to connect to Redis: {e}")
        print(f"        Start Redis with: docker run -p {args.redis_port}:6379 redis:7.2")
        return

    print("\n[Setup] Make sure API server is running:")
    print(f"        python -m graflow.api --backend redis --redis-host {args.redis_host} --redis-port {args.redis_port}")
    print()

    # Create workflow
    # Note: workflow() context doesn't accept backend config
    # We'll configure it when creating ExecutionContext later
    with workflow("distributed_feedback") as wf:

        @task(inject_context=True)
        def prepare_deployment(context):
            """Prepare deployment package."""
            print("\n[Task 1] Preparing deployment package...")
            time.sleep(0.5)
            deployment_info = {
                "version": "v2.0.0",
                "environment": "production",
                "services": ["api", "web", "worker", "database"],
            }
            print(f"[Task 1] Deployment ready: {deployment_info}")
            return deployment_info

        @task(inject_context=True)
        def request_approval(context):
            """Request approval via REST API."""
            print("\n[Task 2] Requesting approval for deployment...")

            # Get deployment info
            channel = context.get_channel()
            deployment_info = channel.get("prepare_deployment.__result__")

            # Fallback if channel doesn't have the info
            if deployment_info is None:
                print("[Task 2] Warning: Could not retrieve deployment info from channel")
                deployment_info = {
                    "version": "v2.0.0",
                    "environment": "production"
                }

            print("[Task 2] Waiting for feedback via REST API...")
            print("[Task 2] Timeout: 300 seconds (5 minutes)")

            # Instructions for user
            print("\n" + "=" * 80)
            print("PROVIDE FEEDBACK VIA API:")
            print("=" * 80)
            print("\n1. List pending requests:")
            print("   curl http://localhost:8000/api/feedback\n")
            print("2. Get request details:")
            print("   curl http://localhost:8000/api/feedback/{feedback_id}\n")
            print("3. Provide approval:")
            print("   curl -X POST http://localhost:8000/api/feedback/{feedback_id}/respond \\")
            print('     -H "Content-Type: application/json" \\')
            print('     -d \'{"approved": true, "reason": "Approved via API"}\'')
            print("\n" + "=" * 80 + "\n")

            # Request feedback (will poll for 5 minutes)
            try:
                response = context.request_feedback(
                    feedback_type="approval",
                    prompt=f"Approve deployment {deployment_info['version']} to {deployment_info['environment']}?",
                    timeout=300.0,  # 5 minutes
                )

                if response.approved:
                    print("\n[Task 2] ✓ Deployment approved!")
                    print(f"[Task 2]   Reason: {response.reason}")
                    print(f"[Task 2]   Responded by: {response.responded_by}")
                    return True
                else:
                    print("\n[Task 2] ✗ Deployment rejected!")
                    print(f"[Task 2]   Reason: {response.reason}")
                    return False

            except Exception as e:
                print(f"\n[Task 2] Timeout: {e}")
                print("[Task 2] Checkpoint will be created...")
                raise

        @task(inject_context=True)
        def execute_deployment(context):
            """Execute deployment if approved."""
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
        _ = prepare_deployment >> request_approval >> execute_deployment

        # Execute with Redis backend
        print("\n" + "=" * 80)
        print("Starting workflow execution...")
        print("=" * 80)

        try:
            from graflow.core.context import ExecutionContext
            from graflow.core.engine import WorkflowEngine

            # Debug: Check graph structure
            print(f"[Debug] Graph nodes: {list(wf.graph._graph.nodes())}")
            print(f"[Debug] Start nodes: {wf.graph.get_start_nodes()}")

            # Create execution context with Redis backend
            start_node = wf.graph.get_start_nodes()[0] if wf.graph.get_start_nodes() else None
            print(f"[Debug] Using start_node: {start_node}")

            exec_context = ExecutionContext.create(
                wf.graph,
                start_node=start_node,
                channel_backend="redis",
                max_steps=10000,
                config={"redis_client": redis_client},
            )

            # Execute workflow
            engine = WorkflowEngine()
            result = engine.execute(exec_context)

            print("\n" + "=" * 80)
            print("Workflow completed!")
            print(f"Result: {result}")
            print("=" * 80)

        except Exception as e:
            print("\n" + "=" * 80)
            print(f"Workflow interrupted: {e}")
            print("=" * 80)


if __name__ == "__main__":
    main()
