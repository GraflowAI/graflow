"""HITL API Server Example.

This example demonstrates how to run a FastAPI server for providing feedback
to running workflows via HTTP API.

Requirements:
    pip install fastapi uvicorn redis

Usage:
    1. Start Redis:
       docker run -p 6379:6379 redis:7.2

    2. Run this API server:
       python examples/11_hitl/04_api_server.py

    3. Run a workflow that requests feedback (in another terminal):
       python examples/11_hitl/05_distributed_feedback.py

    4. Provide feedback via API:
       curl -X POST http://localhost:8000/api/feedback/{feedback_id}/respond \\
         -H "Content-Type: application/json" \\
         -d '{"approved": true, "reason": "Looks good!"}'
"""

from __future__ import annotations


def main():
    """Run the feedback API server."""
    try:
        import redis

        from graflow.api.app import create_feedback_api
        from graflow.hitl.manager import FeedbackManager
    except ImportError as e:
        print(f"Error: {e}")
        print("Install requirements: pip install fastapi uvicorn redis")
        return

    print("=" * 80)
    print("HITL API Server Example")
    print("=" * 80)

    # Create Redis client
    print("\n[1] Connecting to Redis...")
    try:
        redis_client = redis.Redis(
            host="localhost", port=6379, db=0, decode_responses=True
        )
        redis_client.ping()
        print("    ✓ Connected to Redis")
    except Exception as e:
        print(f"    ✗ Failed to connect to Redis: {e}")
        print("    Start Redis with: docker run -p 6379:6379 redis:7.2")
        return

    # Create feedback manager with Redis backend
    print("\n[2] Creating FeedbackManager with Redis backend...")
    feedback_manager = FeedbackManager(
        backend="redis", backend_config={"redis_client": redis_client}
    )
    print("    ✓ FeedbackManager created")

    # Create FastAPI app
    print("\n[3] Creating FastAPI application...")
    app = create_feedback_api(feedback_manager=feedback_manager)
    print("    ✓ FastAPI app created")

    # Show available endpoints
    print("\n[4] Available endpoints:")
    print("    GET  /api/feedback              - List pending feedback requests")
    print("    GET  /api/feedback/{id}         - Get feedback details")
    print("    POST /api/feedback/{id}/respond - Provide feedback response")
    print("    DELETE /api/feedback/{id}       - Cancel feedback request")

    print("\n" + "=" * 80)
    print("Starting API server on http://localhost:8000")
    print("=" * 80)
    print("\nAPI Docs: http://localhost:8000/docs")
    print("Press Ctrl+C to stop\n")

    # Run server
    try:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except ImportError:
        print("Error: uvicorn not installed")
        print("Install with: pip install uvicorn")
    except KeyboardInterrupt:
        print("\n\nServer stopped")


if __name__ == "__main__":
    main()
