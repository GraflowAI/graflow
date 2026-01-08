"""CLI entrypoint for Graflow Feedback API server.

This module provides a command-line interface to launch the Feedback API server
using uvicorn with configurable backend and server options.

Usage:
    # Filesystem backend (default)
    python -m graflow.api.main --backend filesystem --data-dir feedback_data

    # Redis backend
    python -m graflow.api.main --backend redis --redis-host localhost --redis-port 6379

    # Custom host and port
    python -m graflow.api.main --host 0.0.0.0 --port 8080

    # With auto-reload for development
    python -m graflow.api.main --reload
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fastapi import FastAPI


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Command-line arguments (default: sys.argv[1:])

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Graflow Feedback API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with filesystem backend
  python -m graflow.api.main --backend filesystem --data-dir feedback_data

  # Start with Redis backend
  python -m graflow.api.main --backend redis --redis-host localhost --redis-port 6379

  # Development mode with auto-reload
  python -m graflow.api.main --reload

  # Custom host and port
  python -m graflow.api.main --host 0.0.0.0 --port 8080
        """,
    )

    # Backend options
    backend_group = parser.add_argument_group("Backend Options")
    backend_group.add_argument(
        "--backend",
        choices=["filesystem", "redis"],
        default="filesystem",
        help="Feedback backend type (default: filesystem)",
    )
    backend_group.add_argument(
        "--data-dir",
        type=str,
        default="feedback_data",
        help="Data directory for filesystem backend (default: feedback_data)",
    )
    backend_group.add_argument(
        "--redis-host", type=str, default="localhost", help="Redis host for redis backend (default: localhost)"
    )
    backend_group.add_argument(
        "--redis-port", type=int, default=6379, help="Redis port for redis backend (default: 6379)"
    )
    backend_group.add_argument(
        "--redis-db", type=int, default=0, help="Redis database number for redis backend (default: 0)"
    )
    backend_group.add_argument(
        "--redis-expiration-days", type=int, default=7, help="Days before Redis keys expire (default: 7)"
    )

    # Server options
    server_group = parser.add_argument_group("Server Options")
    server_group.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    server_group.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    server_group.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development (default: False)"
    )
    server_group.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    server_group.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default="info",
        help="Log level (default: info)",
    )

    # API options
    api_group = parser.add_argument_group("API Options")
    api_group.add_argument(
        "--title", type=str, default="Graflow Feedback API", help="API title (default: Graflow Feedback API)"
    )
    api_group.add_argument("--enable-cors", action="store_true", help="Enable CORS (default: False)")
    api_group.add_argument(
        "--cors-origins", type=str, nargs="+", default=["*"], help="CORS allowed origins (default: *)"
    )
    api_group.add_argument(
        "--disable-web-ui", action="store_true", help="Disable Web UI (API only mode) (default: False)"
    )

    return parser.parse_args(args)


def create_app_from_args(args: argparse.Namespace) -> FastAPI:
    """Create FastAPI app from parsed arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        FastAPI application instance
    """
    # Check if FastAPI is available
    try:
        from graflow.api.app import create_feedback_api
    except ImportError:
        print("ERROR: FastAPI is not installed.")
        print("Install with: uv sync --all-extras")
        sys.exit(1)

    # Build backend config
    if args.backend == "filesystem":
        backend_config = {"data_dir": args.data_dir}
    elif args.backend == "redis":
        # Import redis here to check availability
        try:
            import redis
        except ImportError:
            print("ERROR: Redis is not installed but redis backend was selected.")
            print("Install with: uv sync --all-extras")
            sys.exit(1)

        # Create Redis client
        redis_client = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db, decode_responses=True)

        # Test connection
        try:
            redis_client.ping()
            print(f"Connected to Redis at {args.redis_host}:{args.redis_port}")
        except redis.ConnectionError as e:
            print(f"ERROR: Cannot connect to Redis at {args.redis_host}:{args.redis_port}")
            print(f"Error: {e}")
            print("\nMake sure Redis is running:")
            print("  docker run -p 6379:6379 redis:7.2")
            sys.exit(1)

        backend_config = {"redis_client": redis_client, "expiration_days": args.redis_expiration_days}
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # Create app
    app = create_feedback_api(
        feedback_backend=args.backend,
        feedback_config=backend_config,
        title=args.title,
        enable_cors=args.enable_cors,
        cors_origins=args.cors_origins if args.enable_cors else None,
        enable_web_ui=not args.disable_web_ui,
    )

    print("\nGraflow Feedback API")
    print(f"Backend: {args.backend}")
    if args.backend == "filesystem":
        print(f"Data directory: {args.data_dir}")
    elif args.backend == "redis":
        print(f"Redis: {args.redis_host}:{args.redis_port} (db={args.redis_db})")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    if not args.disable_web_ui:
        print(f"Web UI: http://{args.host}:{args.port}/ui/feedback/{{feedback_id}}")
    print()

    return app


def main(args: Optional[list[str]] = None) -> None:
    """Main entrypoint for the API server.

    Args:
        args: Command-line arguments (default: sys.argv[1:])
    """
    # Parse arguments
    parsed_args = parse_args(args)

    # Create app
    app = create_app_from_args(parsed_args)

    # Check if uvicorn is available
    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is not installed.")
        print("Install with: uv sync --all-extras")
        sys.exit(1)

    # Run server
    uvicorn.run(
        app,
        host=parsed_args.host,
        port=parsed_args.port,
        reload=parsed_args.reload,
        workers=parsed_args.workers if not parsed_args.reload else 1,  # reload doesn't work with multiple workers
        log_level=parsed_args.log_level,
    )


if __name__ == "__main__":
    main()
