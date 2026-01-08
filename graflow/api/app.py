"""FastAPI application factory for Graflow Feedback API."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from graflow.hitl.backend.base import FeedbackBackend
from graflow.hitl.manager import FeedbackManager


def create_feedback_api(
    feedback_backend: str | FeedbackBackend = "filesystem",
    feedback_config: Optional[dict] = None,
    title: str = "Graflow Feedback API",
    description: str = "Human-in-the-Loop feedback management API for Graflow workflows",
    version: str = "1.0.0",
    enable_cors: bool = True,
    cors_origins: Optional[list[str]] = None,
    enable_web_ui: bool = True,
) -> FastAPI:
    """Create FastAPI application with feedback endpoints.

    This factory function creates a FastAPI application with all feedback routes configured.
    The FeedbackManager is initialized with the specified backend and stored in app state.

    Args:
        feedback_backend: Backend type ("filesystem" or "redis") or FeedbackBackend instance
        feedback_config: Backend-specific configuration
            - filesystem: {"data_dir": "feedback_data"}
            - redis: {"redis_client": redis.Redis, "host": "localhost", "port": 6379, "db": 0}
        title: API title (shown in docs)
        description: API description (shown in docs)
        version: API version
        enable_cors: Whether to enable CORS middleware
        cors_origins: List of allowed CORS origins (default: ["*"])
        enable_web_ui: Whether to enable Web UI endpoints (default: True)

    Returns:
        FastAPI application with feedback routes

    Raises:
        ImportError: If FastAPI is not installed

    Example:
        Basic usage with filesystem backend:
        ```python
        from graflow.api.app import create_feedback_api

        app = create_feedback_api()

        # Run with: uvicorn app:app --host 0.0.0.0 --port 8000
        ```

        With Redis backend:
        ```python
        import redis
        from graflow.api.app import create_feedback_api

        redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )

        app = create_feedback_api(
            feedback_backend="redis",
            feedback_config={"redis_client": redis_client}
        )
        ```

        With custom CORS settings:
        ```python
        app = create_feedback_api(
            enable_cors=True,
            cors_origins=["http://localhost:3000", "https://myapp.com"]
        )
        ```
    """

    # Create FastAPI app
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Enable CORS if requested
    if enable_cors:
        origins = cors_origins or ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Initialize feedback manager
    feedback_config = feedback_config or {}
    feedback_manager = FeedbackManager(backend=feedback_backend, backend_config=feedback_config)

    # Store in app state
    app.state.feedback_manager = feedback_manager

    # Setup Web UI if enabled
    if enable_web_ui:
        # Setup Jinja2 templates
        template_dir = Path(__file__).parent / "templates"
        if template_dir.exists():
            templates = Jinja2Templates(directory=str(template_dir))
            app.state.templates = templates

            # Setup static files (optional)
            static_dir = Path(__file__).parent / "static"
            if static_dir.exists():
                app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

            # Include feedback UI router
            from graflow.api.endpoints.feedback_ui import router as feedback_ui_router

            app.include_router(feedback_ui_router)

    # Import and include API router
    from graflow.api.endpoints.feedback import router as feedback_router

    app.include_router(feedback_router)

    # Add root endpoint
    @app.get("/", summary="Root endpoint with API information", tags=["info"])
    async def root() -> dict:
        """Root endpoint with API information.

        Returns:
            Dictionary with API info and links
        """
        endpoints_info = {
            "api": {
                "list_feedback": "GET /api/feedback",
                "get_feedback": "GET /api/feedback/{feedback_id}",
                "respond_feedback": "POST /api/feedback/{feedback_id}/respond",
                "cancel_feedback": "DELETE /api/feedback/{feedback_id}",
            }
        }

        if enable_web_ui:
            endpoints_info["web_ui"] = {
                "feedback_list": "GET /ui/feedback/",
                "feedback_form": "GET /ui/feedback/{feedback_id}",
                "submit_feedback": "POST /ui/feedback/{feedback_id}/submit",
                "success_page": "GET /ui/feedback/{feedback_id}/success",
                "expired_page": "GET /ui/feedback/{feedback_id}/expired",
            }

        return {
            "name": title,
            "version": version,
            "description": description,
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "health": "/health",
            "web_ui_enabled": enable_web_ui,
            "endpoints": endpoints_info,
        }

    # Add health check endpoint
    @app.get("/health", summary="Health check endpoint", tags=["health"])
    async def health_check() -> dict:
        """Health check endpoint.

        Returns:
            Dictionary with status and backend info
        """
        backend_type = "unknown"
        if hasattr(feedback_manager._backend, "__class__"):
            backend_type = feedback_manager._backend.__class__.__name__

        return {"status": "healthy", "service": "graflow-feedback-api", "version": version, "backend": backend_type}

    return app


def create_feedback_api_with_redis(
    redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0, **kwargs
) -> FastAPI:
    """Convenience function to create feedback API with Redis backend.

    Args:
        redis_host: Redis host
        redis_port: Redis port
        redis_db: Redis database number
        **kwargs: Additional arguments passed to create_feedback_api()

    Returns:
        FastAPI application with Redis backend

    Example:
        ```python
        from graflow.api.app import create_feedback_api_with_redis

        app = create_feedback_api_with_redis(
            redis_host="localhost",
            redis_port=6379,
            redis_db=0
        )
        ```
    """
    try:
        import redis
    except ImportError:
        raise ImportError("Redis is required for Redis backend. Install with: uv sync --all-extras")

    redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)

    return create_feedback_api(feedback_backend="redis", feedback_config={"redis_client": redis_client}, **kwargs)


# Export main factory function
__all__ = ["create_feedback_api", "create_feedback_api_with_redis"]
