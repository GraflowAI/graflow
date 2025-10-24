"""
FastAPI Backend for GPT Newspaper
==================================

Provides REST API endpoints for newspaper generation using graflow workflows.

Prerequisites:
--------------
- TAVILY_API_KEY environment variable
- OPENAI_API_KEY (or other LLM provider API key)

Run server:
-----------
uvicorn api:app --reload --port 8000

Endpoints:
----------
GET  /              - Health check
POST /api/generate  - Generate newspaper from topics
GET  /outputs/{path} - Serve generated newspaper files
"""

import time
from pathlib import Path
from typing import List

from config import Config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from newspaper_workflow import run_newspaper_workflow
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI(
    title="GPT Newspaper API",
    description="Generate personalized newspapers using AI agents",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and outputs (if available)
frontend_static_dir = Path(__file__).parent / "frontend" / "static"
if frontend_static_dir.exists():
    app.mount("/frontend/static", StaticFiles(directory=frontend_static_dir), name="static")
else:
    print("⚠️  Warning: Frontend static assets not found; skipping /frontend/static mount")

outputs_dir = Path("outputs")
outputs_dir.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")


# Pydantic Models
class NewspaperRequest(BaseModel):
    """Request model for newspaper generation."""

    topics: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of topics to include in the newspaper",
        example=["AI developments", "Climate change"]
    )
    layout: str = Field(
        default="layout_1.html",
        description="Layout template to use",
        example="layout_1.html"
    )
    max_workers: int | None = Field(
        default=None,
        description="Maximum number of parallel workers for article processing",
        ge=1,
        le=10
    )


class NewspaperResponse(BaseModel):
    """Response model for newspaper generation."""

    path: str = Field(
        ...,
        description="Path to the generated newspaper HTML file",
        example="/outputs/run_1234567890/newspaper.html"
    )
    article_count: int = Field(
        ...,
        description="Number of articles generated",
        example=3
    )
    timestamp: int = Field(
        ...,
        description="Unix timestamp when newspaper was generated",
        example=1234567890
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok", example="ok")
    message: str = Field(default="GPT Newspaper API is running", example="GPT Newspaper API is running")


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse with status information
    """
    # Check for required environment variables
    if not Config.TAVILY_API_KEY:
        return HealthResponse(
            status="warning",
            message="TAVILY_API_KEY not configured"
        )

    if not Config.OPENAI_API_KEY and "api_base" not in Config.DEFAULT_MODEL_PARAMS:
        return HealthResponse(
            status="warning",
            message="LLM credentials or API base not configured"
        )

    return HealthResponse(
        status="ok",
        message="GPT Newspaper API is running"
    )


@app.post("/api/generate", response_model=NewspaperResponse)
async def generate_newspaper(request: NewspaperRequest):
    """
    Generate a newspaper from given topics.

    Args:
        request: NewspaperRequest with topics and layout

    Returns:
        NewspaperResponse with path to generated newspaper

    Raises:
        HTTPException: If generation fails
    """
    # Validate environment variables
    if not Config.TAVILY_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="TAVILY_API_KEY environment variable is required"
        )

    if not Config.OPENAI_API_KEY and "api_base" not in Config.DEFAULT_MODEL_PARAMS:
        raise HTTPException(
            status_code=500,
            detail="LLM credentials not configured. Provide OPENAI_API_KEY or set "
                   "GPT_NEWSPAPER_MODEL_PARAMS with connection details."
        )

    try:
        # Generate newspaper using graflow workflow
        newspaper_path = run_newspaper_workflow(
            queries=request.topics,
            layout=request.layout,
            max_workers=request.max_workers
        )

        # Convert absolute path to relative URL path
        newspaper_path = Path(newspaper_path)
        relative_path = f"/{newspaper_path.relative_to(Path.cwd())}"

        # Count articles
        output_dir = newspaper_path.parent
        article_count = len(list(output_dir.glob("article_*.html")))

        return NewspaperResponse(
            path=relative_path,
            article_count=article_count,
            timestamp=int(time.time())
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate newspaper: {str(e)}"
        )


@app.get("/frontend")
async def serve_frontend():
    """
    Serve the frontend HTML page.

    Returns:
        HTML file response
    """
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if not frontend_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")

    return FileResponse(frontend_path)


# For testing
if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("🗞️  GPT NEWSPAPER API")
    print("=" * 80)
    print("\nStarting FastAPI server...")
    print("Frontend: http://localhost:8000/frontend")
    print("API docs: http://localhost:8000/docs")
    print("\n" + "=" * 80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
