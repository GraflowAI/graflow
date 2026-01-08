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

import asyncio
import os
import signal
import sys
import threading
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple
from uuid import uuid4

from config import Config
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from newspaper_agent_workflow import run_newspaper_workflow as run_agent_newspaper_workflow
from newspaper_dynamic_workflow import run_newspaper_workflow as run_dynamic_newspaper_workflow
from newspaper_workflow import run_newspaper_workflow as run_original_newspaper_workflow
from pydantic import BaseModel, ConfigDict, Field

# Initialize FastAPI app
app = FastAPI(
    title="GPT Newspaper API", description="Generate personalized newspapers using AI agents", version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

outputs_dir = Path("outputs")
outputs_dir.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")


# Pydantic Models
SUPPORTED_LAYOUTS: dict[str, str] = {
    "single": "layout_1.html",
    "two-column": "layout_2.html",
}

SupportedWorkflow = Literal["original", "dynamic", "agent"]
DEFAULT_WORKFLOW: SupportedWorkflow = "original"
WORKFLOW_RUNNERS: Dict[SupportedWorkflow, Callable[..., str]] = {
    "original": run_original_newspaper_workflow,
    "dynamic": run_dynamic_newspaper_workflow,
    "agent": run_agent_newspaper_workflow,
}


class LogStreamManager:
    """Manages WebSocket subscribers per run and broadcasts log events."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Tuple[asyncio.Queue, asyncio.AbstractEventLoop]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, run_id: str, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._subscribers.setdefault(run_id, []).append((queue, loop))

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            subscribers = self._subscribers.get(run_id)
            if not subscribers:
                return
            self._subscribers[run_id] = [
                (stored_queue, stored_loop) for stored_queue, stored_loop in subscribers if stored_queue is not queue
            ]
            if not self._subscribers[run_id]:
                self._subscribers.pop(run_id)

    def _publish(self, run_id: str, payload: Dict[str, str]) -> None:
        if not run_id:
            return
        with self._lock:
            subscribers = list(self._subscribers.get(run_id, []))
        if not subscribers:
            return
        for queue, loop in subscribers:
            loop.call_soon_threadsafe(queue.put_nowait, payload)

    def log(self, run_id: str, message: str) -> None:
        if not message.strip():
            return
        payload = {
            "type": "log",
            "message": message,
            "runId": run_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._publish(run_id, payload)

    def status(self, run_id: str, status: str) -> None:
        payload = {
            "type": "status",
            "status": status,
            "runId": run_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._publish(run_id, payload)

    def complete(self, run_id: str) -> None:
        payload = {
            "type": "complete",
            "status": "finished",
            "runId": run_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._publish(run_id, payload)


log_stream_manager = LogStreamManager()


@contextmanager
def redirect_stdout_to_stream(run_id: str | None):
    """Mirror stdout/stderr writes to websocket subscribers for a given run."""

    if not run_id:
        yield
        return

    log_stream_manager.status(run_id, "started")

    original_stdout_write = sys.stdout.write
    original_stderr_write = sys.stderr.write

    def _broadcasting_writer(original_write):
        def _write(data: str):
            result = original_write(data)
            log_stream_manager.log(run_id, data)
            return result

        return _write

    sys.stdout.write = _broadcasting_writer(original_stdout_write)  # type: ignore[assignment]
    sys.stderr.write = _broadcasting_writer(original_stderr_write)  # type: ignore[assignment]

    try:
        yield
    finally:
        sys.stdout.write = original_stdout_write  # type: ignore[assignment]
        sys.stderr.write = original_stderr_write  # type: ignore[assignment]
        log_stream_manager.status(run_id, "completed")
        log_stream_manager.complete(run_id)


def _stdin_shutdown_listener():
    """Watch stdin for EOF (Ctrl+D) and trigger a graceful shutdown."""

    try:
        while True:
            chunk = sys.stdin.read(1)
            if chunk == "":
                os.kill(os.getpid(), signal.SIGINT)
                break
    except Exception:
        # stdin might not be available (e.g., running under a supervisor). Ignore.
        pass


class NewspaperRequest(BaseModel):
    """Request model for newspaper generation."""

    queries: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of article topics",
        example=["AI developments", "Climate change"],
    )  # type: ignore
    layout: Literal["single", "two-column", "layout_1.html", "layout_2.html"] = Field(
        default="two-column",
        description="Layout template to use (friendly name or template file)",
        example="two-column",
    )  # type: ignore
    output_dir: str | None = Field(
        default=None,
        alias="outputDir",
        description="Optional override for the output directory",
    )
    max_workers: int | None = Field(
        default=None,
        alias="maxWorkers",
        description="Maximum number of parallel workers for article processing",
        ge=1,
        le=10,
    )
    run_id: str | None = Field(
        default=None,
        alias="runId",
        description="Optional run identifier supplied by the client for log streaming",
        min_length=1,
        max_length=100,
    )
    workflow: SupportedWorkflow = Field(
        default=DEFAULT_WORKFLOW,
        description="Workflow variant to execute: 'original' (simple LLM tasks), 'dynamic' (parallel execution), or 'agent' (LLM agents with tools)",
        example=DEFAULT_WORKFLOW,
    )  # type: ignore

    model_config = ConfigDict(populate_by_name=True)

    def resolved_layout(self) -> str:
        return SUPPORTED_LAYOUTS.get(self.layout, self.layout)


class NewspaperResponse(BaseModel):
    """Response model for newspaper generation."""

    output_path: str = Field(
        alias="outputPath",
        description="Relative path to the generated newspaper HTML file",
        example="/outputs/run_1234567890/newspaper.html",
    )  # type: ignore
    html: str = Field(..., description="The rendered newspaper HTML content")
    created_at: str = Field(alias="createdAt", description="ISO timestamp of creation")
    layout: str = Field(description="Layout option used to generate the newspaper")
    queries: List[str] = Field(description="Queries used to generate the newspaper")
    run_id: str = Field(alias="runId", description="Identifier for the workflow run that produced this output")
    workflow: SupportedWorkflow = Field(description="Workflow variant that produced this newspaper")

    model_config = ConfigDict(populate_by_name=True)


class NewspaperSummary(BaseModel):
    """Summary metadata for a generated newspaper."""

    filename: str = Field(description="Output folder name (e.g., run_1234567890)")
    created_at: str = Field(alias="createdAt", description="ISO timestamp of creation")
    output_path: str = Field(alias="outputPath", description="Relative path to the HTML output")
    run_id: str = Field(alias="runId", description="Identifier for the workflow run")

    model_config = ConfigDict(populate_by_name=True)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok", example="ok")
    message: str = Field(default="GPT Newspaper API is running", example="GPT Newspaper API is running")


def _ensure_generation_ready() -> None:
    if not Config.TAVILY_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="TAVILY_API_KEY environment variable is required",
        )

    if not Config.OPENAI_API_KEY and "api_base" not in Config.DEFAULT_MODEL_PARAMS:
        raise HTTPException(
            status_code=500,
            detail="LLM credentials not configured. Provide OPENAI_API_KEY or set "
            "GRAFLOW_MODEL_PARAMS with connection details.",
        )


def _generate_run_id() -> str:
    timestamp = int(datetime.now(tz=timezone.utc).timestamp())
    return f"run_{timestamp}_{uuid4().hex[:6]}"


def _build_newspaper_response(path: Path, request: NewspaperRequest, run_id: str) -> NewspaperResponse:
    html = path.read_text(encoding="utf-8")
    created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    relative_path = f"/{path.resolve().relative_to(Path.cwd().resolve())}"
    return NewspaperResponse(
        output_path=relative_path,
        html=html,
        created_at=created_at,
        layout=request.layout,
        queries=request.queries,
        run_id=run_id,
        workflow=request.workflow,
    )


def _list_recent_newspapers(limit: int) -> List[NewspaperSummary]:
    outputs_root = Path("outputs")
    if not outputs_root.exists():
        return []

    files = sorted(
        outputs_root.glob("run_*/newspaper.html"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    summaries: List[NewspaperSummary] = []
    for file_path in files[:limit]:
        created_at = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc).isoformat()
        relative_path = f"/{file_path.resolve().relative_to(Path.cwd().resolve())}"
        summaries.append(
            NewspaperSummary(
                filename=file_path.parent.name,
                created_at=created_at,
                output_path=relative_path,
                run_id=file_path.parent.name,
            )
        )
    return summaries


def _generate_newspaper_impl(request: NewspaperRequest, run_id: str) -> NewspaperResponse:
    _ensure_generation_ready()
    sanitized_queries = [query.strip() for query in request.queries if query.strip()]
    if not sanitized_queries:
        raise HTTPException(status_code=400, detail="At least one non-empty query is required.")

    workflow_choice: SupportedWorkflow = request.workflow or DEFAULT_WORKFLOW
    workflow_runner = WORKFLOW_RUNNERS.get(workflow_choice, WORKFLOW_RUNNERS[DEFAULT_WORKFLOW])

    newspaper_path = workflow_runner(
        queries=sanitized_queries,
        layout=request.resolved_layout(),
        max_workers=request.max_workers,
        output_dir=request.output_dir,
    )
    response = _build_newspaper_response(Path(newspaper_path), request, run_id=run_id)
    response.layout = request.layout
    response.queries = sanitized_queries
    response.workflow = workflow_choice
    return response


def _process_generation_request(request: NewspaperRequest) -> NewspaperResponse:
    run_id = (request.run_id or "").strip() or _generate_run_id()
    with redirect_stdout_to_stream(run_id):
        return _generate_newspaper_impl(request, run_id=run_id)


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
        return HealthResponse(status="warning", message="TAVILY_API_KEY not configured")

    if not Config.OPENAI_API_KEY and "api_base" not in Config.DEFAULT_MODEL_PARAMS:
        return HealthResponse(status="warning", message="LLM credentials or API base not configured")

    return HealthResponse(status="ok", message="GPT Newspaper API is running")


@app.post("/api/generate", response_model=NewspaperResponse, response_model_by_alias=True)
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
    try:
        # Run blocking workflow in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _process_generation_request, request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate newspaper: {str(e)}")


@app.post("/api/newspaper", response_model=NewspaperResponse, response_model_by_alias=True)
async def create_newspaper(request: NewspaperRequest):
    try:
        # Run blocking workflow in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _process_generation_request, request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate newspaper: {str(e)}")


@app.get("/api/newspaper", response_model=List[NewspaperSummary], response_model_by_alias=True)
async def list_newspapers(limit: int = Query(10, ge=1, le=50)):
    return _list_recent_newspapers(limit)


@app.websocket("/ws/logs/{run_id}")
async def stream_logs(websocket: WebSocket, run_id: str):
    """
    Stream live logs for a given workflow run over WebSocket.
    """

    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    log_stream_manager.subscribe(run_id, queue, loop)

    try:
        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
            if payload.get("type") == "complete":
                break
    except WebSocketDisconnect:
        pass
    finally:
        log_stream_manager.unsubscribe(run_id, queue)
        with suppress(RuntimeError):
            await websocket.close()


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
    import threading

    import uvicorn

    print("=" * 80)
    print("🗞️  GPT NEWSPAPER API")
    print("=" * 80)
    print("\nStarting FastAPI server...")
    print("Frontend: http://localhost:8000/frontend")
    print("API docs: http://localhost:8000/docs")
    print("\n" + "=" * 80 + "\n")

    if sys.stdin and sys.stdin.isatty():
        threading.Thread(target=_stdin_shutdown_listener, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
