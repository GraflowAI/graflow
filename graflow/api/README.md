# Graflow Feedback API

REST API for Human-in-the-Loop (HITL) feedback management in Graflow workflows.

## Installation

```bash
# Install with API dependencies
uv sync --all-extras
```

## Quick Start

### 1. Start the API Server

**Filesystem backend (default):**
```bash
python -m graflow.api --backend filesystem
```

**Redis backend:**
```bash
# Start Redis
docker run -p 6379:6379 redis:7.2

# Start API server
python -m graflow.api --backend redis --redis-host localhost --redis-port 6379
```

The server will start at `http://localhost:8000` with:
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 2. Run a Workflow with Feedback

```python
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.core.context import ExecutionContext

@task(inject_context=True)
def request_approval(ctx: ExecutionContext) -> bool:
    response = ctx.request_feedback(
        feedback_type="approval",
        prompt="Approve deployment?",
        timeout=180.0  # 3 minutes
    )
    return response.approved

with workflow("my_workflow", backend="filesystem") as wf:
    approved = request_approval()
    wf.execute()
```

### 3. Provide Feedback via API

**List pending requests:**
```bash
curl http://localhost:8000/api/feedback
```

**Provide approval:**
```bash
curl -X POST http://localhost:8000/api/feedback/{feedback_id}/respond \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "reason": "LGTM"}'
```

**Provide text feedback:**
```bash
curl -X POST http://localhost:8000/api/feedback/{feedback_id}/respond \
  -H "Content-Type: application/json" \
  -d '{"text": "Please fix the typos", "responded_by": "reviewer@example.com"}'
```

**Provide selection:**
```bash
curl -X POST http://localhost:8000/api/feedback/{feedback_id}/respond \
  -H "Content-Type: application/json" \
  -d '{"selected": "option_b"}'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/feedback` | List pending feedback requests |
| `GET` | `/api/feedback/{id}` | Get feedback details |
| `POST` | `/api/feedback/{id}/respond` | Provide feedback response |
| `DELETE` | `/api/feedback/{id}` | Cancel feedback request |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive API documentation |

## CLI Options

```bash
python -m graflow.api --help
```

**Backend Options:**
- `--backend {filesystem,redis}` - Backend type (default: filesystem)
- `--data-dir PATH` - Data directory for filesystem backend
- `--redis-host HOST` - Redis host (default: localhost)
- `--redis-port PORT` - Redis port (default: 6379)
- `--redis-db DB` - Redis database (default: 0)

**Server Options:**
- `--host HOST` - Server host (default: 127.0.0.1)
- `--port PORT` - Server port (default: 8000)
- `--reload` - Enable auto-reload for development
- `--workers N` - Number of worker processes
- `--log-level LEVEL` - Log level (info, debug, etc.)

**API Options:**
- `--enable-cors` - Enable CORS
- `--cors-origins URL [URL ...]` - Allowed CORS origins

## Examples

See `examples/11_hitl/feedback_api_example.py` for a complete example.

## Architecture

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   Workflow   │────────▶│  Feedback    │────────▶│   Backend    │
│   (Task)     │         │   Manager    │         │ (filesystem/ │
│              │         │              │         │   Redis)     │
└──────────────┘         └──────────────┘         └──────────────┘
                                │                         ▲
                                │                         │
                                ▼                         │
                         ┌──────────────┐                 │
                         │  FastAPI     │                 │
                         │  Server      │─────────────────┘
                         └──────────────┘
                                ▲
                                │
                                │ REST API
                                │
                         ┌──────────────┐
                         │  External    │
                         │  Client      │
                         │ (Web UI/CLI) │
                         └──────────────┘
```

## Development

**Run tests:**
```bash
PYTHONPATH=. uvx pytest tests/hitl/test_api.py -v
```

**Start with auto-reload:**
```bash
python -m graflow.api --reload
```

## See Also

- Design document: `docs/hitl_design.md`
- HITL manager: `graflow/hitl/manager.py`
- Feedback types: `graflow/hitl/types.py`
