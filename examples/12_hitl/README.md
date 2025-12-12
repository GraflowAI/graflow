# Human-in-the-Loop (HITL) Examples

This directory contains examples demonstrating Graflow's Human-in-the-Loop (HITL) functionality.

## Overview

HITL allows workflows to pause and wait for human feedback before continuing. Supports:
- Multiple feedback types (approval, text, selection, custom)
- Immediate feedback (polling) or delayed feedback (checkpoint + resume)
- Distributed execution with Redis backend
- REST API for external feedback submission

## Examples

### 01_basic_approval.py
**Basic approval workflow with immediate feedback**

Demonstrates:
- Requesting approval from within a task
- Providing feedback programmatically
- Handling approval/rejection

```bash
python examples/12_hitl/01_basic_approval.py
```

### 02_timeout_checkpoint.py
**Timeout handling and checkpoint creation**

Demonstrates:
- Workflow timeout when feedback not provided
- Automatic checkpoint creation
- Resume from checkpoint after feedback provided
- Memory backend (in-process)

```bash
python examples/12_hitl/02_timeout_checkpoint.py
```

### 03_channel_integration.py
**Channel integration for inter-task communication**

Demonstrates:
- Automatic writing of feedback to channels
- Sharing feedback across tasks
- Using `write_to_channel` parameter

```bash
python examples/12_hitl/03_channel_integration.py
```

### 04_api_feedback.py
**Distributed feedback via REST API**

Demonstrates:
- Using the Feedback API server for workflow feedback
- Distributed execution with Redis backend
- External feedback submission via HTTP API
- Cross-process communication

**Requirements:**
```bash
uv sync --all-extras
```

**Setup:**
1. Start Redis:
   ```bash
   docker run -p 16379:6379 redis:7.2
   ```

2. Start API server (Terminal 1):
   ```bash
   uv run python -m graflow.api --backend redis --redis-host localhost --redis-port 16379
   ```

3. Run workflow (Terminal 2):
   ```bash
   # With custom port (matching Redis configuration)
   uv run python examples/12_hitl/04_api_feedback.py --redis-port 16379

   # Or with default port (6379)
   uv run python examples/12_hitl/04_api_feedback.py

   # With custom host and port
   uv run python examples/12_hitl/04_api_feedback.py --redis-host localhost --redis-port 16379
   ```

4. Provide feedback via API (Terminal 3):
   ```bash
   # List pending requests
   curl http://localhost:8000/api/feedback

   # Get request details
   curl http://localhost:8000/api/feedback/{feedback_id}

   # Provide approval
   curl -X POST http://localhost:8000/api/feedback/{feedback_id}/respond \
     -H "Content-Type: application/json" \
     -d '{"approved": true, "reason": "Approved via API"}'
   ```

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Usage

### Starting the Feedback API Server

**Using CLI (Recommended):**
```bash
# Filesystem backend (local development)
.venv/bin/python -m graflow.api --backend filesystem

# Redis backend (distributed/production)
.venv/bin/python -m graflow.api --backend redis --redis-host localhost --redis-port 6379

# With custom configuration
.venv/bin/python -m graflow.api \
  --backend redis \
  --redis-host localhost \
  --redis-port 6379 \
  --host 0.0.0.0 \
  --port 8080 \
  --enable-cors
```

**Programmatic Usage:**
```python
import redis
from graflow.api.app import create_feedback_api

# Create Redis client
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

# Create FastAPI app
app = create_feedback_api(
    feedback_backend="redis",
    feedback_config={"redis_client": redis_client}
)

# Run with uvicorn
# uvicorn api:app --host 0.0.0.0 --port 8000
```

### Requesting Feedback in Tasks

```python
from graflow.core.decorators import task

@task(inject_context=True)
def my_task(context):
    # Approval
    response = context.request_feedback(
        feedback_type="approval",
        prompt="Approve this action?",
        timeout=180.0  # 3 minutes
    )

    # Text input
    response = context.request_feedback(
        feedback_type="text",
        prompt="Enter your comment:",
        timeout=180.0
    )

    # Selection
    response = context.request_feedback(
        feedback_type="selection",
        prompt="Choose mode:",
        options=["fast", "balanced", "thorough"],
        timeout=180.0
    )

    # With channel integration
    response = context.request_feedback(
        feedback_type="approval",
        prompt="Approve?",
        channel_key="my_approval",
        write_to_channel=True,
        timeout=180.0
    )
```

### Convenience Methods

```python
@task(inject_context=True)
def my_task(context):
    # Approval (returns bool)
    approved = context.request_approval(
        prompt="Approve deployment?",
        timeout=180.0
    )

    # Text input (returns str)
    comment = context.request_text_input(
        prompt="Enter comment:",
        timeout=180.0
    )

    # Selection (returns str)
    mode = context.request_selection(
        prompt="Choose mode:",
        options=["fast", "balanced", "thorough"],
        timeout=180.0
    )
```

## REST API Endpoints

### List Pending Requests
```bash
GET /api/feedback?session_id={session_id}
```

### Get Request Details
```bash
GET /api/feedback/{feedback_id}
```

### Provide Feedback
```bash
POST /api/feedback/{feedback_id}/respond
Content-Type: application/json

{
  "approved": true,          # For approval type
  "reason": "Looks good",    # Optional reason
  "text": "My comment",      # For text type
  "selected": "option_b",    # For selection type
  "selected_multiple": [...], # For multi_selection type
  "custom_data": {...},      # For custom type
  "responded_by": "user@example.com"
}
```

### Cancel Request
```bash
DELETE /api/feedback/{feedback_id}
```

## Timeout Behavior

When a feedback request times out:
1. `FeedbackTimeoutError` is raised
2. WorkflowEngine catches the exception
3. Checkpoint is created automatically
4. Workflow exits gracefully
5. Feedback request remains in backend (pending)
6. Resume workflow after feedback is provided:
   ```python
   from graflow.core.checkpoint import CheckpointManager

   context, metadata = CheckpointManager.resume_from_checkpoint(path)
   engine.execute(context)
   ```

## Backend Configuration

### Memory Backend (Default)
```python
from graflow.core.workflow import workflow

with workflow("my_workflow") as wf:
    # Define tasks
    task1 >> task2

    # Execute with memory backend (default)
    result = wf.execute()
```

### Redis Backend (Distributed)
```python
import redis
from graflow.core.workflow import workflow
from graflow.core.context import ExecutionContext
from graflow.core.engine import WorkflowEngine

# Create Redis client
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

# Define workflow
with workflow("my_workflow") as wf:
    # Define tasks
    task1 >> task2

    # Create execution context with Redis backend
    exec_context = ExecutionContext.create(
        wf.graph,
        start_node=None,  # Auto-detect
        channel_backend="redis",
        config={"redis_client": redis_client}
    )

    # Execute workflow
    engine = WorkflowEngine()
    result = engine.execute(exec_context)
    # Tasks use Redis-based feedback storage
    # Feedback persists across processes
```

## Design Documentation

For detailed design information, see:
- `docs/hitl_design.md` - Complete HITL design specification
