# Checkpoint/Resume Design Document

**Document Version**: 2.2
**Date**: 2025-01-23
**Status**: Draft

**Changelog**:
- v2.2: Deferred checkpoint execution (major design improvement)
  - **Breaking change**: `context.checkpoint()` now sets flag instead of creating checkpoint immediately
  - Checkpoint created by engine AFTER task completes successfully (not during execution)
  - Removed `current_task_spec` from checkpoint state (no longer needed)
  - Task that requested checkpoint is marked as completed on resume
  - Added `checkpoint_requested` and `checkpoint_request_metadata` to ExecutionContext
  - Added `request_checkpoint(metadata)` method to ExecutionContext
  - Updated WorkflowEngine.execute() to handle deferred checkpoint creation
  - Simplified checkpoint semantics: tasks are either completed or pending
- v2.1: Added Resume Execution Flow section and clarified start_task_id usage
  - Added "Resume Execution Flow" section explaining auto-resume vs manual override
  - Clarified `engine.execute(context, start_task_id)` API usage
  - Updated all usage examples to show auto-resume pattern
  - Added detailed explanation of task re-execution semantics (restart from beginning)
  - Added channel-based state machine pattern example for idempotent task execution
- v2.0: Major design revisions based on review feedback
  - Changed `pending_tasks` from task IDs to full TaskSpec (includes task_data, retry_count, etc.)
  - Added `current_task_spec` to capture currently executing task for mid-execution checkpoints
  - Added `schema_version` field to checkpoint state for compatibility validation
  - Changed `CheckpointMetadata.created_at` to ISO 8601 string (was datetime)
  - Added `CheckpointMetadata.create()` class method for timestamp generation
  - Added `_atomic_save()` for atomic write using tmpfile + rename (local backend)
  - Added `_get_base_path()` to handle multi-file naming (prevents double extensions)
  - Added `_validate_schema_version()` for compatibility checking
  - Added Storage Abstraction Design section (Phase 2 implementation plan)
  - Updated Implementation Checklist with all new requirements
- v1.5: Changed CheckpointManager to class methods, storage backend inferred from path format
- v1.4: `mark_task_completed()` adds task_id to completed_tasks set (reverted from no-op)
- v1.3: Removed remaining `__workflow_state__` references from architecture diagrams
- v1.2: Added clarification on TaskQueue persistence strategy (InMemoryTaskQueue vs RedisTaskQueue)
- v1.1: Introduced CheckpointManager abstraction, three-file structure, removed `__workflow_state__`
- v1.0: Initial design

---

## Design Summary

This document describes Graflow's checkpoint/resume functionality for workflow state persistence and recovery.

### Key Design Decisions

1. **Explicit Checkpoints**: Users call `context.checkpoint()` explicitly (no auto-checkpoint)
2. **CheckpointManager**: Class methods only, storage backend inferred from path format
   - Local: `"checkpoints/session_123.pkl"`
   - Redis: `"redis://session_123"` (future)
   - S3: `"s3://bucket/session_123.pkl"` (future)
3. **Three-file Structure**:
   - `.pkl` (ExecutionContext pickle)
   - `.state.json` (checkpoint state with schema_version, pending TaskSpecs)
   - `.meta.json` (metadata with ISO 8601 timestamps)
4. **Full TaskSpec Persistence**:
   - Save complete TaskSpec (task_data, retry_count, execution_strategy) not just IDs
   - Enables dynamic task and retry state restoration
5. **Deferred Checkpoint Execution**:
   - `context.checkpoint()` sets flag, does NOT create checkpoint immediately
   - Checkpoint created by engine AFTER task completes successfully
   - Task that requested checkpoint is marked as completed
6. **Queue State Persistence**:
   - InMemoryTaskQueue: Pending TaskSpecs saved to checkpoint and re-queued on resume
   - RedisTaskQueue: Queue already persisted in Redis, no re-queuing needed
7. **Schema Versioning**: `schema_version` field for compatibility validation
8. **Atomic Write**: tmpfile + atomic rename for local backend (Phase 1)
9. **Storage Abstraction**: Path-based backend inference, extensible to Redis/S3 (Phase 2)
10. **Backend Support**: Works with both MemoryChannel and RedisChannel

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Requirements](#requirements)
4. [Design Goals](#design-goals)
5. [Architecture](#architecture)
6. [API Design](#api-design)
7. [Implementation Details](#implementation-details)
8. [Backend Support](#backend-support)
9. [Resume Execution Flow](#resume-execution-flow)
10. [Usage Examples](#usage-examples)
11. [Edge Cases and Limitations](#edge-cases-and-limitations)
12. [Future Enhancements](#future-enhancements)

---

## Overview

This document describes the design of checkpoint/resume functionality for Graflow workflows. This feature allows users to:

- **Save workflow state** at arbitrary points during execution (checkpoint)
- **Resume execution** from a saved checkpoint after interruption or failure
- **Support both Memory and Redis channels** with consistent behavior

The design leverages Graflow's existing state machine execution model and channel-based state persistence.

---

## Motivation

### Use Cases

1. **Long-running workflows**: ML training, data processing pipelines that run for hours/days
2. **Fault tolerance**: Resume after infrastructure failures, OOM errors, or crashes
3. **Iterative development**: Save intermediate states for debugging and testing
4. **Distributed execution**: Workers can pick up from checkpoints after restarts
5. **Cost optimization**: Pause expensive workflows and resume later

### Current Limitations

Without checkpoint/resume:
- Workflow failures require complete re-execution from the beginning
- No mechanism to persist intermediate progress
- Difficult to debug long-running workflows
- Workers cannot resume after crashes

---

## Requirements

### Functional Requirements

**FR1**: Users can create checkpoints from within tasks using `context.checkpoint()`
**FR2**: Checkpoints save complete workflow state: session_id, graph, steps, completed tasks, cycle counts
**FR3**: `ExecutionContext.resume(path)` restores workflow from checkpoint file
**FR4**: Works transparently with both MemoryChannel and RedisChannel backends
**FR5**: Resumed workflows continue from the next pending task

### Non-Functional Requirements

**NFR1**: Checkpoint creation should be lightweight (< 1 second for typical workflows)
**NFR2**: Checkpoint files should be portable across machines (same Redis instance for RedisChannel)
**NFR3**: Design should not break existing workflows without checkpoints
**NFR4**: API should be simple and intuitive

---

## Design Goals

1. **Explicit control**: Checkpoints created only when user calls `context.checkpoint()`
2. **Backend agnostic**: Same API works for Memory and Redis channels
3. **Minimal overhead**: Leverage existing serialization (`__getstate__`/`__setstate__`)
4. **State machine friendly**: Works seamlessly with `next_iteration()` and state-based workflows
5. **Metadata support**: Allow users to attach custom metadata to checkpoints

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Task Code                           │
│                                                              │
│  @task(inject_context=True)                                 │
│  def my_task(context):                                      │
│      # ... processing ...                                   │
│      context.checkpoint(metadata={"stage": "step1"})  ◄─────┼─── Explicit checkpoint
│      # ... more processing ...                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              TaskExecutionContext                           │
│  .checkpoint(path, metadata) ──────────────────────┐        │
└────────────────────────────────────────────────────┼────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ExecutionContext                               │
│                                                              │
│  .mark_task_completed(task_id)                              │
│    └─> Add task_id to completed_tasks set                  │
│                                                              │
│  .checkpoint(path, metadata) ───────────────────────────┐   │
│    1. Update checkpoint_metadata                        │   │
│    2. Collect checkpoint state (session_id, graph, etc) │   │
│    3. Call self.save(path) ─────────────────────────────┼───┼─> cloudpickle
│                                                          │   │
│  .resume(path) ◄─────────────────────────────────────────┘   │
│    1. Load from pickle (calls __setstate__)                 │
│    2. Restore completed_tasks, cycle_counts                 │
│    3. Restore pending_tasks to queue (InMemoryTaskQueue)    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Channel (Memory or Redis)                      │
│                                                              │
│  Memory: Data saved to pickle via __getstate__              │
│  Redis:  Data persists in Redis, session_id saved to pickle │
└─────────────────────────────────────────────────────────────┘
```

### Component Interactions

```
User Task
    │
    │ context.checkpoint()
    ▼
TaskExecutionContext.checkpoint()
    │
    │ Add task metadata
    ▼
ExecutionContext.checkpoint()
    │
    ├─> Collect checkpoint state (session_id, graph, steps, etc)
    │
    └─> self.save(path)
            │
            ├─> __getstate__()
            │       └─> MemoryChannel: Save channel data to state
            │       └─> RedisChannel: Save only session_id
            │
            └─> cloudpickle.dump()
```

---

## API Design

### CheckpointManager API

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class CheckpointMetadata:
    """Checkpoint metadata."""
    checkpoint_id: str
    session_id: str
    created_at: str  # ISO 8601 format (e.g., "2025-01-23T10:30:00Z")
    steps: int
    start_node: str
    backend: dict[str, str]  # {"queue": "memory", "channel": "memory"}
    user_metadata: dict[str, Any]  # User-defined metadata

    @classmethod
    def create(cls, checkpoint_id, session_id, steps, start_node, backend, user_metadata=None):
        """Create metadata with ISO 8601 timestamp."""
        return cls(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            steps=steps,
            start_node=start_node,
            backend=backend,
            user_metadata=user_metadata or {}
        )

class CheckpointManager:
    """Manages checkpoint creation and restoration with storage backend abstraction.

    Storage backend is automatically determined from the path format:
    - Local filesystem: "checkpoints/session_123.pkl"
    - Redis: "redis://session_123" (future)
    - S3: "s3://bucket/session_123.pkl" (future)
    """

    @classmethod
    def create_checkpoint(
        cls,
        context: ExecutionContext,
        path: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> tuple[str, CheckpointMetadata]:
        """Create checkpoint from execution context.

        Args:
            context: ExecutionContext to checkpoint
            path: Checkpoint path. Storage backend inferred from format:
                  - None: Auto-generate local path
                  - "checkpoints/session_123.pkl": Local filesystem
                  - "redis://session_123": Redis storage (future)
                  - "s3://bucket/session_123.pkl": S3 storage (future)
            metadata: User-defined metadata

        Returns:
            (checkpoint_path, metadata): Path and metadata
        """

    @classmethod
    def resume_from_checkpoint(
        cls,
        checkpoint_path: str
    ) -> tuple[ExecutionContext, CheckpointMetadata]:
        """Resume from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint. Storage backend inferred from format:
                            - "checkpoints/session_123.pkl": Local filesystem
                            - "redis://session_123": Redis storage (future)
                            - "s3://bucket/session_123.pkl": S3 storage (future)

        Returns:
            (context, metadata): Restored context and metadata
        """

    @classmethod
    def list_checkpoints(
        cls,
        path_pattern: Optional[str] = None
    ) -> list[CheckpointMetadata]:
        """List available checkpoints (future enhancement).

        Args:
            path_pattern: Path pattern to filter checkpoints
                         - "checkpoints/*.pkl": Local files
                         - "redis://*": All Redis checkpoints
        """

    @classmethod
    def _infer_backend_from_path(cls, path: Optional[str]) -> str:
        """Infer storage backend from path format.

        Returns:
            "local", "redis", or "s3"
        """
```

### ExecutionContext API

```python
class ExecutionContext:
    # New attributes
    completed_tasks: set[str]           # Tracks completed task IDs
    checkpoint_metadata: dict[str, Any] # Last checkpoint metadata
    checkpoint_requested: bool = False  # Flag for deferred checkpoint
    checkpoint_request_metadata: Optional[dict] = None  # User metadata for checkpoint

    # New methods
    def mark_task_completed(self, task_id: str) -> None:
        """Mark task as completed for checkpoint tracking."""
        self.completed_tasks.add(task_id)

    def request_checkpoint(self, metadata: Optional[dict] = None) -> None:
        """Request checkpoint creation (deferred to engine)."""
        self.checkpoint_requested = True
        self.checkpoint_request_metadata = metadata

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Get current state for checkpointing.

        Returns:
            Dictionary containing all necessary state:
            - schema_version, session_id, start_node, graph
            - steps, completed_tasks, cycle_counts
            - pending_tasks (full TaskSpec list from queue)

        Note:
            No current_task_spec needed - checkpoint created after task completes,
            so all tasks are either completed or in pending_tasks queue.
        """
```

### TaskExecutionContext API

```python
class TaskExecutionContext:
    def checkpoint(self, metadata: Optional[dict] = None) -> None:
        """Request checkpoint creation (executed after task completes).

        Args:
            metadata: User-defined metadata

        Note:
            - Does NOT create checkpoint immediately
            - Sets checkpoint_requested flag in ExecutionContext
            - Actual checkpoint created by engine after task execution completes
            - Automatically includes task_id, cycle_count, elapsed_time in metadata

        Example:
            @task(inject_context=True)
            def my_task(context):
                do_work()
                context.checkpoint(metadata={"stage": "completed"})
                # Task continues, checkpoint created after return
        """
```

### WorkflowEngine Integration

```python
class WorkflowEngine:
    def execute(self, context: ExecutionContext, start_task_id: Optional[str] = None):
        """Execute workflow from context.

        Args:
            context: ExecutionContext with graph and queue
            start_task_id: Optional task ID to start from
                          - If None: use context.get_next_task() (auto-resume from queue)
                          - If specified: start from that specific task

        Usage for checkpoint/resume:
            # Auto-resume (recommended)
            engine.execute(context)  # Gets next task from restored queue

            # Manual override
            engine.execute(context, start_task_id=metadata.start_node)
        """
        while task_id is not None and context.steps < context.max_steps:
            # Reset checkpoint flag
            context.checkpoint_requested = False

            # Execute task
            task = graph.get_node(task_id)
            with context.executing_task(task):
                self._execute_task(task, context)

            # Track completion
            context.mark_task_completed(task_id)
            context.increment_step()

            # Handle deferred checkpoint (NEW)
            if context.checkpoint_requested:
                path, metadata = CheckpointManager.create_checkpoint(
                    context,
                    metadata=context.checkpoint_request_metadata
                )
                print(f"Checkpoint created: {path}")
                context.checkpoint_requested = False
                context.checkpoint_request_metadata = None

            # Schedule successors
            # ...

            # Get next task
            task_id = context.get_next_task()
```

---

## Implementation Details

### TaskExecutionContext.checkpoint() Implementation

```python
class TaskExecutionContext:
    def checkpoint(self, metadata: Optional[dict] = None) -> None:
        """Request checkpoint creation (deferred to engine)."""
        # Enrich metadata with task-specific info
        enriched_metadata = metadata or {}
        enriched_metadata.update({
            "task_id": self.task_id,
            "cycle_count": self.execution_context.cycle_controller.get_count(self.task_id),
            "elapsed_time": time.time() - self.start_time,
        })

        # Set flag for engine to create checkpoint after task completes
        self.execution_context.request_checkpoint(enriched_metadata)
```

### CheckpointManager Implementation

```python
class CheckpointManager:
    @classmethod
    def create_checkpoint(cls, context, path=None, metadata=None):
        # 1. Infer backend from path and generate path if needed
        backend = cls._infer_backend_from_path(path)
        checkpoint_id = cls._generate_checkpoint_id(context)
        if path is None:
            path = cls._generate_path(checkpoint_id, backend)

        # 2. Collect checkpoint state
        checkpoint_state = {
            "schema_version": "1.0",
            "session_id": context.session_id,
            "start_node": context.start_node,
            "steps": context.steps,
            "completed_tasks": list(context.completed_tasks),
            "cycle_counts": dict(context.cycle_controller.cycle_counts),
            "pending_tasks": cls._get_pending_task_specs(context),  # Full TaskSpec
            "backend": {
                "queue": context._queue_backend_type,
                "channel": context._channel_backend_type
            }
        }
        # Note: No current_task_spec - checkpoint created after task completes

        # 3. Create metadata with ISO 8601 timestamp
        metadata_obj = CheckpointMetadata.create(
            checkpoint_id=checkpoint_id,
            session_id=context.session_id,
            steps=context.steps,
            start_node=context.start_node,
            backend=checkpoint_state["backend"],
            user_metadata=metadata or {}
        )

        # 4. Save to storage (atomic write for local backend)
        cls._atomic_save(context, path, checkpoint_state, metadata_obj, backend)

        return path, metadata_obj

    @classmethod
    def resume_from_checkpoint(cls, checkpoint_path):
        # 1. Infer backend from path
        backend = cls._infer_backend_from_path(checkpoint_path)

        # 2. Load checkpoint state
        checkpoint_state = cls._load_checkpoint_state(checkpoint_path, backend)

        # 3. Validate schema version
        cls._validate_schema_version(checkpoint_state.get("schema_version", "1.0"))

        # 4. Load context from pickle
        context = ExecutionContext.load(checkpoint_path)

        # 5. Restore pending tasks to queue (backend-specific)
        # InMemoryTaskQueue: Re-queue TaskSpecs (queue state was lost)
        # RedisTaskQueue: Skip re-queuing (queue already persisted in Redis)
        if context._queue_backend_type != "redis":
            # Restore pending tasks from TaskSpec
            for task_spec_dict in checkpoint_state["pending_tasks"]:
                task_spec = cls._deserialize_task_spec(task_spec_dict)
                context.task_queue.enqueue(task_spec)
        # For Redis: Queue state already exists, no re-queuing needed
        # Note: No current_task_spec to restore - all tasks in completed or pending

        # 6. Restore completed tasks tracking
        context.completed_tasks = set(checkpoint_state["completed_tasks"])

        # 7. Restore cycle counts
        context.cycle_controller.cycle_counts.update(
            checkpoint_state["cycle_counts"]
        )

        # 8. Load metadata
        metadata = cls._load_metadata(checkpoint_path, backend)

        # 9. Set context start_node for resumption
        # Note: Caller should use engine.execute(context, start_task_id=None)
        # which will automatically use context.get_next_task() to resume from queue

        return context, metadata

    @classmethod
    def _infer_backend_from_path(cls, path: Optional[str]) -> str:
        """Infer storage backend from path format."""
        if path is None:
            return "local"
        if path.startswith("redis://"):
            return "redis"
        if path.startswith("s3://"):
            return "s3"
        return "local"

    @classmethod
    def _get_pending_task_specs(cls, context: ExecutionContext) -> list[dict]:
        """Extract full TaskSpec from queue (not just IDs)."""
        if hasattr(context.task_queue, 'get_pending_task_specs'):
            task_specs = context.task_queue.get_pending_task_specs()
            return [cls._serialize_task_spec(spec) for spec in task_specs]
        return []

    @classmethod
    def _serialize_task_spec(cls, task_spec) -> dict:
        """Serialize TaskSpec to dict for JSON storage."""
        return {
            "task_id": task_spec.task_id,
            "task_data": task_spec.task_data,  # Already serialized
            "status": task_spec.status,
            "priority": task_spec.priority,
            "retry_count": task_spec.retry_count,
            "max_retries": task_spec.max_retries,
            "execution_strategy": task_spec.execution_strategy,
            # Add other fields as needed
        }

    @classmethod
    def _deserialize_task_spec(cls, task_spec_dict: dict):
        """Deserialize TaskSpec from dict."""
        from graflow.queue.base import TaskSpec
        return TaskSpec(
            task_id=task_spec_dict["task_id"],
            task_data=task_spec_dict["task_data"],
            status=task_spec_dict.get("status", "pending"),
            priority=task_spec_dict.get("priority", 0),
            retry_count=task_spec_dict.get("retry_count", 0),
            max_retries=task_spec_dict.get("max_retries", 3),
            execution_strategy=task_spec_dict.get("execution_strategy", "direct"),
        )

    @classmethod
    def _validate_schema_version(cls, version: str) -> None:
        """Validate checkpoint schema version."""
        supported_versions = ["1.0"]
        if version not in supported_versions:
            raise ValueError(
                f"Unsupported checkpoint schema version: {version}. "
                f"Supported: {supported_versions}"
            )

    @classmethod
    def _atomic_save(cls, context, path, checkpoint_state, metadata_obj, backend):
        """Atomically save checkpoint files.

        For local backend: Use temporary files and atomic rename.
        For remote backends: Implementation depends on backend capabilities.
        """
        if backend == "local":
            import tempfile
            import os
            import shutil

            # Create temp directory
            temp_dir = tempfile.mkdtemp(prefix="checkpoint_")
            try:
                # Save to temp files
                base_path = cls._get_base_path(path)
                temp_pkl = os.path.join(temp_dir, os.path.basename(base_path) + ".pkl")
                temp_state = os.path.join(temp_dir, os.path.basename(base_path) + ".state.json")
                temp_meta = os.path.join(temp_dir, os.path.basename(base_path) + ".meta.json")

                context.save(temp_pkl)
                cls._save_checkpoint_state(temp_state, checkpoint_state, backend)
                cls._save_metadata(temp_meta, metadata_obj, backend)

                # Atomic rename to final location
                os.rename(temp_pkl, base_path + ".pkl")
                os.rename(temp_state, base_path + ".state.json")
                os.rename(temp_meta, base_path + ".meta.json")
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            # For redis/s3: implementation TBD in Phase 2
            raise NotImplementedError(f"Backend {backend} not yet implemented")

    @classmethod
    def _get_base_path(cls, path: str) -> str:
        """Get base path without extension for multi-file checkpoint.

        Examples:
            "checkpoints/session_123.pkl" -> "checkpoints/session_123"
            "checkpoints/session_123" -> "checkpoints/session_123"
        """
        if path.endswith(".pkl"):
            return path[:-4]
        return path
```

### Storage Abstraction Design (Phase 2)

**Problem**: `context.save(path)` uses `open(path, "wb")`, which fails for `redis://` or `s3://` paths.

**Solution**: Introduce storage abstraction layer.

```python
class StorageBackend:
    """Abstract storage backend."""
    @abstractmethod
    def save_pickle(self, path: str, obj: Any) -> None: ...
    @abstractmethod
    def load_pickle(self, path: str) -> Any: ...
    @abstractmethod
    def save_json(self, path: str, data: dict) -> None: ...
    @abstractmethod
    def load_json(self, path: str) -> dict: ...

class LocalStorage(StorageBackend):
    """Local filesystem storage."""
    def save_pickle(self, path, obj):
        with open(path, "wb") as f:
            cloudpickle.dump(obj, f)

class RedisStorage(StorageBackend):
    """Redis storage (Phase 2)."""
    def save_pickle(self, path, obj):
        # path = "redis://session_123"
        key = path.replace("redis://", "")
        self.redis_client.set(key, cloudpickle.dumps(obj))
```

**Implementation Plan**:
- Phase 1: Local only, direct file I/O
- Phase 2: Add StorageBackend abstraction
- Phase 3: Implement RedisStorage, S3Storage

### State Persistence Strategy

**MemoryChannel**:
```python
# Checkpoint creation
__getstate__():
    - Iterate channel.keys()
    - Save all key-value pairs to state['_channel_data']
    - NO __workflow_state__ needed (tracked separately)

# Resume
__setstate__():
    - Create new MemoryChannel
    - Restore all key-value pairs from state['_channel_data']
```

**RedisChannel**:
```python
# Checkpoint creation
__getstate__():
    - Save session_id (data already in Redis)
    - Do NOT save channel data (already persisted)
    - NO __workflow_state__ needed (tracked separately)

# Resume
__setstate__():
    - Reconnect to Redis using session_id
    - Data automatically available via same channel name
```

### Checkpoint State Schema

Saved separately from ExecutionContext pickle as JSON:

```python
{
    "schema_version": "1.0",                        # Schema version for compatibility
    "session_id": "12345",                           # Session identifier
    "start_node": "my_task",                        # Entry point for execution
    "steps": 42,                                    # Total steps executed
    "completed_tasks": ["task1", "task2", "task3"], # Completed task IDs
    "cycle_counts": {                               # Per-task cycle counts
        "task1": 3,
        "task2": 1
    },
    "pending_tasks": [                              # Full TaskSpec, not just IDs
        {
            "task_id": "task4",
            "task_data": "...",                     # Serialized task
            "status": "pending",
            "priority": 0,
            "retry_count": 0,
            "max_retries": 3,
            "execution_strategy": "direct"
        },
        {
            "task_id": "task5",
            "task_data": "...",
            "status": "pending",
            # ...
        }
    ],
    "backend": {
        "queue": "memory",
        "channel": "memory"
    }
}
```

**Note**: No `current_task_spec` field - checkpoint created after task completes, so all tasks are either in `completed_tasks` or `pending_tasks`.

### Checkpoint Metadata Schema

Stored in `ExecutionContext.checkpoint_metadata`:

```python
{
    "created_at": 1234567890.123,      # Unix timestamp
    "session_id": "12345",             # Workflow session ID
    "start_node": "my_task",           # Entry point task
    "steps": 42,                       # Steps at checkpoint time
    "completed_tasks": 3,              # Number of completed tasks
    "backend": {
        "queue": "memory",             # Queue backend type
        "channel": "memory"            # Channel backend type
    },
    # User-defined metadata
    "stage": "processing",
    "task_id": "ml_training",
    "cycle_count": 5,
    "epoch": 10
}
```

### File Naming Convention

Auto-generated checkpoint paths:
```
checkpoints/session_{session_id}_step_{steps}_{timestamp}.pkl
```

Example:
```
checkpoints/session_12345_step_42_1706024400.pkl
```

---

## Backend Support

### Comparison Matrix

| Feature | MemoryChannel | RedisChannel |
|---------|---------------|--------------|
| **State persistence** | Saved to pickle file | Persisted in Redis |
| **Checkpoint size** | Larger (includes all channel data) | Smaller (only session_id) |
| **Portability** | Self-contained file | Requires same Redis instance |
| **Multi-worker resume** | ❌ Not supported | ✅ Any worker can resume |
| **Durability** | File system dependent | Redis persistence dependent |

### Queue State Persistence Strategy

The persistence strategy for queue state (pending tasks) differs significantly between queue backends:

| Queue Backend | Persistence Strategy | Checkpoint Behavior |
|---------------|---------------------|---------------------|
| **InMemoryTaskQueue** | Not persisted | Must save `pending_tasks` list to checkpoint state |
| **RedisTaskQueue** | Already persisted in Redis | Only save session_id; pending tasks auto-restored on reconnect |

**InMemoryTaskQueue**:
- Queue state exists only in memory and is lost on process termination
- `get_pending_tasks()` returns list of task IDs currently in the queue
- Saved to `{path}.state.json` under `pending_tasks` field
- On resume, tasks are re-queued via `context.add_to_queue()`
- This ensures that pending work is not lost on checkpoint/resume

**RedisTaskQueue**:
- Queue state already persists in Redis under session-specific keys (e.g., `session_12345:queue`)
- `get_pending_tasks()` can return current queue state for verification purposes
- Pending tasks don't need explicit saving since they already exist in Redis
- On resume, reconnecting with same session_id automatically restores queue access
- No explicit re-queuing needed; queue is already populated in Redis
- Workers can pick up tasks from the persisted queue immediately

**Implementation Note**: The `CheckpointManager._get_pending_tasks()` method should handle both cases:
```python
def _get_pending_tasks(self, context: ExecutionContext) -> list[str]:
    """Extract pending tasks from queue (backend-specific)."""
    if context._queue_backend_type == "redis":
        # Redis queue already persisted, save for verification only
        return context.task_queue.get_pending_tasks() if hasattr(context.task_queue, 'get_pending_tasks') else []
    else:
        # InMemoryTaskQueue: must save pending tasks
        return context.task_queue.get_pending_tasks()
```

### MemoryChannel Workflow

```
1. Execution:
   task1 → channel.set("state", "A")
   task2 → channel.set("state", "B")

2. Checkpoint:
   CheckpointManager.create_checkpoint() creates:

   File 1: {checkpoint_path}.pkl (ExecutionContext pickle)
   - Contains: _channel_data = {"state": "B", "order_data": {...}}
   - session_id, graph, backend config

   File 2: {checkpoint_path}.state.json (Checkpoint state)
   - session_id, start_node, steps
   - completed_tasks, cycle_counts
   - pending_tasks (from queue)

   File 3: {checkpoint_path}.meta.json (Metadata)
   - checkpoint_id, created_at
   - user_metadata

3. Resume:
   CheckpointManager.resume_from_checkpoint() →
   - Load ExecutionContext from pickle
   - MemoryChannel restored with _channel_data
   - Load checkpoint state, restore pending_tasks to queue
   - Load metadata
   - Continue execution
```

### RedisChannel Workflow

```
1. Execution (Worker 1):
   task1 → redis.set("session_12345:state", "A")
   task2 → redis.set("session_12345:state", "B")

2. Checkpoint:
   CheckpointManager.create_checkpoint() creates:

   File 1: {checkpoint_path}.pkl (ExecutionContext pickle)
   - Contains: session_id = "12345" (NO channel data)
   - graph, backend config

   File 2: {checkpoint_path}.state.json (Checkpoint state)
   - session_id, start_node, steps
   - completed_tasks, cycle_counts
   - pending_tasks (from queue)

   File 3: {checkpoint_path}.meta.json (Metadata)
   - checkpoint_id, created_at
   - user_metadata

3. Resume (Worker 2):
   CheckpointManager.resume_from_checkpoint() →
   - Load ExecutionContext from pickle
   - Reconnect to Redis with session_id="12345"
   - redis.get("session_12345:state") → "B" (data already in Redis)
   - Load checkpoint state, restore pending_tasks to queue
   - Load metadata
   - Continue execution
```

### Saved Information Summary

**ExecutionContext Pickle** (`{path}.pkl`):
- `session_id`: Workflow session identifier
- `graph`: TaskGraph structure (nodes, edges)
- `start_node`: Entry point task
- `cycle_controller`: Cycle management state
- `_queue_backend_type`: Queue backend type
- `_channel_backend_type`: Channel backend type
- `_original_config`: Backend configuration
- `_channel_data`: Channel data (MemoryChannel only)

**Checkpoint State** (`{path}.state.json`):
- `schema_version`: Schema version (e.g., "1.0") for future compatibility
- `session_id`: Session identifier
- `start_node`: Entry point for resume
- `steps`: Total steps executed
- `completed_tasks`: List of completed task IDs
- `cycle_counts`: Per-task cycle execution counts
- `pending_tasks`: List of TaskSpec dictionaries (full task state, not just IDs)
- `backend`: Queue and channel backend types

**Note**: No `current_task_spec` - checkpoint created after task completes

**Metadata** (`{path}.meta.json`):
- `checkpoint_id`: Unique checkpoint identifier
- `session_id`: Session identifier
- `created_at`: Timestamp
- `steps`: Steps at checkpoint time
- `start_node`: Entry point
- `backend`: Backend configuration
- `user_metadata`: User-defined metadata

**Additional Information (if needed)**:
- Task-specific state: Stored in channel by user
- Results: Stored in channel with `{task_id}.__result__` key
- Custom workflow data: User stores in channel

---

## Resume Execution Flow

### How Resume Works

```
1. Load Checkpoint
   CheckpointManager.resume_from_checkpoint(path)
   ↓
   - Load ExecutionContext from pickle
   - Load checkpoint state from .state.json
   - Restore pending_tasks to queue (InMemoryTaskQueue only)
   - Restore completed_tasks, cycle_counts
   ↓
   Return (context, metadata)

2. Resume Execution
   engine.execute(context)  # No start_task_id specified
   ↓
   - context.get_next_task() retrieves from restored queue
   - Executes pending tasks in queue order
   ↓
   Workflow continues from checkpoint point

3. Manual Start Point (Optional)
   engine.execute(context, start_task_id="specific_task")
   ↓
   - Ignores queue, starts from specified task
   - Use case: Override resume behavior for debugging/testing
```

### Key Points

1. **Auto-resume (Recommended)**:
   ```python
   context, metadata = CheckpointManager.resume_from_checkpoint(path)
   engine.execute(context)  # Automatically resumes from queue
   ```

2. **Manual start_task_id**:
   ```python
   context, metadata = CheckpointManager.resume_from_checkpoint(path)
   engine.execute(context, start_task_id=metadata.start_node)  # Override
   ```

3. **Task re-execution**: Tasks checkpoint mid-execution restart from beginning (use channel state)

4. **Queue backends**:
   - InMemoryTaskQueue: pending_tasks re-queued on resume
   - RedisTaskQueue: queue already in Redis, no re-queueing

---

## Usage Examples

### Example 1: State Machine with Checkpoints

```python
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.core.checkpoint import CheckpointManager

with workflow("order_processing") as ctx:

    @task(inject_context=True)
    def process_order(context):
        channel = context.get_channel()
        state = channel.get("order_state", default="NEW")
        order_data = channel.get("order_data")

        if state == "NEW":
            validate_order(order_data)
            channel.set("order_state", "VALIDATED")

            # Checkpoint after validation
            checkpoint_path, metadata = context.checkpoint(
                metadata={
                    "stage": "validation_complete",
                    "order_id": order_data["id"]
                }
            )
            print(f"Checkpoint saved: {checkpoint_path}")

            context.next_iteration()

        elif state == "VALIDATED":
            process_payment(order_data)
            channel.set("order_state", "PAID")

            # Checkpoint after payment
            checkpoint_path, metadata = context.checkpoint(
                metadata={
                    "stage": "payment_complete",
                    "amount": order_data["amount"]
                }
            )
            print(f"Checkpoint saved: {checkpoint_path}")

            context.next_iteration()

        elif state == "PAID":
            ship_order(order_data)
            return "ORDER_COMPLETE"

    # Initial execution
    channel = ctx.execution_context.get_channel()
    channel.set("order_data", {"id": "ORD123", "amount": 100})

    try:
        ctx.execute("process_order", max_steps=10)
    except Exception as e:
        print(f"Error: {e}")
        # Checkpoint saved at last successful stage

# Resume after failure
from graflow.core.engine import WorkflowEngine

context, metadata = CheckpointManager.resume_from_checkpoint(
    checkpoint_path="checkpoints/session_12345_step_5.pkl"
)
print(f"Resuming from step {metadata.steps}, stage: {metadata.user_metadata.get('stage')}")

engine = WorkflowEngine()
# Option 1: Auto-resume from queue (pending_tasks already enqueued)
engine.execute(context)  # Automatically gets next task from queue

# Option 2: Explicit start_task_id (if you want to override)
# engine.execute(context, start_task_id=metadata.start_node)
```

### Example 2: ML Training with Periodic Checkpoints

```python
from graflow.core.checkpoint import CheckpointManager

@task(inject_context=True)
def ml_training(context):
    channel = context.get_channel()
    epoch = channel.get("epoch", default=0)
    model = channel.get("model")

    if epoch == 0:
        model = initialize_model()
        channel.set("model", model)

    # Training step
    metrics = train_epoch(model, epoch)
    channel.set("epoch", epoch + 1)
    channel.set("metrics", metrics)

    # Checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path, checkpoint_metadata = context.checkpoint(
            metadata={
                "epoch": epoch + 1,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"]
            }
        )
        print(f"Checkpoint saved: {checkpoint_path}")

    # Convergence check
    if metrics["accuracy"] >= 0.95:
        save_model(model)
        return "TRAINING_COMPLETE"
    else:
        context.next_iteration()

# Run training
with workflow("ml_training") as ctx:
    ctx.execute("ml_training", max_steps=100)

# Resume from checkpoint if interrupted
from graflow.core.engine import WorkflowEngine

context, metadata = CheckpointManager.resume_from_checkpoint(
    checkpoint_path="checkpoints/session_67890_step_30.pkl"
)
print(f"Resuming from epoch {metadata.user_metadata.get('epoch')}")

engine = WorkflowEngine()
# Auto-resume: pending_tasks (including current_task_spec) already in queue
engine.execute(context)  # Continues from epoch 30
```

### Example 3: Distributed Execution with Redis

```python
# Worker 1: Start execution
import redis
from graflow.core.checkpoint import CheckpointManager

redis_client = redis.Redis(host='localhost', port=6379, db=0)

with workflow("distributed_workflow") as ctx:
    @task(inject_context=True)
    def distributed_task(context):
        # Process some data
        process_data()

        # Checkpoint (path auto-generated as local file)
        checkpoint_path, metadata = context.checkpoint(
            metadata={"worker": "worker-1"}
        )
        print(f"Checkpoint saved: {checkpoint_path}")
        print(f"Session ID: {context.session_id}")  # Save this for other workers

        # Continue processing
        more_processing()

    exec_context = ExecutionContext.create(
        ctx.graph,
        start_node="distributed_task",
        queue_backend="redis",
        channel_backend="redis",
        config={"redis_client": redis_client}
    )

    try:
        exec_context.execute()
    except Exception as e:
        print(f"Worker 1 crashed: {e}")
        # Checkpoint was saved, can resume on another worker

# Worker 2: Resume from checkpoint
# Use the same Redis instance and checkpoint file
from graflow.core.engine import WorkflowEngine

context, metadata = CheckpointManager.resume_from_checkpoint(
    checkpoint_path="checkpoints/session_12345_step_10.pkl"
)
# context.session_id == "12345" → reconnects to same Redis channel
print(f"Worker 2 resuming session {metadata.session_id} from step {metadata.steps}")

engine = WorkflowEngine()
# RedisTaskQueue: Queue already persisted, no re-queueing happened
# RedisChannel: Data already in Redis at session_12345:*
engine.execute(context)  # Continues where Worker 1 left off
```

---

## Edge Cases and Limitations

### Edge Cases

1. **Checkpoint during parallel execution**:
   - Only the calling task's context is checkpointed
   - Parallel siblings may still be running
   - Resume will not re-execute completed siblings

2. **Multiple checkpoints**:
   - Each checkpoint is independent
   - No automatic cleanup of old checkpoints
   - User responsible for managing checkpoint files

3. **Graph structure changes**:
   - If workflow code changes between checkpoint and resume, behavior is undefined
   - Graph structure saved in checkpoint must match current code

4. **Queue state** (RESOLVED):
   - Pending tasks ARE now saved as full TaskSpec in checkpoint state
   - Includes task_data, retry_count, execution_strategy, etc.
   - Restored to queue on resume
   - Execution continues with pending tasks

5. **Checkpoint timing** (RESOLVED):
   - `context.checkpoint()` **sets a flag**, does NOT create checkpoint immediately
   - Checkpoint created by engine **after task completes successfully**
   - **Task that requested checkpoint is marked as completed**
   - On resume: Execution continues from **next pending task in queue**

   Example:
   ```python
   @task(inject_context=True)
   def process_task(context):
       do_work()
       context.checkpoint(metadata={"stage": "completed"})
       # Task continues to completion
       # Checkpoint created AFTER task returns
   ```

   On resume:
   - `process_task` is already completed (in `completed_tasks`)
   - Execution continues from next task in queue
   - If task crashed before completion, checkpoint was not created

6. **Idempotency for iterative tasks**:
   - For tasks using `next_iteration()`, use channel-based state machine

   Example:
   ```python
   @task(inject_context=True)
   def iterative_task(context):
       channel = context.get_channel()
       state = channel.get("state", default="STEP1")

       if state == "STEP1":
           do_step1()
           channel.set("state", "STEP2")
           context.checkpoint()
           context.next_iteration()  # Re-queue self
       elif state == "STEP2":
           do_step2()
           context.checkpoint()
           # Done
   ```

   On resume: Task re-executes but skips to STEP2 via channel state

### Limitations

**L1**: ~~Task queue state is not persisted~~ (RESOLVED: pending_tasks now saved in checkpoint state)

**L2**: No automatic checkpoint cleanup
- **Impact**: Checkpoint files accumulate over time
- **Workaround**: User must manually delete old checkpoints

**L3**: ~~No checkpoint versioning~~ (RESOLVED: schema_version field added to state.json)
- Schema version "1.0" validates compatibility on resume
- Future format changes will increment version

**L4**: MemoryChannel checkpoints are not portable across workers
- **Impact**: Cannot resume on different worker with MemoryChannel
- **Workaround**: Use RedisChannel for distributed execution

**L5**: Graph structure must remain consistent
- **Impact**: Code changes between checkpoint and resume may fail
- **Workaround**: Maintain backward compatibility in workflow code

**L6**: Atomic write for local backend only (Phase 1)
- **Current**: tmpfile + atomic rename for local filesystem
- **Impact**: Remote backends (Redis/S3) not yet implemented
- **Phase 2**: Add StorageBackend abstraction with atomic guarantees

---

## Future Enhancements

### Phase 2: Automatic Checkpoint Management

```python
class ExecutionContext:
    def __init__(self, ..., auto_checkpoint_interval: Optional[int] = None):
        """
        Args:
            auto_checkpoint_interval: Auto-checkpoint every N steps
        """

    # In execution loop:
    if self.auto_checkpoint_interval and self.steps % self.auto_checkpoint_interval == 0:
        self.checkpoint(metadata={"auto": True})
```

### Phase 3: Checkpoint Manager

```python
class CheckpointManager:
    """Manage multiple checkpoints with retention policies."""

    def list_checkpoints(self, session_id: Optional[str] = None) -> list[CheckpointMetadata]:
        """List available checkpoints."""

    def get_latest(self, session_id: str) -> str:
        """Get latest checkpoint for session."""

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Delete old checkpoints, keeping only last N."""
```

### Phase 4: Incremental Checkpoints

- Delta checkpoints that only save changes since last checkpoint
- Reduces checkpoint file size for large workflows
- Requires checkpoint chain management

### Phase 5: Cloud Storage Integration

```python
class ExecutionContext:
    def checkpoint(self, storage: str = "local"):
        """
        Args:
            storage: "local", "s3", "gcs", "azure"
        """
```

### Phase 6: Checkpoint Validation

- Verify checkpoint integrity before resume
- Check graph compatibility
- Validate channel data consistency

---

## Implementation Checklist

### Phase 1: Core Functionality

- [x] Add `completed_tasks` and `checkpoint_metadata` to `ExecutionContext.__init__`
- [x] Implement `ExecutionContext.mark_task_completed()`
- [ ] Add `checkpoint_requested` and `checkpoint_request_metadata` to `ExecutionContext`
- [ ] Implement `ExecutionContext.request_checkpoint(metadata)`
- [ ] Create `graflow/core/checkpoint.py` module
- [ ] Implement `CheckpointMetadata` dataclass
  - [ ] ISO 8601 timestamp for `created_at`
  - [ ] `create()` class method for timestamp generation
- [ ] Implement `CheckpointManager` class (all class methods)
  - [ ] `create_checkpoint(context, path, metadata)`
  - [ ] `resume_from_checkpoint(checkpoint_path)`
  - [ ] `_infer_backend_from_path(path)`
  - [ ] `_get_pending_task_specs(context)` - Full TaskSpec, not just IDs
  - [ ] `_serialize_task_spec(task_spec)`
  - [ ] `_deserialize_task_spec(task_spec_dict)`
  - [ ] `_validate_schema_version(version)`
  - [ ] `_atomic_save(context, path, state, metadata, backend)` - Atomic write with tmp files
  - [ ] `_get_base_path(path)` - Handle multi-file naming
  - [ ] `_save_checkpoint_state(path, state, backend)`
  - [ ] `_load_checkpoint_state(path, backend)`
  - [ ] `_save_metadata(path, metadata, backend)`
  - [ ] `_load_metadata(path, backend)`
- [ ] Implement `ExecutionContext.get_checkpoint_state()`
- [ ] Implement `TaskExecutionContext.checkpoint()` - Sets flag, does NOT create checkpoint
- [ ] Add `get_pending_task_specs()` to TaskQueue interface
  - [ ] Implement for MemoryTaskQueue (return full TaskSpec list)
  - [ ] Implement for RedisTaskQueue (no-op, already persisted)
- [ ] Modify `WorkflowEngine.execute()` to:
  - [ ] Call `mark_task_completed(task_id)` after each task
  - [ ] Check `context.checkpoint_requested` after task execution
  - [ ] Create checkpoint if flag is set (deferred checkpoint)
- [ ] Schema versioning in checkpoint state ("schema_version": "1.0")
- [ ] Write unit tests for CheckpointManager
- [ ] Write unit tests for MemoryChannel checkpoint/resume
- [ ] Write unit tests for RedisChannel checkpoint/resume
- [ ] Write integration tests for state machine workflows
- [ ] Test deferred checkpoint (flag set during task, created after)
- [ ] Update documentation and examples

### Phase 2: Testing and Validation

- [ ] Test checkpoint/resume with nested workflows
- [ ] Test checkpoint/resume with ParallelGroup
- [ ] Test checkpoint/resume with dynamic tasks
- [ ] Test checkpoint/resume with errors
- [ ] Performance benchmarks
- [ ] Load testing with large workflows

### Phase 3: Documentation

- [ ] API reference documentation
- [ ] User guide with examples
- [ ] Best practices guide
- [ ] Migration guide for existing workflows

---

## Conclusion

This checkpoint/resume design provides a simple, explicit, and backend-agnostic mechanism for workflow state persistence. By leveraging existing serialization infrastructure and channel-based state management, it seamlessly integrates with Graflow's state machine execution model while maintaining minimal overhead and complexity.

The design prioritizes user control (explicit checkpoint calls), portability (works with both Memory and Redis), and simplicity (leverages existing `__getstate__`/`__setstate__`), making it suitable for both local development and production distributed execution.

---

**Document Status**: Draft
**Next Review**: Implementation Phase
**Reviewers**: Graflow Team
