"""Checkpoint management utilities for Graflow workflows."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from graflow.core.context import ExecutionContext
from graflow.queue.base import TaskSpec, TaskStatus


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside a checkpoint."""

    checkpoint_id: str
    session_id: str
    created_at: str
    steps: int
    start_node: Optional[str]
    backend: Dict[str, str]
    user_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "steps": self.steps,
            "start_node": self.start_node,
            "backend": self.backend,
            "user_metadata": self.user_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        return cls(
            checkpoint_id=data["checkpoint_id"],
            session_id=data["session_id"],
            created_at=data["created_at"],
            steps=data["steps"],
            start_node=data.get("start_node"),
            backend=data.get("backend", {}),
            user_metadata=data.get("user_metadata", {}),
        )

    @classmethod
    def create(
        cls,
        *,
        checkpoint_id: str,
        session_id: str,
        steps: int,
        start_node: Optional[str],
        backend: Dict[str, str],
        user_metadata: Dict[str, Any],
    ) -> "CheckpointMetadata":
        created_at = datetime.now(timezone.utc).isoformat()
        return cls(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            created_at=created_at,
            steps=steps,
            start_node=start_node,
            backend=backend,
            user_metadata=user_metadata,
        )


class CheckpointManager:
    """Create and restore workflow checkpoints."""

    @classmethod
    def create_checkpoint(
        cls,
        context: ExecutionContext,
        *,
        path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        include_current_task: Optional[TaskSpec] = None,
    ) -> Tuple[str, CheckpointMetadata]:
        """Create a checkpoint for the provided execution context."""

        backend = cls._infer_backend_from_path(path)
        if backend != "local":
            raise NotImplementedError(f"Checkpoint backend '{backend}' is not implemented yet")

        base_path = cls._resolve_base_path(context, path)
        cls._ensure_directory(os.path.dirname(base_path) or ".")

        pending_specs = cls._get_pending_task_specs(context)
        if include_current_task is not None:
            pending_specs.insert(0, include_current_task)

        serialized_specs = [cls._serialize_task_spec(spec) for spec in pending_specs]

        state = context.get_checkpoint_state()
        state.update(
            {
                "pending_tasks": serialized_specs,
                "resume_from_current_task": include_current_task is not None,
            }
        )

        checkpoint_id = cls._generate_checkpoint_id(context)
        metadata_obj = CheckpointMetadata.create(
            checkpoint_id=checkpoint_id,
            session_id=context.session_id,
            steps=context.steps,
            start_node=context.start_node,
            backend=state["backend"],
            user_metadata=dict(metadata) if metadata else {},
        )

        pickle_path = f"{base_path}.pkl"
        state_path = f"{base_path}.state.json"
        meta_path = f"{base_path}.meta.json"

        # Persist context and checkpoint metadata
        context.save(pickle_path)
        cls._save_json(state_path, state)
        cls._save_json(meta_path, metadata_obj.to_dict())

        context.checkpoint_metadata = metadata_obj.to_dict()
        context.last_checkpoint_path = pickle_path

        return pickle_path, metadata_obj

    @classmethod
    def resume_from_checkpoint(
        cls, checkpoint_path: str
    ) -> Tuple[ExecutionContext, CheckpointMetadata]:
        """Restore execution context and metadata from a checkpoint."""

        backend = cls._infer_backend_from_path(checkpoint_path)
        if backend != "local":
            raise NotImplementedError(f"Checkpoint backend '{backend}' is not implemented yet")

        base_path = cls._get_base_path(checkpoint_path)
        pickle_path = f"{base_path}.pkl"
        state_path = f"{base_path}.state.json"
        meta_path = f"{base_path}.meta.json"

        context = ExecutionContext.load(pickle_path)
        state = cls._load_json(state_path)
        metadata = CheckpointMetadata.from_dict(cls._load_json(meta_path))

        context.completed_tasks = set(state.get("completed_tasks", []))
        context.cycle_controller.cycle_counts.update(state.get("cycle_counts", {}))
        context.checkpoint_metadata = metadata.to_dict()
        context.last_checkpoint_path = pickle_path

        pending_specs = state.get("pending_tasks", [])
        if context._queue_backend_type != "redis":
            # Reset queue state reconstructed during deserialization
            context.task_queue.cleanup()
            for spec_data in pending_specs:
                task_spec = cls._deserialize_task_spec(spec_data, context)
                context.task_queue.enqueue(task_spec)

        context.clear_checkpoint_request()
        return context, metadata

    # === Helpers ===

    @staticmethod
    def _generate_checkpoint_id(context: ExecutionContext) -> str:
        timestamp = int(time.time())
        return f"{context.session_id}_{context.steps}_{timestamp}"

    @staticmethod
    def _infer_backend_from_path(path: Optional[str]) -> str:
        if path is None:
            return "local"
        if path.startswith("redis://"):
            return "redis"
        if path.startswith("s3://"):
            return "s3"
        return "local"

    @classmethod
    def _resolve_base_path(cls, context: ExecutionContext, path: Optional[str]) -> str:
        if path is None:
            directory = os.path.join("checkpoints", context.session_id)
            cls._ensure_directory(directory)
            filename = f"session_{context.session_id}_step_{context.steps}_{int(time.time())}"
            return os.path.join(directory, filename)
        return cls._get_base_path(path)

    @staticmethod
    def _get_base_path(path: str) -> str:
        return path[:-4] if path.endswith(".pkl") else path

    @staticmethod
    def _ensure_directory(directory: str) -> None:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def _save_json(path: str, data: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _get_pending_task_specs(context: ExecutionContext) -> List[TaskSpec]:
        specs = context.task_queue.get_pending_task_specs()
        return list(specs) if specs else []

    @staticmethod
    def _serialize_task_spec(task_spec: TaskSpec) -> Dict[str, Any]:
        return {
            "task_id": task_spec.task_id,
            "task_data": task_spec.task_data,
            "status": task_spec.status.value,
            "created_at": task_spec.created_at,
            "strategy": task_spec.strategy,
            "retry_count": task_spec.retry_count,
            "max_retries": task_spec.max_retries,
            "last_error": task_spec.last_error,
        }

    @staticmethod
    def _deserialize_task_spec(data: Dict[str, Any], context: ExecutionContext) -> TaskSpec:
        task_id = data["task_id"]
        task_data = data.get("task_data")

        executable = None
        if task_data:
            try:
                executable = context.task_resolver.resolve_task(task_data)
            except Exception:
                executable = None

        if executable is None:
            from graflow.core.task import Task

            executable = Task(task_id, register_to_context=False)

        task_spec = TaskSpec(
            executable=executable,
            execution_context=context,
            strategy=data.get("strategy", "reference"),
            status=TaskStatus.READY,
            created_at=data.get("created_at", time.time()),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            last_error=data.get("last_error"),
        )
        return task_spec
