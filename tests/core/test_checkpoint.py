"""Comprehensive tests for checkpoint management functionality.

This module contains test coverage for checkpoint creation, restoration, and
metadata management in Graflow workflows. Checkpoints are essential for:

- Workflow state persistence and recovery
- Long-running workflow resumption after interruption
- Distributed execution fault tolerance
- Debugging and workflow state inspection

The tests validate that CheckpointManager can correctly save and restore
ExecutionContext state, including pending tasks, cycle counts, and results.
"""

import json
import os
import tempfile

import pytest

from graflow.core.checkpoint import CheckpointManager, CheckpointMetadata
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph
from graflow.core.task import TaskWrapper
from graflow.queue.base import TaskSpec, TaskStatus


class TestCheckpointMetadata:
    """Test CheckpointMetadata creation, serialization, and deserialization."""

    def test_metadata_creation(self):
        """Test creating CheckpointMetadata with all required fields."""
        metadata = CheckpointMetadata.create(
            checkpoint_id="test_checkpoint_123",
            session_id="session_abc",
            steps=5,
            start_node="task_a",
            backend={"queue": "memory", "channel": "memory"},
            user_metadata={"project": "test_project", "version": "1.0"}
        )

        assert metadata.checkpoint_id == "test_checkpoint_123"
        assert metadata.session_id == "session_abc"
        assert metadata.steps == 5
        assert metadata.start_node == "task_a"
        assert metadata.backend == {"queue": "memory", "channel": "memory"}
        assert metadata.user_metadata == {"project": "test_project", "version": "1.0"}
        assert metadata.created_at is not None

    def test_metadata_to_dict(self):
        """Test converting CheckpointMetadata to dictionary."""
        metadata = CheckpointMetadata.create(
            checkpoint_id="test_123",
            session_id="session_xyz",
            steps=3,
            start_node="start",
            backend={"queue": "redis"},
            user_metadata={"env": "test"}
        )

        metadata_dict = metadata.to_dict()

        assert isinstance(metadata_dict, dict)
        assert metadata_dict["checkpoint_id"] == "test_123"
        assert metadata_dict["session_id"] == "session_xyz"
        assert metadata_dict["steps"] == 3
        assert metadata_dict["start_node"] == "start"
        assert metadata_dict["backend"] == {"queue": "redis"}
        assert metadata_dict["user_metadata"] == {"env": "test"}
        assert "created_at" in metadata_dict

    def test_metadata_from_dict(self):
        """Test creating CheckpointMetadata from dictionary."""
        data = {
            "checkpoint_id": "checkpoint_789",
            "session_id": "session_456",
            "created_at": "2025-01-01T12:00:00Z",
            "steps": 7,
            "start_node": "initial_task",
            "backend": {"queue": "memory", "channel": "redis"},
            "user_metadata": {"owner": "alice"}
        }

        metadata = CheckpointMetadata.from_dict(data)

        assert metadata.checkpoint_id == "checkpoint_789"
        assert metadata.session_id == "session_456"
        assert metadata.created_at == "2025-01-01T12:00:00Z"
        assert metadata.steps == 7
        assert metadata.start_node == "initial_task"
        assert metadata.backend == {"queue": "memory", "channel": "redis"}
        assert metadata.user_metadata == {"owner": "alice"}

    def test_metadata_serialization_roundtrip(self):
        """Test that metadata survives to_dict/from_dict roundtrip."""
        original = CheckpointMetadata.create(
            checkpoint_id="roundtrip_test",
            session_id="session_roundtrip",
            steps=10,
            start_node="node_x",
            backend={"queue": "memory"},
            user_metadata={"test": "roundtrip"}
        )

        # Convert to dict and back
        metadata_dict = original.to_dict()
        restored = CheckpointMetadata.from_dict(metadata_dict)

        assert restored.checkpoint_id == original.checkpoint_id
        assert restored.session_id == original.session_id
        assert restored.created_at == original.created_at
        assert restored.steps == original.steps
        assert restored.start_node == original.start_node
        assert restored.backend == original.backend
        assert restored.user_metadata == original.user_metadata


class TestCheckpointManagerBasic:
    """Test basic CheckpointManager functionality."""

    def test_checkpoint_id_generation(self):
        """Test checkpoint ID generation includes session_id, steps, and timestamp."""
        graph = TaskGraph()
        context = ExecutionContext.create(graph, start_node=None, max_steps=10)

        # Set some execution state
        context.increment_step()
        context.increment_step()

        checkpoint_id = CheckpointManager._generate_checkpoint_id(context)

        # Checkpoint ID format: {session_id}_{steps}_{timestamp}
        assert context.session_id in checkpoint_id
        assert "_2_" in checkpoint_id  # steps = 2
        assert checkpoint_id.count("_") == 2

    def test_backend_inference_local(self):
        """Test inferring local backend from path."""
        assert CheckpointManager._infer_backend_from_path(None) == "local"
        assert CheckpointManager._infer_backend_from_path("/tmp/checkpoint.pkl") == "local"
        assert CheckpointManager._infer_backend_from_path("checkpoints/test.pkl") == "local"

    def test_backend_inference_redis(self):
        """Test inferring Redis backend from path."""
        assert CheckpointManager._infer_backend_from_path("redis://checkpoint") == "redis"

    def test_backend_inference_s3(self):
        """Test inferring S3 backend from path."""
        assert CheckpointManager._infer_backend_from_path("s3://bucket/checkpoint") == "s3"

    def test_base_path_extraction(self):
        """Test extracting base path from checkpoint path."""
        assert CheckpointManager._get_base_path("checkpoint.pkl") == "checkpoint"
        assert CheckpointManager._get_base_path("/tmp/test.pkl") == "/tmp/test"
        assert CheckpointManager._get_base_path("checkpoint") == "checkpoint"

    def test_resolve_base_path_auto_generation(self):
        """Test auto-generating base path when no path provided."""
        graph = TaskGraph()
        context = ExecutionContext.create(graph, start_node=None)

        base_path = CheckpointManager._resolve_base_path(context, None)

        # Should generate path in checkpoints/{session_id}/session_{session_id}_step_{steps}_{timestamp}
        assert "checkpoints" in base_path
        assert context.session_id in base_path
        assert f"step_{context.steps}" in base_path

    def test_resolve_base_path_provided(self):
        """Test using provided base path."""
        graph = TaskGraph()
        context = ExecutionContext.create(graph, start_node=None)

        base_path = CheckpointManager._resolve_base_path(context, "/custom/path/checkpoint.pkl")

        assert base_path == "/custom/path/checkpoint"


class TestCheckpointCreation:
    """Test checkpoint creation functionality."""

    def test_create_checkpoint_basic(self):
        """Test creating a basic checkpoint with minimal state."""
        graph = TaskGraph()

        @task("task_a")
        def task_a_func():
            return "task_a_result"

        graph.add_node(task_a_func, "task_a")

        # Create context without start_node to avoid queue initialization issues
        context = ExecutionContext.create(graph, start_node=None, max_steps=10)
        # Manually set start_node after context creation
        context.start_node = "task_a"

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint")

            # Create checkpoint
            pkl_path, metadata = CheckpointManager.create_checkpoint(
                context,
                path=checkpoint_path
            )

            # Verify checkpoint files exist
            assert os.path.exists(pkl_path)
            assert os.path.exists(f"{checkpoint_path}.state.json")
            assert os.path.exists(f"{checkpoint_path}.meta.json")

            # Verify metadata
            assert metadata.session_id == context.session_id
            assert metadata.steps == context.steps
            assert metadata.start_node == "task_a"

    def test_create_checkpoint_with_execution_state(self):
        """Test creating checkpoint with execution results and cycle state."""
        graph = TaskGraph()

        @task("task_a")
        def task_a_func():
            return "task_a_result"

        graph.add_node(task_a_func, "task_a")

        context = ExecutionContext.create(graph, "task_a", max_steps=15)

        # Set execution state
        context.set_result("task_a", {"status": "completed", "value": 42})
        context.set_result("task_b", "result_b")
        context.increment_step()
        context.increment_step()
        context.cycle_controller.cycle_counts["task_a"] = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint_with_state")

            pkl_path, metadata = CheckpointManager.create_checkpoint(
                context,
                path=checkpoint_path
            )

            # Verify state file contains execution state
            with open(f"{checkpoint_path}.state.json") as f:
                state = json.load(f)

            # Results are stored in channel (saved in .pkl), not in state.json
            assert state["cycle_counts"]["task_a"] == 3
            assert state["steps"] == 2
            assert state["completed_tasks"] == []  # No tasks marked as completed yet

    def test_create_checkpoint_with_user_metadata(self):
        """Test creating checkpoint with custom user metadata."""
        graph = TaskGraph()
        context = ExecutionContext.create(graph, start_node=None)

        user_metadata = {
            "project": "workflow_project",
            "version": "2.0",
            "author": "test_user",
            "tags": ["production", "critical"]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint_metadata")

            pkl_path, metadata = CheckpointManager.create_checkpoint(
                context,
                path=checkpoint_path,
                metadata=user_metadata
            )

            # Verify user metadata in metadata file
            with open(f"{checkpoint_path}.meta.json") as f:
                meta_dict = json.load(f)

            assert meta_dict["user_metadata"] == user_metadata

    def test_create_checkpoint_with_pending_tasks(self):
        """Test creating checkpoint with pending tasks in queue."""
        graph = TaskGraph()

        @task("task_a")
        def task_a_func():
            return "a"

        @task("task_b")
        def task_b_func():
            return "b"

        graph.add_node(task_a_func, "task_a")
        graph.add_node(task_b_func, "task_b")
        graph.add_edge("task_a", "task_b")

        context = ExecutionContext.create(graph, "task_a", max_steps=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint_pending")

            # Create checkpoint before execution
            pkl_path, metadata = CheckpointManager.create_checkpoint(
                context,
                path=checkpoint_path
            )

            # Verify pending tasks in state file
            with open(f"{checkpoint_path}.state.json") as f:
                state = json.load(f)

            assert "pending_tasks" in state
            assert len(state["pending_tasks"]) >= 1
            assert any(task["task_id"] == "task_a" for task in state["pending_tasks"])

    def test_create_checkpoint_with_current_task(self):
        """Test creating checkpoint including currently running task."""
        graph = TaskGraph()
        task_wrapper = TaskWrapper("running_task", lambda: "result")
        graph.add_node(task_wrapper, "running_task")

        context = ExecutionContext.create(graph, "running_task", max_steps=10)

        # Create TaskSpec for currently running task
        current_task_spec = TaskSpec(
            executable=task_wrapper,
            execution_context=context,
            status=TaskStatus.RUNNING
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint_current")

            pkl_path, metadata = CheckpointManager.create_checkpoint(
                context,
                path=checkpoint_path,
                include_current_task=current_task_spec
            )

            # Verify current task is first in pending tasks
            with open(f"{checkpoint_path}.state.json") as f:
                state = json.load(f)

            assert state["resume_from_current_task"] is True
            assert len(state["pending_tasks"]) >= 1
            assert state["pending_tasks"][0]["task_id"] == "running_task"

    def test_create_checkpoint_auto_path_generation(self):
        """Test checkpoint creation with auto-generated path."""
        graph = TaskGraph()
        context = ExecutionContext.create(graph, start_node=None)

        # Don't provide path - should auto-generate
        pkl_path, metadata = CheckpointManager.create_checkpoint(context)

        try:
            # Verify files were created in auto-generated location
            assert os.path.exists(pkl_path)
            assert pkl_path.endswith(".pkl")

            # Verify path contains session_id
            assert context.session_id in pkl_path

            base_path = pkl_path[:-4]
            assert os.path.exists(f"{base_path}.state.json")
            assert os.path.exists(f"{base_path}.meta.json")
        finally:
            # Cleanup auto-generated files
            if os.path.exists(pkl_path):
                base_path = pkl_path[:-4]
                for ext in [".pkl", ".state.json", ".meta.json"]:
                    file_path = f"{base_path}{ext}"
                    if os.path.exists(file_path):
                        os.remove(file_path)

    def test_create_checkpoint_updates_context(self):
        """Test that creating checkpoint updates context metadata."""
        graph = TaskGraph()
        context = ExecutionContext.create(graph, start_node=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint_context")

            pkl_path, metadata = CheckpointManager.create_checkpoint(
                context,
                path=checkpoint_path
            )

            # Verify context was updated with checkpoint info
            assert context.last_checkpoint_path == pkl_path
            assert context.checkpoint_metadata is not None
            assert context.checkpoint_metadata["checkpoint_id"] == metadata.checkpoint_id
            assert context.checkpoint_metadata["session_id"] == context.session_id


class TestCheckpointRestoration:
    """Test checkpoint restoration functionality."""

    def test_resume_from_checkpoint_basic(self):
        """Test resuming from a basic checkpoint."""
        graph = TaskGraph()

        @task("task_a")
        def task_a_func():
            return "completed"

        graph.add_node(task_a_func, "task_a")

        original_context = ExecutionContext.create(graph, "task_a", max_steps=10)
        original_context.set_result("task_a", "completed")
        original_context.increment_step()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "resume_test")

            # Create checkpoint
            pkl_path, _ = CheckpointManager.create_checkpoint(
                original_context,
                path=checkpoint_path
            )

            # Resume from checkpoint
            restored_context, metadata = CheckpointManager.resume_from_checkpoint(pkl_path)

            # Verify basic attributes
            assert restored_context.session_id == original_context.session_id
            assert restored_context.start_node == "task_a"
            assert restored_context.max_steps == 10
            assert restored_context.steps == 1

            # Verify results
            assert restored_context.get_result("task_a") == "completed"

    def test_resume_preserves_execution_state(self):
        """Test that resume preserves complete execution state."""
        graph = TaskGraph()

        @task("task_x")
        def task_x_func():
            return {"data": [1, 2, 3], "status": "ok"}

        graph.add_node(task_x_func, "task_x")

        original_context = ExecutionContext.create(graph, "task_x", max_steps=20)

        # Set complex state
        original_context.set_result("task_x", {"data": [1, 2, 3], "status": "ok"})
        original_context.set_result("task_y", "result_y")
        original_context.increment_step()
        original_context.increment_step()
        original_context.increment_step()
        original_context.cycle_controller.cycle_counts["task_x"] = 5
        original_context.cycle_controller.set_node_max_cycles("task_x", 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "state_test")

            pkl_path, _ = CheckpointManager.create_checkpoint(
                original_context,
                path=checkpoint_path
            )

            restored_context, metadata = CheckpointManager.resume_from_checkpoint(pkl_path)

            # Verify all state preserved
            assert restored_context.steps == 3
            assert restored_context.get_result("task_x") == {"data": [1, 2, 3], "status": "ok"}
            assert restored_context.get_result("task_y") == "result_y"
            assert restored_context.cycle_controller.cycle_counts["task_x"] == 5
            assert restored_context.cycle_controller.get_max_cycles_for_node("task_x") == 10

    def test_resume_with_pending_tasks(self):
        """Test resuming checkpoint with pending tasks in queue."""
        graph = TaskGraph()

        @task("task_1")
        def task_1_func():
            return "result_1"

        @task("task_2")
        def task_2_func():
            return "result_2"

        graph.add_node(task_1_func, "task_1")
        graph.add_node(task_2_func, "task_2")
        graph.add_edge("task_1", "task_2")

        original_context = ExecutionContext.create(graph, "task_1", max_steps=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "pending_test")

            # Create checkpoint before execution
            pkl_path, _ = CheckpointManager.create_checkpoint(
                original_context,
                path=checkpoint_path
            )

            # Resume and verify queue is restored
            restored_context, metadata = CheckpointManager.resume_from_checkpoint(pkl_path)

            # Get pending tasks
            pending = restored_context.task_queue.get_pending_task_specs()
            assert pending is not None
            assert len(list(pending)) >= 1

    def test_resume_metadata_preservation(self):
        """Test that resume preserves checkpoint metadata."""
        graph = TaskGraph()
        original_context = ExecutionContext.create(graph, start_node=None)

        user_metadata = {"project": "test", "version": "1.0"}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "metadata_test")

            pkl_path, original_metadata = CheckpointManager.create_checkpoint(
                original_context,
                path=checkpoint_path,
                metadata=user_metadata
            )

            restored_context, restored_metadata = CheckpointManager.resume_from_checkpoint(pkl_path)

            # Verify metadata preserved
            assert restored_metadata.checkpoint_id == original_metadata.checkpoint_id
            assert restored_metadata.session_id == original_metadata.session_id
            assert restored_metadata.user_metadata == user_metadata
            assert restored_context.checkpoint_metadata == restored_metadata.to_dict()

    def test_resume_clears_checkpoint_request(self):
        """Test that resume clears any pending checkpoint request flags."""
        graph = TaskGraph()
        original_context = ExecutionContext.create(graph, start_node=None)

        # Request checkpoint (simulate flag being set)
        original_context.request_checkpoint()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "clear_test")

            pkl_path, _ = CheckpointManager.create_checkpoint(
                original_context,
                path=checkpoint_path
            )

            # Resume should clear checkpoint request
            restored_context, _ = CheckpointManager.resume_from_checkpoint(pkl_path)

            # Verify checkpoint request is cleared
            assert not restored_context.checkpoint_requested


class TestCheckpointErrorHandling:
    """Test error handling in checkpoint operations."""

    def test_create_checkpoint_nonlocal_backend_error(self):
        """Test that non-local backends raise NotImplementedError."""
        graph = TaskGraph()
        context = ExecutionContext.create(graph, start_node=None)

        # Redis backend should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="redis"):
            CheckpointManager.create_checkpoint(context, path="redis://checkpoint")

        # S3 backend should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="s3"):
            CheckpointManager.create_checkpoint(context, path="s3://bucket/checkpoint")

    def test_resume_checkpoint_nonlocal_backend_error(self):
        """Test that resuming from non-local backends raises NotImplementedError."""
        # Redis backend should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="redis"):
            CheckpointManager.resume_from_checkpoint("redis://checkpoint.pkl")

        # S3 backend should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="s3"):
            CheckpointManager.resume_from_checkpoint("s3://bucket/checkpoint.pkl")

    def test_resume_checkpoint_missing_files(self):
        """Test resuming from checkpoint with missing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "missing.pkl")

            # Should raise FileNotFoundError for missing checkpoint
            with pytest.raises(FileNotFoundError):
                CheckpointManager.resume_from_checkpoint(checkpoint_path)

    def test_resume_checkpoint_corrupted_state(self):
        """Test resuming from checkpoint with corrupted state file."""
        graph = TaskGraph()
        context = ExecutionContext.create(graph, start_node=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "corrupted")

            # Create valid checkpoint
            pkl_path, _ = CheckpointManager.create_checkpoint(
                context,
                path=checkpoint_path
            )

            # Corrupt the state file
            state_path = f"{checkpoint_path}.state.json"
            with open(state_path, "w") as f:
                f.write("corrupted json content")

            # Should raise JSONDecodeError
            with pytest.raises(json.JSONDecodeError):
                CheckpointManager.resume_from_checkpoint(pkl_path)

    def test_resume_checkpoint_corrupted_metadata(self):
        """Test resuming from checkpoint with corrupted metadata file."""
        graph = TaskGraph()
        context = ExecutionContext.create(graph, start_node=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "corrupted_meta")

            # Create valid checkpoint
            pkl_path, _ = CheckpointManager.create_checkpoint(
                context,
                path=checkpoint_path
            )

            # Corrupt the metadata file
            meta_path = f"{checkpoint_path}.meta.json"
            with open(meta_path, "w") as f:
                f.write("invalid json")

            # Should raise JSONDecodeError
            with pytest.raises(json.JSONDecodeError):
                CheckpointManager.resume_from_checkpoint(pkl_path)


class TestCheckpointIntegration:
    """Integration tests for checkpoint functionality with workflows."""

    def test_checkpoint_and_resume_simple_workflow(self):
        """Test checkpointing and resuming a simple workflow."""
        graph = TaskGraph()

        @task("step_1")
        def step_1_func():
            return "step_1_complete"

        @task("step_2")
        def step_2_func():
            return "step_2_complete"

        graph.add_node(step_1_func, "step_1")
        graph.add_node(step_2_func, "step_2")
        graph.add_edge("step_1", "step_2")

        # Execute first step and checkpoint
        context = ExecutionContext.create(graph, "step_1", max_steps=10)

        # Execute only the first task
        next_task_id = context.task_queue.get_next_task()
        assert next_task_id == "step_1"
        task_obj = context.graph.get_node(next_task_id)
        task_obj.set_execution_context(context)
        result = task_obj.run()
        context.set_result(next_task_id, result)
        context.mark_task_completed(next_task_id)
        context.increment_step()

        # Schedule successors (step_2) to queue
        for successor_id in graph.successors("step_1"):
            successor_task = graph.get_node(successor_id)
            task_spec = TaskSpec(executable=successor_task, execution_context=context)
            context.task_queue.enqueue(task_spec)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "workflow_checkpoint")

            # Create checkpoint after first step
            pkl_path, metadata = CheckpointManager.create_checkpoint(
                context,
                path=checkpoint_path
            )

            # Resume from checkpoint
            restored_context, restored_metadata = CheckpointManager.resume_from_checkpoint(pkl_path)

            # Verify first step result is preserved
            assert restored_context.get_result("step_1") == "step_1_complete"

            # Continue execution with second step
            next_task_id = restored_context.task_queue.get_next_task()
            if next_task_id:
                task_obj = restored_context.graph.get_node(next_task_id)
                task_obj.set_execution_context(restored_context)
                result = task_obj.run()
                restored_context.set_result(next_task_id, result)

            # Verify both steps completed
            assert restored_context.get_result("step_1") == "step_1_complete"
            assert restored_context.get_result("step_2") == "step_2_complete"

    def test_multiple_checkpoints_same_session(self):
        """Test creating multiple checkpoints for the same session."""
        graph = TaskGraph()
        context = ExecutionContext.create(graph, start_node=None, max_steps=20)

        checkpoints = []

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint at step 0
            checkpoint_0 = os.path.join(tmpdir, "checkpoint_0")
            pkl_0, meta_0 = CheckpointManager.create_checkpoint(context, path=checkpoint_0)
            checkpoints.append((pkl_0, meta_0))

            # Advance and create checkpoint at step 3
            context.increment_step()
            context.increment_step()
            context.increment_step()
            checkpoint_3 = os.path.join(tmpdir, "checkpoint_3")
            pkl_3, meta_3 = CheckpointManager.create_checkpoint(context, path=checkpoint_3)
            checkpoints.append((pkl_3, meta_3))

            # Advance and create checkpoint at step 7
            context.increment_step()
            context.increment_step()
            context.increment_step()
            context.increment_step()
            checkpoint_7 = os.path.join(tmpdir, "checkpoint_7")
            pkl_7, meta_7 = CheckpointManager.create_checkpoint(context, path=checkpoint_7)
            checkpoints.append((pkl_7, meta_7))

            # Verify all checkpoints are different
            assert meta_0.checkpoint_id != meta_3.checkpoint_id != meta_7.checkpoint_id

            # Resume from each checkpoint and verify steps
            restored_0, _ = CheckpointManager.resume_from_checkpoint(pkl_0)
            assert restored_0.steps == 0

            restored_3, _ = CheckpointManager.resume_from_checkpoint(pkl_3)
            assert restored_3.steps == 3

            restored_7, _ = CheckpointManager.resume_from_checkpoint(pkl_7)
            assert restored_7.steps == 7

    def test_checkpoint_with_channel_data(self):
        """Test checkpoint preserves channel data."""
        graph = TaskGraph()
        context = ExecutionContext.create(graph, start_node=None, channel_backend="memory")

        # Set channel data
        context.channel.set("shared_value", 42)
        context.channel.set("config", {"mode": "test", "retry": 3})

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "channel_checkpoint")

            pkl_path, _ = CheckpointManager.create_checkpoint(context, path=checkpoint_path)

            # Resume and verify channel data
            restored_context, _ = CheckpointManager.resume_from_checkpoint(pkl_path)

            assert restored_context.channel.get("shared_value") == 42
            assert restored_context.channel.get("config") == {"mode": "test", "retry": 3}
