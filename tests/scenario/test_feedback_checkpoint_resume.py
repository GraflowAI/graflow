"""Scenario test for HITL feedback with timeout and checkpoint resume.

This test demonstrates the complete HITL workflow:
1. Workflow starts and requests feedback
2. Feedback request times out (no response within timeout)
3. Checkpoint is created automatically
4. External process provides feedback via API
5. Workflow resumes from checkpoint
6. Workflow completes successfully
"""

from __future__ import annotations

import os
import time

import pytest

from graflow.core.checkpoint import CheckpointManager
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.workflow import workflow
from graflow.hitl.types import FeedbackResponse, FeedbackType


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def feedback_manager_filesystem(tmp_path):
    """Create FeedbackManager with filesystem backend."""
    from graflow.hitl.manager import FeedbackManager

    feedback_dir = tmp_path / "feedback_data"
    feedback_dir.mkdir()

    return FeedbackManager(backend="filesystem", backend_config={"data_dir": str(feedback_dir)})


class TestFeedbackCheckpointScenario:
    """Scenario tests for HITL feedback with checkpoint resume."""

    def test_timeout_checkpoint_api_feedback_resume(self, tmp_path, temp_checkpoint_dir, feedback_manager_filesystem):
        """Complete scenario: timeout -> checkpoint -> API feedback -> resume.

        This test simulates the real-world HITL workflow:
        1. Task requests approval with short timeout
        2. No response arrives -> timeout occurs
        3. Checkpoint created manually after timeout
        4. External system provides feedback via API
        5. Workflow resumes from checkpoint
        6. Task completes with approved result
        """
        checkpoint_path = None
        feedback_id = None

        # Step 1: Define workflow with feedback request
        with workflow("deployment_approval") as wf:

            @task(inject_context=True)
            def request_deployment_approval(context):
                """Request deployment approval with short timeout."""
                response = context.request_feedback(
                    feedback_type="approval",
                    prompt="Approve deployment v2.0.0 to production?",
                    timeout=0.1,  # Very short timeout for immediate timeout in tests
                )
                return response.approved

            @task(inject_context=True)
            def execute_deployment(context):
                """Execute deployment if approved."""
                # Get approval from previous task via channel
                channel = context.get_channel()
                approved = channel.get("request_deployment_approval.__result__")
                return "DEPLOYED" if approved else "CANCELLED"

            # Define workflow
            _ = request_deployment_approval >> execute_deployment

            # Step 2: Create execution context with feedback manager
            # Find start node from graph
            start_nodes = wf.graph.get_start_nodes()
            assert len(start_nodes) == 1
            start_node = start_nodes[0]

            exec_context = ExecutionContext.create(wf.graph, start_node=start_node, channel_backend="memory", config={})

            # Inject feedback manager
            exec_context.feedback_manager = feedback_manager_filesystem

            # Step 3: Execute workflow (should timeout and return None)
            engine = WorkflowEngine()
            result = engine.execute(exec_context)

            # Step 4: Verify timeout handling
            # Engine catches FeedbackTimeoutError internally and returns None
            assert result is None, "Expected None when workflow times out waiting for feedback"

            # Step 5: Verify checkpoint was created automatically
            checkpoint_path = exec_context.last_checkpoint_path
            assert checkpoint_path is not None, "Expected checkpoint to be created on timeout"
            assert os.path.exists(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"

            # Get feedback_id from checkpoint metadata
            assert hasattr(exec_context, "checkpoint_metadata")
            feedback_id = exec_context.checkpoint_metadata.get("user_metadata", {}).get("feedback_id")
            assert feedback_id is not None, "Expected feedback_id in checkpoint metadata"

            print(f"\n[Test] Timeout occurred, checkpoint created at: {checkpoint_path}")
            print(f"[Test] Feedback ID: {feedback_id}")

        # Step 6: Simulate external API providing feedback
        # (In real scenario, this would be done by external process via HTTP API)
        print("\n[Test] Simulating API feedback submission...")

        feedback_response = FeedbackResponse(
            feedback_id=feedback_id,
            response_type=FeedbackType.APPROVAL,
            approved=True,
            reason="Approved by DevOps team",
            responded_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            responded_by="devops@example.com",
        )

        # Submit feedback via FeedbackManager (simulates API endpoint behavior)
        success = feedback_manager_filesystem.provide_feedback(feedback_id, feedback_response)
        assert success is True
        print("[Test] Feedback submitted successfully")

        # Step 7: Resume from checkpoint
        print("\n[Test] Resuming from checkpoint...")

        resumed_context, _metadata = CheckpointManager.resume_from_checkpoint(checkpoint_path)

        # Re-inject feedback manager (not serialized in checkpoint)
        resumed_context.feedback_manager = feedback_manager_filesystem

        # Step 8: Execute resumed workflow
        result = engine.execute(resumed_context)

        # Step 9: Verify workflow completed successfully
        assert result == "DEPLOYED", f"Expected 'DEPLOYED', got {result}"

        print(f"[Test] Workflow completed successfully: {result}")

        # Cleanup
        os.remove(checkpoint_path)

    def test_timeout_checkpoint_rejection_resume(self, tmp_path, temp_checkpoint_dir, feedback_manager_filesystem):
        """Scenario: timeout -> checkpoint -> reject via API -> resume -> cancelled.

        This tests the rejection path of the approval workflow.
        """
        checkpoint_path = None
        feedback_id = None

        # Define workflow
        with workflow("deployment_with_rejection") as wf:

            @task(inject_context=True)
            def request_approval(context):
                """Request approval."""
                response = context.request_feedback(
                    feedback_type="approval",
                    prompt="Approve risky deployment?",
                    timeout=0.1,  # Very short timeout for immediate timeout in tests
                )
                return response.approved

            @task(inject_context=True)
            def execute_if_approved(context):
                """Execute only if approved."""
                channel = context.get_channel()
                approved = channel.get("request_approval.__result__")
                return "EXECUTED" if approved else "SKIPPED"

            _ = request_approval >> execute_if_approved

            # Create context
            # Find start node from graph
            start_nodes = wf.graph.get_start_nodes()
            assert len(start_nodes) == 1
            start_node = start_nodes[0]

            exec_context = ExecutionContext.create(wf.graph, start_node=start_node, channel_backend="memory", config={})
            exec_context.feedback_manager = feedback_manager_filesystem

            # Execute and expect timeout
            engine = WorkflowEngine()

            # Execute workflow (should timeout and return None)
            result = engine.execute(exec_context)

            # Verify timeout handling
            assert result is None, "Expected None when workflow times out"

            # Verify checkpoint created automatically
            checkpoint_path = exec_context.last_checkpoint_path
            assert checkpoint_path is not None
            assert os.path.exists(checkpoint_path)

            # Get feedback_id from checkpoint metadata
            feedback_id = exec_context.checkpoint_metadata.get("user_metadata", {}).get("feedback_id")
            assert feedback_id is not None

        # Provide REJECTION via API
        feedback_response = FeedbackResponse(
            feedback_id=feedback_id,
            response_type=FeedbackType.APPROVAL,
            approved=False,  # REJECTED
            reason="Too risky, need more testing",
            responded_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            responded_by="security@example.com",
        )

        success = feedback_manager_filesystem.provide_feedback(feedback_id, feedback_response)
        assert success is True

        # Resume from checkpoint
        resumed_context, _metadata = CheckpointManager.resume_from_checkpoint(checkpoint_path)
        resumed_context.feedback_manager = feedback_manager_filesystem

        # Execute resumed workflow
        result = engine.execute(resumed_context)

        # Verify rejection resulted in SKIPPED
        assert result == "SKIPPED"

        # Cleanup
        os.remove(checkpoint_path)

    def test_timeout_checkpoint_text_feedback_resume(self, tmp_path, temp_checkpoint_dir, feedback_manager_filesystem):
        """Scenario: text feedback timeout -> API response -> resume.

        Tests non-approval feedback type (text input).
        """
        checkpoint_path = None
        feedback_id = None

        # Define workflow with text feedback
        with workflow("code_review") as wf:

            @task(inject_context=True)
            def request_review_comment(context):
                """Request code review comment."""
                response = context.request_feedback(
                    feedback_type="text",
                    prompt="Please provide your code review comments:",
                    timeout=0.1,  # Very short timeout for immediate timeout in tests
                )
                return response.text

            @task(inject_context=True)
            def process_review(context):
                """Process review comment."""
                channel = context.get_channel()
                comment = channel.get("request_review_comment.__result__")
                return f"Processed: {comment}"

            _ = request_review_comment >> process_review

            # Create context
            # Find start node from graph
            start_nodes = wf.graph.get_start_nodes()
            assert len(start_nodes) == 1
            start_node = start_nodes[0]

            exec_context = ExecutionContext.create(wf.graph, start_node=start_node, channel_backend="memory", config={})
            exec_context.feedback_manager = feedback_manager_filesystem

            # Execute and expect timeout
            engine = WorkflowEngine()

            # Execute workflow (should timeout and return None)
            result = engine.execute(exec_context)

            # Verify timeout handling
            assert result is None, "Expected None when workflow times out"

            # Verify checkpoint created automatically
            checkpoint_path = exec_context.last_checkpoint_path
            assert checkpoint_path is not None
            assert os.path.exists(checkpoint_path)

            # Get feedback_id from checkpoint metadata
            feedback_id = exec_context.checkpoint_metadata.get("user_metadata", {}).get("feedback_id")
            assert feedback_id is not None

        # Provide text feedback via API
        feedback_response = FeedbackResponse(
            feedback_id=feedback_id,
            response_type=FeedbackType.TEXT,
            text="Code looks good, but please add more unit tests for edge cases.",
            responded_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            responded_by="reviewer@example.com",
        )

        success = feedback_manager_filesystem.provide_feedback(feedback_id, feedback_response)
        assert success is True

        # Resume from checkpoint
        resumed_context, _metadata = CheckpointManager.resume_from_checkpoint(checkpoint_path)
        resumed_context.feedback_manager = feedback_manager_filesystem

        # Execute resumed workflow
        result = engine.execute(resumed_context)

        # Verify text feedback was processed
        assert "Processed:" in result
        assert "unit tests" in result

        # Cleanup
        os.remove(checkpoint_path)
