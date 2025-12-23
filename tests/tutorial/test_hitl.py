"""
Unit tests for Human-in-the-Loop (HITL) from Tasks and Workflows Guide.

Tests ctx.request_feedback() feature with various feedback types and patterns.
Uses threading and mocking to simulate human responses.
"""

import threading
import time
from typing import Optional

import pytest

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.exceptions import GraflowWorkflowCanceledError
from graflow.hitl.types import FeedbackResponse, FeedbackTimeoutError, FeedbackType


def provide_feedback_after_delay(
    context: TaskExecutionContext,
    approved: bool = True,
    text: Optional[str] = None,
    selected: Optional[str] = None,
    delay: float = 0.5
):
    """Helper to provide feedback after a delay in a separate thread"""
    def _provide():
        time.sleep(delay)
        manager = context.execution_context.feedback_manager
        pending = manager.list_pending_requests()
        if pending:
            feedback_id = pending[0].feedback_id
            response_type = pending[0].feedback_type

            response = FeedbackResponse(
                feedback_id=feedback_id,
                response_type=response_type,
                approved=approved if response_type == FeedbackType.APPROVAL else None,
                text=text if response_type == FeedbackType.TEXT else None,
                selected=selected if response_type == FeedbackType.SELECTION else None,
                responded_by="test_user"
            )
            manager.provide_feedback(feedback_id, response)

    thread = threading.Thread(target=_provide, daemon=True)
    thread.start()


class TestBasicApproval:
    """Tests for basic approval feedback"""

    def test_approval_approved(self):
        """Test approval request that gets approved"""

        with workflow("approval_test") as wf:

            @task(inject_context=True)
            def request_approval(ctx: TaskExecutionContext):
                # Provide feedback in background
                provide_feedback_after_delay(ctx, approved=True)

                response = ctx.request_feedback(
                    feedback_type="approval",
                    prompt="Approve this action?",
                    timeout=5.0
                )

                return response.approved

            result = wf.execute()

        assert result is True

    def test_approval_rejected(self):
        """Test approval request that gets rejected"""

        with workflow("rejection_test") as wf:

            @task(inject_context=True)
            def request_approval(ctx: TaskExecutionContext):
                provide_feedback_after_delay(ctx, approved=False)

                response = ctx.request_feedback(
                    feedback_type="approval",
                    prompt="Approve this action?",
                    timeout=5.0
                )

                return response.approved

            result = wf.execute()

        assert result is False

    def test_approval_with_reason(self):
        """Test approval with reason field"""

        with workflow("approval_reason") as wf:

            @task(inject_context=True)
            def request_approval(ctx: TaskExecutionContext):
                def _provide():
                    time.sleep(0.5)
                    manager = ctx.execution_context.feedback_manager
                    pending = manager.list_pending_requests()
                    if pending:
                        feedback_id = pending[0].feedback_id
                        response = FeedbackResponse(
                            feedback_id=feedback_id,
                            response_type=FeedbackType.APPROVAL,
                            approved=True,
                            reason="Looks good to me!",
                            responded_by="approver"
                        )
                        manager.provide_feedback(feedback_id, response)

                thread = threading.Thread(target=_provide, daemon=True)
                thread.start()

                response = ctx.request_feedback(
                    feedback_type="approval",
                    prompt="Approve deployment?",
                    timeout=5.0
                )

                return {"approved": response.approved, "reason": response.reason}

            result = wf.execute()

        assert isinstance(result, dict)
        assert result["approved"] is True
        assert result["reason"] == "Looks good to me!"


class TestTextInput:
    """Tests for text input feedback"""

    def test_text_input_basic(self):
        """Test basic text input"""

        with workflow("text_input") as wf:

            @task(inject_context=True)
            def request_comment(ctx: TaskExecutionContext):
                provide_feedback_after_delay(ctx, text="This is my comment")

                response = ctx.request_feedback(
                    feedback_type="text",
                    prompt="Enter your comment:",
                    timeout=5.0
                )

                return response.text

            result = wf.execute()

        assert result == "This is my comment"

    def test_text_input_with_metadata(self):
        """Test text input with custom metadata"""

        with workflow("text_metadata") as wf:

            @task(inject_context=True)
            def request_input(ctx: TaskExecutionContext):
                provide_feedback_after_delay(ctx, text="User input")

                response = ctx.request_feedback(
                    feedback_type="text",
                    prompt="Provide feedback:",
                    metadata={"task": "review", "version": "1.0"},
                    timeout=5.0
                )

                return response.text

            result = wf.execute()

        assert result == "User input"


class TestSelection:
    """Tests for selection feedback"""

    def test_selection_basic(self):
        """Test basic selection from options"""

        with workflow("selection") as wf:

            @task(inject_context=True)
            def request_selection(ctx: TaskExecutionContext):
                provide_feedback_after_delay(ctx, selected="option2")

                response = ctx.request_feedback(
                    feedback_type="selection",
                    prompt="Choose an option:",
                    options=["option1", "option2", "option3"],
                    timeout=5.0
                )

                return response.selected

            result = wf.execute()

        assert result == "option2"

    def test_selection_with_options(self):
        """Test selection with descriptive options"""

        with workflow("selection_modes") as wf:

            @task(inject_context=True)
            def choose_mode(ctx: TaskExecutionContext):
                provide_feedback_after_delay(ctx, selected="balanced")

                response = ctx.request_feedback(
                    feedback_type="selection",
                    prompt="Choose execution mode:",
                    options=["fast", "balanced", "thorough"],
                    timeout=5.0
                )

                return response.selected

            result = wf.execute()

        assert result == "balanced"


class TestChannelIntegration:
    """Tests for feedback with channel integration"""

    def test_write_to_channel(self):
        """Test automatic writing of feedback response to channel"""

        with workflow("channel_feedback") as wf:

            @task(inject_context=True)
            def request_with_channel(ctx: TaskExecutionContext):
                provide_feedback_after_delay(ctx, approved=True)

                response = ctx.request_feedback(
                    feedback_type="approval",
                    prompt="Approve?",
                    channel_key="deployment_approved",
                    write_to_channel=True,
                    timeout=5.0
                )

                # Response should be automatically written to channel
                from_channel = ctx.get_channel().get("deployment_approved")

                return {"response": response.approved, "from_channel": from_channel}

            result = wf.execute()

        assert isinstance(result, dict)
        assert result["response"] is True
        # Channel should contain the approval status
        assert result["from_channel"] is not None

    def test_manual_channel_write(self):
        """Test manually writing feedback to channel"""

        with workflow("manual_channel") as wf:

            @task(inject_context=True)
            def request_and_store(ctx: TaskExecutionContext):
                provide_feedback_after_delay(ctx, text="Important note")

                response = ctx.request_feedback(
                    feedback_type="text",
                    prompt="Enter note:",
                    timeout=5.0
                )

                # Manually write to channel
                ctx.get_channel().set("user_note", response.text)

                return response.text

            @task(inject_context=True)
            def use_feedback(ctx: TaskExecutionContext):
                # Retrieve feedback from channel
                note = ctx.get_channel().get("user_note")
                return f"Processed: {note}"

            request_and_store >> use_feedback  # type: ignore

            result = wf.execute()

        assert result == "Processed: Important note"


class TestTimeoutBehavior:
    """Tests for feedback timeout scenarios"""

    def test_quick_response_within_timeout(self):
        """Test response received well within timeout"""

        with workflow("quick_response") as wf:

            @task(inject_context=True)
            def quick_feedback(ctx: TaskExecutionContext):
                # Provide feedback almost immediately
                provide_feedback_after_delay(ctx, approved=True, delay=0.1)

                response = ctx.request_feedback(
                    feedback_type="approval",
                    prompt="Quick approval?",
                    timeout=10.0
                )

                return response.approved

            result = wf.execute()

        assert result is True

    def test_timeout_error_handling(self):
        """Test handling of timeout errors"""

        with workflow("timeout_test") as wf:

            @task(inject_context=True)
            def timeout_feedback(ctx: TaskExecutionContext):
                # Don't provide feedback - let it timeout
                try:
                    ctx.request_feedback(
                        feedback_type="approval",
                        prompt="This will timeout",
                        timeout=1.0  # Short timeout
                    )
                    return "Should not reach here"
                except FeedbackTimeoutError:
                    return "Timeout handled"

            result = wf.execute()

        assert result == "Timeout handled"


class TestWorkflowIntegration:
    """Tests for HITL integration in complex workflows"""

    def test_approval_gates_in_pipeline(self):
        """Test multiple approval gates in a pipeline"""

        with workflow("approval_pipeline") as wf:

            @task(inject_context=True)
            def stage1(ctx: TaskExecutionContext):
                provide_feedback_after_delay(ctx, approved=True)

                response = ctx.request_feedback(
                    feedback_type="approval",
                    prompt="Approve stage 1?",
                    timeout=5.0
                )

                if not response.approved:
                    ctx.cancel_workflow("Stage 1 rejected")

                return "Stage 1 complete"

            @task(inject_context=True)
            def stage2(ctx: TaskExecutionContext):
                provide_feedback_after_delay(ctx, approved=True, delay=0.6)

                response = ctx.request_feedback(
                    feedback_type="approval",
                    prompt="Approve stage 2?",
                    timeout=5.0
                )

                if not response.approved:
                    ctx.cancel_workflow("Stage 2 rejected")

                return "Stage 2 complete"

            @task
            def final_stage():
                return "Pipeline complete"

            stage1 >> stage2 >> final_stage  # type: ignore

            result = wf.execute()

        assert result == "Pipeline complete"

    def test_rejected_approval_cancels_workflow(self):
        """Test that rejected approval can cancel workflow"""

        with workflow("cancel_on_reject") as wf:

            @task(inject_context=True)
            def gate_task(ctx: TaskExecutionContext):
                provide_feedback_after_delay(ctx, approved=False)

                response = ctx.request_feedback(
                    feedback_type="approval",
                    prompt="Approve?",
                    timeout=5.0
                )

                if not response.approved:
                    ctx.cancel_workflow("User rejected")

                return "Approved"

            @task
            def should_not_run():
                return "Should not execute"

            gate_task >> should_not_run  # type: ignore

            with pytest.raises(GraflowWorkflowCanceledError):
                wf.execute()

    def test_conditional_branching_with_feedback(self):
        """Test conditional workflow branching based on feedback"""

        with workflow("conditional_feedback") as wf:

            @task(inject_context=True)
            def choose_path(ctx: TaskExecutionContext):
                provide_feedback_after_delay(ctx, selected="fast")

                response = ctx.request_feedback(
                    feedback_type="selection",
                    prompt="Choose processing mode:",
                    options=["fast", "thorough"],
                    timeout=5.0
                )

                ctx.get_channel().set("mode", response.selected)
                return response.selected

            @task(inject_context=True)
            def process(ctx: TaskExecutionContext):
                mode = ctx.get_channel().get("mode")
                if mode == "fast":
                    return "Fast processing complete"
                else:
                    return "Thorough processing complete"

            choose_path >> process  # type: ignore

            result = wf.execute()

        assert result == "Fast processing complete"


class TestFeedbackHandler:
    """Tests for custom feedback handlers"""

    def test_feedback_with_custom_handler(self):
        """Test feedback with custom handler callbacks"""

        handler_calls = {
            "created": False,
            "received": False,
            "timeout": False
        }

        class CustomHandler:
            def on_request_created(self, request):
                handler_calls["created"] = True

            def on_response_received(self, request, response):
                handler_calls["received"] = True

            def on_request_timeout(self, request):
                handler_calls["timeout"] = True

        with workflow("custom_handler") as wf:

            @task(inject_context=True)
            def with_handler(ctx: TaskExecutionContext):
                provide_feedback_after_delay(ctx, approved=True)

                handler = CustomHandler()
                response = ctx.request_feedback(
                    feedback_type="approval",
                    prompt="Test handler",
                    timeout=5.0,
                    handler=handler
                )

                return response.approved

            result = wf.execute()

        assert result is True
        assert handler_calls["created"] is True
        assert handler_calls["received"] is True
        assert handler_calls["timeout"] is False


class TestMultipleFeedbackTypes:
    """Tests for different feedback types in same workflow"""

    def test_mixed_feedback_types(self):
        """Test workflow with approval, text, and selection"""

        with workflow("mixed_feedback") as wf:

            @task(inject_context=True)
            def approval_step(ctx: TaskExecutionContext):
                provide_feedback_after_delay(ctx, approved=True)

                response = ctx.request_feedback(
                    feedback_type="approval",
                    prompt="Start process?",
                    timeout=5.0
                )

                return response.approved

            @task(inject_context=True)
            def text_step(ctx: TaskExecutionContext):
                provide_feedback_after_delay(ctx, text="Process notes", delay=0.6)

                response = ctx.request_feedback(
                    feedback_type="text",
                    prompt="Enter notes:",
                    timeout=5.0
                )

                ctx.get_channel().set("notes", response.text)
                return response.text

            @task(inject_context=True)
            def selection_step(ctx: TaskExecutionContext):
                provide_feedback_after_delay(ctx, selected="medium", delay=0.7)

                response = ctx.request_feedback(
                    feedback_type="selection",
                    prompt="Choose priority:",
                    options=["low", "medium", "high"],
                    timeout=5.0
                )

                return response.selected

            approval_step >> text_step >> selection_step  # type: ignore

            _, ctx = wf.execute(ret_context=True)

        assert ctx.get_result("approval_step") is True
        assert ctx.get_result("text_step") == "Process notes"
        assert ctx.get_result("selection_step") == "medium"


class TestRealWorldPatterns:
    """Tests for real-world HITL patterns"""

    def test_deployment_approval_pattern(self):
        """Test deployment approval pattern with metadata"""

        with workflow("deployment") as wf:

            @task
            def prepare_deployment():
                return {
                    "version": "v1.2.3",
                    "environment": "production",
                    "changes": ["feature-x", "bugfix-y"]
                }

            @task(inject_context=True)
            def request_approval(ctx: TaskExecutionContext):
                deployment_info = ctx.get_channel().get("prepare_deployment.__result__")

                provide_feedback_after_delay(ctx, approved=True)

                response = ctx.request_feedback(
                    feedback_type="approval",
                    prompt=f"Approve deployment {deployment_info['version']} to {deployment_info['environment']}?",
                    metadata=deployment_info,
                    timeout=5.0
                )

                if not response.approved:
                    ctx.cancel_workflow("Deployment cancelled by user")

                return response.approved

            @task(inject_context=True)
            def deploy(ctx: TaskExecutionContext):
                approved = ctx.get_channel().get("request_approval.__result__")
                if approved:
                    return {"status": "deployed", "timestamp": time.time()}
                return {"status": "cancelled"}

            prepare_deployment >> request_approval >> deploy  # type: ignore

            result = wf.execute()

        assert isinstance(result, dict)
        assert result["status"] == "deployed"

    def test_data_validation_pattern(self):
        """Test human validation of processed data"""

        with workflow("data_validation") as wf:

            @task
            def process_data():
                return {
                    "records_processed": 1000,
                    "errors": 5,
                    "success_rate": 0.995
                }

            @task(inject_context=True)
            def validate_results(ctx: TaskExecutionContext):
                results = ctx.get_channel().get("process_data.__result__")

                provide_feedback_after_delay(ctx, text="Errors are acceptable, proceed")

                response = ctx.request_feedback(
                    feedback_type="text",
                    prompt=f"Processed {results['records_processed']} records with {results['errors']} errors. Comments?",
                    metadata=results,
                    timeout=5.0
                )

                ctx.get_channel().set("validation_comment", response.text)
                return "validated"

            @task(inject_context=True)
            def finalize(ctx: TaskExecutionContext):
                comment = ctx.get_channel().get("validation_comment")
                return {"status": "complete", "comment": comment}

            process_data >> validate_results >> finalize  # type: ignore

            result = wf.execute()

        assert isinstance(result, dict)
        assert result["status"] == "complete"
        assert "acceptable" in result["comment"]
