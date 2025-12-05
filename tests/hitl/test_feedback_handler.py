"""Tests for FeedbackHandler callback system."""

from __future__ import annotations

import threading
import time

import pytest

from graflow.hitl.handler import FeedbackHandler
from graflow.hitl.manager import FeedbackManager
from graflow.hitl.types import FeedbackResponse, FeedbackType


class TestFeedbackHandler:
    """Test FeedbackHandler callbacks."""

    def test_handler_callbacks_on_request_created(self, tmp_path):
        """Test on_request_created callback is called."""
        feedback_dir = tmp_path / "feedback"
        feedback_dir.mkdir()

        # Track callback invocations
        callbacks = []

        class TestHandler(FeedbackHandler):
            def on_request_created(self, request):
                callbacks.append(("on_request_created", request.feedback_id))

        handler = TestHandler()
        manager = FeedbackManager(
            backend="filesystem",
            backend_config={"data_dir": str(feedback_dir)},
        )

        # Provide feedback in background thread
        def provide_feedback():
            time.sleep(0.5)
            pending = manager.list_pending_requests()
            if pending:
                response = FeedbackResponse(
                    feedback_id=pending[0].feedback_id,
                    response_type=FeedbackType.APPROVAL,
                    approved=True,
                )
                manager.provide_feedback(pending[0].feedback_id, response)

        thread = threading.Thread(target=provide_feedback, daemon=True)
        thread.start()

        # Request feedback (will trigger on_request_created)
        response = manager.request_feedback(
            task_id="test_task",
            session_id="test_session",
            feedback_type=FeedbackType.APPROVAL,
            prompt="Test prompt",
            timeout=2.0,
            handler=handler,
        )

        # Verify callback was called
        assert len(callbacks) == 1
        assert callbacks[0][0] == "on_request_created"
        assert response.approved is True

    def test_handler_callbacks_on_response_received(self, tmp_path):
        """Test on_response_received callback is called."""
        feedback_dir = tmp_path / "feedback"
        feedback_dir.mkdir()

        # Track callback invocations
        callbacks = []

        class TestHandler(FeedbackHandler):
            def on_request_created(self, request):
                callbacks.append(("on_request_created", request.feedback_id))

            def on_response_received(self, request, response):
                callbacks.append(("on_response_received", request.feedback_id, response.approved))

        handler = TestHandler()
        manager = FeedbackManager(
            backend="filesystem",
            backend_config={"data_dir": str(feedback_dir)},
        )

        # Provide feedback in background thread
        def provide_feedback():
            time.sleep(0.5)
            pending = manager.list_pending_requests()
            if pending:
                response = FeedbackResponse(
                    feedback_id=pending[0].feedback_id,
                    response_type=FeedbackType.APPROVAL,
                    approved=True,
                )
                manager.provide_feedback(pending[0].feedback_id, response)

        thread = threading.Thread(target=provide_feedback, daemon=True)
        thread.start()

        # Request feedback
        response = manager.request_feedback(
            task_id="test_task",
            session_id="test_session",
            feedback_type=FeedbackType.APPROVAL,
            prompt="Test prompt",
            timeout=2.0,
            handler=handler,
        )

        # Verify both callbacks were called
        assert len(callbacks) == 2
        assert callbacks[0][0] == "on_request_created"
        assert callbacks[1][0] == "on_response_received"
        assert callbacks[1][2] is True  # approved
        assert response.approved is True

    def test_handler_callbacks_on_request_timeout(self, tmp_path):
        """Test on_request_timeout callback is called."""
        feedback_dir = tmp_path / "feedback"
        feedback_dir.mkdir()

        # Track callback invocations
        callbacks = []

        class TestHandler(FeedbackHandler):
            def on_request_created(self, request):
                callbacks.append(("on_request_created", request.feedback_id))

            def on_request_timeout(self, request):
                callbacks.append(("on_request_timeout", request.feedback_id))

        handler = TestHandler()
        manager = FeedbackManager(
            backend="filesystem",
            backend_config={"data_dir": str(feedback_dir)},
        )

        # Request feedback with very short timeout (will timeout)
        from graflow.hitl.types import FeedbackTimeoutError

        with pytest.raises(FeedbackTimeoutError):
            manager.request_feedback(
                task_id="test_task",
                session_id="test_session",
                feedback_type=FeedbackType.APPROVAL,
                prompt="Test prompt",
                timeout=0.1,  # Very short timeout
                handler=handler,
            )

        # Verify both callbacks were called
        assert len(callbacks) == 2
        assert callbacks[0][0] == "on_request_created"
        assert callbacks[1][0] == "on_request_timeout"

    def test_handler_with_multiple_callbacks(self, tmp_path):
        """Test handler receives all callbacks for a single request."""
        feedback_dir = tmp_path / "feedback"
        feedback_dir.mkdir()

        # Track callback invocations
        callbacks = []

        class TestHandler(FeedbackHandler):
            def on_request_created(self, request):
                callbacks.append("on_request_created")

            def on_response_received(self, request, response):
                callbacks.append("on_response_received")

        handler = TestHandler()
        manager = FeedbackManager(
            backend="filesystem",
            backend_config={"data_dir": str(feedback_dir)},
        )

        # Provide feedback in background thread
        def provide_feedback():
            time.sleep(0.5)
            pending = manager.list_pending_requests()
            if pending:
                response = FeedbackResponse(
                    feedback_id=pending[0].feedback_id,
                    response_type=FeedbackType.APPROVAL,
                    approved=True,
                )
                manager.provide_feedback(pending[0].feedback_id, response)

        thread = threading.Thread(target=provide_feedback, daemon=True)
        thread.start()

        # Request feedback
        response = manager.request_feedback(
            task_id="test_task",
            session_id="test_session",
            feedback_type=FeedbackType.APPROVAL,
            prompt="Test prompt",
            timeout=2.0,
            handler=handler,
        )

        # Verify all callbacks were invoked
        assert response.approved is True
        assert callbacks == ["on_request_created", "on_response_received"]

    def test_handler_error_is_caught(self, tmp_path):
        """Test that errors in handlers are caught and don't break the flow."""
        feedback_dir = tmp_path / "feedback"
        feedback_dir.mkdir()

        class BrokenHandler(FeedbackHandler):
            def on_request_created(self, request):
                raise ValueError("Intentional error for testing")

        handler = BrokenHandler()
        manager = FeedbackManager(
            backend="filesystem",
            backend_config={"data_dir": str(feedback_dir)},
        )

        # Provide feedback in background thread
        def provide_feedback():
            time.sleep(0.5)
            pending = manager.list_pending_requests()
            if pending:
                response = FeedbackResponse(
                    feedback_id=pending[0].feedback_id,
                    response_type=FeedbackType.APPROVAL,
                    approved=True,
                )
                manager.provide_feedback(pending[0].feedback_id, response)

        thread = threading.Thread(target=provide_feedback, daemon=True)
        thread.start()

        # Request feedback should still work despite handler error
        response = manager.request_feedback(
            task_id="test_task",
            session_id="test_session",
            feedback_type=FeedbackType.APPROVAL,
            prompt="Test prompt",
            timeout=2.0,
            handler=handler,
        )

        # Workflow continues despite handler error
        assert response.approved is True
