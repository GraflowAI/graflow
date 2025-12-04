"""Tests for FeedbackManager."""

from __future__ import annotations

import threading
import time

from graflow.hitl.manager import FeedbackManager
from graflow.hitl.types import (
    FeedbackResponse,
    FeedbackTimeoutError,
    FeedbackType,
)


class TestFeedbackManagerMemory:
    """Tests for FeedbackManager with memory backend."""

    def test_create_feedback_request(self):
        """Test creating a feedback request."""
        manager = FeedbackManager(backend="memory")

        # Create a request in a thread (since request_feedback blocks)
        result = []
        exception = []

        def request_thread():
            try:
                response = manager.request_feedback(
                    task_id="test_task",
                    session_id="test_session",
                    feedback_type=FeedbackType.APPROVAL,
                    prompt="Test approval?",
                    timeout=1.0,  # Short timeout for test
                )
                result.append(response)
            except FeedbackTimeoutError as e:
                exception.append(e)

        thread = threading.Thread(target=request_thread)
        thread.start()

        # Wait a bit for request to be created
        time.sleep(0.1)

        # Check pending requests
        pending = manager.list_pending_requests()
        assert len(pending) == 1
        assert pending[0].task_id == "test_task"
        assert pending[0].feedback_type == FeedbackType.APPROVAL

        # Wait for thread to finish (should timeout)
        thread.join(timeout=2.0)

        # Should have raised FeedbackTimeoutException
        assert len(exception) == 1
        assert isinstance(exception[0], FeedbackTimeoutError)
        assert len(result) == 0

    def test_provide_feedback(self):
        """Test providing feedback response."""
        manager = FeedbackManager(backend="memory")

        result = []

        def request_thread():
            response = manager.request_feedback(
                task_id="test_task",
                session_id="test_session",
                feedback_type=FeedbackType.APPROVAL,
                prompt="Test approval?",
                timeout=5.0,
            )
            result.append(response)

        thread = threading.Thread(target=request_thread)
        thread.start()

        # Wait for request to be created
        time.sleep(0.1)

        # Get pending request
        pending = manager.list_pending_requests()
        assert len(pending) == 1
        feedback_id = pending[0].feedback_id

        # Provide response
        response = FeedbackResponse(
            feedback_id=feedback_id,
            response_type=FeedbackType.APPROVAL,
            approved=True,
            reason="Looks good",
            responded_by="test_user",
        )
        success = manager.provide_feedback(feedback_id, response)
        assert success is True

        # Wait for thread to complete
        thread.join(timeout=2.0)

        # Check result
        assert len(result) == 1
        assert result[0].approved is True
        assert result[0].reason == "Looks good"

    def test_response_persistence(self):
        """Test that responses persist and can be retrieved by feedback_id."""
        manager = FeedbackManager(backend="memory")

        # First request
        result1 = []

        def request_thread1():
            response = manager.request_feedback(
                task_id="test_task",
                session_id="test_session",
                feedback_type=FeedbackType.APPROVAL,
                prompt="Test approval?",
                timeout=5.0,
            )
            result1.append(response)

        thread1 = threading.Thread(target=request_thread1)
        thread1.start()

        # Wait and provide feedback
        time.sleep(0.1)
        pending = manager.list_pending_requests()
        feedback_id = pending[0].feedback_id

        response = FeedbackResponse(
            feedback_id=feedback_id,
            response_type=FeedbackType.APPROVAL,
            approved=True,
        )
        manager.provide_feedback(feedback_id, response)
        thread1.join(timeout=2.0)

        # Verify response was stored
        stored_response = manager._backend.get_response(feedback_id)
        assert stored_response is not None
        assert stored_response.approved is True

        # In a real workflow, the checkpoint would save the feedback_id,
        # and on resume, the same feedback_id would be used to check for
        # existing response before creating a new request
        existing_response = manager._backend.get_response(feedback_id)
        assert existing_response is not None
        assert existing_response.approved is True

    def test_text_feedback(self):
        """Test text feedback type."""
        manager = FeedbackManager(backend="memory")

        result = []

        def request_thread():
            response = manager.request_feedback(
                task_id="test_task",
                session_id="test_session",
                feedback_type=FeedbackType.TEXT,
                prompt="Enter your comment:",
                timeout=5.0,
            )
            result.append(response)

        thread = threading.Thread(target=request_thread)
        thread.start()

        time.sleep(0.1)
        pending = manager.list_pending_requests()
        feedback_id = pending[0].feedback_id

        response = FeedbackResponse(
            feedback_id=feedback_id,
            response_type=FeedbackType.TEXT,
            text="This is my comment",
        )
        manager.provide_feedback(feedback_id, response)
        thread.join(timeout=2.0)

        assert len(result) == 1
        assert result[0].text == "This is my comment"

    def test_selection_feedback(self):
        """Test selection feedback type."""
        manager = FeedbackManager(backend="memory")

        result = []

        def request_thread():
            response = manager.request_feedback(
                task_id="test_task",
                session_id="test_session",
                feedback_type=FeedbackType.SELECTION,
                prompt="Choose an option:",
                options=["Option A", "Option B", "Option C"],
                timeout=5.0,
            )
            result.append(response)

        thread = threading.Thread(target=request_thread)
        thread.start()

        time.sleep(0.1)
        pending = manager.list_pending_requests()
        feedback_id = pending[0].feedback_id

        response = FeedbackResponse(
            feedback_id=feedback_id,
            response_type=FeedbackType.SELECTION,
            selected="Option B",
        )
        manager.provide_feedback(feedback_id, response)
        thread.join(timeout=2.0)

        assert len(result) == 1
        assert result[0].selected == "Option B"

    def test_session_filtering(self):
        """Test filtering pending requests by session_id."""
        manager = FeedbackManager(backend="memory")

        # Create requests in different sessions
        threads = []

        def request_thread(session_id: str):
            try:
                manager.request_feedback(
                    task_id=f"task_{session_id}",
                    session_id=session_id,
                    feedback_type=FeedbackType.APPROVAL,
                    prompt="Test?",
                    timeout=0.5,
                )
            except FeedbackTimeoutError:
                pass

        for session in ["session1", "session2", "session3"]:
            thread = threading.Thread(target=request_thread, args=(session,))
            thread.start()
            threads.append(thread)

        time.sleep(0.1)

        # Check filtering
        all_pending = manager.list_pending_requests()
        assert len(all_pending) == 3

        session1_pending = manager.list_pending_requests(session_id="session1")
        assert len(session1_pending) == 1
        assert session1_pending[0].session_id == "session1"

        # Wait for threads
        for thread in threads:
            thread.join(timeout=2.0)

    def test_write_to_channel(self):
        """Test automatic writing of feedback to channel."""
        from graflow.channels.memory_channel import MemoryChannel

        # Create manager with channel
        channel = MemoryChannel(name="test_channel")
        manager = FeedbackManager(backend="memory", channel_manager=channel)

        result = []

        def request_thread():
            response = manager.request_feedback(
                task_id="test_task",
                session_id="test_session",
                feedback_type=FeedbackType.APPROVAL,
                prompt="Approve?",
                timeout=5.0,
                channel_key="deployment_approved",
                write_to_channel=True,
            )
            result.append(response)

        thread = threading.Thread(target=request_thread)
        thread.start()

        time.sleep(0.1)
        pending = manager.list_pending_requests()
        feedback_id = pending[0].feedback_id

        # Provide feedback
        response = FeedbackResponse(
            feedback_id=feedback_id,
            response_type=FeedbackType.APPROVAL,
            approved=True,
            reason="Approved",
        )
        manager.provide_feedback(feedback_id, response)

        thread.join(timeout=2.0)

        # Check channel was written
        assert channel.get("deployment_approved") is True
        full_response = channel.get("deployment_approved.__feedback_response__")
        assert full_response is not None
        assert full_response["approved"] is True


class TestConvenienceMethods:
    """Tests for convenience methods in TaskExecutionContext."""

    def test_request_approval(self):
        """Test request_approval convenience method."""
        from graflow.core.context import ExecutionContext, TaskExecutionContext
        from graflow.core.graph import TaskGraph
        from graflow.core.task import Task

        # Create minimal context
        graph = TaskGraph()
        task = Task(task_id="test")
        graph.add_node(task)

        exec_context = ExecutionContext.create(
            graph=graph,
            start_node="test",  # task_id
            channel_backend="memory",  # Feedback uses same backend as channel
        )
        task_context = TaskExecutionContext("test", exec_context)

        # Test in thread
        result = []

        def request_thread():
            approved = task_context.request_approval(
                prompt="Approve this?",
                timeout=5.0
            )
            result.append(approved)

        thread = threading.Thread(target=request_thread)
        thread.start()

        time.sleep(0.1)

        # Provide approval
        manager = exec_context.feedback_manager
        pending = manager.list_pending_requests()
        assert len(pending) == 1

        response = FeedbackResponse(
            feedback_id=pending[0].feedback_id,
            response_type=FeedbackType.APPROVAL,
            approved=True,
        )
        manager.provide_feedback(pending[0].feedback_id, response)

        thread.join(timeout=2.0)

        # Check result
        assert len(result) == 1
        assert result[0] is True

    def test_request_text_input(self):
        """Test request_text_input convenience method."""
        from graflow.core.context import ExecutionContext, TaskExecutionContext
        from graflow.core.graph import TaskGraph
        from graflow.core.task import Task

        graph = TaskGraph()
        task = Task(task_id="test")
        graph.add_node(task)

        exec_context = ExecutionContext.create(
            graph=graph,
            start_node="test",  # task_id
            channel_backend="memory",  # Feedback uses same backend as channel
        )
        task_context = TaskExecutionContext("test", exec_context)

        result = []

        def request_thread():
            text = task_context.request_text_input(
                prompt="Enter comment:",
                timeout=5.0
            )
            result.append(text)

        thread = threading.Thread(target=request_thread)
        thread.start()

        time.sleep(0.1)

        manager = exec_context.feedback_manager
        pending = manager.list_pending_requests()

        response = FeedbackResponse(
            feedback_id=pending[0].feedback_id,
            response_type=FeedbackType.TEXT,
            text="This is my input",
        )
        manager.provide_feedback(pending[0].feedback_id, response)

        thread.join(timeout=2.0)

        assert len(result) == 1
        assert result[0] == "This is my input"

    def test_request_selection(self):
        """Test request_selection convenience method."""
        from graflow.core.context import ExecutionContext, TaskExecutionContext
        from graflow.core.graph import TaskGraph
        from graflow.core.task import Task

        graph = TaskGraph()
        task = Task(task_id="test")
        graph.add_node(task)

        exec_context = ExecutionContext.create(
            graph=graph,
            start_node="test",  # task_id
            channel_backend="memory",  # Feedback uses same backend as channel
        )
        task_context = TaskExecutionContext("test", exec_context)

        result = []

        def request_thread():
            selected = task_context.request_selection(
                prompt="Choose option:",
                options=["A", "B", "C"],
                timeout=5.0
            )
            result.append(selected)

        thread = threading.Thread(target=request_thread)
        thread.start()

        time.sleep(0.1)

        manager = exec_context.feedback_manager
        pending = manager.list_pending_requests()

        response = FeedbackResponse(
            feedback_id=pending[0].feedback_id,
            response_type=FeedbackType.SELECTION,
            selected="B",
        )
        manager.provide_feedback(pending[0].feedback_id, response)

        thread.join(timeout=2.0)

        assert len(result) == 1
        assert result[0] == "B"
