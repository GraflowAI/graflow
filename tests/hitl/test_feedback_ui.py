"""Tests for Feedback UI endpoints."""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from graflow.api.app import create_feedback_api
from graflow.hitl.types import FeedbackRequest, FeedbackResponse, FeedbackType


@pytest.fixture
def app_with_web_ui():
    """Create FastAPI app with Web UI enabled."""
    return create_feedback_api(
        feedback_backend="filesystem", feedback_config={"data_dir": "test_feedback_data"}, enable_web_ui=True
    )


@pytest.fixture
def client(app_with_web_ui):
    """Create test client."""
    return TestClient(app_with_web_ui)


@pytest.fixture
def feedback_manager(app_with_web_ui):
    """Get feedback manager from app state."""
    return app_with_web_ui.state.feedback_manager


def create_test_request(
    feedback_type: str = "approval",
    feedback_id: str = "test-feedback-123",
    prompt: str = "Test prompt",
    options: list[str] | None = None,
) -> FeedbackRequest:
    """Create test feedback request."""
    return FeedbackRequest(
        task_id="test-task",
        feedback_id=feedback_id,
        session_id="test-session",
        feedback_type=FeedbackType(feedback_type),
        prompt=prompt,
        options=options or [],
        metadata={},
        status="pending",
        created_at=datetime.now().isoformat(),
    )


class TestFeedbackUIEndpoints:
    """Test Feedback UI endpoints."""

    def test_list_pending_feedback_empty(self, client):
        """Test listing pending feedback when empty."""
        response = client.get("/ui/feedback/")
        assert response.status_code == 200
        assert "Pending Feedback Requests" in response.text
        assert "No pending feedback requests" in response.text

    def test_list_pending_feedback_with_requests(self, client, feedback_manager):
        """Test listing pending feedback with multiple requests."""
        # Create multiple pending requests
        request1 = create_test_request(feedback_id="test-1", feedback_type="approval", prompt="Approve deployment?")
        request2 = create_test_request(feedback_id="test-2", feedback_type="text", prompt="Enter feedback")
        feedback_manager.store_request(request1)
        feedback_manager.store_request(request2)

        # GET list
        response = client.get("/ui/feedback/")
        assert response.status_code == 200
        assert "Total: 2 request(s)" in response.text
        assert "Approve deployment?" in response.text
        assert "Enter feedback" in response.text
        # Check masked IDs (1 char + 15 asterisks)
        assert "t**************" in response.text  # test-1 and test-2 both start with 't'
        # Full IDs should NOT be visible
        assert "test-1" not in response.text
        assert "test-2" not in response.text

    def test_show_feedback_form_approval(self, client, feedback_manager):
        """Test showing approval feedback form."""
        # Create pending request
        request = create_test_request(feedback_type="approval")
        feedback_manager.store_request(request)

        # GET form
        response = client.get(f"/ui/feedback/{request.feedback_id}")
        assert response.status_code == 200
        assert "Feedback Request" in response.text
        assert request.prompt in response.text
        assert "Approve" in response.text
        assert "Reject" in response.text

    def test_show_feedback_form_text(self, client, feedback_manager):
        """Test showing text input feedback form."""
        # Create pending request
        request = create_test_request(feedback_type="text", prompt="Enter your comment")
        feedback_manager.store_request(request)

        # GET form
        response = client.get(f"/ui/feedback/{request.feedback_id}")
        assert response.status_code == 200
        assert "Your Response" in response.text
        assert request.prompt in response.text

    def test_show_feedback_form_selection(self, client, feedback_manager):
        """Test showing selection feedback form."""
        # Create pending request
        request = create_test_request(
            feedback_type="selection", prompt="Choose an option", options=["option_a", "option_b", "option_c"]
        )
        feedback_manager.store_request(request)

        # GET form
        response = client.get(f"/ui/feedback/{request.feedback_id}")
        assert response.status_code == 200
        assert "Select an Option" in response.text
        assert "option_a" in response.text
        assert "option_b" in response.text
        assert "option_c" in response.text

    def test_show_feedback_form_not_found(self, client):
        """Test showing form for non-existent feedback."""
        response = client.get("/ui/feedback/non-existent-id")
        assert response.status_code == 404

    def test_show_feedback_form_expired_redirect(self, client, feedback_manager):
        """Test redirect to expired page for expired request."""
        # Create expired request
        request = create_test_request(feedback_type="approval")
        request.status = "completed"
        feedback_manager.store_request(request)

        # GET form - should redirect to expired
        response = client.get(f"/ui/feedback/{request.feedback_id}", follow_redirects=False)
        assert response.status_code == 307  # Redirect
        assert f"/ui/feedback/{request.feedback_id}/expired" in response.headers["location"]

    def test_submit_approval_approved(self, client, feedback_manager):
        """Test submitting approval (approved)."""
        # Create pending request
        request = create_test_request(feedback_type="approval")
        feedback_manager.store_request(request)

        # POST submission
        response = client.post(
            f"/ui/feedback/{request.feedback_id}/submit",
            data={"approved": "true", "reason": "Looks good!", "responded_by": "test@example.com"},
            follow_redirects=False,
        )
        assert response.status_code == 303  # See Other (redirect)
        assert f"/ui/feedback/{request.feedback_id}/success" in response.headers["location"]

        # Check response stored
        feedback_response = feedback_manager.get_response(request.feedback_id)
        assert feedback_response is not None
        assert feedback_response.approved is True
        assert feedback_response.reason == "Looks good!"
        assert feedback_response.responded_by == "test@example.com"

    def test_submit_approval_rejected(self, client, feedback_manager):
        """Test submitting approval (rejected)."""
        # Create pending request
        request = create_test_request(feedback_type="approval")
        feedback_manager.store_request(request)

        # POST submission
        response = client.post(
            f"/ui/feedback/{request.feedback_id}/submit",
            data={"approved": "false", "reason": "Not ready yet"},
            follow_redirects=False,
        )
        assert response.status_code == 303

        # Check response stored
        feedback_response = feedback_manager.get_response(request.feedback_id)
        assert feedback_response is not None
        assert feedback_response.approved is False
        assert feedback_response.reason == "Not ready yet"

    def test_submit_text(self, client, feedback_manager):
        """Test submitting text input."""
        # Create pending request
        request = create_test_request(feedback_type="text")
        feedback_manager.store_request(request)

        # POST submission
        response = client.post(
            f"/ui/feedback/{request.feedback_id}/submit", data={"text": "This is my response"}, follow_redirects=False
        )
        assert response.status_code == 303

        # Check response stored
        feedback_response = feedback_manager.get_response(request.feedback_id)
        assert feedback_response is not None
        assert feedback_response.text == "This is my response"

    def test_submit_selection(self, client, feedback_manager):
        """Test submitting selection."""
        # Create pending request
        request = create_test_request(feedback_type="selection", options=["option_a", "option_b"])
        feedback_manager.store_request(request)

        # POST submission
        response = client.post(
            f"/ui/feedback/{request.feedback_id}/submit", data={"selected": "option_b"}, follow_redirects=False
        )
        assert response.status_code == 303

        # Check response stored
        feedback_response = feedback_manager.get_response(request.feedback_id)
        assert feedback_response is not None
        assert feedback_response.selected == "option_b"

    def test_submit_multi_selection(self, client, feedback_manager):
        """Test submitting multi-selection."""
        # Create pending request
        request = create_test_request(feedback_type="multi_selection", options=["option_a", "option_b", "option_c"])
        feedback_manager.store_request(request)

        # POST submission
        response = client.post(
            f"/ui/feedback/{request.feedback_id}/submit",
            data={"selected_multiple": ["option_a", "option_c"]},
            follow_redirects=False,
        )
        assert response.status_code == 303

        # Check response stored
        feedback_response = feedback_manager.get_response(request.feedback_id)
        assert feedback_response is not None
        assert feedback_response.selected_multiple == ["option_a", "option_c"]

    def test_submit_custom(self, client, feedback_manager):
        """Test submitting custom data."""
        # Create pending request
        request = create_test_request(feedback_type="custom")
        feedback_manager.store_request(request)

        # POST submission
        response = client.post(
            f"/ui/feedback/{request.feedback_id}/submit",
            data={"custom_data": '{"key": "value", "number": 42}'},
            follow_redirects=False,
        )
        assert response.status_code == 303

        # Check response stored
        feedback_response = feedback_manager.get_response(request.feedback_id)
        assert feedback_response is not None
        assert feedback_response.custom_data == {"key": "value", "number": 42}

    def test_submit_invalid_json(self, client, feedback_manager):
        """Test submitting invalid JSON for custom type."""
        # Create pending request
        request = create_test_request(feedback_type="custom")
        feedback_manager.store_request(request)

        # POST submission with invalid JSON
        response = client.post(
            f"/ui/feedback/{request.feedback_id}/submit",
            data={"custom_data": "not valid json{"},
            follow_redirects=False,
        )
        assert response.status_code == 400

    def test_submit_already_responded(self, client, feedback_manager):
        """Test submitting to already responded request."""
        # Create completed request
        request = create_test_request(feedback_type="approval")
        request.status = "completed"
        feedback_manager.store_request(request)

        # POST submission - should redirect to expired
        response = client.post(
            f"/ui/feedback/{request.feedback_id}/submit", data={"approved": "true"}, follow_redirects=False
        )
        assert response.status_code == 307  # Redirect
        assert f"/ui/feedback/{request.feedback_id}/expired" in response.headers["location"]

    def test_show_success_page(self, client, feedback_manager):
        """Test showing success page."""
        # Create request and response
        request = create_test_request(feedback_type="approval")
        feedback_manager.store_request(request)

        response_obj = FeedbackResponse(
            feedback_id=request.feedback_id,
            response_type=FeedbackType.APPROVAL,
            approved=True,
            reason="Test reason",
            responded_at=datetime.now().isoformat(),
            responded_by="test@example.com",
        )
        feedback_manager.store_response(response_obj)

        # GET success page
        response = client.get(f"/ui/feedback/{request.feedback_id}/success")
        assert response.status_code == 200
        assert "Feedback Submitted" in response.text
        assert "Approved" in response.text
        assert "Test reason" in response.text
        assert "test@example.com" in response.text

    def test_show_expired_page(self, client, feedback_manager):
        """Test showing expired page."""
        # Create completed request
        request = create_test_request(feedback_type="approval")
        request.status = "completed"
        feedback_manager.store_request(request)

        # GET expired page
        response = client.get(f"/ui/feedback/{request.feedback_id}/expired")
        assert response.status_code == 200
        assert "Request Expired" in response.text
        assert "completed" in response.text

    def test_app_with_web_ui_disabled(self):
        """Test creating app with Web UI disabled."""
        app = create_feedback_api(enable_web_ui=False)
        client = TestClient(app)

        # Root endpoint should not have web_ui in endpoints
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["web_ui_enabled"] is False
        assert "web_ui" not in data["endpoints"]

        # Web UI endpoints should not exist
        request = create_test_request()
        app.state.feedback_manager.store_request(request)

        response = client.get(f"/ui/feedback/{request.feedback_id}")
        assert response.status_code == 404

    def test_root_endpoint_with_web_ui(self, client):
        """Test root endpoint includes Web UI info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["web_ui_enabled"] is True
        assert "web_ui" in data["endpoints"]
        assert "feedback_form" in data["endpoints"]["web_ui"]
