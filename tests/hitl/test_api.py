"""Tests for Feedback API endpoints."""

from __future__ import annotations

import pytest

from graflow.api.app import create_feedback_api
from graflow.hitl.manager import FeedbackManager
from graflow.hitl.types import FeedbackRequest, FeedbackResponse, FeedbackType

# Check if FastAPI is available
try:
    from fastapi.testclient import TestClient  # noqa: F401
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# Skip all tests in this module if FastAPI is not available
pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")


@pytest.fixture
def feedback_manager(tmp_path):
    """Create a FeedbackManager with filesystem backend in temp directory."""
    return FeedbackManager(
        backend="filesystem",
        backend_config={"data_dir": str(tmp_path / "feedback_data")}
    )


@pytest.fixture
def api_app(feedback_manager):
    """Create FastAPI app with FeedbackManager."""

    app = create_feedback_api(
        feedback_backend=feedback_manager._backend,
        title="Test Feedback API",
        enable_cors=False
    )
    return app


@pytest.fixture
def client(api_app):
    """Create test client."""
    from fastapi.testclient import TestClient
    return TestClient(api_app)


class TestFeedbackAPI:
    """Tests for Feedback API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Feedback API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "graflow-feedback-api"
        assert "backend" in data

    def test_list_pending_feedback_empty(self, client):
        """Test listing pending feedback when none exist."""
        response = client.get("/api/feedback")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["requests"] == []

    def test_list_pending_feedback_with_requests(self, client, feedback_manager):
        """Test listing pending feedback with existing requests."""
        # Create a feedback request
        request = FeedbackRequest(
            feedback_id="test_task_abc123",
            task_id="test_task",
            session_id="test_session",
            feedback_type=FeedbackType.APPROVAL,
            prompt="Test approval?",
            status="pending",
            timeout=180.0
        )
        feedback_manager._backend.store_request(request)

        # List pending requests
        response = client.get("/api/feedback")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["requests"]) == 1
        assert data["requests"][0]["feedback_id"] == "test_task_abc123"
        assert data["requests"][0]["prompt"] == "Test approval?"

    def test_list_pending_feedback_filter_by_session(self, client, feedback_manager):
        """Test listing pending feedback filtered by session ID."""
        # Create requests for different sessions
        request1 = FeedbackRequest(
            feedback_id="task1_abc",
            task_id="task1",
            session_id="session_1",
            feedback_type=FeedbackType.APPROVAL,
            prompt="Approval 1",
            status="pending",
            timeout=180.0
        )
        request2 = FeedbackRequest(
            feedback_id="task2_def",
            task_id="task2",
            session_id="session_2",
            feedback_type=FeedbackType.TEXT,
            prompt="Text input 2",
            status="pending",
            timeout=180.0
        )
        feedback_manager._backend.store_request(request1)
        feedback_manager._backend.store_request(request2)

        # List all requests
        response = client.get("/api/feedback")
        assert response.status_code == 200
        assert response.json()["count"] == 2

        # Filter by session_1
        response = client.get("/api/feedback?session_id=session_1")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["requests"][0]["session_id"] == "session_1"

    def test_get_feedback_not_found(self, client):
        """Test getting feedback that doesn't exist."""
        response = client.get("/api/feedback/nonexistent_id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_feedback_request_only(self, client, feedback_manager):
        """Test getting feedback request without response."""
        # Create a feedback request
        request = FeedbackRequest(
            feedback_id="test_task_abc123",
            task_id="test_task",
            session_id="test_session",
            feedback_type=FeedbackType.APPROVAL,
            prompt="Test approval?",
            status="pending",
            timeout=180.0
        )
        feedback_manager._backend.store_request(request)

        # Get feedback details
        response = client.get("/api/feedback/test_task_abc123")
        assert response.status_code == 200
        data = response.json()
        assert data["request"]["feedback_id"] == "test_task_abc123"
        assert data["request"]["prompt"] == "Test approval?"
        assert data["response"] is None

    def test_get_feedback_with_response(self, client, feedback_manager):
        """Test getting feedback request with response."""
        # Create request and response
        request = FeedbackRequest(
            feedback_id="test_task_abc123",
            task_id="test_task",
            session_id="test_session",
            feedback_type=FeedbackType.APPROVAL,
            prompt="Test approval?",
            status="completed",
            timeout=180.0
        )
        response_obj = FeedbackResponse(
            feedback_id="test_task_abc123",
            response_type=FeedbackType.APPROVAL,
            approved=True,
            reason="Looks good"
        )
        feedback_manager._backend.store_request(request)
        feedback_manager._backend.store_response(response_obj)

        # Get feedback details
        response = client.get("/api/feedback/test_task_abc123")
        assert response.status_code == 200
        data = response.json()
        assert data["request"]["feedback_id"] == "test_task_abc123"
        assert data["response"] is not None
        assert data["response"]["approved"] is True
        assert data["response"]["reason"] == "Looks good"

    def test_respond_to_feedback_approval(self, client, feedback_manager):
        """Test providing approval feedback response."""
        # Create a pending request
        request = FeedbackRequest(
            feedback_id="test_task_abc123",
            task_id="test_task",
            session_id="test_session",
            feedback_type=FeedbackType.APPROVAL,
            prompt="Test approval?",
            status="pending",
            timeout=180.0
        )
        feedback_manager._backend.store_request(request)

        # Provide approval response
        response = client.post(
            "/api/feedback/test_task_abc123/respond",
            json={
                "approved": True,
                "reason": "Approved by manager",
                "responded_by": "alice@example.com"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Feedback provided successfully"
        assert data["feedback_id"] == "test_task_abc123"

        # Verify response was stored
        stored_response = feedback_manager.get_response("test_task_abc123")
        assert stored_response is not None
        assert stored_response.approved is True
        assert stored_response.reason == "Approved by manager"

    def test_respond_to_feedback_text(self, client, feedback_manager):
        """Test providing text feedback response."""
        # Create a pending text request
        request = FeedbackRequest(
            feedback_id="test_task_def456",
            task_id="test_task",
            session_id="test_session",
            feedback_type=FeedbackType.TEXT,
            prompt="Enter your comments:",
            status="pending",
            timeout=180.0
        )
        feedback_manager._backend.store_request(request)

        # Provide text response
        response = client.post(
            "/api/feedback/test_task_def456/respond",
            json={
                "text": "Please fix the typos in section 3",
                "responded_by": "bob@example.com"
            }
        )
        assert response.status_code == 200

        # Verify response was stored
        stored_response = feedback_manager.get_response("test_task_def456")
        assert stored_response is not None
        assert stored_response.text == "Please fix the typos in section 3"

    def test_respond_to_feedback_selection(self, client, feedback_manager):
        """Test providing selection feedback response."""
        # Create a pending selection request
        request = FeedbackRequest(
            feedback_id="test_task_ghi789",
            task_id="test_task",
            session_id="test_session",
            feedback_type=FeedbackType.SELECTION,
            prompt="Choose an option:",
            options=["option_a", "option_b", "option_c"],
            status="pending",
            timeout=180.0
        )
        feedback_manager._backend.store_request(request)

        # Provide selection response
        response = client.post(
            "/api/feedback/test_task_ghi789/respond",
            json={
                "selected": "option_b",
                "responded_by": "charlie@example.com"
            }
        )
        assert response.status_code == 200

        # Verify response was stored
        stored_response = feedback_manager.get_response("test_task_ghi789")
        assert stored_response is not None
        assert stored_response.selected == "option_b"

    def test_respond_to_feedback_not_found(self, client):
        """Test responding to non-existent feedback request."""
        response = client.post(
            "/api/feedback/nonexistent_id/respond",
            json={"approved": True}
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_cancel_feedback(self, client, feedback_manager):
        """Test cancelling a feedback request."""
        # Create a pending request
        request = FeedbackRequest(
            feedback_id="test_task_abc123",
            task_id="test_task",
            session_id="test_session",
            feedback_type=FeedbackType.APPROVAL,
            prompt="Test approval?",
            status="pending",
            timeout=180.0
        )
        feedback_manager._backend.store_request(request)

        # Cancel the request
        response = client.delete("/api/feedback/test_task_abc123")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Feedback request cancelled"
        assert data["feedback_id"] == "test_task_abc123"

        # Verify status was updated
        updated_request = feedback_manager.get_request("test_task_abc123")
        assert updated_request is not None
        assert updated_request.status == "cancelled"

    def test_cancel_feedback_not_found(self, client):
        """Test cancelling non-existent feedback request."""
        response = client.delete("/api/feedback/nonexistent_id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_openapi_schema(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "Test Feedback API"
        assert "paths" in schema
        assert "/api/feedback" in schema["paths"]


class TestCreateFeedbackAPI:
    """Tests for create_feedback_api factory function."""

    def test_create_with_filesystem_backend(self, tmp_path):
        """Test creating API with filesystem backend."""
        app = create_feedback_api(
            feedback_backend="filesystem",
            feedback_config={"data_dir": str(tmp_path / "feedback_data")}
        )
        assert app is not None
        assert hasattr(app.state, "feedback_manager")
        assert app.state.feedback_manager is not None

    def test_create_with_cors_enabled(self, tmp_path):
        """Test creating API with CORS enabled."""
        app = create_feedback_api(
            feedback_backend="filesystem",
            feedback_config={"data_dir": str(tmp_path / "feedback_data")},
            enable_cors=True,
            cors_origins=["http://localhost:3000"]
        )
        assert app is not None

    def test_create_with_custom_title(self, tmp_path):
        """Test creating API with custom title and description."""
        app = create_feedback_api(
            feedback_backend="filesystem",
            feedback_config={"data_dir": str(tmp_path / "feedback_data")},
            title="Custom API",
            description="Custom description",
            version="2.0.0"
        )
        assert app.title == "Custom API"
        assert app.description == "Custom description"
        assert app.version == "2.0.0"


class TestAPIErrorHandling:
    """Tests for API error handling."""

    def test_missing_feedback_manager(self):
        """Test that endpoints fail gracefully if FeedbackManager not initialized."""
        # Create app without proper initialization
        from fastapi import FastAPI

        from graflow.api.endpoints.feedback import router

        app = FastAPI()
        app.include_router(router)
        from fastapi.testclient import TestClient
        client = TestClient(app)

        # Should return 500 error
        response = client.get("/api/feedback")
        assert response.status_code == 500
        assert "FeedbackManager not initialized" in response.json()["detail"]
