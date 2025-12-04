"""Graflow Feedback API package.

This package provides REST API endpoints for Human-in-the-Loop (HITL) feedback management.

Usage:
    # Run the API server via CLI
    python -m graflow.api --backend filesystem

    # Or programmatically
    from graflow.api.app import create_feedback_api

    app = create_feedback_api(feedback_backend="filesystem")
"""
