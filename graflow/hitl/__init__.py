"""Human-in-the-Loop (HITL) functionality for Graflow workflows.

This module provides internal HITL implementation.
Users should access HITL functionality through TaskExecutionContext.request_feedback().

Example:
    @task(inject_context=True)
    def my_task(context):
        response = context.request_feedback(
            feedback_type="approval",
            prompt="Approve this?",
            timeout=180
        )
        if response.approved:
            proceed()
"""

__all__: list[str] = []
