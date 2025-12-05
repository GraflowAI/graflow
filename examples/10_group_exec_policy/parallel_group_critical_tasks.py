"""Example: Critical tasks handler for parallel group error handling.

This example demonstrates using the built-in :class:`CriticalGroupPolicy`
that distinguishes between critical and optional tasks. The workflow fails
only if critical tasks fail.

Key concepts:
- Different failure policies for different tasks in the same group
- Critical tasks must succeed, optional tasks can fail
- Useful for workflows with core and auxiliary operations
- Flexible priority-based execution
"""

import logging

from graflow.coordination.coordinator import CoordinationBackend
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.handlers.group_policy import CriticalGroupPolicy
from graflow.core.workflow import workflow
from graflow.exceptions import ParallelGroupError


def example_data_pipeline_with_optional_tasks():
    """Example: Data pipeline with critical and optional operations."""
    print("=" * 60)
    print("Example 1: Data pipeline (critical + optional tasks)")
    print("=" * 60)

    with workflow("data_pipeline") as wf:

        @task
        def extract_data():
            """Critical: Must succeed."""
            print("  [extract_data] Extracting data from source...")
            return {"records": 1000}

        @task
        def validate_schema():
            """Critical: Must succeed."""
            print("  [validate_schema] Validating data schema...")
            return {"valid": True}

        @task
        def enrich_with_metadata():
            """Optional: Nice to have but not required."""
            print("  [enrich_with_metadata] Adding metadata...")
            # Simulate optional task failure
            raise Exception("Metadata service unavailable")

        @task
        def calculate_statistics():
            """Optional: Analytics task."""
            print("  [calculate_statistics] Computing statistics...")
            # Simulate optional task failure
            raise Exception("Stats calculation timeout")

        # Only extract_data and validate_schema are critical
        processing = (extract_data | validate_schema | enrich_with_metadata | calculate_statistics).with_execution(
            backend=CoordinationBackend.THREADING,
            policy=CriticalGroupPolicy(critical_task_ids=["extract_data", "validate_schema"]),
        )

        @task
        def load_to_warehouse():
            print("\n  [load_to_warehouse] Loading data to warehouse...")
            return "loaded"

        processing >> load_to_warehouse

        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, processing.task_id)
        engine.execute(context)

        result = context.get_result("load_to_warehouse")
        assert result == "loaded"

        # Optional tasks should store exception objects
        assert context.get_result("enrich_with_metadata") is None
        assert context.get_result("calculate_statistics") is None

    print(f"✓ Pipeline completed: {result}")
    print("  Note: Optional tasks failed, but critical tasks succeeded\n")


def example_notification_with_fallback():
    """Example: Send notifications with required and optional channels."""
    print("=" * 60)
    print("Example 2: Notifications with required channels")
    print("=" * 60)

    with workflow("notification_system") as wf:

        @task
        def send_user_email():
            """Critical: User must receive email."""
            print("  [send_user_email] Sending email to user...")
            return "email_sent"

        @task
        def update_notification_log():
            """Critical: Must log notification."""
            print("  [update_notification_log] Logging notification...")
            return "logged"

        @task
        def send_sms():
            """Optional: SMS is nice-to-have."""
            print("  [send_sms] Sending SMS...")
            # Simulate SMS failure
            raise Exception("SMS service rate limit exceeded")

        @task
        def post_to_slack():
            """Optional: Slack notification."""
            print("  [post_to_slack] Posting to Slack...")
            # Simulate Slack failure
            raise Exception("Slack webhook unavailable")

        # Email and logging are critical
        _notifications = (send_user_email | update_notification_log | send_sms | post_to_slack).with_execution(
            backend=CoordinationBackend.THREADING,
            policy=CriticalGroupPolicy(critical_task_ids=["send_user_email", "update_notification_log"]),
        )

        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, _notifications.task_id)
        engine.execute(context)

    assert context.get_result("send_user_email") == "email_sent"
    assert context.get_result("update_notification_log") == "logged"
    assert context.get_result("send_sms") is None
    assert context.get_result("post_to_slack") is None

    print("✓ Notification workflow completed")
    print("  Note: User received email and notification was logged\n")


def example_critical_task_failure():
    """Example: What happens when a critical task fails."""
    print("=" * 60)
    print("Example 3: Critical task failure (workflow fails)")
    print("=" * 60)

    with workflow("critical_failure") as wf:

        @task
        def validate_auth():
            """Critical: Authentication required."""
            print("  [validate_auth] Validating authentication...")
            # Simulate auth failure
            raise PermissionError("Invalid credentials")

        @task
        def fetch_user_profile():
            """Critical: User profile needed."""
            print("  [fetch_user_profile] Fetching user profile...")
            return {"user_id": 123}

        @task
        def load_preferences():
            """Optional: User preferences."""
            print("  [load_preferences] Loading user preferences...")
            return {"theme": "dark"}

        # Auth and profile are critical
        _initialization = (validate_auth | fetch_user_profile | load_preferences).with_execution(
            backend=CoordinationBackend.THREADING,
            policy=CriticalGroupPolicy(critical_task_ids=["validate_auth", "fetch_user_profile"]),
        )

        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, _initialization.task_id)

        try:
            engine.execute(context)
        except ParallelGroupError as e:
            print(f"\n✗ Workflow failed: {e}")
            print(f"  Critical tasks that failed: {[tid for tid, _ in e.failed_tasks]}")
            print(f"  Successful tasks: {e.successful_tasks}")
            print("  Note: Cannot proceed without authentication\n")
            assert "validate_auth" in [tid for tid, _ in e.failed_tasks]
        else:
            raise AssertionError("Expected ParallelGroupError was not raised")


def example_ml_training_with_optional_steps():
    """Example: ML training with critical and optional steps."""
    print("=" * 60)
    print("Example 4: ML training pipeline")
    print("=" * 60)

    with workflow("ml_training") as wf:

        @task
        def load_training_data():
            """Critical: Cannot train without data."""
            print("  [load_training_data] Loading training data...")
            return "data_loaded"

        @task
        def train_model():
            """Critical: Model training is essential."""
            print("  [train_model] Training model...")
            return "model_trained"

        @task
        def compute_feature_importance():
            """Optional: Analysis task."""
            print("  [compute_feature_importance] Computing feature importance...")
            # Simulate failure
            raise Exception("Feature importance computation failed")

        @task
        def generate_training_report():
            """Optional: Documentation task."""
            print("  [generate_training_report] Generating report...")
            # Simulate failure
            raise Exception("Report generation failed")

        # Only data loading and training are critical
        training = (
            load_training_data | train_model | compute_feature_importance | generate_training_report
        ).with_execution(
            backend=CoordinationBackend.THREADING,
            policy=CriticalGroupPolicy(critical_task_ids=["load_training_data", "train_model"]),
        )

        @task
        def save_model():
            print("\n  [save_model] Saving trained model...")
            return "model_saved"

        training >> save_model

        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, training.task_id)
        engine.execute(context)

        result = context.get_result("save_model")
        assert result == "model_saved"
        assert context.get_result("compute_feature_importance") is None
        assert context.get_result("generate_training_report") is None

    print(f"✓ Training completed: {result}")
    print("  Note: Model trained successfully despite analysis failures\n")


def example_dynamic_critical_tasks():
    """Example: Dynamically specify critical tasks based on context."""
    print("=" * 60)
    print("Example 5: Dynamic critical task selection")
    print("=" * 60)

    def run_deployment(environment: str):
        """Run deployment with environment-specific critical tasks."""
        print(f"\nDeploying to {environment} environment:")

        # In production, all tasks are critical
        # In staging, only core deployment is critical
        if environment == "production":
            critical_tasks = ["deploy_app", "run_migrations", "warm_cache", "health_check"]
        else:
            critical_tasks = ["deploy_app"]

        with workflow(f"deploy_{environment}") as wf:

            @task
            def deploy_app():
                print("  [deploy_app] Deploying application...")
                return "deployed"

            @task
            def run_migrations():
                print("  [run_migrations] Running database migrations...")
                return "migrated"

            @task
            def warm_cache():
                print("  [warm_cache] Warming cache...")
                # Simulate failure in staging
                if environment == "staging":
                    raise Exception("Cache warming failed")
                return "cache_warmed"

            @task
            def health_check():
                print("  [health_check] Running health checks...")
                return "healthy"

            _deployment = (deploy_app | run_migrations | warm_cache | health_check).with_execution(
                backend=CoordinationBackend.THREADING,
                policy=CriticalGroupPolicy(critical_task_ids=critical_tasks),
            )

            engine = WorkflowEngine()
            context = ExecutionContext.create(wf.graph, _deployment.task_id)
            engine.execute(context)

        print(f"  ✓ {environment} deployment completed")

    # Deploy to staging (allows cache failure)
    run_deployment("staging")

    # Deploy to production (requires all tasks)
    run_deployment("production")

    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "=" * 60)
    print("PARALLEL GROUP ERROR HANDLING: CRITICAL TASKS")
    print("=" * 60 + "\n")

    example_data_pipeline_with_optional_tasks()
    example_notification_with_fallback()
    example_critical_task_failure()
    example_ml_training_with_optional_steps()
    example_dynamic_critical_tasks()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
