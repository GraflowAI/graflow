"""Example: Best-effort handler for parallel group error handling.

This example demonstrates using the built-in :class:`BestEffortGroupPolicy`
that continues execution even when some tasks fail. This is useful for
scenarios where partial success is acceptable.

Key concepts:
- Built-in handler extends the default direct execution policy
- ``on_group_finished()`` logs failures but never raises an exception
- Useful for non-critical operations (notifications, logging, etc.)
"""

import logging

from graflow.coordination.coordinator import CoordinationBackend
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.handlers.group_policy import BestEffortGroupPolicy
from graflow.core.workflow import workflow
from graflow.exceptions import ParallelGroupError


def example_notification_system():
    """Example: Send notifications via multiple channels (best-effort)."""
    print("=" * 60)
    print("Example 1: Multi-channel notifications (best-effort)")
    print("=" * 60)

    with workflow("notification_workflow") as wf:

        @task
        def send_email():
            print("  [send_email] Sending email notification...")
            # Simulate email failure
            raise ConnectionError("SMTP server unavailable")

        @task
        def send_sms():
            print("  [send_sms] Sending SMS notification...")
            return {"status": "sent", "channel": "sms"}

        @task
        def send_slack():
            print("  [send_slack] Sending Slack notification...")
            return {"status": "sent", "channel": "slack"}

        @task
        def send_webhook():
            print("  [send_webhook] Calling webhook...")
            # Simulate webhook failure
            raise TimeoutError("Webhook timeout")

        # Use best-effort handler - workflow continues even if some fail
        notifications = (send_email | send_sms | send_slack | send_webhook).with_execution(
            backend=CoordinationBackend.THREADING, policy=BestEffortGroupPolicy()
        )

        @task
        def log_notification_results():
            print("  [log_notification_results] Recording notification attempts...")
            return "logged"

        notifications >> log_notification_results

        # Execute workflow - won't raise exception even with failures
        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, notifications.task_id)
        engine.execute(context)

        result = context.get_result("log_notification_results")
        assert result == "logged"

    print(f"✓ Workflow completed: {result}")
    print("  Note: Some notifications failed, but workflow continued\n")


def example_data_enrichment():
    """Example: Enrich data from multiple sources (best-effort)."""
    print("=" * 60)
    print("Example 2: Data enrichment from multiple sources")
    print("=" * 60)

    with workflow("data_enrichment") as wf:

        @task
        def enrich_with_geo_data():
            print("  [enrich_with_geo_data] Adding geographic data...")
            return {"country": "US", "city": "San Francisco"}

        @task
        def enrich_with_weather():
            print("  [enrich_with_weather] Adding weather data...")
            # Simulate weather API failure
            raise Exception("Weather API rate limit exceeded")

        @task
        def enrich_with_demographics():
            print("  [enrich_with_demographics] Adding demographic data...")
            return {"age_group": "25-34", "income": "high"}

        @task
        def enrich_with_social():
            print("  [enrich_with_social] Adding social media data...")
            # Simulate social API failure
            raise Exception("Social API authentication failed")

        # Use best-effort - partial enrichment is acceptable
        enrichment = (
            enrich_with_geo_data | enrich_with_weather | enrich_with_demographics | enrich_with_social
        ).with_execution(backend=CoordinationBackend.THREADING, policy=BestEffortGroupPolicy())

        @task
        def process_enriched_data():
            print("  [process_enriched_data] Processing with available data...")
            return "processed_with_partial_data"

        enrichment >> process_enriched_data

        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, enrichment.task_id)
        engine.execute(context)

        result = context.get_result("process_enriched_data")
        assert result == "processed_with_partial_data"

    print(f"✓ Workflow completed: {result}")
    print("  Note: Data enriched with available sources\n")


def example_parallel_exports():
    """Example: Export data to multiple formats (best-effort)."""
    print("=" * 60)
    print("Example 3: Parallel data exports")
    print("=" * 60)

    with workflow("parallel_exports") as wf:

        @task
        def export_to_csv():
            print("  [export_to_csv] Exporting to CSV...")
            return "export_data.csv"

        @task
        def export_to_json():
            print("  [export_to_json] Exporting to JSON...")
            return "export_data.json"

        @task
        def export_to_parquet():
            print("  [export_to_parquet] Exporting to Parquet...")
            # Simulate parquet library issue
            raise ImportError("parquet library not available")

        @task
        def export_to_xml():
            print("  [export_to_xml] Exporting to XML...")
            return "export_data.xml"

        # Use best-effort - at least some exports should succeed
        exports = (export_to_csv | export_to_json | export_to_parquet | export_to_xml).with_execution(
            backend=CoordinationBackend.THREADING, policy=BestEffortGroupPolicy()
        )

        @task
        def notify_export_completion():
            print("  [notify_export_completion] Notifying users...")
            return "notifications_sent"

        exports >> notify_export_completion

        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, exports.task_id)
        engine.execute(context)

        result = context.get_result("notify_export_completion")
        assert result == "notifications_sent"

    print(f"✓ Workflow completed: {result}")
    print("  Note: Data exported in available formats\n")


def example_comparison_with_strict_mode():
    """Example: Compare best-effort vs strict mode behavior."""
    print("=" * 60)
    print("Example 4: Comparing best-effort vs strict mode")
    print("=" * 60)

    # First, try with strict mode (default)
    print("\nAttempt 1: Using strict mode (default)\n")

    try:
        with workflow("strict_mode_test") as wf:

            @task
            def task_a():
                print("  [task_a] Success")
                return "a"

            @task
            def task_b():
                print("  [task_b] Failure")
                raise Exception("Task B failed!")

            _parallel = (task_a | task_b).with_execution(backend=CoordinationBackend.THREADING)

            engine = WorkflowEngine()
            context = ExecutionContext.create(wf.graph, _parallel.task_id)
            engine.execute(context)
    except ParallelGroupError:
        print("✗ Strict mode: Workflow failed with ParallelGroupError")
    else:
        raise AssertionError("Strict mode should have failed")

    # Now try with best-effort
    print("\nAttempt 2: Using best-effort mode\n")

    with workflow("best_effort_test") as wf:

        @task
        def task_c():
            print("  [task_c] Success")
            return "c"

        @task
        def task_d():
            print("  [task_d] Failure")
            raise Exception("Task D failed!")

        _parallel = (task_c | task_d).with_execution(
            backend=CoordinationBackend.THREADING, policy=BestEffortGroupPolicy()
        )

        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, _parallel.task_id)
        engine.execute(context)

        assert context.get_result("task_c") == "c"

    print("✓ Best-effort mode: Workflow completed successfully")
    print("  Key difference: Best-effort continues despite failures\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "=" * 60)
    print("PARALLEL GROUP ERROR HANDLING: BEST-EFFORT MODE")
    print("=" * 60 + "\n")

    example_notification_system()
    example_data_enrichment()
    example_parallel_exports()
    example_comparison_with_strict_mode()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
