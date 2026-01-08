"""Example: At-least-N-success handler for parallel group error handling.

This example demonstrates using the built-in :class:`AtLeastNGroupPolicy`
that enforces a minimum number of successful tasks before continuing the
workflow. It also provides a percentage-based custom handler for scenarios
where the success threshold should scale with the number of tasks.

Key concepts:
- Built-in handler with configurable success threshold
- Fails only if too few tasks succeed
- Useful for redundant operations (multi-region, multi-source)
- Percentage-based variant for scalable thresholds
"""

from graflow.coordination.coordinator import CoordinationBackend
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.handlers.group_policy import AtLeastNGroupPolicy, GroupExecutionPolicy
from graflow.core.workflow import workflow
from graflow.exceptions import ParallelGroupError


class PercentageSuccessPolicy(GroupExecutionPolicy):
    """Policy that requires a minimum percentage of tasks to succeed."""

    def __init__(self, min_percentage: float):
        if not 0 <= min_percentage <= 1:
            raise ValueError("min_percentage must be between 0 and 1")
        self.min_percentage = min_percentage

    def get_name(self) -> str:
        return f"percentage_{int(self.min_percentage * 100)}"

    def on_group_finished(self, group_id, tasks, results, context):
        self._validate_group_results(group_id, tasks, results)

        success_count = sum(1 for r in results.values() if r.success)
        total_count = len(results)
        success_rate = success_count / total_count if total_count > 0 else 0

        if success_rate < self.min_percentage:
            failed = [(tid, r.error_message) for tid, r in results.items() if not r.success]
            raise ParallelGroupError(
                f"Success rate {success_rate:.1%} < {self.min_percentage:.1%}",
                group_id=group_id,
                failed_tasks=failed,
                successful_tasks=[tid for tid, r in results.items() if r.success],
            )

        print(f"✓ Group '{group_id}': {success_rate:.1%} success rate (threshold: {self.min_percentage:.1%})")


def example_multi_region_deployment():
    """Example: Deploy to multiple regions with quorum requirement."""
    print("=" * 60)
    print("Example 1: Multi-region deployment (at least 2 succeed)")
    print("=" * 60)

    with workflow("multi_region_deploy") as wf:

        @task
        def deploy_us_east():
            print("  [deploy_us_east] Deploying to US East...")
            return "deployed_us_east"

        @task
        def deploy_us_west():
            print("  [deploy_us_west] Deploying to US West...")
            return "deployed_us_west"

        @task
        def deploy_eu_central():
            print("  [deploy_eu_central] Deploying to EU Central...")
            # Simulate deployment failure
            raise Exception("EU Central deployment failed - region unavailable")

        @task
        def deploy_asia_pacific():
            print("  [deploy_asia_pacific] Deploying to Asia Pacific...")
            # Simulate deployment failure
            raise Exception("Asia Pacific deployment failed - quota exceeded")

        # Require at least 2 regions to succeed
        deployments = (deploy_us_east | deploy_us_west | deploy_eu_central | deploy_asia_pacific).with_execution(
            backend=CoordinationBackend.THREADING, policy=AtLeastNGroupPolicy(min_success=2)
        )

        @task
        def update_load_balancer():
            print("\n  [update_load_balancer] Updating load balancer config...")
            return "load_balancer_updated"

        deployments >> update_load_balancer

        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, deployments.task_id)
        engine.execute(context)

        result = context.get_result("update_load_balancer")
        assert result == "load_balancer_updated"

    print(f"\n✓ Deployment workflow completed: {result}")
    print("  Note: 2/4 regions deployed successfully (minimum met)\n")


def example_data_source_redundancy():
    """Example: Fetch data from multiple sources with redundancy."""
    print("=" * 60)
    print("Example 2: Redundant data sources (at least 1 succeeds)")
    print("=" * 60)

    with workflow("redundant_data_fetch") as wf:

        @task
        def fetch_from_primary_db():
            print("  [fetch_from_primary_db] Fetching from primary database...")
            # Simulate primary DB failure
            raise ConnectionError("Primary database connection timeout")

        @task
        def fetch_from_replica_db():
            print("  [fetch_from_replica_db] Fetching from replica database...")
            return {"source": "replica", "records": 1000}

        @task
        def fetch_from_cache():
            print("  [fetch_from_cache] Fetching from cache...")
            # Simulate cache miss
            raise KeyError("Data not found in cache")

        # Only need 1 source to succeed
        data_fetch = (fetch_from_primary_db | fetch_from_replica_db | fetch_from_cache).with_execution(
            backend=CoordinationBackend.THREADING, policy=AtLeastNGroupPolicy(min_success=1)
        )

        @task
        def process_data():
            print("\n  [process_data] Processing fetched data...")
            return "data_processed"

        data_fetch >> process_data

        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, data_fetch.task_id)
        engine.execute(context)

        result = context.get_result("process_data")
        assert result == "data_processed"

    print(f"\n✓ Data fetch workflow completed: {result}")
    print("  Note: At least 1 source succeeded (redundancy achieved)\n")


def example_threshold_not_met():
    """Example: What happens when threshold is not met."""
    print("=" * 60)
    print("Example 3: Threshold not met (failure scenario)")
    print("=" * 60)

    with workflow("threshold_failure") as wf:

        @task
        def task_a():
            print("  [task_a] Executing...")
            return "success"

        @task
        def task_b():
            print("  [task_b] Executing...")
            raise Exception("Task B failed!")

        @task
        def task_c():
            print("  [task_c] Executing...")
            raise Exception("Task C failed!")

        @task
        def task_d():
            print("  [task_d] Executing...")
            raise Exception("Task D failed!")

        # Require at least 3 successes, but only 1 will succeed
        _parallel = (task_a | task_b | task_c | task_d).with_execution(
            backend=CoordinationBackend.THREADING, policy=AtLeastNGroupPolicy(min_success=3)
        )

        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, _parallel.task_id)

        try:
            engine.execute(context)
        except ParallelGroupError as e:
            print(f"\n✗ Workflow failed: {e}")
            print(f"  Successful: {len(e.successful_tasks)}")
            print(f"  Failed: {len(e.failed_tasks)}")
            print("  Note: Only 1/3 required tasks succeeded\n")
            assert len(e.failed_tasks) == 3
        else:
            raise AssertionError("Expected ParallelGroupError was not raised")


def example_percentage_based():
    """Example: Percentage-based success threshold."""
    print("=" * 60)
    print("Example 4: Percentage-based threshold (70% required)")
    print("=" * 60)

    with workflow("percentage_threshold") as wf:

        @task
        def worker_1():
            print("  [worker_1] Processing batch 1...")
            return "completed"

        @task
        def worker_2():
            print("  [worker_2] Processing batch 2...")
            return "completed"

        @task
        def worker_3():
            print("  [worker_3] Processing batch 3...")
            return "completed"

        @task
        def worker_4():
            print("  [worker_4] Processing batch 4...")
            # Simulate failure
            raise Exception("Worker 4 crashed!")

        # Require 70% success rate (3/4 = 75% meets threshold)
        processing = (worker_1 | worker_2 | worker_3 | worker_4).with_execution(
            backend=CoordinationBackend.THREADING, policy=PercentageSuccessPolicy(min_percentage=0.70)
        )

        @task
        def aggregate_results():
            print("\n  [aggregate_results] Aggregating completed batches...")
            return "aggregation_complete"

        processing >> aggregate_results

        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, processing.task_id)
        engine.execute(context)

        result = context.get_result("aggregate_results")
        assert result == "aggregation_complete"

    print(f"\n✓ Workflow completed: {result}")
    print("  Note: 75% success rate meets 70% threshold\n")


def example_configurable_thresholds():
    """Example: Using different thresholds for different scenarios."""
    print("=" * 60)
    print("Example 5: Configurable thresholds")
    print("=" * 60)

    scenarios = [
        ("Critical operations", 4, "All 4 must succeed"),
        ("Important operations", 3, "At least 3 must succeed"),
        ("Best-effort operations", 1, "At least 1 must succeed"),
    ]

    for scenario_name, threshold, description in scenarios:
        print(f"\nScenario: {scenario_name} ({description})")

        with workflow(f"scenario_{threshold}") as wf:

            @task
            def op_1():
                return "success"

            @task
            def op_2():
                return "success"

            @task
            def op_3():
                raise Exception("Failed!")

            @task
            def op_4():
                raise Exception("Failed!")

            _parallel = (op_1 | op_2 | op_3 | op_4).with_execution(
                backend=CoordinationBackend.THREADING, policy=AtLeastNGroupPolicy(min_success=threshold)
            )

            engine = WorkflowEngine()
            context = ExecutionContext.create(wf.graph, _parallel.task_id)

            try:
                engine.execute(context)
                print("  ✓ Passed with 2/4 successes")
                assert threshold <= 2
            except ParallelGroupError:
                print(f"  ✗ Failed: only 2/4 succeeded (needed {threshold})")
                assert threshold > 2

    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PARALLEL GROUP ERROR HANDLING: AT-LEAST-N-SUCCESS")
    print("=" * 60 + "\n")

    example_multi_region_deployment()
    example_data_source_redundancy()
    example_threshold_not_met()
    example_percentage_based()
    example_configurable_thresholds()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
