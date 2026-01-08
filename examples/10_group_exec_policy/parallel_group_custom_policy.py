"""Example: Defining and using a custom group execution policy.

This example shows how to implement a custom :class:`GroupExecutionPolicy`
that enforces domain-specific success criteria. The policy below requires at
least one critical task to succeed and limits the number of tolerated failures
overall.

Key concepts:
- Subclass :class:`GroupExecutionPolicy` for custom logic
- Call ``_validate_group_results`` to ensure all task results are present
- Raise :class:`ParallelGroupError` when the policy decides the group failed
- Use ``parallel.with_execution(policy=...)`` to apply the policy
"""

from __future__ import annotations

import logging
from typing import Iterable

from graflow.coordination.coordinator import CoordinationBackend
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.handlers.group_policy import GroupExecutionPolicy
from graflow.core.workflow import workflow
from graflow.exceptions import ParallelGroupError


class CriticalLimitedFailuresPolicy(GroupExecutionPolicy):
    """Custom policy that enforces critical success and limits failures.

    Args:
        critical_task_ids: iterable of task IDs that must succeed
        max_failures: maximum number of total task failures allowed
    """

    def __init__(self, critical_task_ids: Iterable[str], max_failures: int) -> None:
        ids = list(critical_task_ids)
        if not ids:
            raise ValueError("critical_task_ids must contain at least one task id")
        if max_failures < 0:
            raise ValueError("max_failures must be non-negative")

        self._critical_task_ids = frozenset(ids)
        self.max_failures = max_failures

    def get_name(self) -> str:  # pragma: no cover - used for logging / introspection only
        return "critical_limited_failures"

    def on_group_finished(self, group_id, tasks, results, context):
        # Ensure we received a result for every task defined in the group.
        self._validate_group_results(group_id, tasks, results)

        task_ids = {task.task_id for task in tasks}
        missing = self._critical_task_ids - task_ids
        if missing:
            raise ValueError(f"Critical task ids {sorted(missing)} are not part of group {group_id}")

        successful_tasks, failed_tasks = self._partition_group_results(results)

        # Fail if any critical task did not succeed.
        failed_critical = [
            (task_id, results[task_id].error_message or "Unknown error")
            for task_id in self._critical_task_ids
            if not results[task_id].success
        ]
        if failed_critical:
            raise ParallelGroupError(
                f"Critical tasks failed: {[task_id for task_id, _ in failed_critical]}",
                group_id=group_id,
                failed_tasks=failed_critical,
                successful_tasks=successful_tasks,
            )

        # Enforce failure budget for optional tasks.
        if len(failed_tasks) > self.max_failures:
            raise ParallelGroupError(
                f"Too many task failures ({len(failed_tasks)} > {self.max_failures})",
                group_id=group_id,
                failed_tasks=failed_tasks,
                successful_tasks=successful_tasks,
            )

        optional_failures = [tid for tid, _ in failed_tasks if tid not in self._critical_task_ids]
        if optional_failures:
            logging.warning(
                "Group %s succeeded with %d optional failure(s): %s",
                group_id,
                len(optional_failures),
                optional_failures,
            )
        else:
            logging.info("Group %s succeeded with all tasks passing", group_id)


def example_custom_policy():
    """Demonstrate the custom policy in action."""
    print("=" * 60)
    print("Example: Custom policy requiring critical success and limited failures")
    print("=" * 60)

    with workflow("custom_policy_workflow") as wf:

        @task
        def provision_database():
            print("  [provision_database] Provisioning database ...")
            return "db_ready"

        @task
        def seed_reference_data():
            print("  [seed_reference_data] Seeding reference data ...")
            raise RuntimeError("Seed step failed")

        @task
        def invalidate_cache():
            print("  [invalidate_cache] Invalidating cache ...")
            raise RuntimeError("Cache node offline")

        @task
        def warm_feature_flags():
            print("  [warm_feature_flags] Warming feature flags ...")
            return "flags_ready"

        deployment = (provision_database | seed_reference_data | invalidate_cache | warm_feature_flags).with_execution(
            backend=CoordinationBackend.THREADING,
            policy=CriticalLimitedFailuresPolicy(
                critical_task_ids=["provision_database"],
                max_failures=1,
            ),
        )

        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, deployment.task_id)

        try:
            engine.execute(context)
            print("\n✓ Workflow completed successfully\n")
        except ParallelGroupError as exc:
            print("\n✗ Workflow failed!")
            print(f"  Group: {exc.group_id}")
            print(f"  Failed tasks ({len(exc.failed_tasks)}): {exc.failed_tasks}")
            print(f"  Successful tasks: {exc.successful_tasks}\n")
            assert len(exc.failed_tasks) == 2
        else:
            raise AssertionError("Expected ParallelGroupError was not raised")


def example_policy_rejects_too_many_failures():
    """Show the policy rejecting a group when failure budget is exceeded."""
    print("=" * 60)
    print("Example: Policy rejects when optional failure budget is exceeded")
    print("=" * 60)

    with workflow("custom_policy_failure") as wf:

        @task
        def primary_task():
            print("  [primary_task] Executing primary task ...")
            return "primary_success"

        @task
        def optional_task_a():
            print("  [optional_task_a] Optional task A ...")
            raise RuntimeError("Optional task A failed")

        @task
        def optional_task_b():
            print("  [optional_task_b] Optional task B ...")
            raise RuntimeError("Optional task B failed")

        group = (primary_task | optional_task_a | optional_task_b).with_execution(
            backend=CoordinationBackend.THREADING,
            policy=CriticalLimitedFailuresPolicy(
                critical_task_ids=["primary_task"],
                max_failures=1,
            ),
        )

        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, group.task_id)

        try:
            engine.execute(context)
        except ParallelGroupError as exc:
            print("\n✗ Workflow failed as expected:")
            print(f"  Reason: {exc}")
            print(f"  Failed tasks: {exc.failed_tasks}")
            print(f"  Successful tasks: {exc.successful_tasks}\n")
            assert len(exc.failed_tasks) == 2
        else:
            raise AssertionError("Expected ParallelGroupError was not raised")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "=" * 60)
    print("PARALLEL GROUP ERROR HANDLING: CUSTOM POLICY")
    print("=" * 60 + "\n")

    example_custom_policy()
    example_policy_rejects_too_many_failures()

    print("=" * 60)
    print("All custom policy examples completed!")
    print("=" * 60)
