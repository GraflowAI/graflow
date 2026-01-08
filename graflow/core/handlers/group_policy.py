"""Group execution policies for parallel task result handling."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

from graflow.core.context import ExecutionContext
from graflow.core.handler import TaskResult
from graflow.core.task import Executable
from graflow.exceptions import ParallelGroupError

logger = logging.getLogger(__name__)


class GroupExecutionPolicy(ABC):
    """Abstract base class for parallel group execution policies."""

    def get_name(self) -> str:
        """Return policy name for introspection or debugging."""
        return self.__class__.__name__

    def _validate_group_results(
        self,
        group_id: str,
        tasks: Sequence[Executable],
        results: Dict[str, TaskResult],
    ) -> None:
        """Ensure that collected results match provided tasks."""

        expected_task_ids = {task.task_id for task in tasks}
        actual_task_ids = set(results.keys())

        if expected_task_ids != actual_task_ids:
            missing = expected_task_ids - actual_task_ids
            unexpected = actual_task_ids - expected_task_ids

            error_parts = []
            if missing:
                error_parts.append(f"missing results for tasks: {sorted(missing)}")
            if unexpected:
                error_parts.append(f"unexpected results for tasks: {sorted(unexpected)}")

            raise ParallelGroupError(
                f"Parallel group {group_id} result mismatch: {', '.join(error_parts)}",
                group_id=group_id,
                failed_tasks=[],
                successful_tasks=list(actual_task_ids),
            )

    def _partition_group_results(
        self,
        results: Dict[str, TaskResult],
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Split results into successes and failures."""

        successful_tasks: List[str] = []
        failed_tasks: List[Tuple[str, str]] = []

        for task_id, result in results.items():
            if result.success:
                successful_tasks.append(task_id)
            else:
                failed_tasks.append((task_id, result.error_message or "Unknown error"))

        return successful_tasks, failed_tasks

    @abstractmethod
    def on_group_finished(
        self,
        group_id: str,
        tasks: Sequence[Executable],
        results: Dict[str, TaskResult],
        context: ExecutionContext,
    ) -> None:
        """Evaluate group results and optionally raise ``ParallelGroupError``."""


class StrictGroupPolicy(GroupExecutionPolicy):
    """Strict policy: any task failure aborts the group."""

    def get_name(self) -> str:
        return "strict"

    def on_group_finished(
        self,
        group_id: str,
        tasks: Sequence[Executable],
        results: Dict[str, TaskResult],
        context: ExecutionContext,
    ) -> None:
        self._validate_group_results(group_id, tasks, results)
        successful_tasks, failed_tasks = self._partition_group_results(results)

        if failed_tasks:
            raise ParallelGroupError(
                f"Parallel group {group_id} failed: {len(failed_tasks)} task(s) failed",
                group_id=group_id,
                failed_tasks=failed_tasks,
                successful_tasks=successful_tasks,
            )


class BestEffortGroupPolicy(GroupExecutionPolicy):
    """Best-effort policy: log failures but continue execution."""

    def get_name(self) -> str:
        return "best_effort"

    def on_group_finished(
        self,
        group_id: str,
        tasks: Sequence[Executable],
        results: Dict[str, TaskResult],
        context: ExecutionContext,
    ) -> None:
        self._validate_group_results(group_id, tasks, results)
        _successful_tasks, failed_tasks = self._partition_group_results(results)

        if failed_tasks:
            failed_ids = [task_id for task_id, _ in failed_tasks]
            logger.warning(
                "Parallel group %s completed with %d failure(s); continuing (best-effort). Failed tasks: %s",
                group_id,
                len(failed_tasks),
                failed_ids,
            )
        else:
            logger.debug("Parallel group %s completed successfully (best-effort).", group_id)


class CriticalGroupPolicy(GroupExecutionPolicy):
    """Critical policy: only critical task failures abort the group."""

    def __init__(self, critical_task_ids: Iterable[str]):
        ids = list(critical_task_ids)
        if not ids:
            raise ValueError("critical_task_ids must contain at least one task id")
        self._critical_task_ids = frozenset(ids)

    def get_name(self) -> str:
        return "critical"

    @property
    def critical_task_ids(self) -> List[str]:
        """Return critical task identifiers as a sorted list."""
        return sorted(self._critical_task_ids)

    def on_group_finished(
        self,
        group_id: str,
        tasks: Sequence[Executable],
        results: Dict[str, TaskResult],
        context: ExecutionContext,
    ) -> None:
        self._validate_group_results(group_id, tasks, results)

        task_ids = {task.task_id for task in tasks}
        missing_from_group = [task_id for task_id in self._critical_task_ids if task_id not in task_ids]
        if missing_from_group:
            raise ValueError(
                f"Critical task ids {sorted(missing_from_group)} are not part of parallel group {group_id}"
            )

        successful_tasks, failed_tasks = self._partition_group_results(results)

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

        optional_failures = [task_id for task_id, _ in failed_tasks if task_id not in self._critical_task_ids]
        if optional_failures:
            logger.warning(
                "Parallel group %s completed with optional failures; continuing. Optional tasks: %s",
                group_id,
                optional_failures,
            )
        else:
            logger.debug("Parallel group %s completed successfully (critical policy).", group_id)


class AtLeastNGroupPolicy(GroupExecutionPolicy):
    """Require at least ``min_success`` tasks to complete successfully."""

    def __init__(self, min_success: int):
        if min_success < 0:
            raise ValueError("min_success must be non-negative")
        self.min_success = min_success

    def get_name(self) -> str:
        return f"at_least_{self.min_success}"

    def on_group_finished(
        self,
        group_id: str,
        tasks: Sequence[Executable],
        results: Dict[str, TaskResult],
        context: ExecutionContext,
    ) -> None:
        self._validate_group_results(group_id, tasks, results)

        total_tasks = len(tasks)
        if self.min_success > total_tasks:
            raise ValueError(f"min_success ({self.min_success}) cannot exceed number of tasks ({total_tasks})")

        successful_tasks, failed_tasks = self._partition_group_results(results)

        if len(successful_tasks) < self.min_success:
            raise ParallelGroupError(
                f"Only {len(successful_tasks)}/{self.min_success} tasks succeeded in group {group_id}",
                group_id=group_id,
                failed_tasks=failed_tasks,
                successful_tasks=successful_tasks,
            )

        if failed_tasks:
            logger.info(
                "Parallel group %s met at-least-%d policy despite %d failure(s).",
                group_id,
                self.min_success,
                len(failed_tasks),
            )
        else:
            logger.debug(
                "Parallel group %s met at-least-%d policy with all tasks succeeding.",
                group_id,
                self.min_success,
            )


POLICY_REGISTRY: Dict[str, type[GroupExecutionPolicy]] = {
    "strict": StrictGroupPolicy,
    "best_effort": BestEffortGroupPolicy,
    "critical": CriticalGroupPolicy,
    "at_least_n": AtLeastNGroupPolicy,
}


def canonicalize_group_policy(policy: Union[str, GroupExecutionPolicy]) -> Union[str, Dict[str, Any]]:
    """Return a canonical representation for a group policy.

    Built-in policies are normalized to strings or plain dictionaries to avoid
    relying on pickle. Custom policies are wrapped in a dictionary that marks
    the payload as `__custom__` while keeping the instance accessible for
    immediate execution.
    """

    if isinstance(policy, str):
        return policy

    if isinstance(policy, StrictGroupPolicy):
        return "strict"

    if isinstance(policy, BestEffortGroupPolicy):
        return "best_effort"

    if isinstance(policy, CriticalGroupPolicy):
        return {
            "type": "critical",
            "critical_task_ids": policy.critical_task_ids,
        }

    if isinstance(policy, AtLeastNGroupPolicy):
        return {
            "type": "at_least_n",
            "min_success": policy.min_success,
        }

    if isinstance(policy, GroupExecutionPolicy):
        return {"type": "__custom__", "policy": policy}

    raise TypeError(f"Unsupported policy specification type: {type(policy)!r}")


def resolve_group_policy(policy: Union[str, Mapping[str, Any], GroupExecutionPolicy]) -> GroupExecutionPolicy:
    """Resolve a group policy from a serialized form or return the given instance."""

    if isinstance(policy, GroupExecutionPolicy):
        return policy

    if isinstance(policy, str):
        policy_name = policy.lower()
        if policy_name not in POLICY_REGISTRY:
            raise ValueError(
                f"Unknown group execution policy '{policy}'. Available policies: {sorted(POLICY_REGISTRY)}"
            )
        policy_cls = POLICY_REGISTRY[policy_name]
        return policy_cls()

    if isinstance(policy, Mapping):
        policy_type = policy.get("type")
        if not isinstance(policy_type, str):
            raise ValueError("Serialized policy must include a string 'type' field")

        policy_name = policy_type.lower()

        if policy_name == "critical":
            critical_ids = policy.get("critical_task_ids")
            if not isinstance(critical_ids, list):
                raise ValueError("'critical_task_ids' must be provided as a list")
            return CriticalGroupPolicy(critical_ids)

        if policy_name == "at_least_n":
            min_success = policy.get("min_success")
            if not isinstance(min_success, int):
                raise ValueError("'min_success' must be provided as an integer")
            return AtLeastNGroupPolicy(min_success)

        if policy_name in ("strict", "best_effort"):
            return resolve_group_policy(policy_name)

        if policy_name == "__custom__":
            instance = policy.get("policy")
            if not isinstance(instance, GroupExecutionPolicy):
                raise ValueError("Serialized custom policy must include a GroupExecutionPolicy instance")
            return instance

        raise ValueError(f"Unknown serialized group execution policy '{policy_name}'.")

    raise TypeError(f"Unsupported policy specification type: {type(policy)!r}")
