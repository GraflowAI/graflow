from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.handlers.group_policy import GroupExecutionPolicy
    from graflow.core.task import Executable


class CoordinationBackend(Enum):
    """Types of coordination backends for parallel execution."""

    REDIS = "redis"
    THREADING = "threading"
    DIRECT = "direct"


class TaskCoordinator(ABC):
    """Abstract base class for task coordination."""

    @abstractmethod
    def execute_group(
        self, group_id: str, tasks: List[Executable], execution_context: ExecutionContext, policy: GroupExecutionPolicy
    ) -> None:
        """Execute parallel group with policy.

        Args:
            group_id: Parallel group identifier
            tasks: List of tasks to execute
            execution_context: Execution context
            policy: GroupExecutionPolicy instance for result evaluation
        """
        pass
