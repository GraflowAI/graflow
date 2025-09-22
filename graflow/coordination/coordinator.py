from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.task import Executable


class CoordinationBackend(Enum):
    """Types of coordination backends for parallel execution."""
    REDIS = "redis"
    MULTIPROCESSING = "multiprocessing"
    DIRECT = "direct"

class TaskCoordinator(ABC):
    """Abstract base class for task coordination."""

    @abstractmethod
    def execute_group(self, group_id: str, tasks: List['Executable'], execution_context: 'ExecutionContext') -> None:
        """Execute parallel group with barrier synchronization."""
        pass
