from abc import ABC, abstractmethod
from enum import Enum

from graflow.coordination.executor import TaskSpec


class CoordinationBackend(Enum):
    """Types of coordination backends for parallel execution."""
    REDIS = "redis"
    MULTIPROCESSING = "multiprocessing"

class TaskCoordinator(ABC):
    """Abstract base class for task coordination."""

    @abstractmethod
    def create_barrier(self, barrier_id: str, participant_count: int) -> str:
        """Create a barrier for parallel task synchronization."""
        pass

    @abstractmethod
    def wait_barrier(self, barrier_id: str, timeout: int = 30) -> bool:
        """Wait at barrier until all participants arrive."""
        pass

    @abstractmethod
    def signal_barrier(self, barrier_id: str) -> None:
        """Signal barrier completion."""
        pass

    @abstractmethod
    def dispatch_task(self, task_spec: TaskSpec, group_id: str) -> None:
        """Dispatch task to worker."""
        pass

    @abstractmethod
    def cleanup_barrier(self, barrier_id: str) -> None:
        """Clean up barrier resources."""
        pass
