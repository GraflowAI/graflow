from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from graflow.coordination.task_spec import TaskSpec


class CoordinationBackend(Enum):
    """Types of coordination backends for parallel execution."""
    REDIS = "redis"
    MULTIPROCESSING = "multiprocessing"
    DIRECT = "direct"

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
    def dispatch_task(self, task_spec: Any, group_id: str) -> None:
        """Dispatch task to worker."""
        pass

    @abstractmethod
    def cleanup_barrier(self, barrier_id: str) -> None:
        """Clean up barrier resources."""
        pass

    def execute_group(self, group_id: str, tasks: List['TaskSpec']) -> None:
        """Execute parallel group with barrier synchronization."""
        barrier_id = self.create_barrier(group_id, len(tasks))
        try:
            for task in tasks:
                self.dispatch_task(task, group_id)
            if not self.wait_barrier(barrier_id):
                raise TimeoutError(f"Barrier wait timeout for group {group_id}")
            print(f"Parallel group {group_id} completed successfully")
        finally:
            self.cleanup_barrier(barrier_id)
