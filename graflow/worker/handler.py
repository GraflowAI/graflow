"""Task handler interface and implementations for task execution."""

import logging
import time
from abc import ABC, abstractmethod

from graflow.core.task import Executable

logger = logging.getLogger(__name__)


class TaskHandler(ABC):
    """Abstract base class for task processing handlers."""

    def process_task(self, task: Executable) -> bool:
        """Process a task and return success status.

        Args:
            task: Task object to execute

        Returns:
            bool: True if task completed successfully, False otherwise
        """
        start_time = time.time()
        try:
            success = self._process_task(task)
            duration = time.time() - start_time

            if success:
                self.on_task_success(task, duration)
            else:
                self.on_task_failure(task, Exception("Task handler returned False"), duration)

            return success
        except Exception as e:
            duration = time.time() - start_time
            self.on_task_failure(task, e, duration)
            return False

    @abstractmethod
    def _process_task(self, task: Executable) -> bool:
        """Custom task processing implementation point.

        Args:
            task: Task object to execute

        Returns:
            bool: True if task completed successfully, False otherwise
        """
        pass

    def on_task_success(self, task: Executable, duration: float) -> None:
        """Callback for successful task completion.

        Args:
            task: The completed task
            duration: Execution duration in seconds
        """
        logger.info(f"Task {getattr(task, 'task_id', 'unknown')} succeeded after {duration:.3f}s")

    def on_task_failure(self, task: Executable, error: Exception, duration: float) -> None:
        """Callback for task failure.

        Args:
            task: The failed task
            error: Exception that caused the failure (may be None)
            duration: Execution duration in seconds
        """
        task_id = getattr(task, 'task_id', 'unknown')
        error_msg = str(error) if error else "Unknown error"
        logger.error(f"Task {task_id} failed after {duration:.3f}s: {error_msg}")

    def on_task_timeout(self, task: Executable, duration: float) -> None:
        """Callback for task timeout.

        Args:
            task: The timed out task
            duration: Execution duration in seconds
        """
        task_id = getattr(task, 'task_id', 'unknown')
        logger.warning(f"Task {task_id} timed out after {duration:.3f}s")


class DirectTaskExecutor(TaskHandler):
    """Task executor that runs tasks directly in the worker process."""

    def _process_task(self, task: Executable) -> bool:
        """Execute task directly in the current process.

        Args:
            task: Task object to execute

        Returns:
            bool: True if task completed successfully, False otherwise
        """
        try:
            # Execute the task (assumes task is callable)
            result = task()
            logger.debug(f"Task executed successfully with result: {result}")
            return True
        except Exception as e:
            logger.error(f"InProcess task execution failed: {e}")
            return False
