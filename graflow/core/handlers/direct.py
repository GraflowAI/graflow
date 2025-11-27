"""Direct task execution handler."""

import logging
from typing import Any

from graflow.core.context import ExecutionContext
from graflow.core.handler import TaskHandler
from graflow.core.task import Executable

logger = logging.getLogger(__name__)


class DirectTaskHandler(TaskHandler):
    """Execute tasks directly in the current process.

    This handler simply calls task.run() without any containerization
    or process isolation. It's the default and most straightforward
    execution method.

    Provides concrete implementation of execute_task() (abstract in TaskHandler).

    Examples:
        >>> handler = DirectTaskHandler()
        >>> handler.execute_task(my_task, context)
    """

    def get_name(self) -> str:
        """Return handler name."""
        return "direct"

    def execute_task(self, task: Executable, context: ExecutionContext) -> Any:
        """Execute task and store result in context.

        Args:
            task: Executable task to execute
            context: Execution context
        """
        task_id = task.task_id
        logger.debug(f"[DirectTaskHandler] Executing task {task_id}")

        try:
            # Execute task
            result = task.run()
            # Store result in context (including None)
            context.set_result(task_id, result)
            logger.debug(f"[DirectTaskHandler] Task {task_id} completed successfully")
            return result
        except Exception as e:
            # Store exception in context
            context.set_result(task_id, e)
            logger.debug(f"[DirectTaskHandler] Task {task_id} failed: {e}")
            raise
