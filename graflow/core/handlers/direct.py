"""Direct task execution handler."""

import logging
from typing import TYPE_CHECKING

from graflow.core.handler import TaskHandler

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.task import Executable

logger = logging.getLogger(__name__)


class DirectTaskHandler(TaskHandler):
    """Execute tasks directly in the current process.

    This handler simply calls task.run() without any containerization
    or process isolation. It's the default and most straightforward
    execution method.

    Examples:
        >>> handler = DirectTaskHandler()
        >>> handler.execute_task(my_task, context)
    """

    def execute_task(self, task: 'Executable', context: 'ExecutionContext') -> None:
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
            # Store result in context
            context.set_result(task_id, result)
            logger.debug(f"[DirectTaskHandler] Task {task_id} completed successfully")
        except Exception as e:
            # Store exception in context
            context.set_result(task_id, e)
            logger.error(f"[DirectTaskHandler] Task {task_id} failed: {e}")
            raise
