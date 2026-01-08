"""Task specification for parallel execution."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext


class TaskSpec:
    """Task specification for parallel execution."""

    def __init__(
        self, task_id: str, execution_context: "ExecutionContext", func, args: tuple = (), kwargs: Optional[dict] = None
    ):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.execution_context = execution_context
