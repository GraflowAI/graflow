"""Common TypedDict schemas for typed channel communication."""

from typing import Any, TypedDict

from graflow.channels.typed import ChannelTypeRegistry


class TaskResultMessage(TypedDict):
    """Standard message for task execution results."""

    task_id: str
    result: Any
    timestamp: float
    status: str


class TaskProgressMessage(TypedDict):
    """Message for reporting task progress."""

    task_id: str
    progress: float
    message: str
    timestamp: float


class TaskErrorMessage(TypedDict):
    """Message for reporting task errors."""

    task_id: str
    error_type: str
    error_message: str
    timestamp: float


class WorkflowStateMessage(TypedDict):
    """Message for workflow state updates."""

    workflow_id: str
    state: str
    current_task: str
    completed_tasks: list[str]
    timestamp: float


class DataTransferMessage(TypedDict):
    """Generic message for data transfer between tasks."""

    from_task: str
    to_task: str
    data: Any
    data_type: str
    timestamp: float


class CycleNotificationMessage(TypedDict):
    """Message for cycle execution notifications."""

    task_id: str
    cycle_count: int
    max_cycles: int
    data: Any


# Register common message types
ChannelTypeRegistry.register("task_result", TaskResultMessage)
ChannelTypeRegistry.register("task_progress", TaskProgressMessage)
ChannelTypeRegistry.register("task_error", TaskErrorMessage)
ChannelTypeRegistry.register("workflow_state", WorkflowStateMessage)
ChannelTypeRegistry.register("data_transfer", DataTransferMessage)
ChannelTypeRegistry.register("cycle_notification", CycleNotificationMessage)
