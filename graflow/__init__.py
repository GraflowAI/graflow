"""Graflow - An executable task graph engine."""

try:
    from importlib.metadata import version

    __version__ = version("graflow")
except Exception:
    pass

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.task import chain, parallel
from graflow.core.workflow import workflow

__all__ = ["TaskExecutionContext", "chain", "parallel", "task", "workflow"]
