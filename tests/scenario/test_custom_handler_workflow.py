"""Scenario test executing a workflow that registers custom task handlers."""

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.handler import TaskHandler
from graflow.core.task import Executable
from graflow.core.workflow import workflow


def test_custom_handler_workflow_execution():
    """Execute workflow using custom handlers and verify results."""

    class RecordingLoggingHandler(TaskHandler):
        def __init__(self) -> None:
            self.executed: list[str] = []

        def execute_task(self, task: Executable, context: ExecutionContext):
            self.executed.append(task.task_id)
            result = task.run()
            context.set_result(task.task_id, result)
            return result

    class RecordingTimingHandler(TaskHandler):
        def __init__(self) -> None:
            self.executed: list[str] = []

        def execute_task(self, task: Executable, context: ExecutionContext):
            self.executed.append(task.task_id)
            result = task.run()
            context.set_result(task.task_id, result)
            return result

    logging_handler = RecordingLoggingHandler()
    timing_handler = RecordingTimingHandler()

    with workflow("custom_handler_test") as ctx:

        @task(handler="direct")
        def normal_task():
            return "normal_result"

        @task(handler="logging")
        def custom_task():
            return "custom_result"

        @task(handler="timing")
        def timed_task():
            return "timed_result"

        @task(handler="direct", inject_context=True)
        def results_task(task_ctx):
            return {
                "normal": task_ctx.get_result("normal_task"),
                "custom": task_ctx.get_result("custom_task"),
                "timed": task_ctx.get_result("timed_task"),
            }

        normal_task >> custom_task >> timed_task >> results_task

        engine = WorkflowEngine()
        engine.register_handler("logging", logging_handler)
        engine.register_handler("timing", timing_handler)

        exec_context = ExecutionContext.create(ctx.graph, "normal_task", max_steps=10)
        engine.execute(exec_context)

    assert logging_handler.executed == ["custom_task"]
    assert timing_handler.executed == ["timed_task"]

    assert exec_context.get_result("normal_task") == "normal_result"
    assert exec_context.get_result("custom_task") == "custom_result"
    assert exec_context.get_result("timed_task") == "timed_result"

    final_results = exec_context.get_result("results_task")
    assert final_results == {
        "normal": "normal_result",
        "custom": "custom_result",
        "timed": "timed_result",
    }

    assert exec_context.steps == 4
