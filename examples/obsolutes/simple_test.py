"""Simple test of execution."""

from graflow.core.decorators import task
from graflow.core.workflow import current_workflow_context


@task
def start():
    print("Starting!")

@task
def middle():
    print("Middle!")

@task
def end():
    print("End!")

pipeline = start >> middle >> end

print("Graph before execution:")
ctx = current_workflow_context()
ctx.show_info()

print("\nExecuting:")
ctx.execute("start", max_steps=5)
