"""Simple test of execution."""

from graflow.core.context import execute_with_cycles
from graflow.core.decorators import task
from graflow.utils.graph import show_graph_info


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
show_graph_info()

print("\nExecuting:")
execute_with_cycles("start", max_steps=5)
