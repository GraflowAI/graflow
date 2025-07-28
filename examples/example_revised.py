"""Example usage of revised graflow with @task decorator and global graph."""

from graflow.core.context import execute_with_cycles
from graflow.core.decorators import task
from graflow.core.task import Task
from graflow.core.workflow import clear_workflow_context
from graflow.utils.graph import show_graph_info, visualize_dependencies

# Clear any existing tasks
clear_workflow_context()

# Create tasks using @task decorator
@task
def task_A():  # noqa: N802
    print("Executing task A logic")
    return "Result A"

@task
def task_B():  # noqa: N802
    print("Executing task B logic")
    return "Result B"

@task
def task_C():  # noqa: N802
    print("Executing task C logic")
    return "Result C"

@task(id="custom_D")
def task_D():  # noqa: N802
    print("Executing task D logic")
    return "Result D"

@task
def task_E():  # noqa: N802
    print("Executing task E logic")
    return "Result E"

print("=== Building Dependencies with Operators ===")
# Build dependencies using operators (auto-registers to global graph)
task_A >> task_B >> task_C
task_D >> task_E

# Create parallel group
parallel_group = task_B | task_C
print(f"Created parallel group: {parallel_group}")

# Complex flow
task_A >> (task_B | task_C) >> task_D

print("\n=== Graph Information ===")
show_graph_info()

print("\n=== Dependencies Visualization ===")
visualize_dependencies()

print("\n=== Execution ===")
execute_with_cycles("task_A", max_steps=10)

print("\n=== Traditional Task Objects ===")
# Traditional Task objects still work
clear_workflow_context()

X = Task("X")
Y = Task("Y")
Z = Task("Z")

X >> Y >> Z

show_graph_info()
execute_with_cycles("X", max_steps=5)
